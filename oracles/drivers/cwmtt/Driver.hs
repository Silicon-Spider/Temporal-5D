-- The harness's state driver for penteract/cwmtt.
--
-- It runs beside cwmtt's stock CLI, never in its place: `cwmtt count` remains
-- the source of cwmtt's answers (the action counts, and how far the record is
-- replayed). This program reports what the CLI has no command for: the side to
-- move and every board cwmtt holds at each position of the record, so that the
-- harness can fingerprint the position cwmtt parsed (decision H30; design spec
-- 11.0, "Running the gate"). The adapter, oracles/oracle_harness/adapters/cwmtt.py,
-- merges the two by position index, and refuses the merge when the two did not
-- read and replay the record the same way.
--
-- It reads the record as the stock CLI reads it: the same parser (parsePGN), the
-- same set-up (getGame), the same replay (buildGame scans the same concretize
-- that the CLI's build scans), and standard input left in the runtime's default
-- encoding and newline mode, which is what the CLI leaves it in.
--
-- stdin : one 5DPGN record.
-- stdout: JSON Lines, every line flushed as soon as it is written. All output is
--         ASCII: anything else in a string is written as a \u escape.
--   {"event":"start","driver":"cwmtt","schema":1,"algorithm":"fast"|"naive",
--    "layout":"<the cwmtt layout it set up>"|null,"parsed_actions":n,
--    "unparsed":"<what the parser left unread>"}
--   (the start event says what the parse read, so that a run cut short for
--   time can still be held to the stock CLI's reading of the record)
--   then, for each position k (k actions replayed; 0 is the set-up):
--   {"event":"state","index":k,"size":"WxH","markers":true,
--    "boards":["<L>:<T>:<c>:<placement>", ...]}
--   {"event":"explain","index":k, ...}                       (--explain only)
--   {"event":"position","index":k,"side":"white"|"black"}
--   {"event":"actions","index":k,"count":n,"truncated":bool,"actions":[...]}
--                                                            (--actions k only)
--   and last:
--   {"event":"end","status":"ok"|"rejected"|"error","actions_accepted":n,
--    "parsed_actions":m,"unparsed":"<what the parser left unread>",
--    "failure":"<cwmtt's replay failure>"|null, "reason":..., "detail":...}
--
-- The state event lists every board cwmtt holds - superseded and head boards,
-- every timeline, both colours - as "<L>:<T>:<c>:<placement>": L the timeline
-- (cwmtt knows only games that start with one timeline, or three, so its
-- central timeline is "0" and the others are signed integers); T the board's
-- turn, as the notation numbers it (cwmtt's own T); c the side to move on that
-- board; and the placement the 5DFEN pieces field, ranks from the top, with "*"
-- after each pawn, rook, king and brawn that cwmtt holds as never moved. cwmtt
-- tracks that for every piece, hence "markers":true. The harness, not this
-- program, canonicalises and hashes it.
--
-- The position event carries the side to move and nothing else: the CLI answers
-- the counts. A replay failure is reported in the end event as cwmtt words it,
-- unclassified ("rejected", reason "unrecognised"); the adapter classifies the
-- CLI's copy of the same message. "error" means cwmtt threw while this program
-- replayed or rendered a position; what it threw is the detail.
--
-- Diagnostics, never compared:
--   explain  what cwmtt computes about the position before it searches:
--            "playable", the boards the side to move may play on
--            (Moves.playableBoards); "present", its present (Datatypes.present);
--            "num_active", its count of active timelines on the side that has
--            more of them (Datatypes.numActive); "new_timeline", the timeline a
--            branch by the side to move would create (Datatypes.getNewL).
--   actions  the legal actions at that position as the chosen search yields them
--            (--algorithm fast: FastCheckmate.fastLegalMoveSets, the CLI's
--            default; naive: Moves.legalMoveSets), at most 5000; "count" is how
--            many are listed and "truncated" whether the search had more (if
--            cwmtt throws part-way, the list ends there, is marked truncated,
--            and the event carries "error": what it threw). Each
--            move is {"from":{"L","T","c","square"},"to":{...},"promotion"}. cwmtt
--            writes a castling as two steps, the rook's and then the king's, and
--            an en passant capture as the pawn's step followed by a second step
--            that empties the captured pawn's square; the move listed is the
--            king's step and the pawn's step. cwmtt promotes a pawn or a brawn
--            that reaches its last rank to a queen, so "promotion" is "Q" for
--            such a step and null otherwise.

module Main (main) where

import Control.Exception (SomeException, displayException, evaluate, try)
import Data.Char (chr, intToDigit, ord, toLower)
import Data.List (intercalate, transpose)
import qualified Data.List as List (foldl')
import System.Environment (getArgs)
import System.Exit (ExitCode (..), exitWith)
import System.IO

import Game.Chess.TimeTravel.BuildGame (buildGame)
import Game.Chess.TimeTravel.Datatypes
  ( Board, Cell (..), Coords, Move, MoveSet, Moved (..), Piece (..), Player (..), State
  , getAt, getBoard, getNewL, numActive, present )
import Game.Chess.TimeTravel.FastCheckmate (fastLegalMoveSets)
import Game.Chess.TimeTravel.Layouts (getGame, layouts)
import Game.Chess.TimeTravel.Moves (legalMoveSets, playableBoards)
import Game.Chess.TimeTravel.PGNParser (NotatedGame (Notated), parsePGN)
import Game.Chess.TimeTravel.Utils (Result (..))

schema :: Int
schema = 1

-- The most actions one actions event lists.
actionsListed :: Int
actionsListed = 5000

data Algorithm = Fast | Naive

algorithmName :: Algorithm -> String
algorithmName Fast = "fast"
algorithmName Naive = "naive"

legalActions :: Algorithm -> State -> [MoveSet]
legalActions Fast = fastLegalMoveSets
legalActions Naive = legalMoveSets

data Options = Options
  { optExplain :: Bool
  , optActionsAt :: Maybe Int
  , optAlgorithm :: Algorithm
  }

usage :: String
usage = unlines
  [ "usage: cwmtt_state_driver [--algorithm fast|naive] [--explain] [--actions K]"
  , "  Reads a 5DPGN record from stdin; prints one JSON object per line."
  , "  --explain    after each state event, what cwmtt computes about the position"
  , "  --actions K  at position K, list up to " ++ show actionsListed ++ " legal actions"
  , "  --algorithm  the search --actions lists the actions of (default: fast)"
  ]

readOptions :: [String] -> Options -> Either String Options
readOptions [] options = Right options
readOptions ("--explain" : rest) options = readOptions rest options {optExplain = True}
readOptions ("--actions" : value : rest) options
  | not (null value), all (`elem` ['0' .. '9']) value, length value < 10 =
      readOptions rest options {optActionsAt = Just (read value)}
readOptions ("--algorithm" : "fast" : rest) options = readOptions rest options {optAlgorithm = Fast}
readOptions ("--algorithm" : "naive" : rest) options = readOptions rest options {optAlgorithm = Naive}
readOptions (argument : _) _ = Left ("unexpected argument: " ++ argument)

main :: IO ()
main = do
  arguments <- getArgs
  if any (`elem` ["-h", "--help"]) arguments
    then putStr usage
    else case readOptions arguments (Options False Nothing Fast) of
      Left problem -> do
        hPutStrLn stderr problem
        hPutStr stderr usage
        exitWith (ExitFailure 2)
      Right options -> drive options

-- --- JSON ----------------------------------------------------------------------

-- A JSON string literal, in ASCII.
text :: String -> String
text value = '"' : concatMap escape value ++ "\""
  where
    escape '"' = "\\\""
    escape '\\' = "\\\\"
    escape '\n' = "\\n"
    escape '\r' = "\\r"
    escape '\t' = "\\t"
    escape c
      | c >= ' ' && c <= '~' = [c]
      | ord c > 0xFFFF = let code = ord c - 0x10000
                         in unit (0xD800 + code `div` 0x400) ++ unit (0xDC00 + code `mod` 0x400)
      | otherwise = unit (ord c)
    unit code = "\\u" ++ [intToDigit (code `div` 16 ^ power `mod` 16) | power <- [3, 2, 1, 0 :: Int]]

object :: [(String, String)] -> String
object fields = "{" ++ intercalate "," [text key ++ ":" ++ value | (key, value) <- fields] ++ "}"

array :: [String] -> String
array items = "[" ++ intercalate "," items ++ "]"

bool :: Bool -> String
bool True = "true"
bool False = "false"

orNull :: Maybe String -> String
orNull = maybe "null" text

-- --- The position cwmtt holds ----------------------------------------------------

-- A timeline as the notation writes it. Every layout cwmtt knows starts with an
-- odd number of timelines, the central one numbered 0.
timelineLabel :: Int -> String
timelineLabel l
  | l > 0 = '+' : show l
  | otherwise = show l

colourLetter :: Player -> String
colourLetter White = "w"
colourLetter Black = "b"

sideName :: Player -> String
sideName White = "white"
sideName Black = "black"

pieceLetter :: Piece -> Char
pieceLetter piece = case piece of
  King -> 'K'
  Queen -> 'Q'
  Rook -> 'R'
  Bishop -> 'B'
  Knight -> 'N'
  Pawn -> 'P'
  Unicorn -> 'U'
  Dragon -> 'D'
  Brawn -> 'W'
  RoyalQueen -> 'Y'
  Commoner -> 'C'

-- The pieces whose first move the notation marks with "*" while it is to come.
marked :: Piece -> Bool
marked piece = case piece of
  Pawn -> True
  Rook -> True
  King -> True
  Brawn -> True
  _ -> False

pieceText :: (Player, Piece, Moved) -> String
pieceText (colour, piece, moved) =
  (if colour == White then id else toLower) (pieceLetter piece) : ['*' | moved == Still, marked piece]

-- One rank, files from a: pieces as letters, runs of empty squares as numbers.
rankText :: [Cell] -> String
rankText = go (0 :: Int)
  where
    go blanks [] = blank blanks
    go blanks (Empty : rest) = go (blanks + 1) rest
    go blanks (Full piece : rest) = blank blanks ++ pieceText piece ++ go 0 rest
    blank 0 = ""
    blank blanks = show blanks

-- cwmtt holds a board as its files, a first, each from rank 1 up; the
-- notation writes its ranks from the top down.
placement :: Board -> String
placement board = intercalate "/" (map rankText (reverse (transpose board)))

-- Every board cwmtt holds, as (timeline, turn, colour to move on it, board).
-- cwmtt keeps the boards white is to move on and those black is to move on in
-- two lists of timelines, both ordered from the newest timeline white created
-- (the highest L) down, and each timeline's boards newest first.
boardsOf :: State -> [(Int, Int, Player, Board)]
boardsOf (newest, whiteLines, blackLines, _) = collect White whiteLines ++ collect Black blackLines
  where
    collect colour timelines =
      [ (newest - offset, turn, colour, board)
      | (offset, (lastTurn, boards)) <- zip [0 ..] timelines
      , (turn, board) <- zip [lastTurn, lastTurn - 1 ..] boards
      ]

sizeOf :: State -> String
sizeOf state = case [board | (_, _, _, board) <- boardsOf state] of
  (board : _) -> show (length board) ++ "x" ++ show (fileLength board)
  [] -> "0x0"

fileLength :: Board -> Int
fileLength (file : _) = length file
fileLength [] = 0

stateEvent :: Int -> State -> String
stateEvent index state = object
  [ ("event", text "state"), ("index", show index), ("size", text (sizeOf state)), ("markers", bool True)
  , ("boards", array [ text (timelineLabel l ++ ":" ++ show t ++ ":" ++ colourLetter colour ++ ":" ++ placement board)
                     | (l, t, colour, board) <- boardsOf state ])
  ]

positionEvent :: Int -> State -> String
positionEvent index (_, _, _, mover) =
  object [("event", text "position"), ("index", show index), ("side", text (sideName mover))]

boardRef :: Player -> (Int, Int) -> [(String, String)]
boardRef colour (l, t) = [("L", text (timelineLabel l)), ("T", show t), ("c", text (colourLetter colour))]

explainEvent :: Int -> State -> String
explainEvent index state@(_, _, _, mover) = object
  [ ("event", text "explain"), ("index", show index)
  , ("playable", array [object (boardRef mover lt) | lt <- playableBoards state])
  , ("present", show (present state))
  , ("num_active", show (numActive state))
  , ("new_timeline", text (timelineLabel (getNewL state)))
  ]

-- --- Diagnostics: the legal actions ------------------------------------------------

-- Up to `actionsListed` actions as the search yields them, each rendered in full,
-- and whether the search had more. A failure inside cwmtt's search ends the list
-- where it happened: what was listed stands, the list is marked truncated, and
-- the failure is returned.
listActions :: Algorithm -> State -> IO ([String], Bool, Maybe String)
listActions algorithm state = go (0 :: Int) (legalActions algorithm state) []
  where
    go listed remaining done = do
      next <- catchAll (evaluate (case remaining of
                                    [] -> Nothing
                                    (action : rest) -> Just (action, rest)))
      case next of
        Left problem -> (\message -> (reverse done, True, Just message)) <$> describe problem
        Right Nothing -> return (reverse done, False, Nothing)
        Right (Just (action, rest))
          | listed == actionsListed -> return (reverse done, True, Nothing)
          | otherwise -> do
              rendered <- computed (array (map (moveObject state) action))
              case rendered of
                Left problem -> return (reverse done, True, Just problem)
                Right line -> go (listed + 1) rest (line : done)

actionsEvent :: Int -> ([String], Bool, Maybe String) -> String
actionsEvent index (listed, truncated, failure) = object (
  [ ("event", text "actions"), ("index", show index), ("count", show (length listed))
  , ("truncated", bool truncated), ("actions", array listed) ]
  ++ maybe [] (\problem -> [("error", text problem)]) failure)

-- The step a player makes: the king's in a castling, which cwmtt writes as the
-- rook's step and then the king's; otherwise the first, which in an en passant
-- capture is the pawn's.
playerStep :: State -> Move -> (Coords, Coords)
playerStep state steps = case [step | step@(from, _) <- steps, isKing (getAt state from)] ++ steps of
  (step : _) -> step
  [] -> error "cwmtt listed a move with no steps"
  where
    isKing (Just (Full (_, King, _))) = True
    isKing _ = False

moveObject :: State -> Move -> String
moveObject state@(_, _, _, mover) steps =
  object [("from", end from), ("to", end to), ("promotion", promotion)]
  where
    (from, to) = playerStep state steps
    end (l, t, x, y) = object (boardRef mover (l, t) ++ [("square", text (chr (ord 'a' + x) : show (y + 1)))])
    promotion = case getAt state from of
      Just (Full (_, piece, _)) | piece == Pawn || piece == Brawn, lastRank to -> text "Q"
      _ -> "null"
    lastRank (l, t, _, y) = case mover of
      White -> y == maybe 0 fileLength (getBoard state (l, t)) - 1
      Black -> y == 0

-- --- Running ---------------------------------------------------------------------

catchAll :: IO a -> IO (Either SomeException a)
catchAll = try

-- What an exception says, or a stand-in when even that cannot be computed.
describe :: SomeException -> IO String
describe problem = do
  let message = displayException problem
  shown <- catchAll (evaluate (List.foldl' (flip seq) () message))
  return (either (const "an exception whose message could not be computed") (const message) shown)

-- A string with every character computed, or what went wrong computing it. cwmtt
-- fails lazily (error, fromJust, irrefutable patterns), so nothing is printed
-- until it has been computed in full.
computed :: String -> IO (Either String String)
computed value = do
  outcome <- catchAll (evaluate (List.foldl' (flip seq) () value))
  either (fmap Left . describe) (const (return (Right value))) outcome

emit :: String -> IO ()
emit line = putStr (line ++ "\n") >> hFlush stdout

data Step = Reached State (Result String State) | Finished | Failed String

view :: Result String State -> Step
view (Cons state rest) = Reached state rest
view Nil = Finished
view (Fail message) = Failed message

-- How a replay ended: the positions it reported, and either cwmtt's replay
-- failure (Left) or what was thrown (Right).
data Ending = Ending Int (Maybe (Either String String))

-- Report positions from `index` on; returns how the replay ended.
walk :: Options -> Int -> Result String State -> IO Ending
walk options index result = do
  stepped <- catchAll (evaluate (view result))
  case stepped of
    Left problem -> Ending index . Just . Right <$> describe problem
    Right Finished -> return (Ending index Nothing)
    Right (Failed message) -> Ending index . Just . either Right Left <$> computed message
    Right (Reached state rest) -> do
      -- The position's lines are computed in full before any is printed, so a
      -- failure part-way never leaves a state without its position.
      rendered <- computed (unlines (stateEvent index state
                                     : [explainEvent index state | optExplain options]
                                     ++ [positionEvent index state]))
      case rendered of
        Left problem -> return (Ending index (Just (Right problem)))
        Right lines' -> do
          putStr lines' >> hFlush stdout
          if optActionsAt options == Just index then reportActions options index state else return ()
          walk options (index + 1) rest

reportActions :: Options -> Int -> State -> IO ()
reportActions options index state = listActions (optAlgorithm options) state >>= emit . actionsEvent index

-- The layout cwmtt set up, by its name in cwmtt's own table.
layoutName :: State -> Maybe String
layoutName start = case [name | (name, layout) <- layouts, layout == start] of
  (name : _) -> Just name
  [] -> Nothing

startEvent :: Options -> State -> String -> String -> String
startEvent options start parsedActions unread = object
  [ ("event", text "start"), ("driver", text "cwmtt"), ("schema", show schema)
  , ("algorithm", text (algorithmName (optAlgorithm options)))
  , ("layout", orNull (layoutName start)), ("parsed_actions", parsedActions), ("unparsed", text unread)
  ]

-- The end event, after `reported` positions.
finish :: Int -> String -> String -> Maybe (Either String String) -> IO ()
finish reported parsedActions unread outcome = emit (object (
  [ ("event", text "end"), ("status", text status), ("actions_accepted", show (max (reported - 1) 0))
  , ("parsed_actions", parsedActions), ("unparsed", text unread) ] ++ extra))
  where
    (status, extra) = case outcome of
      Nothing -> ("ok", [("failure", "null")])
      Just (Left failure) ->
        ("rejected", [("failure", text failure), ("reason", text "unrecognised"), ("detail", text failure)])
      Just (Right problem) ->
        ("error", [("failure", "null"), ("reason", text "crash"), ("detail", text problem)])

drive :: Options -> IO ()
drive options = do
  hSetEncoding stdout utf8
  hSetNewlineMode stdout noNewlineTranslation
  hSetBuffering stdout LineBuffering
  input <- getContents
  parsed <- catchAll (evaluate (parsePGN input))
  case parsed of
    Left problem -> describe problem >>= crash
    -- It cannot fail: every part of cwmtt's grammar is optional.
    Right Nothing -> crash "cwmtt's parser returned no parse"
    Right (Just (Notated tags actions remainder, _)) -> do
      -- What the stock CLI computes before its first position: the parse, to its
      -- end, and the text it left unread (which it prints).
      counted <- computed (show (length actions))
      unread <- computed remainder
      case (,) <$> counted <*> unread of
        Left problem -> crash problem
        Right (parsedActions, unreadText) -> do
          let start = getGame tags
          header <- computed (startEvent options start parsedActions unreadText)
          case header of
            Left problem -> finish 0 parsedActions unreadText (Just (Right problem))
            Right line -> do
              emit line
              Ending reported outcome <- walk options 0 (buildGame start actions)
              finish reported parsedActions unreadText outcome
  where
    crash problem = finish 0 "null" "" (Just (Right problem))
