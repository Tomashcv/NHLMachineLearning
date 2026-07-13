# NHL Pilot

## Purpose

The pilot validates data identity, temporal integrity and market settlement
before any large historical backfill.

## Pilot stages

### F2A — Manual raw import

- One manually saved daily-score JSON file
- No automated network collection
- Raw bytes preserved unchanged
- SHA-256 recorded
- Local import manifest created
- Import must be idempotent

### F2B — Canonical game parsing

- Provider game ID
- Season and game type
- Scheduled start in UTC
- Home and away team IDs
- Final score
- Regulation, overtime and shootout status
- Source and ingestion timestamps

### F2C — Five-game validation

The first five real games must cover several outcome types, including at least:

- regulation result;
- overtime result;
- shootout result.

### F2D — Forty-game pilot

The forty-game pilot will span multiple seasons and include:

- regular season;
- playoffs;
- postponed or rescheduled cases where available;
- overtime and shootouts;
- odds-coverage checks;
- goalie and player identity checks.

## Restrictions

No large NHL web backfill is permitted during the pilot.

The source remains research-only and its raw files are not distributed through
the public repository.

## Canonical game rules

Canonical NHL games are produced deterministically from immutable raw files.

Every canonical record retains:

- the raw relative path;
- the raw SHA-256;
- the provider game ID;
- the season ID;
- home and away team IDs;
- scheduled start in UTC;
- final scores;
- overtime and shootout indicators;
- source observation and ingestion timestamps.

For manually retrieved historical result files, the import timestamp is only a
proxy for when the source was observed. These records are classified as
postgame result data and cannot be used as proof of pregame availability.

Final games must explicitly identify whether the last period type was:

- `REG`;
- `OT`;
- `SO`.

The parser must not infer overtime or shootout solely from the final score.

## Five-game pilot selection

The first real pilot contains five final games selected deterministically.

The selector prioritizes:

1. one regulation game;
2. one overtime game without a shootout;
3. one shootout game;
4. the earliest remaining final games.

If one of the required outcome types is unavailable, the pilot remains usable
for structural validation but is classified as incomplete outcome coverage.

Team IDs are treated as canonical identities.

Team abbreviations are stored only as observed aliases because names and
abbreviations may change historically.

## Single-game play-by-play audit

The first play-by-play pilot uses game `2024020669`, an overtime game already
validated in the canonical result pilot.

The raw play-by-play file is saved manually and processed offline.

The structural audit records:

- top-level payload keys;
- event count;
- event types;
- regulation and overtime period types;
- event and sort-order completeness;
- duplicate event IDs;
- available detail fields;
- coordinate coverage;
- team IDs and final scoreboard values.

No event is converted into a modelling feature at this stage.

The purpose is to inspect the real provider schema before defining canonical
shot, goal, penalty or faceoff records.

## Canonical play-by-play events

Each provider play is converted into one canonical event while preserving:

- game ID;
- event ID;
- source order and sort order;
- event type;
- period number and period type;
- period clock;
- event-owning team;
- player identifiers;
- coordinates;
- shot type;
- penalty information;
- score and shot snapshots;
- raw-file lineage.

Official shots on goal are reconciled using:

`shot-on-goal events + goal events`, excluding shootout events.

Goals are reconciled against the official final score for games without a
shootout. Shootout score reconciliation requires separate settlement logic and
is deliberately not inferred from regulation and overtime goal events.

A goal without `goalieInNetId` is classified only as an
`empty_net_candidate`. It is not treated as definitively empty-net until the
situation state and goalie information are audited.

## Five-game play-by-play batch

The first PBP batch contains the five real games previously selected for the
small result pilot.

It includes:

- three regulation games;
- one overtime game;
- one shootout game.

The batch is approved only if:

- raw and canonical event counts match;
- every official shots-on-goal total reconciles;
- every non-shootout final score reconciles;
- shootout score reconciliation is explicitly not applicable;
- observed period types match the expected result type;
- regulation, overtime and shootout are all represented;
- core team-owned events retain a team identifier.

The batch retains individual raw hashes, canonical hashes and one aggregate
batch hash.

### Goals without a provider shot type

A goal is not automatically treated as a shot on goal when the provider does
not supply a `shotType`.

The five-game pilot identified one real example in game `2024020672`:

- Vancouver had 16 `shot-on-goal` events;
- Vancouver had three goal events;
- one goal had no provider `shotType`;
- the official final total was 18 shots on goal.

The canonical goal event is retained unchanged. For team-game SOG
reconciliation, non-shootout goals are counted only when the provider supplies
a shot type. Goals without a shot type are separately counted in the audit and
must not be silently deleted or reclassified.
