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
