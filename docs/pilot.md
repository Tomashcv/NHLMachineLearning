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

## Forty-game play-by-play manifest

The frozen forty-game result pilot is converted into a deterministic PBP
download manifest before any larger batch is processed.

The manifest retains:

- source selection ID and SHA-256;
- exactly forty unique game IDs;
- season and game type;
- scheduled UTC start;
- expected regulation, overtime or shootout outcome;
- deterministic local filename.

A separate offline coverage audit classifies files as valid, missing, invalid
or unexpected. A partial download directory is not treated as a completed PBP
batch.

No network retrieval is performed by the project. The audit only inspects
manually saved local JSON files.

## Forty-game play-by-play batch

After local download coverage reaches forty out of forty valid files, the
generic offline PBP batch processor is applied to the frozen multiseason
manifest.

The forty-game batch requires:

- exactly forty unique game IDs;
- raw and canonical event-count equality;
- expected regulation, overtime and shootout outcomes;
- official shots-on-goal reconciliation;
- applicable final-score reconciliation;
- no missing team identifier in core team-owned events;
- deterministic per-game and aggregate hashes.

All raw JSON, canonical event files, manifests and local audits remain outside
version control.

A failed reconciliation remains visible in the aggregate audit. No tolerance
is introduced merely to force the batch to pass.

## Team-game aggregates from canonical PBP

The forty-game canonical event batch is reduced to exactly two team-game rows
per game.

Each row contains event-derived counts for:

- non-shootout goals;
- official and event-derived shots on goal;
- shot attempts, missed shots and blocked shots;
- shot coordinate coverage;
- penalties and penalty minutes;
- delayed penalties;
- faceoff wins;
- hits;
- giveaways and takeaways;
- conservative empty-net goal candidates;
- regulation, overtime and shootout owned-event counts.

Official team-game score and shots-on-goal values remain separate from
event-derived values.

Shootout final-score reconciliation is marked as not applicable because the
official winner receives a shootout score increment that is not a normal goal
event.

A goal without a provider shot type remains a goal and a shot-attempt event,
but is not counted as a provider-defined shot on goal.

These rows are post-game aggregates. They are not yet safe pre-game model
features. Any later rolling feature must use only rows from games completed
before the prediction timestamp.

## Event-owner semantic validation

The meaning of `eventOwnerTeamId` is not assumed to be identical across event
types.

For blocked shots, faceoffs, hits, giveaways and takeaways, the owner team is
compared with the team of every relevant player role using the raw
`rosterSpots` mapping.

The audit infers the owner role only when:

- all relevant player identifiers are present;
- every player maps to one of the two teams;
- the owner team is valid;
- multi-player roles belong to opposing teams;
- exactly one candidate role matches the owner team for every event of that
  type.

This validation determines whether existing aggregate names represent actions
performed by a team or actions attempted against an opponent. No metric is
accepted as a modelling feature solely because its provider event name appears
intuitive.

## Corrected blocked-shot aggregate semantics

The forty-game player-team semantic audit confirmed that
`eventOwnerTeamId` on a `blocked-shot` event identifies the team of the
shooting player.

Therefore, an owner-team `blocked-shot` count represents a shot attempt by
that team that was blocked. It does not represent a defensive block made by
that team.

The aggregate field formerly named `blocked_shots` is renamed to
`blocked_shot_attempts`.

No `blocks_made` metric is inferred from the opponent team because some
blocked-shot events contain shooting and blocking players mapped to the same
team. A defensive-block metric would require a separately validated
player-role aggregation.

The following identity is enforced for every team-game row:

`shot_attempt_events = pbp_shots_on_goal + goals_without_shot_type
+ missed_shots + blocked_shot_attempts`

Goals without provider shot type remain shot attempts, even though they are
excluded from provider-defined shots on goal.

## Leakage-safe pre-game rolling team features

Team-game post-match aggregates are transformed into pre-game rolling
features using a conservative temporal availability rule.

For a game on a given UTC calendar date, eligible history is restricted to:

- the same team;
- the same NHL season;
- games scheduled on strictly earlier UTC calendar dates.

Games from the same UTC date are excluded even when their scheduled start time
is earlier. This avoids relying on an assumed game duration or an unavailable
verified completion timestamp.

The feature output includes:

- season-to-date history;
- last-three-game history;
- last-five-game history;
- win rate;
- goals for and against;
- shots on goal for and against;
- shot attempts for and against;
- blocked shot attempts for and against;
- shooting percentage;
- save-percentage proxy;
- shot-attempt share;
- faceoff win percentage;
- penalties and penalty minutes;
- hits, giveaways and takeaways;
- previous-game timing;
- exact game-ID lineage for every rolling window.

No current-game result or current-game PBP statistic is included in the
pre-game feature row.

The forty-game pilot validates the temporal and lineage machinery only. Its
sparse selected-game history is not sufficient for model training or betting
evaluation.

## Season-scale regular-season target corpus

The successful forty-game pilot validates the raw, canonical, semantic,
team-game and temporal feature machinery, but it is too sparse for modelling.

The first season-scale target contains the complete regular-season game-ID
ranges for:

- 2021-22: development;
- 2022-23: development;
- 2023-24: validation;
- 2024-25: sealed holdout.

Each season contains 1,312 candidate regular-season IDs, for a total target of
5,248 games.

The target manifest is intentionally labelled `candidate_unverified`.
Constructing a syntactically valid game ID does not prove that its payload,
date, teams, state or event contents are valid.

A candidate becomes locally valid only when its PBP payload confirms:

- the exact game ID;
- the expected season;
- regular-season game type;
- a final game state;
- a start timestamp;
- distinct home and away team IDs;
- a non-empty play list.

Playoffs are excluded from the first model corpus because their scheduling,
incentives, overtime rules and matchup structure differ from the regular
season.

The 2024-25 season remains sealed. Its outcomes may be stored and processed
mechanically, but must not influence feature selection, hyperparameters,
calibration choices or betting rules.

## Verified season-scale game inventory

The 5,248 candidate regular-season IDs are converted into a verified local
inventory before season-scale event processing begins.

Each local PBP payload must confirm:

- exact game ID and expected season;
- regular-season game type;
- final provider state;
- timezone-aware scheduled start;
- distinct home and away teams;
- official final scores and shots on goal;
- supported final period type: regulation, overtime or shootout;
- non-empty play and roster lists;
- roster teams restricted to the two participating teams.

The verified inventory records source filename, path, byte size and SHA-256
for every game. It also freezes the expected REG, OT or SO outcome later used
by canonical PBP reconciliation.

The sealed-holdout season is inventoried mechanically. Its outcomes remain
unavailable for model-selection decisions.

## Season-scale PBP batches

The verified 5,248-game inventory is divided into one immutable PBP batch per
regular season.

Each season config freezes:

- exactly 1,312 game IDs;
- expected regulation, overtime or shootout outcome;
- scheduled start timestamp;
- home and away team IDs;
- local source filename and source SHA-256;
- split role;
- a deterministic hash of the season-specific inventory subset.

Before processing, the config is reconstructed from the verified inventory and
must match exactly. This prevents a manually edited config from silently
changing game membership, outcomes or source lineage.

The 2021-22 and 2022-23 batches are development data. The 2023-24 batch is
validation data. The 2024-25 batch is a sealed holdout and must not influence
feature, model, calibration or betting-rule selection.

## Season-scale team-game aggregates

The four verified regular seasons produce deterministic post-game team-level
aggregates before any rolling pregame features are calculated.

Results:

- 5,248 games;
- 10,496 team-game rows;
- exactly two rows per game;
- 1,647,137 canonical PBP events aggregated;
- 32 participating teams per season;
- all score, event-owner and shot-attempt identity gates passed.

Official boxscore shots on goal remain the authoritative aggregate value.
The PBP-derived value, PBP-minus-official delta and provider-correction flag
are preserved separately for auditability.

Seventeen 2021-22 games use the previously audited provider correction policy:
exactly one team has PBP SOG equal to official SOG plus one. All remaining
5,231 games reconcile exactly.

The 2024-25 sealed holdout was transformed mechanically and remains excluded
from feature, model, calibration and policy selection.

## Season-scale leakage-safe pregame features

A deterministic combined panel was built from four regular seasons:

- 5,248 games;
- 10,496 team-game source rows;
- 10,496 pregame feature rows;
- 128 team-season groups;
- 82 games in every team-season group.

History is restricted to the same team and season and to strictly earlier UTC
dates. Games starting earlier on the same UTC date are deliberately excluded
because their post-game information may not have been safely available at the
current decision time.

The audit recorded:

- 10,367 rows with at least one eligible history game;
- 10,112 rows with at least three eligible history games;
- 9,855 rows with at least five eligible history games;
- 86 same-UTC-date history candidates excluded;
- zero current-game, future-date, cross-season, wrong-team or missing-lineage
  references.

Official boxscore shots on goal are used for rolling features. Canonical PBP
goals excluding shootouts remain the goals source.

Combined team-game SHA-256:
`071edf90e1bec9b5003da74b70a82beb6a0d91e534f563995aaa3e974d1ec179`

Pregame feature SHA-256:
`74f5cdad379ba3ab8e25b0c8549a4b91bc63de912684ce081ef5d5715c98e3ad`

The 2024-25 sealed holdout was transformed mechanically and remains excluded
from feature, model, calibration and policy selection.
