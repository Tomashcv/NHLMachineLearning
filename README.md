# NHL Machine Learning

Leakage-safe probabilistic modelling and betting-market research for the NHL.

## Current status

The project is in the data-source and temporal-integrity audit phase.

No model profitability claims are made.

## Core principles

- Temporal splits and walk-forward evaluation only
- No random train/test split
- Train → validation → untouched future test
- Market probabilities are always a baseline
- No executable ROI claims without bet-time-safe odds
- Features must only use information available before puck drop
- Raw data, canonical tables, manifests and audits are preserved
- “No edge” is a valid result

## Initial scope

The first pilot will validate approximately 20–50 games across multiple seasons:

- NHL game and team IDs
- schedules and results
- regulation, overtime and shootout
- play-by-play
- shots and coordinates
- players and goalies
- market definitions
- odds timestamps
- settlement logic

## Planned first market

Full-game moneyline including overtime and shootout.

## Data policy

Raw NHL data, licensed odds data, Betfair historical files, credentials and API keys
are not distributed in this repository.

The public repository contains code, schemas, manifests, tests and aggregated reports.
