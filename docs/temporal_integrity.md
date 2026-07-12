# Temporal Integrity

Every feature must satisfy:

feature_observed_at_utc <= decision_time_utc

Important timestamps include:

- scheduled_start_utc
- actual_start_utc
- decision_time_utc
- source_observed_at_utc
- provider_updated_at_utc
- ingested_at_utc

Starting goalies, lineups, scratches and injuries may only be used if their
availability timestamp is no later than the model decision cutoff.

The starting goalie recorded in the final boxscore is not automatically a valid
pregame feature.

Rolling features must exclude the target game and use an equivalent of shift(1).

Games sharing the same effective timestamp must not influence each other through
an arbitrary input-file ordering.
