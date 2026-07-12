# Source Registry

Every external source must be registered before its data enters the project.

## Audit statuses

- `pending_terms_review`: documentation or terms have not been fully audited.
- `candidate_noncommercial_source`: potentially usable for non-commercial research.
- `account_access_confirmed`: account access exists, but usage constraints remain.
- `candidate_paid_source`: paid source requiring a pilot cost and coverage audit.
- `commercial_candidate`: access requires a commercial agreement.

## Automation statuses

- `blocked_until_approved`: automated collection is prohibited.
- `approved_download_files_only`: only provider-approved downloads may be used.
- `manual_download_only`: files must be obtained manually through the account.
- `blocked_until_plan_review`: API use waits for plan and credit review.
- `blocked_until_contract`: no data collection without a contract.

## Redistribution statuses

Raw data must not be published unless redistribution rights are explicit.

The public repository may contain:

- source schemas;
- ingestion code;
- synthetic fixtures;
- file hashes;
- coverage summaries;
- aggregated metrics.

It must not contain:

- credentials;
- API keys;
- session tokens;
- certificates;
- private odds files;
- licensed raw data.
