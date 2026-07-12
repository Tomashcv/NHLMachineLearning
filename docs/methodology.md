# Methodology

## Experimental design

Only temporal and walk-forward evaluation is allowed.

The required ordering is:

1. Training period
2. Validation period
3. Untouched future test period

Random splitting is prohibited.

## Market baseline

The bookmaker or exchange market is always evaluated as a baseline.

A model must not be described as useful merely because it beats a naive baseline.

## Profitability

ROI is classified as research-only unless the price:

- existed before the decision cutoff;
- has a documented timestamp;
- matches the market settlement rules;
- was available on the relevant bookmaker or exchange;
- has an explicitly documented execution assumption.

## Model complexity

Initial models:

- Elo
- regularized logistic regression
- Poisson or negative-binomial models

Complex models are not allowed before honest baselines are complete.
