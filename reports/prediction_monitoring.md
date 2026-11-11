# La Liga Predictor — Prediction Monitoring

Generated: 2026-11-11T00:33:23+00:00
Overall status: **warning**

> Live performance reliable: **False**. Minimum settled sample: 30. Below that threshold, metrics are arithmetic only and not conclusions.

## Prediction ledger

- Prediction rows: 17
- Unique fixtures: 3
- Multi-snapshot fixtures: 2
- Settled rows: 0
- Pending rows: 17
- Missing predictions: 0
- Overdue pending: 0
- Invalid probability rows: 0

## Settled performance

- No settled predictions with usable probabilities.

## Checks

| check | status | message |
|---|---|---|
| prediction_counts | ok | 17 prediction rows cover 3 fixtures; 0 settled, 17 pending. |
| missing_predictions | ok | Every configured dated upcoming fixture has a prediction. |
| prediction_errors | ok | 0 row(s) have invalid probabilities; 0 settled-age row(s) are still pending. |
| settled_performance | insufficient_data | No settled predictions yet; performance and calibration are unknown. |
| data_freshness | ok | Data and fixture freshness are within threshold. |
| fixture_coverage | warning | 1 upcoming fixture(s) stored; schedule completeness is unknown_partial_provider_window. |

## Notes

- Live metrics below the minimum settled sample are not conclusions.
- A confirmed drift label requires a sufficiently large sample and a supportive bootstrap interval.
- This monitor never retrains, promotes or rewrites a model.

