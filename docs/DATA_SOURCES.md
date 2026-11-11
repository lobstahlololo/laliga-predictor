# Data sources, coverage and quality

This document states what is **real**, what is **measured**, and what is
**unavailable**. No fixture, xG value or player-availability value is fabricated.

## 1. Results and odds (real, cached)

| Source | Content | Status |
|---|---|---|
| Football-Data.co.uk `SP1` | Results, goals, shots, shots on target, Bet365 decimal odds | Cached for `23/24`, `24/25`, `25/26`, `26/27` |

Measured by `python main.py quality`:

- **1,259 matches**, 2023-08-11 to 2026-11-08.
- `23/24` 380, `24/25` 380, `25/26` 380, `26/27` **119 in progress**.
- 20 teams per season.
- **0** duplicate fixture rows, invalid scores, invalid dates, score/result
  mismatches or missing odds.
- 19 canonical features; **98.57%** of rows have every feature present.
- Largest feature missing fraction is **1.19%** (early-season rolling-form
  warm-up), below the 5% threshold.

Reports:

- `reports/data_quality.json` — machine-readable audit.
- `reports/data_quality_by_column.csv` and `data_quality_by_season.csv`.
- `reports/data_quality.md` — human-readable summary.
- `data/processed/data_quality.json` — ingestion-quality report from the refresh
  step (separate artifact).

Thresholds live in `src/data/data_quality.py` (`QualityThresholds`) and
`src/utils/config.py`. An in-progress season is allowed to be smaller than a
full season; a completed season outside 340–420 matches is flagged.

## 2. Upcoming fixtures (provider window, incomplete by design)

Two providers are implemented behind one interface in
`src/data/fixture_provider.py`:

| Provider | Endpoint | Credentials | Notes |
|---|---|---|---|
| TheSportsDB | `eventsnextleague.php?id=4335` | Free public key by default | Returns a small next-events window |
| API-Football | `fixtures?league=140&next=20` | Registered key (`LALIGA_FOOTBALL_API_KEY`) | Free plan is request-limited |

Measured by `python main.py coverage` on the current store:

- **1** stored upcoming fixture from TheSportsDB.
- 2 of 20 current teams covered; the other 18 have no returned fixture.
- 0 duplicate fixture IDs and 0 conflicting kickoff/status rows in this snapshot.
- Schedule completeness: **`unknown_partial_provider_window`**.
- The provider's "next events" response is a window, not a league calendar.
  Missing fixtures are reported as unknown, never invented. A provider returning
  no events does not mean there are no events.

Merging rules (`merge_fixtures`, `merge_provider_batches`):

- Same provider event ID updates in place.
- Same stable fixture ID (canonical teams + calendar date) is reconciled across
  providers; a kickoff-time move keeps the ID.
- An incomplete payload with no kickoff never erases a known kickoff.
- Cross-provider kickoff disagreements are preserved as conflict rows for the
  audit rather than silently choosing one.
- Stored fixtures that a provider no longer returns are kept; a partial response
  is not treated as a deletion list.
- Finished events are not returned to the upcoming store; stored history is not
  deleted.

Provider telemetry (`data/processed/fixture_provider_status.json`) records last
attempt, last success, last status, fixture count, total requests/failures and
consecutive failures. It is opt-in through `LALIGA_FIXTURE_TELEMETRY=1` for
scheduled updates.

Coverage reports:

- `reports/fixture_coverage.json`
- `reports/fixture_coverage_by_team_season.csv`
- `reports/fixture_coverage.md`

Measured team/season finding: reference season `25/26` contains Girona,
Mallorca and Oviedo; `26/27` contains La Coruna, Malaga and Santander instead.
That is a real promotion/relegation change, not a missing-data bug.

## 3. Historical xG (unavailable)

`python main.py sources` audits candidates without any network request:

| Candidate | Type | Status in this repository |
|---|---|---|
| Understat | community/scrape | Not configured; no timestamped local extract |
| FBref / StatsBomb open data | published tables | Not configured; incomplete La Liga archive |
| API-Football xG | commercial API | Not configured; free plan does not provide a verified historical xG archive |
| Local `data/raw/xg_history.csv` | repository CSV | File absent |

**No xG value is used by any trained model.** The optional feature interface
exists (`src/features/optional_features.py`) and requires:

```csv
match_id,team,date,available_at,xg_for,xg_against
```

`available_at` must be before the fixture kickoff. If it is absent, the loader
conservatively treats the observation as available one day after the match, so
it cannot enter that match's pre-kickoff features. Coverage by season/team is
reported only when an extract exists. No external coverage is claimed.

## 4. Historical player availability (unavailable)

| Candidate | Type | Status |
|---|---|---|
| API-Football injuries/lineups | commercial API | Not configured; free tier and fixture provider do not supply a historical timestamped archive |
| Transfermarkt | community scrape | Not configured; reliable publication timestamps/redistribution are unresolved |
| Local `data/raw/player_availability.csv` | repository CSV | File absent |

No lineup, injury or suspension feature is trained. The optional interface
requires `team,player,status,available_at`; only snapshots published before
kickoff are eligible. Missing snapshots stay NaN with explicit `*_known` flags
for the training-only imputer.

## 5. Adding a new source safely

1. Put a legitimate extract under `data/raw/` (do not commit restricted data).
2. Include a publication/availability timestamp for every observation.
3. Run `python main.py sources` and confirm measured coverage.
4. Run the optional-group ablation: `python main.py ablation`.
5. A group is retained only if validation log-loss improves without a Brier
   regression. The test season is never used to choose it.
6. Production promotion remains a separate evidence-gated registry action.

## 6. What remains unavailable

- A complete league-wide upcoming fixture calendar from the free providers.
- Historical xG with a verified publication timestamp.
- Historical player availability/lineups.
- Live closing-line odds for an independent betting-edge audit.
