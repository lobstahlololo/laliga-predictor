# Fixture providers and calendar verification

Phase 7 replaces the single "next events" feed with a provider layer that can
state, and then verify, La Liga fixture coverage.

| Provider | Key | Free | Scope | Implemented |
|---|---|---|---|---|
| `openfootball` | none | yes (CC0) | full 380-match season calendar | yes |
| `thesportsdb` | shared `123`; premium optional | yes | 1-event free window | yes |
| `api-football` | registered key required | 100 req/day, 10/min | `next=20` window | yes |

`openfootball` is first in the default list because it is the only source that
publishes the whole season, which is what makes coverage verifiable. The window
providers remain for kickoff times and for when the calendar lags.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LALIGA_FIXTURE_PROVIDER` | `openfootball,thesportsdb` | Comma-separated priority list; duplicates are removed |
| `LALIGA_FIXTURE_DISCOVERY` | `1` | `0`/`false`/`off`/`no` disables all network discovery |
| `LALIGA_FOOTBALL_API_KEY` | unset | API-Football key; also the optional TheSportsDB personal key |
| `LALIGA_FOOTBALL_API_BASE` | TheSportsDB/API-Football default | Base URL override for the window providers |
| `LALIGA_FOOTBALL_LEAGUE_ID` | `4335` (TheSportsDB) / `140` (API-Football) | League ID override |
| `LALIGA_FIXTURE_CALENDAR_BASE` | `https://raw.githubusercontent.com/openfootball/football.json/master` | Calendar base URL |
| `LALIGA_FIXTURE_SEASON` | derived from today (`2026-27` in November 2026) | Season folder in the calendar dataset |
| `LALIGA_FIXTURE_TELEMETRY` | `0` | `1` records provider freshness/failure state in scheduled runs |

Semantics that are enforced in code and covered by tests:

- **Unknown provider name → error**, even when discovery is switched off. A typo
  must never look like a deliberate disable.
- `none`, `off`, `disabled`, `local` or an empty value disable discovery and make
  no request; `none` inside a list is ignored while the real entries still run.
- One provider failing does not discard another's batch; the result is `partial`
  and `failed_providers` names the outage.
- Every failure leaves the stored CSV byte-identical.

## Commands

```bash
python -m src.data.football_calendar --refresh   # cache the season calendar
python -m src.data.football_calendar             # compare cache with the store
python -m src.data.fixture_provider              # refresh fixtures from the env list
python main.py calendar                          # same as the calendar CLI
python main.py coverage                          # coverage audit incl. calendar section
```

`data/raw/season_calendar.json` is a cache with `source`, `season`, `fetched_at`
and the raw payload. It is git-ignored, so CI and fresh clones report
`unknown_partial_provider_window` until someone runs `--refresh`.

## Merge and completeness rules

- Stable fixture identity is `sha1(canonical home | canonical away | date)`;
  a kickoff-time move keeps the ID, a date change does not.
- Same provider event ID updates in place; same stable ID reconciles across
  providers.
- A calendar row without a kickoff time (`00:00Z`) merges into a timed row from
  another provider; two different real times stay as separate rows so the audit
  can report the disagreement instead of choosing one.
- Postponed, cancelled and finished statuses are preserved, never rewritten to
  `scheduled`; finished events never re-enter the upcoming store.
- A fixture a provider no longer returns is kept: a window is not a deletion list.
- `FixtureDiscoveryResult.schedule_scope` is `full_season_calendar` only when
  every successful provider covers the full season, otherwise
  `partial_provider_window`.
- The coverage audit only claims `verified_against_calendar` when a cached
  calendar exists; otherwise completeness stays unknown.

## Known limitations

- OpenFootball is community-maintained and refreshed about once a day, so it can
  lag official announcements by hours; recent results may be missing (the store
  then holds a past fixture as `scheduled`, which the audit shows but never
  treats as a result).
- Kickoff clock times are only present for part of the season; dates are always
  present. Missing times stay `00:00Z` rather than being invented.
- API-Football cannot be measured without a key, and its free plan does not cover
  the current season in 2026.
- Football-Data.co.uk `fixtures.csv` and football-data.org were evaluated but not
  integrated: the first was unreachable from this environment and publishes no
  full calendar, the second needs an account and limits free-tier schedules.
