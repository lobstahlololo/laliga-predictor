# La Liga Predictor — Historical xG and Player Availability Source Audit

Generated: 2026-11-11T00:27:21+00:00

No network requests were made. 'Available' means a local timestamped extract exists; external provider coverage is not claimed.

## Candidate sources

| name | kind | provider_type | credentials | status |
|---|---|---|---|---|
| Football-Data.co.uk SP1 | results_and_odds | published CSV | none | available_local_extract |
| Understat | xg | community dataset / scrape | none | not_configured |
| FBref / StatsBomb data | xg | published tables / open datasets | none for public pages | not_configured |
| API-Football (xG) | xg | commercial API | API key required | not_configured |
| API-Football (injuries/lineups) | player_availability | commercial API | API key required | not_configured |
| Transfermarkt | player_availability | community scrape | none | not_configured |
| Local xG history extract | xg | repository CSV | none | not_configured |
| Local player-availability extract | player_availability | repository CSV | none | not_configured |

## Measured local coverage

### xg

- Available: **False**
- Source rows: 0
- Seasons: none
- Teams: 0
- Matches with a pre-kickoff observation: 0
- Mean team coverage fraction: None

### player_availability

- Available: **False**
- Source rows: 0
- Seasons: none
- Teams: 0
- Matches with a pre-kickoff observation: 0
- Mean team coverage fraction: None

## Findings

- **xg**: No timestamped historical xG extract is present. The xG feature group is unavailable and the production model is unchanged.
- **player_availability**: No timestamped player-availability extract is present. The availability feature group is unavailable and no values are fabricated.

## What remains unavailable

- No historical xG extract is committed, so xG-based models are not trained and no xG improvement is claimed.
- No historical player-availability archive is committed, so lineup/injury features are not trained.
- The fixture APIs used elsewhere are not historical xG or availability providers and are not described as such.

