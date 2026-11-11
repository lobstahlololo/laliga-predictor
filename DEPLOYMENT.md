# Deployment guide

This project runs as two independent processes over one shared codebase:

| Process | Command | Port |
|---|---|---|
| Streamlit dashboard | `streamlit run dashboard/app.py` | 8501 |
| FastAPI service | `uvicorn src.api.main:app` | 8000 |

Both read the same artifacts (`data/`, `models/`, `reports/`) and import the
same prediction engine (`src/predictions/predictor.py`), so they cannot produce
different probabilities for the same fixture and model version.

Nothing here is a hosted deployment: the commands below are verified to start
locally and in Docker. Streamlit Community Cloud is supported by the repository
layout but was not deployed from this environment.

---

## 1. Local setup

```bash
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
# source venv/bin/activate          # Linux/macOS
python -m pip install -r requirements.txt
```

Build the artifacts once:

```bash
python main.py data
python main.py features
python main.py train
python -m src.models.registry register
```

Audit the data, coverage and evaluation before deploying:

```bash
python main.py quality
python main.py coverage
python main.py sources
python main.py evaluation
python main.py monitor
```

Then run the dashboard (`python main.py dashboard`) or API (`python main.py api`).

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LALIGA_API_HOST` | `127.0.0.1` | FastAPI bind address |
| `LALIGA_API_PORT` | `8000` | FastAPI port |
| `LALIGA_ALERT_WEBHOOK` | unset | JSON POST target for failed update cycles |
| `LALIGA_ALERT_TIMEOUT` | `10` | Webhook timeout in seconds |
| `LALIGA_FIXTURE_PROVIDER` | `thesportsdb` | `thesportsdb`, `api-football`, or a disabled value |
| `LALIGA_FIXTURE_DISCOVERY` | `1` | `0` disables network fixture discovery |
| `LALIGA_FOOTBALL_API_KEY` | unset | Optional personal provider key |
| `LALIGA_FOOTBALL_API_BASE` | provider default | Optional base URL override |
| `LALIGA_FOOTBALL_LEAGUE_ID` | `4335` / `140` | Optional league ID override |
| `LALIGA_FIXTURE_TELEMETRY` | `0` | `1` records provider freshness/failure telemetry in scheduled runs |
| `LALIGA_CACHE_BACKEND` | `memory` | `memory`, `redis`, or `none` |
| `LALIGA_CACHE_TTL` | `300` | Default cache TTL in seconds |
| `LALIGA_REDIS_URL` | `redis://localhost:6379/0` | Redis URL; unreachable Redis degrades to memory |

Copy `.env.example` to `.env`. `.env`, `.streamlit/secrets.toml` and `logs/` are
git-ignored; no credential is required for any core feature.

---

## 2. Docker

```bash
docker compose up --build
```

- Dashboard: <http://127.0.0.1:8501>
- API: <http://127.0.0.1:8000> (docs at `/docs`)

The image includes `libgomp1` for xgboost and ships the cached dataset and
trained artifacts from the repository. Both services have health checks
(`/_stcore/health` and `/health`).

Run only the API:

```bash
docker compose up --build api
```

Run one update cycle in the container:

```bash
docker compose run --rm dashboard python -m src.pipeline.automation --steps data,backfill,reports,monitoring
```

Container files are ephemeral unless you mount `data/`, `models/` and `reports/`
as volumes. Mount them if you want performance to accumulate across restarts.

---

## 3. Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create an app pointing at `dashboard/app.py`.
3. Add `LALIGA_ALERT_WEBHOOK` under *Secrets* only if you want failure alerts;
   the dashboard itself needs no secret.
4. The committed `models/saved_models/best_model.pkl`,
   `data/processed/preprocessor.joblib` and `models/registry/` must be present,
   because Community Cloud does not run the training pipeline on deploy.

Caveat: Community Cloud has an ephemeral filesystem. Predictions generated
inside the app are not persisted there. Run `update`/`predict` somewhere durable
(CI, a scheduled job, or a container with mounted volumes) if you need the ledger
to accumulate.

---

## 4. Scheduled updates

One cycle is:

```
data → quality → coverage → predict → backfill → reports → monitoring
```

and is safe to repeat: downloads merge and de-duplicate, predictions are keyed by
fixture/timestamp/model version, and reconciliation only fills blank results.
The cycle never registers, promotes or overwrites a registry snapshot; set
`LALIGA_FIXTURE_TELEMETRY=1` to record provider freshness/failure state.

### GitHub Actions (included)

`.github/workflows/update.yml` runs daily at 06:15 UTC and on demand. It runs the
data-quality and coverage audits, records provider telemetry, and uploads
`reports/`, `logs/`, `models/evaluation/` and the provider-status file as workflow
artifacts. It verifies the registry before and after the cycle and fails if any
file under `models/registry/` changed. It does not commit generated files back to
the repository; add an explicit commit step only if you intend to version the
ledger. `ci.yml` runs the test suite, registry verification, all Phase 6 audits,
the full-season evaluation and a credential-free pipeline smoke test on both
Ubuntu and Windows.

Add the webhook as a repository secret named `LALIGA_ALERT_WEBHOOK` to receive
failure notifications.

### cron (Linux/macOS)

```cron
15 6 * * * cd /path/to/laliga-predictor && ./venv/bin/python -m src.pipeline.automation --steps data,predict,backfill,reports,monitoring >> logs/cron.log 2>&1
```

### Windows Task Scheduler

Program: `C:\path\venv\Scripts\python.exe`
Arguments: `-m src.pipeline.automation --steps data,predict,backfill,reports,monitoring`
Start in: the repository root.

### Exit codes

`0` all steps ran; `1` at least one step failed (already logged, recorded in
`reports/pipeline_runs.jsonl`, and sent to the configured webhook); `2` invalid
arguments.

---

## 5. Model promotion and rollback

The registry lives at `models/registry/registry.json`, with an immutable snapshot
per version under `models/registry/versions/<version>/`.

```bash
python -m src.models.registry status
python -m src.models.registry register    # register the latest training artifacts
python -m src.models.registry list
python -m src.models.registry promote <version> --reason "validation evidence"
python -m src.models.registry rollback --reason "observed regression"
python -m src.models.registry verify
```

Promotion requires the challenger's **validation** log-loss to be strictly better
than production's and validation Brier not worse. Test metrics are recorded but
never used to decide a promotion. Missing metrics block promotion. Rollback
restores the previous snapshot and re-checks artifact hashes first. Scheduled
update cycles only refresh data, predictions and reports; they never register or
promote a model, and the workflow fails if `models/registry/` changes.

---

## 6. API

```bash
python main.py api
# or
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | artifact presence and production version |
| GET | `/model` | production metadata, metrics, feature list |
| GET | `/fixtures` | configured upcoming fixtures with full predictions |
| POST | `/predict` | one fixture: probabilities, expected goals, scorelines |
| GET | `/teams` | canonical team names |
| GET | `/teams/{team}/stats` | record, form, Elo rating, attack/defence |
| GET | `/performance` | ledger summary and grouped performance |
| GET | `/registry` | registered versions and production designation |
| GET | `/runs` | recent update-cycle records |

Errors: `404` unknown team, `422` invalid/ambiguous fixture input, `503` missing
model artifacts or an unreadable registry. OpenAPI docs are served at `/docs`.

---

## 7. Verification performed

| Check | Command | Result |
|---|---|---|
| Automated tests | `python -m pytest` | see README test count |
| Data refresh | `python -m src.pipeline.automation --steps data` | uses cache if the host is unreachable; never writes an empty dataset |
| Prediction | `python predict_upcoming.py` | appends ledger rows, prints xG/scorelines |
| Reconciliation | `python backfill_results.py` | fills only blank results |
| Dashboard | `streamlit run dashboard/app.py` | eight pages, AppTest smoke test in suite |
| API | `uvicorn src.api.main:app` | all endpoints exercised by `tests/test_api.py` |
| Clean environment | `pip install -r requirements.txt` in a fresh venv | no credentials or external services required |
| Data-quality audit | `python main.py quality` | 0 errors on the current cache; JSON/CSV/Markdown written |
| Fixture coverage | `python main.py coverage` | 1 upcoming fixture from one provider; schedule marked incomplete, not fabricated |
| Historical sources | `python main.py sources` | xG and availability unavailable; 0 network requests |
| Full-season evaluation | `python main.py evaluation` | validation-selected ensemble, one-shot 119-match test, bootstrap intervals |
| Live monitoring | `python main.py monitor` | 0 settled rows, explicitly `insufficient_data` |
| Registry | `python -m src.models.registry verify` | `random_forest-20261105T224238Z: OK`; unchanged |

Linux and Windows CI are configured in `.github/workflows/ci.yml`. They were not
executed from this Windows development environment, so this table does not claim
a Linux pass.

---

## 8. Security notes

- No secret is committed. `.env`, `.streamlit/secrets.toml`, `logs/` and
  `.pytest_cache/` are git-ignored.
- The API is read-only for artifacts; it does not write the ledger or promote
  models. Long-lived credentials are therefore not exposed through it.
- Put the API behind TLS and authentication if it is exposed publicly; it has no
  built-in auth and is intended for local/internal use.
- `POST /predict` validates team names, odds parity and date input; unknown teams
  return `422` instead of a fabricated forecast.
