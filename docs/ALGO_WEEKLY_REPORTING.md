# Mossy 4X Weekly ALGO Reporting

This monitoring layer reads the persistent SQLite trade journal and generates a verified weekly operations report without changing the Champion strategy.

## What it reports

- completed trades, wins, losses and win rate
- net realised P/L, expectancy and profit factor
- reconstructed maximum drawdown and longest losing streak
- results by instrument, direction, session and exit reason
- journal health, open journal rows, readiness score and alerts

The report deliberately leaves macro FX intelligence, the next-week event watch list and COT positioning as external-data placeholders when no verified data source is connected.

## Schedule

On Render, the monitor checks hourly and generates one report on Sunday after 7:00 pm Australia/Perth time. Reports are written to:

- `/var/data/algo-reports/YYYY-MM-DD.md`
- `/var/data/algo-reports/YYYY-MM-DD.json`
- `/var/data/algo-reports/LATEST_ALGO_REPORT.md`

## Optional automatic GitHub publishing

Set these environment variables on the Mossy 4X Render worker:

- `GITHUB_REPORT_TOKEN`: fine-grained GitHub token with Contents read/write permission for this repository
- `GITHUB_REPORT_REPOSITORY=lukeblanc/mossy-4x-render`
- `GITHUB_REPORT_BRANCH=main`
- `ALGO_WEEKLY_REPORT_ENABLED=true`

The report is then archived to `reports/algo-weekly/YYYY-MM-DD.md` and the latest pointer is updated at `reports/algo-weekly/LATEST_ALGO_REPORT.md`.

Do not place the token in GitHub source code or `render.yaml`. Store it only as a secret environment variable in Render.

## Safety

- No entry, signal, stop, take-profit, risk or sizing code is changed.
- The monitor runs in a daemon thread and failures are caught so reporting cannot stop trading.
- Shadow auto-apply remains disabled.
- All P/L figures come from closed rows in the persistent journal; missing data is reported as missing rather than estimated.
