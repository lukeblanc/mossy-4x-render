# ishy-bot

Ishy Bot is a small Flask service that runs the Mushy Merch automation agent on a schedule.
It exposes simple HTTP endpoints for manual triggering and health checks while a background
scheduler polls the storefront, reasons about the latest inventory state using OpenAI, and
pushes actionable instructions to your operations channel.

## Features

- Flask API with `/health` and `/run` endpoints
- APScheduler background job to execute the Mushy Merch agent on an interval
- OpenAI Responses API integration for reasoning about storefront inventory
- Webhook delivery so instructions can reach Slack, Discord, or another downstream system
- Render deployment configuration for one-click hosting

## Requirements

- Python 3.11+
- An OpenAI API key with access to the configured model
- (Optional) A JSON API endpoint that returns current inventory/status data

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | API key used to authenticate with OpenAI. |
| `OPENAI_RESPONSE_MODEL` | No | OpenAI model name for the reasoning step (default: `gpt-4o-mini`). |
| `MUSHY_MERCH_WEBHOOK_URL` | No | HTTPS endpoint that receives the generated instructions. |
| `STORE_STATUS_URL` | No | Endpoint returning JSON payload with current inventory/status. |
| `CHECK_INTERVAL_MIN` | No | Minutes between automated runs (default: `30`). |
| `LOG_LEVEL` | No | Python logging level (default: `INFO`). |

## Local development

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` (create one if needed) to `.env` and populate environment variables.
3. Run the Flask server:
   ```bash
   python app.py
   ```
4. Manually trigger the agent with:
   ```bash
   curl -X POST http://localhost:8000/run
   ```

## Render deployment

The repository includes a `render.yaml` blueprint and GitHub Actions workflow to automate
deployments.

1. Create a new Render Web Service from the repo, or use the CLI:
   ```bash
   render services create web --name ishy-web \
     --env python \
     --branch main \
     --build-command "pip install -r requirements.txt" \
     --start-command "gunicorn app:app -b 0.0.0.0:$PORT"
   ```
2. Add the environment variables listed above (at minimum `OPENAI_API_KEY`).
3. Enable automatic deploys on Render.

## GitHub Actions deployment

The workflow at `.github/workflows/deploy.yml` logs in to Render using the
`RENDER_API_KEY` and `RENDER_SERVICE_NAME` secrets, then triggers a deployment
of the configured service on pushes to `main` or when run manually.

## Manual run via Render CLI

If you want to trigger the job locally using the same logic as the scheduler, run:
```bash
python app.py --once
```
(you can also POST to the `/run` endpoint as shown above).

## License

MIT
