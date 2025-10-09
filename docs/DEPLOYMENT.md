# Render Deployment Guide

This project is deployed as a Python worker on Render. Follow the steps below to ship a new version or trigger a redeploy of the existing code.

## 1. Verify prerequisites
- Render CLI installed and authenticated (`render login`).
- Access to push to the `main` branch *or* the ability to trigger redeploys from the Render dashboard.
- Required OANDA demo credentials saved as environment variables on Render (`OANDA_API_KEY`, `OANDA_ACCOUNT_ID`).

## 2. Update the codebase
1. Create a feature branch for any changes and open a pull request.
2. Run the test suite locally to confirm the worker builds clean:
   ```bash
   pytest
   ```
3. Once the PR is approved, merge it into `main`. If you do not have merge rights, ask a maintainer to merge for you.

## 3. Deploy via Render CLI
After `main` contains the latest code:
```bash
git checkout main
git pull origin main
render services redeploy mossy-4x
```
The CLI redeploy command waits for the build to finish and reports the status. Auto-deploy is enabled, so pushing to `main` also triggers the build automatically—running the command above forces an immediate redeploy if you need one right away.

## 4. Alternative deployment without CLI
If you cannot use the CLI:
1. Push your changes to `main` (or ask a maintainer to do so).
2. Visit the Render dashboard, open the **mossy-4x** service, and click **Manual Deploy → Deploy latest commit**.

## 5. Confirm the worker is live
- Check the Render logs for "Decision cycle started" entries to ensure the bot is running.
- Verify trade activity by tailing `logs/trade_activity.log` in the Render shell or running the `python -m src.live_monitor 30` helper locally with the same environment variables.

Following these steps ensures the trading bot is redeployed safely even if you do not merge directly yourself.
