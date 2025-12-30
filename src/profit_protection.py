"""Profit Open
Reconcile profit protection when broker position already closed
#77
lukeblanc wants to merge 1 commit into main from codex/fix-broker-position-close-handling-ylsgth 
+201 −33 
 Conversation 1
 Commits 1
 Checks 0
 Files changed 2
Conversation
@lukeblanc
Owner
lukeblanc
commented
1 minute ago
Summary
treat CLOSEOUT_POSITION_DOESNT_EXIST and related reject reasons as successful closes, reconciling local state and marking trades closed without retries
guard profit protection from re-closing locally closed trades while logging broker-confirmed vs broker-missing-position outcomes at INFO level
expand profit protection tests to cover missing-position handling, snapshot hints, and no-repeat close attempts
Testing
pytest tests/test_profit_protection.py -q
Codex Task

@lukeblanc
Treat missing broker positions as closed in profit protection
db4792f
@lukeblanc lukeblanc added the codex label 1 minute ago — with  ChatGPT Codex Connector
@chatgpt-codex-connector
chatgpt-codex-connector bot
commented
1 minute ago
You have reached your Codex usage limits for code reviews. You can see your limits in the Codex usage dashboard.
To continue using code reviews, you can upgrade your account or add credits to your account and enable them for code reviews in your settings.

Merge info
This branch has conflicts that must be resolved
Use the web editor or the command line to resolve conflicts before continuing.

src/profit_protection.py
tests/test_profit_protection.py
You can also merge this with the command line. 
Still in progress?
@lukeblanc


Add a comment
Comment
 
Add your comment here...
 
Remember, contributions to this repository should follow our GitHub Community Guidelines.
 ProTip! Add .patch or .diff to the end of URLs for Git’s plaintext views.
Reviewers
No reviews
Still in progress?
Assignees
No one—
Labels
codex
Projects
None yet
Milestone
No milestone
Development
Successfully merging this pull request may close these issues.

None yet


Notifications
Customize
You’re receiving notifications because you authored the thread.
1 participant
@lukeblanc
Footer
© 2025 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
                    f"{log_prefix}[INFO] Trade already closed at broker; marking closed ticket={trade_id} instrument={instrument}{spread_clause}",
                    flush=True,
                )
                return True
            if closed_status is None and self._response_indicates_missing_position(result):
                self._mark_locally_closed(trade_id, instrument)
                print(
                    f"{log_prefix}[INFO] Broker response indicates no open position; marking closed ticket={trade_id} instrument={instrument}{spread_clause}",
                    flush=True,
                )
                return True
            print(
                f"{log_prefix}[WARN] Broker reported CLOSEOUT_POSITION_DOESNT_EXIST but {instrument} still appears open; ticket={trade_id} resp={result}{spread_clause}",
                flush=True,
            )
        else:
            # If the broker did not acknowledge the close, perform a follow-up check
            # to see whether the position already disappeared (e.g., previously closed
            # or closed by another rule). Only then can we safely treat it as closed.
            if closed_status is True:
                self._mark_locally_closed(trade_id, instrument)
                print(
                    f"{log_prefix}[INFO] Broker confirmed close via snapshot ticket={trade_id} instrument={instrument}{spread_clause}",
                    flush=True,
                )
                return True

        print(
            f"{log_prefix}[WARN] Close failed ticket={trade_id} {metric_clause} floor={floor:.2f} "
            f"high_water={high_water:.2f} reason={reason} resp={result}{spread_clause}",
            flush=True,
        )
        return False

    @staticmethod
    def _extract_error_code(result: Dict) -> Optional[str]:
        if not isinstance(result, dict):
            return None
        for key in ("errorCode", "error_code"):
            if key in result:
                val = result.get(key)
                if isinstance(val, str):
                    return val
        text = result.get("text")
        if isinstance(text, str):
            try:
                import json

                parsed = json.loads(text)
                for key in ("errorCode", "error_code"):
                    val = parsed.get(key)
                    if isinstance(val, str):
                        return val
            except Exception:
                return None
        return None

    def _response_indicates_missing_position(self, result: Dict) -> bool:
        """Return True when the broker response clearly states no position exists."""

        if not isinstance(result, dict):
            return False

        payload = None
        text = result.get("text")
        if isinstance(text, str):
            try:
                payload = json.loads(text)
            except Exception:
                if "CLOSEOUT_POSITION_DOESNT_EXIST" in text or "POSITION_CLOSEOUT_DOESNT_EXIST" in text:
                    return True

        payload = payload or result

        for key in ("errorCode", "error_code"):
            code = payload.get(key)
            if isinstance(code, str) and code == "CLOSEOUT_POSITION_DOESNT_EXIST":
                return True

        for leg in ("longOrderRejectTransaction", "shortOrderRejectTransaction"):
            reject_reason = (payload.get(leg) or {}).get("rejectReason")
            if isinstance(reject_reason, str) and (
                "CLOSEOUT_POSITION_DOESNT_EXIST" in reject_reason or "POSITION_CLOSEOUT_DOESNT_EXIST" in reject_reason
            ):
                return True

        message = payload.get("errorMessage")
        if isinstance(message, str) and "does not exist" in message:
            return True

        return False

    @staticmethod
    def _raw_units(trade: Dict) -> Optional[float]:
        """Return the raw units value if present, preserving zeros."""

        for key in ("currentUnits", "current_units", "units"):
            if key in trade:
                return trade.get(key)
        return None

    def _broker_confirms_closed(self, trade_id: Optional[str], instrument: str) -> Optional[bool]:
        """Return True only if broker reports no open position for the instrument.

        Returns False when the instrument is still present. Returns None when the broker
        cannot confirm (missing capability or transient failure).
        """

        try:
            if not hasattr(self.broker, "list_open_trades"):
                return None
            trades = self.broker.list_open_trades()
            if trades is None:
                return None
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[TRAIL][WARN] Unable to confirm closure for {instrument}: {exc}",
                flush=True,
            )
            return None

        return not self._instrument_open_in_snapshot(trades, instrument, trade_id)

    def _pip_size(self, instrument: str) -> float:
        try:
            return float(self.broker._pip_size(instrument))
        except Exception:
            if instrument.endswith("JPY"):
                return 0.01
            if instrument.startswith("XAU"):
                return 0.1
            if instrument.startswith("XAG"):
                return 0.01
            return 0.0001

    def _current_spread(self, instrument: str) -> Optional[float]:
        try:
            spread = self.broker.current_spread(instrument)
            return None if spread is None else float(spread)
        except Exception:
            return None

    def _list_open_trades_quietly(self) -> Optional[List[Dict]]:
        try:
            if not hasattr(self.broker, "list_open_trades"):
                return None
            return self.broker.list_open_trades()
        except Exception:
            return None

    def _instrument_open_in_snapshot(
        self, trades: Optional[List[Dict]], instrument: str, trade_id: Optional[str]
    ) -> bool:
        for trade in trades or []:
            inst = trade.get("instrument")
            if instrument and inst != instrument:
                continue
            raw_units = self._raw_units(trade)
            if raw_units is not None:
                units = self._units_from_trade(trade)
                if units == 0:
                    continue
            if trade_id is None:
                return True
            live_id = trade.get("id") or trade.get("tradeID") or trade.get("position_id")
            if live_id is None:
                return True
            if str(live_id) == str(trade_id):
                return True
        return False

    def _reconcile_closed(
        self,
        trade_id: Optional[str],
        instrument: str,
        open_trades: Optional[List[Dict]],
        state: Optional[TrailingState],
    ) -> None:
        state = state or self._state.get(trade_id or "")
        if state:
            state.armed = False
            state.max_profit_ccy = None
            state.last_update = None
            state.open_time = None
        self._mark_locally_closed(trade_id, instrument)
        if open_trades is not None:
            remaining = []
            for trade in open_trades:
                tid = self._trade_id(trade)
                inst = trade.get("instrument")
                if (trade_id is not None and tid is not None and str(tid) == str(trade_id)) or (
                    instrument and inst == instrument
                ):
                    if isinstance(trade, dict):
                        trade["state"] = "CLOSED"
                    continue
                remaining.append(trade)
            open_trades[:] = remaining
        if trade_id is not None:
            self._state.pop(trade_id, None)
