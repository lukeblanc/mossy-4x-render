from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    main_path = root / "src" / "main.py"
    old_tp = '''            tp_distance = (
                risk.tp_distance_from_atr(atr_val, instrument=evaluation.instrument)
                if tp_enabled
                else 0.0
            )
'''
    new_tp = '''            if tp_enabled:
                tp_distance_fn = getattr(risk, "tp_distance_from_atr", None)
                if callable(tp_distance_fn):
                    tp_distance = tp_distance_fn(atr_val, instrument=evaluation.instrument)
                else:
                    # Compatibility/safety fallback for older risk-manager builds and
                    # lightweight test doubles. This preserves take-profit protection
                    # instead of crashing the entire decision cycle.
                    tp_mult = float(
                        config.get("risk", {}).get(
                            "tp_atr_mult",
                            config.get("risk", {}).get("tp_rr_multiple", 1.0),
                        )
                    )
                    tp_distance = max(0.0, float(atr_val or 0.0) * tp_mult)
            else:
                tp_distance = 0.0
'''
    replace_once(main_path, old_tp, new_tp)

    journal_path = root / "src" / "trade_journal.py"
    dead_report_block = '''    analysis_ts = datetime.now(timezone.utc)
    report_text = _format_performance_report(
        analysis_ts=analysis_ts,
        metrics=metrics,
        max_drawdown=max_drawdown,
        longest_losing_streak=longest_losing_streak,
        instrument_metrics=instrument_metrics,
    )

    report_dir = Path(os.getenv("PERFORMANCE_REPORT_DIR", "/var/data/performance_reports/"))
    report_path = _save_performance_pdf(
        report_text=report_text,
        analysis_ts=analysis_ts,
        total_trades=int(metrics["total_trades"]),
        report_dir=report_dir,
    )
    print(f"[PERFORMANCE_PDF_SAVED] path={report_path.resolve()}", flush=True)
'''
    replace_once(journal_path, dead_report_block, "")


if __name__ == "__main__":
    main()
