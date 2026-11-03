"""
EvalRunner — base class for all eval suites.
Tracks pass/fail per case, computes aggregate score, writes JSON + Markdown reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORTS_DIR = Path("reports")
REPORT_FILE = REPORTS_DIR / "eval_results.json"
SUMMARY_FILE = REPORTS_DIR / "eval_summary.md"
GOLDEN_VERSION = "v1"


@dataclass
class CaseResult:
    name: str
    passed: bool
    score: float
    detail: str = ""


@dataclass
class EvalResult:
    eval_name: str
    threshold: float
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score for c in self.cases) / len(self.cases)

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "n_cases": len(self.cases),
            "n_passed": sum(1 for c in self.cases if c.passed),
            "cases": [
                {"name": c.name, "passed": c.passed, "score": round(c.score, 4), "detail": c.detail}
                for c in self.cases
            ],
        }


class EvalRunner:
    def __init__(self, name: str, threshold: float, golden_version: str = GOLDEN_VERSION) -> None:
        self.name = name
        self.threshold = threshold
        self.golden_version = golden_version
        self.result = EvalResult(eval_name=name, threshold=threshold)

    def add(self, case_name: str, passed: bool, score: float, detail: str = "") -> None:
        self.result.cases.append(CaseResult(name=case_name, passed=passed, score=score, detail=detail))

    def finish(self) -> EvalResult:
        return self.result

    @staticmethod
    def write_report(results: dict[str, EvalResult]) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        report: dict[str, Any] = {
            "golden_version": GOLDEN_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": {name: r.to_dict() for name, r in results.items()},
        }

        # Merge with existing report to preserve results from other eval runs.
        if REPORT_FILE.exists():
            existing = json.loads(REPORT_FILE.read_text())
            existing["timestamp"] = report["timestamp"]
            existing.setdefault("results", {}).update(report["results"])
            report = existing

        REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Write markdown summary (ASCII only for Windows compatibility).
        lines = [
            f"# Eval Results -- golden {GOLDEN_VERSION}",
            f"*{report['timestamp']}*",
            "",
            "| Eval | Score | Threshold | Pass |",
            "|------|-------|-----------|------|",
        ]
        for name, r in report["results"].items():
            icon = "PASS" if r["passed"] else "FAIL"
            lines.append(f"| {name} | {r['score']:.3f} | {r['threshold']:.2f} | {icon} |")
        SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
