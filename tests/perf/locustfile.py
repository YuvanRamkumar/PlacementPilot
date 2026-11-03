"""
Locust performance test for PlacementPilot.

Run with FakeLLM (fast, for CI throughput baseline):
  locust -f tests/perf/locustfile.py --headless -u 10 -r 2 --run-time 30s \\
         --host http://localhost:8000 --csv reports/perf --html reports/perf.html

Run with real LLM (rate-limited, for latency profiling):
  LLM_PROVIDER=groq locust -f tests/perf/locustfile.py --headless -u 2 -r 1 \\
         --run-time 60s --host http://localhost:8000 --csv reports/perf-real
"""
from locust import HttpUser, between, task

_RESUME = (
    "Experienced Python developer with 5 years building data pipelines and REST APIs. "
    "Proficient in Apache Spark, SQL, and cloud data engineering on AWS."
)
_JD = (
    "We are looking for a Python developer with 3+ years of experience in data "
    "engineering, Apache Spark, ETL pipelines, and SQL databases."
)
_BULLET = "Built ETL pipelines in Python processing 2 TB/day across 15 sources."


class PlacementPilotUser(HttpUser):
    """Simulates a user calling the four main endpoints."""

    wait_time = between(0.5, 1.5)

    @task(4)
    def health(self) -> None:
        self.client.get("/health", name="GET /health")

    @task(3)
    def analyze(self) -> None:
        self.client.post(
            "/analyze",
            json={"resume_text": _RESUME, "jd_text": _JD},
            name="POST /analyze",
        )

    @task(2)
    def rewrite(self) -> None:
        self.client.post(
            "/rewrite",
            json={"resume_text": _RESUME, "bullet": _BULLET},
            name="POST /rewrite",
        )

    @task(1)
    def chat(self) -> None:
        self.client.post(
            "/chat",
            json={"message": "What skills should I highlight?", "history": []},
            name="POST /chat",
        )
