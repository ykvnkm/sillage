"""Prometheus metrics: what the service is asked, and how it answers.

Request rate, latency and errors are the generic part. The part specific to this
model answers the question the roadmap asked of monitoring: **how often do real users
land where the model is weak?** Measured offline, a molecule with an unseen element
scores AP 0.270 against 0.562 (journal E-010). If live traffic hits that zone in a
third of requests, the narrow element vocabulary stops being a footnote in the model
card and becomes the main problem. These counters are how we would find out.

Each app gets its own registry, so tests and multiple app instances never share
counters through the global default.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sillage.domain import BANDS, OdorReport, Refusal

SIMILARITY_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
# Bucket edges are the confidence bands, so the histogram reads as "how many answers
# topped out at 'likely', at 'probable'..." rather than at arbitrary numbers.
PROBABILITY_BUCKETS = (*sorted(edge for edge, _ in BANDS), 1.0)


class ServiceMetrics:
    def __init__(self, model_name: str) -> None:
        self.registry = CollectorRegistry()
        r = self.registry

        model = Gauge("sillage_model_info", "The model being served", ["model"], registry=r)
        model.labels(model=model_name).set(1)

        self.http_requests = Counter(
            "sillage_http_requests_total",
            "HTTP requests by route template and status",
            ["method", "route", "status"],
            registry=r,
        )
        self.http_latency = Histogram(
            "sillage_http_request_duration_seconds",
            "Wall time per HTTP request",
            ["route"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.predictions = Counter(
            "sillage_predictions_total",
            "Prediction requests by outcome",
            ["outcome"],
            registry=r,
        )
        self.refusals = Counter(
            "sillage_refusals_total",
            "Refusals by reason: how often users ask outside the declared scope",
            ["reason"],
            registry=r,
        )
        self.prediction_latency = Histogram(
            "sillage_prediction_duration_seconds",
            "Time in the prediction use case, parsing to report",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.nearest_similarity = Histogram(
            "sillage_nearest_similarity",
            "Tanimoto similarity of each answered input to its closest known molecule: "
            "input drift shows up as this distribution moving left",
            buckets=SIMILARITY_BUCKETS,
            registry=r,
        )
        self.top_probability = Histogram(
            "sillage_top_probability",
            "Highest calibrated probability per answer, bucketed by confidence band",
            buckets=PROBABILITY_BUCKETS,
            registry=r,
        )
        self.confident_descriptors = Histogram(
            "sillage_confident_descriptors",
            "Descriptors at 'possible' or above per answer",
            buckets=(0, 1, 2, 3, 5, 8, 13),
            registry=r,
        )
        self.queries = Counter(
            "sillage_queries_total",
            "Answered requests by what the input was read as: a name or a SMILES string",
            ["kind"],
            registry=r,
        )
        self.known_molecules = Counter(
            "sillage_known_molecules_total",
            "Answers about molecules already in the dataset, by whether they were trained on",
            ["seen_in_training"],
            registry=r,
        )

    def observe(self, result: OdorReport | Refusal, seconds: float) -> None:
        self.prediction_latency.observe(seconds)
        if isinstance(result, Refusal):
            self.predictions.labels(outcome="refused").inc()
            self.refusals.labels(reason=str(result.reason)).inc()
            return

        self.predictions.labels(outcome="answered").inc()
        self.queries.labels(kind=result.query_kind).inc()
        if result.neighbours:
            self.nearest_similarity.observe(result.neighbours[0].similarity)
        confident = result.profile.confident()
        self.confident_descriptors.observe(len(confident))
        calibrated = [s.score for s in result.profile.scores if s.calibrated]
        if calibrated:
            self.top_probability.observe(max(calibrated))
        if result.known is not None:
            self.known_molecules.labels(
                seen_in_training=str(result.known.seen_in_training).lower()
            ).inc()
