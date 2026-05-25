from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple

import math


Edge = Tuple[Any, Any]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
	return float(sum(x * y for x, y in zip(a, b)))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
	da = math.sqrt(sum(x * x for x in a))
	db = math.sqrt(sum(x * x for x in b))
	if da == 0.0 or db == 0.0:
		return 0.0
	return _dot(a, b) / (da * db)


ScoreFn = Literal["dot", "cosine"]


def score_edges_from_embeddings(
	embeddings: Dict[Any, Sequence[float]],
	edges: Iterable[Edge],
	*,
	method: ScoreFn = "dot",
) -> List[float]:
	"""Compute a score for each edge using node embeddings."""

	scorer = _dot if method == "dot" else _cosine
	scores: List[float] = []
	for u, v in edges:
		eu = embeddings.get(u)
		ev = embeddings.get(v)
		if eu is None or ev is None:
			scores.append(0.0)
		else:
			scores.append(float(scorer(eu, ev)))
	return scores


@dataclass(frozen=True)
class LinkPredictionReport:
	auc: float
	ap: float


def evaluate_link_prediction(
	pos_scores: Sequence[float],
	neg_scores: Sequence[float],
) -> LinkPredictionReport:
	"""Return ROC-AUC and Average Precision.

	This is a thin wrapper around sklearn metrics.
	"""

	from sklearn.metrics import average_precision_score, roc_auc_score

	y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
	y_score = list(pos_scores) + list(neg_scores)

	if len(set(y_true)) < 2:
		raise ValueError("Need both positive and negative samples")

	return LinkPredictionReport(
		auc=float(roc_auc_score(y_true, y_score)),
		ap=float(average_precision_score(y_true, y_score)),
	)

