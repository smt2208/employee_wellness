"""Configuration helpers for pipeline startup validation."""

from config.settings import (
	FATIGUE_WEIGHTS,
	STRESS_WEIGHTS,
	ENGAGEMENT_WEIGHTS,
)


def _validate_weight_table(name: str, weights: dict):
	"""Ensures an index weight table sums to ~1.0."""
	total = float(sum(weights.values()))
	if not 0.99 <= total <= 1.01:
		raise ValueError(f"{name} must sum to 1.0 (+/-0.01), got {total:.4f}")


def validate_configuration():
	"""Runs startup checks for critical configuration tables."""
	_validate_weight_table("FATIGUE_WEIGHTS", FATIGUE_WEIGHTS)
	_validate_weight_table("STRESS_WEIGHTS", STRESS_WEIGHTS)
	_validate_weight_table("ENGAGEMENT_WEIGHTS", ENGAGEMENT_WEIGHTS)
