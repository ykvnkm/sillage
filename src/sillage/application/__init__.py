"""Use cases. They orchestrate ports and know nothing about what implements them."""

from sillage.application.predict_odor import MAX_INPUT_LENGTH, PredictOdor

__all__ = ["MAX_INPUT_LENGTH", "PredictOdor"]
