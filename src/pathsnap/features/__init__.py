"""Candidate-level feature construction and supervised assembly."""

from .engineering import build_feature_table, attach_reference_labels, reference_candidate_coverage

__all__ = ["build_feature_table", "attach_reference_labels", "reference_candidate_coverage"]
