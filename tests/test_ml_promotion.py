"""
GeoDemand AI — Model Promotion & Rollback Test Suite (Member 3)

Dispatches to tests/unit/test_ml_promotion.py
"""

from tests.unit.test_ml_promotion import (
    dummy_validation_data,
    test_atomic_backup_and_rollback,
    test_atomic_promotion_and_synchronization,
    test_gate_evaluation_extreme_prediction_rejected,
    test_gate_evaluation_inferior_candidate_rejected,
    test_gate_evaluation_latency_exceeded_rejected,
    test_gate_evaluation_missing_keys_rejected,
    test_gate_evaluation_nan_predictions_rejected,
    test_gate_evaluation_quantile_inversion_rejected,
    test_gate_evaluation_schema_mismatch_rejected,
    test_gate_evaluation_superior_candidate_approved,
    test_promotion_failure_triggers_automatic_rollback,
    test_promotion_policy_defaults,
)

__all__ = [
    "dummy_validation_data",
    "test_promotion_policy_defaults",
    "test_gate_evaluation_superior_candidate_approved",
    "test_gate_evaluation_inferior_candidate_rejected",
    "test_gate_evaluation_schema_mismatch_rejected",
    "test_gate_evaluation_nan_predictions_rejected",
    "test_gate_evaluation_quantile_inversion_rejected",
    "test_gate_evaluation_latency_exceeded_rejected",
    "test_gate_evaluation_extreme_prediction_rejected",
    "test_gate_evaluation_missing_keys_rejected",
    "test_atomic_backup_and_rollback",
    "test_atomic_promotion_and_synchronization",
    "test_promotion_failure_triggers_automatic_rollback",
]
