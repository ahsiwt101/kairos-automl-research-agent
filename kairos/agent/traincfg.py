"""train_cfg validation, shared between the sandbox (where a bad key must be repairable)
and the scoring subprocess (where values are clipped to safe ranges).

Key-name validation MUST happen inside sandbox.run_candidate(), in the same subprocess
that runs the candidate's build(), not later in evaluate_candidate.py's separate scoring
subprocess - that subprocess has no repair loop wired to it at all, so a bad key there
silently discards the ENTIRE iteration, including a build() that ran successfully. This
cost a real live iteration: the coder wrote 'n_estimators' (an XGBoost/sklearn name, not
LightGBM's), and the whole candidate was thrown away instead of getting a one-line repair.
"""
import numpy as np

TUNABLE = ('learning_rate', 'num_leaves', 'min_data_in_leaf', 'feature_fraction',
          'bagging_fraction', 'bagging_freq', 'lambda_l1', 'lambda_l2',
          'max_depth', 'min_gain_to_split')
_BOUNDS = dict(learning_rate=(0.005, 0.3), num_leaves=(7, 255), min_data_in_leaf=(20, 2000),
              feature_fraction=(0.3, 1.0), bagging_fraction=(0.3, 1.0), bagging_freq=(0, 10),
              lambda_l1=(0.0, 10.0), lambda_l2=(0.0, 10.0), max_depth=(-1, 16),
              min_gain_to_split=(0.0, 1.0))


def validate_keys(hparams):
    """Raise immediately, with a repairable hint, on any unrecognised hyperparameter."""
    for k in (hparams or {}):
        if k not in TUNABLE:
            raise ValueError(
                f"train_cfg.hparams: '{k}' is not a tunable LightGBM parameter. Choose "
                f"from {TUNABLE}. Common mistake: 'n_estimators'/'max_leaf_nodes' are "
                f"XGBoost/sklearn names - LightGBM's boosting-round count is set by "
                f"num_boost_round (a train() argument, not a param) and its leaf-count "
                f"control is num_leaves.")


def sanitize(hparams):
    """Clip to safe ranges. Call validate_keys() first so a bad NAME is repairable;
    this only guards against a bad VALUE (which is safe to just clip)."""
    validate_keys(hparams)
    out = {}
    for k, v in (hparams or {}).items():
        lo, hi = _BOUNDS[k]
        out[k] = type(lo)(np.clip(v, lo, hi))
    return out
