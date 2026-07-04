"""
Unit tests for ``app.ml.explainability.SHAPExplainer``.

The real ``shap`` package is optional/heavy (and not required for these
tests to exercise FinSight's own logic), so a minimal fake ``shap`` module
is injected into ``sys.modules`` before import. This tests OUR unwrapping,
normalisation, and narrative logic — not SHAP's internals.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import ExplainabilityError

# ─────────────────────────────────────────────────────────────────────────────
# Fake shap module
# ─────────────────────────────────────────────────────────────────────────────


class _FakeTreeExplainer:
    """Returns a deterministic (n, features) SHAP array."""

    def __init__(self, model, data=None):
        self.model = model
        self.expected_value = [0.4, 0.6]  # [class0, class1] baseline

    def shap_values(self, X):
        n, f = X.shape
        # Deterministic values: feature i gets weight (i+1)/10, alternating sign.
        base_row = np.array([((-1) ** i) * (i + 1) / 10.0 for i in range(f)])
        return np.tile(base_row, (n, 1))


class _FakeKernelExplainer:
    def __init__(self, predict_fn, background):
        self.predict_fn = predict_fn
        self.expected_value = 0.5

    def shap_values(self, X):
        n, f = X.shape
        return np.tile(np.linspace(0.1, 0.05, f), (n, 1))


@pytest.fixture
def fake_shap(monkeypatch):
    fake_module = SimpleNamespace(
        TreeExplainer=_FakeTreeExplainer,
        KernelExplainer=_FakeKernelExplainer,
        sample=lambda data, n: data.iloc[:n] if hasattr(data, "iloc") else data,
    )
    monkeypatch.setitem(sys.modules, "shap", fake_module)
    yield fake_module


@pytest.fixture
def feature_columns():
    return ["rsi_14", "macd", "bb_upper", "obv"]


@pytest.fixture
def sample_X(feature_columns):
    return pd.DataFrame(
        np.random.rand(10, len(feature_columns)), columns=feature_columns
    )


# ─────────────────────────────────────────────────────────────────────────────
# Explainer selection / unwrapping
# ─────────────────────────────────────────────────────────────────────────────


class TestExplainerSelection:
    def test_tree_based_model_uses_tree_explainer(
        self, fake_shap, feature_columns, sample_X
    ):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        explainer = SHAPExplainer(XGBClassifier(), feature_columns)
        explainer._build_explainer(sample_X)
        assert isinstance(explainer._explainer, _FakeTreeExplainer)

    def test_non_tree_model_uses_kernel_explainer(
        self, fake_shap, feature_columns, sample_X
    ):
        from app.ml.explainability import SHAPExplainer

        class LogisticRegression:
            def predict_proba(self, X):
                return np.column_stack([1 - X.iloc[:, 0], X.iloc[:, 0]])

        explainer = SHAPExplainer(LogisticRegression(), feature_columns)
        explainer._build_explainer(sample_X)
        assert isinstance(explainer._explainer, _FakeKernelExplainer)

    def test_pipeline_unwraps_named_steps_clf(self, fake_shap, feature_columns):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        pipeline = SimpleNamespace(named_steps={"clf": XGBClassifier()})
        explainer = SHAPExplainer(pipeline, feature_columns)
        underlying = explainer._get_underlying_model()
        assert type(underlying).__name__ == "XGBClassifier"

    def test_calibrated_classifier_unwraps_to_base_estimator(
        self, fake_shap, feature_columns
    ):
        from app.ml.explainability import SHAPExplainer

        class RandomForestClassifier:
            pass

        calibrated = SimpleNamespace(
            calibrated_classifiers_=[
                SimpleNamespace(estimator=RandomForestClassifier())
            ]
        )
        explainer = SHAPExplainer(calibrated, feature_columns)
        underlying = explainer._get_underlying_model()
        assert type(underlying).__name__ == "RandomForestClassifier"

    def test_explainer_is_built_only_once(self, fake_shap, feature_columns, sample_X):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        explainer = SHAPExplainer(XGBClassifier(), feature_columns)
        explainer._build_explainer(sample_X)
        first = explainer._explainer
        explainer._build_explainer(sample_X)
        assert explainer._explainer is first  # not rebuilt

    def test_explainer_construction_failure_raises_explainability_error(
        self, fake_shap, feature_columns, sample_X, monkeypatch
    ):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        def boom(*args, **kwargs):
            raise RuntimeError("shap internal failure")

        monkeypatch.setattr(fake_shap, "TreeExplainer", boom)
        explainer = SHAPExplainer(XGBClassifier(), feature_columns)
        with pytest.raises(ExplainabilityError):
            explainer._build_explainer(sample_X)


# ─────────────────────────────────────────────────────────────────────────────
# compute_shap_values — normalisation across SHAP output formats
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeShapValues:
    def _tree_explainer_instance(self, fake_shap, feature_columns):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        return SHAPExplainer(XGBClassifier(), feature_columns)

    def test_returns_2d_array_matching_input_shape(
        self, fake_shap, feature_columns, sample_X
    ):
        explainer = self._tree_explainer_instance(fake_shap, feature_columns)
        result = explainer.compute_shap_values(sample_X)
        assert result.shape == (len(sample_X), len(feature_columns))

    def test_legacy_list_format_uses_positive_class_index(
        self, fake_shap, feature_columns, sample_X, monkeypatch
    ):
        explainer = self._tree_explainer_instance(fake_shap, feature_columns)
        explainer._build_explainer(sample_X)

        class0 = np.zeros((len(sample_X), len(feature_columns)))
        class1 = np.ones((len(sample_X), len(feature_columns))) * 0.5
        monkeypatch.setattr(
            explainer._explainer, "shap_values", lambda X: [class0, class1]
        )

        result = explainer.compute_shap_values(sample_X)
        assert np.allclose(result, 0.5)

    def test_3d_array_format_takes_positive_class_slice(
        self, fake_shap, feature_columns, sample_X, monkeypatch
    ):
        explainer = self._tree_explainer_instance(fake_shap, feature_columns)
        explainer._build_explainer(sample_X)

        n, f = len(sample_X), len(feature_columns)
        arr_3d = np.zeros((n, f, 2))
        arr_3d[:, :, 1] = 0.75
        monkeypatch.setattr(explainer._explainer, "shap_values", lambda X: arr_3d)

        result = explainer.compute_shap_values(sample_X)
        assert np.allclose(result, 0.75)

    def test_max_samples_caps_row_count_passed_to_explainer(
        self, fake_shap, feature_columns
    ):
        explainer = self._tree_explainer_instance(fake_shap, feature_columns)
        big_X = pd.DataFrame(
            np.random.rand(1000, len(feature_columns)), columns=feature_columns
        )
        result = explainer.compute_shap_values(big_X, max_samples=50)
        assert result.shape[0] == 50

    def test_unexpected_shape_raises_explainability_error(
        self, fake_shap, feature_columns, sample_X, monkeypatch
    ):
        explainer = self._tree_explainer_instance(fake_shap, feature_columns)
        explainer._build_explainer(sample_X)
        # 1-D output is invalid.
        monkeypatch.setattr(
            explainer._explainer, "shap_values", lambda X: np.zeros(len(sample_X))
        )
        with pytest.raises(ExplainabilityError):
            explainer.compute_shap_values(sample_X)


# ─────────────────────────────────────────────────────────────────────────────
# global_feature_importance / local_explanation
# ─────────────────────────────────────────────────────────────────────────────


class TestGlobalFeatureImportance:
    def test_returns_sorted_dataframe_with_expected_columns(
        self, fake_shap, feature_columns, sample_X
    ):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        explainer = SHAPExplainer(XGBClassifier(), feature_columns)
        df = explainer.global_feature_importance(sample_X, top_n=3)

        assert list(df.columns) == ["feature", "mean_abs_shap"]
        assert len(df) == 3
        # Descending sort by importance.
        assert (df["mean_abs_shap"].diff().dropna() <= 0).all()


class TestLocalExplanation:
    def test_local_explanation_returns_expected_keys(
        self, fake_shap, feature_columns, sample_X
    ):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        explainer = SHAPExplainer(XGBClassifier(), feature_columns)
        single_row = sample_X.iloc[[0]]
        result = explainer.local_explanation(single_row, top_n=2)

        assert set(result.keys()) == {
            "base_value",
            "prediction_probability",
            "predicted_class",
            "top_features",
        }
        assert len(result["top_features"]) == 2
        for feat in result["top_features"]:
            assert feat["direction"] in {"bullish", "bearish"}

    def test_top_features_sorted_by_absolute_shap_value(
        self, fake_shap, feature_columns, sample_X
    ):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        explainer = SHAPExplainer(XGBClassifier(), feature_columns)
        single_row = sample_X.iloc[[0]]
        result = explainer.local_explanation(single_row, top_n=4)

        abs_values = [abs(f["shap_value"]) for f in result["top_features"]]
        assert abs_values == sorted(abs_values, reverse=True)


class TestGenerateNarrative:
    def test_narrative_uses_authoritative_prediction_over_shap_internal(
        self, fake_shap, feature_columns, sample_X
    ):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        explainer = SHAPExplainer(XGBClassifier(), feature_columns)
        local_exp = explainer.local_explanation(sample_X.iloc[[0]])

        # Force a narrative claiming BEARISH even if SHAP-internal said bullish.
        narrative = explainer.generate_narrative(
            local_exp,
            ticker="AAPL",
            authoritative_prediction=0,
            authoritative_p_bullish=0.15,
        )
        assert "BEARISH" in narrative
        assert "AAPL" in narrative

    def test_narrative_falls_back_to_shap_internal_when_no_override(
        self, fake_shap, feature_columns, sample_X
    ):
        from app.ml.explainability import SHAPExplainer

        class XGBClassifier:
            pass

        explainer = SHAPExplainer(XGBClassifier(), feature_columns)
        local_exp = explainer.local_explanation(sample_X.iloc[[0]])

        narrative = explainer.generate_narrative(local_exp, ticker="MSFT")
        assert "MSFT" in narrative
        assert ("BULLISH" in narrative) or ("BEARISH" in narrative)
