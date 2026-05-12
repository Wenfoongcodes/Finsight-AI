"""
Tests for app.ml.training.trainer
"""

from __future__ import annotations

import pytest

from app.ml.training.trainer import ModelTrainer, WalkForwardSplitter


class TestWalkForwardSplitter:
    def test_produces_correct_fold_count(self, X_y):
        X, _ = X_y
        splitter = WalkForwardSplitter(n_folds=5)
        splits = splitter.split(X)
        assert len(splits) == 5

    def test_no_overlap_between_train_and_test(self, X_y):
        X, _ = X_y
        splitter = WalkForwardSplitter(n_folds=3)
        for train_idx, test_idx in splitter.split(X):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_train_always_precedes_test(self, X_y):
        X, _ = X_y
        splitter = WalkForwardSplitter(n_folds=3)
        for train_idx, test_idx in splitter.split(X):
            assert train_idx.max() < test_idx.min()

    def test_increasing_train_sizes(self, X_y):
        X, _ = X_y
        splitter = WalkForwardSplitter(n_folds=3)
        splits = splitter.split(X)
        train_sizes = [len(t) for t, _ in splits]
        assert train_sizes == sorted(train_sizes)


class TestModelTrainer:
    def test_train_returns_model_and_result(self, X_y, tmp_path):
        X, y = X_y
        trainer = ModelTrainer(model_dir=tmp_path)
        model, result = trainer.train(
            "random_forest",
            X,
            y,
            ticker="TEST",
            hyperparams={"n_estimators": 10, "max_depth": 3},
        )
        assert model is not None
        assert result.mean_roc_auc > 0

    def test_artifacts_persisted(self, X_y, tmp_path):
        X, y = X_y
        trainer = ModelTrainer(model_dir=tmp_path)
        trainer.train(
            "random_forest",
            X,
            y,
            ticker="TEST",
            hyperparams={"n_estimators": 10},
        )
        assert (tmp_path / "TEST_random_forest.pkl").exists()
        assert (tmp_path / "TEST_random_forest_meta.json").exists()

    def test_load_model_roundtrip(self, X_y, tmp_path):
        X, y = X_y
        trainer = ModelTrainer(model_dir=tmp_path)
        trainer.train(
            "random_forest",
            X,
            y,
            ticker="TEST",
            hyperparams={"n_estimators": 10},
        )
        model, feature_cols = trainer.load_model("TEST", "random_forest")
        assert model is not None
        assert len(feature_cols) == X.shape[1]

    def test_load_model_raises_if_not_found(self, tmp_path):
        from app.core.exceptions import ModelNotFoundError

        trainer = ModelTrainer(model_dir=tmp_path)
        with pytest.raises(ModelNotFoundError):
            trainer.load_model("NONEXISTENT", "xgboost")
