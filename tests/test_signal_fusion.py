"""
Unit tests for ``app.services.signal_fusion.SignalFusionService``.

The service's contract is that it must NEVER raise to the caller and must
degrade through three tiers:

    1. LLM fusion succeeds                       -> fusion_applied=True
    2. News retrieved but LLM fails / bad JSON    -> rule-based fallback
    3. News retrieval fails / empty               -> ML-only signal

All three tiers are exercised here with a mocked LLM client and a mocked
``FinancialIntelligenceService`` so no network or API key is required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import LLMError
from app.services.news_intelligence import IntelligenceBrief, NewsItem
from app.services.signal_fusion import SignalFusionService

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_prediction_response(
    prediction: int = 1, p_bullish: float = 0.68, confidence_label: str = "high"
) -> SimpleNamespace:
    """
    Minimal stand-in for ``PredictionResponse``.

    Signal fusion only touches ``.prediction``, ``.p_bullish``,
    ``.confidence_label``, and ``.shap_explanation`` — a SimpleNamespace
    avoids depending on the full pydantic schema (and its validators) here.
    """
    return SimpleNamespace(
        prediction=prediction,
        p_bullish=p_bullish,
        confidence_label=confidence_label,
        shap_explanation={
            "top_features": [
                {"feature": "rsi_14", "shap_value": 0.12},
                {"feature": "macd", "shap_value": -0.05},
            ]
        },
    )


def _make_news_item(sentiment: str = "positive") -> NewsItem:
    return NewsItem(
        title="Company beats earnings expectations",
        snippet="Strong quarter driven by cloud growth.",
        url="https://reuters.com/article/123",
        sentiment=sentiment,
        sentiment_score=0.4 if sentiment == "positive" else -0.4,
        final_weight=0.8,
    )


def _make_brief(
    retrieval_success=True,
    top_news=None,
    aggregate_sentiment="positive",
    sentiment_score=0.4,
) -> IntelligenceBrief:
    return IntelligenceBrief(
        ticker="AAPL",
        situation_summary="Positive earnings momentum.",
        aggregate_sentiment=aggregate_sentiment,
        sentiment_score=sentiment_score,
        top_news=top_news if top_news is not None else [_make_news_item()],
        retrieval_success=retrieval_success,
        recency_note="last 3 days",
    )


@pytest.fixture
def mock_intel_service():
    svc = MagicMock()
    svc.get_brief.return_value = _make_brief()
    return svc


@pytest.fixture
def fusion_service(mock_intel_service):
    llm = MagicMock()
    return SignalFusionService(
        llm_client=llm, intelligence_service=mock_intel_service
    ), llm


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: LLM fusion succeeds
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMFusionSuccess:
    def test_valid_llm_json_produces_fused_signal(self, fusion_service):
        service, llm = fusion_service
        llm.chat.return_value = (
            json.dumps(
                {
                    "final_direction": "BULLISH",
                    "final_confidence": "HIGH",
                    "fusion_probability": 0.74,
                    "news_sentiment": "positive",
                    "synthesis_narrative": "ML and news both bullish; strong conviction.",
                }
            ),
            123,
        )

        result = service.fuse("AAPL", _make_prediction_response(), horizon="1d")

        assert result.fusion_applied is True
        assert result.final_direction == "BULLISH"
        assert result.final_confidence == "HIGH"
        assert result.fusion_probability == pytest.approx(0.74)
        assert result.ml_direction == "BULLISH"
        assert len(result.news_items) == 1

    def test_llm_response_wrapped_in_markdown_fences_is_parsed(self, fusion_service):
        service, llm = fusion_service
        llm.chat.return_value = (
            "```json\n"
            + json.dumps(
                {
                    "final_direction": "NEUTRAL",
                    "final_confidence": "LOW",
                    "fusion_probability": 0.5,
                    "news_sentiment": "neutral",
                    "synthesis_narrative": "Conflicting signals.",
                }
            )
            + "\n```",
            50,
        )

        result = service.fuse("AAPL", _make_prediction_response(), horizon="1d")

        assert result.fusion_applied is True
        assert result.final_direction == "NEUTRAL"

    def test_llm_json_missing_optional_fields_falls_back_to_ml_defaults(
        self, fusion_service
    ):
        service, llm = fusion_service
        llm.chat.return_value = (json.dumps({}), 10)

        pred = _make_prediction_response(prediction=1, p_bullish=0.68)
        result = service.fuse("AAPL", pred, horizon="1d")

        assert result.fusion_applied is True
        assert result.final_direction == "BULLISH"  # defaults to ml_dir
        assert result.final_confidence == "HIGH"  # defaults to ml_conf


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: LLM failure -> rule-based fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestRuleBasedFallback:
    def test_llm_error_triggers_rule_based_fusion(self, fusion_service):
        service, llm = fusion_service
        llm.chat.side_effect = LLMError("provider unavailable")

        result = service.fuse("AAPL", _make_prediction_response(), horizon="1d")

        assert result.fusion_applied is False
        assert result.final_direction in {"BULLISH", "BEARISH", "NEUTRAL"}
        assert result.synthesis_narrative  # non-empty deterministic narrative

    def test_malformed_json_triggers_rule_based_fusion(self, fusion_service):
        service, llm = fusion_service
        llm.chat.return_value = ("this is not json at all", 5)

        result = service.fuse("AAPL", _make_prediction_response(), horizon="1d")

        assert result.fusion_applied is False

    def test_empty_llm_response_triggers_rule_based_fusion(self, fusion_service):
        service, llm = fusion_service
        llm.chat.return_value = ("", 0)

        result = service.fuse("AAPL", _make_prediction_response(), horizon="1d")

        assert result.fusion_applied is False

    def test_unexpected_exception_from_llm_is_caught(self, fusion_service):
        service, llm = fusion_service
        llm.chat.side_effect = RuntimeError("socket exploded")

        result = service.fuse("AAPL", _make_prediction_response(), horizon="1d")

        assert result.fusion_applied is False

    def test_strong_agreeing_news_keeps_ml_direction_moderate_confidence(
        self, mock_intel_service
    ):
        llm = MagicMock()
        llm.chat.side_effect = LLMError("down")
        mock_intel_service.get_brief.return_value = _make_brief(
            aggregate_sentiment="positive", sentiment_score=0.5
        )
        service = SignalFusionService(
            llm_client=llm, intelligence_service=mock_intel_service
        )

        # ML is bullish, news is strongly positive -> agreement.
        result = service.fuse(
            "AAPL",
            _make_prediction_response(prediction=1, p_bullish=0.68),
            horizon="1d",
        )

        assert result.final_direction == "BULLISH"
        assert result.final_confidence == "MODERATE"

    def test_strong_conflicting_news_forces_neutral(self, mock_intel_service):
        llm = MagicMock()
        llm.chat.side_effect = LLMError("down")
        mock_intel_service.get_brief.return_value = _make_brief(
            aggregate_sentiment="negative", sentiment_score=-0.5
        )
        service = SignalFusionService(
            llm_client=llm, intelligence_service=mock_intel_service
        )

        # ML is bullish, news is strongly negative -> conflict -> NEUTRAL.
        result = service.fuse(
            "AAPL",
            _make_prediction_response(prediction=1, p_bullish=0.68),
            horizon="1d",
        )

        assert result.final_direction == "NEUTRAL"
        assert result.final_confidence == "LOW"
        assert result.fusion_probability == pytest.approx(0.5)

    def test_weak_news_preserves_ml_direction_and_confidence(self, mock_intel_service):
        llm = MagicMock()
        llm.chat.side_effect = LLMError("down")
        mock_intel_service.get_brief.return_value = _make_brief(
            aggregate_sentiment="positive",
            sentiment_score=0.05,  # below 0.30 threshold
        )
        service = SignalFusionService(
            llm_client=llm, intelligence_service=mock_intel_service
        )

        pred = _make_prediction_response(
            prediction=0, p_bullish=0.30, confidence_label="moderate"
        )
        result = service.fuse("AAPL", pred, horizon="1d")

        assert result.final_direction == "BEARISH"
        assert result.final_confidence == "MODERATE"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: News retrieval fails entirely -> ML-only signal
# ─────────────────────────────────────────────────────────────────────────────


class TestMLOnlyFallback:
    def test_retrieval_failure_returns_ml_only_signal(self, mock_intel_service):
        mock_intel_service.get_brief.return_value = _make_brief(
            retrieval_success=False, top_news=[]
        )
        service = SignalFusionService(
            llm_client=MagicMock(), intelligence_service=mock_intel_service
        )

        result = service.fuse("AAPL", _make_prediction_response(), horizon="1d")

        assert result.fusion_applied is False
        assert result.news_items == []
        assert "ML-only" in result.synthesis_narrative

    def test_empty_top_news_returns_ml_only_signal(self, mock_intel_service):
        mock_intel_service.get_brief.return_value = _make_brief(
            retrieval_success=True, top_news=[]
        )
        service = SignalFusionService(
            llm_client=MagicMock(), intelligence_service=mock_intel_service
        )

        result = service.fuse("AAPL", _make_prediction_response(), horizon="1d")

        assert result.fusion_applied is False
        assert result.news_items == []

    def test_ml_only_direction_matches_prediction(self, mock_intel_service):
        mock_intel_service.get_brief.return_value = _make_brief(
            retrieval_success=False, top_news=[]
        )
        service = SignalFusionService(
            llm_client=MagicMock(), intelligence_service=mock_intel_service
        )

        bearish_pred = _make_prediction_response(prediction=0, p_bullish=0.22)
        result = service.fuse("AAPL", bearish_pred, horizon="1d")

        assert result.ml_direction == "BEARISH"
        assert result.final_direction == "BEARISH"


# ─────────────────────────────────────────────────────────────────────────────
# Model resolution across providers
# ─────────────────────────────────────────────────────────────────────────────


class TestModelResolution:
    @pytest.mark.parametrize(
        "base_url,configured_model,expected",
        [
            ("https://api.groq.com/openai/v1", "gpt-4o-mini", "openai/gpt-oss-120b"),
            ("http://ollama.local:11434/v1", "gpt-4o-mini", "llama3"),
            ("https://myazure.openai.azure.com", "gpt-4o-mini", "gpt-4o-mini"),
            ("", "gpt-4o-mini", "gpt-4o-mini"),
            ("https://api.groq.com/openai/v1", "llama-3.3-70b", "llama-3.3-70b"),
        ],
    )
    def test_resolve_model_maps_provider_to_expected_model(
        self, fusion_service, monkeypatch, base_url, configured_model, expected
    ):
        service, _ = fusion_service
        monkeypatch.setattr(
            "app.services.signal_fusion.settings.LLM_BASE_URL", base_url
        )
        monkeypatch.setattr(
            "app.services.signal_fusion.settings.LLM_MODEL", configured_model
        )
        assert service._resolve_model() == expected


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing helper
# ─────────────────────────────────────────────────────────────────────────────


class TestParseHelper:
    def test_parse_extracts_json_from_surrounding_prose(self):
        text = 'Here is my answer:\n{"a": 1, "b": 2}\nHope that helps!'
        result = SignalFusionService._parse(text)
        assert result == {"a": 1, "b": 2}

    def test_parse_raises_value_error_when_no_braces(self):
        with pytest.raises(ValueError):
            SignalFusionService._parse("no json here")

    def test_parse_raises_value_error_on_malformed_json(self):
        with pytest.raises(ValueError):
            SignalFusionService._parse("{not: valid, json}")
