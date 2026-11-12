"""Unit tests for lexical/DLP algorithms, nickname resolution, and zero future leakage."""

from __future__ import annotations

import pandas as pd
from numpy.random import default_rng

from outguard.config import PipelineConfig
from outguard.data.catalog import make_luhn_card
from outguard.features.extractor import OutGuardFeatureExtractor
from outguard.features.lexical import (
    NICKNAME_EQUIV,
    SALUTATION_RE,
    WARM_INTRO_RE,
    damerau_levenshtein,
    jaro_winkler_similarity,
)
from outguard.features.semantic_dlp import verify_luhn


def test_jaro_winkler_and_damerau_levenshtein() -> None:
    assert jaro_winkler_similarity("michael", "michael") == 1.0
    assert jaro_winkler_similarity("michael", "micheal") > 0.93
    assert jaro_winkler_similarity("alice", "roberto") < 0.55

    # Transposition ("nrth" <-> "rnth") is 1 edit in Damerau-Levenshtein
    assert damerau_levenshtein("northstarbank.com", "nrothstarbank.com") == 1
    assert damerau_levenshtein("summitretail.com", "summitretail.com") == 0
    assert damerau_levenshtein("summitretail.com", "gmail.com") >= 6


def test_nickname_canonicalization_and_warm_intro() -> None:
    assert "margaret" in NICKNAME_EQUIV["peggy"]
    assert "michael" in NICKNAME_EQUIV["mike"]
    assert "alexander" in NICKNAME_EQUIV["sasha"]
    assert "david" not in NICKNAME_EQUIV["michael"]

    m = SALUTATION_RE.search("Hi Margaret,\n\nAttached is the deck.")
    assert m is not None and m.group(1).lower() == "margaret"
    assert WARM_INTRO_RE.search("Hi Priya,\n\nLooping in Marcus from Northstar to help.")
    assert not WARM_INTRO_RE.search("Hi Priya,\n\nInternal only payroll sheet attached.")


def test_luhn_payment_card_verifier() -> None:
    rng = default_rng(123)
    valid_card = make_luhn_card(rng)
    assert verify_luhn(valid_card)
    corrupted = valid_card[:-1] + str((int(valid_card[-1]) + 1) % 10)
    assert not verify_luhn(corrupted)


def test_strict_point_in_time_zero_future_leakage() -> None:
    """Verify that appending future interactions NEVER alters historical features."""
    cfg = PipelineConfig()
    extractor = OutGuardFeatureExtractor(cfg)

    base_events = [
        {
            "event_id": "e1",
            "timestamp_epoch": 1_760_000_000.0,
            "split": "train",
            "tenant_id": "t1",
            "tenant_domain": "acme.io",
            "sender_email": "alice@acme.io",
            "sender_name": "Alice Smith",
            "sender_department": "Finance",
            "sender_is_vip": False,
            "recipients": [
                {
                    "email": "bob@partner.com",
                    "display_name": "Bob Lee",
                    "domain": "partner.com",
                    "field": "to",
                }
            ],
            "is_reply": False,
            "thread_prior_messages": 0,
            "subject": "Q1 Kickoff",
            "body": "Hi Bob,\n\nNice meeting you.",
            "attachment_names": [],
            "true_label": 0,
            "observed_train_label": 0,
            "label_confidence": 1.0,
            "label_source": "implicit",
            "misdirection_type": "BENIGN",
            "eval_slice": "standard_business",
        },
    ]
    future_events = [
        {
            **base_events[0],
            "event_id": f"e_fut_{i}",
            "timestamp_epoch": 1_760_000_000.0 + 3600.0 * (i + 1),
        }
        for i in range(10)
    ]

    df_single, _ = extractor.build_offline_feature_table(pd.DataFrame(base_events))
    df_with_future, _ = extractor.build_offline_feature_table(
        pd.DataFrame(base_events + future_events)
    )

    row_a = df_single.iloc[0]
    row_b = df_with_future.iloc[0]
    assert row_a["graph_min_recipient_prior_count"] == 0.0
    assert row_a["graph_any_first_time_external"] == 1.0
    for col in df_single.columns[13:]:
        assert abs(float(row_a[col]) - float(row_b[col])) < 1e-12, f"Future leak in {col}"

    assert df_with_future.iloc[1]["graph_any_first_time_external"] == 0.0
    assert df_with_future.iloc[1]["graph_min_recipient_prior_count"] > 0.69
