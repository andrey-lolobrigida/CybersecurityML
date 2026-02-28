import pandas as pd
import pytest

from src.preprocessing import preprocess


@pytest.fixture
def base_df():
    return pd.DataFrame([{
        "session_id": "SID_TEST",
        "network_packet_size": 599,
        "protocol_type": "TCP",
        "login_attempts": 4,
        "session_duration": 492.98,
        "encryption_used": "DES",
        "ip_reputation_score": 0.607,
        "failed_logins": 1,
        "browser_type": "Edge",
        "unusual_time_access": 0,
    }])


def test_output_columns_match_model_features(base_df, cfg):
    result = preprocess(base_df, cfg)
    assert list(result.columns) == cfg.model_features


def test_session_id_dropped(base_df, cfg):
    result = preprocess(base_df, cfg)
    assert "session_id" not in result.columns


def test_attack_detected_ignored_if_present(base_df, cfg):
    base_df["attack_detected"] = 1
    result = preprocess(base_df, cfg)
    assert "attack_detected" not in result.columns
    assert list(result.columns) == cfg.model_features


def test_encryption_used_null_fills_unencrypted(base_df, cfg):
    base_df["encryption_used"] = None
    result = preprocess(base_df, cfg)
    assert result["encryption_used_Unencrypted"].iloc[0] == 1
    assert result["encryption_used_AES"].iloc[0] == 0
    assert result["encryption_used_DES"].iloc[0] == 0


def test_ohe_tcp_protocol(base_df, cfg):
    result = preprocess(base_df, cfg)
    assert result["protocol_type_TCP"].iloc[0] == 1
    assert result["protocol_type_UDP"].iloc[0] == 0
    assert result["protocol_type_ICMP"].iloc[0] == 0


def test_ohe_unknown_category_produces_all_zeros(base_df, cfg):
    # A category value not in the config produces all-zero OHE columns for that group
    base_df["protocol_type"] = "UNKNOWN_PROTOCOL"
    result = preprocess(base_df, cfg)
    assert result["protocol_type_TCP"].iloc[0] == 0
    assert result["protocol_type_UDP"].iloc[0] == 0
    assert result["protocol_type_ICMP"].iloc[0] == 0


def test_multiple_rows_shape(base_df, cfg):
    df = pd.concat([base_df, base_df], ignore_index=True)
    result = preprocess(df, cfg)
    assert result.shape == (2, len(cfg.model_features))


def test_passthrough_values_preserved(base_df, cfg):
    result = preprocess(base_df, cfg)
    assert result["network_packet_size"].iloc[0] == 599
    assert result["ip_reputation_score"].iloc[0] == pytest.approx(0.607)
    assert result["failed_logins"].iloc[0] == 1