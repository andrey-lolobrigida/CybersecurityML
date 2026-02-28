"""
Pydantic request and response models for the inference API.
"""

from typing import Literal

from pydantic import BaseModel


class SessionInput(BaseModel):
    session_id: str | None = None
    network_packet_size: int
    protocol_type: Literal["ICMP", "TCP", "UDP"]
    login_attempts: int
    session_duration: float
    encryption_used: Literal["AES", "DES"] | None = None  # None → filled as "Unencrypted"
    ip_reputation_score: float
    failed_logins: int
    browser_type: Literal["Chrome", "Edge", "Firefox", "Safari", "Unknown"]
    unusual_time_access: Literal[0, 1]


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    label: Literal["Normal session", "Possible attack session"]
    probability: float