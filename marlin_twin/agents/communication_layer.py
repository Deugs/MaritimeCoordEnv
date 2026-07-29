# ============================================================================
# FILE: marlin_twin/agents/communication_layer.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import VesselState, Route, MaritimeMessage, MessagePriority

class CommunicationLayer:
    """Encodes and decodes inter-vessel coordination messages."""

    @staticmethod
    def encode_message(sender_id: int, receiver_id: int, state: VesselState, priority: MessagePriority = MessagePriority.MEDIUM) -> MaritimeMessage:
        content = np.array([state.x, state.y, state.heading, state.speed], dtype=np.float32)
        return MaritimeMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            priority=priority,
            timestamp=0.0,
            size_bits=256
        )

    @staticmethod
    def decode_message(message: MaritimeMessage) -> tuple[float, float, float, float]:
        x, y, heading, speed = message.content
        return float(x), float(y), float(heading), float(speed)
