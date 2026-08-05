"""Encoding/decoding of inter-vessel coordination messages."""

import struct

import numpy as np
from marlin_twin.data_classes import VesselState, MaritimeMessage, MessagePriority


class CommunicationLayer:
    """Encodes and decodes inter-vessel coordination messages."""

    @staticmethod
    def encode_message(
        sender_id: int,
        receiver_id: int,
        state: VesselState,
        priority: MessagePriority = MessagePriority.MEDIUM,
    ) -> MaritimeMessage:
        content = np.array([state.x, state.y, state.heading, state.speed], dtype=np.float32)
        return MaritimeMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            priority=priority,
            timestamp=0.0,
            size_bits=256,
        )

    @staticmethod
    def decode_message(message: MaritimeMessage) -> tuple[float, float, float, float]:
        x, y, heading, speed = message.content
        return float(x), float(y), float(heading), float(speed)

    @staticmethod
    def encode_binary(sender_id: int, receiver_id: int, state: VesselState) -> bytes:
        """Packs state telemetry into a compact 16-byte (128-bit) binary payload."""
        # Pack 4 floats (4 * 4 = 16 bytes = 128 bits)
        return struct.pack("!ffff", state.x, state.y, state.heading, state.speed)

    @staticmethod
    def decode_binary(payload: bytes) -> tuple[float, float, float, float]:
        """Unpacks 16-byte (128-bit) binary payload into state telemetry."""
        if len(payload) != 16:
            raise ValueError(f"Invalid payload length {len(payload)} bytes (expected 16 bytes)")
        x, y, heading, speed = struct.unpack("!ffff", payload)
        return float(x), float(y), float(heading), float(speed)
