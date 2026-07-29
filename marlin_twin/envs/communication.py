# ============================================================================
# FILE: marlin_twin/envs/communication.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import (
    MaritimeMessage, MessagePriority, MaritimeCommunicationChannel, CommunicationStatus
)

class CommunicationChannelManager:
    """
    Manages bandwidth-constrained priority message transmission across vessels.
    Handles message queues (CRITICAL > HIGH > MEDIUM > LOW), weather attenuation, and jamming.
    """

    def __init__(self, bandwidth_bps: float = 9600.0, base_latency: float = 0.5):
        self.channel = MaritimeCommunicationChannel(
            channel_id="maritime_vhf_channel",
            bandwidth_bps=bandwidth_bps,
            base_latency=base_latency,
            packet_loss_rate=0.05
        )

    def process_step(
        self,
        outgoing_messages: list[MaritimeMessage],
        weather_degradation: float = 0.0,
        jamming_active: bool = False
    ) -> list[MaritimeMessage]:
        """Process one time step of message transmissions."""
        self.channel.weather_degradation = weather_degradation
        self.channel.jamming_active = jamming_active

        # Sort incoming messages by priority (CRITICAL = 0 comes first)
        all_messages = self.channel.message_queue + outgoing_messages
        all_messages.sort(key=lambda m: m.priority.value)

        delivered = []
        remaining = []

        avail_bits = self.channel.available_bandwidth(time_window=1.0)

        for msg in all_messages:
            if msg.size_bits <= avail_bits:
                # Check packet loss
                loss_prob = self.channel.packet_loss_rate + weather_degradation * 0.2
                if np.random.rand() > loss_prob:
                    msg_delivered = MaritimeMessage(
                        sender_id=msg.sender_id,
                        receiver_id=msg.receiver_id,
                        content=msg.content,
                        priority=msg.priority,
                        timestamp=msg.timestamp,
                        size_bits=msg.size_bits,
                        latency=self.channel.base_latency,
                        delivered=True,
                        delivery_confirmed=True
                    )
                    delivered.append(msg_delivered)
                    avail_bits -= msg.size_bits
                else:
                    msg_failed = MaritimeMessage(
                        sender_id=msg.sender_id, receiver_id=msg.receiver_id,
                        content=msg.content, priority=msg.priority, timestamp=msg.timestamp,
                        size_bits=msg.size_bits, delivered=False
                    )
                    remaining.append(msg_failed)
            else:
                remaining.append(msg)

        self.channel.message_queue = remaining
        return delivered
