"""
ARIA Sentiment Engine — Velocity & Momentum Tracker
Tracks rate of change of sentiment — key insight: velocity predicts price moves better than level alone.
"""
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SentimentVelocityTracker:
    def __init__(self, window_hours=4, max_history=500):
        self.history = deque(maxlen=max_history)
        self.window_hours = window_hours

    def update(self, score, timestamp=None):
        ts = timestamp or datetime.utcnow()
        self.history.append({'score': score, 'timestamp': ts})

    def get_velocity(self, window_minutes=60):
        if len(self.history) < 2:
            return 0.0
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=window_minutes)
        recent = [h for h in self.history if h['timestamp'] >= cutoff]
        if len(recent) < 2:
            return 0.0
        times = [(h['timestamp'] - recent[0]['timestamp']).total_seconds() / 3600 for h in recent]
        scores = [h['score'] for h in recent]
        if len(set(times)) < 2:
            return 0.0
        n = len(times)
        sum_x = sum(times); sum_y = sum(scores)
        sum_xy = sum(t * s for t, s in zip(times, scores))
        sum_x2 = sum(t * t for t in times)
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.0
        return float((n * sum_xy - sum_x * sum_y) / denom)

    def get_acceleration(self):
        return self.get_velocity(30) - self.get_velocity(120)

    def detect_reversal(self, threshold=15.0):
        if len(self.history) < 10:
            return False
        recent_scores = [h['score'] for h in list(self.history)[-20:]]
        mid = len(recent_scores) // 2
        first_half_mean = np.mean(recent_scores[:mid])
        second_half_mean = np.mean(recent_scores[mid:])
        reversal_magnitude = abs(second_half_mean - first_half_mean)
        direction_change = (first_half_mean > 0) != (second_half_mean > 0)
        return direction_change and reversal_magnitude > threshold

    def get_full_signal(self):
        velocity_1h = self.get_velocity(60)
        velocity_4h = self.get_velocity(240)
        acceleration = self.get_acceleration()
        reversal = self.detect_reversal()
        current = self.history[-1]['score'] if self.history else 0.0
        return {
            'current_score': current,
            'velocity_1h': velocity_1h,
            'velocity_4h': velocity_4h,
            'acceleration': acceleration,
            'reversal_detected': reversal,
            'signal': _classify(velocity_1h, acceleration, reversal),
            'timestamp': datetime.utcnow().isoformat(),
        }

def _classify(velocity, acceleration, reversal):
    if reversal: return 'REVERSAL'
    if velocity > 20 and acceleration > 5: return 'STRONG_BULLISH_MOMENTUM'
    if velocity > 10: return 'BULLISH_MOMENTUM'
    if velocity < -20 and acceleration < -5: return 'STRONG_BEARISH_MOMENTUM'
    if velocity < -10: return 'BEARISH_MOMENTUM'
    return 'NEUTRAL'
