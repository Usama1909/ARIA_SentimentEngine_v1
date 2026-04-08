"""
ARIA Sentiment Engine — Live Integration Connector
Drop-in replacement for static sentiment in agent_loop_v4.py
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentiment_engine import get_sentiment_signal

logger = logging.getLogger('ARIA.Connector')

_last_signal = None
_last_fetch_time = None
_REFRESH_INTERVAL = 300  # 5 minutes

def get_live_sentiment(force_refresh=False):
    global _last_signal, _last_fetch_time
    from datetime import datetime
    now = datetime.utcnow()
    should_refresh = (
        force_refresh or _last_signal is None or _last_fetch_time is None or
        (now - _last_fetch_time).total_seconds() > _REFRESH_INTERVAL
    )
    if should_refresh:
        try:
            _last_signal = get_sentiment_signal()
            _last_fetch_time = now
            logger.info(f"Refreshed: {_last_signal['score']:+.1f} ({_last_signal['stance']})")
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            if _last_signal is None:
                _last_signal = _default_signal()
    return _last_signal

def update_bus_with_sentiment(bus, s):
    bus['sentiment'] = s['score']
    bus['sentiment_confidence'] = s['confidence']
    bus['sentiment_velocity'] = s['velocity_1h']
    bus['sentiment_velocity_4h'] = s['velocity_4h']
    bus['sentiment_acceleration'] = s['acceleration']
    bus['sentiment_signal'] = s['velocity_signal']
    bus['sentiment_stance'] = s['stance']
    bus['sentiment_reversal'] = s['reversal_detected']
    bus['fomc_stance'] = s['fomc_stance']
    bus['fear_greed'] = s['fear_greed_raw']
    bus['fear_greed_regime'] = s['fear_greed_regime']
    bus['vix_live'] = s['vix']
    bus['yield_spread'] = s['yield_curve_spread']
    bus['sentiment_by_asset'] = s['by_asset']
    bus['sentiment_regime'] = s['regime']
    return bus

def _default_signal():
    from datetime import datetime
    return {
        'score': 0.0, 'confidence': 0.0, 'velocity_1h': 0.0, 'velocity_4h': 0.0,
        'acceleration': 0.0, 'velocity_signal': 'NEUTRAL', 'reversal_detected': False,
        'regime': 'NORMAL', 'stance': 'NEUTRAL', 'by_asset': {},
        'fomc_stance': 'NEUTRAL', 'fomc_key_phrases': [], 'fear_greed_raw': 50,
        'fear_greed_regime': 'NEUTRAL', 'vix': 20.0, 'yield_curve_spread': 0.0,
        'source_count': 0, 'fetch_time_seconds': 0.0,
        'timestamp': datetime.utcnow().isoformat(),
    }
