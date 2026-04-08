"""
ARIA Sentiment Engine — Crypto Fear & Greed Index
Source: alternative.me API (free, no key needed)
"""
import requests, logging
from datetime import datetime

logger = logging.getLogger(__name__)
FNG_URL = "https://api.alternative.me/fng/?limit=7&format=json"

def get_crypto_fear_greed():
    try:
        r = requests.get(FNG_URL, timeout=10)
        data = r.json()['data']
        current = int(data[0]['value'])
        classification = data[0]['value_classification']
        values = [int(d['value']) for d in data]
        trend = values[0] - values[-1]
        normalised = (current - 50) * 2
        if current >= 75: regime = 'EXTREME_GREED'
        elif current >= 55: regime = 'GREED'
        elif current >= 45: regime = 'NEUTRAL'
        elif current >= 25: regime = 'FEAR'
        else: regime = 'EXTREME_FEAR'
        return {'raw_score': current, 'score': float(normalised), 'classification': classification,
                'regime': regime, 'trend_7d': float(trend),
                'trend_direction': 'IMPROVING' if trend > 5 else ('WORSENING' if trend < -5 else 'STABLE'),
                'confidence': 0.9, 'source': 'fear_greed', 'timestamp': datetime.utcnow().isoformat()}
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
        return {'raw_score': 50, 'score': 0.0, 'classification': 'Neutral', 'regime': 'NEUTRAL',
                'trend_7d': 0.0, 'trend_direction': 'STABLE', 'confidence': 0.0,
                'source': 'fear_greed', 'timestamp': datetime.utcnow().isoformat()}

def get_vix_sentiment():
    try:
        import feedparser
        feed = feedparser.parse("https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EVIX")
        if feed.entries:
            import re
            title = feed.entries[0].get('title', '')
            match = re.search(r'[\d.]+', title)
            if match:
                vix = float(match.group())
                score = max(-100, min(100, (20 - vix) * 4))
                return {'vix': vix, 'score': score,
                        'regime': 'CALM' if vix < 15 else ('ELEVATED' if vix < 25 else ('FEARFUL' if vix < 35 else 'PANIC')),
                        'confidence': 0.8, 'source': 'vix', 'timestamp': datetime.utcnow().isoformat()}
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")
    return {'vix': 20.0, 'score': 0.0, 'regime': 'UNKNOWN', 'confidence': 0.0,
            'source': 'vix', 'timestamp': datetime.utcnow().isoformat()}
