"""
ARIA Sentiment Engine — FOMC Hawkish/Dovish Parser
Sources: Fed RSS feeds, FOMC statements
Method: Custom financial NLP lexicon scoring
"""
import feedparser, re, logging
from datetime import datetime

logger = logging.getLogger(__name__)

FOMC_LEXICON = {
    'inflation': -2.5, 'inflationary': -2.8, 'overheat': -3.0,
    'tighten': -2.5, 'tightening': -2.5, 'restrictive': -2.0,
    'sufficiently restrictive': -3.5, 'rate hike': -3.0, 'raise rates': -3.0,
    'higher for longer': -3.5, 'above target': -2.0, 'persistent inflation': -3.0,
    'hawkish': -3.0, 'wage growth': -1.5, 'strong labor': -1.5,
    'pivot': 3.5, 'pause': 2.5, 'cut rates': 3.5, 'rate cut': 3.5,
    'easing': 3.0, 'accommodative': 3.0, 'support growth': 2.5,
    'below target': 2.0, 'slowing inflation': 3.0, 'disinflation': 2.5,
    'cooling': 2.0, 'dovish': 3.0, 'data dependent': 1.0, 'patient': 1.5,
}

FED_RSS_FEEDS = [
    'https://www.federalreserve.gov/feeds/press_all.xml',
    'https://www.federalreserve.gov/feeds/speeches.xml',
]

def get_fomc_sentiment():
    all_scores = []
    key_phrases = []
    latest_statement = None
    for feed_url in FED_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')[:1000]
                text = f"{title} {summary}".lower()
                sentences = re.split(r'[.!?]', text)
                entry_scores = []
                for sentence in sentences:
                    score = _score_fed_sentence(sentence)
                    if abs(score) > 0.5:
                        entry_scores.append(score)
                        for phrase in FOMC_LEXICON:
                            if phrase in sentence:
                                key_phrases.append(phrase)
                if entry_scores:
                    import numpy as np
                    all_scores.append(float(np.mean(entry_scores)))
                    if latest_statement is None:
                        latest_statement = {'title': title, 'date': entry.get('published', ''),
                                           'score': float(np.mean(entry_scores))}
        except Exception as e:
            logger.warning(f"Fed RSS failed: {e}")
    if not all_scores:
        return {'score': 0.0, 'stance': 'NEUTRAL', 'confidence': 0.1,
                'key_phrases': [], 'latest_statement': None,
                'source': 'fomc', 'timestamp': datetime.utcnow().isoformat()}
    import numpy as np
    overall = float(np.mean(all_scores)) * 10
    overall = max(-100, min(100, overall))
    stance = 'DOVISH' if overall > 20 else ('HAWKISH' if overall < -20 else 'NEUTRAL')
    return {'score': overall, 'stance': stance, 'confidence': min(1.0, len(all_scores) / 10),
            'key_phrases': list(set(key_phrases))[:10], 'latest_statement': latest_statement,
            'source': 'fomc', 'timestamp': datetime.utcnow().isoformat()}

def _score_fed_sentence(sentence):
    score = 0.0
    for phrase, weight in FOMC_LEXICON.items():
        if phrase in sentence.lower():
            score += weight
    return max(-10, min(10, score))

def get_yield_curve_sentiment():
    try:
        import requests
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
        r = requests.get(url, timeout=10)
        lines = r.text.strip().split('\n')
        for line in reversed(lines):
            parts = line.split(',')
            if len(parts) == 2 and parts[1] != '.':
                spread = float(parts[1])
                score = max(-100, min(100, spread * 50))
                return {'spread': spread, 'score': score,
                        'regime': 'INVERTED' if spread < 0 else 'NORMAL',
                        'confidence': 0.9, 'source': 'yield_curve',
                        'timestamp': datetime.utcnow().isoformat()}
    except Exception as e:
        logger.warning(f"Yield curve failed: {e}")
    return {'spread': 0.0, 'score': 0.0, 'regime': 'UNKNOWN', 'confidence': 0.0,
            'source': 'yield_curve', 'timestamp': datetime.utcnow().isoformat()}
