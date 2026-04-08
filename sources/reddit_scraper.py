"""
ARIA Sentiment Engine — Reddit Scraper
Sources: WSB, investing, stocks, crypto, options, algotrading
Method: VADER + financial lexicon on titles via RSS
"""
import feedparser, logging, time
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)
vader = SentimentIntensityAnalyzer()

FINANCIAL_LEXICON = {
    'moon': 3.5, 'mooning': 3.5, 'bullish': 2.8, 'bearish': -2.8,
    'rekt': -3.5, 'dump': -2.5, 'dumping': -2.5, 'pumping': 2.5,
    'crash': -3.5, 'crashing': -3.5, 'yolo': 1.5, 'hodl': 2.0,
    'tendies': 2.5, 'calls': 1.5, 'puts': -1.5, 'squeeze': 2.8,
    'short': -1.2, 'shorting': -2.0, 'capitulation': -3.0,
    'fud': -2.5, 'fomo': 1.5, 'dip': -1.0, 'buying the dip': 2.5,
    'recession': -3.0, 'inflation': -2.0, 'rate hike': -2.5,
    'rate cut': 2.5, 'pivot': 2.0, 'overvalued': -2.0,
    'undervalued': 2.0, 'bubble': -2.5, 'correction': -1.5,
    'apes': 1.8, 'diamond hands': 2.5, 'paper hands': -1.5,
    'to the moon': 3.5, 'bear market': -2.8, 'bull market': 2.8,
    'going up': 2.0, 'going down': -2.0, 'buy the dip': 2.5,
    'sell off': -2.5, 'selloff': -2.5, 'rally': 2.5,
}
vader.lexicon.update(FINANCIAL_LEXICON)

SUBREDDITS = [
    ('wallstreetbets', 2.0),
    ('investing', 1.5),
    ('stocks', 1.2),
    ('StockMarket', 1.2),
    ('options', 1.4),
    ('SecurityAnalysis', 1.3),
    ('CryptoCurrency', 1.8),
    ('Bitcoin', 1.5),
    ('ethereum', 1.4),
    ('algotrading', 1.3),
    ('Economics', 1.2),
    ('finance', 1.2),
]

ASSET_KEYWORDS = {
    'BTC':  ['bitcoin', 'btc', 'crypto', 'satoshi', 'digital currency'],
    'ETH':  ['ethereum', 'eth', 'ether', 'defi'],
    'SPY':  ['spy', 'sp500', 's&p', 'market', 'stocks', 'equities', 'nasdaq', 'dow'],
    'GLD':  ['gold', 'gld', 'precious metals', 'xau'],
    'NVDA': ['nvidia', 'nvda', 'gpu', 'ai chips', 'jensen'],
    'AAPL': ['apple', 'aapl', 'iphone', 'tim cook'],
    'TSLA': ['tesla', 'tsla', 'elon', 'musk', 'ev'],
}

def get_reddit_sentiment():
    results = {'overall': [], 'by_asset': {k: [] for k in ASSET_KEYWORDS}}
    sources_hit = 0
    for subreddit, weight in SUBREDDITS:
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.rss?limit=50"
            headers = {'User-Agent': 'ARIA:SentimentEngine:v1 (by /u/aria_bot)'}
            import urllib.request
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                content = response.read()
            feed = feedparser.parse(content)
            if not feed.entries:
                continue
            sources_hit += 1
            for entry in feed.entries[:40]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')[:300]
                text = f"{title} {summary}"
                score = vader.polarity_scores(text)['compound']
                weighted_score = score * weight * 100
                results['overall'].append(weighted_score)
                text_lower = text.lower()
                for asset, keywords in ASSET_KEYWORDS.items():
                    if any(kw in text_lower for kw in keywords):
                        results['by_asset'][asset].append(weighted_score)
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"Reddit {subreddit} failed: {e}")
    return _aggregate(results, sources_hit)

def _aggregate(results, sources_hit):
    import numpy as np
    def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
    def safe_std(lst): return float(np.std(lst)) if len(lst) > 1 else 10.0
    overall = results['overall']
    by_asset = {}
    for asset, scores in results['by_asset'].items():
        by_asset[asset] = {'score': safe_mean(scores), 'count': len(scores), 'std': safe_std(scores)}
    return {
        'score': safe_mean(overall),
        'confidence': max(0.1, min(1.0, len(overall) / 200)),
        'post_count': len(overall),
        'sources_hit': sources_hit,
        'by_asset': by_asset,
        'source': 'reddit',
        'timestamp': datetime.utcnow().isoformat(),
    }
