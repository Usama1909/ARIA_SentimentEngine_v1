"""
ARIA Sentiment Engine — Financial News Feed Parser
Sources: 15 RSS feeds across equity, crypto, forex, macro
"""
import feedparser, logging
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)
vader = SentimentIntensityAnalyzer()

NEWS_FEEDS = [
    # Tier 1 — Highest weight institutional sources
    ('https://feeds.reuters.com/reuters/businessNews', 'Reuters Business', 2.0),
    ('https://feeds.reuters.com/reuters/technologyNews', 'Reuters Tech', 1.8),
    ('https://feeds.reuters.com/reuters/companyNews', 'Reuters Company', 1.8),
    ('https://www.cnbc.com/id/100003114/device/rss/rss.html', 'CNBC Markets', 2.0),
    ('https://www.cnbc.com/id/10000664/device/rss/rss.html', 'CNBC Business', 1.8),
    # Tier 2 — Strong financial sources
    ('https://feeds.marketwatch.com/marketwatch/topstories/', 'MarketWatch', 1.5),
    ('https://feeds.marketwatch.com/marketwatch/marketpulse/', 'MarketWatch Pulse', 1.8),
    ('https://feeds.finance.yahoo.com/rss/2.0/headline', 'Yahoo Finance', 1.3),
    ('https://www.investing.com/rss/news.rss', 'Investing.com', 1.5),
    ('https://www.investing.com/rss/news_301.rss', 'Investing.com Forex', 1.5),
    ('https://www.benzinga.com/feed', 'Benzinga', 1.4),
    # Tier 3 — Crypto specific
    ('https://www.coindesk.com/arc/outboundfeeds/rss/', 'CoinDesk', 1.8),
    ('https://cointelegraph.com/rss', 'CoinTelegraph', 1.8),
    ('https://decrypt.co/feed', 'Decrypt', 1.5),
    ('https://bitcoinmagazine.com/.rss/full/', 'Bitcoin Magazine', 1.5),
    # Tier 4 — Macro/Global
    ('https://feeds.bbci.co.uk/news/business/rss.xml', 'BBC Business', 1.4),
    ('https://www.ft.com/rss/home/uk', 'Financial Times', 2.0),
    ('https://www.forexfactory.com/ff_calendar_thisweek.xml', 'ForexFactory', 1.6),
    ('https://seekingalpha.com/feed.xml', 'Seeking Alpha', 1.4),
    ('https://www.zerohedge.com/fullrss2.xml', 'ZeroHedge', 1.2),
]

ASSET_KEYWORDS = {
    'BTC':  ['bitcoin', 'btc', 'cryptocurrency', 'crypto', 'satoshi', 'digital asset'],
    'ETH':  ['ethereum', 'eth', 'defi', 'smart contract', 'vitalik'],
    'SPY':  ['s&p 500', 'sp500', 'stock market', 'equities', 'wall street', 'dow', 'nasdaq', 'russell'],
    'GLD':  ['gold', 'precious metal', 'xau', 'commodities', 'bullion'],
    'NVDA': ['nvidia', 'nvda', 'semiconductor', 'ai chip', 'gpu', 'jensen huang'],
    'AAPL': ['apple', 'aapl', 'iphone', 'tim cook', 'app store'],
    'TSLA': ['tesla', 'tsla', 'elon musk', 'electric vehicle', 'ev', 'cybertruck'],
    'MACRO': ['federal reserve', 'fed', 'fomc', 'interest rate', 'inflation', 'cpi',
              'gdp', 'recession', 'employment', 'jobs report', 'yield curve', 'treasury'],
}

CRISIS_KEYWORDS = [
    'crash', 'collapse', 'crisis', 'panic', 'plunge', 'meltdown',
    'bankruptcy', 'default', 'contagion', 'systemic', 'black swan',
    'bank run', 'liquidity crisis', 'margin call', 'forced selling',
]

EUPHORIA_KEYWORDS = [
    'record high', 'all-time high', 'ath', 'rally', 'surge', 'soar',
    'boom', 'breakthrough', 'milestone', 'historic', 'new high',
    'bull market', 'golden cross', 'breakout',
]

def get_news_sentiment():
    all_scores = []
    asset_scores = {k: [] for k in ASSET_KEYWORDS}
    crisis_count = 0
    euphoria_count = 0
    headlines = []
    sources_hit = 0

    for feed_url, source_name, weight in NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            if not feed.entries:
                continue
            sources_hit += 1
            for entry in feed.entries[:25]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')[:300]
                text = f"{title} {summary}"
                text_lower = text.lower()
                score = vader.polarity_scores(text)['compound'] * weight * 100
                all_scores.append(score)
                for asset, keywords in ASSET_KEYWORDS.items():
                    if any(kw in text_lower for kw in keywords):
                        asset_scores[asset].append(score)
                if any(kw in text_lower for kw in CRISIS_KEYWORDS):
                    crisis_count += 1
                if any(kw in text_lower for kw in EUPHORIA_KEYWORDS):
                    euphoria_count += 1
                if abs(score) > 30:
                    headlines.append({'title': title[:100], 'score': score, 'source': source_name})
        except Exception as e:
            logger.warning(f"News feed {source_name} failed: {e}")

    import numpy as np
    def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
    overall = safe_mean(all_scores)
    if crisis_count > 5:
        overall = min(overall, -30.0)
    by_asset = {asset: {'score': safe_mean(scores), 'count': len(scores)}
                for asset, scores in asset_scores.items()}
    return {
        'score': overall,
        'confidence': min(1.0, len(all_scores) / 200),
        'article_count': len(all_scores),
        'sources_hit': sources_hit,
        'crisis_signals': crisis_count,
        'euphoria_signals': euphoria_count,
        'top_headlines': sorted(headlines, key=lambda x: abs(x['score']), reverse=True)[:10],
        'by_asset': by_asset,
        'source': 'news',
        'timestamp': datetime.utcnow().isoformat(),
    }
