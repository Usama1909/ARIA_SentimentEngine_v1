"""
ARIA Sentiment Engine v1 — Problem B
Multi-source Bayesian sentiment fusion for autonomous trading
"""
import time, logging, json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from sources.reddit_scraper import get_reddit_sentiment
from sources.fomc_parser import get_fomc_sentiment, get_yield_curve_sentiment
from sources.news_feed import get_news_sentiment
from sources.crypto_fear_greed import get_crypto_fear_greed, get_vix_sentiment
from fusion.bayesian_fusion import bayesian_fuse, detect_regime
from fusion.velocity_tracker import SentimentVelocityTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('ARIA.SentimentEngine')

_velocity_tracker = SentimentVelocityTracker(window_hours=4)
_cache = {}
_cache_ttl = {'reddit':300,'fomc':1800,'news':180,'fear_greed':600,'vix':60,'yield_curve':3600}

def _is_cached(key):
    if key not in _cache: return False
    return (datetime.utcnow() - _cache[key]['time']).total_seconds() < _cache_ttl.get(key, 300)

def _get_cached(key): return _cache[key]['data'] if key in _cache else None
def _set_cache(key, data): _cache[key] = {'data': data, 'time': datetime.utcnow()}

def fetch_all_sources(timeout=25):
    results = {}
    tasks = {
        'reddit': get_reddit_sentiment,
        'fomc': get_fomc_sentiment,
        'news': get_news_sentiment,
        'fear_greed': get_crypto_fear_greed,
        'vix': get_vix_sentiment,
        'yield_curve': get_yield_curve_sentiment,
    }
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for name, func in tasks.items():
            if _is_cached(name):
                results[name] = _get_cached(name)
            else:
                futures[executor.submit(func)] = name
        for future in as_completed(futures, timeout=timeout):
            name = futures[future]
            try:
                data = future.result(timeout=5)
                results[name] = data
                _set_cache(name, data)
                logger.info(f'{name}: score={data.get("score",0):.1f} conf={data.get("confidence",0):.2f}')
            except Exception as e:
                logger.warning(f'{name} failed: {e}')
                results[name] = _get_cached(name)
    return results

def get_sentiment_signal():
    import numpy as np
    start = time.time()
    sources = fetch_all_sources(timeout=25)
    regime = detect_regime(
        fomc_data=sources.get('fomc'),
        vix_data=sources.get('vix'),
        fear_greed_data=sources.get('fear_greed')
    )
    fusion_input = {k: v for k, v in sources.items() if v is not None and 'score' in v and 'confidence' in v}
    fused = bayesian_fuse(fusion_input, regime=regime)
    _velocity_tracker.update(fused['score'])
    velocity = _velocity_tracker.get_full_signal()
    by_asset = {}
    for asset in ['BTC','ETH','SPY','GLD','NVDA','AAPL','TSLA']:
        asset_scores = []
        for source_data in sources.values():
            if source_data and 'by_asset' in source_data:
                ad = source_data['by_asset'].get(asset, {})
                if isinstance(ad, dict) and 'score' in ad:
                    asset_scores.append(ad['score'])
                elif isinstance(ad, (int, float)):
                    asset_scores.append(float(ad))
        by_asset[asset] = float(np.mean(asset_scores)) if asset_scores else fused['score']
    score = fused['score']
    stance = 'BULLISH' if score > 20 else ('BEARISH' if score < -20 else 'NEUTRAL')
    elapsed = time.time() - start
    signal = {
        'score': score,
        'confidence': fused['confidence'],
        'uncertainty': fused.get('uncertainty', 50.0),
        'velocity_1h': velocity['velocity_1h'],
        'velocity_4h': velocity['velocity_4h'],
        'acceleration': velocity['acceleration'],
        'velocity_signal': velocity['signal'],
        'reversal_detected': velocity['reversal_detected'],
        'regime': regime,
        'stance': stance,
        'by_asset': by_asset,
        'fomc_stance': (sources.get('fomc') or {}).get('stance','NEUTRAL'),
        'fomc_key_phrases': (sources.get('fomc') or {}).get('key_phrases',[]),
        'fear_greed_raw': (sources.get('fear_greed') or {}).get('raw_score',50),
        'fear_greed_regime': (sources.get('fear_greed') or {}).get('regime','NEUTRAL'),
        'vix': (sources.get('vix') or {}).get('vix',20.0),
        'yield_curve_spread': (sources.get('yield_curve') or {}).get('spread',0.0),
        'source_count': fused['source_count'],
        'fetch_time_seconds': round(elapsed, 2),
        'timestamp': datetime.utcnow().isoformat(),
    }
    logger.info(f"Sentiment: {score:+.1f} ({stance}) | Velocity: {velocity['velocity_1h']:+.1f}/hr | Regime: {regime} | FOMC: {signal['fomc_stance']} | F&G: {signal['fear_greed_raw']} | {elapsed:.1f}s")
    return signal

if __name__ == '__main__':
    print("ARIA Sentiment Engine v1 — Test Run")
    print("=" * 60)
    signal = get_sentiment_signal()
    print(json.dumps(signal, indent=2))
