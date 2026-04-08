"""
ARIA Sentiment Engine — Full Test Suite
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fear_greed():
    print("Testing Fear & Greed...")
    from sources.crypto_fear_greed import get_crypto_fear_greed
    d = get_crypto_fear_greed()
    assert 'score' in d and -100 <= d['score'] <= 100
    print(f"  Score: {d['score']:+.1f} | {d['classification']} | {d['regime']}")

def test_vix():
    print("Testing VIX...")
    from sources.crypto_fear_greed import get_vix_sentiment
    d = get_vix_sentiment()
    assert 'score' in d
    print(f"  VIX: {d['vix']:.1f} | Score: {d['score']:+.1f} | {d['regime']}")

def test_yield_curve():
    print("Testing Yield Curve...")
    from sources.fomc_parser import get_yield_curve_sentiment
    d = get_yield_curve_sentiment()
    assert 'score' in d
    print(f"  Spread: {d['spread']:.3f}% | Score: {d['score']:+.1f} | {d['regime']}")

def test_fomc():
    print("Testing FOMC...")
    from sources.fomc_parser import get_fomc_sentiment
    d = get_fomc_sentiment()
    assert 'score' in d and 'stance' in d
    print(f"  Score: {d['score']:+.1f} | Stance: {d['stance']} | Phrases: {d['key_phrases'][:3]}")

def test_news():
    print("Testing News (20 sources)...")
    from sources.news_feed import get_news_sentiment
    d = get_news_sentiment()
    assert 'score' in d
    print(f"  Score: {d['score']:+.1f} | Articles: {d['article_count']} | Sources hit: {d['sources_hit']} | Crisis: {d['crisis_signals']}")
    if d['top_headlines']:
        print(f"  Top: {d['top_headlines'][0]['title'][:70]}")

def test_reddit():
    print("Testing Reddit (12 subreddits)...")
    from sources.reddit_scraper import get_reddit_sentiment
    d = get_reddit_sentiment()
    assert 'score' in d
    print(f"  Score: {d['score']:+.1f} | Posts: {d['post_count']} | Sources: {d['sources_hit']}")

def test_fusion():
    print("Testing Bayesian Fusion...")
    from fusion.bayesian_fusion import bayesian_fuse
    sources = {
        'fear_greed': {'score': 30.0, 'confidence': 0.9},
        'fomc':       {'score': -20.0, 'confidence': 0.8},
        'news':       {'score': 10.0, 'confidence': 0.7},
        'reddit':     {'score': 50.0, 'confidence': 0.5},
    }
    r = bayesian_fuse(sources, regime='NORMAL')
    assert 'score' in r and -100 <= r['score'] <= 100
    print(f"  Fused: {r['score']:+.1f} | Confidence: {r['confidence']:.2f} | Uncertainty: {r['uncertainty']:.1f}")

def test_velocity():
    print("Testing Velocity Tracker...")
    from fusion.velocity_tracker import SentimentVelocityTracker
    from datetime import datetime, timedelta
    t = SentimentVelocityTracker()
    base = datetime.utcnow() - timedelta(hours=3)
    for i in range(20):
        t.update(float(i * 3 - 30), base + timedelta(minutes=i * 9))
    s = t.get_full_signal()
    assert 'velocity_1h' in s
    print(f"  Velocity 1h: {s['velocity_1h']:+.1f}/hr | Signal: {s['signal']}")

def test_full_engine():
    print("Testing Full Engine...")
    from sentiment_engine import get_sentiment_signal
    s = get_sentiment_signal()
    assert 'score' in s and 'velocity_1h' in s and 'by_asset' in s
    print(f"  Score: {s['score']:+.1f} | Stance: {s['stance']} | Regime: {s['regime']}")
    print(f"  FOMC: {s['fomc_stance']} | F&G: {s['fear_greed_raw']} | VIX: {s['vix']:.1f}")
    print(f"  Velocity: {s['velocity_1h']:+.1f}/hr | Signal: {s['velocity_signal']}")
    print(f"  Sources: {s['source_count']} | Fetch: {s['fetch_time_seconds']}s")

if __name__ == '__main__':
    print("=" * 60)
    print("ARIA Sentiment Engine v1 — Full Test Suite")
    print("=" * 60)
    tests = [test_fear_greed, test_vix, test_yield_curve, test_fomc,
             test_news, test_reddit, test_fusion, test_velocity, test_full_engine]
    passed = failed = 0
    for test in tests:
        try:
            test(); passed += 1; print("  PASS\n")
        except Exception as e:
            print(f"  FAIL: {e}\n"); failed += 1
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed")
    print("ALL TESTS PASSED — Engine ready for ARIA" if failed == 0 else f"{failed} tests failed")
