"""
ARIA Sentiment Engine — Bayesian Multi-Source Fusion
Method: Confidence-weighted Bayesian combination with regime conditioning
"""
import numpy as np
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)

SOURCE_WEIGHTS = {
    'fear_greed':   0.30,
    'fomc':         0.25,
    'news':         0.25,
    'reddit':       0.15,
    'vix':          0.05,
}

REGIME_WEIGHT_MULTIPLIERS = {
    'FOMC_DAY': {'fomc': 2.0, 'fear_greed': 0.5, 'news': 1.5, 'reddit': 0.3, 'vix': 1.0},
    'CRISIS':   {'fear_greed': 1.5, 'vix': 2.0, 'fomc': 1.0, 'news': 1.2, 'reddit': 0.2},
    'NORMAL':   {k: 1.0 for k in SOURCE_WEIGHTS},
}

def bayesian_fuse(sources, regime='NORMAL'):
    if not sources:
        return {'score': 0.0, 'confidence': 0.0, 'uncertainty': 50.0,
                'regime': regime, 'source_count': 0, 'total_weight': 0.0,
                'timestamp': datetime.utcnow().isoformat()}
    multipliers = REGIME_WEIGHT_MULTIPLIERS.get(regime, REGIME_WEIGHT_MULTIPLIERS['NORMAL'])
    weighted_scores = []
    total_weight = 0.0
    for source_name, data in sources.items():
        if data is None:
            continue
        score = data.get('score', 0.0)
        confidence = data.get('confidence', 0.5)
        base_w = SOURCE_WEIGHTS.get(source_name, 0.1)
        regime_mult = multipliers.get(source_name, 1.0)
        weight = base_w * confidence * regime_mult
        weighted_scores.append((score, weight))
        total_weight += weight
    if total_weight == 0:
        return {'score': 0.0, 'confidence': 0.0, 'uncertainty': 50.0,
                'regime': regime, 'source_count': 0, 'total_weight': 0.0,
                'timestamp': datetime.utcnow().isoformat()}
    fused_score = sum(s * w for s, w in weighted_scores) / total_weight
    fused_score = max(-100, min(100, fused_score))
    if len(weighted_scores) > 1:
        scores_array = np.array([s for s, _ in weighted_scores])
        weights_array = np.array([w for _, w in weighted_scores])
        weights_norm = weights_array / weights_array.sum()
        variance = np.average((scores_array - fused_score)**2, weights=weights_norm)
        uncertainty = float(np.sqrt(variance))
    else:
        uncertainty = 50.0
    agreement = max(0.0, 1.0 - uncertainty / 100.0)
    overall_confidence = min(1.0, (total_weight / sum(SOURCE_WEIGHTS.values())) * agreement)
    return {
        'score': float(fused_score),
        'confidence': float(overall_confidence),
        'uncertainty': float(uncertainty),
        'regime': regime,
        'source_count': len(weighted_scores),
        'total_weight': float(total_weight),
        'timestamp': datetime.utcnow().isoformat(),
    }

def detect_regime(fomc_data=None, vix_data=None, fear_greed_data=None):
    today = date.today()
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    is_fomc_week = today.month in fomc_months and 15 <= today.day <= 22
    if is_fomc_week:
        return 'FOMC_DAY'
    vix_score = (vix_data or {}).get('vix', 20)
    fg_score = (fear_greed_data or {}).get('raw_score', 50)
    if vix_score > 35 or fg_score < 20:
        return 'CRISIS'
    return 'NORMAL'
