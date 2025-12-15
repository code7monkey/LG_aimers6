"""
losses.py
=========

이 모듈에서는 커스텀 평가 함수나 앙상블 기법을 정의합니다.
현재는 가중 랭크 앙상블 함수를 제공합니다. 이는 모델별로 예측한 확률값을
랭크 형태로 변환한 후 가중 평균을 통해 최종 예측을 생성하는 방식입니다.
이 방식을 통해 서로 다른 모델이 출력하는 확률 분포의 스케일 차이를 완화하고
상대적인 순위를 기반으로 보다 안정적인 앙상블 결과를 얻을 수 있습니다.
"""

from __future__ import annotations

from typing import List
import numpy as np
from scipy.stats import rankdata


def weighted_rank_ensemble(predictions: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """가중 랭크 변환을 이용한 앙상블.

    여러 모델의 예측값을 받아 각 예측값을 rank 로 변환한 뒤,
    가중치를 적용하여 평균을 구합니다. 최종적으로 0~1 사이로 정규화된
    랭크 점수를 반환합니다.

    Args:
        predictions: shape (n_models, n_samples) 형태의 예측값 리스트.
        weights: 각 모델에 적용할 가중치. predictions 와 길이가 같아야 합니다.

    Returns:
        np.ndarray: 0~1 범위로 정규화된 가중 랭크 앙상블 결과.
    """
    preds_array = np.array(predictions)
    if preds_array.ndim != 2:
        # (n_models, n_samples) 형태가 아닐 경우 reshape 시도
        preds_array = preds_array.reshape(len(predictions), -1)
    # 각 모델의 예측값을 랭크로 변환
    ranked_preds = np.array([rankdata(pred) for pred in preds_array])
    # 가중 평균 랭크 계산
    weighted_avg_rank = np.average(ranked_preds, axis=0, weights=weights)
    # 0~1 범위로 정규화
    return weighted_avg_rank / np.max(weighted_avg_rank)