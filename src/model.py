"""
model.py
========

이 모듈은 사용할 수 있는 머신러닝 모델을 정의하고 초기화하는 기능을 제공합니다.
LightGBM, XGBoost, CatBoost 등 여러 트리 기반 부스팅 모델을 지원하며,
각 모델에 대한 초기 하이퍼파라미터를 설정하는 데 도움을 줍니다.

함수 설명
---------
get_model(name, params)
    주어진 이름과 하이퍼파라미터로 모델 인스턴스를 반환합니다.

MODEL_TYPES
    지원하는 모델 이름 리스트입니다.
"""

from __future__ import annotations

from typing import Any, Dict

MODEL_TYPES = ["lgbm", "xgb", "cat"]


def get_model(name: str, params: Dict[str, Any]):
    """모델 이름과 하이퍼파라미터로 모델 객체를 생성합니다.

    Args:
        name: 모델 이름. "lgbm", "xgb", "cat" 중 하나.
        params: 모델 생성에 사용될 하이퍼파라미터 딕셔너리.

    Returns:
        초기화된 모델 객체.

    Raises:
        ValueError: 지원하지 않는 모델 이름이 주어진 경우.
    """
    name = name.lower()
    if name == "lgbm":
        import lightgbm as lgb
        # LightGBM 의 LGBMClassifier 를 사용합니다.
        return lgb.LGBMClassifier(**params)
    elif name == "xgb":
        import xgboost as xgb
        # XGBoost 의 XGBClassifier 를 사용합니다. enable_categorical 은 train 단계에서 사용됩니다.
        return xgb.XGBClassifier(**params)
    elif name == "cat":
        from catboost import CatBoostClassifier
        # CatBoostClassifier 를 사용합니다.
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"지원하지 않는 모델 이름: {name}")