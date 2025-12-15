"""
trainer.py
==========

이 모듈은 K-Fold 교차 검증을 통해 여러 모델을 학습하고 평가하는 함수를 제공합니다.
LightGBM, XGBoost, CatBoost 모델을 지원하며, 클래스 불균형을 고려한 가중치 적용,
범주형 변수 처리 등 노트북에서 수행했던 로직을 재현합니다.

사용 예::

    from src.trainer import train_models
    from sklearn.model_selection import StratifiedKFold
    from src.dataset import preprocess_data, load_data
    from src.model import get_model
    from src.utils import set_seed

    # 데이터 로드 및 전처리
    train_df, test_df = load_data('data/train.csv', 'data/test.csv')
    X_train, y_train, X_test, test_ids, categorical_features = preprocess_data(train_df, test_df)

    # 모델 설정
    model_configs = {
        'lgbm': {...},
        'xgb': {...},
        'cat': {...},
    }
    drop_features = {
        'lgbm': [],
        'xgb': ['PGS 검사를 받고도 안한 사람'],
        'cat': ['PGS 검사를 받고도 안한 사람'],
    }
    result = train_models(X_train, y_train, X_test, categorical_features, model_configs,
                          drop_features, n_splits=5, random_seed=42)

함수 설명
---------
train_models(X_train, y_train, X_test, categorical_features, model_configs,
             drop_features, n_splits, random_seed)
    각 모델을 K-Fold 전략으로 학습하고 검증 AUC를 계산합니다.
    테스트 데이터에 대한 예측 결과도 fold 별로 저장합니다.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: List[str],
    model_configs: Dict[str, Dict[str, Any]],
    drop_features: Dict[str, List[str]],
    n_splits: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """여러 모델을 교차검증으로 학습하고 평가합니다.

    Args:
        X_train: 학습 데이터 특징.
        y_train: 학습 데이터 타겟.
        X_test: 테스트 데이터 특징.
        categorical_features: 범주형 변수 목록.
        model_configs: 각 모델 이름에 대한 하이퍼파라미터 딕셔너리.
        drop_features: 모델 별 제거할 컬럼 목록.
        n_splits: K-Fold 분할 수.
        random_seed: 난수 시드.

    Returns:
        dict: 다음 정보를 담은 딕셔너리
            - auc_scores: {model_name: [(auc, fold_index), ...]}
            - models: {model_name: [model_per_fold, ...]}
            - test_preds: {model_name: [np.ndarray_predictions_per_fold, ...]}
    """

    # 클래스 불균형 대비 pos_weight 계산
    class_counts = y_train.value_counts()
    pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts and class_counts[1] != 0 else 1.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    auc_scores: Dict[str, List[Tuple[float, int]]] = {key: [] for key in model_configs.keys()}
    models: Dict[str, List[Any]] = {key: [] for key in model_configs.keys()}
    test_preds: Dict[str, List[np.ndarray]] = {key: [] for key in model_configs.keys()}

    # 범주형 feature 인덱스 (CatBoost 용)
    cat_features_indices = [X_train.columns.get_loc(col) for col in categorical_features]

    for model_name, params in model_configs.items():
        # 각 모델마다 데이터를 별도로 준비한다.
        # 지정된 drop_features 에 맞추어 컬럼을 제거한다.
        drop_cols = drop_features.get(model_name, [])
        X_train_model = X_train.drop(columns=drop_cols, errors='ignore')
        X_test_model = X_test.drop(columns=drop_cols, errors='ignore')
        # categorical_features 에서 drop 한 컬럼은 제거
        cat_feats_model = [col for col in categorical_features if col not in drop_cols and col in X_train_model.columns]
        # 인덱스 변환
        cat_feats_indices = [X_train_model.columns.get_loc(c) for c in cat_feats_model]

        # 각 fold 학습
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_model, y_train)):
            X_tr, X_val = X_train_model.iloc[train_idx], X_train_model.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            if model_name == 'lgbm':
                import lightgbm as lgb
                # 학습용 Dataset
                lgb_train = lgb.Dataset(
                    X_tr,
                    label=y_tr,
                    weight=y_tr.map({0: 1, 1: pos_weight}),
                    categorical_feature=cat_feats_indices
                )
                lgb_valid = lgb.Dataset(
                    X_val,
                    label=y_val,
                    weight=y_val.map({0: 1, 1: pos_weight}),
                    categorical_feature=cat_feats_indices
                )
                # LightGBM 학습
                model = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_valid],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
                )
                # 검증 예측 및 AUC
                y_pred_proba = model.predict(X_val)
                auc = roc_auc_score(y_val, y_pred_proba)
                auc_scores[model_name].append((auc, fold))
                models[model_name].append(model)
                # 테스트 예측 저장
                test_pred = model.predict(X_test_model)
                test_preds[model_name].append(test_pred)

            elif model_name == 'xgb':
                import xgboost as xgb
                # XGBoost 는 DMatrix 를 사용하며 enable_categorical=True 설정
                dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
                dvalid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
                # num_boost_round 와 early_stopping_rounds 설정
                # XGBoost 의 파라미터에 n_estimators 가 있다면 이를 round 로 사용
                params_copy = params.copy()
                num_boost_round = params_copy.pop('n_estimators', 1000)
                early_rounds = params_copy.pop('early_stopping_rounds', 100)
                model = xgb.train(
                    params_copy,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dvalid, 'validation')],
                    early_stopping_rounds=early_rounds,
                    verbose_eval=False,
                )
                # 검증 예측 및 AUC
                y_pred_proba = model.predict(xgb.DMatrix(X_val, enable_categorical=True))
                auc = roc_auc_score(y_val, y_pred_proba)
                auc_scores[model_name].append((auc, fold))
                models[model_name].append(model)
                # 테스트 예측 저장
                test_pred = model.predict(xgb.DMatrix(X_test_model, enable_categorical=True))
                test_preds[model_name].append(test_pred)

            elif model_name == 'cat':
                from catboost import CatBoostClassifier, Pool
                params_copy = params.copy()
                # CatBoost 는 scale_pos_weight 를 사용해 클래스 불균형을 보정할 수 있다.
                # 사용자가 입력하지 않은 경우 자동으로 설정한다.
                if 'scale_pos_weight' not in params_copy:
                    params_copy['scale_pos_weight'] = pos_weight
                # Pool 생성
                train_pool = Pool(X_tr, label=y_tr, cat_features=cat_feats_indices)
                valid_pool = Pool(X_val, label=y_val, cat_features=cat_feats_indices)
                model = CatBoostClassifier(**params_copy)
                model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100, verbose=False)
                # 검증 예측 및 AUC
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba)
                auc_scores[model_name].append((auc, fold))
                models[model_name].append(model)
                # 테스트 예측 저장
                test_pred = model.predict_proba(X_test_model)[:, 1]
                test_preds[model_name].append(test_pred)

            else:
                raise ValueError(f"알 수 없는 모델명: {model_name}")

    return {
        'auc_scores': auc_scores,
        'models': models,
        'test_preds': test_preds,
    }