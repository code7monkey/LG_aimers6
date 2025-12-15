"""
inference.py
============

이 스크립트는 configs/submit.yaml 파일을 읽어 학습된 모델을 로드한 뒤
테스트 데이터에 대한 추론을 수행합니다. 추론 결과는 지정된 경로에
submission 파일로 저장됩니다.

사용 방법::

    python inference.py --config configs/submit.yaml
"""

from __future__ import annotations

import argparse
import os
import yaml
import pickle

import pandas as pd

from src.dataset import load_data, preprocess_data
from src.losses import weighted_rank_ensemble
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Inference using trained models and create submission")
    parser.add_argument('--config', type=str, default='configs/submit.yaml', help='Path to the inference config YAML file')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('random_seed', 42))
    # 데이터 로드 (train 데이터는 전처리 consistency를 위해 필요)
    train_df, test_df = load_data(cfg['train_path'], cfg['test_path'])
    # 전처리. 타겟 컬럼이 없는 경우에도 preprocess_data 가 동작하도록 구현되어 있다.
    X_train, y_train_unused, X_test, test_ids, categorical_features = preprocess_data(
        train_df, test_df, target_col=cfg.get('target_col', '임신 성공 여부')
    )
    # 모델 로드
    models_dir = cfg.get('model_dir', 'assets')
    model_names = cfg['models']  # ex: ['lgbm', 'xgb', 'cat']
    models = {name: [] for name in model_names}
    # 각 모델 fold 파일 로드
    for name in model_names:
        fold_idx = 0
        while True:
            model_file = os.path.join(models_dir, f"{name}_fold{fold_idx}.pkl")
            if not os.path.exists(model_file):
                break
            with open(model_file, 'rb') as mf:
                models[name].append(pickle.load(mf))
            fold_idx += 1
        if not models[name]:
            raise FileNotFoundError(f"No saved models found for {name} in {models_dir}")

    # 테스트 데이터 예측
    test_preds = {name: [] for name in model_names}
    # drop_features: 모델별 제거할 컬럼
    drop_features = cfg.get('drop_features', {})
    for name in model_names:
        drop_cols = drop_features.get(name, [])
        X_test_model = X_test.drop(columns=drop_cols, errors='ignore')
        # 예측 수행
        for model in models[name]:
            if name == 'lgbm':
                pred = model.predict(X_test_model)
            elif name == 'xgb':
                import xgboost as xgb
                pred = model.predict(xgb.DMatrix(X_test_model, enable_categorical=True))
            elif name == 'cat':
                pred = model.predict_proba(X_test_model)[:, 1]
            else:
                raise ValueError(f"Unknown model name: {name}")
            test_preds[name].append(pred)

    # fold 별 AUC 정보는 추론 단계에서 제공되지 않으므로 n_select_folds 와 weights 로 Top n fold 선택만 수행
    ensemble_cfg = cfg.get('weighted_rank_ensemble', {})
    n_select = ensemble_cfg.get('n_select_folds', 3)
    weights = ensemble_cfg.get('weights', {})
    selected_test_preds = []
    selected_weights = []
    # 단순히 앞에서부터 n_select fold 사용
    for name in model_names:
        preds_list = test_preds[name]
        # fold 수가 n_select 보다 작을 경우 모두 사용
        for idx, pred in enumerate(preds_list[:n_select]):
            selected_test_preds.append(pred)
            selected_weights.append(weights.get(name, 1.0))

    final_preds = weighted_rank_ensemble(selected_test_preds, selected_weights)

    # submission 생성
    if 'submission_sample_path' in cfg and os.path.exists(cfg['submission_sample_path']):
        submission_df = pd.read_csv(cfg['submission_sample_path'])
        if 'ID' in submission_df.columns:
            submission_df['ID'] = test_ids
        if submission_df.shape[1] > 1:
            submission_df.iloc[:, 1] = final_preds
    else:
        submission_df = pd.DataFrame({'ID': test_ids, 'label': final_preds})
    # 저장
    os.makedirs(cfg.get('output_dir', 'outputs'), exist_ok=True)
    submission_path = os.path.join(cfg.get('output_dir', 'outputs'), cfg.get('submission_filename', 'submission.csv'))
    submission_df.to_csv(submission_path, index=False)
    print(f"Inference submission saved to {submission_path}")


if __name__ == '__main__':
    main()