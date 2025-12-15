"""
train.py
========

이 스크립트는 configs/train.yaml 파일을 읽어 학습을 수행합니다.
데이터 로드 및 전처리, 모델 학습, 교차검증 평가, 앙상블까지 전 과정을 담당합니다.

사용 방법::

    python train.py --config configs/train.yaml

Colab 환경에서는 경로를 조정하여 실행할 수 있으며, 모든 학습 결과는
assets/ 디렉터리에 저장됩니다. 출력 submission 파일은 outputs/ 디렉터리에
저장됩니다.
"""

from __future__ import annotations

import argparse
import os
import yaml
import pickle

import numpy as np
import pandas as pd

from src.dataset import load_data, preprocess_data
from src.trainer import train_models
from src.losses import weighted_rank_ensemble
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train models and produce ensemble submission")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to the training config YAML file')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # config 파일 읽기
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # 시드 고정
    set_seed(cfg.get('random_seed', 42))
    # 데이터 로드
    train_df, test_df = load_data(cfg['train_path'], cfg['test_path'])
    # 전처리
    X_train, y_train, X_test, test_ids, categorical_features = preprocess_data(
        train_df,
        test_df,
        target_col=cfg.get('target_col', '임신 성공 여부'),
    )
    # 학습
    result = train_models(
        X_train,
        y_train,
        X_test,
        categorical_features,
        model_configs=cfg['models'],
        drop_features=cfg.get('drop_features', {}),
        n_splits=cfg.get('n_splits', 5),
        random_seed=cfg.get('random_seed', 42),
    )
    auc_scores = result['auc_scores']
    models = result['models']
    test_preds = result['test_preds']

    # fold 별 AUC 출력
    for model_name, scores in auc_scores.items():
        auc_values = [s for s, _ in scores]
        if auc_values:
            print(f"{model_name} 평균 AUC: {np.mean(auc_values):.5f}")
        else:
            print(f"{model_name} AUC 계산 불가")

    # 앙상블 진행
    ensemble_cfg = cfg.get('weighted_rank_ensemble', {})
    n_select = ensemble_cfg.get('n_select_folds', 3)
    weights = ensemble_cfg.get('weights', {})
    selected_test_preds = []
    selected_weights = []
    # 모델별로 AUC 상위 n_select fold 선택
    for model_name in cfg['models'].keys():
        # fold 별 AUC 정보를 내림차순으로 정렬
        scores = auc_scores.get(model_name, [])
        top_n = sorted(scores, reverse=True, key=lambda x: x[0])[:n_select]
        top_indices = [fold_idx for _, fold_idx in top_n]
        print(f"{model_name} 선택된 fold: {top_indices}")
        # 예측값과 가중치 축적
        for idx in top_indices:
            selected_test_preds.append(test_preds[model_name][idx])
            selected_weights.append(weights.get(model_name, 1.0))

    # 가중 랭크 앙상블 수행
    final_preds = weighted_rank_ensemble(selected_test_preds, selected_weights)

    # submission 생성
    # sample_submission 경로를 지정하지 않은 경우 test_ids 와 final_preds 로 DataFrame 생성
    if 'submission_sample_path' in cfg and os.path.exists(cfg['submission_sample_path']):
        submission_df = pd.read_csv(cfg['submission_sample_path'])
        if 'ID' in submission_df.columns:
            submission_df['ID'] = test_ids
        if submission_df.shape[1] > 1:
            submission_df.iloc[:, 1] = final_preds
    else:
        submission_df = pd.DataFrame({'ID': test_ids, 'label': final_preds})

    # outputs 디렉터리 생성
    os.makedirs(cfg.get('output_dir', 'outputs'), exist_ok=True)
    submission_path = os.path.join(cfg.get('output_dir', 'outputs'), cfg.get('submission_filename', 'submission.csv'))
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # 모델 저장 (옵션)
    models_dir = cfg.get('model_dir', 'assets')
    os.makedirs(models_dir, exist_ok=True)
    # 각 모델의 fold 별로 저장한다.
    for model_name, model_list in models.items():
        for fold_idx, model in enumerate(model_list):
            model_file = os.path.join(models_dir, f"{model_name}_fold{fold_idx}.pkl")
            with open(model_file, 'wb') as mf:
                pickle.dump(model, mf)

    print(f"모델들이 {models_dir}에 저장되었습니다.")


if __name__ == '__main__':
    main()