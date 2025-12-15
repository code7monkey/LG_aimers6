"""
dataset.py
===========

이 모듈은 훈련 및 테스트 데이터를 로드하고 전처리하는 함수를 제공합니다.
노트북에서 수행한 데이터 정제, 결측치 보간, 파생변수 생성과 같은
모든 로직을 함수 형태로 재구성했습니다.

사용 방법::

    from src.dataset import load_data, preprocess_data

    train_df, test_df = load_data('data/train.csv', 'data/test.csv')
    X_train, y_train, X_test, test_ids, categorical_features = preprocess_data(train_df, test_df)

함수 설명
---------
load_data(train_path, test_path)
    CSV 파일을 읽어 pandas DataFrame 으로 반환합니다.

preprocess_data(train_df, test_df)
    주어진 DataFrame 을 전처리하여 모델 학습에 사용할 특징과 타겟을 반환합니다.

각 함수는 학습/추론 파이프라인에서 재사용될 수 있도록 모듈화되어 있습니다.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, List


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """주어진 경로에서 학습 및 테스트 데이터를 읽어 반환합니다.

    Args:
        train_path: 학습 데이터 CSV 파일 경로.
        test_path: 테스트 데이터 CSV 파일 경로.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "임신 성공 여부"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """데이터를 전처리하여 학습/추론을 위한 특징을 생성합니다.

    노트북에서 수행한 모든 데이터 전처리 작업을 함수화했습니다.
    카테고리형 변수는 pandas 의 ``category`` 타입으로 변환하여
    각 모델에서 범주형으로 인식될 수 있도록 처리합니다.

    Args:
        train_df: 로드된 학습 데이터.
        test_df: 로드된 테스트 데이터.
        target_col: 타겟 변수 컬럼명. 기본값은 "임신 성공 여부".

    Returns:
        Tuple containing:
            - X_train: pd.DataFrame, 전처리 후 학습 특징.
            - y_train: pd.Series, 타겟 값.
            - X_test: pd.DataFrame, 전처리 후 테스트 특징.
            - test_ids: pd.Series, 테스트 데이터의 ID 컬럼.
            - categorical_features: List[str], 범주형 변수 목록.
    """

    # ID 분리
    test_ids = test_df["ID"].copy() if "ID" in test_df.columns else pd.Series()
    if "ID" in train_df.columns:
        train_df = train_df.drop(columns=["ID"])
    if "ID" in test_df.columns:
        test_df = test_df.drop(columns=["ID"])

    # 난자 출처 알 수 없음 -> 본인 제공 (컬럼 존재 시)
    for df in (train_df, test_df):
        if "난자 출처" in df.columns:
            df["난자 출처"] = df["난자 출처"].replace("알 수 없음", "본인 제공")

    # 특정 시술 유형 결측치 대체 (컬럼 존재 시)
    for df in (train_df, test_df):
        if "특정 시술 유형" in df.columns:
            df["특정 시술 유형"] = df["특정 시술 유형"].fillna("Unknown")
    # PGD, 착상 전 유전 검사, PGS 시술 여부 결측치 대체
    s_col = ["PGD 시술 여부", "착상 전 유전 검사 사용 여부", "PGS 시술 여부"]
    for df in (train_df, test_df):
        for col in s_col:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    # DI 사례의 결측치 처리
    columns_to_update = [
        "단일 배아 이식 여부", "착상 전 유전 진단 사용 여부", "배아 생성 주요 이유",
        "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
        "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수",
        "해동된 배아 수", "해동 난자 수", "수집된 신선 난자 수", "저장된 신선 난자 수",
        "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
        "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부",
    ]
    for df in (train_df, test_df):
        for column in columns_to_update:
            if column not in df.columns:
                continue
            if df[column].dtype == object:
                df[column] = df[column].fillna("Not Answer(DI)")
            else:
                df[column] = df[column].fillna(0)

    # 경과일 컬럼 결측치 대체
    o_col = ["난자 채취 경과일"]
    for df in (train_df, test_df):
        for col in o_col:
            if col in df.columns:
                df[col] = df[col].fillna(1)
    d_col = ["난자 혼합 경과일", "배아 이식 경과일", "배아 해동 경과일"]
    for df in (train_df, test_df):
        for col in d_col:
            if col in df.columns:
                df[col] = df[col].fillna(999)

    # 횟수 컬럼의 문자열 숫자 추출 후 정수형 변환
    int_col = [
        "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
        "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수",
        "DI 출산 횟수",
    ]
    for df in (train_df, test_df):
        for col in int_col:
            # 데이터셋에 해당 컬럼이 없는 경우 건너뜀
            if col not in df.columns:
                continue
            # 일부 데이터는 '3번' 과 같이 문자열로 표기되어 있어 숫자만 추출한다.
            df[col] = df[col].astype(str).str.extract(r"(\d+)").astype(float).fillna(0).astype(int)

    # 훈련 데이터에서만 등장하는 희귀 범주 제거
    if '배란 유도 유형' in train_df.columns:
        to_drop = train_df['배란 유도 유형'].value_counts()
        to_drop = to_drop[to_drop == 1].index
        train_df = train_df[~train_df['배란 유도 유형'].isin(to_drop)].copy()

    # 난자 출처가 "본인 제공"인 경우 난자 기증자 나이를 시술 당시 나이로 설정
    def fill_donor_age(df: pd.DataFrame) -> None:
        """난자 출처가 본인 제공일 때 난자 기증자 나이를 시술 당시 나이로 대체합니다.

        컬럼이 존재하지 않는 경우 아무 동작을 하지 않습니다.
        """
        # 필수 컬럼이 모두 존재하는지 확인
        if not all(col in df.columns for col in ['난자 출처', '난자 기증자 나이', '시술 당시 나이']):
            return
        cond = df['난자 출처'] == "본인 제공"
        df.loc[cond, '난자 기증자 나이'] = df.loc[cond, '시술 당시 나이']
    fill_donor_age(train_df)
    fill_donor_age(test_df)

    # 필요없는 컬럼 제거
    drop_cols = [
        '임신 시도 또는 마지막 임신 경과 연수', '난자 해동 경과일', '불임 원인 - 정자 형태',
        '불임 원인 - 정자 운동성', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 농도',
        '불임 원인 - 자궁경부 문제', '불임 원인 - 여성 요인', '대리모 여부', '부부 부 불임 원인'
    ]
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # 배아 생성 주요 이유 키워드 특징 추출
    keywords = ['기증용', '현재 시술용', '난자 저장용', '배아 저장용']
    for df in (train_df, test_df):
        if '배아 생성 주요 이유' in df.columns:
            col_series = df['배아 생성 주요 이유'].astype(str)
            for keyword in keywords:
                df[keyword] = col_series.str.contains(keyword).astype(int)
            # 원본 컬럼 제거
            df.drop(columns=['배아 생성 주요 이유'], inplace=True)
        else:
            # 컬럼이 없을 경우 키워드별 컬럼을 0으로 생성
            for keyword in keywords:
                df[keyword] = 0

    # 특정 시술 유형 문자열을 키워드 개수로 변환
    unique_types = ['ICSI', 'IVF', 'Unknown', 'IUI', 'BLASTOCYST', 'AH']
    for df in (train_df, test_df):
        if '특정 시술 유형' in df.columns:
            col_series = df['특정 시술 유형'].astype(str)
            for keyword in unique_types:
                df[keyword] = col_series.str.count(keyword)
            # 원본 컬럼 제거
            df.drop(columns=['특정 시술 유형'], inplace=True)
        else:
            for keyword in unique_types:
                df[keyword] = 0

    # 파생 변수: 난임 여부, 유산 여부
    # 임신 시술 실패 횟수
    # 총 시술 횟수와 총 임신 횟수가 없는 경우 0으로 간주하여 계산
    train_df['난임 여부'] = train_df.get('총 시술 횟수', 0) - train_df.get('총 임신 횟수', 0)
    test_df['난임 여부'] = test_df.get('총 시술 횟수', 0) - test_df.get('총 임신 횟수', 0)
    # 임신 후 출산까지 이어지지 않은 경우 (값이 1 이상이면 유산)
    # 총 임신 횟수와 총 출산 횟수가 없는 경우 0으로 채우되, Series 형태로 만들어 apply 가 가능하도록 한다.
    diff_train = train_df.get('총 임신 횟수', pd.Series([0] * len(train_df))) - train_df.get('총 출산 횟수', pd.Series([0] * len(train_df)))
    train_df['유산 여부'] = diff_train.apply(lambda x: 1 if x > 0 else 0)
    diff_test = test_df.get('총 임신 횟수', pd.Series([0] * len(test_df))) - test_df.get('총 출산 횟수', pd.Series([0] * len(test_df)))
    test_df['유산 여부'] = diff_test.apply(lambda x: 1 if x > 0 else 0)

    # 파생 변수: 미세주입이 아닌 배아 통계
    train_df['미세주입이 아닌 배아 이식 수'] = train_df.get('이식된 배아 수', 0) - train_df.get('미세주입 배아 이식 수', 0)
    test_df['미세주입이 아닌 배아 이식 수'] = test_df.get('이식된 배아 수', 0) - test_df.get('미세주입 배아 이식 수', 0)
    train_df['미세주입이 아닌 배아 생성 수'] = train_df.get('총 생성 배아 수', 0) - train_df.get('미세주입에서 생성된 배아 수', 0)
    test_df['미세주입이 아닌 배아 생성 수'] = test_df.get('총 생성 배아 수', 0) - test_df.get('미세주입에서 생성된 배아 수', 0)
    train_df['미세주입이 아닌 배아 저장 수'] = train_df.get('저장된 배아 수', 0) - train_df.get('미세주입 후 저장된 배아 수', 0)
    test_df['미세주입이 아닌 배아 저장 수'] = test_df.get('저장된 배아 수', 0) - test_df.get('미세주입 후 저장된 배아 수', 0)

    # 파생 변수: 외부 시술 횟수 및 PGS 검사 후 미시술
    train_df['클리닉 외 총 시술 횟수'] = train_df.get('총 시술 횟수', 0) - train_df.get('클리닉 내 총 시술 횟수', 0)
    test_df['클리닉 외 총 시술 횟수'] = test_df.get('총 시술 횟수', 0) - test_df.get('클리닉 내 총 시술 횟수', 0)
    train_df['PGS 검사를 받고도 안한 사람'] = train_df.get('착상 전 유전 검사 사용 여부', 0) - train_df.get('PGS 시술 여부', 0)
    test_df['PGS 검사를 받고도 안한 사람'] = test_df.get('착상 전 유전 검사 사용 여부', 0) - test_df.get('PGS 시술 여부', 0)

    # 파생 변수: 배아 변수 조정 (단일 배아 이식 성공률 향상 전략)
    # 단일 배아를 이식한 경우, 이식된 배아 수가 1이면 1.5로 조정
    for df in (train_df, test_df):
        # 단일 배아 이식 조정
        if '단일 배아 이식 여부' in df.columns and '이식된 배아 수' in df.columns:
            mask = (df['단일 배아 이식 여부'] == 1) & (df['이식된 배아 수'] == 1)
            df.loc[mask, '이식된 배아 수'] = 1.5
        # 착상 전 유전 검사 후 PGS 시술 미실시 조정
        if '착상 전 유전 검사 사용 여부' in df.columns and 'PGS 시술 여부' in df.columns:
            mask2 = (df['착상 전 유전 검사 사용 여부'] == 1) & (df['PGS 시술 여부'] == 0)
            df.loc[mask2, 'PGS 시술 여부'] = -1

    # 파생 변수: 나이별 배아 수 비율
    age_bin_mapping = {
        "만18-34세": 1,
        "만35-37세": 2,
        "만38-39세": 3,
        "만40-42세": 4,
        "만43-44세": 5,
        "만45-50세": 6,
        "알 수 없음": 7,
    }
    for df in (train_df, test_df):
        # 나이 매핑. 컬럼이 없는 경우 1로 대체
        if '시술 당시 나이' in df.columns:
            df['나이'] = df['시술 당시 나이'].map(age_bin_mapping).fillna(7)
        else:
            df['나이'] = 1
        # 배아 수 컬럼이 없는 경우 0으로 채움
        if '이식된 배아 수' not in df.columns:
            df['이식된 배아 수'] = 0
        df['이식 배아 수/나이'] = df['이식된 배아 수'] / df['나이']
        df.drop(columns=['나이'], inplace=True)

    # 범주형 변수 목록 정의
    cat_col = [
        '배란 자극 여부', '단일 배아 이식 여부', '착상 전 유전 검사 사용 여부', '착상 전 유전 진단 사용 여부',
        '남성 주 불임 원인', '남성 부 불임 원인', '여성 주 불임 원인', '여성 부 불임 원인', '부부 주 불임 원인',
        '불명확 불임 원인', '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애',
        '불임 원인 - 자궁내막증', '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수',
        'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수', '동결 배아 사용 여부',
        '신선 배아 사용 여부', '기증 배아 사용 여부', 'PGD 시술 여부', 'PGS 시술 여부', '기증용', '현재 시술용',
        '난자 저장용', '배아 저장용', 'ICSI', 'IVF', 'Unknown', 'IUI', 'BLASTOCYST', 'AH', '난임 여부', '유산 여부',
        'PGS 검사를 받고도 안한 사람'
    ]
    # 존재하지 않는 컬럼은 제거
    cat_col = [col for col in cat_col if col in train_df.columns]

    # 범주형 변수를 category 타입으로 변환
    for df in (train_df, test_df):
        for col in cat_col:
            df[col] = df[col].astype(str).astype('category')

    # 학습/테스트 특징 및 타겟 생성
    y_train = train_df[target_col].copy() if target_col in train_df.columns else pd.Series()
    X_train = train_df.drop(columns=[target_col]) if target_col in train_df.columns else train_df.copy()
    X_test = test_df.copy()

    # categorical_features list 반환
    categorical_features = cat_col

    return X_train, y_train, X_test, test_ids, categorical_features