"""
utils.py
========

공통적으로 사용되는 유틸리티 함수를 정의합니다. 현재는 재현성을 위한 시드 고정
함수만을 제공하지만, 필요에 따라 로깅 설정, 폴더 생성 등의 기능을 추가할 수
있습니다.
"""

from __future__ import annotations

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """난수 시드를 고정하여 실험의 재현성을 높입니다.

    Args:
        seed: 고정할 시드 값. 기본값은 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)