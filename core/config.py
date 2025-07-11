# 文件名: core/config.py
"""
存放问题的所有常量、参数和配置。
"""
import numpy as np

# --- 物理和地理参数 ---
CELL_SIZE = 5  # 栅格边长 (m)
ELEVATION_FACTOR_K = 5  # 高程变化因子

# --- 车辆技术指标 (附件3, 表1) ---
VEHICLE_PARAMS = {
    'A': {
        'max_slope': 30,
        'speed': {(0, 10): 30, (10, 20): 20, (20, 30): 10},  # km/h
        'power_consumption': {(0, 10): 1.0, (10, 20): 1.5, (20, 30): 2.0}  # %/km
    }
}

# --- 里程加权系数 (附件3, 表2) ---
# key: |dx|+|dy|, value: {d_theta: coeff}
MILEAGE_COEFFS = {
    1: {0: 1.0, 45: 1.5, 90: 2.0},
    2: {0: np.sqrt(2), 45: np.sqrt(2) + 0.5, 90: np.sqrt(2) + 1.0}
}

# --- 车头方向定义 ---
DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTIONS)}
IDX_TO_DIR = {i: d for i, d in enumerate(DIRECTIONS)}

