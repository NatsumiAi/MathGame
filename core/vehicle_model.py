# 文件名: core/vehicle_model.py (最终注释版)
# 描述: 封装与无人车模型相关的所有计算逻辑，包括速度、里程、平稳性以及车辆运动学转向规则。

import numpy as np
from collections import defaultdict
from . import config  # 从同级包中导入config


def get_speed_by_slope(slope, vehicle_type='A'):
    """
    根据地形坡度获取无人车的行驶速度。
    (此函数用于问题1)

    :param slope: 当前栅格的坡度 (度)。
    :param vehicle_type: 车辆型号，默认为'A'。
    :return: 对应的行驶速度 (km/h)。
    """
    # 从配置中获取指定型号车辆的速度参数
    speed_rules = config.VEHICLE_PARAMS[vehicle_type]['speed']

    # 特殊处理：对于坡度规则中的最大值，其区间通常是左闭右闭
    last_slope_range = max(speed_rules.keys())
    if slope == last_slope_range[1]:
        return speed_rules[last_slope_range]

    # 遍历其余速度规则（通常是左闭右开）
    for (s_min, s_max), speed in speed_rules.items():
        if s_min <= slope < s_max:
            return speed

    # 如果没有匹配到任何规则（理论上不应该发生），返回一个安全的默认值
    return speed_rules.get((10, 20), 0)


def calculate_segment_mileage(dx, dy, d_theta):
    """
    计算单个路段的里程。
    (此函数用于问题1)

    :param dx: x方向的栅格位移。
    :param dy: y方向的栅格位移。
    :param d_theta: 发生的转向角度 (度)。
    :return: 该路段的里程 (米)。
    """
    # 根据位移计算移动类型 (直行=1, 斜行=2)
    move_type = abs(dx) + abs(dy)
    if move_type not in config.MILEAGE_COEFFS:
        return 0.0

    # 根据移动类型和转向角度，从配置中查找里程加权系数
    mileage_coeff = config.MILEAGE_COEFFS[move_type].get(d_theta, 1.0)

    # 最终里程 = 加权系数 * 栅格尺寸
    return mileage_coeff * config.CELL_SIZE

def calculate_angle_diff(angle1, angle2):
    """
    计算两个角度之间的最小正向差值 (0-180度)。
    (此函数用于问题1)

    :param angle1: 第一个角度 (度)。
    :param angle2: 第二个角度 (度)。
    :return: 两个角度之间的最小正向差值。
    """
    diff = abs(angle1 - angle2)
    # 角度差值可能是大于180的大角，也可能是小于180的小角，取小者
    return min(diff, 360 - diff)


def calculate_stability_cost(normal_vec1, normal_vec2, slope1, slope2):
    """
    计算单个路段的平稳性成本。
    (此函数用于问题4)

    :param normal_vec1: 起点的地表法向量。
    :param normal_vec2: 终点的地表法向量。
    :param slope1: 起点的坡度。
    :param slope2: 终点的坡度。
    :return: 该路段的平稳性成本。
    """
    # 计算两个法向量之间的夹角的余弦值
    cos_theta = np.dot(normal_vec1, normal_vec2) / (np.linalg.norm(normal_vec1) * np.linalg.norm(normal_vec2))

    # 使用反余弦得到法向量的夹角（弧度），np.clip确保输入在[-1, 1]范围内以避免数学错误
    angle_change_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 用该段的平均坡度对地表法向量的角度变化进行加权，作为平稳性成本
    avg_slope = (slope1 + slope2) / 2
    return avg_slope * angle_change_rad


def generate_turn_rules():
    rules = defaultdict(lambda: defaultdict(list))

    # --- PART 1: 主方向 (0, 90, 180, 270) ---
    moves_cardinal = {(-1, 1): -45, (0, 1): 0, (1, 1): 45}
    for base_dir in [0, 90, 180, 270]:
        for move, angle_change in moves_cardinal.items():
            # 坐标旋转
            if base_dir == 0:
                actual_move = move
            elif base_dir == 90:
                actual_move = (move[1], -move[0])
            elif base_dir == 180:
                actual_move = (-move[0], -move[1])
            else:  # base_dir == 270
                actual_move = (-move[1], move[0])

            # 应用图2规则
            if angle_change == 0:
                offsets = [-45, 0, 45]
            else:
                offsets = [0, angle_change]

            # 统一计算最终朝向
            final_center = base_dir + angle_change
            possible_headings = sorted([(final_center + d + 360) % 360 for d in offsets])
            rules[base_dir][actual_move] = possible_headings

    # --- PART 2: 斜方向 (45, 135, 225, 315) ---
    moves_diagonal = {(0, 1): -45, (1, 1): 0, (1, 0): 45}
    for base_dir in [45, 135, 225, 315]:
        for move, angle_change in moves_diagonal.items():
            # 坐标旋转
            if base_dir == 45:
                actual_move = move
            elif base_dir == 135:
                actual_move = (move[1], -move[0])
            elif base_dir == 225:
                actual_move = (-move[0], -move[1])
            else:  # base_dir == 315
                actual_move = (-move[1], move[0])

            if angle_change == 0:
                offsets = [-45, 0, 45]
            else:
                offsets = [0, angle_change]

            # 统一计算最终朝向
            final_center = base_dir + angle_change
            possible_headings = sorted([(final_center + d + 360) % 360 for d in offsets])
            rules[base_dir][actual_move] = possible_headings

    return rules


