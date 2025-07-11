# 文件名: problem1_solver.py (重构并优化绘图)
# 描述: 调用核心模块来解决问题一。绘图函数被修改，以生成四张独立的、更大、更清晰的图像。

import numpy as np
import matplotlib.pyplot as plt
import os
# --- 核心修改：现在只从 vehicle_model 调用所需函数 ---
from core import config, data_loader, geo_calculator, vehicle_model

plt.rcParams['font.sans-serif'] = ['STSONG']
plt.rcParams['axes.unicode_minus'] = False


def solve():
    """主解决函数，负责整个问题一的计算和评估流程。"""
    # === 步骤 1: 加载数据 ===
    df = data_loader.load_path_data('P1-P2的行驶路径.xlsx')
    bad_zones = data_loader.load_bad_zones()
    map_data = data_loader.load_map_data()

    # === 步骤 2: 初始化计算器 ===
    calculator = geo_calculator.GeoCalculator(map_data)

    # === 步骤 3: 迭代计算路径性能指标 (重构后) ===
    total_mileage, total_time, total_stability, total_safety_time = 0.0, 0.0, 0.0, 0.0

    start_elevation = calculator.get_elevation(df.iloc[0]['x'], df.iloc[0]['y'])
    mileage_curve, time_curve, elevation_curve = [0], [0], [start_elevation]
    slope_curve, speed_curve = [], []

    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1], df.iloc[i]

        # --- 使用 vehicle_model 和 geo_calculator 进行计算 ---

        # 里程计算
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        d_theta = vehicle_model.calculate_angle_diff(curr['heading'], prev['heading'])
        step_mileage = vehicle_model.calculate_segment_mileage(dx, dy, d_theta)

        # 时效性计算
        curr_slope, _ = calculator.get_slope_and_aspect(curr['x'], curr['y'])
        curr_speed_kmh = vehicle_model.get_speed_by_slope(curr_slope)
        step_time = step_mileage / (curr_speed_kmh / 3.6) if curr_speed_kmh > 0 else 0

        # 安全性计算
        if (curr['x'], curr['y']) in bad_zones:
            total_safety_time += step_time

        # 平稳性计算
        vec_prev = calculator.get_normal_vector(prev['x'], prev['y'])
        vec_curr = calculator.get_normal_vector(curr['x'], curr['y'])
        prev_slope, _ = calculator.get_slope_and_aspect(prev['x'], prev['y'])
        step_stability = vehicle_model.calculate_stability_cost(vec_prev, vec_curr, prev_slope, curr_slope)

        # --- 累加总值 ---
        total_mileage += step_mileage
        total_time += step_time
        total_stability += step_stability

        # --- 更新绘图数据 ---
        mileage_curve.append(total_mileage)
        time_curve.append(total_time)
        elevation_curve.append(calculator.get_elevation(curr['x'], curr['y']))
        slope_curve.append(curr_slope)
        speed_curve.append(curr_speed_kmh)

    # === 步骤 4: 打印结果和生成图表 ===
    print(
        f"\n问题1 计算结果:\n"
        f"  总里程: {total_mileage:.4f} 米\n"
        f"  时效性: {total_time:.4f} 秒\n"
        f"  平稳性: {total_stability:.4f}\n"
        f"  安全性: {total_safety_time:.4f} 秒"
    )
    plot_curves({
        'mileage': mileage_curve, 'time': time_curve, 'elevation': elevation_curve,
        'slope': slope_curve, 'speed': speed_curve
    })


# --- 这是被完全重构的函数 ---
def plot_curves(curves):
    """
    为每个性能指标生成独立的、更大、更清晰的图表，并保存到文件。
    """
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n问题1 曲线图将保存至目录: {output_dir}")

    # 内部帮助函数，用于绘制和保存单个图表，避免代码重复
    def _save_single_plot(x_data, y_data, title, xlabel, ylabel, filename, color):
        # 1. 创建一个独立的、更大的画布
        plt.figure(figsize=(12, 8))

        # 2. 绘制更清晰的线条
        plt.plot(x_data, y_data, color=color, linewidth=2.5)

        # 3. 设置更大的字体
        plt.title(title, fontsize=20)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 4. 添加网格并收紧布局
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # 5. 保存为高分辨率图像
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()  # 关闭画布，释放内存
        print(f"  - 图表已保存: {full_path}")

    # --- 为四个指标分别调用绘图函数 ---

    # 图1: 里程-时间
    _save_single_plot(
        x_data=curves['time'], y_data=curves['mileage'],
        title='路径P1-P2: 里程-时间关系图',
        xlabel='时间 (秒)', ylabel='里程 (米)',
        filename='curve_mileage_vs_time.png', color='b'
    )

    # 图2: 高程-里程
    _save_single_plot(
        x_data=curves['mileage'], y_data=curves['elevation'],
        title='路径P1-P2: 高程-里程关系图',
        xlabel='里程 (米)', ylabel='高程 (米)',
        filename='curve_elevation_vs_mileage.png', color='g'
    )

    # 图3: 坡度-里程 (注意x轴数据长度匹配)
    _save_single_plot(
        x_data=curves['mileage'][1:], y_data=curves['slope'],
        title='路径P1-P2: 坡度-里程关系图',
        xlabel='里程 (米)', ylabel='坡度 (度)',
        filename='curve_slope_vs_mileage.png', color='r'
    )

    # 图4: 速度-里程 (注意x轴数据长度匹配)
    _save_single_plot(
        x_data=curves['mileage'][1:], y_data=curves['speed'],
        title='路径P1-P2: 速度-里程关系图',
        xlabel='里程 (米)', ylabel='速度 (km/h)',
        filename='curve_speed_vs_mileage.png', color='m'
    )


if __name__ == '__main__':
    solve()
