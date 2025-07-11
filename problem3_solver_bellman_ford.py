# 文件名: problem3_solver_bellman_ford.py (使用 plot_tools 升级版)

import pandas as pd
import time
from collections import defaultdict

# 导入您提供的所有核心模块
from core import data_loader, config, vehicle_model, plot_tools
from core.geo_calculator import GeoCalculator


def solve_with_bellman_ford(path_points, turn_rules):
    """
    使用 Bellman-Ford 算法思想（等价于本场景的动态规划）寻找总里程最短的路径。
    (此函数逻辑保持不变)

    :param path_points: 路径点列表，每个点是一个字典，如 {'id': 1, 'x': 4697, 'y': 6162}。
    :param turn_rules: 车辆运动学转向规则字典。
    :return: (DataFrame, float) 包含最优朝向的路径DataFrame，以及最小总里程。
    """
    print("🚀 开始使用 Bellman-Ford / 动态规划算法求解...")
    start_time = time.time()

    num_points = len(path_points)
    headings = config.DIRECTIONS

    # 1. 初始化距离和前驱字典
    distance = defaultdict(lambda: float('inf'))
    predecessor = {}

    # 2. 设置起点 (问题3，所有8个朝向皆有可能)
    for h in headings:
        distance[(0, h)] = 0

    # 3. 核心：迭代松弛 (Relaxation)
    for i in range(num_points - 1):
        p_curr = path_points[i]
        p_next = path_points[i + 1]
        x1, y1, x2, y2 = p_curr['x'], p_curr['y'], p_next['x'], p_next['y']
        dx, dy = x2 - x1, y2 - y1

        for h_prev in headings:
            if distance[(i, h_prev)] == float('inf'):
                continue

            possible_next_headings = turn_rules[h_prev].get((dx, dy))
            if possible_next_headings is None:
                continue

            for h_next in possible_next_headings:
                d_theta = vehicle_model.calculate_angle_diff(h_next, h_prev)
                step_mileage = vehicle_model.calculate_segment_mileage(dx, dy, d_theta)
                new_dist = distance[(i, h_prev)] + step_mileage

                if new_dist < distance[(i + 1, h_next)]:
                    distance[(i + 1, h_next)] = new_dist
                    predecessor[(i + 1, h_next)] = h_prev

    end_time = time.time()
    print(f"✅ 算法计算完成，耗时: {end_time - start_time:.4f} 秒")

    # 4. 回溯路径
    final_point_idx = num_points - 1
    best_final_heading, min_total_mileage = -1, float('inf')
    for h in headings:
        if distance[(final_point_idx, h)] < min_total_mileage:
            min_total_mileage = distance[(final_point_idx, h)]
            best_final_heading = h

    if min_total_mileage == float('inf'):
        print("❌ 错误：未能找到从起点到终点的有效路径！")
        return None, float('inf')

    print(f"✔️ 找到最优路径，总里程: {min_total_mileage:.2f} m")

    optimal_path_headings = [0] * num_points
    optimal_path_headings[final_point_idx] = best_final_heading
    current_h = best_final_heading
    for i in range(final_point_idx, 0, -1):
        prev_h = predecessor.get((i, current_h))
        if prev_h is None:
            print(f"❌ 警告: 回溯路径时，状态 ({i}, {current_h}) 没有前驱节点。")
            break
        optimal_path_headings[i - 1] = prev_h
        current_h = prev_h

    result_df = pd.DataFrame(path_points)
    result_df['heading'] = optimal_path_headings

    return result_df, min_total_mileage


if __name__ == '__main__':
    # --- 1. 数据加载 ---
    print("=" * 30 + " 1. 数据加载 " + "=" * 30)
    map_data = data_loader.load_map_data()
    bad_zones = data_loader.load_bad_zones()
    path_df = data_loader.load_path_data('P5-P6的行驶路径.xlsx')
    path_points_list = path_df.to_dict('records')
    turn_rules = vehicle_model.generate_turn_rules()

    # --- 2. 准备绘图和计算资源 ---
    print("\n" + "=" * 25 + " 2. 准备绘图和计算资源 " + "=" * 25)
    # GeoCalculator 是计算坡度的基础
    calculator = GeoCalculator(map_data)
    # 使用 plot_tools 中的 Numba 加速函数计算全图坡度，为绘图做准备
    slope_map = plot_tools.calculate_full_slope_map(calculator, map_data.shape)

    # --- 3. 算法执行 ---
    print("\n" + "=" * 30 + " 3. 算法执行 " + "=" * 30)
    bf_path_df, bf_mileage = solve_with_bellman_ford(path_points_list, turn_rules)

    # --- 4. 结果输出与可视化 ---
    print("\n" + "=" * 28 + " 4. 结果与可视化 " + "=" * 28)
    if bf_path_df is not None:
        print("\n--- Bellman-Ford 算法找到的路径 (前5个点) ---")
        print(bf_path_df.head())

        # 保存结果到Excel
        output_filename = 'output/problem3_path_bellman_ford.xlsx'
        bf_path_df.to_excel(output_filename, index=False)
        print(f"\n路径已保存至: '{output_filename}'")

        # 使用 plot_tools 生成高质量的分析图
        print("\n--- 正在调用 plot_tools 生成最终分析图 ---")
        plot_title = f"问题3(Bellman-Ford) P5-P6最优朝向路径(总里程{bf_mileage:.2f}m)"
        plot_tools.plot_slope_analysis_map(
            slope_map=slope_map,
            bad_zones=bad_zones,
            path_df=bf_path_df,
            error_df=None,  # 问题3不涉及错误标记
            output_folder='output',
            map_title=plot_title
        )
    else:
        print("\n未能生成路径，跳过结果保存和绘图。")

