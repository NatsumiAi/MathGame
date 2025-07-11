# 文件名: problem3_astar_solver.py
# 描述: 使用A*算法解决问题3，为给定路径设计最优车头朝向以最小化总里程。

import heapq  # A*算法的核心：优先队列
from collections import defaultdict
from core import data_loader, vehicle_model, config, geo_calculator
from core.plot_tools import plot_slope_analysis_map, calculate_full_slope_map
import time

def solve_problem3_astar():
    """
    主函数，使用A*算法解决问题3并生成可视化结果。
    """
    print("--- 开始使用 A* 算法解决问题3 ---")

    # 1. 加载和初始化
    print("正在加载数据和模型...")
    path_df = data_loader.load_path_data('P5-P6的行驶路径.xlsx')
    turn_rules = vehicle_model.generate_turn_rules()

    HEADINGS = config.DIRECTIONS
    CELL_SIZE = config.CELL_SIZE
    start_index, end_index = 0, len(path_df) - 1

    start_time = time.perf_counter()  # 开始计时
    # --- A* 数据结构 ---
    # open_set: 优先队列，存储 (f_score, g_score, node)。g_score用于在f_score相同时进行比较。
    open_set = []
    # came_from[node]: 记录到达当前节点的最优路径上的前一个节点
    came_from = {}
    # g_score[node]: 从起点到当前节点的实际最小成本
    g_score = defaultdict(lambda: float('inf'))

    # --- 启发函数 h(n) ---
    def heuristic(point_index):
        # 乐观估计：假设所有剩余步骤都是最低成本的直行+无转向
        remaining_steps = end_index - point_index
        return remaining_steps * 1.0 * CELL_SIZE

    # 2. 初始化起点
    print("初始化A*起点...")
    # 车辆在起点时可以有任意朝向，我们将所有可能的起始状态加入open_set
    for h in HEADINGS:
        start_node = (start_index, h)
        g_score[start_node] = 0
        f_score = heuristic(start_index)
        heapq.heappush(open_set, (f_score, 0, start_node))

    final_node = None  # 用于存储到达终点的最佳状态

    # 3. A* 主循环
    print("开始A*搜索...")
    while open_set:
        # 从优先队列中取出 f_score 最小的节点
        current_f, current_g, current_node = heapq.heappop(open_set)
        current_index, current_heading = current_node

        # 如果取出的节点的 g_score 比记录的还大，说明已有一条更优路径到达此节点，跳过
        if current_g > g_score[current_node]:
            continue

        # 如果已到达终点，则搜索成功，结束循环
        if current_index == end_index:
            final_node = current_node
            print("已找到最优路径！")
            break

        # --- 探索邻居 ---
        next_index = current_index + 1
        p_prev = path_df.iloc[current_index]
        p_curr = path_df.iloc[next_index]
        dx = p_curr['x'] - p_prev['x']
        dy = p_curr['y'] - p_prev['y']

        # 根据当前朝向和移动，获取所有合法的下一个朝向
        allowed_next_headings = turn_rules[current_heading].get((dx, dy))
        if not allowed_next_headings:
            continue  # 如果没有合法的下一步，则此路不通

        # 遍历所有合法的邻居状态
        for next_heading in allowed_next_headings:
            neighbor_node = (next_index, next_heading)

            # 计算从当前节点到邻居节点的成本
            d_theta = vehicle_model.calculate_angle_diff(next_heading, current_heading)
            segment_mileage = vehicle_model.calculate_segment_mileage(dx, dy, d_theta)

            # 计算经由当前节点到达邻居节点的 g_score
            tentative_g_score = current_g + segment_mileage

            # 如果发现了到达邻居的更优路径
            if tentative_g_score < g_score[neighbor_node]:
                came_from[neighbor_node] = current_node
                g_score[neighbor_node] = tentative_g_score

                # 计算 f_score 并将邻居节点加入优先队列
                f_score = tentative_g_score + heuristic(next_index)
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor_node))

    # 4. 路径回溯和结果处理
    if not final_node:
        print("错误：A*算法未能找到通往终点的路径！")
        return

    print("正在回溯以构建最优朝向序列...")
    optimal_path = []
    current = final_node
    while current in came_from:
        optimal_path.append(current)
        current = came_from[current]
    optimal_path.append(current)  # 添加起点
    optimal_path.reverse()  # 翻转为正序：从起点到终点

    # 提取最优朝向序列
    optimal_headings = [h for i, h in optimal_path]
    min_total_mileage = g_score[final_node]

    # --- 记录结束时间并计算耗时 ---
    end_time = time.perf_counter()
    duration = end_time - start_time
    print("\n" + "=" * 50)
    print(f"算法总运行时间: {duration:.4f} 秒")
    print("=" * 50 + '\n')

    # 5. 输出和可视化
    path_df['heading'] = optimal_headings
    output_filename = 'output/problem3_A-Star_optimal_headings.xlsx'
    path_df.to_excel(output_filename, index=False)
    print(f"包含最优朝向的完整路径已保存至 '{output_filename}'")

    print("\n--- 问题3 (A* 算法) 最终结果 ---")
    print(f"最小总行驶里程为: {min_total_mileage:.4f} 米")

    heading_lookup = path_df.set_index('id')['heading'].to_dict()
    required_ids = ['L4', 'L17', 'L28', 'L36', 'L45', 'L52', 'L64', 'L70', 'L86', 'L97']
    print("\n请将以下结果填入答题纸的 表2:车头方向：")
    print("=" * 40)
    print(f"{'序号':<5}{'栅格编号':<10}{'无人车车头方向 (度)':<20}")
    print("-" * 40)
    for i, grid_id in enumerate(required_ids, 1):
        heading = heading_lookup.get(grid_id, "未找到")
        print(f"{i:<5}{grid_id:<10}{str(heading):<20}")
    print("=" * 40)

    print("\n--- 开始生成A*结果的可视化地图 ---")
    map_data = data_loader.load_map_data()
    calculator = geo_calculator.GeoCalculator(map_data)
    slope_map_data = calculate_full_slope_map(calculator, map_data.shape)
    map_title = f"问题3 (A*算法): P5-P6 最优朝向路径 (总里程: {min_total_mileage:.2f} m)"

    plot_slope_analysis_map(
        slope_map=slope_map_data,
        bad_zones=None,
        path_df=path_df,
        error_df=None,
        output_folder='output',
        map_title=map_title
    )


if __name__ == '__main__':
    solve_problem3_astar()
