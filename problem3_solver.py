# 文件名: problem3_solver.py (最终版 - 带可视化功能)
# 描述: 使用动态规划解决问题3，并调用绘图工具将结果可视化。
import cProfile
from core import data_loader, vehicle_model, geo_calculator
from core.plot_tools import plot_slope_analysis_map, calculate_full_slope_map  # 导入绘图工具
import time

def solve_problem3_and_visualize():
    print("正在加载路径文件和车辆模型...")
    path_df = data_loader.load_path_data('P5-P6的行驶路径.xlsx')
    turn_rules = vehicle_model.generate_turn_rules()
    HEADINGS = [0, 45, 90, 135, 180, 225, 270, 315]
    num_points = len(path_df)

    start_time = time.perf_counter() #开始计时

    dp_table = [{} for _ in range(num_points)]
    path_table = [{} for _ in range(num_points)]

    for h in HEADINGS:
        dp_table[0][h] = 0.0
        path_table[0][h] = None

    for i in range(1, num_points):
        p_prev = path_df.iloc[i - 1];
        p_curr = path_df.iloc[i]
        dx = p_curr['x'] - p_prev['x'];
        dy = p_curr['y'] - p_prev['y']
        for h_curr in HEADINGS:
            min_mileage_for_h_curr = float('inf');
            best_prev_h = -1
            for h_prev in HEADINGS:
                if h_prev not in dp_table[i - 1]: continue
                allowed_next_headings = turn_rules[h_prev].get((dx, dy))
                if allowed_next_headings is None or h_curr not in allowed_next_headings: continue
                d_theta = vehicle_model.calculate_angle_diff(h_curr, h_prev)
                segment_mileage = vehicle_model.calculate_segment_mileage(dx, dy, d_theta)
                total_mileage = dp_table[i - 1][h_prev] + segment_mileage
                if total_mileage < min_mileage_for_h_curr:
                    min_mileage_for_h_curr = total_mileage
                    best_prev_h = h_prev
            if best_prev_h != -1:
                dp_table[i][h_curr] = min_mileage_for_h_curr
                path_table[i][h_curr] = best_prev_h
        if i % 10 == 0 or i == num_points - 1:
            print(f"  已处理 {i}/{num_points - 1} 个路径点...")
    print("动态规划计算完成！")

    print("正在回溯以构建最优朝向序列...")
    optimal_headings = [-1] * num_points
    last_point_dp = dp_table[-1]
    if not last_point_dp:
        print("错误：无法找到到达终点的任何有效路径！");
        return
    min_total_mileage = min(last_point_dp.values())
    final_heading = min(h for h, m in last_point_dp.items() if m == min_total_mileage)
    optimal_headings[-1] = final_heading
    for i in range(num_points - 1, 0, -1):
        optimal_headings[i - 1] = path_table[i][optimal_headings[i]]
    print("最优朝向序列已构建。")

    # --- 记录结束时间并计算耗时 ---
    end_time = time.perf_counter()
    duration = end_time - start_time
    print("\n" + "=" * 50)
    print(f"算法总运行时间: {duration:.4f} 秒")
    print("=" * 50 + '\n')

    path_df['heading'] = optimal_headings

    output_filename = 'output/problem3_P5-P6_with_optimal_headings.xlsx'
    path_df.to_excel(output_filename, index=False)
    print(f"包含最优朝向的完整路径已保存至 '{output_filename}'")

    print("\n--- 问题3 最终结果 ---")
    print(f"最小总行驶里程为: {min_total_mileage:.4f} 米")

    heading_lookup = path_df.set_index('id')['heading'].to_dict()
    required_ids = ['L4', 'L17', 'L28', 'L36', 'L45', 'L52', 'L64', 'L70', 'L86', 'L97']
    print("=" * 40)
    print(f"{'序号':<5}{'栅格编号':<10}{'无人车车头方向 (度)':<20}")
    print("-" * 40)
    for i, grid_id in enumerate(required_ids, 1):
        heading = heading_lookup.get(grid_id, "未找到")
        print(f"{i:<5}{grid_id:<10}{str(heading):<20}")
    print("=" * 40)

    print("\n--- 开始生成问题3结果的可视化地图 ---")
    map_data = data_loader.load_map_data()
    calculator = geo_calculator.GeoCalculator(map_data)
    slope_map_data = calculate_full_slope_map(calculator, map_data.shape)

    map_title = f"问题3: P5-P6 最优朝向路径规划 (总里程: {min_total_mileage:.2f} m)"

    # c. 调用绘图函数
    #    - slope_map: 坡度图
    #    - bad_zones: 本问不关心，设为None
    #    - path_df: 我们刚刚计算出的、包含'heading'列的DataFrame
    #    - error_df: 本问没有错误分析，设为None
    plot_slope_analysis_map(
        slope_map=slope_map_data,
        bad_zones=None,
        path_df=path_df,
        error_df=None,
        output_folder='output',
        map_title=map_title
    )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


if __name__ == '__main__':
    solve_problem3_and_visualize()
