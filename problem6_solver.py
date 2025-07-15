# 文件名: Problems/problem6_solver.py
# 版本: v1.1 (更新初始配置)

import pandas as pd
import os
import time
from core import config, global_cache
from BCVRP_utils import cost_matrix_builder, daily_planner


def solve_problem_6():
    start_total_time = time.time()
    print("\n" + "=" * 80)
    print("🚀 开始执行问题六：15天连续物资保障方案...")
    print("=" * 80)

    # --- 步骤 1: 加载基础数据和成本矩阵 ---
    print("[步骤 1/3] 加载数据和预计算的成本矩阵...")
    points_df = pd.read_excel('./data/各点位位置信息.xlsx')
    points_info = {row['编号']: {'x': row['栅格x坐标'], 'y': row['栅格y坐标'], 'type': row['类别']} for _, row in
                   points_df.iterrows()}
    cost_matrix = cost_matrix_builder.build_and_cache_cost_matrix(points_info, objective='time')

    # --- 步骤 2: 定义15天保障任务的初始配置 ---
    print("[步骤 2/3] 设置15天保障任务的初始配置...")

    # --- [核心修正] 使用新的、更优的分配方案 ---
    site_to_depot_map = {
        'Z1': 'C2', 'Z2': 'C2',
        'Z6': 'C3',
        'Z3': 'C6', 'Z4': 'C6', 'Z5': 'C6',
        'Z7': 'C9', 'Z8': 'C9', 'Z9': 'C9'
    }
    vehicle_deployment = {
        'C2': 2,
        'C3': 1,
        'C6': 2,
        'C9': 2
    }
    # --- 修正结束 ---

    daily_demands_kg = {
        'Z1': 510, 'Z2': 450, 'Z3': 285, 'Z4': 450, 'Z5': 675,
        'Z6': 450, 'Z7': 285, 'Z8': 675, 'Z9': 450
    }

    initial_config = {
        'site_to_depot_map': site_to_depot_map,
        'vehicle_deployment': vehicle_deployment,
        'daily_demands_kg': daily_demands_kg,
        'initial_stock_days': 2.0,
        'max_stock_days': 3.5,
        'min_stock_days': 0.5,
        'replenish_trigger_days': 1.5
    }

    # --- 步骤 3: 运行15天动态模拟 ---
    print("[步骤 3/3] 开始运行15天动态模拟...")
    planner = daily_planner.DailyPlanner(initial_config, cost_matrix)
    full_log = planner.run_simulation(num_days=15)

    # --- 结果输出 ---
    print("\n" + "=" * 80)
    print("✅ 15天模拟完成！正在保存运行日志...")
    log_path = "output/q6_daily_run_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("      问题六：15天连续保障运行日志\n")
        f.write("=" * 40 + "\n\n")
        for line in full_log:
            f.write(line + "\n")
    print(f"✅ 详细运行日志已保存到: {log_path}")
    duration = time.time() - start_total_time
    print(f"\n🎉 问题六求解流程全部完成！总耗时: {duration:.2f} 秒。")


if __name__ == '__main__':
    if not os.path.exists('output'):
        os.makedirs('output')
    solve_problem_6()
