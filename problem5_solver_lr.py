# 文件名: Problems/problem5_solver_lr.py (最终版)

import time
import pandas as pd
from core import config, global_cache
from BCVRP_utils import (
    cost_matrix_builder,
    demand_driven_planner,
    simulation_engine,
    result_analyzer
)
import os


def solve_problem_5_final():
    start_total_time = time.time()
    print("\n" + "=" * 80)
    print("🚀 开始执行问题五最终方案 (基于论文思想)...")
    print("=" * 80)

    # 步骤 1 & 2: 加载数据和构建成本矩阵
    points_df = pd.read_excel('./data/各点位位置信息.xlsx')
    points_info = {row['编号']: {'x': row['栅格x坐标'], 'y': row['栅格y坐标'], 'type': row['类别']} for _, row in
                   points_df.iterrows()}
    # global_cache.load_all_precomputed_data()
    cost_matrix = cost_matrix_builder.build_and_cache_cost_matrix(points_info, objective='time')

    # 步骤 3: 使用需求驱动规划器生成初步方案
    print("\n[步骤 3/5] 执行逆向需求驱动规划...")
    planner = demand_driven_planner.DemandDrivenPlanner(points_info, cost_matrix, config)
    final_schedule = planner.solve()

    if not final_schedule:
        print("规划失败，无法继续。")
        return

    # 步骤 4: 仿真验证
    print("\n[步骤 4/5] 对最终方案进行动态仿真验证...")
    sim = simulation_engine.SimulationEngine(cost_matrix)
    final_result = sim.run(final_schedule)

    # 步骤 5: 结果分析
    print("\n[步骤 5/5] 生成最终分析报告...")
    analyzer = result_analyzer.ResultAnalyzer(final_schedule, final_result, {}, points_info)
    analyzer.analyze()

    # ... (保存日志和结束语) ...
    print("\n[附加步骤] 保存详细仿真日志到文件...")
    with open("output/simulation_full_log.txt", "w", encoding="utf-8") as f:
        for line in final_result.get('log', []): f.write(line + "\n")
    print("✅ 详细日志已保存到: output/simulation_full_log.txt")
    duration = time.time() - start_total_time
    print(f"\n🎉 最终方案求解流程全部完成！总耗时: {duration:.2f} 秒。")


if __name__ == '__main__':
    if not os.path.exists('output'): os.makedirs('output')
    solve_problem_5_final()
