# 文件名: problem4_solver.py
# 描述: 终极简化与加速版。完全依赖预处理数据，并修复了所有已知错误。

import time
import heapq
import numpy as np
import pandas as pd
from collections import defaultdict
from core import data_loader, vehicle_model, config, plot_tools

SAFETY_PENALTY = 1e9
GEO_DATA_CACHE = {}


def load_precomputed_data():
    """加载预处理好的地理数据到全局缓存。"""
    global GEO_DATA_CACHE
    if GEO_DATA_CACHE:
        return
    precomputed_file = './data/precomputed_geo_data.npz'
    try:
        print(f"正在加载预处理的地理数据文件: {precomputed_file}...")
        data = np.load(precomputed_file)
        GEO_DATA_CACHE['slope'] = data['slope']
        GEO_DATA_CACHE['normals'] = data['normals']
        GEO_DATA_CACHE['rows'], GEO_DATA_CACHE['cols'] = data['slope'].shape
        print("✅ 预处理地理数据加载成功！")
    except FileNotFoundError:
        print(f"❌ 致命错误: 未找到预处理文件 '{precomputed_file}'。")
        print("💡 请先运行 'preprocess_vectorized.py' 来生成此文件。")
        exit()


def get_geo_info(x, y):
    """从缓存中快速获取地理信息。"""
    rows = GEO_DATA_CACHE['rows']
    r, c = (rows - 1) - int(y), int(x)
    if not (0 <= r < rows and 0 <= c < GEO_DATA_CACHE['cols']):
        return config.VEHICLE_PARAMS['A']['max_slope'] + 1, np.array([0., 0., 1.])
    return GEO_DATA_CACHE['slope'][r, c], GEO_DATA_CACHE['normals'][r, c]


def evaluate_path(path, bad_zones):
    """评估一条给定路径的全部四个指标。"""
    total_mileage, total_time, total_stability, safety_time = 0, 0, 0, 0
    for i in range(1, len(path)):
        p_from, h_from = path[i - 1]
        p_to, h_to = path[i]
        dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
        d_theta = vehicle_model.calculate_angle_diff(h_to, h_from)
        mileage = vehicle_model.calculate_segment_mileage(dx, dy, d_theta)
        slope_to, normal_to = get_geo_info(p_to[0], p_to[1])
        speed_kmh = vehicle_model.get_speed_by_slope(slope_to)
        speed_mps = speed_kmh / 3.6
        time_cost = mileage / speed_mps if speed_mps > 0 else float('inf')
        total_mileage += mileage
        total_time += time_cost
        if p_to in bad_zones:
            safety_time += time_cost
        slope_from, normal_from = get_geo_info(p_from[0], p_from[1])
        total_stability += vehicle_model.calculate_stability_cost(normal_from, normal_to, slope_from, slope_to)
    return {'平稳性': total_stability, '里程(米)': total_mileage, '行驶时长(秒)': total_time, '安全性(秒)': safety_time}


class FastAStarSolver:
    """一个只依赖预处理数据的高速单向A*求解器。"""

    def __init__(self, bad_zones, turn_rules):
        self.bad_zones = bad_zones
        self.turn_rules = turn_rules
        self.max_slope = config.VEHICLE_PARAMS['A']['max_slope']

    def _get_cost(self, cost_type, p_from, p_to, h_from, h_to):
        dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
        d_theta = vehicle_model.calculate_angle_diff(h_to, h_from)
        mileage = vehicle_model.calculate_segment_mileage(dx, dy, d_theta)
        slope_to, normal_to = get_geo_info(p_to[0], p_to[1])
        speed_kmh = vehicle_model.get_speed_by_slope(slope_to)
        speed_mps = speed_kmh / 3.6
        time_cost = mileage / speed_mps if speed_mps > 0 else float('inf')
        cost = 0
        if cost_type == 'stability':
            slope_from, normal_from = get_geo_info(p_from[0], p_from[1])
            cost = vehicle_model.calculate_stability_cost(normal_from, normal_to, slope_from, slope_to)
        elif cost_type == 'time':
            cost = time_cost
        elif cost_type == 'mileage':
            cost = mileage
        if p_to in self.bad_zones:
            cost += SAFETY_PENALTY
        return cost

    def _get_heuristic(self, p_curr, p_goal):
        dx = abs(p_curr[0] - p_goal[0])
        dy = abs(p_curr[1] - p_goal[1])
        return (config.CELL_SIZE * (dx + dy) + (config.CELL_SIZE * np.sqrt(2) - 2 * config.CELL_SIZE) * min(dx, dy))

    def search(self, start_coords, goal_coords, cost_type):
        start_node, goal_coords = (start_coords, 0), tuple(goal_coords)
        h_start = self._get_heuristic(start_coords, goal_coords)
        open_set = [(h_start, 0, start_node)]
        came_from, g_costs = {}, defaultdict(lambda: float('inf'))
        g_costs[start_node] = 0
        node_count = 0
        while open_set:
            node_count += 1
            if node_count % 50000 == 0:
                print(f"  (单向)已探索 {node_count} 个状态节点...")
            f, g, u_node = heapq.heappop(open_set)
            u_coords, u_heading = u_node
            if g > g_costs[u_node]:
                continue
            if u_coords == goal_coords:
                print(f"✅ (单向)找到路径！总共探索了 {node_count} 个状态。")
                return self._reconstruct_path(came_from, u_node)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    v_coords = (u_coords[0] + dx, u_coords[1] + dy)
                    if get_geo_info(v_coords[0], v_coords[1])[0] > self.max_slope:
                        continue
                    allowed_headings = self.turn_rules[u_heading].get((dx, dy))
                    if not allowed_headings:
                        continue
                    for v_heading in allowed_headings:
                        v_node = (v_coords, v_heading)
                        step_cost = self._get_cost(cost_type, u_coords, v_coords, u_heading, v_heading)
                        tentative_g = g_costs[u_node] + step_cost
                        if tentative_g < g_costs[v_node]:
                            came_from[v_node] = u_node
                            g_costs[v_node] = tentative_g
                            h = self._get_heuristic(v_coords, goal_coords)
                            heapq.heappush(open_set, (tentative_g + h, tentative_g, v_node))
        print(f"❌ (单向)未能找到路径！探索了 {node_count} 个节点。")
        return None

    def _reconstruct_path(self, came_from, current_node):
        path = [current_node]
        while current_node in came_from:
            path.append(current_node)
            current_node = came_from[current_node]
        return path[::-1]


class FastBidirectionalAStarSolver:
    def __init__(self, solver: FastAStarSolver):
        self.solver = solver
        self.max_slope = solver.max_slope
        self.turn_rules = solver.turn_rules

    def search(self, start_coords, goal_coords, cost_type):
        start_node, goal_node = (start_coords, 0), (goal_coords, 0)
        open_fwd, g_fwd, came_from_fwd = [(self.solver._get_heuristic(start_coords, goal_coords), 0, start_node)], {
            start_node: 0}, {}
        open_bwd, g_bwd, came_from_bwd = [(self.solver._get_heuristic(goal_coords, start_coords), 0, goal_node)], {
            goal_node: 0}, {}

        # [修复] 正确初始化两个空字典
        closed_fwd, closed_bwd = {}, {}

        mu, meet_node, node_count = float('inf'), None, 0

        while open_fwd and open_bwd:
            node_count += 1
            if node_count % 10000 == 0:
                print(f"  (双向)已探索 {node_count}*2 个状态节点... mu={mu:.2f}")

            _, g_u, u_node = heapq.heappop(open_fwd)
            u_coords, u_heading = u_node

            if g_u > g_fwd.get(u_node, float('inf')):
                continue
            closed_fwd[u_node] = g_u

            if g_u + open_bwd[0][1] >= mu:
                print(f"✅ (双向)找到最优路径！总共探索了 {len(closed_fwd) + len(closed_bwd)} 个节点。")
                return self._reconstruct_path(came_from_fwd, came_from_bwd, meet_node)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    v_coords = (u_coords[0] + dx, u_coords[1] + dy)
                    if get_geo_info(v_coords[0], v_coords[1])[0] > self.max_slope:
                        continue
                    allowed_headings = self.turn_rules[u_heading].get((dx, dy))
                    if not allowed_headings:
                        continue
                    for v_heading in allowed_headings:
                        v_node = (v_coords, v_heading)
                        if v_node in closed_fwd:
                            continue
                        cost = self.solver._get_cost(cost_type, u_coords, v_coords, u_heading, v_heading)
                        if g_u + cost < g_fwd.get(v_node, float('inf')):
                            g_fwd[v_node] = g_u + cost
                            came_from_fwd[v_node] = u_node
                            h = self.solver._get_heuristic(v_coords, goal_coords)
                            heapq.heappush(open_fwd, (g_fwd[v_node] + h, g_fwd[v_node], v_node))
                        if v_node in closed_bwd:
                            path_cost = g_fwd[v_node] + closed_bwd[v_node]
                            if path_cost < mu:
                                mu, meet_node = path_cost, v_node
            # 交换方向
            open_fwd, open_bwd = open_bwd, open_fwd
            g_fwd, g_bwd = g_bwd, g_fwd
            came_from_fwd, came_from_bwd = came_from_bwd, came_from_bwd
            closed_fwd, closed_bwd = closed_bwd, closed_fwd
            start_coords, goal_coords = goal_coords, start_coords

        print(f"❌ (双向)未能找到路径，探索了 {len(closed_fwd) + len(closed_bwd)} 个节点。")
        return None

    def _reconstruct_path(self, came_from_fwd, came_from_bwd, meet_node):
        path_fwd, curr = [], meet_node
        while curr in came_from_fwd:
            path_fwd.append(curr)
            curr = came_from_fwd[curr]
        path_fwd.append(curr)
        path_fwd.reverse()

        path_bwd, curr = [], meet_node
        while curr in came_from_bwd:
            curr = came_from_bwd[curr]
            path_bwd.append(curr)

        return path_fwd + path_bwd


def solve_problem4():
    print("=" * 20 + " 开始求解问题4 (终极优化版) " + "=" * 20)
    start_all = time.perf_counter()

    load_precomputed_data()
    bad_zones = data_loader.load_bad_zones()
    turn_rules = vehicle_model.generate_turn_rules()
    locations_df = pd.read_excel('./data/各点位位置信息.xlsx')
    locations = locations_df.set_index('编号').to_dict('index')

    uni_solver = FastAStarSolver(bad_zones, turn_rules)
    bi_solver = FastBidirectionalAStarSolver(uni_solver)

    tasks = [
        {'start': 'C6', 'goal': 'Z5', 'objective': 'stability'},
        {'start': 'C3', 'goal': 'Z4', 'objective': 'time'},
        {'start': 'C5', 'goal': 'Z7', 'objective': 'mileage'},
    ]
    results_table = []

    for task in tasks:
        if task['objective'] == 'stability':
            print(f"\n--- [双向A*] 开始任务: {task['start']} -> {task['goal']} (目标: {task['objective']}) ---")
            solver_to_use = bi_solver
        else:
            print(f"\n--- [单向A*] 开始任务: {task['start']} -> {task['goal']} (目标: {task['objective']}) ---")
            solver_to_use = uni_solver

        start_coords = (locations[task['start']]['栅格x坐标'], locations[task['start']]['栅格y坐标'])
        goal_coords = (locations[task['goal']]['栅格x坐标'], locations[task['goal']]['栅格y坐标'])

        start_task_time = time.perf_counter()
        path = solver_to_use.search(start_coords, goal_coords, task['objective'])
        print(f"任务耗时: {time.perf_counter() - start_task_time:.2f} 秒。")

        if path:
            metrics = evaluate_path(path, bad_zones)
            metrics['路径起止点'] = f"{task['start']}-{task['goal']}"
            results_table.append(metrics)

            path_data = [{'编号': f"L{i}", '栅格x坐标': c[0], '栅格y坐标': c[1], '车头朝向': h} for i, (c, h) in
                         enumerate(path)]
            path_df = pd.DataFrame(path_data)
            outfile_map = {'stability': '附件8：C6-Z5平稳性最优路径', 'time': '附件9：C3-Z4时效性最优路径',
                           'mileage': '附件10：C5-Z7路程最短路径'}
            output_path = f"output/{outfile_map[task['objective']]}.xlsx"
            path_df.to_excel(output_path, index=False)
            print(f"路径已保存至: {output_path}")

    print("\n\n" + "=" * 25 + " 问题4 最终路径评估结果 " + "=" * 25)
    if results_table:
        df_results = pd.DataFrame(results_table)
        df_results = df_results[['路径起止点', '平稳性', '里程(米)', '行驶时长(秒)', '安全性(秒)']]
        print(df_results.to_string(index=False))
    print(f"\n问题4总耗时: {(time.perf_counter() - start_all) / 60:.2f} 分钟")


if __name__ == '__main__':
    solve_problem4()

