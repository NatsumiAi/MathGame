# 文件名: BCVRP_utils/pathfinder.py (100% 移植 problem4_solver 逻辑版)
# 描述: 直接移植经过验证的高效双向A*算法，确保性能和正确性。

import heapq
from collections import defaultdict
import numpy as np
import math
from core import config, vehicle_model, global_cache


class AStarPathfinder:
    """
    本寻路器直接采用了 problem4_solver.py 中经过验证的 FastAStarSolver 和
    FastBidirectionalAStarSolver 的核心逻辑，以确保最高的性能和稳定性。
    """

    def __init__(self):
        print("🚀 初始化寻路器 (移植自 problem4_solver)...")
        # --- 1. 初始化所有必需的数据 ---
        self.geo_data = global_cache.CACHE['geo_data']
        self.bad_zones = global_cache.CACHE['bad_zones']
        self.slope_map = self.geo_data['slope']
        self.normal_vectors_map = self.geo_data['normals']
        self.rows, self.cols = self.geo_data['rows'], self.geo_data['cols']
        self.max_slope = config.VEHICLE_PARAMS['A']['max_slope']

        # --- 2. 生成与 problem4_solver 兼容的转向规则 ---
        # 正向规则
        original_rules = vehicle_model.generate_turn_rules()
        self.fwd_turn_rules = defaultdict(lambda: defaultdict(list))
        for h_start, moves in original_rules.items():
            for move, h_ends in moves.items():
                self.fwd_turn_rules[h_start][move] = h_ends
        # 反向规则
        self.bwd_turn_rules = defaultdict(lambda: defaultdict(list))
        for h_start, moves in self.fwd_turn_rules.items():
            for move, h_ends in moves.items():
                for h_end in h_ends:
                    rev_move = (-move[0], -move[1])
                    self.bwd_turn_rules[h_end][rev_move].append(h_start)

        # --- 3. 安全惩罚项 ---
        self.SAFETY_PENALTY = 1e9  # 使用一个巨大的惩罚值

    def _get_cost(self, cost_type, p0, p1, h0, h1):
        """完全复制 problem4_solver.py 的成本计算逻辑"""
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dtheta = vehicle_model.calculate_angle_diff(h1, h0)
        seg_len = vehicle_model.calculate_segment_mileage(dx, dy, dtheta)

        r, c = (self.rows - 1) - p1[1], p1[0]
        slope = self.slope_map[r, c]

        speed_kmh = vehicle_model.get_speed_by_slope(slope)
        time_sec = (seg_len / 1000.0) / speed_kmh * 3600.0 if speed_kmh > 0 else float('inf')

        if cost_type == 'time':
            cost = time_sec
        elif cost_type == 'stability':
            pr0, pc0 = (self.rows - 1) - p0[1], p0[0]
            slope0 = self.slope_map[pr0, pc0]
            normal0 = self.normal_vectors_map[pr0, pc0]
            normal1 = self.normal_vectors_map[r, c]
            cost = vehicle_model.calculate_stability_cost(normal0, normal1, slope0, slope)
        else:  # 默认为 time
            cost = time_sec

        if p1 in self.bad_zones:
            cost += self.SAFETY_PENALTY

        return cost

    def _get_heuristic(self, p, goal):
        """完全复制 problem4_solver.py 的启发式函数"""
        dx = abs(p[0] - goal[0]);
        dy = abs(p[1] - goal[1])
        # Octile distance
        return config.CELL_SIZE * (dx + dy) + (config.CELL_SIZE * math.sqrt(2) - 2 * config.CELL_SIZE) * min(dx, dy)

    def _reconstruct_path(self, cf, cb, meet_node):
        """完全复制 problem4_solver.py 的路径重构逻辑"""
        path_fwd = []
        curr = meet_node
        while curr in cf:
            path_fwd.append(curr)
            curr = cf[curr]
        path_fwd.append(curr)
        path_fwd.reverse()

        path_bwd = []
        curr = meet_node
        while curr in cb:
            curr = cb[curr]
            path_bwd.append(curr)

        return path_fwd + path_bwd

    def _evaluate_final_path(self, path_nodes):
        """在得到最终路径后，评估其各项成本"""
        total_costs = defaultdict(float)
        final_path_dicts = []

        start_node = path_nodes[0]
        final_path_dicts.append({'x': start_node[0][0], 'y': start_node[0][1], 'h': start_node[1]})

        for i in range(1, len(path_nodes)):
            p0_node, p1_node = path_nodes[i - 1], path_nodes[i]
            p0, h0 = p0_node
            p1, h1 = p1_node

            # 重新计算所有维度的成本
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            dtheta = vehicle_model.calculate_angle_diff(h1, h0)
            dist_m = vehicle_model.calculate_segment_mileage(dx, dy, dtheta)

            r1, c1 = (self.rows - 1) - p1[1], p1[0]
            slope1 = self.slope_map[r1, c1]
            speed_kmh = vehicle_model.get_speed_by_slope(slope1)
            time_sec = (dist_m / 1000.0) / speed_kmh * 3600.0 if speed_kmh > 0 else 0

            power_cost = vehicle_model.get_power_consumption(slope1) * (dist_m / 1000.0)

            r0, c0 = (self.rows - 1) - p0[1], p0[0]
            slope0 = self.slope_map[r0, c0]
            normal0 = self.normal_vectors_map[r0, c0]
            normal1 = self.normal_vectors_map[r1, c1]
            stability_cost = vehicle_model.calculate_stability_cost(normal0, normal1, slope0, slope0)

            total_costs['distance'] += dist_m
            total_costs['time'] += time_sec
            total_costs['power'] += power_cost
            total_costs['stability'] += stability_cost

            final_path_dicts.append({'x': p1[0], 'y': p1[1], 'h': h1})

        return {'path': final_path_dicts, 'total_costs': dict(total_costs)}

    def find_path(self, start_pos, end_pos, objective='time'):
        """双向A*主函数，完全复制 problem4_solver.py 的逻辑"""
        start_node = (tuple(start_pos), 0)
        goal_node = (tuple(end_pos), 0)  # 假设终点朝向为0，因为A*会自己找到最优的进入朝向

        open_fwd = [(self._get_heuristic(start_pos, end_pos), 0, start_node)]
        open_bwd = [(self._get_heuristic(end_pos, start_pos), 0, goal_node)]

        g_fwd, g_bwd = {start_node: 0}, {goal_node: 0}
        came_from_fwd, came_from_bwd = {}, {}

        mu = float('inf')
        meet_node = None

        while open_fwd and open_bwd:
            # 剪枝条件
            if open_fwd[0][0] + open_bwd[0][0] >= mu:
                break

            # 选择扩展方向
            if len(open_fwd) <= len(open_bwd):
                direction, g_list, cf_list, rules, p_goal = 'fwd', g_fwd, came_from_fwd, self.fwd_turn_rules, end_pos
                _, g_curr, u_node = heapq.heappop(open_fwd)
                g_other = g_bwd
            else:
                direction, g_list, cf_list, rules, p_goal = 'bwd', g_bwd, came_from_bwd, self.bwd_turn_rules, start_pos
                _, g_curr, u_node = heapq.heappop(open_bwd)
                g_other = g_fwd

            if g_curr > g_list.get(u_node, float('inf')):
                continue

            u_pos, u_h = u_node

            # 扩展邻居
            for move, possible_headings in rules[u_h].items():
                v_pos = (u_pos[0] + move[0], u_pos[1] + move[1])

                # 边界和坡度检查
                r_v, c_v = (self.rows - 1) - v_pos[1], v_pos[0]
                if not (0 <= r_v < self.rows and 0 <= c_v < self.cols and self.slope_map[r_v, c_v] <= self.max_slope):
                    continue

                for v_h in possible_headings:
                    v_node = (v_pos, v_h)

                    p_start, p_end = (u_pos, v_pos) if direction == 'fwd' else (v_pos, u_pos)
                    h_start, h_end = (u_h, v_h) if direction == 'fwd' else (v_h, u_h)

                    cost = self._get_cost(objective, p_start, p_end, h_start, h_end)

                    new_g = g_curr + cost
                    if new_g < g_list.get(v_node, float('inf')):
                        g_list[v_node] = new_g
                        cf_list[v_node] = u_node
                        priority = new_g + self._get_heuristic(v_pos, p_goal)
                        heapq.heappush(open_fwd if direction == 'fwd' else open_bwd, (priority, new_g, v_node))

                        if v_node in g_other:
                            potential_mu = new_g + g_other[v_node]
                            if potential_mu < mu:
                                mu = potential_mu
                                meet_node = v_node

        if meet_node is None:
            return None

        # 路径重构与最终评估
        path_nodes = self._reconstruct_path(came_from_fwd, came_from_bwd, meet_node)
        return self._evaluate_final_path(path_nodes)
