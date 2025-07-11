# 文件名: verify_optimality.py
# 描述: 基于正确的 problem4_solver.py，仅为可视化添加必要的返回值支持

import time
import heapq
import numpy as np
import pandas as pd
from collections import defaultdict
from core import data_loader, vehicle_model, config
import matplotlib.pyplot as plt

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['DengXian', 'SimHei', 'Microsoft YaHei', 'Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False

# --- 全局常量和缓存 ---
SAFETY_PENALTY = 1e9
GEO_DATA_CACHE = {}


def load_precomputed_data():
    """加载预处理好的地理数据到全局缓存"""
    global GEO_DATA_CACHE
    if GEO_DATA_CACHE:
        return
    precomputed_file = './data/precomputed_geo_data.npz'
    try:
        print(f"加载预处理地理数据: {precomputed_file}")
        data = np.load(precomputed_file)
        GEO_DATA_CACHE['slope'] = data['slope']
        GEO_DATA_CACHE['normals'] = data['normals']
        GEO_DATA_CACHE['rows'], GEO_DATA_CACHE['cols'] = data['slope'].shape
    except FileNotFoundError:
        print(f"未找到 '{precomputed_file}'，请先运行 preprocess_vectorized.py")
        exit()


def get_geo_info(x, y):
    """从缓存中快速获取坡度和法向量"""
    rows = GEO_DATA_CACHE['rows']
    r, c = (rows - 1) - int(y), int(x)
    if not (0 <= r < rows and 0 <= c < GEO_DATA_CACHE['cols']):
        return config.VEHICLE_PARAMS['A']['max_slope'] + 1, np.array([0., 0., 1.])
    return GEO_DATA_CACHE['slope'][r, c], GEO_DATA_CACHE['normals'][r, c]


def evaluate_path(path, bad_zones):
    """评估路径：平稳性、行程、时间、安全性"""
    total_mileage = total_time = total_stability = safety_time = 0
    for i in range(1, len(path)):
        (x0, y0), h0 = path[i - 1]
        (x1, y1), h1 = path[i]
        dx, dy = x1 - x0, y1 - y0
        dtheta = vehicle_model.calculate_angle_diff(h1, h0)
        segment_len = vehicle_model.calculate_segment_mileage(dx, dy, dtheta)
        slope1, normal1 = get_geo_info(x1, y1)
        speed_mps = vehicle_model.get_speed_by_slope(slope1) / 3.6
        t = segment_len / speed_mps if speed_mps > 0 else float('inf')

        total_mileage += segment_len
        total_time += t
        if (x1, y1) in bad_zones:
            safety_time += t

        slope0, normal0 = get_geo_info(x0, y0)
        total_stability += vehicle_model.calculate_stability_cost(
            normal0, normal1, slope0, slope1)

    return {
        '平稳性': total_stability,
        '里程(米)': total_mileage,
        '行驶时长(秒)': total_time,
        '安全性(秒)': safety_time
    }


class FastAStarSolver:
    """单向 A* 求解器"""

    def __init__(self, bad_zones, turn_rules):
        self.bad_zones = bad_zones
        self.turn_rules = turn_rules
        self.max_slope = config.VEHICLE_PARAMS['A']['max_slope']

    def _get_cost(self, cost_type, p0, p1, h0, h1):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dtheta = vehicle_model.calculate_angle_diff(h1, h0)
        seg_len = vehicle_model.calculate_segment_mileage(dx, dy, dtheta)
        slope, normal = get_geo_info(p1[0], p1[1])
        speed_mps = vehicle_model.get_speed_by_slope(slope) / 3.6
        t = seg_len / speed_mps if speed_mps > 0 else float('inf')
        if cost_type == 'stability':
            slope0, normal0 = get_geo_info(p0[0], p0[1])
            c = vehicle_model.calculate_stability_cost(normal0, normal, slope0, slope)
        elif cost_type == 'time':
            c = t
        else:  # mileage
            c = seg_len
        if p1 in self.bad_zones:
            c += SAFETY_PENALTY
        return c

    def _get_heuristic(self, p, goal):
        dx = abs(p[0] - goal[0]); dy = abs(p[1] - goal[1])
        c1 = config.CELL_SIZE * (dx + dy)
        c2 = (config.CELL_SIZE * np.sqrt(2) - 2 * config.CELL_SIZE) * min(dx, dy)
        return c1 + c2

    def search(self, start, goal, cost_type):
        start_node = (start, 0)
        goal = tuple(goal)
        open_set = [(self._get_heuristic(start, goal), 0, start_node)]
        came_from = {}
        g_cost = defaultdict(lambda: float('inf'))
        g_cost[start_node] = 0
        count = 0
        while open_set:
            count += 1
            _, g0, u = heapq.heappop(open_set)
            if g0 > g_cost[u]:
                continue
            ux, uy = u[0]
            if (ux, uy) == goal:
                print(f"✅ (单向)找到路径，节点数={count}")
                return self._reconstruct_path(came_from, u), g_cost, None

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == dy == 0: continue
                    v = (ux+dx, uy+dy)
                    slope_v, _ = get_geo_info(v[0], v[1])
                    if slope_v > self.max_slope: continue
                    for vh in self.turn_rules[u[1]].get((dx, dy), []):
                        vn = (v, vh)
                        c = self._get_cost(cost_type, u[0], v, u[1], vh)
                        ng = g_cost[u] + c
                        if ng < g_cost[vn]:
                            g_cost[vn] = ng
                            came_from[vn] = u
                            f = ng + self._get_heuristic(v, goal)
                            heapq.heappush(open_set, (f, ng, vn))

        print(f"❌ (单向)未找到路径，节点数={count}")
        return None, g_cost, None

    def _reconstruct_path(self, came_from, node):
        path = [node]
        while node in came_from:
            node = came_from[node]
            path.append(node)
        return path[::-1]


class FastBidirectionalAStarSolver:
    """双向 A* 求解器"""

    def __init__(self, solver: FastAStarSolver):
        self.solver = solver
        self.max_slope = solver.max_slope
        self.turn_rules = solver.turn_rules

    def search(self, start, goal, cost_type):
        start_node = (start, 0)
        goal_node = (tuple(goal), 0)
        open_fwd = [(self.solver._get_heuristic(start, goal), 0, start_node)]
        open_bwd = [(self.solver._get_heuristic(goal, start), 0, goal_node)]
        g_fwd = {start_node: 0}
        g_bwd = {goal_node: 0}
        cf = {}
        cb = {}
        closed_fwd = {}
        closed_bwd = {}
        mu = float('inf')
        meet_node = None
        count = 0

        while open_fwd and open_bwd:
            count += 1
            # 交替展开
            f_list, g_list, cf_list, closed_list = \
                (open_fwd, g_fwd, cf, closed_fwd) if len(open_fwd) <= len(open_bwd) else \
                (open_bwd, g_bwd, cb, closed_bwd)
            h_other = open_bwd[0][1] if f_list is open_fwd else open_fwd[0][1]

            _, gu, un = heapq.heappop(f_list)
            if gu > g_list.get(un, float('inf')):
                continue
            closed_list[un] = gu

            # 检查剪枝条件
            if gu + h_other >= mu:
                print(f"✅ (双向)找到最优路径，已探索 {count*2} 次扩展，μ={mu:.2f}")
                break

            ux, uy = un[0]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == dy == 0: continue
                    v = (ux+dx, uy+dy)
                    slope_v, _ = get_geo_info(v[0], v[1])
                    if slope_v > self.max_slope: continue
                    for vh in self.turn_rules[un[1]].get((dx, dy), []):
                        vn = (v, vh)
                        cost = self.solver._get_cost(cost_type, un[0], v, un[1], vh)
                        if un in g_fwd and vn in g_bwd:
                            # 双向相遇
                            cand = g_fwd[un] + cost + g_bwd[vn]
                            if cand < mu:
                                mu = cand
                                meet_node = vn if f_list is open_fwd else un
                        # 松弛
                        got = gu + cost
                        if got < g_list.get(vn, float('inf')):
                            g_list[vn] = got
                            cf_list[vn] = un
                            heapq.heappush(f_list,
                                           (got + self.solver._get_heuristic(v, goal if f_list is open_fwd else start),
                                            got, vn))

        # 重建路径
        path = None
        if meet_node is not None:
            path = self._reconstruct_path(cf, cb, meet_node)
        print(f"总共探索节点: 前向={len(g_fwd)}, 后向={len(g_bwd)}, μ={mu:.2f}")
        return path, g_fwd, g_bwd

    def _reconstruct_path(self, cf, cb, meet_node):
        # 从 meet_node 向两端回溯
        left, curr = [], meet_node
        while curr in cf:
            left.append(curr)
            curr = cf[curr]
        left.append(curr)
        left.reverse()

        right = []
        curr = meet_node
        while curr in cb:
            curr = cb[curr]
            right.append(curr)
        return left + right


def visualize_bidirectional_search_simple(
        g_fwd, g_bwd, path,
        start_coords, goal_coords,
        meet_states, task):
    """用散点高亮前/后向探索，以及真正的碰头状态。"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 前向探索
    x0 = [s[0][0] for s in g_fwd.keys()]
    y0 = [s[0][1] for s in g_fwd.keys()]
    ax.scatter(x0, y0, c='C0', s=5, alpha=0.4, label='前向探索')

    # 后向探索
    x1 = [s[0][0] for s in g_bwd.keys()]
    y1 = [s[0][1] for s in g_bwd.keys()]
    ax.scatter(x1, y1, c='C3', s=5, alpha=0.4, label='后向探索')

    # 最优路径
    if path:
        pts = np.array([p[0] for p in path])
        ax.plot(pts[:, 0], pts[:, 1], c='cyan', lw=2.5, label='最优路径')

    # 起点/终点
    ax.scatter(start_coords[0], start_coords[1],
               c='lime', edgecolor='k', s=150, zorder=10, label='起点')
    ax.scatter(goal_coords[0], goal_coords[1],
               c='magenta', marker='s', edgecolor='k', s=150, zorder=10, label='终点')

    # 真正的碰头状态
    for coords, heading in meet_states:
        ax.scatter(coords[0], coords[1],
                   c='yellow', marker='X', s=200,
                   edgecolor='black', linewidth=2, zorder=15,
                   label=f'碰头 {coords},{heading}')

    ax.set_title(f'双向A* 探索范围及路径 ({task["start"]}-{task["goal"]})')
    ax.set_xlabel('栅格 x 坐标'); ax.set_ylabel('栅格 y 坐标')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.subplots_adjust(
        top=0.96,  # 图顶到 figure 顶的距离
        bottom=0.08,  # 图底到 figure 底的距离
        left=0.06,  # 图左…
        right=0.98  # 图右…
    )
    return fig


def solve_problem4():
    print("=" * 20, "开始求解问题4", "=" * 20)
    start_all = time.perf_counter()

    load_precomputed_data()
    bad_zones = data_loader.load_bad_zones()
    turn_rules = vehicle_model.generate_turn_rules()
    loc_df = pd.read_excel('./data/各点位位置信息.xlsx')
    locs = loc_df.set_index('编号').to_dict('index')

    uni = FastAStarSolver(bad_zones, turn_rules)
    bi = FastBidirectionalAStarSolver(uni)

    tasks = [
        {'start': 'C6', 'goal': 'Z5', 'objective': 'stability'},
        {'start': 'C3', 'goal': 'Z4', 'objective': 'time'},
        {'start': 'C5', 'goal': 'Z7', 'objective': 'mileage'},
    ]
    results = []

    for t in tasks:
        s = (locs[t['start']]['栅格x坐标'], locs[t['start']]['栅格y坐标'])
        g = (locs[t['goal']]['栅格x坐标'], locs[t['goal']]['栅格y坐标'])
        print(f"\n>> 任务 {t['start']}->{t['goal']} (目标={t['objective']})")
        tic = time.perf_counter()
        if t['objective'] == 'stability':
            path, g_fwd, g_bwd = bi.search(s, g, t['objective'])
            meet_states = set(g_fwd.keys()) & set(g_bwd.keys())
            print(">>> 碰头状态数:", len(meet_states))
        else:
            path, g_fwd, _ = uni.search(s, g, t['objective'])
            g_bwd = {}
            meet_states = set()

        print(f"任务耗时: {time.perf_counter() - tic:.2f}s")
        if not path:
            continue

        # 评估并存表
        m = evaluate_path(path, bad_zones)
        m['路径'] = f"{t['start']}-{t['goal']}"
        results.append(m)

        # 保存路径明细
        df_path = pd.DataFrame([
            {'编号': f'L{i}', 'x': p[0][0], 'y': p[0][1], 'heading': p[1]}
            for i, p in enumerate(path)
        ])
        fn_map = {
            'stability': '附件8：C6-Z5平稳性最优路径',
            'time':      '附件9：C3-Z4时效性最优路径',
            'mileage':   '附件10：C5-Z7路程最短路径',
        }
        df_path.to_excel(f'output/{fn_map[t["objective"]]}.xlsx', index=False)

        # 只对 stability 画散点图
        if t['objective'] == 'stability':
            fig = visualize_bidirectional_search_simple(
                g_fwd, g_bwd, path,
                s, g,
                meet_states, t
            )
            fig.savefig(f'output/双向散点_{t["start"]}-{t["goal"]}.png', dpi=300)
            plt.show()

    # 打印汇总
    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res[['路径','平稳性','里程(米)','行驶时长(秒)','安全性(秒)']]
        print("\n最终评估：\n", df_res.to_string(index=False))

    print(f"\n总耗时 {(time.perf_counter() - start_all)/60:.2f} 分钟")


if __name__ == '__main__':
    solve_problem4()
