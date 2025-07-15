# 文件名: BCVRP_utils/cost_matrix_builder.py
# 描述: 构建并缓存点对点之间的成本矩阵。
# 版本: v2.1 (在v2.0基础上修复了数据加载和节点筛选问题)

import os
import pickle
import pandas as pd
from collections import defaultdict
from . import pathfinder
from tqdm import tqdm
from core import global_cache # 导入全局缓存

def build_and_cache_cost_matrix(points_info, objective='time', force_rebuild=False):
    """
    构建VRP运营层所需的全连接成本矩阵。
    如果已存在缓存则直接加载。
    """
    cache_file = f'./data/precomputed_path_costs_{objective}.pkl'

    if os.path.exists(cache_file) and not force_rebuild:
        print(f"-> 发现已缓存的VRP成本矩阵，正在加载: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"-> 未发现缓存或强制重建，开始构建VRP成本矩阵 (目标: {objective})...")

    # --- 核心修复 1: 确保底层地理数据已加载 ---
    # 这一行是解决 'NoneType' object is not subscriptable 错误的关键
    if 'geo_data' not in global_cache.CACHE:
        global_cache.load_all_precomputed_data()

    finder = pathfinder.AStarPathfinder()
    cost_matrix = defaultdict(dict)

    # --- 核心修复 2: 修正节点筛选逻辑，包含所有可能的点 ---
    # 这是解决下游 KeyError 的关键
    vrp_nodes = list(points_info.keys())
    print(f"-> VRP节点已确定 (所有点位)，共 {len(vrp_nodes)} 个。")

    tasks = []
    for p1 in vrp_nodes:
        for p2 in vrp_nodes:
            if p1 == p2:
                continue
            tasks.append((p1, p2))

    print(f"-> 将为VRP节点计算 {len(tasks)} 条路径...")

    LARGE_COST = 1e9

    for p1, p2 in tqdm(tasks, desc="VRP路径成本计算进度"):
        start_pos = (points_info[p1]['x'], points_info[p1]['y'])
        end_pos = (points_info[p2]['x'], points_info[p2]['y'])

        try:
            result = finder.find_path(start_pos, end_pos, objective)
            if result:
                cost_matrix[p1][p2] = result['total_costs']
            else:
                cost_matrix[p1][p2] = {'time': LARGE_COST, 'distance': LARGE_COST, 'power': LARGE_COST, 'stability': LARGE_COST}
        except Exception as e:
            print(f"\n[错误] 计算路径 {p1} -> {p2} 时发生异常: {e}")
            cost_matrix[p1][p2] = {'time': LARGE_COST, 'distance': LARGE_COST, 'power': LARGE_COST, 'stability': LARGE_COST}

    final_matrix = dict(cost_matrix)
    with open(cache_file, 'wb') as f:
        pickle.dump(final_matrix, f)
    print(f"✅ VRP成本矩阵已构建并缓存到: {cache_file}")

    return final_matrix
