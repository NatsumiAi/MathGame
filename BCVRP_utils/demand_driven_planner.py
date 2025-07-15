# 文件名: BCVRP_utils/demand_driven_planner.py
# 版本: v5.3 (基于完整往返成本进行规划)

from collections import defaultdict
import math


class DemandDrivenPlanner:
    def __init__(self, points_info, cost_matrix, config):
        print("🚀 初始化规划器 (v5.3 - 基于完整往返成本)...")
        self.points_info = points_info
        self.cost_matrix = cost_matrix
        self.config = config
        self.UNREACHABLE_THRESHOLD = 1e8
        self._prepare_data()

    def _prepare_data(self):
        SITE_TYPE_NAME = '前沿阵地'
        WAREHOUSE_TYPE_NAME = '中转仓库候选点'
        self.sites = {p: info for p, info in self.points_info.items() if info['type'] == SITE_TYPE_NAME}
        demands_1day = {
            'Z1': 510, 'Z2': 450, 'Z3': 285, 'Z4': 450, 'Z5': 675,
            'Z6': 450, 'Z7': 285, 'Z8': 675, 'Z9': 450
        }
        # 问题六要求15天，这里我们先按问题五的1.5天需求来确定最优仓库
        self.total_demands = {name: d * 1.5 for name, d in demands_1day.items()}
        self.warehouses = {p: info for p, info in self.points_info.items() if info['type'] == WAREHOUSE_TYPE_NAME}
        if not self.warehouses: raise ValueError("没有可用的仓库来满足需求。")
        self.vehicle_capacity = self.config.VEHICLE_PARAMS['A']['max_load']

    def _get_path_cost(self, u, v):
        cost = self.cost_matrix.get(u, {}).get(v, {})
        time_sec = cost.get('time', float('inf'))
        power = cost.get('power', float('inf'))

        if time_sec > self.UNREACHABLE_THRESHOLD:
            return {'time': float('inf'), 'power': float('inf')}
        return {'time': time_sec / 3600.0, 'power': power}

    def solve(self):
        print("\n-> 步骤1: 宏观规划 - 指派最佳补给仓库 (基于完整往返成本)...")
        site_to_depot_map = {}

        # --- [核心修正] ---
        for site in self.sites:
            candidate_costs = []
            for depot in self.warehouses:
                # 计算从该仓库出发的完整往返成本
                cost_to_site = self._get_path_cost(depot, site)
                cost_back_to_depot = self._get_path_cost(site, depot)

                # 检查电量是否足够完成一个往返
                total_power_needed = cost_to_site['power'] + cost_back_to_depot['power']
                if total_power_needed > 100:
                    total_round_trip_time = float('inf')
                else:
                    # 综合成本 = 去程时间 + 卸货时间(0.5h) + 返程时间
                    total_round_trip_time = cost_to_site['time'] + 0.5 + cost_back_to_depot['time']

                candidate_costs.append((depot, total_round_trip_time))

            if not candidate_costs:
                print(f"❌ 致命警告: 阵地 {site} 无法规划任何可行往返路径！")
                continue

            # 找到往返时间最短的仓库
            best_depot, min_cost = min(candidate_costs, key=lambda x: x[1])

            if min_cost == float('inf'):
                print(f"❌ 致命警告: 阵地 {site} 虽然可达，但无法从任何仓库安全往返！")
                continue

            site_to_depot_map[site] = best_depot
            print(f"  - 阵地 {site} -> 由仓库 {best_depot} 负责 (单趟往返总耗时: {min_cost:.2f}h)")
        # --- 修正结束 ---

        print("\n-> 步骤2: 中观规划 - 生成所有必需的运输趟次...")
        tasks_by_depot = defaultdict(list)
        for site, total_demand in self.total_demands.items():
            if site not in site_to_depot_map: continue
            depot = site_to_depot_map[site]
            num_deliveries = math.ceil(total_demand / self.vehicle_capacity)
            for _ in range(num_deliveries):
                tasks_by_depot[depot].append(site)

        print("\n-> 步骤3: 微观规划 - 组织行程链并分配车辆...")
        final_schedule = {}
        total_tasks = sum(len(sites) for sites in tasks_by_depot.values())
        if total_tasks == 0:
            print("警告：没有生成任何运输任务。")
            return {}

        # 动态计算各仓库所需车辆数
        # 假设总车辆数仍为7，按任务比例分配
        TOTAL_VEHICLES = 7
        assigned_vehicles = 0
        depot_list = sorted(tasks_by_depot.keys())  # 排序以保证结果一致性

        for i, depot in enumerate(depot_list):
            sites_to_visit = tasks_by_depot[depot]
            if not sites_to_visit: continue

            if i == len(depot_list) - 1:  # 最后一个仓库获得剩余所有车辆
                num_vehicles_for_depot = TOTAL_VEHICLES - assigned_vehicles
            else:
                num_vehicles_for_depot = max(1, round(TOTAL_VEHICLES * len(sites_to_visit) / total_tasks))
                assigned_vehicles += num_vehicles_for_depot

            print(f"  - 仓库 {depot}: 分配到 {len(sites_to_visit)} 趟任务, 部署 {num_vehicles_for_depot} 辆车。")
            final_schedule[depot] = {'vehicles': num_vehicles_for_depot, 'trips': []}

            # 将任务均分给车辆
            chunk_size = math.ceil(len(sites_to_visit) / num_vehicles_for_depot)
            for i in range(0, len(sites_to_visit), chunk_size):
                trip_chunk = sites_to_visit[i:i + chunk_size]
                final_schedule[depot]['trips'].append({'sites': trip_chunk, 'load': self.vehicle_capacity})

        return final_schedule
