# 文件名: BCVRP_utils/daily_planner.py
# 版本: v1.1 (修复TypeError并优化日志)

import heapq
from collections import deque, defaultdict
import math
from core import config, vehicle_model


class DailyPlanner:
    def __init__(self, initial_config, cost_matrix):
        print("🚀 初始化15天动态规划器 (v1.1 - 修复日志)...")
        self.cost_matrix = cost_matrix
        self.config = initial_config
        self.log = []
        self._initialize_world_state()

    def _initialize_world_state(self):
        # ... (此函数无变化) ...
        self.time = 0.0
        self.events = []
        self._event_counter = 0
        self.sites = {}
        for name, daily_demand in self.config['daily_demands_kg'].items():
            self.sites[name] = {
                'name': name,
                'stock_days': self.config['initial_stock_days'],
                'daily_demand_kg': daily_demand,
                'max_stock_days': self.config['max_stock_days'],
                'min_stock_days': self.config['min_stock_days']
            }
        self.vehicles = {}
        vid_counter = 0
        for depot, num_vehicles in self.config['vehicle_deployment'].items():
            for _ in range(num_vehicles):
                vid = f"V{vid_counter}"
                self.vehicles[vid] = {
                    'id': vid, 'depot': depot, 'location': depot,
                    'status': 'IDLE', 'power': 100.0, 'available_at': 0.0
                }
                vid_counter += 1
        self.depots = {}
        for depot_name in self.config['vehicle_deployment'].keys():
            self.depots[depot_name] = {
                'name': depot_name, 'charging_slots_used': 0,
                'max_charging_slots': config.MAX_CHARGING_VEHICLES_PER_WAREHOUSE
            }
        self.task_queue = deque()

    def _schedule_event(self, delay_hours, event_type, data):
        # ... (此函数无变化) ...
        event_time = self.time + delay_hours
        heapq.heappush(self.events, (event_time, self._event_counter, event_type, data))
        self._event_counter += 1

    # --- [核心修正] ---
    def _log_event(self, message):
        """记录日志，自动计算当前是第几天"""
        # 从全局时间 self.time 计算当前天数 (从1开始)
        current_day = math.floor(self.time / 24) + 1
        if current_day > 15: current_day = 15  # 确保天数不超过15
        self.log.append(f"[Day {current_day:02d} | {self.time:07.2f}h] {message}")

    # --- 修正结束 ---

    def _get_path_cost(self, u, v):
        # ... (此函数无变化) ...
        return self.cost_matrix.get(u, {}).get(v, {'time': 1e9})

    def run_simulation(self, num_days=15):
        for day in range(num_days):
            self._log_event("====== 新的一天开始 ======")
            self._daily_consumption_and_task_generation()
            self._schedule_event(0, 'DISPATCH_CHECK', {})

            next_day_start_time = (day + 1) * 24.0
            while self.events and self.events[0][0] < next_day_start_time:
                event_time, _, event_type, data = heapq.heappop(self.events)
                self.time = event_time

                handler = getattr(self, f"_handle_{event_type.lower()}", None)
                if handler:
                    handler(data)

            self.time = next_day_start_time

        self._log_event("====== 15天保障任务完成 ======")
        return self.log

    def _daily_consumption_and_task_generation(self):
        self._log_event("各阵地消耗物资，更新库存...")
        for site in self.sites.values():
            site['stock_days'] -= 1.0

        tasks_today = []
        for site in self.sites.values():
            if site['stock_days'] < self.config['replenish_trigger_days']:
                needed_days = site['max_stock_days'] - site['stock_days']
                needed_kg = needed_days * site['daily_demand_kg']
                num_trips = math.ceil(needed_kg / config.VEHICLE_PARAMS['A']['max_load'])
                remaining_load = needed_kg
                for i in range(num_trips):
                    load = min(remaining_load, config.VEHICLE_PARAMS['A']['max_load'])
                    tasks_today.append({'site': site['name'], 'load': load, 'priority': site['stock_days']})
                    remaining_load -= load

        tasks_today.sort(key=lambda x: x['priority'])
        self.task_queue.extend(tasks_today)
        self._log_event(f"生成了 {len(tasks_today)} 个运输趟次任务。总任务队列: {len(self.task_queue)}。")

    def _handle_dispatch_check(self, data):
        if not self.task_queue: return
        dispatched_this_round = False
        for task in list(self.task_queue):
            target_site = task['site']
            depot = self.config['site_to_depot_map'][target_site]
            best_vehicle = None
            for vid, v_state in self.vehicles.items():
                if v_state['depot'] == depot and v_state['status'] == 'IDLE' and v_state['available_at'] <= self.time:
                    best_vehicle = v_state
                    break
            if best_vehicle:
                self.task_queue.remove(task)
                vid = best_vehicle['id']
                best_vehicle['status'] = 'DRIVING'
                cost_to_site = self._get_path_cost(depot, target_site)
                travel_time_to_site = cost_to_site['time'] / 3600.0
                self._log_event(
                    f"车辆 {vid} | 状态: 出发 | 仓库: {depot} | 路线: {depot}->{target_site} | 载货: {task['load']:.0f}kg")
                self._schedule_event(travel_time_to_site, 'ARRIVE_AT_SITE',
                                     {'vid': vid, 'site': target_site, 'load': task['load']})
                dispatched_this_round = True
        if self.task_queue and not dispatched_this_round:
            self._schedule_event(0.5, 'DISPATCH_CHECK', {})

    def _handle_arrive_at_site(self, data):
        vid, site_name, load = data['vid'], data['site'], data['load']
        self.vehicles[vid]['location'] = site_name
        self.sites[site_name]['stock_days'] += load / self.sites[site_name]['daily_demand_kg']
        self._log_event(
            f"车辆 {vid} | 状态: 抵达 | 阵地: {site_name} | 送达: {load:.0f}kg | {site_name}库存: {self.sites[site_name]['stock_days']:.2f}天")
        self._schedule_event(0.5, 'FINISH_UNLOAD', data)

    def _handle_finish_unload(self, data):
        vid, site_name = data['vid'], data['site']
        best_return_depot = self.config['site_to_depot_map'][site_name]
        cost_back = self._get_path_cost(site_name, best_return_depot)
        travel_time_back = cost_back['time'] / 3600.0
        self._log_event(f"车辆 {vid} | 状态: 返程 | 从: {site_name} | 路线: {site_name}->{best_return_depot}")
        self._schedule_event(travel_time_back, 'ARRIVE_AT_DEPOT', {'vid': vid, 'depot': best_return_depot})

    def _handle_arrive_at_depot(self, data):
        vid, depot_name = data['vid'], data['depot']
        self.vehicles[vid]['location'] = depot_name
        self._log_event(f"车辆 {vid} | 状态: 抵达 | 仓库: {depot_name} | 任务完成")
        depot_state = self.depots[depot_name]
        if depot_state['charging_slots_used'] < depot_state['max_charging_slots']:
            depot_state['charging_slots_used'] += 1
            self.vehicles[vid]['status'] = 'CHARGING'
            charge_duration = 8.0
            self.vehicles[vid]['available_at'] = self.time + charge_duration
            self._log_event(
                f"车辆 {vid} | 状态: 开始充电 | 仓库: {depot_name} | 持续: {charge_duration}h | 预计结束: {self.vehicles[vid]['available_at']:.2f}h")
            self._schedule_event(charge_duration, 'FINISH_CHARGE', {'vid': vid, 'depot': depot_name})
        else:
            self._log_event(f"车辆 {vid} | 状态: 等待充电 | 仓库: {depot_name} (充电位已满)")
            self._schedule_event(1.0, 'ARRIVE_AT_DEPOT', data)

    def _handle_finish_charge(self, data):
        vid, depot_name = data['vid'], data['depot']
        self.vehicles[vid]['status'] = 'IDLE'
        self.vehicles[vid]['power'] = 100.0
        self.depots[depot_name]['charging_slots_used'] -= 1
        self._log_event(f"车辆 {vid} | 状态: 充电结束 | 仓库: {depot_name} | 电量: 100%")
        self._schedule_event(0, 'DISPATCH_CHECK', {})
