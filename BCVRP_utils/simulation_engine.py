# 文件名: BCVRP_utils/simulation_engine.py
# 版本: v3.5 (完整最终版)

import heapq
from collections import deque
from core import vehicle_model, config


class SimulationEngine:
    def __init__(self, cost_matrix):
        print("⚡️ 初始化仿真引擎 (v3.5 - 完整最终版)...")
        self.cost_matrix = cost_matrix
        self.events = []
        self.time = 0
        self.log = []
        self._event_counter = 0
        self.UNREACHABLE_THRESHOLD = 1e8
        self.all_warehouses = [p for p in cost_matrix.keys() if p.startswith('C')]

        self.last_delivery_time = 0
        self.total_distance_km = 0
        self.total_stability_cost = 0
        self.max_stability_on_any_leg = 0

    def _safe_cost(self, u, v):
        return self.cost_matrix.get(u, {}).get(v, {
            'time': 1e9, 'power': 1e9, 'distance': 1e9, 'stability': 1e9
        })

    def _schedule_event(self, delay_hours, event_type, data):
        event_time = self.time + delay_hours
        heapq.heappush(self.events, (event_time, self._event_counter, event_type, data))
        self._event_counter += 1

    def run(self, schedule):
        self._initialize_state(schedule)
        while self.events:
            if self.time > 1000:
                self.log.append(f"{self.time:.4f} SYSTEM 仿真超时")
                break
            event_time, _, event_type, data = heapq.heappop(self.events)
            self.time = event_time
            handler = getattr(self, f"_handle_{event_type.lower()}", None)
            if handler and handler(data) == 'FATAL_ERROR':
                self.log.append(f"{self.time:.4f} SYSTEM 仿真因致命错误中断。")
                break
        return self._collect_results()

    def _accumulate_costs(self, cost_dict):
        self.total_distance_km += cost_dict.get('distance', 0) / 1000.0
        stability = cost_dict.get('stability', 0)
        self.total_stability_cost += stability
        self.max_stability_on_any_leg = max(self.max_stability_on_any_leg, stability)

    def _handle_dispatch_check(self, data):
        for did, trips_queue in self.pending_trips.items():
            if not trips_queue: continue
            vid = self._find_available_vehicle(did)
            if not vid: continue
            trip = trips_queue.popleft()
            v_state = self.vehicles[vid]
            if v_state['power'] < 99.9:
                trips_queue.appendleft(trip)
                self._start_or_queue_charge(vid)
                continue
            first_site = trip['sites'][0]
            cost = self._safe_cost(did, first_site)
            if cost['time'] > self.UNREACHABLE_THRESHOLD:
                self.log.append(f"{self.time:.4f} FATAL_ERROR: 车辆 {vid} 无法从 {did} 到达 {first_site}。")
                return 'FATAL_ERROR'

            self._accumulate_costs(cost)

            v_state['status'] = 'DRIVING'
            v_state['current_trip'] = trip
            raw_sites = trip['sites']
            display_path = []
            if raw_sites:
                display_path.append(raw_sites[0])
                for i in range(1, len(raw_sites)):
                    if raw_sites[i] != raw_sites[i - 1]:
                        display_path.append(raw_sites[i])
            task_str = '->'.join(display_path)
            self.log.append(f"{self.time:.4f} LOG START_TRIP VID {vid} TASK {task_str}")
            self._schedule_event(cost['time'] / 3600.0, "ARRIVE_AT_SITE", {'vid': vid, 'site_idx': 0})
        if any(q for q in self.pending_trips.values()):
            self._schedule_event(0.1, "DISPATCH_CHECK", {})

    def _handle_arrive_at_site(self, data):
        vid, site_idx = data['vid'], data['site_idx']
        v_state = self.vehicles[vid]
        trip = v_state['current_trip']
        current_site = trip['sites'][site_idx]
        prev_node = v_state['location']
        cost = self._safe_cost(prev_node, current_site)
        v_state['power'] -= cost['power']
        v_state['location'] = current_site
        self.log.append(f"{self.time:.4f} LOG ARRIVE_SITE VID {vid} AT {current_site} POWER {v_state['power']:.1f}%")
        self._schedule_event(0.5, "FINISH_UNLOAD", {'vid': vid, 'site_idx': site_idx})

    def _handle_finish_unload(self, data):
        self.last_delivery_time = max(self.last_delivery_time, self.time)
        vid, site_idx = data['vid'], data['site_idx']
        v_state = self.vehicles[vid]
        trip = v_state['current_trip']
        current_location = v_state['location']
        if site_idx + 1 < len(trip['sites']):
            next_site = trip['sites'][site_idx + 1]
            if next_site == current_location:
                self.log.append(f"{self.time:.4f} LOG CONSECUTIVE_DELIVERY VID {vid} AT {current_location}")
                self._schedule_event(0.5, "FINISH_UNLOAD", {'vid': vid, 'site_idx': site_idx + 1})
                return
            cost = self._safe_cost(current_location, next_site)
            if cost['time'] > self.UNREACHABLE_THRESHOLD:
                self.log.append(f"{self.time:.4f} FATAL_ERROR: 车辆 {vid} 无法从 {current_location} 到达 {next_site}。")
                return 'FATAL_ERROR'

            self._accumulate_costs(cost)

            self._schedule_event(cost['time'] / 3600.0, "ARRIVE_AT_SITE", {'vid': vid, 'site_idx': site_idx + 1})
        else:
            best_return_time, best_return_depot, best_return_cost = float('inf'), None, None
            for depot in self.all_warehouses:
                cost = self._safe_cost(current_location, depot)
                t_sec, p = cost.get('time', float('inf')), cost.get('power', float('inf'))
                if t_sec < best_return_time and v_state['power'] - p >= 0:
                    best_return_time, best_return_depot, best_return_cost = t_sec, depot, cost
            if best_return_depot:
                self._accumulate_costs(best_return_cost)
                self.log.append(f"{self.time:.4f} LOG 车辆 {vid} 将返回最近仓库 {best_return_depot}。")
                self._schedule_event(best_return_time / 3600.0, "ARRIVE_AT_DEPOT",
                                     {'vid': vid, 'target_depot': best_return_depot})
            else:
                self.log.append(
                    f"{self.time:.4f} FATAL_ERROR: 车辆 {vid} 在 {current_location} 完成任务后无法返回任何仓库！")
                return 'FATAL_ERROR'

    def _handle_arrive_at_depot(self, data):
        vid, target_depot = data['vid'], data['target_depot']
        v_state = self.vehicles[vid]
        cost = self._safe_cost(v_state['location'], target_depot)
        v_state['power'] -= cost['power']
        v_state['location'] = target_depot
        v_state['depot'] = target_depot
        v_state['current_trip'] = None
        self.log.append(f"{self.time:.4f} LOG END_TRIP VID {vid} AT {target_depot} POWER {v_state['power']:.1f}%")
        self._start_or_queue_charge(vid)

    def _start_or_queue_charge(self, vid):
        v_state = self.vehicles[vid]
        depot_id = v_state['depot']
        if depot_id not in self.depots:
            self.log.append(f"{self.time:.4f} FATAL_ERROR: 车辆 {vid} 尝试在未初始化的仓库 {depot_id} 充电。")
            return 'FATAL_ERROR'
        d_state = self.depots[depot_id]
        if v_state['status'] in ['CHARGING', 'WAITING_FOR_CHARGE']: return
        if v_state['power'] >= 99.9:
            v_state['status'] = 'IDLE'
            self._schedule_event(0.01, "DISPATCH_CHECK", {})
            return
        if d_state['charging_slots_used'] < config.MAX_CHARGING_VEHICLES_PER_WAREHOUSE:
            d_state['charging_slots_used'] += 1
            v_state['status'] = 'CHARGING'
            best_duration = 0
            for duration in sorted(config.VALID_CHARGE_DURATIONS, reverse=True):
                if vehicle_model.calculate_final_charge(v_state['power'], duration) >= 99.9:
                    best_duration = duration
                    break
            if best_duration == 0: best_duration = max(config.VALID_CHARGE_DURATIONS)
            self.log.append(
                f"{self.time:.4f} LOG START_CHARGE VID {vid} AT {depot_id} FOR {best_duration}h (电量 {v_state['power']:.1f}%)")
            self._schedule_event(best_duration, "FINISH_CHARGE", {'vid': vid, 'duration': best_duration})
        else:
            v_state['status'] = 'WAITING_FOR_CHARGE'
            d_state['charge_queue'].append(vid)
            self.log.append(f"{self.time:.4f} LOG QUEUE_CHARGE VID {vid} AT {depot_id} (充电位已满)")

    def _handle_finish_charge(self, data):
        vid, duration = data['vid'], data['duration']
        v_state = self.vehicles[vid]
        depot_id = v_state['depot']
        d_state = self.depots[depot_id]
        v_state['power'] = vehicle_model.calculate_final_charge(v_state['power'], duration)
        v_state['status'] = 'IDLE'
        d_state['charging_slots_used'] -= 1
        self.log.append(f"{self.time:.4f} LOG END_CHARGE VID {vid} AT {depot_id} POWER {v_state['power']:.1f}%")
        if d_state['charge_queue']:
            self._start_or_queue_charge(d_state['charge_queue'].popleft())
        self._schedule_event(0.01, "DISPATCH_CHECK", {})

    # --- [补回] 之前被意外删除的函数 ---
    def _initialize_state(self, schedule):
        self.vehicles = {}
        self.depots = {}
        vid_counter = 0
        for depot_name in self.all_warehouses:
            if depot_name not in self.depots:
                self.depots[depot_name] = {'charging_slots_used': 0, 'charge_queue': deque()}
        for did, part_schedule in schedule.items():
            if did not in self.depots:
                self.depots[did] = {'charging_slots_used': 0, 'charge_queue': deque()}
            num_vehicles = part_schedule.get('vehicles', 0)
            for _ in range(num_vehicles):
                vid = f"V{vid_counter}"
                self.vehicles[vid] = {'id': vid, 'location': did, 'status': 'IDLE', 'power': 100.0, 'depot': did,
                                      'current_trip': None}
                vid_counter += 1
        self.pending_trips = {did: deque(part_schedule.get('trips', [])) for did, part_schedule in schedule.items()}
        self._schedule_event(0, "DISPATCH_CHECK", {})

    def _find_available_vehicle(self, depot_id):
        for vid, v_state in self.vehicles.items():
            if v_state['depot'] == depot_id and v_state['status'] == 'IDLE':
                return vid
        return None

    # --- 补回结束 ---

    def _collect_results(self):
        remaining_tasks = sum(len(q) for q in self.pending_trips.values())
        if remaining_tasks == 0:
            return {
                'status': 'success',
                'makespan': self.last_delivery_time,
                'log': self.log,
                'total_distance_km': self.total_distance_km,
                'total_stability_cost': self.total_stability_cost,
                'max_stability_on_any_leg': self.max_stability_on_any_leg
            }
        else:
            return {
                'status': 'failed',
                'message': f'仿真结束时仍有 {remaining_tasks} 个任务未完成。',
                'makespan': float('inf'),
                'log': self.log
            }
