# 文件名: problem2_solver.py (最终修正版 - 匹配新规则)
# 描述: 使用最终正确的"7种状态"车辆运动学模型检查路径可通行性。

import pandas as pd
from core import data_loader, config
from core.geo_calculator import GeoCalculator
from core import vehicle_model  # 导入整个模块


def check_path_passability(path_df, calculator, bad_zones, turn_rules):
    """
    使用最终正确的"7种状态"车辆运动学模型检查路径的可通行性。
    """
    print("🚀 开始检查路径的可通行性 ...")

    max_slope = config.VEHICLE_PARAMS['A']['max_slope']
    errors = []

    # --- 1. 逐点检查 (坡度 & 不良区域) ---
    # 这部分逻辑不变，但为了完整性，我们把它写全
    for index, row in path_df.iterrows():
        point_id, x, y = row['id'], int(row['x']), int(row['y'])

        # 检查坡度
        slope, _ = calculator.get_slope_and_aspect(x, y)
        if slope > max_slope:
            errors.append({'栅格编号1': point_id, '栅格编号2': '-',
                           '错误类型': f'超过最大通行坡度 ({slope:.2f}° > {max_slope}°)'})
        # 检查不良区域
        if (x, y) in bad_zones:
            errors.append({'栅格编号1': point_id, '栅格编号2': '-', '错误类型': '进入不良区域'})

    # --- 2. 逐段检查 (车辆运动学约束) ---
    # 这部分逻辑被完全重写以匹配新规则
    for i in range(len(path_df) - 1):
        current_point = path_df.iloc[i]
        next_point = path_df.iloc[i + 1]

        id1, x1, y1, h1 = current_point['id'], current_point['x'], current_point['y'], current_point['heading']
        id2, x2, y2, h2 = next_point['id'], next_point['x'], next_point['y'], next_point['heading']
        dx, dy = x2 - x1, y2 - y1

        # 核心修改1: 获取允许的下一个朝向的列表
        possible_next_headings = turn_rules[h1].get((dx, dy))

        # 核心修改2: 检查移动是否合法
        if possible_next_headings is None:
            # 如果 get 返回 None，说明 (dx, dy) 这个 key 在 turn_rules[h1] 中不存在
            error_msg = f'非法移动 (从 {h1}° 朝向无法移动 {dx},{dy})'
            print(f"❌ 错误发现: 从 {id1} 到 {id2} - {error_msg}")
            errors.append({'栅格编号1': id1, '栅格编号2': id2, '错误类型': '非法移动'})
            continue  # 跳过后续检查

        # 核心修改3: 检查实际新朝向 h2 是否在允许的列表内
        if h2 not in possible_next_headings:
            error_msg = f'车头方向错误 (实际: {h2}°, 规则允许: {possible_next_headings})'
            print(f"❌ 错误发现: 从 {id1} 到 {id2} - {error_msg}")
            errors.append({'栅格编号1': id1, '栅格编号2': id2, '错误类型': '车头方向错误'})

    print("\n✅ 路径检查完成！")
    if not errors:
        return pd.DataFrame()

    error_df = pd.DataFrame(errors)
    error_df.sort_values(by=['栅格编号1', '栅格编号2'], inplace=True)
    error_df.drop_duplicates(inplace=True)
    return error_df


if __name__ == '__main__':
    print("--- 正在加载数据 ---")
    map_data = data_loader.load_map_data()
    bad_zones_set = data_loader.load_bad_zones()

    path_df = data_loader.load_path_data('P3-P4的行驶路径.xlsx')
    print("--- 数据加载完毕 ---\n")
    calculator = GeoCalculator(map_data)
    turn_rules = vehicle_model.generate_turn_rules()
    print("--- 车辆运动学最终模型已加载 ---\n")
    passability_errors_df = check_path_passability(path_df, calculator, bad_zones_set, turn_rules)

    if not passability_errors_df.empty:
        print(f"\n发现 {len(passability_errors_df)} 个独立的不可通行问题。结果报告如下：")
        print(passability_errors_df.to_string())
        output_filename = 'output/problem2_report_final_correct.xlsx'
        passability_errors_df.to_excel(output_filename, index=False)
        print(f"\n报告已保存至: '{output_filename}'")
    else:
        print("\n报告：该路径完全可通行。")
