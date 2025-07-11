# 文件名: visualize_report.py
# 描述: 独立的可视化工具，用于读取路径文件和错误报告，并生成带有错误标记的分析图。

import pandas as pd
from core import data_loader
from core.geo_calculator import GeoCalculator
from core.plot_tools import plot_slope_analysis_map, calculate_full_slope_map
import os

def visualize_from_excel(path_file, report_file, map_title):
    """
    从Excel文件加载路径和错误报告，并进行可视化。

    :param path_file: 路径数据文件的名称 (e.g., 'P3-P4的行驶路径.xlsx')
    :param report_file: 错误报告文件的名称 (e.g., 'problem2_report_final_correct.xlsx')
    :param map_title: 生成的地图的标题
    """
    print("--- 开始执行独立的可视化任务 ---")

    # --- 1. 加载所有需要的数据 ---
    print("--- 正在加载底图和路径数据 ---")
    map_data = data_loader.load_map_data()
    # bad_zones_set = data_loader.load_bad_zones() # 可选：如果不需要画不良区域，可以注释掉以提速

    # 加载指定的路径文件
    path_df = data_loader.load_path_data(path_file)
    print(f"✅ 路径文件 '{path_file}' 加载成功。")

    # 加载指定的错误报告文件
    try:
        error_df = pd.read_excel(os.path.join('output', report_file))
        print(f"✅ 错误报告 '{report_file}' 加载成功，共 {len(error_df)} 条错误。")
    except FileNotFoundError:
        print(f"⚠️ 警告: 找不到错误报告 '{report_file}'。将只绘制路径。")
        error_df = None

    # --- 2. 初始化工具并计算坡度图 ---
    calculator = GeoCalculator(map_data)
    slope_map_data = calculate_full_slope_map(calculator, map_data.shape)

    # --- 3. 调用绘图函数 ---
    plot_slope_analysis_map(
        slope_map=slope_map_data,
        bad_zones=None,  # 设置为None，避免地图过于杂乱
        path_df=path_df,
        error_df=error_df,  # 传入从Excel读取的错误报告
        output_folder='output',
        map_title=map_title
    )
    print("\n--- 可视化任务完成 ---")


if __name__ == '__main__':
    # --- 在这里配置你要可视化的文件 ---

    # --- 示例1: 可视化 问题2 (P3-P4) 的报告 ---
    PATH_FILENAME_P3_P4 = 'P3-P4的行驶路径.xlsx'
    REPORT_FILENAME_P3_P4 = 'problem2_report_final_correct.xlsx'
    MAP_TITLE_P3_P4 = 'P3-P4路径可通行性分析图'

    visualize_from_excel(
        path_file=PATH_FILENAME_P3_P4,
        report_file=REPORT_FILENAME_P3_P4,
        map_title=MAP_TITLE_P3_P4
    )

    # # --- 示例2: 如果你想可视化其他报告，可以取消下面的注释并修改文件名 ---
    # PATH_FILENAME_OTHER = 'P5-P6的行驶路径.xlsx'
    # REPORT_FILENAME_OTHER = 'some_other_report.xlsx'
    # MAP_TITLE_OTHER = 'P5-P6的行驶路径的分析图'
    #
    # visualize_from_excel(
    #     path_file=PATH_FILENAME_OTHER,
    #     report_file=REPORT_FILENAME_OTHER,
    #     map_title=MAP_TITLE_OTHER
    # )
# --- 示例1: 可视化 问题2 (P3-P4) 的报告 ---
#     PATH_FILENAME_P1_P2 = 'P1-P2的行驶路径.xlsx'
#     REPORT_FILENAME_P1_P2 = ''
#     MAP_TITLE_P1_P2 = 'P1-P2路径分析图'
#
#     visualize_from_excel(
#         path_file=PATH_FILENAME_P1_P2,
#         report_file=REPORT_FILENAME_P1_P2,
#         map_title=MAP_TITLE_P1_P2
#     )
