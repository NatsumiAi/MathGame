# 文件名: visualize_q4_results.py
# 描述: 独立的可视化工具，用于读取问题4生成的路径文件，并利用预处理数据高效生成分析图。

import os
import pandas as pd
import numpy as np

# 导入我们需要的核心模块
from core import data_loader
from core import plot_tools  # 只导入 plot_tools，不再需要 GeoCalculator

# --- 全局地理数据缓存 ---
# 和主程序一样的逻辑，加载一次，反复使用
GEO_DATA_CACHE = {}


def load_precomputed_data():
    """加载预处理好的地理数据到全局缓存。"""
    global GEO_DATA_CACHE, bad_zones
    if GEO_DATA_CACHE:
        return

    precomputed_file = './data/precomputed_geo_data.npz'
    try:
        print(f"正在加载预处理的地理数据文件: {precomputed_file}...")
        data = np.load(precomputed_file)
        # 只需要坡度图用于绘图
        GEO_DATA_CACHE['slope'] = data['slope']
        print("✅ 预处理坡度图加载成功！")
    except FileNotFoundError:
        print(f"❌ 致命错误: 未找到预处理文件 '{precomputed_file}'。")
        print("💡 请先运行 'preprocess_vectorized.py' 来生成此文件。")
        exit()

    # 加载不良区域（用于在地图上绘制）
    bad_zones = data_loader.load_bad_zones()


def visualize_q4_path(path_filename, map_title):
    """
    从Excel文件加载问题4的路径，并进行可视化。

    :param path_filename: 在 'output' 文件夹中的路径数据文件名。
    :param map_title: 生成的地图的标题。
    """
    print(f"\n--- 开始可视化任务: {map_title} ---")

    # --- 1. 加载数据 ---
    # 加载指定的路径文件
    full_path = os.path.join('output', path_filename)
    try:
        path_df = pd.read_excel(full_path)
        print(f"✅ 路径文件 '{full_path}' 加载成功。")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到路径文件 '{full_path}'。已跳过此任务。")
        return

    # --- 2. 调用绘图函数 ---
    # 直接使用预加载的坡度图数据
    slope_map_data = GEO_DATA_CACHE['slope']

    # 动态计算稀疏度，让车头方向不至于太密集
    quiver_step = max(1, len(path_df) // 100)

    plot_tools.plot_slope_analysis_map(
        slope_map=slope_map_data,
        bad_zones=bad_zones,
        path_df=path_df,
        error_df=None,  # 问题4的路径理论上没有错误，不传入错误报告
        output_folder='output/images',  # 建议将图片统一保存到子文件夹
        map_title=map_title,
        quiver_step=quiver_step
    )
    print("--- 可视化任务完成 ---")


if __name__ == '__main__':
    # --- 0. 首先，加载一次预处理数据 ---
    load_precomputed_data()

    # --- 1. 定义问题4生成的所有路径文件和标题 ---
    q4_tasks = [
        {
            'file': '附件8：C6-Z5平稳性最优路径.xlsx',
            'title': 'Q4 路径分析 (C6-Z5 平稳性最优)'
        },
        {
            'file': '附件9：C3-Z4时效性最优路径.xlsx',
            'title': 'Q4 路径分析 (C3-Z4 时效性最优)'
        },
        {
            'file': '附件10：C5-Z7路程最短路径.xlsx',
            'title': 'Q4 路径分析 (C5-Z7 路程最短)'
        }
    ]

    # --- 2. 循环执行所有可视化任务 ---
    for task in q4_tasks:
        visualize_q4_path(
            path_filename=task['file'],
            map_title=task['title']
        )

    print("\n所有问题4的路径图已生成完毕！")

