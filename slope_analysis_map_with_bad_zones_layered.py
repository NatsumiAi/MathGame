# 文件名: slope_analysis_map_with_bad_zones_layered.py (版本更新: 增加点位信息标注)
# 描述: 使用 Numba JIT 加速计算，并绘制包含不良区域和各类点位的坡度分析图。

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
# +++ 1. 新增导入，用于图例 +++
from matplotlib.lines import Line2D

import numba

from core import data_loader
from core.geo_calculator import GeoCalculator


# JIT 加速的核心计算函数 (无变化)
@numba.jit(nopython=True, parallel=True, cache=True)
def calculate_slope_map_numba_core(map_data, rows, cols, CELL_SIZE, K_FACTOR):
    slope_map = np.zeros((rows, cols), dtype=np.float32)
    for y in numba.prange(rows):
        for x in range(cols):
            if not (0 < x < cols - 1 and 0 < y < rows - 1):
                continue
            r, c = (rows - 1) - y, x
            h_a = map_data[r - 1, c - 1];
            h_b = map_data[r - 1, c];
            h_c = map_data[r - 1, c + 1]
            h_d = map_data[r, c - 1];
            h_f = map_data[r, c + 1]
            h_g = map_data[r + 1, c - 1];
            h_h = map_data[r + 1, c];
            h_i = map_data[r + 1, c + 1]
            dz_dx = K_FACTOR * ((h_c + 2.0 * h_f + h_i) - (h_a + 2.0 * h_d + h_g)) / (8.0 * CELL_SIZE)
            dz_dy = K_FACTOR * ((h_a + 2.0 * h_b + h_c) - (h_g + 2.0 * h_h + h_i)) / (8.0 * CELL_SIZE)
            if dz_dx == 0.0 and dz_dy == 0.0:
                slope = 0.0
            else:
                slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
                slope = np.rad2deg(slope_rad)
            slope_map[r, c] = slope
    return slope_map


def get_chinese_font():
    # ... (此函数无变化) ...
    fonts = ['STSONG', 'Microsoft YaHei', 'SimHei', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']
    for font_name in fonts:
        try:
            return FontProperties(fname=None, family=font_name)
        except Exception:
            continue
    return FontProperties()


def calculate_full_slope_map(calculator, map_shape):
    # ... (此函数无变化) ...
    rows, cols = map_shape
    print("=" * 60)
    print("🚀 使用 Numba JIT + 并行计算加速生成坡度图...")
    print("首次运行会进行编译，可能需要一些时间。后续运行将直接加载缓存，速度飞快！")
    print("=" * 60)
    start_time = time.time()
    slope_map = calculate_slope_map_numba_core(
        calculator.map_data, rows, cols, calculator.CELL_SIZE, calculator.K_FACTOR
    )
    total_time = time.time() - start_time
    print(f"\n✅ Numba JIT 坡度图计算完成！总耗时: {total_time:.2f} 秒。")
    return slope_map


# +++ 2. 修改绘图函数签名，增加 points_df 参数 +++
def plot_slope_analysis_map(slope_map, bad_zones, points_df, output_folder='output'):
    """
    根据坡度图数据，绘制精美的坡度分析图，叠加显示不良区域，并标注各类点位。

    :param slope_map: 坡度图数据 (numpy array)
    :param bad_zones: 不良区域坐标的集合, e.g., {(x1, y1), ...}
    :param points_df: 包含点位信息的 pandas DataFrame
    :param output_folder: 图像输出文件夹
    """
    print("开始绘制坡度分析图 (含不良区域和点位信息)...")
    font = get_chinese_font()

    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = mcolors.LinearSegmentedColormap.from_list("slope_cmap", ["#2ca02c", "yellow", "red"])

    # 1. 绘制底层的坡度图 (无变化)
    im = ax.imshow(slope_map, cmap=cmap, vmin=0, vmax=45,
                   origin='upper', extent=[0, slope_map.shape[1] - 1, 0, slope_map.shape[0] - 1])

    # 2. 绘制等高线和标签 (无变化)
    levels = [5, 10, 15, 20, 25, 30, 35, 40]
    CS = ax.contour(slope_map, levels=levels, colors='white', linewidths=1.0, alpha=0.6,
                    origin='upper', extent=[0, slope_map.shape[1] - 1, 0, slope_map.shape[0] - 1])
    ax.clabel(CS, inline=True, fontsize=9, fmt='%d°', colors='#FFFFFFB3')

    # 3. 绘制不良区域 (无变化)
    if bad_zones:
        rows, cols = slope_map.shape
        bad_zone_overlay = np.zeros((rows, cols, 4), dtype=float)
        bad_zones_array = np.array(list(bad_zones))
        bad_x, bad_y = bad_zones_array[:, 0], bad_zones_array[:, 1]
        r_indices = (rows - 1) - bad_y
        c_indices = bad_x
        valid_indices = (r_indices >= 0) & (r_indices < rows) & (c_indices >= 0) & (c_indices < cols)
        r_indices, c_indices = r_indices[valid_indices], c_indices[valid_indices]
        bad_zone_overlay[r_indices, c_indices] = [0, 0, 0, 0.5]
        ax.imshow(bad_zone_overlay, origin='upper',
                  extent=[0, slope_map.shape[1] - 1, 0, slope_map.shape[0] - 1])

    # +++ 4. 新增：绘制各点点位信息 +++
    legend_handles = []  # 用于统一管理图例项

    if points_df is not None and not points_df.empty:
        # 定义不同类别点的样式 (颜色, 标记, 图例名称)
        style_map = {
            '前沿阵地': {'color': '#FF1493', 'marker': 'P', 'label': '前沿阵地'},  # 深粉色, 加号
            '中转仓库候选点': {'color': '#00BFFF', 'marker': 'o', 'label': '中转仓库'},  # 深天蓝, 圆圈
            '后方基地': {'color': '#ADFF2F', 'marker': 's', 'label': '后方基地'}  # 绿黄色, 方块
        }

        # 遍历DataFrame中的每一行来绘制点和标签
        for index, point in points_df.iterrows():
            category = point['类别']
            label = point['编号']
            x, y = point['栅格x坐标'], point['栅格y坐标']

            style = style_map.get(category, {'color': 'white', 'marker': 'x', 'label': '未知点'})

            # 绘制点标记
            ax.scatter(x, y,
                       c=style['color'],
                       marker=style['marker'],
                       s=80,  # 点的大小
                       edgecolors='black',  # 点的边缘颜色，使其更突出
                       linewidth=1,
                       label=label,  # 这个label不会直接用于图例
                       zorder=10)  # zorder确保点在最上层

            # 在点的旁边绘制编号文本
            ax.text(x + 50, y + 50, label,  # x, y偏移量，防止文字覆盖点
                    color='white',
                    fontsize=9,
                    fontweight='bold',
                    fontproperties=font,
                    # 添加背景使其在复杂背景下更清晰
                    bbox=dict(facecolor=style['color'], alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'),
                    zorder=11)

        # 为点位类别创建图例句柄
        for cat, style in style_map.items():
            legend_handles.append(
                Line2D([0], [0], marker=style['marker'], color='w', label=style['label'],
                       markerfacecolor=style['color'], markeredgecolor='black', markersize=10)
            )

    # 将不良区域的图例句柄也加入列表
    if bad_zones:
        legend_handles.append(Patch(facecolor='black', alpha=0.5, label='不良区域'))

    # 统一显示所有图例
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', prop=font, fontsize=10)

    # --- 图表美化部分 (无变化) ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('坡度 (度)', fontproperties=font, size=14)

    ax.set_title('高原高寒地区坡度分析图 (含不良区域及关键点位)', fontproperties=font, size=18, pad=20)
    ax.set_xlabel('栅格X坐标', fontproperties=font, size=14)
    ax.set_ylabel('栅格Y坐标', fontproperties=font, size=14)

    ax.set_xlim(0, slope_map.shape[1])
    ax.set_ylim(0, slope_map.shape[0])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = os.path.join(output_folder, 'slope_analysis_map_with_all_points.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 包含所有点位的地图已保存至 '{output_filename}'。")
    plt.show()


if __name__ == '__main__':
    # 主程序部分
    map_data = data_loader.load_map_data(data_folder='data')
    bad_zones_set = data_loader.load_bad_zones(data_folder='data')
    # +++ 5. 新增：调用函数加载点位信息 +++
    points_info_df = data_loader.load_points_info(data_folder='data')

    calculator = GeoCalculator(map_data)
    print("...GeoCalculator 已成功初始化。")

    slope_map_data = calculate_full_slope_map(calculator, map_data.shape)

    # +++ 6. 修改：将读取到的点位信息DataFrame传递给绘图函数 +++
    plot_slope_analysis_map(slope_map_data, bad_zones_set, points_info_df, output_folder='output')

