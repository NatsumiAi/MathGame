# 文件名: slope_analysis_map_with_bad_zones_layered.py (版本更新: 使用图层法解决不良区域透明度问题)
# 描述: 使用 Numba JIT 加速计算，并绘制包含不良区域的坡度分析图。

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
# +++ 1. 新增导入 +++
from matplotlib.patches import Patch

import numba

from core import data_loader
from core.geo_calculator import GeoCalculator


# JIT 加速的核心计算函数，保持不变
@numba.jit(nopython=True, parallel=True, cache=True)
def calculate_slope_map_numba_core(map_data, rows, cols, CELL_SIZE, K_FACTOR):
    # ... (此函数内容无变化，为简洁省略) ...
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
    fonts = ['STSONG', 'Microsoft YaHei', 'SimHei', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']
    for font_name in fonts:
        try:
            return FontProperties(fname=None, family=font_name)
        except Exception:
            continue
    return FontProperties()


def calculate_full_slope_map(calculator, map_shape):
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


# ++++++++++++++++++++++ 函数修改点 (START) - 完全替换旧函数 ++++++++++++++++++++++
def plot_slope_analysis_map(slope_map, bad_zones, output_folder='output'):
    """
    根据坡度图数据，绘制精美的坡度分析图，并叠加显示不良区域。
    (版本更新：使用图层法正确渲染半透明区域，解决过度绘制问题)

    :param slope_map: 坡度图数据 (numpy array)
    :param bad_zones: 不良区域坐标的集合, e.g., {(x1, y1), (x2, y2), ...}
    :param output_folder: 图像输出文件夹
    """
    print("开始绘制坡度分析图 (图层法)...")
    font = get_chinese_font()

    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = mcolors.LinearSegmentedColormap.from_list("slope_cmap", ["#2ca02c", "yellow", "red"])

    # 1. 绘制底层的坡度图
    im = ax.imshow(slope_map, cmap=cmap, vmin=0, vmax=45,
                   origin='upper', extent=[0, slope_map.shape[1] - 1, 0, slope_map.shape[0] - 1])

    # 2. 绘制等高线和标签
    levels = [5, 10, 15, 20, 25, 30, 35, 40]
    CS = ax.contour(slope_map, levels=levels, colors='white', linewidths=1.0, alpha=0.6,
                    origin='upper', extent=[0, slope_map.shape[1] - 1, 0, slope_map.shape[0] - 1])
    ax.clabel(CS, inline=True, fontsize=9, fmt='%d°', colors='#FFFFFFB3')

    # --- 核心修改：使用图层法绘制不良区域 ---
    if bad_zones:
        # A. 创建一个与地图同样大小的4通道 (RGBA) 图像数组，初始为完全透明
        bad_zone_overlay = np.zeros((slope_map.shape[0], slope_map.shape[1], 4), dtype=float)

        # B. 将不良区域的(x, y)坐标转换为numpy数组索引(r, c)
        rows, cols = slope_map.shape
        bad_zones_array = np.array(list(bad_zones))
        bad_x, bad_y = bad_zones_array[:, 0], bad_zones_array[:, 1]

        # 坐标转换：y -> row_index
        r_indices = (rows - 1) - bad_y
        c_indices = bad_x

        # 边界检查，防止有坐标点在地图外导致程序崩溃
        valid_indices = (r_indices >= 0) & (r_indices < rows) & (c_indices >= 0) & (c_indices < cols)
        r_indices = r_indices[valid_indices]
        c_indices = c_indices[valid_indices]

        # C. 在不良区域的像素位置，设置颜色为半透明的黑色 [R, G, B, Alpha]
        bad_zone_overlay[r_indices, c_indices] = [0, 0, 0, 0.5]  # 0.5 表示 50% 透明度

        # D. 使用 imshow 将这个新创建的图层叠加到主图上
        ax.imshow(bad_zone_overlay, origin='upper',
                  extent=[0, slope_map.shape[1] - 1, 0, slope_map.shape[0] - 1])

        # E. 手动为图例创建一个图例句柄 (Patch)
        #    因为 imshow 不会自动创建图例项
        bad_zone_patch = Patch(facecolor='black', alpha=0.5, label='不良区域')
        ax.legend(handles=[bad_zone_patch], loc='upper right', prop=font)

    # --- 修改结束 ---

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('坡度 (度)', fontproperties=font, size=14)

    ax.set_title('高原高寒地区坡度分析图 (含不良区域)', fontproperties=font, size=18, pad=20)
    ax.set_xlabel('经度', fontproperties=font, size=14)
    ax.set_ylabel('纬度', fontproperties=font, size=14)

    ax.set_xlim(0, slope_map.shape[1])
    ax.set_ylim(0, slope_map.shape[0])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = os.path.join(output_folder, 'slope_analysis_map_with_bad_zones_layered.png')  # 新文件名
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已保存至 '{output_filename}'。")
    plt.show()


# ++++++++++++++++++++++ 函数修改点 (END) ++++++++++++++++++++++


if __name__ == '__main__':
    # 主程序部分无需任何改动，保持原样
    map_data = data_loader.load_map_data(data_folder='data')
    bad_zones_set = data_loader.load_bad_zones(data_folder='data')
    calculator = GeoCalculator(map_data)
    print("...GeoCalculator 已成功初始化。")
    slope_map_data = calculate_full_slope_map(calculator, map_data.shape)
    plot_slope_analysis_map(slope_map_data, bad_zones_set, output_folder='output')

