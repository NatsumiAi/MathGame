# 文件名: plot_tools.py (图层法美化版 - 最终)
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch  # [新增] 导入Patch用于手动创建图例
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numba


# ... (calculate_slope_map_numba_core, get_chinese_font, calculate_full_slope_map 函数不变) ...
@numba.jit(nopython=True, parallel=True, cache=True)
def calculate_slope_map_numba_core(map_data, rows, cols, CELL_SIZE, K_FACTOR):
    slope_map = np.zeros((rows, cols), dtype=np.float32)
    for y in numba.prange(rows):
        for x in range(cols):
            if not (0 < x < cols - 1 and 0 < y < rows - 1): continue
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
                slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)); slope = np.rad2deg(slope_rad)
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
    print("=" * 60 + "\n🚀 使用 Numba JIT + 并行计算加速生成坡度图...\n" + "=" * 60)
    start_time = time.time()
    slope_map = calculate_slope_map_numba_core(calculator.map_data, rows, cols, calculator.CELL_SIZE,
                                               calculator.K_FACTOR)
    print(f"\n✅ Numba JIT 坡度图计算完成！总耗时: {time.time() - start_time:.2f} 秒。")
    return slope_map


# ++++++++++++++++++++++++++++ 核心绘图函数 (新版本) ++++++++++++++++++++++++++++
def plot_slope_analysis_map(slope_map, bad_zones, path_df=None, error_df=None, output_folder='output',
                            map_title='综合分析图', quiver_step=20):
    print(f"开始绘制综合分析图: {map_title}")
    font = get_chinese_font()
    fig, ax = plt.subplots(figsize=(20, 16))
    cmap_slope = mcolors.LinearSegmentedColormap.from_list("slope_cmap", ["#2ca02c", "yellow", "red"])

    im = ax.imshow(slope_map, cmap=cmap_slope, vmin=0, vmax=45, origin='upper',
                   extent=[0, slope_map.shape[1] - 1, 0, slope_map.shape[0] - 1])
    levels = np.arange(0, 45, 5)
    CS = ax.contour(slope_map, levels=levels, colors='white', linewidths=0.8, alpha=0.6, origin='upper',
                    extent=[0, slope_map.shape[1] - 1, 0, slope_map.shape[0] - 1])
    ax.clabel(CS, inline=True, fontsize=9, fmt='%d°', colors='#FFFFFFB3')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('坡度 (度)', fontproperties=font, size=14)

    # --- [核心修改] 使用图层法绘制不良区域 ---
    legend_handles = []  # 用于收集所有图例项
    if bad_zones:
        print("正在创建不良区域叠加图层...")
        bad_zone_overlay = np.zeros((slope_map.shape[0], slope_map.shape[1], 4), dtype=float)
        rows, cols = slope_map.shape
        bad_zones_array = np.array(list(bad_zones), dtype=int)
        bad_x, bad_y = bad_zones_array[:, 0], bad_zones_array[:, 1]

        r_indices = (rows - 1) - bad_y
        c_indices = bad_x

        valid_indices = (r_indices >= 0) & (r_indices < rows) & (c_indices >= 0) & (c_indices < cols)
        r_indices = r_indices[valid_indices]
        c_indices = c_indices[valid_indices]

        # 设置颜色为半透明的灰色，比纯黑柔和
        bad_zone_overlay[r_indices, c_indices] = [0.2, 0.2, 0.2, 0.5]

        ax.imshow(bad_zone_overlay, origin='upper',
                  extent=[0, slope_map.shape[1] - 1, 0, slope_map.shape[0] - 1], zorder=2)

        # 手动创建图例项
        legend_handles.append(Patch(facecolor=[0.2, 0.2, 0.2], alpha=0.5, label='不良区域'))

    # --- 路径绘制逻辑 ---
    if path_df is not None and not path_df.empty:
        x_col = '栅格x坐标' if '栅格x坐标' in path_df.columns else 'x'
        y_col = '栅格y坐标' if '栅格y坐标' in path_df.columns else 'y'
        h_col = '车头朝向' if '车头朝向' in path_df.columns else 'heading'

        path_x, path_y = path_df[x_col].values, path_df[y_col].values

        # 为了让路径图例项显示为一条线，而不是一个点
        line, = ax.plot(path_x, path_y, color='teal', linewidth=1.5, label='路径', zorder=5)
        legend_handles.append(line)

        if h_col in path_df.columns:
            path_h = path_df[h_col].values;
            quiver_x = path_x[::quiver_step];
            quiver_y = path_y[::quiver_step]
            quiver_h = path_h[::quiver_step];
            math_angles_rad = np.deg2rad(90 - quiver_h)
            u, v = np.cos(math_angles_rad), np.sin(math_angles_rad)
            quiver = ax.quiver(quiver_x, quiver_y, u, v, color='black', alpha=0.7, scale=45, width=0.0035,
                               label=f"车头方向 (每{quiver_step}点)", zorder=6)
            legend_handles.append(quiver)

        start_pt, = ax.plot(path_x[0], path_y[0], 'o', color='lime', markersize=10, markeredgecolor='black',
                            label='起点', zorder=12)
        end_pt, = ax.plot(path_x[-1], path_y[-1], 's', color='red', markersize=10, markeredgecolor='black',
                          label='终点', zorder=12)
        legend_handles.extend([start_pt, end_pt])

        padding_factor = 0.1
        path_width = np.max(path_x) - np.min(path_x) if len(path_x) > 1 else 100
        path_height = np.max(path_y) - np.min(path_y) if len(path_y) > 1 else 100
        padding_x = path_width * padding_factor;
        padding_y = path_height * padding_factor
        ax.set_xlim(np.min(path_x) - padding_x, np.max(path_x) + padding_x)
        ax.set_ylim(np.min(path_y) - padding_y, np.max(path_y) + padding_y)

    # --- 错误标记绘制 ---
    # ...

    # --- 图表属性设置 ---
    ax.legend(handles=legend_handles, loc='upper right', prop=font, facecolor='#FFFFFFBF', frameon=True,
              framealpha=0.75, fancybox=True)
    ax.set_title(map_title, fontproperties=font, size=20, pad=20)
    ax.set_xlabel('栅格 x 坐标', fontproperties=font, size=14)
    ax.set_ylabel('栅格 y 坐标', fontproperties=font, size=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
    ax.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.08)

    # --- 保存图像 ---
    os.makedirs(output_folder, exist_ok=True)
    safe_title = "".join([c for c in map_title if c.isalnum() or c in (' ', '_')]).rstrip().replace(" ", "_").replace(
        ":", "")
    output_filename = os.path.join(output_folder, f'{safe_title}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已保存至 '{output_filename}'。")
    plt.close(fig)
