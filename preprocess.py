# 文件名: preprocess_vectorized.py
# 描述: 终极优化版预处理脚本。使用NumPy向量化操作和Numba JIT并行计算
#       全地图的坡度、坡向和法向量，并将结果保存。

import numpy as np
import time
from core import data_loader, config
import numba


@numba.njit(parallel=True)
def vectorized_slope_aspect_normals(map_data, CELL_SIZE, K_FACTOR):
    """
    使用Numba JIT并行计算全地图的坡度、坡向和法向量。
    返回: slope_map, aspect_map, normal_vectors_map (3D array)
    """
    rows, cols = map_data.shape
    slope_map = np.zeros_like(map_data, dtype=np.float32)
    aspect_map = np.zeros_like(map_data, dtype=np.float32)
    # 创建一个 (rows, cols, 3) 的数组来存储每个点的 (nx, ny, nz)
    normal_vectors_map = np.zeros((rows, cols, 3), dtype=np.float32)

    # Numba的并行循环
    for r in numba.prange(1, rows - 1):
        for c in range(1, cols - 1):
            # 获取3x3邻域高程
            h_a = map_data[r - 1, c - 1]
            h_b = map_data[r - 1, c]
            h_c = map_data[r - 1, c + 1]
            h_d = map_data[r, c - 1]
            h_f = map_data[r, c + 1]
            h_g = map_data[r + 1, c - 1]
            h_h = map_data[r + 1, c]
            h_i = map_data[r + 1, c + 1]

            # 梯度计算 (Sobel算子)
            dz_dx = K_FACTOR * ((h_c + 2 * h_f + h_i) - (h_a + 2 * h_d + h_g)) / (8 * CELL_SIZE)
            dz_dy = K_FACTOR * ((h_a + 2 * h_b + h_c) - (h_g + 2 * h_h + h_i)) / (8 * CELL_SIZE)

            if dz_dx == 0.0 and dz_dy == 0.0:
                # 水平面
                slope_rad = 0.0
                aspect_rad = 0.0  # 坡向无定义，设为0
                nx, ny, nz = 0.0, 0.0, 1.0
            else:
                # 坡度
                slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
                # 坡向
                aspect_rad = np.arctan2(-dz_dx, -dz_dy)  # 与GeoCalculator一致

                # 法向量
                S, A = slope_rad, aspect_rad
                nx = np.sin(A) * np.sin(S)
                ny = np.cos(A) * np.sin(S)
                nz = np.cos(S)

            slope_map[r, c] = np.rad2deg(slope_rad)
            aspect_deg = np.rad2deg(aspect_rad)
            aspect_map[r, c] = aspect_deg if aspect_deg >= 0 else aspect_deg + 360
            normal_vectors_map[r, c, 0] = nx
            normal_vectors_map[r, c, 1] = ny
            normal_vectors_map[r, c, 2] = nz

    # 单独处理边界，法向量设为(0,0,1)
    for r in range(rows):
        normal_vectors_map[r, 0, 2] = 1.0
        normal_vectors_map[r, cols - 1, 2] = 1.0
    for c in range(cols):
        normal_vectors_map[0, c, 2] = 1.0
        normal_vectors_map[rows - 1, c, 2] = 1.0

    return slope_map, aspect_map, normal_vectors_map


def run_vectorized_preprocessing():
    """
    主函数，执行全地图的向量化预处理。
    """
    print("=" * 60)
    print("🚀 开始执行【终极优化版】全地图预处理...")
    print("=" * 60)

    # 1. 加载地形数据
    print("正在加载原始地形数据 map.tif...")
    map_data = data_loader.load_map_data()
    print(f"地形数据加载成功，尺寸: {map_data.shape}")

    # 2. 调用Numba优化的函数进行并行计算
    print("\n正在使用Numba JIT并行计算坡度、坡向和法向量...")
    start_time = time.time()

    slope_map, aspect_map, normal_vectors_map = vectorized_slope_aspect_normals(
        map_data,
        config.CELL_SIZE,
        config.ELEVATION_FACTOR_K
    )

    duration = time.time() - start_time
    # Numba第一次运行时需要编译，会慢一些，第二次运行会快很多
    print(f"✅ 全地图地理信息计算完成！总耗时: {duration:.2f} 秒。")

    # 3. 保存所有结果到一个文件
    output_filename = './data/precomputed_geo_data.npz'
    print(f"\n正在将所有预计算数据保存到: {output_filename}")

    np.savez_compressed(
        output_filename,
        slope=slope_map,
        aspect=aspect_map,
        normals=normal_vectors_map
    )

    print("✅ 数据保存成功！")
    print("\n现在 problem4_solver.py 可以光速加载这些数据。")
    print("=" * 60)


if __name__ == '__main__':
    run_vectorized_preprocessing()
