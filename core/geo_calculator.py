# 文件名: core/geo_calculator.py
# 描述: 地理信息计算器，严格按照官方附件1的定义和公式实现。

import numpy as np
from . import config

class GeoCalculator:
    """
    地理信息计算器。
    负责根据官方规则计算高程、坡度、坡向和坡面法向量。
    代码中的所有计算均严格遵循《附件1：任务区域地形数据及文件格式说明》文档。
    """

    def __init__(self, map_data):
        """
        初始化计算器。
        :param map_data: 从 map.tif 加载的完整地形高程二维数组。
        """
        self.map_data = map_data
        self.rows, self.cols = map_data.shape

        # --- 核心参数 (源自附件1) ---
        self.CELL_SIZE = config.CELL_SIZE
        self.K_FACTOR = config.ELEVATION_FACTOR_K

        # --- 缓存机制 ---
        # 避免对同一栅格重复进行复杂的梯度、坡度和法向量计算，显著提升性能。
        self.gradient_cache = {}
        self.slope_aspect_cache = {}
        self.normal_vector_cache = {}

    def get_elevation(self, x, y):
        """
        根据栅格坐标获取高程。
        严格实现了《附件1》中"使用Python根据栅格坐标读取高程示例代码"的转换逻辑。
        :param x: 栅格的x坐标 (0-12499)
        :param y: 栅格的y坐标 (0-12499)
        :return: 对应栅格的高程值。若坐标越界，返回np.nan。
        """
        # 检查坐标是否在允许范围内
        if not (0 <= x < self.cols and 0 <= y < self.rows):
            return np.nan

        # 官方坐标转换公式: row_index = 12499 - y, col_index = x
        # self.rows - 1 等价于 12499
        r, c = (self.rows - 1) - y, x
        return self.map_data[r, c]

    def _calculate_gradients(self, x, y):
        """
        [私有] 计算高程变化率 (梯度)。
        严格实现了《附件1》中的修正后的计算公式(2)，即Sobel算子。
        返回值为本题定义的、包含了风险因子k的“高程变化率”。
        """
        if (x, y) in self.gradient_cache:
            return self.gradient_cache[(x, y)]

        # 根据《附件1》注记：“由于任务区域边界上的栅格没有3×3邻域，因此不计算这些栅格的坡度和坡向”
        # 我们通过返回0梯度来实现这一点，这将导致坡度为0，符合规定。
        if not (0 < x < self.cols - 1 and 0 < y < self.rows - 1):
            return 0.0, 0.0

        # --- 获取计算所需的3x3邻域高程 ---
        # 命名(a,b,c..i)与《附件1》图2完全对应，便于核对公式。
        h = {
            'a': self.get_elevation(x - 1, y + 1), 'b': self.get_elevation(x, y + 1),
            'c': self.get_elevation(x + 1, y + 1),
            'd': self.get_elevation(x - 1, y),     'f': self.get_elevation(x + 1, y),
            'g': self.get_elevation(x - 1, y - 1), 'h': self.get_elevation(x, y - 1),
            'i': self.get_elevation(x + 1, y - 1)
        }

        # 健壮性检查：如果任何邻居点在地图外（理论上边界检查已覆盖），则不计算。
        if any(np.isnan(v) for v in h.values()):
            return 0.0, 0.0

        # --- 严格应用官方公式(2) ---
        # Δz/Δx = k * [(c + 2f + i) - (a + 2d + g)] / (8 * L)
        dz_dx = self.K_FACTOR * ((h['c'] + 2 * h['f'] + h['i']) - (h['a'] + 2 * h['d'] + h['g'])) / (8 * self.CELL_SIZE)

        # Δz/Δy = k * [(a + 2b + c) - (g + 2h + i)] / (8 * L)
        dz_dy = self.K_FACTOR * ((h['a'] + 2 * h['b'] + h['c']) - (h['g'] + 2 * h['h'] + h['i'])) / (8 * self.CELL_SIZE)

        self.gradient_cache[(x, y)] = (dz_dx, dz_dy)
        return dz_dx, dz_dy

    def get_slope_and_aspect(self, x, y):
        """
        计算栅格的坡度(度)和坡向(度)。
        """
        # 检查缓存，如果已计算过则直接返回结果
        if (x, y) in self.slope_aspect_cache:
            # 外部调用通常只需要角度值，因此我们返回元组的第一个元素
            return self.slope_aspect_cache[(x, y)][0]

        # 步骤1: 获取官方定义的“高程变化率” (Δz/Δx, Δz/Δy)
        dz_dx, dz_dy = self._calculate_gradients(x, y)

        # 步骤2: 处理水平面情况
        if dz_dx == 0 and dz_dy == 0:
            result_deg = (0.0, '水平面')
            # 同时缓存弧度值(0, 0)，方便法向量函数调用
            self.slope_aspect_cache[(x, y)] = (result_deg, (0.0, 0.0))
            return result_deg

        # 步骤3: 计算坡度 (Slope)，严格遵循《附件1》坡度公式
        # S = arctan(sqrt((Δz/Δx)² + (Δz/Δy)²))
        slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
        slope_deg = np.rad2deg(slope_rad)

        # 步骤4: 计算坡向 (Aspect)
        # 《附件1》定义:"...投影与正北方向的夹角为坡向，以正北方向为0度，顺时针为正方向。"
        # np.arctan2(dz_dx, dz_dy) 是该定义的标准、高效且数值稳定的实现。
        aspect_rad = np.arctan2(-dz_dx, -dz_dy)
        aspect_deg = np.rad2deg(aspect_rad)
        # 将(-180, 180]范围的角度转换为[0, 360)范围
        if aspect_deg < 0:
            aspect_deg += 360

        # 将角度制的最终结果和弧度制的中间结果一并缓存
        result_deg = (slope_deg, aspect_deg)
        self.slope_aspect_cache[(x, y)] = (result_deg, (slope_rad, aspect_rad))

        return result_deg

    def get_normal_vector(self, x, y):
        """
        计算坡面法向量。
        严格实现了《附件1》中法向量的最终计算公式:
        (nx, ny, nz) = (sinA * sinS, cosA * sinS, cosS)
        """
        if (x, y) in self.normal_vector_cache:
            return self.normal_vector_cache[(x, y)]

        # 步骤1: 获取坡度和坡向的计算结果。
        # 此处我们直接从缓存中获取完整的元组(角度结果, 弧度结果)
        # self.get_slope_and_aspect(x,y) 会确保计算并填充缓存
        self.get_slope_and_aspect(x, y)
        result_deg, result_rad = self.slope_aspect_cache[(x, y)]

        # 步骤2: 处理水平面情况
        if result_deg[1] == '水平面':
            # 根据《附件1》注记：“规定坡面是水平面时的法向量为：(0, 0, 1)”
            vec = np.array([0.0, 0.0, 1.0])
        else:
            # 步骤3: 应用官方球坐标法向量公式
            slope_rad, aspect_rad = result_rad
            S = slope_rad  # S 为弧度制坡度
            A = aspect_rad  # A 为弧度制坡向

            # (nx,ny,nz)=(sinA·sinS, cosA·sinS, cosS)
            nx = np.sin(A) * np.sin(S)
            ny = np.cos(A) * np.sin(S)
            nz = np.cos(S)

            vec = np.array([nx, ny, nz])

        # 官方公式推导出的已经是单位向量，无需归一化。
        # 为防止浮点数精度问题，保留归一化操作作为健壮性保障。
        norm = np.linalg.norm(vec)
        normalized_vec = vec / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

        self.normal_vector_cache[(x, y)] = normalized_vec
        return normalized_vec
