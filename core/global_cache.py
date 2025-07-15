import numpy as np
import os
from typing import Dict, Any, Union, Optional  # 导入必要的类型

GeoDataType = Dict[str, Any]
# 全局缓存字典，并提供明确的类型标注
CACHE: Dict[str, Union[GeoDataType, None]] = {
    'geo_data': None,
    'bad_zones': None
}


def load_all_precomputed_data(force_reload=False):
    """
    加载所有需要预计算的数据到全局缓存中。
    使用单例模式，只加载一次。
    """
    if isinstance(CACHE.get('geo_data'), dict) and not force_reload:
        print("-> 全局数据已缓存，跳过重复加载。")
        return
    print("=" * 30)
    print("🚀 首次加载全局数据到缓存...")

    geo_file = './data/precomputed_geo_data.npz'
    try:
        print(f"  -> 正在加载地理数据: {geo_file}")
        data = np.load(geo_file)
        CACHE['geo_data'] = {
            'slope': data['slope'],
            'normals': data['normals'],
            'rows': data['slope'].shape[0],
            'cols': data['slope'].shape[1]
        }
    except FileNotFoundError:
        print(f"❌ 错误: 未找到 '{geo_file}'！请先运行 preprocess_vectorized.py。")
        exit()

    from . import data_loader
    print("  -> 正在加载不良区域数据...")
    CACHE['bad_zones'] = data_loader.load_bad_zones()

    print("✅ 全局数据加载并缓存成功！")
    print("=" * 30)