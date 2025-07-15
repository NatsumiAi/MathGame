# 文件名: core/data_loader.py

import rasterio
import pandas as pd
import os


def load_map_data(data_folder='data'):
    """读取TIF地形数据"""
    print("正在加载地形数据 (map.tif)...")
    file_path = os.path.join(data_folder, 'map.tif')
    with rasterio.open(file_path) as src:
        return src.read(1)

def load_bad_zones(data_folder='data'):
    """
    读取不良区域Excel文件中所有工作表(sheet)的坐标，
    并将它们合并到一个Set中以便快速查询。
    """
    print("正在加载不良区域数据 (所有工作表)...")
    file_path = os.path.join(data_folder, '不良区域位置信息.xlsx')

    # 1. 使用 sheet_name=None 读取所有工作表。
    #    这将返回一个字典，格式为: {'工作表1名': DataFrame1, '工作表2名': DataFrame2, ...}
    all_sheets_data = pd.read_excel(file_path, sheet_name=None)

    # 2. 初始化一个空集合，用于存放所有不良区域的坐标
    all_bad_zones_set = set()

    # 3. 遍历字典中的每一个工作表数据
    for sheet_name, df_sheet in all_sheets_data.items():
        print(f"  -> 正在处理工作表: '{sheet_name}'")

        # 4. 确保当前工作表包含必需的列，增加代码的健壮性
        if '栅格x坐标' in df_sheet.columns and '栅格y坐标' in df_sheet.columns:
            # 从当前工作表中提取坐标
            sheet_coords = set(zip(df_sheet['栅格x坐标'], df_sheet['栅格y坐标']))

            # 5. 使用 update 方法将当前工作表的坐标集合并到总集合中
            all_bad_zones_set.update(sheet_coords)
        else:
            print(f"  -> 警告: 工作表 '{sheet_name}' 缺少坐标列，已跳过。")

    print(f"✅ 所有不良区域加载完毕，共计 {len(all_bad_zones_set)} 个独立栅格。")
    # 6. 返回包含所有工作表坐标的总集合
    return all_bad_zones_set


def load_path_data(file_name, data_folder='data'):
    """读取指定的路径文件"""
    print(f"正在加载路径文件: {file_name}...")
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_excel(file_path)
    # 统一列名，方便处理
    if len(df.columns) == 4:
        df.columns = ['id', 'x', 'y', 'heading']
    elif len(df.columns) == 3:
        df.columns = ['id', 'x', 'y']
    else:
        df.columns = ['x', 'y']
    return df


def load_points_info(data_folder='data'):
    """
    加载各点位置信息.xlsx文件。

    :param data_folder: 包含 '各点位置信息.xlsx' 的文件夹路径。
    :return: 包含点位信息的 pandas DataFrame，如果文件不存在则返回 None。
    """
    file_path = os.path.join(data_folder, '各点位位置信息.xlsx')
    if not os.path.exists(file_path):
        print(f"⚠️  警告: 点位信息文件未找到: {file_path}")
        return None
    try:
        df = pd.read_excel(file_path)
        print(f"✅ 成功加载点位信息: {file_path} ({len(df)}个点)")
        return df
    except Exception as e:
        print(f"❌ 错误: 读取点位信息文件失败: {e}")
        return None