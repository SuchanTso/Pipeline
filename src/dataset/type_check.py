import pandas as pd

import argparse
import json

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="data/whole_Haishu.xlsx",
        help="path of the data file",
    )
    parser.add_argument(
        "-s",
        "--sheet",
        type=str,
        default="节点",
        help="sheet of data file",
    )
    parser.add_argument(
        "-c",
        "--column",
        type=str,
        default="材质",
        help="column to check",
    )
    
    parser.add_argument(
        "-m",
        "--map",
        type=str,
        default="data/predefine/node_epa.json",
        help="path of the map file",
    )
    
    return parser.parse_args()
def create_column_mapping(file_path, sheet_name=0, column='I' , map_file = None):
    """
    读取Excel文件，创建指定列的映射关系
    
    参数:
    file_path: Excel文件路径
    sheet_name: 工作表名称或索引(默认第一个工作表)
    column: 要处理的列字母(默认'I')
    
    返回:
    映射字典和打印映射关系
    """
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # 检查指定列是否存在
    # print(f"df.columns: {df.columns}")
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于工作表中")
    
    # 提取列数据并去重
    unique_values = df[column].dropna().astype(str).unique()
    
    # 创建映射字典
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    
    # 如果提供了 map_file，则加载并进行 key 验证
    if map_file:
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"无法找到 map_file: {map_file}")
        
        missing_keys = [key for key in mapping if key not in json_data]
        
        if missing_keys:
            error_msg = f"{map_file} 中, {column}列存在[{missing_keys}]未定义"
            print(error_msg)
            exit(1)
            raise KeyError(error_msg)
    
    # 打印映射关系
    # print("映射关系:")
    # for key, value in mapping.items():
    #     print(f"{key}: {value}")
    
    return mapping

# 示例用法
if __name__ == "__main__":
    # 替换为你的Excel文件路径
    args = parse_args()
    excel_file =args.file
    
    # 可选: 指定工作表名称(如果不是第一个工作表)
    # sheet = "Sheet2"
    
    try:
        mapping_result = create_column_mapping(excel_file,sheet_name=args.sheet, column=args.column , map_file=args.map)
        # 如果需要指定工作表: create_column_mapping(excel_file, sheet_name="Sheet2")
        print(f"checked {args.column} in {args.sheet} of {excel_file} successfully.")
    except Exception as e:
        print(f"发生错误: {e}")