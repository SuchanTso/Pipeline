import pandas as pd
import sys
import argparse


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--part_file",
        type=str,
        help="path of the part graph file",
    )
    parser.add_argument(
        "-w",
        "--whole_file",
        type=str,
        help="path of the whole graph file",
    )
    
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="path of the part graph file",
    )
    
    return parser.parse_args()
def filter_and_save(input_a, input_b, output_path):
    # 读取表格B
    df_b = pd.read_excel(input_b, sheet_name=0, header=0)
    
    # 获取B表中的ID列表（忽略空值）
    sheet1_ids = df_b.iloc[:, 0].dropna().astype(str).unique()
    sheet2_ids = df_b.iloc[:, 1].dropna().astype(str).unique()
    
    # 读取表格A的Sheet1和Sheet2
    with pd.ExcelFile(input_a) as xls:
        df_sheet1 = pd.read_excel(xls, sheet_name=0, header=0)
        df_sheet2 = pd.read_excel(xls, sheet_name=1, header=0)
    
    # 获取ID列名称（假设表头中包含ID列）
    def find_id_col(df):
        for col in df.columns:
            if 'id' in col.lower():
                return col
        raise ValueError("无法自动识别ID列，请确保表头包含'ID'字样")
    
    id_col_sheet1 = find_id_col(df_sheet1)
    id_col_sheet2 = find_id_col(df_sheet2)
    
    # 筛选Sheet1中存在于B表的行
    filtered_sheet1 = df_sheet1[df_sheet1[id_col_sheet1].astype(str).isin(sheet1_ids)]
    
    # 筛选Sheet2中存在于B表的行
    filtered_sheet2 = df_sheet2[df_sheet2[id_col_sheet2].astype(str).isin(sheet2_ids)]
    
    # 保存结果到output文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        filtered_sheet1.to_excel(writer, sheet_name='Sheet1', index=False)
        filtered_sheet2.to_excel(writer, sheet_name='Sheet2', index=False)
    
    print(f"处理完成！结果已保存至: {output_path}")
    print(f"Sheet1 匹配行数: {len(filtered_sheet1)}")
    print(f"Sheet2 匹配行数: {len(filtered_sheet2)}")

if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("用法: python script.py <表格A路径> <表格B路径> <输出路径>")
    #     sys.exit(1)
    args = parse_args()
    input_a_path = args.whole_file
    input_b_path = args.part_file
    output_path = args.output_path
    
    filter_and_save(input_a_path, input_b_path, output_path)