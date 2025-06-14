import os

def convert_breastcancer(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as fin, \
         open(output_file, 'w', newline='\n', encoding='utf-8') as fout:
        
        for line in fin:
            # 移除行尾换行符并分割字段
            line = line.rstrip('\n\r')
            parts = [part.strip() for part in line.split(',')]
            if not parts:
                continue
                
            # 验证基本格式 (至少需要ID和诊断标签)
            if len(parts) < 2:
                continue
                
            # 提取ID、诊断标签和特征
            id_part = parts[0]
            diag_char = parts[1]
            features = parts[2:]
            
            # 转换诊断标签 (M->4, B->2)
            if diag_char == 'M':
                label = '4'
            elif diag_char == 'B':
                label = '2'
            else:
                # 保留未知标签原样（DeepBoost会处理无效标签）
                label = diag_char
                
            # 构建新行: ID + 特征 + 数字标签
            # 保留所有特征值（包括"?"）
            new_line = id_part + ',' + ','.join(features) + ',' + label + '\n'
            fout.write(new_line)

if __name__ == "__main__":
    input_path = 'E:/25springcourse/MachineLearning/deepboost/testdata/wdbc.data'    # 输入文件名
    output_path = 'E:/25springcourse/MachineLearning/deepboost/testdata/wdbc_deepboost.data'  # 输出文件名
    
    convert_breastcancer(input_path, output_path)
    print(f"转换完成! 已保存至: {os.path.abspath(output_path)}")
    print(f"输入行数: {sum(1 for _ in open(input_path, 'r'))}")
    print(f"输出行数: {sum(1 for _ in open(output_path, 'r'))}")