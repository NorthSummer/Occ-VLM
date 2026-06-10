def calculate_average_from_file(file_path):
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 确保不是空行
                try:
                    value = float(line)
                    values.append(value)
                except ValueError:
                    print(f"警告：无法转换为浮点数的行：{line}")

    if values:
        average = sum(values) / len(values)
        print(f"平均值为: {average}")
        return average
    else:
        print("文件中没有有效的数值。")
        return None

# 使用示例
if __name__ == "__main__":
    file_path = "./visual_length_sqa3d_24v.txt"  # 替换为你的文件路径
    calculate_average_from_file(file_path)
