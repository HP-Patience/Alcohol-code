import os
import pandas as pd
from tqdm import tqdm  # ✅ 进度条

root_dir = r"E:\pycharm all files\眼动数据处理\GSR\原始数据预处理\Data_GSR"
all_data = []

# 先收集所有文件
all_files = []
for group in ["A", "B"]:
    group_path = os.path.join(root_dir, group)
    for person in os.listdir(group_path):
        person_path = os.path.join(group_path, person)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.startswith("GSR") and file.endswith(".csv"):
                    all_files.append((group, person, file))

# 遍历所有文件（总体进度条）
for group, person, file in tqdm(all_files, desc="总体进度", unit="文件"):
    seq = int(file.split("-")[1].split(".")[0])  # 飞行天数
    file_path = os.path.join(root_dir, group, person, file)

    # 读取数据
    df = pd.read_csv(file_path, usecols=["data", "eventName"])

    # 找到所有事件标记行
    events_idx = df.index[df["eventName"].notna()].tolist()

    # 遍历每个事件起点
    for i in range(len(events_idx) - 1):
        start_idx = events_idx[i]
        phase_name = str(df.loc[start_idx, "eventName"]).strip()

        # 向后找下一个相同阶段的标记
        for j in range(i + 1, len(events_idx)):
            end_idx = events_idx[j]
            phase_end = str(df.loc[end_idx, "eventName"]).strip()
            if phase_end == phase_name:  # 找到匹配的结束标记
                phase_df = df.loc[start_idx:end_idx, ["data"]].copy()

                # 添加标识信息
                phase_df = phase_df.assign(
                    组别=group,
                    姓名=person,
                    飞行天数=seq,
                    阶段=phase_name
                )

                phase_df = phase_df[["组别", "姓名", "飞行天数", "阶段", "data"]]
                all_data.append(phase_df)
                break  # 找到第一个匹配就退出，避免重复

# 合并所有阶段数据
final_df = pd.concat(all_data, ignore_index=True)

# 删除缺失值
final_df = final_df.dropna()

# 保存为 Excel
final_df.to_excel("GSR_SCL_SCR_AllEvents.xlsx", index=False)

print("处理完成！结果已保存为 GSR_SCL_SCR_AllEvents.xlsx")
