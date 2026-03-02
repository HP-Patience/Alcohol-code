import pandas as pd
import numpy as np
from scipy import interpolate
import os


def interpolate_missing_data(df):
    """
    对眼动数据进行缺失值插值处理（确保结果大于等于0）

    参数:
    df -- 包含所有眼动数据的DataFrame


    返回:
    插值处理后的DataFrame
    """
    # 定义需要插值的指标列
    metrics = ['AOI转换次数', '静态注视熵(SGE)', '眼跳注视熵(GTE)']

    # 创建结果副本
    interpolated_df = df.copy()

    # 将0值视为缺失值（替换为NaN）
    for metric in metrics:
        interpolated_df[metric] = interpolated_df[metric].replace(0, np.nan)

    # 按被试者、阶段分组
    grouped = interpolated_df.groupby(['被试者', '阶段'])

    # 遍历每个分组
    for (participant, phase), group_data in grouped:
        # 按天数排序
        group_data = group_data.sort_values('天数')

        # 获取组索引
        group_index = group_data.index

        # 检查是否有足够的数据点进行插值
        if len(group_data) < 2:
            continue

        # 对每个指标进行插值
        for metric in metrics:
            # 提取该指标的数值
            values = group_data[metric].values

            # 检查是否有缺失值（包括0值）
            if np.isnan(values).any():
                # 创建有效索引（非NaN值）
                valid_idx = np.where(~np.isnan(values))[0]
                valid_values = values[valid_idx]

                # 如果有效点少于2个，无法插值
                if len(valid_idx) < 2:
                    continue

                try:
                    # 创建插值函数（允许外推）
                    interp_func = interpolate.interp1d(
                        valid_idx,
                        valid_values,
                        kind='linear',
                        fill_value="extrapolate"
                    )

                    # 生成所有索引的插值
                    all_idx = np.arange(len(values))
                    interpolated_values = interp_func(all_idx)

                    # 确保插值结果大于等于0
                    interpolated_values = np.maximum(interpolated_values, 0.0)

                    # 更新DataFrame中的值
                    for i, idx in enumerate(group_index):
                        interpolated_df.at[idx, metric] = interpolated_values[i]

                except:
                    continue

    return interpolated_df


# 主程序
if __name__ == "__main__":
    # 1. 加载原始数据
    input_file = "眼动数据预处理文件.xlsx"
    print(f"正在加载数据: {input_file}")

    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 - {input_file}")
        exit()

    original_df = pd.read_excel(input_file)
    print(f"原始数据记录数: {len(original_df)}")

    # # === 新增：输出缺失值占比 ===
    metrics = ['AOI转换次数', '静态注视熵(SGE)', '眼跳注视熵(GTE)']
    for metric in metrics:
        # 这里把0也当作缺失
        missing_count = ((original_df[metric] == 0) | (original_df[metric].isna())).sum()
        missing_ratio = missing_count / len(original_df) * 100
        print(f"{metric} 缺失值占比: {missing_ratio:.2f}%")

    # 分阶段输出缺失比例
    # metrics = ['AOI转换次数', '静态注视熵(SGE)', '眼跳注视熵(GTE)']
    turn_stages = ['第1次转弯', '第2次转弯', '第3次转弯', '第4次转弯']

    print("=== 缺失值占比 ===")
    for metric in metrics:
        # 先算四次转弯平均
        ratios = []
        for turn in turn_stages:
            values = original_df.loc[original_df['阶段'] == turn, metric]
            if len(values) == 0:
                continue
            missing_count = ((values == 0) | (values.isna())).sum()
            missing_ratio = missing_count / len(values) * 100
            ratios.append(missing_ratio)
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"{metric} 阶段:转弯阶段 缺失值占比: {avg_ratio:.2f}%")
        else:
            print(f"{metric} 四次转弯没有数据")

        # 再算其他阶段单独缺失率
        other_stages = original_df['阶段'].unique()
        # 排除四次转弯
        other_stages = [s for s in other_stages if s not in turn_stages]

        for stage in other_stages:
            values = original_df.loc[original_df['阶段'] == stage, metric]
            if len(values) == 0:
                continue
            missing_count = ((values == 0) | (values.isna())).sum()
            missing_ratio = missing_count / len(values) * 100
            print(f"{metric} 阶段:{stage} 缺失值占比: {missing_ratio:.2f}%")
        print()

    print("=== 缺失率均值 ± 标准差 ===")

    for metric in metrics:
        # 四次转弯
        ratios = []
        for turn in turn_stages:
            stage_df = original_df[original_df['阶段'] == turn]
            # 按被试计算缺失率
            per_subject = stage_df.groupby('被试者')[metric].apply(
                lambda x: ((x == 0) | x.isna()).sum() / len(x) * 100
            )
            ratios.append(per_subject)
        if ratios:
            # 拼接成一个 Series
            concat_ratios = pd.concat(ratios)
            mean_val = concat_ratios.mean()
            std_val = concat_ratios.std()
            print(f"{metric} 四次转弯缺失率均值±标准差: {mean_val:.2f}% ± {std_val:.2f}%")

        # 其他阶段
        other_stages = [s for s in original_df['阶段'].unique() if s not in turn_stages]
        for stage in other_stages:
            stage_df = original_df[original_df['阶段'] == stage]
            per_subject = stage_df.groupby('被试者')[metric].apply(
                lambda x: ((x == 0) | x.isna()).sum() / len(x) * 100
            )
            mean_val = per_subject.mean()
            std_val = per_subject.std()
            print(f"{metric} 阶段:{stage} 缺失率均值±标准差: {mean_val:.2f}% ± {std_val:.2f}%")

    # 2. 执行插值处理（确保结果大于等于0）
    interpolated_df = interpolate_missing_data(original_df)

    # 3. 保存插值结果
    output_file = "眼动数据插值处理文件.xlsx"
    interpolated_df.to_excel(output_file, index=False)
    print(f"插值结果已保存至: {output_file}")

    # 4. 验证结果是否大于等于0
    metrics = ['AOI转换次数', '静态注视熵(SGE)', '眼跳注视熵(GTE)']
    for metric in metrics:
        min_value = interpolated_df[metric].min()
        print(f"{metric}最小值: {min_value}")
        if min_value < 0:
            print(f"警告: {metric}存在负值!")

    print("插值处理完成！")
