import numpy as np
import pandas as pd
from pycirclize import Circos

def plot_chord_pos_neg(
    corr_df,
    cmap,
    thresholds,
    prefix,
    space=5,
    label_size=15
):

    corr_ut = corr_df.where(
        np.triu(np.ones(corr_df.shape), k=1).astype(bool)
    )

    for th in thresholds:

        # ========== 正相关 ==========
        pos_weights = corr_ut.where(corr_ut >= th, 0).fillna(0)

        if (pos_weights.values > 0).any():
            # 获取当前有正相关的指标
            current_nodes = [col for col in pos_weights.columns if pos_weights[col].sum() > 0]
            
            # 根据 cmap 分组并排序
            # 1. 创建颜色到指标的映射
            color_to_nodes = {}
            for node in current_nodes:
                color = cmap.get(node, "gray")
                if color not in color_to_nodes:
                    color_to_nodes[color] = []
                color_to_nodes[color].append(node)
            
            # 2. 根据 cmap1 中的顺序确定颜色顺序
            # 提取 cmap 中所有颜色的顺序
            cmap_order = []
            seen_colors = set()
            for node in cmap:
                color = cmap[node]
                if color not in seen_colors:
                    cmap_order.append(color)
                    seen_colors.add(color)
            
            # 3. 按照 cmap 顺序和颜色分组排序指标
            sorted_nodes = []
            for color in cmap_order:
                if color in color_to_nodes:
                    # 对于同颜色的指标，按照它们在 cmap1 中的顺序排序
                    nodes_in_color = color_to_nodes[color]
                    # 创建指标到其在 cmap 中位置的映射
                    node_positions = {node: list(cmap.keys()).index(node) for node in nodes_in_color}
                    # 按位置排序
                    nodes_in_color.sort(key=lambda x: node_positions[x])
                    sorted_nodes.extend(nodes_in_color)
            
            # 4. 重新排序矩阵
            sorted_pos_weights = pos_weights.loc[sorted_nodes, sorted_nodes]
            
            # 5. 创建排序后的颜色映射
            sorted_cmap = {node: cmap.get(node, "gray") for node in sorted_nodes}
            
            circos = Circos.chord_diagram(
                sorted_pos_weights,
                space=space,
                cmap=sorted_cmap,
                label_kws=dict(orientation="vertical", size=label_size),
                link_kws=dict(lw=0.6, ec="none")
            )
            fname = f"{prefix}_POS_th{th}.png"
            circos.savefig(fname, dpi=300)
            print("Saved:", fname)
        else:
            print(f"[{prefix}] th={th} 无正相关")

        # ========== 负相关 ==========
        neg_weights = corr_ut.where(corr_ut <= -th, 0).fillna(0).abs()

        if (neg_weights.values > 0).any():
            # 获取当前有负相关的指标
            current_nodes = [col for col in neg_weights.columns if neg_weights[col].sum() > 0]
            
            # 根据 cmap 分组并排序
            # 1. 创建颜色到指标的映射
            color_to_nodes = {}
            for node in current_nodes:
                color = cmap.get(node, "gray")
                if color not in color_to_nodes:
                    color_to_nodes[color] = []
                color_to_nodes[color].append(node)
            
            # 2. 根据 cmap1 中的顺序确定颜色顺序
            # 提取 cmap 中所有颜色的顺序
            cmap_order = []
            seen_colors = set()
            for node in cmap:
                color = cmap[node]
                if color not in seen_colors:
                    cmap_order.append(color)
                    seen_colors.add(color)
            
            # 3. 按照 cmap 顺序和颜色分组排序指标
            sorted_nodes = []
            for color in cmap_order:
                if color in color_to_nodes:
                    # 对于同颜色的指标，按照它们在 cmap1 中的顺序排序
                    nodes_in_color = color_to_nodes[color]
                    # 创建指标到其在 cmap 中位置的映射
                    node_positions = {node: list(cmap.keys()).index(node) for node in nodes_in_color}
                    # 按位置排序
                    nodes_in_color.sort(key=lambda x: node_positions[x])
                    sorted_nodes.extend(nodes_in_color)
            
            # 4. 重新排序矩阵
            sorted_neg_weights = neg_weights.loc[sorted_nodes, sorted_nodes]
            
            # 5. 创建排序后的颜色映射
            sorted_cmap = {node: cmap.get(node, "gray") for node in sorted_nodes}
            
            circos = Circos.chord_diagram(
                sorted_neg_weights,
                space=space,
                cmap=sorted_cmap,
                label_kws=dict(orientation="vertical", size=label_size),
                link_kws=dict(lw=0.6, ec="none")
            )
            fname = f"{prefix}_NEG_th{th}.png"
            circos.savefig(fname, dpi=300)
            print("Saved:", fname)
        else:
            print(f"[{prefix}] th={th} 无负相关")
