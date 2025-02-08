import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PyWGCNA import WGCNA
from scipy.cluster.hierarchy import dendrogram
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def wgcna(species="tomato", net_type="unsigned", ntraits=5, resdir="results/fig1/"):

    # 数据加载与预处理
    if species == "tomato":
        df = pd.read_csv("data/input/tom_imputed_scaled.csv", index_col=0)
    elif species == "blueberry":
        df = pd.read_csv("data/input/bb_imputed_scaled.csv", index_col=0)
    # 提取代谢物数据 (假设前ntraits+1列为性状数据)
    metabolites = df.iloc[:, ntraits+1:]  # 保留代谢物列
    print(f"原始数据维度: {metabolites.shape}")

    # 样本聚类筛选（改进版）
    scaled_data = StandardScaler().fit_transform(metabolites)
    
    # 计算距离矩阵
    dist_matrix = distance.pdist(scaled_data, 'euclidean')
    Z = linkage(dist_matrix, 'average')
    
    # 可视化聚类结果
    plt.figure(figsize=(15, 6))
    dendrogram(Z, labels=metabolites.index, leaf_rotation=90)
    plt.title('Sample Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.savefig(f"{resdir}sample_clustering.pdf")
    plt.close()
    
    # 自动确定聚类数（基于轮廓系数）
    from sklearn.metrics import silhouette_score
    best_score = -1
    best_clusters = 2
    for n_clusters in range(2, 5):
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        score = silhouette_score(scaled_data, labels)
        if score > best_score:
            best_score = score
            best_clusters = n_clusters
    
    print(f"\n建议聚类数: {best_clusters}")
    
    # 保留最大的两个类
    cluster_labels = fcluster(Z, best_clusters, criterion='maxclust')
    unique, counts = np.unique(cluster_labels, return_counts=True)
    # 修改后的保留策略（保留最大类）
    top_clusters = [unique[np.argmax(counts)]]  # 只保留数量最多的类
    keep_samples = cluster_labels == top_clusters[0]
    
    filtered_data = metabolites.iloc[keep_samples, :]
    print(f"\n筛选后数据维度: {filtered_data.shape}")

    # 格式转换 (PyWGCNA要求genes×samples)
    expr_matrix = filtered_data.T  # 转置为(代谢物×样本)

    # 初始化WGCNA对象（使用类定义中已有的参数）
    wgcna = WGCNA(
        name=species,
        geneExp=expr_matrix,  # 确保格式为 samples×genes
        networkType=net_type,
        minModuleSize=3,
        RsquaredCut=0.6,
        save=True,
        outputPath=resdir
    )

    # 软阈值选择（移除了verbose参数）
    powers = list(range(1, 51))
    sft = wgcna.pickSoftThreshold(
        expr_matrix,
        powerVector=powers,
        networkType=net_type,
        RsquaredCut=0.6
    )
    # 格式为tuple
    # sft[0]：整数，推荐的最佳软阈值 (powerEstimate)
    # sft[1]：Pandas DataFrame，包含详细的拟合指标数据
    powerEstimate = sft[0]  # 直接获取元组第一个元素
    fitIndices = sft[1]     # 获取数据框

    print(f"推荐软阈值: {powerEstimate}")

   # 可视化调整后的正确代码
    plt.figure(figsize=(12,5))

    # 左图：Scale Free Topology Fit
    plt.subplot(121)
    plt.plot(fitIndices['Power'], 
            -np.sign(fitIndices['slope'])*fitIndices['SFT.R.sq'],
            'r-o')
    plt.axhline(0.7, color='r', linestyle='--')
    plt.xlabel('Soft Threshold')
    plt.ylabel('Scale Free Topology Fit')
    
    # 右图：Mean Connectivity
    plt.subplot(122)
    plt.plot(fitIndices['Power'], 
            fitIndices['mean(k)'],  # 使用正确的列名
            'b-o')
    plt.xlabel('Soft Threshold')
    plt.ylabel('Mean Connectivity')
    plt.show()

    # 调用 findModules
    wgcna.findModules()

    # 获取模块颜色和唯一颜色列表
    module_colors = wgcna.datExpr.var['moduleColors']
    unique_colors = np.unique(module_colors).tolist()

    # 创建颜色到数字的映射字典
    color_to_num = {color: i for i, color in enumerate(unique_colors)}

    # 根据基因在树状图中的顺序获取颜色索引
    gene_order = dendrogram(wgcna.geneTree, no_plot=True)['leaves']
    color_indices = [color_to_num[module_colors[i]] for i in gene_order]

    # 创建自定义颜色映射
    cmap = ListedColormap(unique_colors)

    # 绘制图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)

    # 绘制树状图
    dendrogram(wgcna.geneTree, ax=ax1, 
            labels=expr_matrix.columns.tolist(),
            leaf_rotation=90, leaf_font_size=8)
    ax1.set_title('Gene Clustering Dendrogram with Module Colors')

    # 绘制颜色条
    ax2.imshow([color_indices], aspect='auto', cmap=cmap, 
            vmin=0, vmax=len(unique_colors)-1)
    ax2.set_yticks([])
    ax2.set_xticks([])

    # 添加图例（直接使用颜色名称）
    patches = [mpatches.Patch(color=c, label=f"Module {i+1}") 
            for i, c in enumerate(unique_colors)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{resdir}gene_dendrogram_with_colors.pdf",format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    wgcna()