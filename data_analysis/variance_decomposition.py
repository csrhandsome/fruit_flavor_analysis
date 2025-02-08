import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from statsmodels.regression.mixed_linear_model import MixedLM
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 配置参数
resdir = Path("results/fig3/")
resdir.mkdir(parents=True, exist_ok=True)
np.random.seed(100)

#%% 数据预处理函数
def preprocess_data(file_path, key_path, species='tomato'):
    # 加载数据
    df = pd.read_csv(file_path)
    key = pd.read_csv(key_path)
    
    # 生成唯一ID
    df['id'] = [f"{i}_{x}" for i, x in enumerate(df['id'])] if species=='tomato' \
              else [f"bb{i}_{x}" for i, x in enumerate(df['id'])]
    
    # 分离特征
    sensory_cols = ["liking","sweetness", "sour", "umami", "intensity"] if species=='tomato' \
                 else ["liking","sweetness", "sour", "intensity"]
    metab_cols = [c for c in df.columns if c not in sensory_cols + ['id']]
    
    # 合并关键信息
    idx_df = pd.merge(pd.DataFrame({'Metabolite': metab_cols}), key, on='Metabolite')
    return df, sensory_cols, idx_df

#%% 遗传相似矩阵构建函数
def build_genetic_matrix(data, metabolites):
    """构建遗传相似性矩阵"""
    W = data[metabolites].values
    dist_matrix = squareform(pdist(W, metric='euclidean'))
    G = 1 - dist_matrix / np.nanmax(dist_matrix)
    np.fill_diagonal(G, G.diagonal() + 0.001)  # 确保正定性
    return G

#%% 混合模型方差分析
def variance_component_analysis(df, sensory_trait, genetic_matrices):
    """运行混合效应模型并提取方差成分"""
    # 准备数据
    model_df = pd.DataFrame({
        'y': df[sensory_trait],
        'genotype': df['id']
    }).dropna()
    
    # 构建模型公式
    exog_vars = []
    group_vars = []
    var_names = []
    
    # 添加每个遗传矩阵作为随机效应
    for name, G in genetic_matrices.items():
        # 对齐矩阵与数据
        idx = model_df['genotype'].index
        sub_G = G[idx.values, :][:, idx.values]
        
        # 转换为方差组件
        exog_vars.append(sub_G)
        group_vars.append(np.ones(len(model_df)))
        var_names.append(name)
    
    # 拟合模型 
    try:
        model = MixedLM(model_df['y'], exog=np.ones(len(model_df)), 
                       groups=group_vars, exog_vc=exog_vars)
        result = model.fit(reml=True)
        
        # 提取方差成分
        vc = result.vcomp
        total_var = sum(vc.values())
        return {k: v/total_var*100 for k, v in vc.items()}
    except Exception as e:
        print(f"Error analyzing {sensory_trait}: {str(e)}")
        return None

#%% 主流程
def main_analysis(species='tomato'):
    # 加载数据
    if species == 'tomato':
        df, sensory_cols, idx_df = preprocess_data(
            "data/input/tom_imputed_scaled.csv",
            "data/input/tom_metabolites_clusters_key.csv",
            species
        )
        metab_groups = {
            'Acid/Sugar': ['citric_acid', 'malic_acid', 'glucose', 'fructose'],
            'AA_derived': [...],
            # 补充其他代谢物分组
        }
    else:
        # 蓝莓数据处理
        pass
    
    # 构建遗传矩阵
    genetic_matrices = {}
    for group, metabolites in metab_groups.items():
        genetic_matrices[group] = build_genetic_matrix(df, metabolites)
    
    # 进行方差分析
    results = []
    for trait in sensory_cols:
        vc = variance_component_analysis(df, trait, genetic_matrices)
        if vc:
            for k, v in vc.items():
                results.append({
                    'species': species,
                    'trait': trait,
                    'component': k,
                    'variance_percent': v
                })
    
    return pd.DataFrame(results)

#%% 执行分析
tomato_results = main_analysis('tomato')
blueberry_results = main_analysis('blueberry')

#%% 结果可视化
def plot_results(df, species):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='trait', y='variance_percent', hue='component', data=df)
    plt.title(f'Variance Components - {species.capitalize()}')
    plt.ylabel('Variance Explained (%)')
    plt.savefig(resdir/f'varcomp_{species}.png')

plot_results(tomato_results, 'tomato')
plot_results(blueberry_results, 'blueberry')

#%% 保存结果
full_results = pd.concat([tomato_results, blueberry_results])
full_results.to_csv(resdir/'varcomp_results.csv', index=False)