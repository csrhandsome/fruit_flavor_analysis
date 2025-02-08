import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
def metabolite_histograms(species="tomato", resdir="results/fig1/"):
    # 设置结果目录
    resdir = Path("results/fig1/")
    os.makedirs(resdir, exist_ok=True)

    # 设置随机种子
    np.random.seed(100)

    # 加载数据
    if species == "tomato":
        key = pd.read_csv("data/input/tom_metabolites_clusters_key.csv")
        tom = pd.read_csv("data/input/tom_imputed.csv")
    elif species == "blueberry":
        key = pd.read_csv("data/input/bb_metabolites_clusters_key.csv")
        tom = pd.read_csv("data/input/bb_imputed.csv")

    # 生成唯一基因型ID
    tom["id"] = [f"{i}_{x}" for i, x in enumerate(tom["id"])]

    # 定义感官属性列
    sensory = ["liking", "sweetness", "sour", "umami", "intensity"]

    # 获取代谢物列
    mets = [col for col in tom.columns if col not in sensory + ["id"]]

    # 检查元数据匹配
    missing_in_key = list(set(mets) - set(key["Metabolite"]))
    missing_in_data = list(set(key["Metabolite"]) - set(mets))
    print("Key中缺失的代谢物:", missing_in_key)
    print("数据中缺失的代谢物:", missing_in_data)

    # 合并元数据
    idx_tom = pd.merge(pd.DataFrame({"Metabolite": mets}), key, on="Metabolite")

    # 处理数据
    tmp = tom.drop(columns=sensory + ["id"])

    # 转置数据并重置索引
    tmp2 = tmp.T.reset_index()
    tmp2.columns = ["metabolite"] + list(tom["id"])

    # 合并元数据
    tmp3 = pd.merge(tmp2, idx_tom[["Metabolite", "fig1b_histogram"]], 
                    left_on="metabolite", right_on="Metabolite").drop(columns="Metabolite")

    # 数据重塑
    tmp4 = tmp3.melt(id_vars=["metabolite", "fig1b_histogram"], 
                    var_name="genotype", value_name="concentration")

    # 数据清洗
    tmp4 = tmp4.dropna()
    tmp4 = tmp4[(~tmp4["fig1b_histogram"].isin(["Acid/Sugar", "unknown"])) & 
                (tmp4["concentration"] != 0)]

    # 设置绘图风格
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update({
        "axes.facecolor": "white",
        "grid.color": "white",
        "axes.labelcolor": "black",
        "axes.labelsize": 14,
        "xtick.color": "black",
        "ytick.color": "black",
    })

    # 创建绘图画布
    plt.figure(figsize=(6, 2.5))

    # 创建小提琴图
    ax = sns.violinplot(
        data=tmp4,
        x="fig1b_histogram",
        y=np.log10(tmp4["concentration"]),
        hue="fig1b_histogram",  # 新增hue参数
        palette=["#9999ffff", "#74a9ccff", "#ffcc99ff", "#e19582ff"],  # 颜色数量与类别匹配
        linewidth=0.5,
        dodge=False,  # 防止自动分组
        legend=False  # 隐藏图例
    )

    # 设置坐标轴
    plt.ylim(-3.25, 3.25)
    plt.yticks(np.arange(-3, 4), [f"{x}" for x in np.arange(-3, 4)])
    plt.ylabel(r"$\log_{10}[ng/gfw/hr]$")
    plt.xlabel("")

    # 隐藏x轴标签
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)

    # 保存图片
    plt.tight_layout()
    plt.savefig(resdir / "fig1_panelB.pdf", format="pdf", bbox_inches="tight")
    plt.close()