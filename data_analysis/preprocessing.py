import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

###
# Initial Preprocessing
###


###
# Tomato Data
###
def preprocess_data(data_dir="data/supplemental_datasets/SD1_dataset_tomato.xlsx"):
    # 从 Excel 文件中读取数据
    tom_og = pd.read_excel(data_dir, sheet_name="SD1_Original_Data")

    # 移除不需要的列
    tom_og = tom_og.drop(columns=["species", "panel number"])

    # 对所有特征和响应进行标准化（均值为0，标准差为1）
    scaler = StandardScaler()
    tom_og.iloc[:, 1:] = scaler.fit_transform(tom_og.iloc[:, 1:])

    # 将标准化后的数据保存为 CSV 文件
    tom_og.to_csv("./data/input/tom_imputed_scaled.csv", index=False)

    ###
    # 验证标准化后的数据是否与 SD1_Imputed_Scaled 一致
    tom_imputed_scaled = pd.read_csv("./data/input/tom_imputed_scaled.csv")
    sd1_tom_imputed_scaled = pd.read_excel("data/supplemental_datasets/SD1_dataset_tomato.xlsx", sheet_name="SD1_Imputed_Scaled")

    # 移除不需要的列
    sd1_tom_imputed_scaled = sd1_tom_imputed_scaled.drop(columns=["species", "panel number"])

    # 计算两个数据集之间的差异
    differences = tom_imputed_scaled.iloc[:, 1:] - sd1_tom_imputed_scaled.iloc[:, 1:]

    # 绘制差异的直方图
    differences_melted = differences.melt(var_name="metabolite", value_name="difference")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=differences_melted, x="difference", bins=100)
    plt.title("Differences between Scaled Datasets (Tomato)")
    plt.show()

    # 尽管差异很小，但为了尽可能保持可重复性，我们将使用 SD1_dataset_tomato.xlsx 文件中的标准化数据
    sd1_tom_imputed_scaled.to_csv("./data/input/tom_imputed_scaled.csv", index=False)

    ###
    # Blueberry Data
    ###

    # 从 Excel 文件中读取数据
    bb_og = pd.read_excel("data/supplemental_datasets/SD2_dataset_blueberry.xlsx", sheet_name="SD2_Original_Data", na_values="NA")

    # 移除不需要的列
    bb_og = bb_og.drop(columns=["species"])

    # 将 id 列名改为与番茄数据集一致
    bb_og = bb_og.rename(columns={bb_og.columns[0]: "id"})

    # 以下代谢物在某些样本中存在缺失值
    mets_with_missing_data = [
        "firmness", "citric", "fructose", "glucose", "sucrose", "Limonene", "Nerylacetone"
    ]

    # 用非缺失样本的均值填补缺失值
    means = bb_og[mets_with_missing_data].mean()

    # 对每个有缺失值的代谢物，用均值填补缺失值
    for met in mets_with_missing_data:
        bb_og[met].fillna(means[met], inplace=True)

    # 将填补后的数据保存为 CSV 文件
    bb_og.to_csv("./data/input/bb_imputed.csv", index=False)

    # 对所有特征和响应进行标准化（均值为0，标准差为1）
    bb_og.iloc[:, 1:] = scaler.fit_transform(bb_og.iloc[:, 1:])

    # 将标准化后的数据保存为 CSV 文件
    bb_og.to_csv("./data/input/bb_imputed_scaled.csv", index=False)

    ###
    # 验证标准化后的数据是否与 SD2_Imputed_Scaled 一致
    bb_imputed_scaled = pd.read_csv("./data/input/bb_imputed_scaled.csv")
    sd2_bb_imputed_scaled = pd.read_excel("data/supplemental_datasets/SD2_dataset_blueberry.xlsx", sheet_name="SD2_Imputed_Scaled")

    # 移除不需要的列
    sd2_bb_imputed_scaled = sd2_bb_imputed_scaled.drop(columns=["species"])
    sd2_bb_imputed_scaled = sd2_bb_imputed_scaled.rename(columns={sd2_bb_imputed_scaled.columns[0]: "id"})

    # 计算两个数据集之间的差异
    differences = bb_imputed_scaled.iloc[:, 1:] - sd2_bb_imputed_scaled.iloc[:, 1:]

    # 绘制差异的直方图
    differences_melted = differences.melt(var_name="metabolite", value_name="difference")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=differences_melted, x="difference", bins=100)
    plt.title("Differences between Scaled Datasets (Blueberry)")
    plt.show()

    # 尽管差异很小，但为了尽可能保持可重复性，我们将使用 SD2_dataset_blueberry.xlsx 文件中的标准化数据
    sd2_bb_imputed_scaled.to_csv("./data/input/bb_imputed_scaled.csv", index=False)