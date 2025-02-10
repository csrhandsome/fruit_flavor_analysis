from sklearn.kernel_ridge import KernelRidge
import numpy as np
# 基因组选择(genomic_selction):
# 使用SNP标记构建亲缘关系矩阵
# 主要使用gBLUP模型
def calculate_grm(genotype_matrix):
    """
    计算亲缘关系矩阵GRM。
    
    参数:
    genotype_matrix (numpy.ndarray): 基因型矩阵，形状为 (n_samples, n_markers)。
    
    返回:
    numpy.ndarray: 亲缘关系矩阵，形状为 (n_samples, n_samples)。
    """
    # 中心化基因型矩阵
    p = np.mean(genotype_matrix, axis=0) / 2  # 计算等位基因频率
    Z = genotype_matrix - 2 * p  # 中心化
    
    # 计算 GRM
    grm = np.dot(Z, Z.T) / (2 * np.sum(p * (1 - p)))
    return grm

def gblup(X, y, grm):
    """
    使用 gBLUP 模型进行基因组预测。
    
    参数:
    X (numpy.ndarray): 固定效应矩阵，形状为 (n_samples, n_fixed_effects)。
    y (numpy.ndarray): 表型值，形状为 (n_samples,)。
    grm (numpy.ndarray): 亲缘关系矩阵，形状为 (n_samples, n_samples)。
    
    返回:
    numpy.ndarray: 预测值。
    """
    model = KernelRidge(kernel='precomputed')
    model.fit(grm, y)
    predictions = model.predict(grm)
    return predictions
# 示例基因型矩阵（假设有 5 个个体和 10 个 SNP）
genotype_matrix = np.array([
    [0, 1, 2, 1, 0, 2, 1, 0, 1, 2],
    [1, 0, 1, 2, 1, 0, 2, 1, 0, 1],
    [2, 1, 0, 1, 2, 1, 0, 2, 1, 0],
    [0, 2, 1, 0, 1, 2, 1, 0, 2, 1],
    [1, 0, 2, 1, 0, 1, 2, 1, 0, 2]
])

# 计算 GRM
grm = calculate_grm(genotype_matrix)
print("Genomic Relationship Matrix (GRM):")
print(grm)


# 示例数据
X = np.ones((5, 1))  # 固定效应矩阵（截距）
y = np.array([10, 15, 12, 14, 13])  # 表型值

# 使用 gBLUP 进行预测
predictions = gblup(X, y, grm)
print("Predictions:", predictions)