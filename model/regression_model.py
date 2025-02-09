from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso,BayesianRidge, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
# 基础模型集合
class Regression_Models:
    def __init__(self):
        self.MODELS= {
        "Bayes": BayesianRidge(
            alpha_init=1.0,  # 初始化 alpha（权重先验的精度）
            lambda_init=1.0,  # 初始化 lambda（噪声先验的精度）
            alpha_1=1e-6,    # alpha 的超先验参数
            alpha_2=1e-6,    # alpha 的超先验参数
            lambda_1=1e-6,   # lambda 的超先验参数
            lambda_2=1e-6    # lambda 的超先验参数
        ),
        "Lasso": Lasso(
            alpha=0.1,       # 正则化强度
            max_iter=1000    # 最大迭代次数
        ),
        "Ridge": Ridge(
                alpha=1.0        # 正则化强度
        ),
        "Gaussian": GaussianProcessRegressor(),
        "RKHS": KernelRidge(kernel='precomputed'),
        "GradientBoosting": GradientBoostingRegressor(),
        "RandomForest": RandomForestRegressor( 
            n_estimators=1000,  # 增加树的数量
            max_features='sqrt',  # 使用平方根特征数量
            max_depth=20,  # 限制树的最大深度
            min_samples_split=5  # 节点分裂的最小样本数
        ),
        "SVM_linear": SVR(
            kernel='rbf',  # 使用径向基核函数
            C=10.0,  # 增加正则化强度
            epsilon=0.1  # 调整误差容忍度
        ),
        "ElasticNet": ElasticNet(
            alpha=0.1,  # 减小正则化强度
            l1_ratio=0.7  # 增加 L1 正则化比例
        ),
        "NeuralNet": MLPRegressor(
            hidden_layer_sizes=(100, 50),  # 增加隐藏层复杂度
            learning_rate_init=0.01,  # 增加初始学习率
            learning_rate='adaptive',  # 自适应学习率
            max_iter=1000,  # 增加迭代次数
            solver='adam'  # 使用 Adam 优化器
        ),
        "PLS": PLSRegression(n_components=5),
        "XGBoost": XGBRegressor(
            learning_rate=0.1,  # 降低学习率
            n_estimators=1000,  # 增加树的数量
            max_depth=10,  # 增加树的最大深度
            subsample=0.8,  # 样本采样比例
            colsample_bytree=0.8  # 特征采样比例
        )
    }
    def get_models(self):
        return self.MODELS
