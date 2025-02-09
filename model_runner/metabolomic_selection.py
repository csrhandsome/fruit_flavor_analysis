import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import BayesianRidge, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, RepeatedKFold
# 配置参数
N_EXPERIMENTS = 10# 实验次数
N_FOLDS = 10# 交叉验证折数
# 相当于进行来N_EXPERIMENTS次实验，每次实验都会进行N_FOLDS次交叉验证 
# 所以总的实验次数是N_EXPERIMENTS*N_FOLDS
N_MODELS = 6# 模型数量
N_TRAITS = 5
N_METABOLITES = 68
RESULTS_DIR = Path("./results/fig4/a.panel/")
MODELS= {
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
# 初始化目录结构
run_id = time.strftime("tom_MS_run_%y%m%d_%I%M%p")
experiment_dirs = [RESULTS_DIR/run_id/f"experiment{i}" for i in range(1, N_EXPERIMENTS+1)]
for d in experiment_dirs:
    d.mkdir(parents=True, exist_ok=True)
    (d/"tmp").mkdir(exist_ok=True)
    (d/"accuracies").mkdir(exist_ok=True)
    (d/"mse").mkdir(exist_ok=True)
    (d/"feature_importance").mkdir(exist_ok=True)

# 日志系统
class ExperimentLogger:
    def __init__(self, path):
        self.path = path
        
    def log(self, message):
        with open(self.path, "a") as f:
            f.write(f"{time.ctime()}\t{message}\n")

# 数据预处理
def load_data(experiment):
    df = pd.read_csv("./data/input/tom_imputed_scaled.csv")
    df = df.sample(n=200, random_state=experiment)
    return df


# 核心训练函数
def train_model(model_name, X_train, y_train, X_test, y_test):
    try:
        model = MODELS[model_name]
        # print(f"Training {model_name}...")
        # model = MODELS[model_name].set_params(**{
        #     'random_state': np.random.randint(1e4)  # 确保随机性控制
        # })
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if np.isnan(pred).any():
            print(f"{model_name}预测结果包含NaN值")
        
        # 特征提取逻辑
        if isinstance(model, SVR) and model.kernel == 'linear':
            # 特殊处理SVR线性核
            if model.coef_.ndim == 2 and model.coef_.shape[0] == 1:
                features = model.coef_.flatten()
            else:
                features = model.coef_.ravel()
        elif hasattr(model, 'feature_importances_'):
            features = model.feature_importances_
        elif hasattr(model, 'coef_'):
            features = model.coef_.ravel()
        else:
            features = np.zeros(X_train.shape[1])
        
        # 强制维度对齐
        features = np.resize(features, X_train.shape[1])
        
        # 添加特征长度验证
        expected_features = X_train.shape[1]  # 应为68
        if len(features) != expected_features:
            raise ValueError(
                f"特征维度不匹配！预期{expected_features}，实际{len(features)}。"
                f"模型类型: {type(model).__name__}，参数: {model.get_params()}"
            )
        # print(f'accuracy is {r2_score(y_test, pred)}')
        # print(f'mse is {mean_squared_error(y_test, pred)}')
        # print(f'features is {features}')
        return {
            "accuracy": float(r2_score(y_test, pred)),
            "mse": float(mean_squared_error(y_test, pred)),
            "features": features
        }
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        return None

# 实验执行函数
def run_experiment(experiment):
    # logger = ExperimentLogger(RESULTS_DIR/run_id/f"experiment{experiment}"/"log.txt")
    # logger.log(f"Starting experiment {experiment}")
    # 抽取不同的样本
    data = load_data(experiment)
    results = {trait: [] for trait in range(N_TRAITS)}
    
    # 把数据集分成k份(nfold次) 这里shuffle是随机分的
    # 每次取其中一份作为测试集,其余k-1份作为训练集
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=experiment)
    
    for trait in range(N_TRAITS):
        X = data.iloc[:, 6:74].values# 特征
        y = data.iloc[:, trait+1].values# 不同的标签
        # 安全并行执行
        fold_results = []
        for fold, (train, test) in enumerate(kfold.split(X, y)):# train和test是索引
            fold_start = time.time()
            # 并行执行train_model([model_name1, model_name2, ...]),因此batch_results返回了结果的list
            batch_results = Parallel(n_jobs=-1)(
                delayed(train_model)(
                    model_name, 
                    X[train], y[train], 
                    X[test], y[test]
                ) for model_name in MODELS
            )
            
            # 过滤无效结果
            valid_results = [res for res in batch_results if res is not None]
            
            # 如果没有过滤掉
            if len(valid_results) == len(MODELS):
                fold_results.extend(valid_results)
            else:# 如果过滤了
                print(f"Experiment {experiment} trait {trait} fold {fold} has missing results")
            
            # logger.log(f"Finished fold {fold} in {time.time()-fold_start:.2f}s")
        
        # 结果整合
        if fold_results:
            for res in fold_results:
                '''for key,value in res.items():
                    if key != "features":
                        print(f"{key}:{value}")
                    else:
                        print(f"{key}: {np.mean(value)}")'''
            results[trait] = {
                "accuracies": [res["accuracy"] for res in fold_results],
                "mse": [res["mse"] for res in fold_results],
                #"features": np.mean([res["features"] for res in fold_results])
            }
            for arr in results[trait]:# 这样遍历的是字典的key 真无语 遍历字典切记要用.items()
                # print(f"{arr} is: {results[trait][arr]}")
                # print(f"{arr} length is: {len(results[trait][arr])}")
                df = pd.DataFrame(results[trait])
                os.makedirs(f"{RESULTS_DIR}/{run_id}/experiment{experiment}", exist_ok=True)
                df.to_csv(f"{RESULTS_DIR}/{run_id}/experiment{experiment}/trait_{trait}_results.csv", index=False)
    # logger.log(f"Completed experiment {experiment}")
    return results

# 主执行流程
if __name__ == "__main__":
    # 并行run_experiment([0,....,N_EXPERIMENTS-1]) 相当于是并行的for循环
    # 每一个experiment都随机抽取了200个样本
    Parallel(n_jobs=-1)(delayed(run_experiment)(exp) for exp in range(N_EXPERIMENTS))
    # 结果汇总分析
    final_results = {}
    for exp in range(N_EXPERIMENTS):# exp是实验次数
        exp_results = {}
        for trait in range(N_TRAITS):# trait是回归的标签的索引 
            df = pd.read_csv(RESULTS_DIR/run_id/f"experiment{exp}"/f"trait_{trait}_results.csv")
            exp_results[trait] = {
                "mean_accuracy": df["accuracies"].mean(),
                "mean_mse": df["mse"].mean()
            }
        final_results[exp] = exp_results
    
    # 生成最终报告
    summary = pd.DataFrame.from_dict({
        (exp, trait): final_results[exp][trait]
        for exp in final_results
        for trait in final_results[exp]
    }).T
    summary.to_csv(RESULTS_DIR/run_id/"final_summary.csv")