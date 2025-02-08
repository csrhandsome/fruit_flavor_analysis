from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
# 基础模型集合
class Mutiple_Models:
    def __init__(self):
        self.models = {
            "RandomForest": RandomForestRegressor(n_estimators=750, max_features=20),
            "SVM_linear": SVR(kernel='linear'),
            "ElasticNet": ElasticNet(),
            "NeuralNet": MLPRegressor(hidden_layer_sizes=(10,)),
            "PLS": PLSRegression(n_components=2),
            "XGBoost": XGBRegressor(tree_method = "hist")
        }
    def get_models(self):
        return self.models
