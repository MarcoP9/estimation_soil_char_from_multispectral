import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor

# Metrics
from sklearn.metrics import r2_score

# Miscellaneous
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# change scaler

import os

from joblib import dump, load

from tqdm import tqdm

class ml_models():

    def __init__(self, path_to_data):

        self.base_train = pd.read_csv(f"{path_to_data}/base_train.csv")
        self.base_val = pd.read_csv(f"{path_to_data}/base_val.csv")
        self.base_test = pd.read_csv(f"{path_to_data}/base_test.csv")

        self.base_train = self.base_train.drop(columns="Unnamed: 0.1")
        self.base_val = self.base_train.drop(columns="Unnamed: 0")
        self.base_test = self.base_test.drop(columns="Unnamed: 0.1")

        self.chemicals = list(self.base_train.iloc[:,3:15].columns)
        self.bands_multi = [col for col in self.base_train.columns if col.startswith("Band")]
        self.bands_hyper = [col for col in self.base_train.columns if col.startswith("spc")]
        self.geo_features = ['CATCH_SLOP','TWI', 'SLOPE_LENG', 'CN_BASE', 'VDCN', 'VALLEY_DEP', 'SLOPE', 'DEM']

        self.all_columns = ["GPS_LAT", "GPS_LONG"] + self.chemicals + self.bands_multi + self.geo_features + self.bands_hyper

        self.base_train = self.base_train[self.all_columns]
        self.base_val = self.base_val[self.all_columns]
        self.base_test = self.base_test[self.all_columns]


    def robust_scaling(self, train, test):
        scale = RobustScaler()
        y_train = np.array(train)
        y_test = np.array(test)
        y_train_scaled = scale.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = scale.transform(y_test.reshape(-1, 1))
        y_train_scaled = y_train_scaled.reshape(y_train_scaled.shape[0],)
        y_test_scaled = y_test_scaled.reshape(y_test_scaled.shape[0],)
        return y_train_scaled, y_test_scaled

    def train_model(self, model, suffix, DEM=False):
        if DEM:
            regressors = self.base_train[self.bands_multi + self.geo_features]
            regressors = regressors.interpolate()
            X_train = np.array(regressors)
            regressors_test = self.base_test[self.bands_multi + self.geo_features]
            regressors_test = regressors_test.interpolate()
            X_test = np.array(regressors_test)
            # personalized scaling
            scale = RobustScaler()
            X_train = scale.fit_transform(X_train)
            X_test = scale.transform(X_test)
            suf_dem = "DEM"
        else:
            regressors = self.base_train[self.bands_multi]
            # Interpolate only one instance
            regressors = regressors.interpolate()
            X_train = np.array(regressors)
            regressors_test = self.base_test[self.bands_multi]
            X_test = np.array(regressors_test)
            suf_dem = ""

        performance = pd.DataFrame(columns = ["Char", "MAE", "RMSE", "R2"])

        for col in tqdm(self.chemicals, desc=f"Training {suffix} model with Sentinel {suf_dem}.."):
            y_train, y_test = self.robust_scaling(self.base_train[col], self.base_test[col])

            model.fit(X_train, y_train)
            dump(model, f'models_save/ml_models/{suffix}_sentinel{suf_dem}_{col}.joblib')
            pred = model.predict(X_test)
            mae_test = np.mean(abs(y_test - pred))
            rmse_test = np.sqrt(np.mean((y_test - pred)**2))
            r2_test = r2_score(y_test, pred)
            performance = pd.concat([performance, pd.DataFrame({"Char": [col], "MAE": [mae_test], "RMSE" : [rmse_test], "R2": [r2_test]})])

        performance = performance.reset_index(drop = True)
        performance["Model"] = suffix
        if DEM:
            performance["Regressors"] = "Sentinel_DEM"
        else:
            performance["Regressors"] = "Sentinel"
        performance.to_csv(f"results/ml_models/{suffix}_sentinel{suf_dem}.csv", index=False)
        return performance

    def wrap_results(self):
        files = os.listdir("results/ml_models")
        result_list = []
        for file in files:
            result = pd.read_csv(f"results/ml_models/{file}")
            result_list.append(result)
        final = pd.concat(result_list)
        final = final.reset_index(drop=True)
        final.to_csv("ML_performances.csv", index=False)

    def feat_importance(self, path_to_model):
        for chem in tqdm(self.chemicals, desc="Generating feature importance plots.."):
            model_rf = load(f'{path_to_model}/RF_sentinelDEM_{chem}.joblib')
            fig, ax = plt.subplots(1,1,figsize = (8,4))
            plt.bar(self.bands_multi ,model_rf.feature_importances_[:21], label = "Multispectral")
            plt.bar(self.geo_features ,model_rf.feature_importances_[21:], label = "Geomorphological")
            plt.title(f"{chem}")
            plt.legend()
            plt.xticks(rotation = 45, ha = "right", fontsize = 8)
            plt.grid(alpha = .2)
            plt.savefig(f"plot/feat_importance/{chem}.png", bbox_inches = "tight")

        band_imp = []
        geo_imp = []
        for chem in self.chemicals:
            model_rf = load(f'{path_to_model}/RF_sentinelDEM_{chem}.joblib')
            band_imp.append(model_rf.feature_importances_[:21].sum())
            geo_imp.append(model_rf.feature_importances_[21:].sum())
        band_imp = pd.DataFrame({"Chem": self.chemicals, "Importance": band_imp})
        band_imp["Source"] = "Bands"
        geo_imp = pd.DataFrame({"Chem": self.chemicals, "Importance": geo_imp})
        geo_imp["Source"] = "Geomorph"

        all_imp = pd.concat([band_imp, geo_imp])
        all_imp = all_imp.reset_index(drop = True)

        fig, ax = plt.subplots(1,1,figsize = (9,5))
        sns.histplot(data = all_imp, x = "Chem", weights = "Importance", hue = "Source", multiple = "stack", edgecolor = "white", shrink = .6)
        plt.title(f"Feature importance", fontsize = 16)
        plt.xticks(rotation = 45, ha = "right", fontsize = 8)
        plt.grid(alpha = .2)
        plt.xlabel(None)
        plt.ylabel("Feature Importance")
        plt.savefig(f"plot/feat_importance/Stacked_barplot.png", bbox_inches = "tight")

ml = ml_models(path_to_data="data/")
print("Data reading completed")

rf = RandomForestRegressor(max_features=9, n_estimators=30)
performance_rf_bands = ml.train_model(model=rf, suffix="RF", DEM=False)
# performance_rf_bands_DEM = ml.train_model(model=rf, suffix="RF", DEM=True)

# vector = svm.SVR(kernel = "rbf", C=10000, gamma=300)
# performance_svm_bands = ml.train_model(model=vector, suffix="SVM", DEM=False)
# vector_DEM = svm.SVR(kernel = "rbf")
# performance_svm_bands_DEM = ml.train_model(model=vector_DEM, suffix="SVM", DEM=True)
#
# gb_reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=300, min_samples_split = 2)
# performance_gb_bands = ml.train_model(model=gb_reg, suffix="GB", DEM=False)
# performance_gb_bands_DEM = ml.train_model(model=gb_reg, suffix="GB", DEM=True)

print("All models have been trained")

# ml.feat_importance(path_to_model="models_save/ml_models")

# ml.wrap_results()

print("All task have been completed")
print(ml.chemicals)
