import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models
# from sklearn.ensemble import RandomForestRegressor
# from sklearn import svm
# from sklearn.ensemble import GradientBoostingRegressor

# Metrics
from sklearn.metrics import r2_score

# Miscellaneous
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# change scaler

import os
import sys

from joblib import dump, load

from tqdm import tqdm

class ml_models():

    def __init__(self, scaler, folder_to_data="data", output="results/ml_models", save_model="models_save/ml_models"):

        self.scaler = scaler
        if not os.path.exists(output):
            os.makedirs(output)
        self.output = output
        if not os.path.exists(save_model):
            os.makedirs(save_model)
        self.save_model = save_model

        self.base_train = pd.read_csv(f"{folder_to_data}/base_train.csv")
        self.base_val = pd.read_csv(f"{folder_to_data}/base_val.csv")
        self.base_test = pd.read_csv(f"{folder_to_data}/base_test.csv")

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


    def scaling(self, train, test):
        scale = self.scaler
        y_train = np.array(train)
        y_test = np.array(test)
        y_train_scaled = scale.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = scale.transform(y_test.reshape(-1, 1))
        y_train_scaled = y_train_scaled.reshape(y_train_scaled.shape[0],)
        y_test_scaled = y_test_scaled.reshape(y_test_scaled.shape[0],)
        return y_train_scaled, y_test_scaled

    def train_model(self, model, suffix, regressors="Sentinel"):
        if regressors == "SentinelDEM":
            regressors = self.base_train[self.bands_multi + self.geo_features]
            regressors = regressors.interpolate()
            X_train = np.array(regressors)
            regressors_test = self.base_test[self.bands_multi + self.geo_features]
            regressors_test = regressors_test.interpolate()
            X_test = np.array(regressors_test)
            # personalized scaling
            scale = self.scaler
            X_train = scale.fit_transform(X_train)
            X_test = scale.transform(X_test)
            suf_dem = "SentinelDEM"
        elif regressors == "onlyDEM":
            regressors = self.base_train[self.geo_features]
            regressors = regressors.interpolate()
            X_train = np.array(regressors)
            regressors_test = self.base_test[self.geo_features]
            regressors_test = regressors_test.interpolate()
            X_test = np.array(regressors_test)
            # personalized scaling
            scale = self.scaler
            X_train = scale.fit_transform(X_train)
            X_test = scale.transform(X_test)
            suf_dem = "onlyDEM"
        elif regressors == "Sentinel":
            regressors = self.base_train[self.bands_multi]
            # Interpolate only one instance
            regressors = regressors.interpolate()
            X_train = np.array(regressors)
            regressors_test = self.base_test[self.bands_multi]
            X_test = np.array(regressors_test)
            suf_dem = "Sentinel"
        else:
            print("Problemi!!")
            sys.exit()

        performance = pd.DataFrame(columns = ["Char", "MAE", "RMSE", "R2"])

        for col in tqdm(self.chemicals, desc=f"Training {suffix} model with {suf_dem}.."):
            y_train, y_test = self.scaling(self.base_train[col], self.base_test[col])

            model.fit(X_train, y_train)
            dump_path = f'{self.save_model}/{suffix}_{suf_dem}_{col}.joblib'
            dump(model, dump_path)
            pred = model.predict(X_test)
            mae_test = np.mean(abs(y_test - pred))
            rmse_test = np.sqrt(np.mean((y_test - pred)**2))
            r2_test = r2_score(y_test, pred)
            performance = pd.concat([performance, pd.DataFrame({"Char": [col], "MAE": [mae_test], "RMSE" : [rmse_test], "R2": [r2_test]})])

        performance = performance.reset_index(drop = True)
        performance["Model"] = suffix
        performance["Regressors"] = suf_dem
        saving_results_path = f"{self.output}/{suffix}_{suf_dem}.csv"
        if os.path.exists(saving_results_path):
            saving_results_path = saving_results_path[:-4] + "_(1).csv"
        performance.to_csv(saving_results_path, index=False)
        return performance


    def test_models(self, models_to_test="models_save/ml_models", output="results/models_tested.csv"):

        all_models = [f for f in os.listdir(models_to_test) if os.path.isfile(os.path.join(models_to_test, f))]
        performance = pd.DataFrame(columns = ["Char", "Model", "Regressors","MAE", "RMSE", "R2"])
        for model in all_models:
            suffs = model.split("_")
            model_suff = suffs[0]
            DEM = suffs[1]
            chem = suffs[2][:-7]

            if DEM == "SentinelDEM":
                regressors = self.base_train[self.bands_multi + self.geo_features]
                regressors = regressors.interpolate()
                X_train = np.array(regressors)
                regressors_test = self.base_test[self.bands_multi + self.geo_features]
                regressors_test = regressors_test.interpolate()
                X_test = np.array(regressors_test)
                # personalized scaling
                scale = self.scaler
                X_train = scale.fit_transform(X_train)
                X_test = scale.transform(X_test)
                suf_dem = "SentinelDEM"

            elif DEM == "onlyDEM":
                regressors = self.base_train[self.geo_features]
                regressors = regressors.interpolate()
                X_train = np.array(regressors)
                regressors_test = self.base_test[self.geo_features]
                regressors_test = regressors_test.interpolate()
                X_test = np.array(regressors_test)
                # personalized scaling
                scale = self.scaler
                X_train = scale.fit_transform(X_train)
                X_test = scale.transform(X_test)
                suf_dem = "onlyDEM"
            else:
                regressors = self.base_train[self.bands_multi]
                # Interpolate only one instance
                regressors = regressors.interpolate()
                X_train = np.array(regressors)
                regressors_test = self.base_test[self.bands_multi]
                X_test = np.array(regressors_test)
                suf_dem = "Sentinel"

            y_train, y_test = self.scaling(self.base_train[chem], self.base_test[chem])
            model = load(f'{models_to_test}/{model_suff}_{suf_dem}_{chem}.joblib')

            pred = model.predict(X_test)
            mae_test = np.mean(abs(y_test - pred))
            rmse_test = np.sqrt(np.mean((y_test - pred)**2))
            r2_test = r2_score(y_test, pred)
            performance = pd.concat([performance, pd.DataFrame({"Char": [chem],
                                                                "Model": [model_suff],
                                                                "Regressors": [suf_dem],
                                                                "MAE": [mae_test],
                                                                "RMSE" : [rmse_test],
                                                                "R2": [r2_test]})])

        performance = performance.reset_index(drop = True)
        if os.path.exists(output):
            output = output[:-4] + "_(1).csv"
            print("A file with the same name already exists, pay attention!")
        performance.to_csv(output, index=False)
        return performance

    def wrap_results(self, result="results/ml_models", filename="ML_performances.csv"):
        self.output = result
        files = os.listdir(self.output)
        result_list = []
        for file in files:
            result = pd.read_csv(f"{self.output}/{file}")
            result_list.append(result)
        final = pd.concat(result_list)
        final = final.reset_index(drop=True)
        final.to_csv(filename, index=False)

    def feat_importance(self, model_suff="RF", DEM=True):

        # PER COME E' STRUTTURATA ADESSO FUNZIONA SOLO CON SENTINEL E DEM!!!!

        if DEM:
            suff_dem="DEM"
        elif DEM == "only":
            suff_dem = "only_DEM"
        else:
            suff_dem=""
        for chem in tqdm(self.chemicals, desc="Generating feature importance plots.."):

            model_rf = load(f'{self.save_model}/{model_suff}_{suff_dem}_{chem}.joblib')
            fig, ax = plt.subplots(1,1,figsize = (8,4))
            plt.bar(self.bands_multi ,model_rf.feature_importances_[:21], label = "Multispectral")
            plt.bar(self.geo_features ,model_rf.feature_importances_[21:], label = "Geomorphological")
            plt.title(f"{chem}")
            plt.legend()
            plt.xticks(rotation = 45, ha = "right", fontsize = 8)
            plt.grid(alpha = .2)
            plt.savefig(f"plot/feat_importance/{model_suff}_{suff_dem}_{chem}.png", bbox_inches = "tight")

        band_imp = []
        geo_imp = []
        for chem in self.chemicals:
            model_rf = load(f'{self.save_model}/{model_suff}_{suff_dem}_{chem}.joblib')
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
        plt.savefig(f"plot/feat_importance/{model_suff}_Stacked_barplot.png", bbox_inches = "tight")



#### RUNNING CODE ##################
