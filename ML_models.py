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

    def __init__(self, scaler, folder_to_data="data", models_saved="models_save/ml_models"):

        self.scaler = scaler

        if not os.path.exists(models_saved):
            os.makedirs(models_saved)
        self.models_saved = models_saved

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

    def regressor_selection(self, regr):
        if regr == "SentinelDEM":
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

        elif regr == "onlyDEM":
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

        elif regr == "Sentinel":
            regressors = self.base_train[self.bands_multi]
            # Interpolate only one instance
            regressors = regressors.interpolate()
            X_train = np.array(regressors)
            regressors_test = self.base_test[self.bands_multi]
            X_test = np.array(regressors_test)

        else:
            print("Problems!!")
            sys.exit()
        return X_train, X_test


    def label_scaling(self, train, test):
        scale = self.scaler
        y_train = np.array(train)
        y_test = np.array(test)
        y_train_scaled = scale.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = scale.transform(y_test.reshape(-1, 1))
        y_train_scaled = y_train_scaled.reshape(y_train_scaled.shape[0],)
        y_test_scaled = y_test_scaled.reshape(y_test_scaled.shape[0],)
        return y_train_scaled, y_test_scaled


    def train_model(self, model, suffix, regr_suff="Sentinel"):

        X_train, _ = self.regressor_selection(regr_suff)

        for col in tqdm(self.chemicals, desc=f"Training {suffix} model with {regr_suff}.."):
            y_train, _ = self.label_scaling(self.base_train[col], self.base_test[col])

            model.fit(X_train, y_train)
            dump_path = f'{self.models_saved}/{suffix}_{regr_suff}_{col}.joblib'
            dump(model, dump_path)


    def test_models(self, models_to_test="models_save/ml_models", output="models_tested.csv"):

        if not os.path.exists("results"):
            os.makedirs("results")

        all_models = [f for f in os.listdir(models_to_test) if os.path.isfile(os.path.join(models_to_test, f))]
        performance = pd.DataFrame(columns = ["Char", "Model", "Regressors","MAE", "RMSE", "R2"])
        for model in tqdm(all_models, desc=f"Testing all models in {models_to_test} ..."):
            suffs = model.split("_")
            model_suff = suffs[0]
            regr_suff = suffs[1]
            chem = suffs[2][:-7]

            _, X_test = self.regressor_selection(regr_suff)

            _, y_test = self.label_scaling(self.base_train[chem], self.base_test[chem])
            model = load(f'{models_to_test}/{model_suff}_{regr_suff}_{chem}.joblib')

            pred = model.predict(X_test)
            mae_test = np.mean(abs(y_test - pred))
            rmse_test = np.sqrt(np.mean((y_test - pred)**2))
            r2_test = r2_score(y_test, pred)
            performance = pd.concat([performance, pd.DataFrame({"Char": [chem],
                                                                "Model": [model_suff],
                                                                "Regressors": [regr_suff],
                                                                "MAE": [mae_test],
                                                                "RMSE" : [rmse_test],
                                                                "R2": [r2_test]})])

        performance = performance.reset_index(drop = True)
        if os.path.exists(output):
            output = output[:-4] + "_(1).csv"
            print("A file with the same name already exists, pay attention!")
        performance.to_csv(output, index=False)
        # return performance

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

    def feat_importance(self, model_suff, models="models_save/ml_models", output="plot/feat_importance"):

        if not os.path.exists(output):
            os.makedirs(output)

        for chem in tqdm(self.chemicals, desc="Generating feature importance plots.."):

            model_rf = load(f'{models}/{model_suff}_SentinelDEM_{chem}.joblib')
            fig, ax = plt.subplots(1,1,figsize = (8,4))
            plt.bar(self.bands_multi ,model_rf.feature_importances_[:21], label = "Multispectral")
            plt.bar(self.geo_features ,model_rf.feature_importances_[21:], label = "Geomorphological")
            plt.title(f"{chem}")
            plt.legend()
            plt.xticks(rotation = 45, ha = "right", fontsize = 8)
            plt.grid(alpha = .2)
            plt.savefig(f"{output}/{model_suff}_SentinelDEM_{chem}.png", bbox_inches = "tight")

        band_imp = []
        geo_imp = []
        for chem in self.chemicals:
            model_rf = load(f'{self.models_saved}/{model_suff}_SentinelDEM_{chem}.joblib')

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
        plt.savefig(f"{output}/{model_suff}_Stacked_barplot.png", bbox_inches = "tight")

    def latex_generator(self, model_suff, models="models_save/ml_models"):
        diz_all = {}
        diz_all["Features"] = self.bands_multi + self.geo_features
        for chem in self.chemicals:
            model_rf = load(f'{models}/{model_suff}_sentinelDEM_{chem}.joblib')
            diz_all[chem] = model_rf.feature_importances_.tolist()
        diz_all_df = pd.DataFrame(diz_all)
        latex_table = diz_all_df.to_latex(float_format="%.2f", bold_rows = True, index = False)
        print(latex_table, file=open("feat_importance.txt", "a"))
