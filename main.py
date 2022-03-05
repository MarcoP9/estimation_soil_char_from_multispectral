from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from ML_models import ml_models


data_path = "data"
saving_models = "models_save/ml_standard_complete"
all_results = "ML_standard_complete.csv"
output_plots = "plot/feat_importance"

if __name__ == "__main__":

    ml_standard = ml_models(scaler=StandardScaler(), folder_to_data=data_path, models_saved=saving_models)

    #### TRAINING MODELS ####

    # # RANDOM FOREST MODEL
    # rf = RandomForestRegressor(max_features=9, n_estimators=30)
    # rf_only = RandomForestRegressor(max_features=8, n_estimators=30)
    # ml_standard.train_model(model=rf, suffix="RF", regr_suff="Sentinel")
    # ml_standard.train_model(model=rf_only, suffix="RF", regr_suff="onlyDEM")
    # ml_standard.train_model(model=rf, suffix="RF", regr_suff="SentinelDEM")
    # #
    # # # SVR MODEL
    # vector = svm.SVR(kernel = "rbf", C=10000, gamma=300)
    # ml_standard.train_model(model=vector, suffix="SVR", regr_suff="Sentinel")
    # vector_DEM = svm.SVR(kernel = "rbf")
    # ml_standard.train_model(model=vector_DEM, suffix="SVR", regr_suff="onlyDEM")
    # ml_standard.train_model(model=vector_DEM, suffix="SVR", regr_suff="SentinelDEM")
    # #
    # # # GRADIENT BOOSTING MODEL
    # gb_reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=300, min_samples_split = 2)
    # ml_standard.train_model(model=gb_reg, suffix="GB", regr_suff="Sentinel")
    # ml_standard.train_model(model=gb_reg, suffix="GB", regr_suff="onlyDEM")
    # ml_standard.train_model(model=gb_reg, suffix="GB", regr_suff="SentinelDEM")

    print("Training done!")

    #### TESTING ALL MODELS ####
    # ml_standard.test_models(models_to_test=saving_models, output=all_results)

    print("Testing done!")


    # ml_standard.test_models(models_to_test=saving_models, output="models_tested.csv")

    # ml_standard.feat_importance(model_suff="RF", models=saving_models, output=output_plots)

    ml_standard.latex_generator(model_suff="RF", models=saving_models)


    print("All tasks done!")
