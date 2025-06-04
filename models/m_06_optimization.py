import pandas as pd
import joblib
from  xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from models.m_01_model_evaluation import model_evaluation_meth
from features.f_00_datasets_creation import df_train_ordinal_up, df_valid_ordinal
from features.f_06_features_target_split import features_target_split

ran =12345
 
# Features selection

model_xgboost_selection = XGBClassifier(objective='binary:logistic',
                              eval_metric='auc',
                                learning_rate = 0.01, 
                                max_depth=8, 
                                subsample=0.8,
                                colsample_bytree=0.8,
                                n_estimators=100,
                                alpha=0,
                                random_state=ran)
model_evaluation_meth(model_xgboost_selection, 
                      'XGBoost', 
                      df_train_ordinal_up, 
                      df_valid_ordinal,
                      show_print=True)


selection_importances_xg = pd.DataFrame({'feature_name_x': model_xgboost_selection.feature_names_in_,
                                    'importance_value_x':model_xgboost_selection.feature_importances_}).sort_values(by='importance_value_x', ascending=False)
selection_importances_xg


model_xgboost_selection = XGBClassifier(objective='binary:logistic',
                              eval_metric='map',
                                learning_rate = 0.05, 
                                max_depth=5, 
                                subsample=0.8,
                                colsample_bytree=0.8,
                                n_estimators=100,
                                alpha=3,
                                random_state=ran,
)
columns_to_drop = [
  'seniorcitizen',
  'dependents',
  'deviceprotection',
  'partner',
  'gender']

def drop_columns(df):
    new_df = df.copy()
    new_df = new_df.drop(columns_to_drop, axis=1)
    return new_df

df_train_drop = drop_columns(df_train_ordinal_up)
df_valid_drop = drop_columns(df_valid_ordinal)

model_evaluation_meth(model_xgboost_selection, 
                      'XGBoost', 
                      df_train_ordinal_up, 
                      df_valid_ordinal,
                      show_print=True)

 

 
# #### GridSearch


model_xgb_gridsearch =XGBClassifier(objective='binary:logistic',
                                eval_metric='auc',
                                alpha=3,
                                random_state=ran)
param_grid = {
    "n_estimators": [80,90,100],  
    "learning_rate": [0.005, 0.01,],  
    "max_depth": [7, 8], 
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8,0.9]
}
grid_search = GridSearchCV(
    estimator=model_xgb_gridsearch,
    param_grid=param_grid,
    scoring="roc_auc",  
    cv=3,  
    verbose=3  
)


df_train_valid = pd.concat([df_train_drop, df_valid_drop], axis=0)
features_search, target_search = features_target_split(df_train_valid)


grid_search.fit(features_search, target_search)


print(f"Mejores parámetros: {grid_search.best_params_}")


best_xgboost = grid_search.best_estimator_


results_gridsearch = model_evaluation_meth(best_xgboost, 
                                            'XGBoost', 
                                            df_train_drop, 
                                            df_valid_drop, 
                                            show_print = True, 
                                            graph=True)

joblib.dump(best_xgboost, 'models/models/xgboost_model.pkl')
