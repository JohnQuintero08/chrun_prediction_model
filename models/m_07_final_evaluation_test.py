from  xgboost import XGBClassifier
import joblib
from models.m_01_model_evaluation import model_evaluation_meth, metrics_results, metrics_graphs
from features.f_00_datasets_creation import df_train_ordinal_up, df_valid_ordinal, df_test_ordinal
from features.f_06_features_target_split import features_target_split
ran = 12345
columns_to_drop = [
  'seniorcitizen',
  'dependents',
  'deviceprotection',
  'partner',
  'gender']

model_xgboost_optimized = XGBClassifier(objective='binary:logistic',
                                eval_metric = 'aucpr',
                                learning_rate = 0.05, 
                                max_depth=5, 
                                subsample=0.8,
                                colsample_bytree=0.8,
                                n_estimators=100,
                                alpha=3,
                                random_state=ran,
)

model_evaluation_meth(model_xgboost_optimized, 
                      'XGBoost', 
                      df_train_ordinal_up.drop(columns_to_drop, axis=1), 
                      df_valid_ordinal.drop(columns_to_drop, axis=1),
                      show_print=True)


df_test_final = df_test_ordinal.drop(columns_to_drop, axis=1)
features_test, target_test = features_target_split(df_test_final)
predictions_test = model_xgboost_optimized.predict(features_test)
predictions_proba_test = model_xgboost_optimized.predict_proba(features_test)


model_xgboost_results = metrics_results(model_xgboost_optimized, target_test, predictions_test, predictions_proba_test, show_print =True)


metrics_graphs(model_xgboost_optimized, 'XGBoost - Test Data', model_xgboost_results)

 
