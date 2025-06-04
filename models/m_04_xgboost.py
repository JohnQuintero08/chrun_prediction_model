from  xgboost import XGBClassifier
from models.m_01_model_evaluation import multiple_model_evaluation, model_evaluation_meth
from features.f_00_datasets_creation import df_train_ordinal_up, df_valid_ordinal
ran=12345

model_xgboost = XGBClassifier(objective='binary:logistic',
                              eval_metric='auc',
                                learning_rate = 0.01, 
                                max_depth=8, 
                                subsample=0.8,
                                colsample_bytree=0.8,
                                n_estimators=100,
                                alpha=0,
                                random_state=ran)

multiple_model_evaluation(model_xgboost, 
                          'XGBoost', 
                          ord_enc=True, 
                          ohe=True, 
                          upsampling=True, 
                          scaled_upsampling=True)

model_evaluation_meth(model_xgboost, 'XGBoost', df_train_ordinal_up, df_valid_ordinal,show_print=True, graph=True)