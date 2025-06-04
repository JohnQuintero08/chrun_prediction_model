import lightgbm as lgbm 
from models.m_01_model_evaluation import multiple_model_evaluation, model_evaluation_meth
from features.f_00_datasets_creation import df_train_ordinal_up, df_valid_ordinal
ran=12345

model_lgbm = lgbm.LGBMClassifier(max_depth=-1, 
                                 learning_rate=0.05,
                                 n_estimators=100,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 random_state=ran)

multiple_model_evaluation(model_lgbm, 
                          'LightGBM', 
                          ord_enc=True, 
                          ohe=True, 
                          upsampling=True, 
                          scaled_upsampling=True)

model_evaluation_meth(model_lgbm, 'LightGBM', df_train_ordinal_up, df_valid_ordinal,show_print=True, graph=True)