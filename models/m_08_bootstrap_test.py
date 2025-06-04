import numpy as np
import pandas as pd
import joblib
from features.f_01_train_test_split import split_df_for_model
from features.f_02_ordinal_encoding import ordinal_encoder
from features.f_05_upsampling import upsampling_dataframe
from features.f_06_features_target_split import features_target_split
from features.f_07_drop_columns_model import drop_columns
from models.m_01_model_evaluation import metrics_results, format_simple_results
random_seed = 42

def load_model(location):
    return joblib.load(location)

def boostrap_random_state_evaluation(model,df_to_bootstrap, model_name = 'Classifier'):
    np.random.seed(random_seed)
    random_list = [np.random.randint(0,100000) for _ in range(10)]
    
    for num in random_list:
        df_train, df_valid, df_test = split_df_for_model(df_to_bootstrap, num, False)
        
        df_train = upsampling_dataframe(df_train)
        list_df = [df_train, df_valid, df_test]
        list_df_mod = []
        
        for data in list_df:
            df_new = ordinal_encoder(data, None, True)
            df_new = drop_columns(df_new)
            list_df_mod.append(df_new)
        
        print(f'Prueba con número rándom state {num}')
        
        results = []
        for data in list_df_mod:
            features, target = features_target_split(data)
            predictions = model.predict(features)
            predictions_proba = model.predict_proba(features)
            results.append(format_simple_results(metrics_results(model_name, target, predictions, predictions_proba, show_print =False)))
        
        results = pd.DataFrame(results, columns=['f1_score', 'auc_roc'], index=['train', 'valid', 'test'])
        print(results)
        print(f'F1 score promedio: {results['f1_score'].mean():.2f}')
        print(f'AUC-ROC promedio: {results['auc_roc'].mean():.2f}')
        print('------->>>>><<<<<<----------')
        
model = load_model('models/models/xgboost_model.pkl')
df = pd.read_feather('data/intermediate/preprocess_df_notnull.feather')
boostrap_random_state_evaluation(model, df, 'XGBoost' )