import pandas as pd
import joblib
from preprocess.pr_01_dataset_construction import df_creation, contract_df_transform, merge_df, boolean_yes_no_transform
from preprocess.pr_01_dataset_construction import fill_null_values, drop_not_important_features
from features.f_07_drop_columns_model import drop_columns
from features.f_02_ordinal_encoding import ordinal_encoder
from features.f_06_features_target_split import features_target_split

def load_model(location):
    return joblib.load(location)

def data_ingestion():
    try:
        df_contract = df_creation('pipeline/data/contract.csv')
        df_internet = df_creation('pipeline/data/internet.csv')
        df_personal = df_creation('pipeline/data/personal.csv')
        df_phone    = df_creation('pipeline/data/phone.csv')
        return df_contract, df_internet, df_personal, df_phone
    except:
        print('Error durante la lectura de los archivos')

def extraction_preprocess(df_contract, df_internet, df_personal, df_phone):
    try:
        df_contract = contract_df_transform(df_contract)
        df = merge_df(df_contract, df_internet, df_personal, df_phone)
        df = boolean_yes_no_transform(df)
        df = fill_null_values(df)
        df = drop_not_important_features(df)
        return df
    except:
        print('Error durante el preprocesamiento de los datos')
        
def feature_engineering(data):
    try:
        df = ordinal_encoder(data, None, True)
        df = drop_columns(df)
        return df
    except:
        print('Error durante la transformación de los datos')

def model_prediction(data, has_target=True):
    try:
        model = load_model('models/models/xgboost_model.pkl')
        if has_target:
            features, target = features_target_split(data)
        else:
            features = data
        predictions = model.predict(features)
        predictions_proba = model.predict_proba(features)[:, 1] 
        df_lift = pd.DataFrame({'probability': predictions_proba}).sort_values(by='probability', ascending=False)
        print(df_lift)
        return df_lift
    except:
        print('Error en la predicción de nuevos datos')

def lift_analysis(df, num_clients=5):
    try:
        print(f'Los {num_clients} clientes con más posibilidad de irse son:')
        for client in df.iloc[:num_clients,:].index.values:
            print(client) 
    except:
        print('Error en la generación de clientes chrun')
    
def exe(num_clients):
    df_contract, df_internet, df_personal, df_phone = data_ingestion()
    df_preprocess = extraction_preprocess(df_contract, df_internet, df_personal, df_phone)
    df_feature = feature_engineering(df_preprocess)
    predictions = model_prediction(df_feature, True)
    lift_analysis(predictions, num_clients)

exe(5)
