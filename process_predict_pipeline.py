import pandas as pd
import joblib


dict_yes_no = {'Yes' : 1,'No' : 0 }
yes_no_columns = ['paperlessbilling', 'onlinesecurity', 'onlinebackup','deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'partner', 'dependents', 'multiplelines']
categorical_variables = ['type', 'paymentmethod', 'internetservice', 'gender']
columns_to_drop = ['seniorcitizen','dependents','deviceprotection','partner','gender']

def merge_dataframes(df_contract_m, df_internet_m, df_personal_m, df_phone_m):
    df_merge = df_contract_m.merge(df_internet_m, on='customerid', how='outer')\
                            .merge(df_personal_m, on='customerid', how='outer')\
                            .merge(df_phone_m, on='customerid', how='outer')
    return df_merge

def column_name_to_lowercase(df):
    return df.columns.str.lower()

def format_dataframe(df):
    df['begindate'] = pd.to_datetime(df['begindate'], format='%Y-%m-%d')
    df['totalcharges'] = df['totalcharges'].replace(' ', 0).astype(float)
    df['months'] = round(df['totalcharges']/df['monthlycharges'],0)
    df['internetservice'] = df['internetservice'].fillna('Not Hired')
    for col in yes_no_columns:
        df[col] = df[col].map(dict_yes_no)
    return df

def fix_df_traintest(df):
    df = df.reset_index(drop=True).drop(['customerid', 'begindate'], axis=1)
    return df

def ordinal_encoder(df):
    ordinal_encoder_pretrained = joblib.load('pretrained_models/ordinal_encoder_with_columns.pkl')
    var_encoded = pd.DataFrame(ordinal_encoder_pretrained.transform(df[categorical_variables]), 
                                columns=categorical_variables)
    return pd.concat([df.drop(categorical_variables, axis=1), var_encoded], axis=1)

def pipeline_procesing(df_contract_m, df_internet_m, df_personal_m, df_phone_m):
    df_merged = merge_dataframes(df_contract_m, df_internet_m, df_personal_m, df_phone_m)
    df_columns_lower = column_name_to_lowercase(df_merged)
    df_formated = format_dataframe(df_columns_lower)
    df_clean = fix_df_traintest(df_formated)
    df_encoded = ordinal_encoder(df_clean).fillna(0).drop(columns_to_drop, axis=1)
    
    model_xgb = joblib.load('pretrained_models/xgb_model_trained.pkl')
    predictions_model = model_xgb.predict(df_encoded)
    print(predictions_model)
    return predictions_model