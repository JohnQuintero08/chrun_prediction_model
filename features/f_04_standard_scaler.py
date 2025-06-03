import pandas as pd
from sklearn.preprocessing import StandardScaler
from features.f_01_train_test_split import split_df_for_model

numerical_variables = ['monthlycharges', 'totalcharges', 'months']

def train_standard_scaler(df_train):
    scaler = StandardScaler()   
    scaler.fit(df_train[numerical_variables])  
    return scaler

def standard_scaler(df, df_train):
    df_c = df.copy()
    scaler = train_standard_scaler(df_train)
    numerical_scaled = pd.DataFrame(scaler.transform(df_c[numerical_variables]), columns=numerical_variables)
    df_c = df_c.reset_index()
    df_c = pd.concat([df_c.drop(numerical_variables, axis=1), numerical_scaled], axis=1)  
    df_c = df_c.set_index('index', drop=True)
    return df_c


def test_standard_scaler():    
    df = pd.read_feather('data/intermediate/preprocess_df_notnull.feather')
    df_train, df_valid, df_test = split_df_for_model(df)
    new_df_train = standard_scaler(df_train, df_train)
    new_df_valid = standard_scaler(df_valid, df_train)
    new_df_test = standard_scaler(df_test, df_train)
    return new_df_train, new_df_valid, new_df_test

new_df_train, new_df_valid, new_df_test = test_standard_scaler()

