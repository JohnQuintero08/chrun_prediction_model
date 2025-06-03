import pandas as pd
from features.f_01_train_test_split import split_df_for_model

categorical_variables = ['type', 'paymentmethod', 'internetservice', 'gender']

def ohe_encoding(df):
    ohe_var_encoded = pd.get_dummies(df[categorical_variables], drop_first=True, dtype=float)
    return pd.concat([df.drop(categorical_variables, axis=1), ohe_var_encoded], axis=1)

def test_ordinal_encoding():    
    df = pd.read_feather('data/intermediate/preprocess_df_notnull.feather')
    df_train, df_valid, df_test = split_df_for_model(df)
    new_df_train = ohe_encoding(df_train)
    new_df_valid = ohe_encoding(df_valid)
    new_df_test = ohe_encoding(df_test)
    return new_df_train, new_df_valid, new_df_test

new_df_train, new_df_valid, new_df_test = test_ordinal_encoding()