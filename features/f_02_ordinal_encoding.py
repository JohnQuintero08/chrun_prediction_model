import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
from features.f_01_train_test_split import split_df_for_model

categorical_variables = ['type', 'paymentmethod', 'internetservice', 'gender']

def train_ordinal_encoder(df_train, is_saved=False):
    categorical_encoder = OrdinalEncoder()
    categorical_encoder.fit(df_train[categorical_variables])
    if is_saved:
        joblib.dump(categorical_encoder, 'features/encoders/ordinal_encoder.pkl')
    return categorical_encoder

def ordinal_encoder(df, df_train, has_encoder=False):
    df_c = df.copy()
    if not has_encoder:
        encoder = train_ordinal_encoder(df_train)
    else:
        encoder = joblib.load('features/encoders/ordinal_encoder.pkl')
    var_encoded = pd.DataFrame(encoder.transform(df_c[categorical_variables]), 
                                columns=df_c[categorical_variables].columns)
    df_c = df_c.reset_index()
    df_c = pd.concat([df_c.drop(categorical_variables, axis=1), var_encoded], axis=1)
    df_c = df_c.set_index('index', drop=True)
    return df_c

def test_ordinal_encoding():    
    df = pd.read_feather('data/intermediate/preprocess_df_notnull.feather')
    df_train, df_valid, df_test = split_df_for_model(df)
    new_df_train = ordinal_encoder(df_train, df_train)
    new_df_valid = ordinal_encoder(df_valid, df_train)
    new_df_test = ordinal_encoder(df_test, df_train)
    return new_df_train, new_df_valid, new_df_test
