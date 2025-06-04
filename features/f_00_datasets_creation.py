import pandas as pd
from features.f_01_train_test_split import split_df_for_model
from features.f_02_ordinal_encoding import ordinal_encoder, train_ordinal_encoder
from features.f_03_ohe import ohe_encoding
from features.f_04_standard_scaler import standard_scaler
from features.f_05_upsampling import upsampling_dataframe


df = pd.read_feather('data/intermediate/preprocess_df_notnull.feather')
df_train, df_valid, df_test = split_df_for_model(df)

def generate_ordinal_no_scaled_no_upsampling(df, df_train):
    new_df = ordinal_encoder(df, df_train)
    return new_df

def generate_ordinal_scaled_no_upsampling(df, df_train):
    new_df = ordinal_encoder(df, df_train)
    new_df = standard_scaler(new_df, df_train)
    return new_df

def generate_ordinal_no_scaled_upsampling(df, df_train):
    new_df = ordinal_encoder(df, df_train)
    new_df = upsampling_dataframe(new_df)
    return new_df
    
def generate_ordinal_scaled_upsampling(df, df_train):
    new_df = ordinal_encoder(df, df_train)
    new_df = standard_scaler(new_df,df_train)
    new_df = upsampling_dataframe(new_df)
    return new_df



def generate_ohe_no_scaled_no_upsampling(df):
    new_df = ohe_encoding(df)
    return new_df

def generate_ohe_scaled_no_upsampling(df, df_train):
    new_df = ohe_encoding(df)
    new_df = standard_scaler(new_df, df_train)
    return new_df

def generate_ohe_no_scaled_upsampling(df):
    new_df = ohe_encoding(df)
    new_df = upsampling_dataframe(new_df)
    return new_df

def generate_ohe_scaled_upsampling(df, df_train):
    new_df = ohe_encoding(df)
    new_df = standard_scaler(new_df, df_train)
    new_df = upsampling_dataframe(new_df)
    return new_df

# ORDINAL ENCODING

df_train_ordinal = generate_ordinal_no_scaled_no_upsampling(df_train, df_train)
df_train_ordinal_up = generate_ordinal_no_scaled_upsampling(df_train, df_train)
df_train_ordinal_scaled_up = generate_ordinal_scaled_upsampling(df_train, df_train)

df_valid_ordinal = generate_ordinal_no_scaled_no_upsampling(df_valid, df_train)
df_valid_ordinal_scaled = generate_ordinal_scaled_no_upsampling(df_valid, df_train)

df_test_ordinal = generate_ordinal_no_scaled_no_upsampling(df_test, df_train)

# ONE HOT ENCODING

df_train_ohe = generate_ohe_no_scaled_no_upsampling(df_train)
df_train_ohe_up = generate_ohe_no_scaled_upsampling(df_train)
df_train_ohe_scaled_up = generate_ohe_scaled_upsampling(df_train, df_train)

df_valid_ohe = generate_ohe_no_scaled_no_upsampling(df_valid)
df_valid_ohe_scaled = generate_ohe_scaled_no_upsampling(df_valid, df_train)

# SAVE ORDINAL ENCODER

train_ordinal_encoder(df_train, True)