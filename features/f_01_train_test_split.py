from sklearn.model_selection import train_test_split
ran = 12345
random_seed = 42

def split_df_for_model(df, ran=ran, has_strata=True):
    df_train, df_pass = train_test_split(df, 
                                         test_size=0.30, 
                                         random_state=ran, 
                                         stratify=df['chrun'] if has_strata else None
                                         )
    df_valid, df_test = train_test_split(df_pass, 
                                         test_size=0.5, 
                                         random_state=ran, 
                                         stratify=df_pass['chrun'] if has_strata else None
                                         )
    return df_train, df_valid, df_test

import pandas as pd

def test_split_df_for_model():
    df = pd.read_feather('data/intermediate/preprocess_df_notnull.feather')
    df_train, df_valid, df_test = split_df_for_model(df)
    print(df_train.shape)

