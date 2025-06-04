import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from features.f_01_train_test_split import split_df_for_model

ran = 12345
random_seed =42

def upsampling_dataframe(df, bootstrap = True, repeat=2):
    np.random.seed(random_seed)
    
    boostrap_data = []
    positive_df = df[df['chrun']==1]
    negative_df = df[df['chrun']==0]
    if bootstrap == True:    
        for i in range(len(positive_df)-1):
            random_index = np.random.randint(0, len(positive_df)-1)
            boostrap_data.append(positive_df.iloc[random_index,:])
        boostrap_data = pd.DataFrame(boostrap_data, columns=positive_df.columns)
        bootstrap_df = pd.concat([positive_df, boostrap_data, negative_df], axis=0)
        bootstrap_df_shuffle = shuffle(bootstrap_df, random_state=ran )
        return bootstrap_df_shuffle
    else:
        df_upsample = pd.concat([negative_df]+[positive_df]*repeat)
        df_shuffled = shuffle(df_upsample, random_state=ran )
        return df_shuffled
    
def test_upsampling():
    df = pd.read_feather('data/intermediate/preprocess_df_notnull.feather')
    df_train, df_valid, df_test = split_df_for_model(df)
    print(df_train.shape)
    new_df_train = upsampling_dataframe(df_train)
    return new_df_train

test_upsampling()
