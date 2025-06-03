def features_target_split(df):
    features = df.drop(['chrun'], axis=1)
    target = df['chrun']
    return features, target