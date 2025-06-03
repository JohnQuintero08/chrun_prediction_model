from preprocess.pr_01_dataset_construction import df_creation, contract_df_transform, merge_df, boolean_yes_no_transform
from preprocess.pr_01_dataset_construction import fill_null_values

df_contract = df_creation('data/input/contract.csv')
df_internet = df_creation('data/input/internet.csv')
df_personal = df_creation('data/input/personal.csv')
df_phone    = df_creation('data/input/phone.csv')

df_contract = contract_df_transform(df_contract)

df = merge_df(df_contract, df_internet, df_personal, df_phone)
df = boolean_yes_no_transform(df)

df.to_feather('data/intermediate/preprocess_df.feather')

df = fill_null_values(df)
df.to_feather('data/intermediate/preprocess_df_notnull.feather')