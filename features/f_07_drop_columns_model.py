columns_to_drop = [
  'seniorcitizen',
  'dependents',
  'deviceprotection',
  'partner',
  'gender']

def drop_columns(df):
    df_c = df.copy()
    df_c = df_c.drop(columns_to_drop, axis=1)
    return df_c