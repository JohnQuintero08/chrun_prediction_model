import pandas as pd

def column_name_to_lowercase(df):
    return df.columns.str.lower()


def contract_df_transform(df):
    # Format contract dataframe
    df_contract_m = df.copy()
    df_contract_m['begindate'] = pd.to_datetime(df_contract_m['begindate'], format='%Y-%m-%d')
    df_contract_m.rename(columns={'enddate': 'chrun'}, inplace=True)
    def end_date_format(value):
        value = value.strip()
        if value == 'No':
            return 0
        else:
            return 1
    df_contract_m['chrun'] = df_contract_m['chrun'].apply(end_date_format)
    df_contract_m['totalcharges'] = df_contract_m['totalcharges'].replace(' ', 0).astype(float)
    df_contract_m['months'] = round(df_contract_m['totalcharges']/df_contract_m['monthlycharges'],0)
    return df_contract_m


def check_df_unique_values(df):
    for column in df.columns:
        print(f' Columna: {column}: {df[column].unique()}')


def merge_df(df_contract_m, df_internet_m, df_personal_m, df_phone_m):
    df_merge = df_contract_m.merge(df_internet_m, on='customerid', how='outer')\
                        .merge(df_personal_m, on='customerid', how='outer')\
                        .merge(df_phone_m, on='customerid', how='outer')
    df_merge['internetservice'] = df_merge['internetservice'].fillna('Not Hired')
    return df_merge


yes_no_columns = ['paperlessbilling', 'onlinesecurity', 'onlinebackup','deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'partner', 'dependents', 'multiplelines']

def boolean_yes_no_transform(df, yes_no_columns = yes_no_columns):
    df_merge = df.copy()    
    dict_yes_no = {
        'Yes' : 1,
        'No' : 0,
    }
    for col in yes_no_columns:
        df_merge[col] = df_merge[col].map(dict_yes_no)
    return df_merge

def fill_null_values(df):
    df_c = df.copy()
    df_c = df.fillna(0)
    return df_c

def df_creation(route):
    df = pd.read_csv(route)
    df.columns = column_name_to_lowercase(df)
    return df