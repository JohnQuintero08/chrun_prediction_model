from matplotlib import pyplot as plt

def check_general_info(df):
    print('--INFO--')
    print(df.info())
    print()
    print('--HEAD--')
    print(df.head())
    print()
    print('--DESCRIBE--')
    print(df.describe(include='all'))
    
def check_duplicates(df):
    print(f"Duplicates: {df.duplicated().sum()}")

def check_null_values(df):
    print(f"""Null values: 
          {df.isna().sum()}""")

def print_function_over_array(dfs_names, dfs, function):
    for name , df in zip(dfs_names, dfs):
        print('')
        print(f'---{name}---')
        print('')
        function(df)


def plot_histogram(df, feature, title, xlab, ylab, bins=300):
    plt.figure(figsize=(15,5))
    df[feature].hist(bins=bins, edgecolor='lightblue')
    plt.title(title)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.tight_layout()
    plt.show()


def plot_bar_categorical(columns, df, name, x_col, y_col):
    total_plots= x_col * y_col
    if total_plots >= 2:
        fig, axs = plt.subplots(x_col, y_col,figsize=(15,5))
        axs =axs.flatten()
        for i,column in enumerate(columns):
            valores = df[column].value_counts()
            axs[i].bar(valores.index, valores.values)
            axs[i].set_title(f'Variable: {column}, dataset: {name}')
            axs[i].tick_params(axis='x', rotation=20)

        for j in range(len(columns), total_plots):
            fig.delaxes(axs[j])    
    else: 
        fig, ax = plt.subplots(figsize=(5,5))
        ax.bar(df[columns].values, df[columns].index)
        ax.set_title(f'Variable: {columns}, dataset: {name}')
        
    plt.tight_layout()
    plt.show()
    