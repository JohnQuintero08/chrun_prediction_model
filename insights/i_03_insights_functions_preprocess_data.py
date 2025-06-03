import pandas as pd
from matplotlib import pyplot as plt


def plot_crosstab(df, columns, target, x_col, y_col, normal =True):
    tota_plots = x_col * y_col
    if tota_plots >=2:
        fig, axs = plt.subplots(x_col,y_col,figsize=(15,8))
        axs= axs.flatten()
        for i,col in enumerate(columns):
            ct = pd.crosstab(df[col], 
                            df[target], 
                            normalize='index' if normal else False
                            )
            ct.plot(kind='bar', stacked=True, ax =axs[i])
            axs[i].set_title(f'{col} vs Churn')
            axs[i].set_ylabel('Proportion')
    else:
        fig, axs = plt.subplots(figsize=(15,5))
        new_ct = pd.crosstab(df[columns], 
                            df[target], 
                            normalize='index' if normal else False
                            )
        new_ct.plot(kind='bar', stacked=True, ax=axs)
        axs.set_title(f'{columns} vs Churn')
        axs.set_ylabel('Proportion')
    plt.tight_layout()
    plt.show()

