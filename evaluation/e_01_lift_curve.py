import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from features.f_07_drop_columns_model import drop_columns
from features.f_02_ordinal_encoding import ordinal_encoder
from features.f_06_features_target_split import features_target_split



def lift_evaluation(df, model):
    # Adapta el nuevo df
    df_to_lift = ordinal_encoder(df, None, True)
    df_to_lift = drop_columns(df_to_lift)
    features_lift, target_lift = features_target_split(df_to_lift)
    # Predice valores
    predict_proba_lift = model.predict_proba(features_lift)[:, 1] 
    # Crea un df con el valor real y la probabilidad
    df_lift = pd.DataFrame({'true': target_lift, 'proba': predict_proba_lift}).sort_values(by='proba', ascending=False)
    df_lift['decil'] = pd.qcut(df_lift['proba'], 10, labels=False)
    
    lift_data_real = df_lift.groupby('decil')['true'].mean()
    baseline = df_lift['true'].mean()
    lift_score = lift_data_real / baseline
    
    return df_lift, lift_score, baseline

def lift_plot(lift_score, baseline):
    plt.figure(figsize=(15,5))
    plt.plot(range(1, 11), lift_score[::-1])  
    plt.plot(range(1,11), np.full(lift_score.shape, baseline) )
    plt.xlabel('Decil')
    plt.ylabel('Lift')
    plt.title('Curva Lift')
    plt.grid()
    plt.tight_layout()
    plt.show()

def read_results(df_lift):
    proba_lift_1 = df_lift[df_lift['decil']==9]['true'].sum() / len(df_lift[df_lift['decil']==9])*100
    print(f'La probabilidad de escoger un cliente en el primer decil que vaya a dejar el servicio es de {proba_lift_1:.0f}% ')
    
def test_lift():
    df = pd.read_feather('data/intermediate/preprocess_df_notnull.feather')
    model = joblib.load('models/models/xgboost_model.pkl')
    
    df_lift, lift_score, baseline = lift_evaluation(df, model)
    read_results(df_lift)
    lift_plot(lift_score, baseline)
    
test_lift()