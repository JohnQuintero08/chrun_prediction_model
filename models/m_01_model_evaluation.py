import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, recall_score, precision_score

from features.f_06_features_target_split import features_target_split
from features.f_00_datasets_creation import df_train_ordinal, df_train_ordinal_up, df_train_ordinal_scaled_up, df_train_ohe, df_train_ohe_up, df_train_ohe_scaled_up, df_valid_ordinal, df_valid_ordinal_scaled, df_valid_ohe, df_valid_ohe_scaled


# Esta función grafica la curva roc, la matriz de confusión y la curva pr.
def metrics_graphs(model, model_name, metrics_results):
    cm, f1_scr, roc_auc_scr, roc_curve_result, pr_curve_result = metrics_results
    precision, recall, _ = pr_curve_result
    
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp_pr = PrecisionRecallDisplay(precision=precision,recall= recall)
    fpr, tpr, threshold = roc_curve_result
    disp_cm.plot(ax=axs[0])
    axs[0].set_title('Matriz de confusión')
    axs[1].scatter(fpr, tpr)
    axs[1].set_title('Curva ROC')
    axs[1].set_xlabel('FPR')
    axs[1].set_ylabel('TPR')
    axs[1].grid(True)
    disp_pr.plot(ax=axs[2])
    axs[2].set_title('Curva Precision-Recall')
    plt.suptitle(f'Evaluación del Modelo {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    # plt.tight_layout()
    plt.show()

# Esta función toma las predicciones y la variable objetivo y genera los valores de las metricas de evaluación
def metrics_results(model_name, target, predictions, predictions_proba, show_print =False):
    recall = recall_score(target, predictions)
    precision = precision_score(target, predictions)
    cm = confusion_matrix(target, predictions)
    f1_scr = f1_score(target, predictions)
    roc_auc_scr = roc_auc_score(target, predictions)
    roc_curve_result = roc_curve(target, predictions_proba[:,1])
    pr_curve_result = precision_recall_curve(target,predictions_proba[:,1])
    
    if show_print == True:
      print(f"""
            Modelo: {model_name}
            F1 score: {f1_scr:.2f}
            ROC AUC score: {roc_auc_scr:.2f}
            Recall: {recall:.2f}
            Precision: {precision:.2f}""")
    
    return cm, f1_scr, roc_auc_scr, roc_curve_result, pr_curve_result

def format_simple_results(metrics_results):
    cm, f1_scr, roc_auc_scr, roc_curve_result, pr_curve_result = metrics_results
    return [np.round(f1_scr,2), np.round(roc_auc_scr,2)]

# Esta función toma un modelo, lo entrena y genera las metricas de entrenamiento y validación
def model_evaluation_meth(model, model_name, df_train, df_valid, show_print = False, graph=False):
    features_t, target_t = features_target_split(df_train)
    features_v, target_v = features_target_split(df_valid)
    model.fit(features_t, target_t)
    predictions_t = model.predict(features_t)
    predictions_v = model.predict(features_v)
    predictions_proba_t = model.predict_proba(features_t)
    predictions_proba_v = model.predict_proba(features_v)
    
    metrics_resuts_train = metrics_results(model_name + ' - Entrenamiento', target_t, predictions_t, predictions_proba_t, show_print)
    metrics_results_valid = metrics_results(model_name + ' - Validacion', target_v, predictions_v, predictions_proba_v, show_print)
        
    if graph == True:
        metrics_graphs(model, model_name + ' - Entrenamiento', metrics_resuts_train)
        metrics_graphs(model, model_name + ' - Validacion', metrics_results_valid)
    
    if show_print == False:
        f1_scr_t, roc_auc_scr_t = format_simple_results(metrics_resuts_train)
        f1_scr_v, roc_auc_scr_v = format_simple_results(metrics_results_valid)
        return [f1_scr_t, roc_auc_scr_t, f1_scr_v, roc_auc_scr_v]
    else:
        print('         ****')
        
# Esta función toma un modelo y lo evalua con distintos dataframes.
def multiple_model_evaluation(model, model_name, ord_enc=True, ohe=True, upsampling=True, scaled_upsampling=True):
    results = []
    indexs = []
    if ord_enc == True:
        results.append(model_evaluation_meth(model, 
                            f'{model_name} - Ordinal encoding / Sin Escalado / Sin Upsampling',
                            df_train_ordinal, 
                            df_valid_ordinal))
        indexs.append('Ordinal encoding / Sin Escalado / Sin Upsampling')
    if ord_enc == True & upsampling == True:
        results.append(model_evaluation_meth(model, 
                            f'{model_name} - Ordinal encoding / Sin Escalado / Con Upsampling',
                            df_train_ordinal_up, 
                            df_valid_ordinal))
        indexs.append('Ordinal encoding / Sin Escalado / Con Upsampling')
    if ord_enc == True & scaled_upsampling == True:
        results.append(model_evaluation_meth(model, 
                            f'{model_name} - Ordinal encoding / Con Escalado / Con Upsampling',
                            df_train_ordinal_scaled_up, 
                            df_valid_ordinal_scaled))
        indexs.append('Ordinal encoding / Con Escalado / Con Upsampling')
    if ohe == True:
        results.append(model_evaluation_meth(model, 
                            f'{model_name} - OHE / Sin Escalado / Sin Upsampling',
                            df_train_ohe, 
                            df_valid_ohe))
        indexs.append('OHE / Sin Escalado / Sin Upsampling')
    if ohe == True & upsampling == True:
        results.append(model_evaluation_meth(model, 
                            f'{model_name} - OHE / Sin Escalado / Con Upsampling',
                            df_train_ohe_up, 
                            df_valid_ohe))
        indexs.append('OHE / Sin Escalado / Con Upsampling')
    if ohe == True & scaled_upsampling == True:
        results.append(model_evaluation_meth(model, 
                            f'{model_name} - OHE / Con Escalado / Con Upsampling',
                            df_train_ohe_scaled_up, 
                            df_valid_ohe_scaled))
        indexs.append('OHE / Con Escalado / Con Upsampling')
    df_results = pd.DataFrame(results,index=indexs, columns=['f1_scr_t', 'roc_auc_scr_t', 'f1_scr_v', 'roc_auc_scr_v'])
    print(df_results)