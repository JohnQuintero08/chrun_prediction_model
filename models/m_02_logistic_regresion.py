from sklearn.linear_model import LogisticRegression
from models.m_01_model_evaluation import multiple_model_evaluation
ran=12345

model_logistic_regression = LogisticRegression(random_state=ran, solver='liblinear')

multiple_model_evaluation(model_logistic_regression,
                          'Regresión Logística',
                          ord_enc=True, 
                          ohe=True, 
                          upsampling=True, 
                          scaled_upsampling=True)