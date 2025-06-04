from sklearn.ensemble import RandomForestClassifier
from models.m_01_model_evaluation import multiple_model_evaluation
ran=12345


model_random_forest = RandomForestClassifier(n_estimators=100, 
                                             criterion='gini',
                                             max_depth=8,
                                             random_state=ran,
                                             )

multiple_model_evaluation(model_random_forest,
                          'Random Forest',
                          ord_enc=True, 
                          ohe=True, 
                          upsampling=True, 
                          scaled_upsampling=True)