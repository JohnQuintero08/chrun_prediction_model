# MODELS EVALUATION

- **Oversampling improves the model's training performance.**
- The models showed similar results; even the baseline model, logistic regression, produced overall results comparable to the others.
- Among the models presented, **XGBoost and LightGBM showed the highest values** for the F1 score and ROC AUC score. In this case, we aim to identify as many users at risk of churning as possible, which makes the F1 score particularly relevant, as we also want to optimize recall — a component of that metric.
- Similarly, tests were carried out using several datasets, applying modifications and observing parameter behavior. **DataFrames with upsampling improved model performance on validation data**, whereas scaling numerical features **did not produce significant changes** for the XGBoost model.

# FEATURES SELECTION

- **The parameters were reduced to the 12 most important ones** to simplify the model. As observed, the ROC AUC score for the validation data **remains unchanged at 0.77**, and overfitting relative to the training data is reduced.

# OPTIMIZATION

- **After performing parameter selection and optimizing XGBoost model**, it was observed that the model continued to behave very similarly, with no improvement in the response metrics compared to previous results, maintaining an **F1 score of 0.63** and a **ROC AUC score of 0.76**.

# TEST DATA

- The performance of the xgboost model is maintained even for the test dataset.

# BOOTSTRAP TEST

- As observed, the model is stable when tested with different combinations for splitting the training, validation, and test data, resulting in an average **ROC AUC between 0.78 and 0.80**.
