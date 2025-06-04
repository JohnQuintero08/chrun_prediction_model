# 1. PROJECT STAGES

The project was divided into 4 stages: Exploration, Preprocessing, Model Training, and Optimization. Below is a description of the objective of each stage and the main findings.

## - Exploration:

The main idea was to understand the data and its behavior—whether there were null, duplicated, or outlier values, or any anomalies that needed correction. Bar charts and histograms were created to visualize the data behavior.

It was found that most of the data consisted of binary values (Yes, No) or categorical variables, and only payment values were continuous numerical variables. Some variables such as dates and total charges had incorrect data types and needed correction. Additionally, the target variable `EndDate` was imbalanced with a 1:4 ratio.

In terms of distribution, some categorical variables like `streamingTV` and `streamingMovies` were balanced both in number and in the proportion of customers who left the platform, suggesting that some features might be unnecessary for training.

Lastly, payment data provided valuable information about customer churn—most of the customers who left had been with the service for a long period, and those who paid more were more likely to leave.

## - Preprocessing:

This stage aimed to fix data formatting and create various dataframes to allow model training from different perspectives in order to find the best option.

After inspecting the data, several errors had to be corrected. First, dataframes were merged using the `customerID` variable as a key. Column names were converted to lowercase for consistency. Date columns were converted to `datetime` type, numerical variables to `float`, and binary Yes/No variables to 1/0.

Additionally, a new variable `Months` was created to estimate how long a customer had used the Telecom service by dividing total charges by monthly charges.

The target variable was transformed to 0 and 1, where 1 indicates the customer left the service and 0 means they are still active. Dates were not used, as the problem only required predicting whether a customer would leave.

The cleaned data was then split into training, validation, and test sets using a 70:15:15 ratio. Given the dataset had only 7,043 records, a larger training portion was used. To ensure consistent class distribution across splits, the `stratify` parameter was used.

Different training dataframes were created based on potential model training approaches:

- Scaling using `StandardScaler` from sklearn to normalize payment data for linear models.
- Ordinal and One-Hot Encoding for categorical variables to test tree-based models.

Missing values were filled with 0, indicating the customer did not use the corresponding service.

Some datasets were adapted using oversampling to handle class imbalance.

The training dataframes were:

- Ordinal encoding / No Scaling / No Oversampling
- Ordinal encoding / No Scaling / With Oversampling
- Ordinal encoding / With Scaling / With Oversampling
- One-Hot Encoding / No Scaling / No Oversampling
- One-Hot Encoding / No Scaling / With Oversampling
- One-Hot Encoding / With Scaling / With Oversampling

## - Model Training:

This stage tested different classification models to predict the target variable.

Functions were created to evaluate models without repeating code. Evaluation metrics included Recall, Precision, F1 Score, and AUC-ROC.

The evaluated models were: Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost. These models were chosen for their effectiveness with categorical variables. Logistic Regression was used as a baseline.

Most models performed similarly, showing an average F1 score of 0.65 and AUC-ROC of 0.77. LightGBM and XGBoost were selected for further comparison due to lower overfitting between training and validation. CatBoost had longer training times.

There was little difference between using OHE and ordinal encoding, or scaling vs. not scaling. However, oversampling the training data significantly improved model performance by 15% to 20%.

## - Optimization:

This stage aimed to improve the performance of the models selected in the previous step.

GridSearch from sklearn was used for hyperparameter tuning, but no significant improvement was observed and overfitting appeared, requiring heavy parameter restrictions.

Dimensionality reduction was also applied. Variables such as `seniorcitizen`, `dependents`, `deviceprotection`, `partner`, and `gender` were found to add no value to the models—removing them yielded the same performance with lower dimensionality.

XGBoost was chosen over LightGBM as it performed better during optimization, maintaining both AUC-ROC and F1 score.

Predictions on the test data showed that the XGBoost model was stable on unseen data.

Finally, bootstrapping was performed to ensure model stability was not dependent on the specific stratified split. Various random partitions showed consistent results, with average AUC-ROC scores between 0.78 and 0.80.

## - Final Results:

A lift curve analysis was conducted to determine how much better the model was compared to random selection. It was found that selecting the top 10% of customers most likely to churn (as predicted by the model) yielded six times more positive churn predictions than random selection.

This demonstrates the model’s reliability.

All planned project stages were completed as intended.

# 2. CHALLENGES

The main difficulties were related to generalizing the training functions. For example, CatBoost handles its own encoding, making some preprocessing steps inapplicable.

Improving the model's performance was also challenging and required the creation of the `Months` variable and parameter tuning with regularization, involving a lengthy trial-and-error process.

Otherwise, the data was relatively clean and easy to handle.

# 3. KEY STEPS

From my point of view, four key steps led to the model’s success:

- Creating various dataframes to explore model performance from different perspectives.
- Discovering that oversampling training data increased model performance by 15% to 20%.
- Creating the `Months` variable, which stabilized the model and increased the AUC-ROC from an average of 0.72 to 0.77.
- Establishing reusable functions for model testing, simplifying experimentation without adding repetitive code.

# 4. FINAL MODEL AND MODEL QUALITY

The final model used was an XGBoost with the following parameters:

```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=100,
    alpha=3,
    random_state=ran,
)

And can be found in the folder models/models/xgboost_model.pkl
```

The model achieved the following prediction metrics:

F1 Score: 0.69 - 0.71

AUC-ROC: 0.78 - 0.80

Since this type of model is used to identify customers likely to churn and offer them promotions, only a small proportion of customers can be targeted. If a promotion is sent to the top 10% most likely to churn (according to the model), it is expected that around 81% of them would actually churn—making it possible to retain them effectively.
