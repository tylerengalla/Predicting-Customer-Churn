# Predicting-Customer-Churn

This was the final group project done in my Advanced Machine Learning class in my Master's of Business Analytics program at The University of Texas at Austin. You can see a full blog report in the files section. 

Our goal was simple - can we predict which customers are likely to churn? 

# Data Set
Our data set to help us answer this consists of 100 thousand records (each indicative of a single telecom customer) and 100 attributes related to each of those customer’s. These attributes relate to a customer’s phone usage, revenue, call behavior, demographics, and equipment (phone) details. 

# Selecting the Right Model for Churn Prediction
When approaching customer churn prediction, selecting the right model is crucial for achieving high accuracy and ensuring that the model generalizes well, allowing for efficient retraining in production. In this section, we’ll walk you through our detailed process of evaluating various machine learning models to identify the optimal solution.

# How We Chose Our Candidate Models
Our first step was to identify candidate models suitable for our churn prediction problem, which involves both categorical and numerical data with potential class imbalance. We selected models known for robust performance on structured tabular datasets:

- Logistic Regression: Simple, interpretable, and effective for linear relationships.
- Random Forest: Reliable and versatile, excellent at capturing complex interactions between features.
- Gradient Boosting Models: Powerful algorithms renowned for their predictive accuracy, including GradientBoostingClassifier (scikit-learn), LightGBM, and XGBoost.

# Defining Evaluation Metrics
To thoroughly assess model performance, we chose multiple complementary metrics:
ROC AUC (Test): Our primary metric, which measures the model's ability to correctly rank customers who churn versus those who do not, making it particularly valuable for imbalanced datasets.

- Accuracy: A simple measure of overall correct predictions, although less reliable when data is imbalanced.
- AUC Gap: Calculated as the difference between the training and testing ROC AUC scores to identify potential overfitting.
- Training Time: Ensures the practical feasibility of frequently updating the model.

# Model Selection Decision
After careful benchmarking, LightGBM emerged as the standout candidate, delivering the best balance among predictive accuracy, computational efficiency, and generalization performance. Its superior ROC AUC scores and reasonable AUC gap, along with faster training times compared to other gradient boosting models, made it an ideal choice for further optimization.  This choice set the foundation for our subsequent hyperparameter tuning phase.

- Model: LightGBM
- Accuracy: .644
- ROC AUC (Train): .748
- ROC AUC (Test): .6967
- AUC Gap: .0513
- Training Time (s): 12.84

# Conclusion
At the heart of the model’s predictive power are several features that capture customer behavior
and engagement patterns. The feature importance table shows that "change_mou" (change in
minutes of use) ranks highest, followed closely by "mou_Mean" (average minutes of use),
"months" (customer tenure), and "totmrc_Mean" (total monthly recurring charge). This indicates
that abrupt changes in usage and long-term engagement are critical signals for identifying
customers at risk of churning.

The preprocessing pipeline is carefully constructed to handle both numerical and categorical
data effectively. Numerical features are processed through a sequence of median imputation,
power transformation, and standard scaling, which helps address missing values, normalize
distributions, and ensure consistent scaling across features. Categorical variables, on the other
hand, are imputed with a constant "missing" value and then one-hot encoded, allowing the
model to handle diverse categories robustly and avoid issues with unseen values during
inference.

Model selection and tuning are performed using RandomizedSearchCV, optimizing a range of
hyperparameters for the LightGBM classifier. The search explores variations in tree depth,
learning rate, subsampling ratios, and other key parameters, ultimately settling on a
configuration that balances complexity and generalizability. The chosen model uses a moderate
learning rate (0.05), shallow trees (max_depth=5), and a substantial number of estimators (500),
which collectively help prevent overfitting while capturing meaningful patterns in the data.
The integration of these components into a unified pipeline ensures that preprocessing and
modeling is tightly coupled, reducing the risk of data leakage and improving reproducibility.
The use of a column transformer allows for parallel processing of different feature types, and the
pipeline structure makes the workflow scalable and maintainable. This design also simplifies the
deployment process, as the same transformations applied during training can be consistently
used during prediction.

In summary, the notebook provides a robust framework for churn prediction, leveraging
advanced feature engineering, systematic preprocessing, and thoughtful model tuning. The
results underscore the importance of monitoring customer usage trends and tenure in retention
strategies. By focusing on these key drivers and maintaining a disciplined modeling approach,
organizations can better identify at-risk customers and take proactive measures to reduce
churn.

