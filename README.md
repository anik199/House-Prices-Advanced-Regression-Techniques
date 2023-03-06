# House Prices Advanced Regression Techniques

This project is based on the House Prices: Advanced Regression Techniques Kaggle competition dataset. The objective of this project is to predict the sales price of each house in the test set using multiple regression techniques.

## Dataset

The dataset contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. The dataset can be downloaded from the Kaggle competition page [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

## Libraries Used

- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-Learn

## Approach

1. **Exploratory Data Analysis (EDA)** - Analyze the data and identify patterns, trends, relationships, and anomalies in the data using visualizations and statistical methods.
2. **Data Preprocessing** - Handle missing values, categorical variables, outliers, skewness, and feature scaling.
3. **Feature Engineering** - Create new features by combining, transforming, or extracting information from existing features to improve the predictive power of the model.
4. **Model Selection** - Compare the performance of different regression models and select the best one based on the evaluation metrics.
5. **Hyperparameter Tuning** - Fine-tune the hyperparameters of the selected model to optimize the performance on the validation set.
6. **Prediction** - Make predictions on the test set using the trained model.

## Models Used

- Linear Regression
- Lasso Regression
- Ridge Regression
- Elastic Net Regression
- XGBoost Regression

## Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## Conclusion

- The XGBoost regression model performed the best with an RMSE score of 0.123 and an MAE score of 0.075 on the test set.
- Feature engineering and hyperparameter tuning played an important role in improving the performance of the models.
- Further improvements could be made by incorporating other regression techniques, ensemble models, or deep learning models.
