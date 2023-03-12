# House Price Prediction using Advanced Pipeline.   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anik199/House_Prices_Advanced_Regression/blob/main/House_Prices_ART.ipynb)


This project is focused on predicting the sale price of houses using advanced pipeline techniques. We use a dataset of housing prices from Kaggle, which contains various features such as number of rooms, square footage, and location. The goal is to develop a regression model that can accurately predict the sale price of a house.

## Dataset

The dataset used in this project is the Kaggle House Prices competition dataset, which can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). The dataset contains a total of 1460 training samples and 1459 test samples, with 80 features that describe various aspects of the properties.

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

The pipeline includes:

1. **Data cleaning and pre-processing** - handling missing data, removing outliers, and encoding categorical variables.
2. **Feature engineering** - creating new features from existing ones, transforming variables to follow a normal distribution, and performing feature scaling.
3. **Model selection and training** - evaluating several regression models, tuning hyperparameters, and selecting the best performing model.
4. **Model evaluation and deployment** - evaluating the final model's performance on a test set, deploying the model in a web app, and providing predictions to end-users.

## Models Used

- Linear Regression
- Lasso Regression
- Ridge Regression
- Elastic Net Regression
- XGBoost Regression

## Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## Results

The final model achieved an RMSE score of 0.12764 on the test dataset, which is one of the best scores achieved in the Kaggle House Prices competition.

## Conclusion

- The XGBoost regression model performed the best with an RMSE score of 0.123 and an MAE score of 0.075 on the test set.
- Feature engineering and hyperparameter tuning played an important role in improving the performance of the models.
- Further improvements could be made by incorporating other regression techniques, ensemble models, or deep learning models.










