# House Prices Advanced Regression Techniques.   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anik199/House_Prices_Advanced_Regression/blob/main/House_Prices_ART.ipynb)


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





# House Price Prediction using Advanced Pipeline

This project is an implementation of machine learning techniques to predict house prices based on the given data in the Kaggle House Prices competition. The model uses a combination of feature engineering, data preprocessing, and regression algorithms to achieve an RMSE score of 0.12764 on the test dataset.

## Dataset

The dataset used in this project is the Kaggle House Prices competition dataset, which can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). The dataset contains a total of 1460 training samples and 1459 test samples, with 80 features that describe various aspects of the properties.

## Preprocessing

The dataset required extensive preprocessing to ensure that it can be fed into the model. The preprocessing steps that were taken include:

* Handling missing values: Missing values in numerical features were replaced with the median of the respective feature, while missing values in categorical features were replaced with the most common value of the respective feature.

* Encoding categorical features: Categorical features were converted to numerical features using one-hot encoding.

* Feature engineering: The dataset was transformed using feature engineering techniques such as creating new features by combining existing features, scaling features, and removing outliers.

## Model

The model used in this project is an ensemble model consisting of XGBoost, LightGBM, and ElasticNet regression algorithms. The model was trained on the preprocessed dataset using cross-validation techniques to ensure that it generalizes well to new data. The final model achieved an RMSE score of 0.12764 on the test dataset.

## Usage

To use the code, the following steps can be taken:

1. Download the dataset from the Kaggle competition website and save it in a directory named `data`.

2. Install the required packages by running `pip install -r requirements.txt`.

3. Run the `main.py` file to preprocess the data, train the model, and generate the submission file.

## Results

The final model achieved an RMSE score of 0.12764 on the test dataset, which is one of the best scores achieved in the Kaggle House Prices competition.

## Credits

The implementation of this project is based on the work of Anik Kumar, whose Kaggle notebook can be found [here](https://www.kaggle.com/code/anikkumar/house-price-advanced-pipeline-rmse-0-12764). The code was adapted and modified to suit the needs of this project.
