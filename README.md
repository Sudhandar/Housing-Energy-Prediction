# Housing-Energy-Prediction

## Business Problem 

**Goal:** Create a model to forecast energy usage (regression) based on numerous household details and attributes.

Residential Energy Consumption Survey (RECS) is a national sample survey that collects energy-related data for housing units occupied as primary residences and the households that live in them. The RECS 2009 data was collected from 12,083 households with 938 features and our goal is to predict the energy consumption in Kilowatt-hour.

## Project Workflow

![alt text](https://github.com/Sudhandar/Housing-Energy-Prediction/blob/main/images/Energy%20Prediction.png)

## Dataset

Link to the dataset: [RECS Survey Data](https://www.eia.gov/consumption/residential/data/2009/index.php?view=microdata)

The dataset contains 12,083 rows with 938 features with both numerical and categorical columns. Since, there are a lot of features a lot of steps were involved with respect to cleaning the data. The various steps are outlined as below,

1. Numerical columns with more than 50% null values were removed.
2. Impute the null values in remaining numerical columns with the median.
3. Categorical  columns with more than 50% null values were  removed.
4. Impute the null values in remaining categorical columns with the mode.
5. Remove the imputation flags.
6. Remove the columns which contain kwh data in British Thermal Units (BTU).
7. Remove the individual energy consumption features (kwhoth, kwhcol, etc.)
8. Remove Outliers using IQR.

- **Before Cleaning:** 12,082 records, 940 features
- **After Cleaning:** 10769 records,  233 features

## Evaluation Metrics

The following metrics have been selected to evaluate the performance of the models,

1.  R^2 error - Helps in measuring the goodness of fit and comparing the performance of models.
2.  Mean Absolute Error - Helps in finding out the value by which the predictions are wrong.


## Model Selection

### Reasons for rejecting a linear model

The data fails the following assumptions of a linear regression model,

1. Independence - All the features are dependent on each (multicollinearity). This was verified using the Variance Inflation Factor (VIF)
2. The data is not linear - This was verified by plotting the residuals of a lasso regression model.

![alt text](https://github.com/Sudhandar/Housing-Energy-Prediction/blob/main/images/Linear%20model%20plots.png)


## Reasons for opting a tree based model:

Due to the non-linearity of the data, it's better to pivot to a tree based regression model since it does not expect linearity between the features and the target variable. Among the tree based models, the following models were used, Random Forest, XGBoost, AdaBoost and Gradient Boosting Regressor. The performance of the models were compared and finally the XGBoost model was selected for the following reasons, 

1. XGBoost gives more importance to functional space when reducing the cost of a model while Random Forest tries to give more preferences to hyperparameters to optimize the model.
2. XGBoost produced the best R2 score and the lowest MAE among the models without overfitting on the training data. 


## Results

The following are the results of the XGBoost model on the train and the test sets, 

- **Train R2:**  0.9438  
- **Train MAE:**  1012.6
- **Test R2:**  0.9160 
- **Test MAE:** 1169.13

### Residuals Plot

![alt text](https://github.com/Sudhandar/Housing-Energy-Prediction/blob/main/images/XGBoost%20plots.png)


The following image shows the feature importance provided by the XGBoost model which helps in explaining the results to the stakeholders,

![alt text](https://github.com/Sudhandar/Housing-Energy-Prediction/blob/main/images/Feature%20Importance.png)


## Implementation details

The repository contains a jupyter notebook [Energy Prediction](https://github.com/Sudhandar/Housing-Energy-Prediction/blob/main/Analysis/Energy_Consumption.ipynb) which provides the walkthrough of the entire process. 

To replicate the entire process and reproduce the results please implement the following steps,

1. Install the dependencies

`pip install -r requirements.txt`

2. Run the run.py python file using the following command,

`python run.py`

The run.py file triggers the entire pipeline and prints the final results. 

To serve the model predictions using API on your local computer, please implement the following steps,

1. Run the application.py file

`python application.py`

2. The request.py file contains a sample API request with input values. The request.py can be run to test if the API is working or Postman can be used to verify the API.


