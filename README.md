# Boston Housing Project README

## Overview
This project involves analyzing housing data to predict the median house value in California districts using machine learning techniques. The dataset used is the **California Housing dataset**, sourced from the UCI machine learning repository. It contains several features, such as median income, average rooms per household, and latitude/longitude, which are used to predict the median house value (the target variable).

The project consists of data preprocessing, exploratory data analysis (EDA), model training, and web application deployment to provide real-time predictions.

![image_alt](https://github.com/iamanirudhnair/Boston_House_Pricing/blob/main/Screenshot%202025-03-08%20091134.png?raw=true)

![image_alt](https://github.com/iamanirudhnair/Boston_House_Pricing/blob/main/Screenshot%202025-03-08%20091151.png?raw=true)

## Dataset Description
The dataset contains the following features:
- **MedInc**: Median income in the block group.
- **HouseAge**: Median house age in the block group.
- **AveRooms**: Average number of rooms per household.
- **AveBedrms**: Average number of bedrooms per household.
- **Population**: Block group population.
- **AveOccup**: Average number of household members.
- **Latitude**: Latitude of the block group.
- **Longitude**: Longitude of the block group.
- **Price**: Median house value in the block group (target variable, in hundreds of thousands of dollars).

## Data Source
The dataset is based on the **1990 U.S. census**, with one row representing a census block group, which typically contains between 600 and 3,000 people. It can be accessed via the `fetch_california_housing` function from `scikit-learn`.

## Requirements
This project requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `flask`

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask
```

## Project Workflow

### 1. Loading the Dataset
The dataset is loaded using the `fetch_california_housing` function from `sklearn.datasets`.

```python
from sklearn.datasets import fetch_california_housing
boston = fetch_california_housing()
```

### 2. Data Preprocessing
The dataset is converted into a pandas DataFrame, and the target variable, `Price`, is added to the dataset.

```python
import pandas as pd
dataset = pd.DataFrame(boston.data, columns=boston.feature_names)
dataset['Price'] = boston.target
```

### 3. Exploratory Data Analysis (EDA)
- Descriptive statistics are computed to understand the distribution of data.
- Missing values are checked (none found).
- Correlations between features and the target variable are analyzed.
- Visualizations, including scatter plots and pair plots, are used to identify relationships in the data.

```python
import seaborn as sns
sns.pairplot(dataset)
```

### 4. Feature Selection
The dataset is split into independent (X) and dependent (y) variables, with `y` representing the target variable, `Price`.

### 5. Train-Test Split
The data is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.

```python
from sklearn.model_selection import train_test_split
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 6. Data Standardization
The features are standardized using `StandardScaler` to bring all values into a similar range, improving the performance of machine learning models.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 7. Model Training
Various machine learning models can be employed to predict the median house value. For this project, we used regression models such as Linear Regression, Decision Trees, or Random Forest.

### 8. Performance Metrics
The model's performance is evaluated using metrics such as Mean Squared Error (MSE) and R-squared (R²) to assess the prediction accuracy.

### 9. Prediction of New Data
Once the model is trained, it can be used to predict house prices for new, unseen data.

### 10. Pickling the Model
The trained model is saved using Python’s `pickle` module for future use or deployment.

```python
import pickle
with open('housing_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 11. Web Application with Flask
The project also includes a web application built using Flask. This application allows users to input new data and receive predictions in real time.

```bash
# Run the Flask application
python app.py
```

### 12. Testing and Running the Application
The web application is tested to ensure it works seamlessly for real-time predictions.

## What I Learned
- **Data Preprocessing**: I gained hands-on experience in data cleaning, handling missing values, and scaling the dataset for machine learning.
- **Exploratory Data Analysis**: By using various visualization techniques (like scatter plots and pair plots), I learned how to uncover insights from the dataset.
- **Model Training**: I learned how to train regression models and evaluate them using different performance metrics.
- **Flask Web Application**: I developed a web application using Flask that allows users to interact with the trained model.

## Techniques Used
- **Data Scaling**: StandardScaler was used to standardize the features, ensuring that they are within the same range, which is crucial for many machine learning algorithms.
- **Train-Test Split**: Data splitting ensures that the model generalizes well and doesn't overfit the training data.
- **Machine Learning Models**: Various regression techniques were employed, such as Linear Regression and Decision Trees.
- **Web Application with Flask**: Flask was used to create an API for the model, allowing users to predict house prices via a simple interface.

## Challenges and Solutions

### 1. **Data Standardization**
- **Challenge**: The initial model performance was low due to varying scales of features.
- **Solution**: Using `StandardScaler` to standardize the data significantly improved model performance.

### 2. **Model Performance**
- **Challenge**: The initial model (Linear Regression) didn’t yield great results due to its simplicity.
- **Solution**: I experimented with more complex models such as Decision Trees and Random Forest, which improved the accuracy.

### 3. **Deployment**
- **Challenge**: Integrating the machine learning model with a Flask web application was challenging.
- **Solution**: After reading relevant Flask documentation, I successfully created an API for real-time predictions. Using `pickle` to save the model was an essential step for deployment.

## Expected Outcomes
By the end of this project, we should be able to:
- Understand the relationship between different features and house prices.
- Develop a regression model to predict the median house value.
- Deploy the model in a Flask web application for real-time predictions.

## Conclusion
This project provides a foundational understanding of performing **Exploratory Data Analysis (EDA)**, **preprocessing** datasets, and applying **machine learning** techniques for **regression tasks**. Additionally, it demonstrates how to deploy a model using **Flask** for web-based prediction applications.

## References
- Dataset Source: [California Housing Dataset](https://archive.ics.uci.edu/ml/datasets/California+Housing)
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- Flask Documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
