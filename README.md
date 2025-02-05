# Boston Housing Project

## Overview

This project involves analyzing housing data for California districts to predict the median house value. The dataset used here is the California Housing dataset, sourced from the UCI machine learning repository. It consists of various features, such as median income, average rooms per household, and latitude/longitude, which are used to predict the median house value (target variable).

The data is pre-processed, analyzed, and explored to uncover meaningful insights. Machine learning techniques, including data scaling and train-test splitting, are employed to prepare the data for prediction tasks.

## Dataset Description

The dataset contains the following attributes:
1. **MedInc**: Median income in the block group.
2. **HouseAge**: Median house age in the block group.
3. **AveRooms**: Average number of rooms per household.
4. **AveBedrms**: Average number of bedrooms per household.
5. **Population**: Block group population.
6. **AveOccup**: Average number of household members.
7. **Latitude**: Latitude of the block group.
8. **Longitude**: Longitude of the block group.
9. **Price (target variable)**: Median house value in the block group (in hundreds of thousands of dollars).

### Data Source
The dataset was derived from the 1990 U.S. census, with one row representing a census block group. A block group typically contains between 600 and 3,000 people. The dataset can be accessed via the `fetch_california_housing` function from scikit-learn.

## Requirements

This project requires the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Workflow

1. **Loading the Dataset**: The dataset is loaded using the `fetch_california_housing` function from `sklearn.datasets`.

2. **Data Preprocessing**: The data is converted into a pandas DataFrame. The target variable, `Price`, is added to the dataset.

3. **Exploratory Data Analysis (EDA)**:
   - Descriptive statistics are computed to understand the distribution and summary of the data.
   - Missing values are checked (none in this case).
   - Correlations between the features and the target variable are examined.
   - Various visualizations such as scatter plots and pair plots are generated to explore relationships.

4. **Feature Selection**: The dataset is split into independent (`X`) and dependent (`y`) variables, with `y` being the target variable (Price).

5. **Train-Test Split**: The data is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.

6. **Data Standardization**: The features are standardized using `StandardScaler` from `sklearn.preprocessing` to bring all values into a similar range, helping improve the performance of machine learning models.

## Key Code Snippets

### Loading the Dataset

```python
from sklearn.datasets import fetch_california_housing
boston = fetch_california_housing()
```

### Checking Dataset Description

```python
print(boston.DESCR)
```

### Preparing the Dataset

```python
import pandas as pd
dataset = pd.DataFrame(boston.data, columns=boston.feature_names)
dataset['Price'] = boston.target
```

### Exploratory Data Analysis

```python
import seaborn as sns
sns.pairplot(dataset)
```

### Train-Test Split

```python
from sklearn.model_selection import train_test_split
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Data Standardization

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Expected Outcomes

This project explores the correlation between various features of the dataset and the target variable (house prices). It provides insights into which factors most strongly influence house prices in California.

Through visualizations and statistical analysis, you should be able to identify:
- Relationships between features such as median income, house age, and population.
- Patterns in the geographical coordinates (latitude and longitude) and their effect on house prices.

## References

- Dataset Source: [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## Conclusion

This project provides a foundation for performing exploratory data analysis on a housing dataset and applying machine learning techniques for regression tasks. By preparing the data, exploring it, and splitting it into training/testing sets, we can later implement more sophisticated predictive models to estimate house prices based on the provided features.
