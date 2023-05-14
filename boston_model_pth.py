import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import torch

# Load the California housing dataset
boston = fetch_california_housing()
boston_df = pd.DataFrame(boston["data"])
boston_df.columns = boston["feature_names"]
boston_df["PRICE"] = boston["target"]

# Prepare the input features and target variable
y = boston_df["PRICE"]
X = boston_df[["AveBedrms", "HouseAge", "AveOccup"]]

# Train a linear regression model
lm = LinearRegression()
lm.fit(X, y)

torch.save(lm, "california.pth")