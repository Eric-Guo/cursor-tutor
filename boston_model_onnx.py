import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

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

# Convert the scikit-learn model to ONNX
onnx_model = convert_sklearn(lm, initial_types=[("float_input", FloatTensorType([None, 3]))])

# Save the ONNX model
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())