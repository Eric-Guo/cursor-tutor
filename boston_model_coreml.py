import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import coremltools as cml

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

# Convert the scikit-learn model to a Core ML model
model = cml.converters.sklearn.convert(lm, ["AveBedrms", "HouseAge", "AveOccup"], "PRICE")

# Assign model metadata
model.author = "Demo Author"
model.license = "MIT"
model.short_description = "Predicts house price in California"

# Assign feature descriptions
model.input_description["AveBedrms"] = "Number of bedrooms"
model.input_description["HouseAge"] = "Proportion of units built pre 1940"
model.input_description["AveOccup"] = "Pupil-teacher ratio by town"

# Assign the output description
model.output_description["PRICE"] = "Median Value in 1k (USD)"

# Save model
model.save('california.mlpackage')