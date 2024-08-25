import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass
from sklearn.utils import resample

# Function to remove outliers based on IQR
# widely used technique that effectively identifies and excludes extreme values
# without being overly influenced by the presence of outliers
def outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Loading dataset from kaggle
df = pd.read_csv("Electricity.csv", low_memory=False)


# Replacing "?" with NaN standardizes missing data handling, allowing for consistent data processing and analysis
df.replace("?", np.nan, inplace=True)

# Convert columns to numeric
columns_to_convert = [
    "ForecastWindProduction", "SystemLoadEA", "SMPEA",
    "ORKTemperature", "ORKWindspeed", "CO2Intensity",
    "ActualWindProduction", "SystemLoadEP2", "SMPEP2"
]
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Remove outliers from the columns used in the model
# Filling missing values with forward fill method
df = outliers(df, columns_to_convert)
df.ffill(inplace=True)

# Select relevant features and target
features = [
    "HolidayFlag", "DayOfWeek", "Day", "Month",
    "PeriodOfDay", "ForecastWindProduction", "SystemLoadEA",
    "ORKTemperature", "ORKWindspeed", "CO2Intensity",
    "ActualWindProduction", "SystemLoadEP2", "SMPEP2"
]
target = "SMPEA"

X = df[features]
y = df[target]

# Split the dataset into training and testing sets, 20 percent is used for training.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalance in the training set for HolidayFlag
# Separate the training data by class
x_train_non_holiday = x_train[x_train['HolidayFlag'] == 0]
x_train_holiday = x_train[x_train['HolidayFlag'] == 1]
y_train_non_holiday = y_train[x_train['HolidayFlag'] == 0]
y_train_holiday = y_train[x_train['HolidayFlag'] == 1]

# Upsample the minority class (holiday days)
# The original training dataset was imbalanced, with the minority class
# (holiday days) being underrepresented compared to the majority class (non holidays)
x_train_holiday_upsampled, y_train_holiday_upsampled = resample(
    x_train_holiday, y_train_holiday,
    replace=True,  # sample with replacement
    n_samples=len(x_train_non_holiday),  # match number of non-holiday observations
    random_state=42
)
# This was important to Avoid Bias Towards the Majority Class
# Combine the upsampled minority class with the majority class
x_train_balanced = pd.concat([x_train_non_holiday, x_train_holiday_upsampled])
y_train_balanced = pd.concat([y_train_non_holiday, y_train_holiday_upsampled])


# Shuffle the combined dataset
# to ensure that the training data is randomly mixed before the model is trained.
# If the training data is ordered in a particular way (e.g., all similar instances grouped together),
# the model might learn patterns that are specific to that order rather than generalizable patterns.
x_train_balanced, y_train_balanced = sklearn.utils.shuffle(x_train_balanced, y_train_balanced, random_state=42)

@dataclass(eq=True, frozen=True, order=True)
class electricity_model_prediction:
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(x_train_balanced, y_train_balanced)
    y_pred = model.predict(x_test)
    model_accuracy = model.score(x_test, y_test)

    # Calculate SMAPE
    # Using this, gives me an idea of how well the prediction model is performing,and helped me find out if it needs fine tuning.
    def calculate_smape(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred) / denominator
        smape = np.mean(diff) * 100  # Convert to percentage
        return 100 - smape

    total_accuracy = calculate_smape(None, y_test, y_pred)

# Create an instance of the class to train the model and calculate accuracy
model_instance = electricity_model_prediction()
print(f"Model Accuracy (R^2): {model_instance.model_accuracy:.4f}")
print(f"Total SMAPE-based Accuracy: {model_instance.total_accuracy:.2f}%")