import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

dataset = sns.load_dataset("titanic")

#----Checking any missing values in the dataset-----#
# However TITANIC datasets contain missing values#
print("Check if any missing vlaues \n", dataset.isnull().mean())

# 1-----handling missing values in the dataset using pandas----#

# Filling missing values for numerical columns
dataset.update(dataset[['age']].fillna(dataset['age'].median()))  # Median for age
dataset.update(dataset[['fare']].fillna(dataset['fare'].mean()))  # Mean for fare

# Filling missing values for categorical columns
dataset.update(dataset[['embarked']].fillna(dataset['embarked'].mode()[0]))  # Mode for embarked (categorical)
dataset.update(dataset[['embark_town']].fillna(dataset['embark_town'].mode()[0]))  # Mode for embark_town (categorical)

# Have to drop the "deck" column due to too many missing values
dataset.drop(columns=['deck'], inplace=True)                        

# 2--------Feature Scaling------------#
# Select numerical columns for scaling (for better perfomance and to reduce biasing)

num_cols = ["age", "fare"]
# Min-Max Scaling (scales values between 0 and 1)
min_max_scaler = MinMaxScaler()
dataset_scaled = dataset.copy()
dataset_scaled[num_cols] = min_max_scaler.fit_transform(dataset[num_cols])
# Standardization (Z-score scaling: mean = 0, std deviation = 1)
standard_scaler = StandardScaler()
dataset_standardized = dataset.copy()
dataset_standardized[num_cols] = standard_scaler.fit_transform(dataset[num_cols])

# 3--------Encoding Categorical data------------

#one-hot encoding to convert categorical data values into binary values
dataset_encoded = pd.get_dummies(dataset, columns=["class"], drop_first=True)

#label encoding to convert catergorical data values into numeric values(0,1,2,3)
Label_encoder  = LabelEncoder()
dataset["EmbarkedEncoded"] = Label_encoder.fit_transform(dataset["embarked"])

# 4---------Exploratory data analysis (EDA)--------

# Heatmap shows the correlation b/w features & detection of highly correlated and not highly correlated features
plt.figure(figsize=(8, 6))
sns.heatmap(dataset.corr(numeric_only=True),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Heatmap")
plt.show()
#Color Interpretation:
#Red (Positive Correlation) → If one feature increases, the other increases.
#Blue (Negative Correlation) → If one feature increases, the other decreases.
#Close to 0 → Weak or no correlation.

#Pair plot shows relationships between age and fare in the Titanic dataset based on survival column
sns.pairplot(dataset, hue="survived", palette="husl", vars=["age", "fare"])
plt.show()
