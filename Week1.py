import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target


print(df.head())
print("*"*100 + "\n")

print(df.info())
print("*"*100 + "\n")

print(df.describe())
print("*"*100 + "\n")

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["sepal length (cm)"], y=df["sepal width (cm)"], hue=df["target"], palette="viridis")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Scatter Plot of Sepal Length vs Sepal Width")
plt.show()





plt.figure(figsize=(6, 4))
sns.histplot(df["sepal length (cm)"], kde=True, bins=20, color="lightblue")
plt.xlabel("Sepal Length (cm)") #different sepal length on x-axis
plt.ylabel("Count") #basically tells the occurrences of sepal length
plt.title("Distribution of Sepal Length")
plt.show()
