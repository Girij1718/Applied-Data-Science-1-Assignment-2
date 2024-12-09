import pandas as pd

# Load the dataset
file_path = r"C:\Users\girij\Downloads\Clustering\Adidas Vs Nike.csv"
data = pd.read_csv(file_path)

# Inspect the first few rows of the dataset
data.head()

# Check for missing values
data.isnull().sum()

# Drop rows with missing values (or impute if necessary)
data = data.dropna()  # Alternatively, use data.fillna() for imputation

from sklearn.preprocessing import MinMaxScaler

# Normalize the numerical columns
scaler = MinMaxScaler()
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Elbow Method
distortions = []
for k in range(1, 11):  # Try from 1 to 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[numerical_columns])  # Fit only numerical data
    distortions.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 11), distortions, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion")
plt.show()

