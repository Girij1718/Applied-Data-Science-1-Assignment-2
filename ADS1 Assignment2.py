import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# Load the dataset
file_path = r"C:\Users\girij\Downloads\Clustering\Adidas Vs Nike.csv"
data = pd.read_csv(file_path)

# Inspect the first few rows of the dataset
data.head()

# Check for missing values
data.isnull().sum()

# Drop rows with missing values (or impute if necessary)
data = data.dropna()  # Alternatively, use data.fillna() for imputation


# Normalize the numerical columns
scaler = MinMaxScaler()
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])


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


# Let's assume the optimal number of clusters is 3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data[numerical_columns])

# Compute the silhouette score
silhouette_avg = silhouette_score(data[numerical_columns], clusters)
print(f"Silhouette Score for {optimal_k} clusters: {silhouette_avg}")

# Visualize the clusters
plt.scatter(data[numerical_columns[0]], data[numerical_columns[1]], c=clusters, cmap='viridis')
plt.title("K-Means Clustering")
plt.xlabel('Listing_Price')
plt.ylabel('Sale_Price')
plt.show()


def plot_histogram(data, column_name, bins=20, color='blue', alpha=0.7):
  
    plt.hist(data[column_name], bins=bins, color=color, alpha=alpha)
    plt.title(f"Distribution of {column_name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

file_path = r"C:\Users\girij\Downloads\Clustering\Adidas Vs Nike.csv"
data = pd.read_csv(file_path)
plot_histogram(data, 'Sale Price', bins=20, color='blue', alpha=0.7)


# Select only numeric columns
numeric_data = data.select_dtypes(include=['number', 'float64', 'int64'])

# Compute correlation matrix for numeric columns only
corr_matrix = numeric_data.corr()

# Plot heatmap
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    annot_kws={"size": 10, "color": "black"}, 
    cmap="coolwarm",
     center=0, 
    linewidths=0.5, 
    linecolor='black', 
    cbar_kws={'shrink': 0.8}
)
plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()




