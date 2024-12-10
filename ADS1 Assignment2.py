import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


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


# Load the dataset
file_path = r"C:\Users\girij\Downloads\Clustering\Adidas Vs Nike.csv"
data = pd.read_csv(file_path)

# Strip any extra spaces from column names
data.columns = data.columns.str.strip()

# Print the cleaned column names to check
print(data.columns)

# Adjust the column names here based on the output above
x = data['Listing Price'].values.reshape(-1, 1)  # Independent variable (replace with actual column name)
y = data['Sale Price'].values  # Dependent variable (replace with actual column name)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(x, y)

# Make predictions using the model
y_pred = model.predict(x)

# Plotting the scatter plot and regression line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.title('Linear Regression: Listing Price vs Sale Price')
plt.xlabel('Adidas')  # Adjust label based on the column
plt.ylabel('Nike')  # Adjust label based on the column
plt.legend()
plt.show()

# Display the regression coefficients (optional)
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)


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




