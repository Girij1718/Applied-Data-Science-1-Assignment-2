
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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

def plot_elbow_curve(data, numerical_columns, max_clusters=10):
    """
    Plots the Elbow Curve to determine the optimal number of clusters for KMeans.
    
    Parameters:
    data (DataFrame): The input DataFrame containing the data.
    numerical_columns (list): List of column names to be used for clustering.
    max_clusters (int): The maximum number of clusters to try (default is 10).
    """
    distortions = []  # To store inertia values for each k
    for k in range(1, max_clusters + 1):  # Try from 1 to max_clusters clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[numerical_columns])  # Fit only the specified numerical columns
        distortions.append(kmeans.inertia_)  # Store the inertia value
    
    # Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title("Elbow Method to Find Optimal Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.show()


# Load the dataset
file_path = r"C:\Users\girij\Downloads\Clustering\Adidas Vs Nike.csv"
data = pd.read_csv(file_path)

# Strip any extra spaces from column names
data.columns = data.columns.str.strip()

# Identify the numerical columns (for clustering, only numerical columns are relevant)
numerical_columns = data.select_dtypes(include=['number', 'float64', 'int64']).columns

# Call the function to plot the Elbow Curve
plot_elbow_curve(data, numerical_columns, max_clusters=10)


from sklearn.metrics import silhouette_score

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

# Function
def plot_correlation_heatmap(data):
 
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

# Replace 'file_path' with the path to your CSV file
file_path = r"C:\Users\girij\Downloads\Clustering\Adidas Vs Nike.csv"
data = pd.read_csv(file_path)

# Call the function to plot the heatmap
plot_correlation_heatmap(data)




