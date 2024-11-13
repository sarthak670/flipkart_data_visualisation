import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\Harshada\Downloads\flipkart_data\flipkart_com-ecommerce_sample.csv")  # Change path as necessary

# Display last 10 rows
last_x_rows = df.tail(10)
print("Last 10 entries:")
print(last_x_rows)

# Display first 10 rows
first_x_rows = df.head(10)
print("First 10 entries:")
print(first_x_rows)

# Check for NaN or infinite values in the dataset and handle them
# We drop rows where any NaN values are present
df = df.dropna(subset=['product_category_tree', 'retail_price', 'discounted_price'])

# Alternatively, if you want to fill NaNs instead of dropping them, you can use:
# df['discounted_price'] = df['discounted_price'].fillna(df['discounted_price'].mean())  # Fill with mean
# df['retail_price'] = df['retail_price'].fillna(df['retail_price'].mean())  # Fill with mean

# Check for inf values and replace them with NaN (this is a common issue if you have infinite values)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values after replacing inf values
df = df.dropna(subset=['discounted_price', 'retail_price'])

# Encode 'product_category_tree' into numeric values (LabelEncoder)
le = LabelEncoder()
df['encoded_category'] = le.fit_transform(df['product_category_tree'].astype(str))  # Encode the category

# Pearson correlation between 'encoded_category' and 'retail_price'
corr = pearsonr(df['encoded_category'], df['retail_price'])
print("Pearson correlation between encoded category and retail price:", corr)

# Linear regression model
X = df[['encoded_category']].values.reshape(-1, 1)  # Independent variable (encoded category)
y = df['retail_price'].values  # Dependent variable (retail price)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict retail prices based on the encoded category
predicted_ratings = model.predict(X)

# Plot the regression line and data points
plt.scatter(df['encoded_category'], df['retail_price'], color='blue', label='Actual Retail Price')
plt.plot(df['encoded_category'], predicted_ratings, color='red', label='Predicted Retail Price')
plt.xlabel('Encoded product_category_tree')
plt.ylabel('Retail Price')
plt.title('Product Category Tree vs Retail Price with Regression Line')
plt.legend()
plt.show()

# Print the model's slope (coefficient) and intercept
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot scatter plot with the encoded 'product_category_tree'
plt.scatter(df['encoded_category'], df['retail_price'])
plt.xlabel('Encoded product_category_tree')
plt.ylabel('Retail Price')
plt.title('Scatter plot: Product Category vs Retail Price')
plt.show()

# Plot histogram of 'discounted_price'
plt.hist(df['discounted_price'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Discounted Prices')
plt.xlabel('Discounted Price')
plt.ylabel('Frequency')
plt.show()

# Box plot by 'product_category_tree'
df.boxplot(by='product_category_tree', column="retail_price", grid=False)
plt.title('Box Plot of Retail Price by Product Category')
plt.show()