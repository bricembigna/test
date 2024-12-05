import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define scopes
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Authenticate using Streamlit secrets
credentials = Credentials.from_service_account_info(
    st.secrets["google_credentials"],
    scopes=scopes
)
client = gspread.authorize(credentials)

# Access Google Sheet
sheet = client.open("Dataset").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Initialize session state for page tracking
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Home Page
if st.session_state.page == "Home":
    st.title("Welcome to the HR monitor!")
    st.write("This application provides you with in-depth analysis of your HR data in real time.")
    st.write("Choose a section to navigate to for different functionalities:")
    st.write("- **Dashboard:** Check your HR dashboards.")
    st.write("- **Machine Learning:** Leverage AI to make predictions about your workforce (Coming Soon).")
    st.write("- **Backend:** Backend management and settings (Coming Soon).")

    # Display navigation buttons under the explanation
    st.write("### Navigate to:")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Dashboard"):
            st.session_state.page = "Dashboard"
    with col2:
        if st.button("Machine Learning"):
            st.session_state.page = "Machine Learning"
    with col3:
        if st.button("Backend"):
            st.session_state.page = "Backend"

# Dashboard Page - Complete Data Analysis Content
elif st.session_state.page == "Dashboard":
    st.title("Google Sheets Data Analysis")

    # Display 'Back to Home' button
    if st.button("Back to Home"):
        st.session_state.page = "Home"


     # 5. Income Statistics
    st.subheader("Income Statistics")
    mean_income = df['MonthlyIncome'].mean()
    median_income = df['MonthlyIncome'].median()
    std_income = df['MonthlyIncome'].std()
    st.write(f"Mean Monthly Income: ${mean_income:.2f}")
    st.write(f"Median Monthly Income: ${median_income:.2f}")
    st.write(f"Standard Deviation of Monthly Income: ${std_income:.2f}")


    
    # 1. Pie Chart - Percentage of Employees by Department
    st.subheader("Percentage of Employees by Department")
    department_counts = df['Department'].value_counts()
    percentages = department_counts / department_counts.sum() * 100
    plt.figure(figsize=(10, 6))
    plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Employees by Department')
    plt.axis('equal')
    st.pyplot(plt)

    # 2. Violin Plot - Age Distribution by Gender
    st.subheader("Age Distribution by Gender")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Gender', y='Age', data=df, hue='Gender', split=True)
    plt.title('Age Distribution by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Age')
    st.pyplot(plt)

    # 3. Histogram - Age Distribution
    st.subheader("Age Distribution Histogram (Ages 18 to 60)")
    bins = list(range(18, 61))
    plt.figure(figsize=(12, 6))
    plt.hist(df['Age'], bins=bins, edgecolor='white', color='skyblue', alpha=0.7, align='left')
    plt.title('Age Distribution Histogram (Individual Ages 18 to 60)')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.xticks(bins)
    st.pyplot(plt)

    # 4. Bar Chart - Distribution of Employees by Department
    st.subheader("Distribution of Employees by Department")
    plt.figure(figsize=(10, 6))
    department_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Employees by Department')
    plt.xlabel('Department')
    plt.ylabel('Number of Employees')
    plt.xticks(rotation=45)
    st.pyplot(plt)


    # 6. Business Travel Frequency
    st.subheader("Business Travel Frequency")
    business_travel_counts = df['BusinessTravel'].value_counts()
    plt.figure(figsize=(8, 5))
    business_travel_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Business Travel Frequency')
    plt.xlabel('Business Travel Category')
    plt.ylabel('Number of Employees')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # 7. Pie Chart - Business Travel Distribution
    st.subheader("Business Travel Distribution")
    plt.figure(figsize=(8, 8))
    plt.pie(business_travel_counts, labels=business_travel_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Business Travel Distribution')
    plt.axis('equal')
    st.pyplot(plt)

    # 8. Violin Plot - Monthly Income by Job Role
    st.subheader("Distribution of Monthly Income by Job Role")
    plt.figure(figsize=(12, 8))
    sns.violinplot(y='JobRole', x='MonthlyIncome', data=df, scale='count', inner='stick')
    plt.title('Distribution of Monthly Income by Job Role')
    plt.xlabel('Monthly Income')
    plt.ylabel('Job Role')
    st.pyplot(plt)

    # 9. Boxen Plot - Monthly Income by Job Role
    st.subheader("Monthly Income by Job Role (Boxen Plot)")
    plt.figure(figsize=(12, 8))
    sns.boxenplot(y='JobRole', x='MonthlyIncome', data=df)
    plt.title('Distribution of Monthly Income by Job Role')
    plt.xlabel('Monthly Income')
    plt.ylabel('Job Role')
    st.pyplot(plt)

    # 10. Box Plot - Monthly Income by Job Role
    st.subheader("Monthly Income by Job Role (Box Plot)")
    plt.figure(figsize=(12, 8))
    sns.boxplot(y='JobRole', x='MonthlyIncome', data=df)
    plt.title('Distribution of Monthly Income by Job Role')
    plt.xlabel('Monthly Income')
    plt.ylabel('Job Role')
    st.pyplot(plt)

    # 11. Violin Plot - Work-Life Balance by Job Role
    st.subheader("Distribution of Work-Life Balance by Job Role")
    plt.figure(figsize=(12, 8))
    sns.violinplot(y='JobRole', x='WorkLifeBalance', data=df, inner='quartile')
    plt.title('Distribution of Work-Life Balance by Job Role')
    plt.xlabel('Work-Life Balance')
    plt.ylabel('Job Role')
    st.pyplot(plt)

# Machine Learning Page - Placeholder Content
elif st.session_state.page == "Machine Learning":
    st.title("Machine Learning")
    if st.button("Back to Home"):
        st.session_state.page = "Home"
    st.write("Machine Learning content will be added here.")


# Import necessary libraries for data manipulation, visualization, and modeling
import pandas as pd  # For working with tabular data
import numpy as np  # For numerical operations (not used here directly but commonly useful)
import streamlit as st  # For creating an interactive web app
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # For performing linear regression
from sklearn.metrics import mean_squared_error, r2_score  # For evaluating the regression model
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For enhanced data visualization (not used directly but useful for customization)
import gspread
from google.oauth2.service_account import Credentials

# Define scopes
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Authenticate using Streamlit secrets
credentials = Credentials.from_service_account_info(
    st.secrets["google_credentials"],
    scopes=scopes
)
client = gspread.authorize(credentials)

# Access Google Sheet
try:
    sheet = client.open("Dataset").sheet1  # Replace "Dataset" with your Google Sheet name
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
except Exception as e:
    st.error(f"Error accessing Google Sheet: {e}")
    st.stop()

# Set the title of the Streamlit application
st.title("Linear Regression: Total Working Years and Job Level vs. Predicted Monthly Income")

# Validate the data to ensure necessary columns are present
try:
    # Select only the relevant columns for regression and drop any rows with missing values
    df = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
except KeyError as e:
    # Display an error message if any required columns are missing and stop execution
    st.error(f"Missing columns in the dataset: {e}")
    st.stop()

# Define the features (independent variables) and the target (dependent variable)
X = df[['TotalWorkingYears', 'JobLevel']]  # Features: Total Working Years and Job Level
y = df['MonthlyIncome']  # Target: Monthly Income

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)  # Fit the model to the training data

# Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using common regression metrics
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error: Measures average squared difference between actual and predicted values
r2 = r2_score(y_test, y_pred)  # R² Score: Indicates how well the model explains the variance in the target variable

# Display regression results in the Streamlit app
st.subheader("Regression Results")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")
st.write("Model Coefficients (Impact of each feature on the prediction):")
st.write(dict(zip(['TotalWorkingYears', 'JobLevel'], model.coef_)))
st.write(f"Intercept (Base value when all features are zero): {model.intercept_:.2f}")

# Create user input fields for Total Working Years and Job Level
st.subheader("Predict Monthly Income")
user_working_years = st.number_input("Enter Total Working Years:", min_value=0, max_value=40, step=1, value=10)
user_job_level = st.number_input("Enter Job Level:", min_value=1, max_value=5, step=1, value=2)

# Prepare the user's input for prediction
user_input = pd.DataFrame({'TotalWorkingYears': [user_working_years], 'JobLevel': [user_job_level]})
predicted_income = model.predict(user_input)[0]  # Predict Monthly Income for the user's input
st.write(f"Predicted Monthly Income: *{predicted_income:.2f}*")

# Visualize the results using a scatter plot
fig, ax = plt.subplots(figsize=(10, 6))  # Create a plot figure
# Scatter plot: Total Working Years vs. Predicted Monthly Income, with Job Level represented by color
scatter = ax.scatter(
    X_test['TotalWorkingYears'], y_pred, c=X_test['JobLevel'], cmap='viridis', s=50, alpha=0.8
)
ax.set_xlabel("Total Working Years")  # Label for x-axis
ax.set_ylabel("Predicted Monthly Income")  # Label for y-axis
ax.set_title("Predicted Monthly Income vs. Total Working Years (Color: Job Level)")  # Plot title

# Add a color bar to indicate Job Level
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Job Level")  # Label for the color bar

# Highlight the user's input on the scatter plot
ax.scatter(
    user_working_years, predicted_income, color='black', s=100, label='Your Input', zorder=5
)
ax.legend()  # Add a legend to the plot

# Display the plot in the Streamlit app
st.pyplot(fig)


# Backend Page - Placeholder Content
elif st.session_state.page == "Backend":
    st.title("Backend")
    if st.button("Back to Home"):
        st.session_state.page = "Home"
    st.write("Backend management content will be added here.")
