import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define scopes for Google Sheets API
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

# Title for the Streamlit app
st.title("Google Sheets Data Analysis Dashboard")

# Display data preview in an expandable section
with st.expander("Data Preview"):
    st.write(df.head())

# Tabs for different sections of the dashboard
tab1, tab2, tab3 = st.tabs(["Employee Demographics", "Income Analysis", "Work Analysis"])

# Employee Demographics tab
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Percentage of Employees by Department")
        department_counts = df['Department'].value_counts()
        percentages = department_counts / department_counts.sum() * 100
        plt.figure(figsize=(5, 5))
        plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
        plt.title('Percentage of Employees by Department')
        plt.axis('equal')
        st.pyplot(plt)

    with col2:
        st.subheader("Age Distribution by Gender")
        plt.figure(figsize=(5, 5))
        sns.violinplot(x='Gender', y='Age', data=df, hue='Gender', split=True)
        plt.title('Age Distribution by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Age')
        st.pyplot(plt)

    # Add histogram in full width
    st.subheader("Age Distribution Histogram (Ages 18 to 60)")
    bins = list(range(18, 61))
    plt.figure(figsize=(10, 4))
    plt.hist(df['Age'], bins=bins, edgecolor='white', color='skyblue', alpha=0.7, align='left')
    plt.title('Age Distribution Histogram (Individual Ages 18 to 60)')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.xticks(bins)
    st.pyplot(plt)

# Income Analysis tab
with tab2:
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribution of Monthly Income by Job Role")
        plt.figure(figsize=(5, 5))
        sns.violinplot(y='JobRole', x='MonthlyIncome', data=df, scale='count', inner='stick')
        plt.title('Distribution of Monthly Income by Job Role')
        plt.xlabel('Monthly Income')
        plt.ylabel('Job Role')
        st.pyplot(plt)

    with col4:
        st.subheader("Monthly Income by Job Role (Box Plot)")
        plt.figure(figsize=(5, 5))
        sns.boxplot(y='JobRole', x='MonthlyIncome', data=df)
        plt.title('Distribution of Monthly Income by Job Role')
        plt.xlabel('Monthly Income')
        plt.ylabel('Job Role')
        st.pyplot(plt)

    # Income statistics
    st.subheader("Income Statistics")
    mean_income = df['MonthlyIncome'].mean()
    median_income = df['MonthlyIncome'].median()
    std_income = df['MonthlyIncome'].std()
    st.write(f"Mean Monthly Income: ${mean_income:.2f}")
    st.write(f"Median Monthly Income: ${median_income:.2f}")
    st.write(f"Standard Deviation of Monthly Income: ${std_income:.2f}")

# Work Analysis tab
with tab3:
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Business Travel Frequency")
        business_travel_counts = df['BusinessTravel'].value_counts()
        plt.figure(figsize=(5, 5))
        business_travel_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
        plt.title('Business Travel Frequency')
        plt.xlabel('Business Travel Category')
        plt.ylabel('Number of Employees')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    with col6:
        st.subheader("Business Travel Distribution")
        plt.figure(figsize=(5, 5))
        plt.pie(business_travel_counts, labels=business_travel_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Business Travel Distribution')
        plt.axis('equal')
        st.pyplot(plt)

    # Full-width scatter plot
    st.subheader("Years at Company vs. Years Since Last Promotion (Attrition Highlighted)")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='YearsAtCompany', y='YearsSinceLastPromotion', hue='Attrition', data=df, palette='coolwarm', alpha=0.7)
    plt.title('Years at Company vs. Years Since Last Promotion (Attrition Highlighted)')
    plt.xlabel('Years at Company')
    plt.ylabel('Years Since Last Promotion')
    plt.legend(title='Attrition')
    st.pyplot(plt)

# Additional Analysis and Description
with st.expander("Additional Information and Analysis"):
    st.write("""
    This dashboard provides insights into employee demographics, work-life balance, and income distribution. 
    Each chart helps explore factors such as age distribution by gender, income by job role, and the effect of business travel on employee distribution.

    **For detailed interpretations, refer to individual charts and explore the data.**
    """)
