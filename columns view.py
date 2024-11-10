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
        st.subheader("Distribution of Monthly
