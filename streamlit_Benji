###################################################################################################################################
##################################################     Source Code      ###########################################################
###################################################################################################################################


import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from fpdf import FPDF
import io

# Define scopes
# The 'scopes' variable defines the level of access the application has to Google Sheets and Drive.
# These specific scopes allow the app to read and write data from Google Sheets and Drive, 
# enabling secure, real-time interaction with HR-related datasets.
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Authenticate using Streamlit secrets
# This method uses a JSON configuration stored in Streamlit secrets for authentication.
# It's a secure way to handle credentials, ensuring sensitive data is not hard-coded.
credentials = Credentials.from_service_account_info(
    st.secrets["google_credentials"],
    scopes=scopes
)
client = gspread.authorize(credentials)

# Access Google Sheet
sheet = client.open("Dataset").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)  # Converts the data into a Pandas DataFrame for easier manipulation and analysis.

# Initialize session state for page tracking
# Streamlit's session state is used to handle page navigation, ensuring a smooth and intuitive user experience.
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Home Page
if st.session_state.page == "Home":
    # The Home page introduces the football analytics application
    # and serves as the main navigation hub.

    st.title("⚽ Football Club Performance MOnitor")

    st.write(
        "This application provides a comprehensive overview of player performance, "
        "training engagement, and team composition across all divisions of the club."
    )

    st.write("### Navigate through the application:")
    
    st.write(
        "- **Dashboard:** Explore interactive visualizations of player data, including performance metrics, "
        "attendance trends, positional analysis, and injury distribution."
    )

    st.write(
        "- **Machine Learning:** Use predictive models to estimate player performance based on key indicators "
        "such as training attendance, match involvement, and physical condition."
    )

    st.write(
        "- **Data Management:** Manage player records efficiently by adding, updating, or removing data "
        "across all divisions."
    )

    st.write(
        "- **Player Report:** Generate a detailed report for individual players, including performance, "
        "fitness, and development insights."
    )

    st.write("---")

    st.write(
        "This tool is designed to support coaches and club managers in making informed, data-driven decisions "
        "to optimize player development and team performance."
    )

    # Display navigation buttons under the explanation
    # The buttons provide quick navigation to other sections of the application.
    st.write("### Navigate to:")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Dashboard"):
            st.session_state.page = "Dashboard"
    with col2:
        if st.button("Machine Learning"):
            st.session_state.page = "Machine Learning"
    with col3:
        if st.button("Data Management"):
            st.session_state.page = "Data Management"
    with col4:
        if st.button("Division Report"):
            st.session_state.page = "Employee Report"





########################################### Dashboard Page ###########################################
# Dashboard Page


elif st.session_state.page == "Dashboard":

    st.title("⚽ Player Performance Dashboard")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    # -------------------------------
    # Key Metrics
    # -------------------------------
    st.subheader("Club Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Players", len(df))
    col2.metric("Avg Performance", round(df["PerformanceScore"].mean(), 1))
    col3.metric("Avg Attendance", f"{round(df['TrainingAttendanceRate'].mean(),1)}%")

    # -------------------------------
    # Age Distribution
    # -------------------------------
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax, color='skyblue')
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    # -------------------------------
    # Position Distribution
    # -------------------------------
    st.subheader("Player Distribution by Position")
    position_counts = df['Position'].value_counts()
    fig, ax = plt.subplots()
    position_counts.plot(kind='bar', ax=ax, color='lightblue')
    ax.set_title("Players by Position")
    ax.set_xlabel("Position")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # -------------------------------
    # Goals by Position
    # -------------------------------
    st.subheader("Goals by Position")
    fig, ax = plt.subplots()
    sns.boxplot(x='Position', y='Goals', data=df, ax=ax)
    ax.set_title("Goals Distribution by Position")
    st.pyplot(fig)

    # -------------------------------
    # Performance vs Attendance
    # -------------------------------
    st.subheader("Performance vs Training Attendance")
    fig, ax = plt.subplots()
    sns.scatterplot(x='TrainingAttendanceRate', y='PerformanceScore', data=df, ax=ax)
    sns.regplot(x='TrainingAttendanceRate', y='PerformanceScore', data=df, ax=ax, scatter=False, color='red')
    ax.set_title("Performance vs Attendance")
    st.pyplot(fig)

    # -------------------------------
    # Fitness vs Age
    # -------------------------------
    st.subheader("Fitness vs Age")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='FitnessScore', data=df, ax=ax)
    sns.regplot(x='Age', y='FitnessScore', data=df, ax=ax, scatter=False, color='red')
    ax.set_title("Fitness vs Age")
    st.pyplot(fig)

    # -------------------------------
    # Injury Status Distribution
    # -------------------------------
    st.subheader("Injury Status Distribution")
    injury_counts = df['InjuryStatus'].value_counts()
    fig, ax = plt.subplots()
    injury_counts.plot(kind='bar', ax=ax, color='salmon')
    ax.set_title("Injury Status")
    st.pyplot(fig)

    # -------------------------------
    # Violin Plot Age Distribution by Gender
    # -------------------------------
    st.subheader("Age Distribution (Male vs Female)")
    
    fig, ax = plt.subplots()
    
    sns.violinplot(
        x=["All"] * len(df),   # forces a single category
        y="Age",
        hue="Gender",
        data=df,
        split=True,
        inner="quartile",
        ax=ax
    )
    
    ax.set_xlabel("")
    ax.set_title("Age Distribution Split by Gender")
    
    st.pyplot(fig)


########################################### ML Page ###########################################




# Machine Learning Page and title 
elif st.session_state.page == "Machine Learning":
    st.title("Machine Learning (page under construction)")
   

########################################### Data Management Page ###########################################


# Backend Page (Data Management)
elif st.session_state.page == "Data Management":
    # Title for the data management page, emphasizing its purpose
    st.title("Data Input Manager (page udner construction)")

