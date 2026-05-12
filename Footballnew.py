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

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Football Club Performance Monitor",
    page_icon="⚽",
    layout="wide"
)

# --------------------------------------------------
# GLOBAL STYLE — TACTICAL FOOTBALL DASHBOARD THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07140d 0%, #0e1f16 45%, #111827 100%);
    color: white;
}

h1, h2, h3 {
    color: #ffffff;
    font-weight: 700;
}

p, li, span, div {
    color: #d1d5db;
}

[data-testid="stMarkdownContainer"] {
    color: #d1d5db;
}

[data-testid="metric-container"] {
    background-color: rgba(17, 24, 39, 0.95);
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.35);
}

[data-testid="metric-container"] label {
    color: #a7f3d0 !important;
}

[data-testid="metric-container"] div {
    color: #ffffff !important;
}

.stButton > button {
    background-color: #14532d;
    color: white;
    border: 1px solid #22c55e;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #16a34a;
    color: white;
    border: 1px solid #86efac;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

hr {
    border-color: rgba(34, 197, 94, 0.35);
}

[data-testid="stHeader"] {
    background: rgba(7, 20, 13, 0);
}

[data-testid="stToolbar"] {
    color: white;
}

.stSlider label,
.stSelectbox label,
.stNumberInput label {
    color: #d1d5db !important;
    font-weight: 600;
}

div[data-baseweb="select"] > div {
    background-color: #111827;
    color: white;
    border-color: #22c55e;
}

input {
    background-color: #111827 !important;
    color: white !important;
    border-color: #22c55e !important;
}

.stAlert {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)




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

    st.title("⚽ Football Club Performance Monitor")

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

    # ==================================================
    # ROW 1 → 3 PLOTS
    # ==================================================

    col1, col2, col3 = st.columns(3)

    # -------------------------------
    # Age Distribution
    # -------------------------------
    with col1:
        st.markdown("#### Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], bins=20, kde=True, ax=ax, color='skyblue')
        ax.set_title("Age Distribution")
        st.pyplot(fig)

    # -------------------------------
    # Position Distribution
    # -------------------------------
    with col2:
        st.markdown("#### Player Distribution by Position")
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
    with col3:
        st.markdown("#### Goals by Position")
        fig, ax = plt.subplots()
        sns.boxplot(x='Position', y='Goals', data=df, ax=ax)
        ax.set_title("Goals Distribution by Position")
        st.pyplot(fig)

    # ==================================================
    # ROW 2 → 3 PLOTS
    # ==================================================

    col4, col5, col6 = st.columns(3)

    # -------------------------------
    # Performance vs Attendance
    # -------------------------------
    with col4:
        st.markdown("#### Performance vs Training Attendance")
        fig, ax = plt.subplots()
        sns.scatterplot(x='TrainingAttendanceRate', y='PerformanceScore', data=df, ax=ax)
        sns.regplot(x='TrainingAttendanceRate', y='PerformanceScore', data=df, ax=ax, scatter=False, color='red')
        ax.set_title("Performance vs Attendance")
        st.pyplot(fig)

    # -------------------------------
    # Fitness vs Age
    # -------------------------------
    with col5:
        st.markdown("#### Fitness vs Age")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Age', y='FitnessScore', data=df, ax=ax)
        sns.regplot(x='Age', y='FitnessScore', data=df, ax=ax, scatter=False, color='red')
        ax.set_title("Fitness vs Age")
        st.pyplot(fig)

    # -------------------------------
    # Injury Status Distribution
    # -------------------------------
    with col6:
        st.markdown("#### Injury Status Distribution")
        injury_counts = df['InjuryStatus'].value_counts()
        fig, ax = plt.subplots()
        injury_counts.plot(kind='bar', ax=ax, color='salmon')
        ax.set_title("Injury Status")
        st.pyplot(fig)

    st.subheader("Female Division")
    st.subheader("Male Division")


########################################### ML Page ###########################################




# Machine Learning Page and title 
# Machine Learning Page
elif st.session_state.page == "Machine Learning":

    st.title("⚽ Player Performance Predictor")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    st.write(
        "This section uses live player data from Google Sheets to predict a player's "
        "performance score based on age, training attendance, fitness score, and goals."
    )

    # Required columns for machine learning
    FEATURES = ["Age", "TrainingAttendanceRate", "FitnessScore", "Goals"]
    TARGET = "PerformanceScore"

    required_columns = FEATURES + [TARGET]

    # Check whether all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing columns in Google Sheet: {missing_columns}")
        st.stop()

    # Prepare ML dataset from live Google Sheets data
    ml_df = df[required_columns].copy()

    # Convert values to numeric, in case Google Sheets imported them as text
    for col in required_columns:
        ml_df[col] = pd.to_numeric(ml_df[col], errors="coerce")

    # Remove rows with missing or invalid values
    ml_df = ml_df.dropna()

    if len(ml_df) < 10:
        st.warning("Not enough valid data to train the model. Please add more player records.")
        st.stop()

    X = ml_df[FEATURES]
    y = ml_df[TARGET]

    # Import machine learning libraries
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    import numpy as np

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Standardise the input variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN regression model
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Rows", len(X_train))
    col2.metric("R² Score", f"{r2:.2f}")
    col3.metric("Error", f"{mae:.1f}")

    st.write("---")

    st.subheader("Try Prediction")

    input_age = st.slider("Age", 15, 50, 25)
    input_attendance = st.slider("Training Attendance Rate (%)", 0.0, 100.0, 80.0)
    input_fitness = st.slider("Fitness Score", 0.0, 100.0, 75.0)
    input_goals = st.slider("Goals", 0, 50, 5)

    input_array = np.array([[input_age, input_attendance, input_fitness, input_goals]])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Performance Score: {prediction:.1f}")
   

########################################### Data Management Page ###########################################


# ─────────────────────────────────────────────────────────────────
# AI Usage Declaration – Data Management Page
# Author: Fabian
# This section was developed with the support of Claude (Anthropic).
# Claude was used as a coding assistant to help write the
# Data Management page including the add, edit and delete functions.
# Reference: Claude (Anthropic), claude.ai, May 2026
# ─────────────────────────────────────────────────────────────────

# Backend Page (Data Management)
elif st.session_state.page == "Data Management":
    st.title("⚽ Player Data Manager")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    worksheet = client.open("Dataset").sheet1

    if "dm_subpage" not in st.session_state:
        st.session_state.dm_subpage = "main"

    if st.session_state.dm_subpage == "main":
        st.subheader("What would you like to do?")

        if st.button("➕ Add New Player"):
            st.session_state.dm_subpage = "add"

        if st.button("✏️ Edit Player"):
            st.session_state.dm_subpage = "edit"

        if st.button("🗑️ Delete Player"):
            st.session_state.dm_subpage = "delete"

    elif st.session_state.dm_subpage == "add":
        st.subheader("➕ Add New Player")

        with st.form("add_form"):
            name = st.text_input("Name")
            gender = st.selectbox("Gender", ["M", "F"])
            age = st.number_input("Age", min_value=8, max_value=50, step=1)
            division = st.selectbox("Division", ["U11", "U15", "U19", "Senior", "Veteran"])
            position = st.selectbox("Position", ["GK", "DF", "MF", "FW"])
            attendance = st.slider("Training Attendance Rate (%)", 0, 100, 85)
            matches = st.number_input("Matches Played", min_value=0, max_value=28, step=1)
            sessions = st.number_input("Training Sessions", min_value=0, max_value=85, step=1)
            goals = st.number_input("Goals", min_value=0, step=1)
            assists = st.number_input("Assists", min_value=0, step=1)
            performance = st.number_input("Performance Score", min_value=0, max_value=400, value=75, step=1)
            fitness = st.slider("Fitness Score", 0, 100, 75)
            injury = st.selectbox("Injury Status", ["Healthy", "Minor Injury", "Injured"])
            years = st.number_input("Years At Club", min_value=0, step=1)

            submitted = st.form_submit_button("Add Player")

            if submitted:
                all_data = worksheet.get_all_values()
                last_player_id = all_data[-1][0]
                last_number = int(last_player_id.replace("P", ""))
                new_player_id = f"P{last_number + 1:04d}"

                worksheet.append_row([new_player_id, name, gender, age, division, position, attendance, matches, sessions, goals, assists, performance, fitness, injury, years])

                st.success(f"✅ Player {new_player_id} added successfully!")

        if st.button("Back"):
            st.session_state.dm_subpage = "main"


    elif st.session_state.dm_subpage == "edit":
        st.subheader("✏️ Edit Player")
    
        all_data = worksheet.get_all_values()
        rows = all_data[1:]
    
        if rows:
            row_labels = [f"{row[0]} - {row[1]}" for row in rows]
            selected = st.selectbox("Select a player to edit", row_labels)
            idx = row_labels.index(selected)
            current = rows[idx]
    
            with st.form("edit_form"):
                player_id = current[0]
                name = st.text_input("Name", value=current[1])
                gender = st.selectbox("Gender", ["M", "F"], index=["M", "F"].index(current[2]) if current[2] in ["M", "F"] else 0)
                age = st.number_input("Age", min_value=8, max_value=50, value=int(current[3]) if current[3] else 20, step=1)
                division = st.selectbox("Division", ["U11", "U15", "U19", "Senior", "Veteran"], index=["U11", "U15", "U19", "Senior", "Veteran"].index(current[4]) if current[4] in ["U11", "U15", "U19", "Senior", "Veteran"] else 0)
                position = st.selectbox("Position", ["GK", "DF", "MF", "FW"], index=["GK", "DF", "MF", "FW"].index(current[5]) if current[5] in ["GK", "DF", "MF", "FW"] else 0)
                attendance = st.slider("Training Attendance Rate (%)", 0, 100, int(current[6]) if current[6] else 85)
                matches = st.number_input("Matches Played", min_value=0, max_value=28, value=int(current[7]) if current[7] else 0, step=1)
                sessions = st.number_input("Training Sessions", min_value=0, max_value=85, value=int(current[8]) if current[8] else 0, step=1)
                goals = st.number_input("Goals", min_value=0, value=int(current[9]) if current[9] else 0, step=1)
                assists = st.number_input("Assists", min_value=0, value=int(current[10]) if current[10] else 0, step=1)
                performance = st.number_input("Performance Score", min_value=0, max_value=400, value=int(current[11]) if current[11] else 75, step=1)
                fitness = st.slider("Fitness Score", 0, 100, int(current[12]) if current[12] else 75)
                injury = st.selectbox("Injury Status", ["Healthy", "Minor Injury", "Injured"], index=["Healthy", "Minor Injury", "Injured"].index(current[13]) if current[13] in ["Healthy", "Minor Injury", "Injured"] else 0)
                years = st.number_input("Years At Club", min_value=0, value=int(current[14]) if current[14] else 0, step=1)
                if st.form_submit_button("Save Changes"):
                    row_number = idx + 2
                    worksheet.update(f"A{row_number}:O{row_number}", [[player_id, name, gender, age, division, position, attendance, matches, sessions, goals, assists, performance, fitness, injury, years]])
                    st.success("✅ Player updated successfully!")
    
        else:
            st.warning("No players found in the sheet.")
    
        if st.button("Back"):
            st.session_state.dm_subpage = "main"

    elif st.session_state.dm_subpage == "delete":
        st.subheader("🗑️ Delete Player")

        all_data = worksheet.get_all_values()
        rows = all_data[1:]

        if rows:
            row_labels = [f"{row[0]} - {row[1]}" for row in rows]
            selected = st.selectbox("Select a player to delete", row_labels)
            idx = row_labels.index(selected)

            if st.button("🗑️ Confirm Delete"):
                worksheet.delete_rows(idx + 2)
                st.success("✅ Player deleted successfully!")
        else:
            st.warning("No players found in the sheet.")

        if st.button("Back"):
            st.session_state.dm_subpage = "main"
