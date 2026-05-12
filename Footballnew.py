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

.stDownloadButton > button {
    background-color: #14532d;
    color: white !important;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 600;
}

.stDownloadButton > button:hover {
    background-color: white;
    color: #14532d !important;
    border: 1px solid #22c55e;
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

    st.markdown("""
### Modern Football Analytics for Data-Driven Clubs

Monitor player development, evaluate performance trends, and generate actionable insights across every division of the club through a centralized analytics platform.

This application combines performance tracking, squad analytics, machine learning, and reporting capabilities into a single operational dashboard designed for coaches, analysts, and club management.

### Platform Capabilities

#### 📊 Interactive Analytics Dashboard
Explore real-time visualizations covering:
- Player performance trends
- Training attendance patterns
- Positional distributions
- Fitness and injury monitoring
- Division-level comparisons

#### 🤖 Machine Learning Predictions
Estimate player performance scores using predictive modelling based on:
- Attendance consistency
- Physical fitness
- Match participation
- Offensive contribution

#### ⚙️ Player Data Management
Maintain and update club records efficiently through:
- Player creation and editing
- Squad database management
- Live synchronization with Google Sheets

#### 📄 Automated PDF Reporting
Generate downloadable professional reports for:
- Overall squad performance
- Female division analytics
- Male division analytics

---

This platform is designed to support evidence-based decision making and provide a clearer understanding of player progression, squad dynamics, and overall club performance.
""")

    
    
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
            st.session_state.page = "Division Report"





########################################### Dashboard Page ###########################################
# Dashboard Page

elif st.session_state.page == "Dashboard":

    st.title("⚽ Player Performance Dashboard")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    # -------------------------------
    # Helper function for dashboards
    # -------------------------------
    def plot_dashboard(dataframe, title):

        st.markdown("---")
        st.title(title)

        if dataframe.empty:
            st.warning("No data available for this section.")
            return

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total Players", len(dataframe))
        metric2.metric("Avg Performance", round(dataframe["PerformanceScore"].mean(), 1))
        metric3.metric("Avg Attendance", f"{round(dataframe['TrainingAttendanceRate'].mean(), 1)}%")

        row1_col1, row1_col2, row1_col3 = st.columns(3)

        with row1_col1:
            st.markdown("#### Age Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(dataframe["Age"], bins=20, kde=True, ax=ax, color="skyblue")
            ax.set_title("Age Distribution")
            st.pyplot(fig, use_container_width=True)

        with row1_col2:
            st.markdown("#### Position Distribution")
            position_counts = dataframe["Position"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            position_counts.plot(kind="bar", ax=ax, color="lightblue")
            ax.set_title("Players by Position")
            ax.set_xlabel("Position")
            ax.set_ylabel("Count")
            st.pyplot(fig, use_container_width=True)

        with row1_col3:
            st.markdown("#### Goals by Position")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(x="Position", y="Goals", data=dataframe, ax=ax)
            ax.set_title("Goals by Position")
            st.pyplot(fig, use_container_width=True)

        row2_col1, row2_col2, row2_col3 = st.columns(3)

        with row2_col1:
            st.markdown("#### Performance vs Attendance")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.scatterplot(x="TrainingAttendanceRate", y="PerformanceScore", data=dataframe, ax=ax)
            if len(dataframe) >= 2:
                sns.regplot(x="TrainingAttendanceRate", y="PerformanceScore", data=dataframe, ax=ax, scatter=False, color="red")
            ax.set_title("Performance vs Attendance")
            st.pyplot(fig, use_container_width=True)

        with row2_col2:
            st.markdown("#### Fitness vs Age")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.scatterplot(x="Age", y="FitnessScore", data=dataframe, ax=ax)
            if len(dataframe) >= 2:
                sns.regplot(x="Age", y="FitnessScore", data=dataframe, ax=ax, scatter=False, color="red")
            ax.set_title("Fitness vs Age")
            st.pyplot(fig, use_container_width=True)

        with row2_col3:
            st.markdown("#### Injury Status")
            injury_counts = dataframe["InjuryStatus"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            injury_counts.plot(kind="bar", ax=ax, color="salmon")
            ax.set_title("Injury Status")
            st.pyplot(fig, use_container_width=True)

    # -------------------------------
    # Overall Dashboard
    # -------------------------------
    plot_dashboard(df, "📊 Overall Player Dashboard")

    # -------------------------------
    # Female Dashboard
    # Your sheet uses F, not Female
    # -------------------------------
    female_df = df[df["Gender"] == "F"]
    plot_dashboard(female_df, "👩 Female Player Dashboard")

    # -------------------------------
    # Male Dashboard
    # Your sheet uses M, not Male
    # -------------------------------
    male_df = df[df["Gender"] == "M"]
    plot_dashboard(male_df, "👨 Male Player Dashboard")


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



########################################### Division Report Page ###########################################

elif st.session_state.page == "Division Report":

    st.title("📄 Division Report Downloads")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    st.markdown("""
### 📊 Club Performance Reports

Generate professional PDF dashboard reports containing:
- Player performance analytics
- Attendance and fitness insights
- Positional and injury analysis
- Division-specific breakdowns

Select a report below to export the latest club data.
""")
    from matplotlib.backends.backend_pdf import PdfPages

    def create_dashboard_pdf(dataframe, report_title):

        pdf_buffer = io.BytesIO()

        with PdfPages(pdf_buffer) as pdf:

            # -------------------------------
            # Cover Page
            # -------------------------------
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")

            ax.text(0.5, 0.72, report_title, ha="center", fontsize=24, fontweight="bold")
            ax.text(0.5, 0.60, f"Total Players: {len(dataframe)}", ha="center", fontsize=16)

            if not dataframe.empty:
                ax.text(0.5, 0.52, f"Average Performance: {round(dataframe['PerformanceScore'].mean(), 1)}", ha="center", fontsize=14)
                ax.text(0.5, 0.46, f"Average Attendance: {round(dataframe['TrainingAttendanceRate'].mean(), 1)}%", ha="center", fontsize=14)

            pdf.savefig(fig)
            plt.close(fig)

            if dataframe.empty:
                return pdf_buffer.getvalue()

            # -------------------------------
            # Plot 1 - Age Distribution
            # -------------------------------
            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.histplot(dataframe["Age"], bins=20, kde=True, ax=ax, color="skyblue")
            ax.set_title("Age Distribution")
            pdf.savefig(fig)
            plt.close(fig)

            # -------------------------------
            # Plot 2 - Position Distribution
            # -------------------------------
            fig, ax = plt.subplots(figsize=(11, 8.5))
            position_counts = dataframe["Position"].value_counts()
            position_counts.plot(kind="bar", ax=ax, color="lightblue")
            ax.set_title("Players by Position")
            ax.set_xlabel("Position")
            ax.set_ylabel("Count")
            pdf.savefig(fig)
            plt.close(fig)

            # -------------------------------
            # Plot 3 - Goals by Position
            # -------------------------------
            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.boxplot(x="Position", y="Goals", data=dataframe, ax=ax)
            ax.set_title("Goals by Position")
            pdf.savefig(fig)
            plt.close(fig)

            # -------------------------------
            # Plot 4 - Performance vs Attendance
            # -------------------------------
            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.scatterplot(x="TrainingAttendanceRate", y="PerformanceScore", data=dataframe, ax=ax)
            if len(dataframe) >= 2:
                sns.regplot(x="TrainingAttendanceRate", y="PerformanceScore", data=dataframe, ax=ax, scatter=False, color="red")
            ax.set_title("Performance vs Attendance")
            pdf.savefig(fig)
            plt.close(fig)

            # -------------------------------
            # Plot 5 - Fitness vs Age
            # -------------------------------
            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.scatterplot(x="Age", y="FitnessScore", data=dataframe, ax=ax)
            if len(dataframe) >= 2:
                sns.regplot(x="Age", y="FitnessScore", data=dataframe, ax=ax, scatter=False, color="red")
            ax.set_title("Fitness vs Age")
            pdf.savefig(fig)
            plt.close(fig)

            # -------------------------------
            # Plot 6 - Injury Status
            # -------------------------------
            fig, ax = plt.subplots(figsize=(11, 8.5))
            injury_counts = dataframe["InjuryStatus"].value_counts()
            injury_counts.plot(kind="bar", ax=ax, color="salmon")
            ax.set_title("Injury Status")
            pdf.savefig(fig)
            plt.close(fig)

        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    female_df = df[df["Gender"] == "F"]
    male_df = df[df["Gender"] == "M"]

    overall_pdf = create_dashboard_pdf(df, "Overall Team Dashboard Report")
    female_pdf = create_dashboard_pdf(female_df, "Female Division Dashboard Report")
    male_pdf = create_dashboard_pdf(male_df, "Male Division Dashboard Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="⬇️ Download Overall Team PDF",
            data=overall_pdf,
            file_name="overall_team_dashboard.pdf",
            mime="application/pdf"
        )

    with col2:
        st.download_button(
            label="⬇️ Download Female Division PDF",
            data=female_pdf,
            file_name="female_division_dashboard.pdf",
            mime="application/pdf"
        )

    with col3:
        st.download_button(
            label="⬇️ Download Male Division PDF",
            data=male_pdf,
            file_name="male_division_dashboard.pdf",
            mime="application/pdf"
        )
