# ======================================================================================================================
# INTRODUCTION
# ======================================================================================================================
# Sehr geehrter Ruben
#
# im Rahmen dieser Arbeit wurde eine Streamlit-Applikation mit dem Titel „Football Club Performance Monitor“ entwickelt.
# Die Anwendung dient dazu, Leistungsdaten von Fussballspielerinnen und Fussballspielern übersichtlich darzustellen,
# zu analysieren und für weitere Auswertungen nutzbar zu machen.
#
# Die App greift auf einen Google-Sheets-Datensatz zu, in dem verschiedene Spielerdaten gespeichert sind. Dazu gehören
# unter anderem Name, Geschlecht, Alter, Division, Position, Trainingsanwesenheit, gespielte Matches, Tore, Assists,
# Fitnessstatus, Verletzungsstatus und ein Performance Score. Die Daten werden beim Start der Anwendung geladen und
# anschliessend mit Python, Pandas, Streamlit, Matplotlib und Seaborn verarbeitet und visualisiert.
#
# Die Anwendung ist in mehrere Hauptbereiche gegliedert. Auf der Startseite erhält der Nutzer einen Überblick über den
# Zweck der App und kann über Navigationsbuttons zu den verschiedenen Bereichen wechseln. Der Dashboard-Bereich zeigt
# zentrale Kennzahlen und Visualisierungen zur Spielerleistung, darunter Altersverteilung, Positionsverteilung, Tore
# nach Position, Performance im Verhältnis zur Trainingsanwesenheit, Fitness im Verhältnis zum Alter sowie den
# Verletzungsstatus. Diese Auswertungen werden sowohl für den gesamten Datensatz als auch getrennt nach männlichen und
# weiblichen Spielern angezeigt.
#
# Ein weiterer Bereich der App ist das Machine-Learning-Modul. Dort wird ein K-Nearest-Neighbors-Regressionsmodell
# eingesetzt, um den Performance Score eines Spielers auf Basis von Alter, Trainingsanwesenheit, Fitness Score und Toren
# vorherzusagen. Die App teilt die vorhandenen Daten in Trainings- und Testdaten auf, skaliert die Eingabewerte und zeigt
# anschliessend Modellkennzahlen wie den R² Score und den durchschnittlichen Fehler an. Zusätzlich kann der Nutzer eigene
# Werte eingeben, um eine beispielhafte Performance-Prognose zu erhalten.
#
# Der Bereich „Data Management“ ermöglicht die direkte Verwaltung der Spielerdaten. Nutzer können neue Spieler hinzufügen,
# bestehende Spieler bearbeiten oder Spieler aus dem Datensatz löschen. Diese Änderungen werden direkt in Google Sheets
# gespeichert, wodurch die App als einfache Verwaltungsoberfläche für den zugrunde liegenden Datensatz funktioniert.
#
# Abschliessend enthält die App einen Reporting-Bereich. Dort können automatisch generierte PDF-Berichte heruntergeladen
# werden. Die Berichte enthalten zentrale Dashboard-Auswertungen für das gesamte Team sowie separat für die weibliche und
# männliche Division. Dadurch können die Analyseergebnisse auch ausserhalb der Anwendung weiterverwendet oder präsentiert
# werden.
#
# Insgesamt kombiniert die Applikation Datenverwaltung, Visualisierung, maschinelles Lernen und Reporting in einer
# einheitlichen Benutzeroberfläche. Ziel der App ist es, Fussballdaten verständlich aufzubereiten und daraus praktische
# Erkenntnisse für Training, Spielerentwicklung und Teammanagement abzuleiten.
#
# Frontend App Link: https://5ojsipufq2ps7vgp7qlmfr.streamlit.app
#
# ======================================================================================================================


# ======================================================================================================================
# ERKLÄRUNG ZUR NUTZUNG VON GENERATIVER KI
# ======================================================================================================================
# Während der Entwicklung dieser Applikation wurden generative KI-Tools als unterstützende Hilfsmittel verwendet.
# Der finale Code wurde vom Autor überprüft, angepasst und in die Applikation integriert.
#
# OpenAI ChatGPT wurde verwendet für:
# - Überprüfung und Unterstützung des PDF-Report-Generation-Bereichs
# - Unterstützung bei der Erstellung der Funktionsreferenz-Bibliothek
#   Hinweis: Das Überprüfen der einzelnen Bibliotheks- und Funktionsreferenzen wurde manuell
#   durch die Autor:innen durchgeführt.
# - Unterstützung beim Debugging, insbesondere bei der Erklärung von Funktionsparametern und Fehlerursachen
# - Übersetzung der Applikations-Einleitung ins Deutsche
#
# Claude von Anthropic wurde verwendet für:
# - Unterstützung beim Bereich Data Management, einschliesslich Add-, Edit- und Delete-Funktionalität
# - Unterstützung beim Machine-Learning-Bereich
#
# Die Autor:innen bleiben verantwortlich für die finale Implementierung, die Auswahl der Quellen, die Anpassung des Codes,
# die Funktionalität sowie die Erklärung der eingereichten Applikation.
#
# Referenzen:
# - OpenAI ChatGPT, chat.openai.com, Mai 2026
# - Claude von Anthropic, claude.ai, Mai 2026
# ======================================================================================================================


# ======================================================================================================================
# 1. IMPORTS
# ======================================================================================================================

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from fpdf import FPDF
import io


# ======================================================================================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ======================================================================================================================

st.set_page_config(page_title="Football Club Performance Monitor", page_icon="⚽", layout="wide")

# ======================================================================================================================
# 3. GLOBAL APPLICATION STYLING
# ======================================================================================================================

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #07140d 0%, #0e1f16 45%, #111827 100%); color: white; }
h1, h2, h3 { color: #ffffff; font-weight: 700; }
p, li, span, div { color: #d1d5db; }
[data-testid="stMarkdownContainer"] { color: #d1d5db; }
[data-testid="metric-container"] { background-color: rgba(17, 24, 39, 0.95); border: 1px solid rgba(34, 197, 94, 0.35); border-radius: 16px; padding: 18px; box-shadow: 0px 4px 18px rgba(0,0,0,0.35); }
[data-testid="metric-container"] label { color: #a7f3d0 !important; }
[data-testid="metric-container"] div { color: #ffffff !important; }
.stButton > button { background-color: #14532d; color: white; border: 1px solid #22c55e; border-radius: 10px; padding: 0.6rem 1rem; font-weight: 600; }
.stButton > button:hover { background-color: #16a34a; color: white; border: 1px solid #86efac; }
.stDownloadButton > button { background-color: #14532d; color: white !important; border: 1px solid #86efac; border-radius: 10px; padding: 0.6rem 1rem; font-weight: 600; }
.stDownloadButton > button:hover { background-color: white; color: #14532d !important; border: 1px solid #22c55e; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
hr { border-color: rgba(34, 197, 94, 0.35); }
[data-testid="stHeader"] { background: rgba(7, 20, 13, 0); }
[data-testid="stToolbar"] { color: white; }
.stSlider label, .stSelectbox label, .stNumberInput label { color: #d1d5db !important; font-weight: 600; }
div[data-baseweb="select"] > div { background-color: #111827; color: white; border-color: #22c55e; }
input { background-color: #111827 !important; color: white !important; border-color: #22c55e !important; }
.stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ======================================================================================================================
# 4. GOOGLE SHEETS AUTHENTICATION AND DATA LOADING
# ======================================================================================================================

scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

credentials = Credentials.from_service_account_info(st.secrets["google_credentials"], scopes=scopes)

client = gspread.authorize(credentials)

sheet = client.open("Dataset").sheet1

data = sheet.get_all_records()

df = pd.DataFrame(data)  # Converts the data into a Pandas DataFrame for easier manipulation and analysis.


# ======================================================================================================================
# 5. SESSION STATE INITIALISATION
# ======================================================================================================================

# Initialize session state for page tracking
# Streamlit's session state is used to handle page navigation, ensuring a smooth and intuitive user experience.
if "page" not in st.session_state:
    st.session_state.page = "Home"


# ======================================================================================================================
# 6. HOME PAGE
# ======================================================================================================================

# Home Page
if st.session_state.page == "Home":
    # The Home page introduces the football analytics application
    # and serves as the main navigation hub.

    st.title("⚽ Football Club Performance Monitor")

    st.markdown("""
    ### Turning Football Data into Performance Insights
    
    Track player development, monitor squad dynamics, and analyse club performance through a unified analytics environment built for modern football operations.
    
    📊 **Visual Analytics**  
    Interactive dashboards for performance, fitness, attendance, injuries, and positional trends.
    
    🤖 **Performance Prediction**  
    Machine learning models designed to estimate player performance based on live club data.
    
    ⚙️ **Squad Management**  
    Create, update, and maintain player records through an integrated database system.
    
    📄 **Professional Reporting**  
    Export downloadable PDF reports for the full squad and individual divisions.
    
    Built to support smarter coaching decisions and a deeper understanding of player progression across the club.
    """)

    
    # --------------------------------------------------------------------------------------------------------------
    # 6.1 HOME PAGE NAVIGATION BUTTONS
    # --------------------------------------------------------------------------------------------------------------

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


# ======================================================================================================================
# 7. DASHBOARD PAGE
# ======================================================================================================================

elif st.session_state.page == "Dashboard":

    st.title("⚽ Player Performance Dashboard")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    # --------------------------------------------------------------------------------------------------------------
    # 7.1 DASHBOARD HELPER FUNCTION
    # --------------------------------------------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------------------------------------------
    # 7.2 OVERALL PLAYER DASHBOARD
    # --------------------------------------------------------------------------------------------------------------

    plot_dashboard(df, "📊 Overall Player Dashboard")

    # --------------------------------------------------------------------------------------------------------------
    # 7.3 FEMALE PLAYER DASHBOARD
    # --------------------------------------------------------------------------------------------------------------

    female_df = df[df["Gender"] == "F"]
    plot_dashboard(female_df, "👩 Female Player Dashboard")

    # --------------------------------------------------------------------------------------------------------------
    # 7.4 MALE PLAYER DASHBOARD
    # --------------------------------------------------------------------------------------------------------------

    male_df = df[df["Gender"] == "M"]
    plot_dashboard(male_df, "👨 Male Player Dashboard")


# ======================================================================================================================
# 8. MACHINE LEARNING PAGE
# ======================================================================================================================

elif st.session_state.page == "Machine Learning":

    st.title("⚽ Player Performance Predictor")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    st.write(
        "This section uses live player data from Google Sheets to predict a player's "
        "performance score based on age, training attendance, fitness score, and goals."
    )

    # --------------------------------------------------------------------------------------------------------------
    # 8.1 DEFINE MODEL INPUTS AND TARGET
    # --------------------------------------------------------------------------------------------------------------

    # Required columns for machine learning
    FEATURES = ["Age", "TrainingAttendanceRate", "FitnessScore", "Goals"]
    TARGET = "PerformanceScore"

    required_columns = FEATURES + [TARGET]

    # --------------------------------------------------------------------------------------------------------------
    # 8.2 VALIDATE AND PREPARE DATASET
    # --------------------------------------------------------------------------------------------------------------

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing columns in Google Sheet: {missing_columns}")
        st.stop()

    ml_df = df[required_columns].copy()

    for col in required_columns:
        ml_df[col] = pd.to_numeric(ml_df[col], errors="coerce")

    ml_df = ml_df.dropna()

    if len(ml_df) < 10:
        st.warning("Not enough valid data to train the model. Please add more player records.")
        st.stop()

    X = ml_df[FEATURES]
    y = ml_df[TARGET]

    # --------------------------------------------------------------------------------------------------------------
    # 8.3 TRAIN MACHINE LEARNING MODEL
    # --------------------------------------------------------------------------------------------------------------

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    import numpy as np

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # --------------------------------------------------------------------------------------------------------------
    # 8.4 DISPLAY MODEL PERFORMANCE
    # --------------------------------------------------------------------------------------------------------------

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Rows", len(X_train))
    col2.metric("R² Score", f"{r2:.2f}")
    col3.metric("Error", f"{mae:.1f}")

    st.write("---")

    # --------------------------------------------------------------------------------------------------------------
    # 8.5 USER INPUT PREDICTION TOOL
    # --------------------------------------------------------------------------------------------------------------

    st.subheader("Try Prediction")

    input_age = st.slider("Age", 15, 50, 25)
    input_attendance = st.slider("Training Attendance Rate (%)", 0.0, 100.0, 80.0)
    input_fitness = st.slider("Fitness Score", 0.0, 100.0, 75.0)
    input_goals = st.slider("Goals", 0, 50, 5)

    input_array = np.array([[input_age, input_attendance, input_fitness, input_goals]])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Performance Score: {prediction:.1f}")
   

# ======================================================================================================================
# 9. DATA MANAGEMENT PAGE
# ======================================================================================================================

# Backend Page (Data Management)
elif st.session_state.page == "Data Management":
    st.title("⚽ Player Data Manager")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    worksheet = client.open("Dataset").sheet1

    if "dm_subpage" not in st.session_state:
        st.session_state.dm_subpage = "main"

    # --------------------------------------------------------------------------------------------------------------
    # 9.1 DATA MANAGEMENT MAIN MENU
    # --------------------------------------------------------------------------------------------------------------

    if st.session_state.dm_subpage == "main":
        st.subheader("What would you like to do?")

        if st.button("➕ Add New Player"):
            st.session_state.dm_subpage = "add"

        if st.button("✏️ Edit Player"):
            st.session_state.dm_subpage = "edit"

        if st.button("🗑️ Delete Player"):
            st.session_state.dm_subpage = "delete"

    # --------------------------------------------------------------------------------------------------------------
    # 9.2 ADD NEW PLAYER
    # --------------------------------------------------------------------------------------------------------------

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


    # --------------------------------------------------------------------------------------------------------------
    # 9.3 EDIT EXISTING PLAYER
    # --------------------------------------------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------------------------------------------
    # 9.4 DELETE PLAYER
    # --------------------------------------------------------------------------------------------------------------

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



# ======================================================================================================================
# 10. DIVISION REPORT PAGE
# ======================================================================================================================

elif st.session_state.page == "Division Report":

    st.title("📄 Division Report Downloads")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    # --------------------------------------------------------------------------------------------------------------
    # 10.1 REPORT PAGE INTRODUCTION
    # --------------------------------------------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------------------------------------------
    # 10.2 PDF DASHBOARD GENERATION FUNCTION
    # --------------------------------------------------------------------------------------------------------------

    def create_dashboard_pdf(dataframe, report_title):

        pdf_buffer = io.BytesIO()

        with PdfPages(pdf_buffer) as pdf:

            # ------------------------------------------------------------------------------------------------------
            # 10.2.1 COVER PAGE
            # ------------------------------------------------------------------------------------------------------

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

            # ------------------------------------------------------------------------------------------------------
            # 10.2.2 AGE DISTRIBUTION CHART
            # ------------------------------------------------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.histplot(dataframe["Age"], bins=20, kde=True, ax=ax, color="skyblue")
            ax.set_title("Age Distribution")
            pdf.savefig(fig)
            plt.close(fig)

            # ------------------------------------------------------------------------------------------------------
            # 10.2.3 POSITION DISTRIBUTION CHART
            # ------------------------------------------------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(11, 8.5))
            position_counts = dataframe["Position"].value_counts()
            position_counts.plot(kind="bar", ax=ax, color="lightblue")
            ax.set_title("Players by Position")
            ax.set_xlabel("Position")
            ax.set_ylabel("Count")
            pdf.savefig(fig)
            plt.close(fig)

            # ------------------------------------------------------------------------------------------------------
            # 10.2.4 GOALS BY POSITION CHART
            # ------------------------------------------------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.boxplot(x="Position", y="Goals", data=dataframe, ax=ax)
            ax.set_title("Goals by Position")
            pdf.savefig(fig)
            plt.close(fig)

            # ------------------------------------------------------------------------------------------------------
            # 10.2.5 PERFORMANCE VS ATTENDANCE CHART
            # ------------------------------------------------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.scatterplot(x="TrainingAttendanceRate", y="PerformanceScore", data=dataframe, ax=ax)
            if len(dataframe) >= 2:
                sns.regplot(x="TrainingAttendanceRate", y="PerformanceScore", data=dataframe, ax=ax, scatter=False, color="red")
            ax.set_title("Performance vs Attendance")
            pdf.savefig(fig)
            plt.close(fig)

            # ------------------------------------------------------------------------------------------------------
            # 10.2.6 FITNESS VS AGE CHART
            # ------------------------------------------------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.scatterplot(x="Age", y="FitnessScore", data=dataframe, ax=ax)
            if len(dataframe) >= 2:
                sns.regplot(x="Age", y="FitnessScore", data=dataframe, ax=ax, scatter=False, color="red")
            ax.set_title("Fitness vs Age")
            pdf.savefig(fig)
            plt.close(fig)

            # ------------------------------------------------------------------------------------------------------
            # 10.2.7 INJURY STATUS CHART
            # ------------------------------------------------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(11, 8.5))
            injury_counts = dataframe["InjuryStatus"].value_counts()
            injury_counts.plot(kind="bar", ax=ax, color="salmon")
            ax.set_title("Injury Status")
            pdf.savefig(fig)
            plt.close(fig)

        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    # --------------------------------------------------------------------------------------------------------------
    # 10.3 CREATE DIVISION DATASETS
    # --------------------------------------------------------------------------------------------------------------

    female_df = df[df["Gender"] == "F"]
    male_df = df[df["Gender"] == "M"]

    # --------------------------------------------------------------------------------------------------------------
    # 10.4 GENERATE PDF REPORTS
    # --------------------------------------------------------------------------------------------------------------

    overall_pdf = create_dashboard_pdf(df, "Overall Team Dashboard Report")
    female_pdf = create_dashboard_pdf(female_df, "Female Division Dashboard Report")
    male_pdf = create_dashboard_pdf(male_df, "Male Division Dashboard Report")

    # --------------------------------------------------------------------------------------------------------------
    # 10.5 DOWNLOAD BUTTONS
    # --------------------------------------------------------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(label="⬇️ Download Overall Team PDF", data=overall_pdf, file_name="overall_team_dashboard.pdf", mime="application/pdf")

    with col2:
        st.download_button(label="⬇️ Download Female Division PDF", data=female_pdf, file_name="female_division_dashboard.pdf", mime="application/pdf")

    with col3:
        st.download_button(label="⬇️ Download Male Division PDF", data=male_pdf, file_name="male_division_dashboard.pdf", mime="application/pdf")


# ======================================================================================================================
# 11. FUNCTION REFERENCE LIBRARY
# ======================================================================================================================
# This section lists the main functions, methods and callable objects used in the application.
# The functions are grouped by library/object so that documentation links can be collected more easily.
# ======================================================================================================================


# ======================================================================================================================
# 11.1 STREAMLIT FUNCTIONS
# ======================================================================================================================
# Streamlit is used to build the application interface, handle navigation, display text, show charts,
# collect user inputs and provide download buttons.
#
# st.set_page_config()
# Purpose: Configures the Streamlit page title, page icon and layout.
# Reference: https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config
#
# st.markdown()
# Purpose: Displays formatted Markdown text and is also used to inject custom CSS styling.
# Reference: https://docs.streamlit.io/develop/api-reference/text/st.markdown
#
# st.title()
# Purpose: Displays main page titles.
# Reference: https://docs.streamlit.io/develop/api-reference/text/st.title
#
# st.subheader()
# Purpose: Displays section-level headings.
# Reference: https://docs.streamlit.io/develop/api-reference/text/st.subheader
#
# st.write()
# Purpose: Displays text, separators and general output on the page.
# Reference: https://docs.streamlit.io/develop/api-reference/write-magic/st.write
#
# st.button()
# Purpose: Creates clickable buttons for navigation and actions.
# Reference: https://docs.streamlit.io/develop/api-reference/widgets/st.button
#
# st.columns()
# Purpose: Splits the app layout into multiple columns.
# Reference: https://docs.streamlit.io/develop/api-reference/layout/st.columns
#
# st.metric()
# Purpose: Displays key performance indicators such as total players and average performance.
# Reference: https://docs.streamlit.io/develop/api-reference/data/st.metric
#
# st.pyplot()
# Purpose: Displays Matplotlib figures inside the Streamlit app.
# Reference: https://docs.streamlit.io/develop/api-reference/charts/st.pyplot
#
# st.warning()
# Purpose: Displays warning messages when data is missing or insufficient.
# Reference: https://docs.streamlit.io/develop/api-reference/status/st.warning
#
# st.error()
# Purpose: Displays error messages when required columns are missing.
# Reference: https://docs.streamlit.io/develop/api-reference/status/st.error
#
# st.success()
# Purpose: Displays success messages after actions such as adding, editing or deleting players.
# Reference: https://docs.streamlit.io/develop/api-reference/status/st.success
#
# st.stop()
# Purpose: Stops the app execution when validation fails.
# Reference: https://docs.streamlit.io/develop/api-reference/execution-flow/st.stop
#
# st.slider()
# Purpose: Creates slider inputs for numerical user inputs.
# Reference: https://docs.streamlit.io/develop/api-reference/widgets/st.slider
#
# st.text_input()
# Purpose: Creates text input fields, for example for player names.
# Reference: https://docs.streamlit.io/develop/api-reference/widgets/st.text_input
#
# st.number_input()
# Purpose: Creates numerical input fields for values such as age, goals, assists and matches played.
# Reference: https://docs.streamlit.io/develop/api-reference/widgets/st.number_input
#
# st.selectbox()
# Purpose: Creates dropdown menus for categories such as gender, division, position and injury status.
# Reference: https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox
#
# st.form()
# Purpose: Groups input widgets into a form that is submitted together.
# Reference: https://docs.streamlit.io/develop/api-reference/execution-flow/st.form
#
# st.form_submit_button()
# Purpose: Creates a submit button inside a Streamlit form.
# Reference: https://docs.streamlit.io/develop/api-reference/execution-flow/st.form_submit_button
#
# st.download_button()
# Purpose: Creates buttons for downloading generated PDF reports.
# Reference: https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
#
# st.session_state
# Purpose: Stores page navigation state and selected subpages during the Streamlit session.
# Reference: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state


# ======================================================================================================================
# 11.2 GOOGLE SHEETS / GSPREAD FUNCTIONS
# ======================================================================================================================
# gspread is used to connect the app to Google Sheets, read the player dataset and write changes back to the sheet.
#
# gspread.authorize()
# Purpose: Authorizes the Google Sheets client using service-account credentials.
# Reference: https://docs.gspread.org/en/latest/oauth2.html
#
# client.open()
# Purpose: Opens the Google Sheet file named "Dataset".
# Reference: https://docs.gspread.org/en/latest/user-guide.html
#
# sheet1
# Purpose: Selects the first worksheet inside the opened Google Sheet.
# Reference: https://docs.gspread.org/en/latest/user-guide.html
#
# worksheet.get_all_records()
# Purpose: Reads the Google Sheet data as a list of records using the first row as headers.
# Reference: https://docs.gspread.org/en/latest/user-guide.html#getting-all-values-from-a-row-or-a-column
#
# worksheet.get_all_values()
# Purpose: Reads all values from the worksheet, including headers.
# Reference: https://docs.gspread.org/en/latest/user-guide.html#getting-all-values-from-a-row-or-a-column
#
# worksheet.append_row()
# Purpose: Adds a new player record as a new row in Google Sheets.
# Reference: https://docs.gspread.org/en/latest/api/models/worksheet.html#gspread.worksheet.Worksheet.append_row
#
# worksheet.update()
# Purpose: Updates an existing player row in Google Sheets.
# Reference: https://docs.gspread.org/en/latest/api/models/worksheet.html#gspread.worksheet.Worksheet.update
#
# worksheet.delete_rows()
# Purpose: Deletes a selected player row from Google Sheets.
# Reference: https://docs.gspread.org/en/latest/api/models/worksheet.html#gspread.worksheet.Worksheet.delete_rows


# ======================================================================================================================
# 11.3 GOOGLE AUTHENTICATION FUNCTIONS
# ======================================================================================================================
# Google authentication is used to securely connect Streamlit to Google Sheets through service-account credentials.
#
# Credentials.from_service_account_info()
# Purpose: Creates Google service-account credentials from the credential information stored in Streamlit secrets.
# Reference: https://google-auth.readthedocs.io/en/latest/reference/google.oauth2.service_account.html


# ======================================================================================================================
# 11.4 PANDAS FUNCTIONS
# ======================================================================================================================
# Pandas is used to convert, clean, filter and analyse the player dataset.
#
# pd.DataFrame()
# Purpose: Converts Google Sheets data into a Pandas DataFrame.
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
#
# dataframe.empty
# Purpose: Checks whether a DataFrame contains no rows.
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.empty.html
#
# dataframe.copy()
# Purpose: Creates a copy of the selected machine-learning dataset.
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html
#
# pd.to_numeric()
# Purpose: Converts selected columns to numeric values and replaces invalid values with missing values.
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
#
# dataframe.dropna()
# Purpose: Removes rows with missing values before training the machine-learning model.
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
#
# series.mean()
# Purpose: Calculates averages such as average performance and average attendance.
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.Series.mean.html
#
# series.value_counts()
# Purpose: Counts category frequencies, for example players by position or injury status.
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
#
# series.plot()
# Purpose: Creates simple charts from Pandas data, such as bar charts.
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.html


# ======================================================================================================================
# 11.5 MATPLOTLIB FUNCTIONS
# ======================================================================================================================
# Matplotlib is used to create figures, axes and PDF report charts.
#
# plt.subplots()
# Purpose: Creates a figure and axes object for each chart.
# Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
#
# ax.set_title()
# Purpose: Sets the title of a chart.
# Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html
#
# ax.set_xlabel()
# Purpose: Sets the label of the x-axis.
# Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html
#
# ax.set_ylabel()
# Purpose: Sets the label of the y-axis.
# Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html
#
# ax.axis()
# Purpose: Shows or hides chart axes, used in the PDF cover page.
# Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axis.html
#
# ax.text()
# Purpose: Adds text to a Matplotlib figure, used for the PDF cover page.
# Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
#
# plt.close()
# Purpose: Closes figures after saving them to avoid unnecessary memory usage.
# Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.close.html


# ======================================================================================================================
# 11.6 SEABORN FUNCTIONS
# ======================================================================================================================
# Seaborn is used to create statistical visualisations for the dashboard and PDF reports.
#
# sns.histplot()
# Purpose: Creates histograms, used for the age distribution chart.
# Reference: https://seaborn.pydata.org/generated/seaborn.histplot.html
#
# sns.boxplot()
# Purpose: Creates boxplots, used for goals by position.
# Reference: https://seaborn.pydata.org/generated/seaborn.boxplot.html
#
# sns.scatterplot()
# Purpose: Creates scatterplots, used for performance vs attendance and fitness vs age.
# Reference: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
#
# sns.regplot()
# Purpose: Adds regression trend lines to scatterplots.
# Reference: https://seaborn.pydata.org/generated/seaborn.regplot.html


# ======================================================================================================================
# 11.7 SCIKIT-LEARN FUNCTIONS
# ======================================================================================================================
# scikit-learn is used to prepare the data, train the machine-learning model and evaluate its performance.
#
# train_test_split()
# Purpose: Splits the dataset into training and testing data.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#
# StandardScaler()
# Purpose: Creates a scaler object to standardise input features before model training.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#
# scaler.fit_transform()
# Purpose: Fits the scaler to the training data and transforms the training data.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#
# scaler.transform()
# Purpose: Applies the fitted scaler to test data and user input data.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#
# KNeighborsRegressor()
# Purpose: Creates the K-Nearest-Neighbors regression model.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
#
# model.fit()
# Purpose: Trains the machine-learning model using the scaled training data.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
#
# model.predict()
# Purpose: Predicts performance scores using the trained model.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
#
# r2_score()
# Purpose: Calculates the R² score of the model.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
#
# mean_absolute_error()
# Purpose: Calculates the average prediction error of the model.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html


# ======================================================================================================================
# 11.8 NUMPY FUNCTIONS
# ======================================================================================================================
# NumPy is used to prepare the user input values for the machine-learning model.
#
# np.array()
# Purpose: Converts the user's prediction inputs into an array format that can be used by the model.
# Reference: https://numpy.org/doc/stable/reference/generated/numpy.array.html


# ======================================================================================================================
# 11.9 MATPLOTLIB PDF FUNCTIONS
# ======================================================================================================================
# PdfPages is used to generate multi-page PDF dashboard reports.
#
# PdfPages()
# Purpose: Creates a multi-page PDF object that can store several Matplotlib figures.
# Reference: https://matplotlib.org/stable/api/backend_pdf_api.html
#
# pdf.savefig()
# Purpose: Saves a Matplotlib figure as a page inside the PDF report.
# Reference: https://matplotlib.org/stable/api/backend_pdf_api.html


# ======================================================================================================================
# 11.10 PYTHON BUILT-IN FUNCTIONS AND METHODS
# ======================================================================================================================
# Python built-ins and standard methods are used for basic data processing, conversion and formatting.
#
# len()
# Purpose: Counts the number of rows, players or data points.
# Reference: https://docs.python.org/3/library/functions.html#len
#
# round()
# Purpose: Rounds numeric values before they are displayed.
# Reference: https://docs.python.org/3/library/functions.html#round
#
# int()
# Purpose: Converts values into integers.
# Reference: https://docs.python.org/3/library/functions.html#int
#
# list.index()
# Purpose: Finds the position of a selected player in a list.
# Reference: https://docs.python.org/3/tutorial/datastructures.html
#
# str.replace()
# Purpose: Removes the "P" from a player ID before generating the next player number.
# Reference: https://docs.python.org/3/library/stdtypes.html#str.replace
#
# f-strings
# Purpose: Formats dynamic text such as player IDs, success messages and model output.
# Reference: https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals


# ======================================================================================================================
# 11.11 PYTHON IO FUNCTIONS
# ======================================================================================================================
# The io library is used to create the PDF reports in memory before they are downloaded.
#
# io.BytesIO()
# Purpose: Creates an in-memory binary buffer for the generated PDF.
# Reference: https://docs.python.org/3/library/io.html#io.BytesIO
#
# pdf_buffer.getvalue()
# Purpose: Returns the PDF content from the in-memory buffer.
# Reference: https://docs.python.org/3/library/io.html#io.BytesIO.getvalue
#
# pdf_buffer.seek()
# Purpose: Resets the buffer position to the beginning before returning the PDF.
# Reference: https://docs.python.org/3/library/io.html#io.IOBase.seek


# ======================================================================================================================
# 11.12 CUSTOM FUNCTIONS DEFINED IN THIS APPLICATION
# ======================================================================================================================
# These functions are written directly inside this application and therefore do not have external documentation pages.
#
# plot_dashboard(dataframe, title)
# Purpose: Generates the dashboard section for a given dataset, including KPIs and charts.
# Used in: Overall Player Dashboard, Female Player Dashboard, Male Player Dashboard.
#
# create_dashboard_pdf(dataframe, report_title)
# Purpose: Generates a downloadable PDF dashboard report for a given dataset.
# Used in: Overall Team Dashboard Report, Female Division Dashboard Report, Male Division Dashboard Report.
# ======================================================================================================================
    
