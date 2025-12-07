import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ------------------------------------------------------
# DARK THEME CSS
# ------------------------------------------------------
st.markdown("""
<style>
    .block-container { background-color: #111 !important; color: #f1f1f1 !important; }
    h1, h2, h3, h4, h5, h6, p, label, span { color: #f2f2f2 !important; }
    div[data-testid="stSidebar"] { background-color: #1a1a1a !important; color: #f1f1f1 !important; }
    table { background-color:#181818 !important; color:#e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

FEATURES = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "TotalWorkingYears"
]

# ------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------
@st.cache_resource
def load_model(path="final_attrition_model.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ------------------------------------------------------
# HELPER
# ------------------------------------------------------
def build_input(values):
    return np.array([[values[f] for f in FEATURES]])

# ------------------------------------------------------
# LANDING PAGE
# ------------------------------------------------------
def landing_page():

    st.markdown("<h1 style='text-align:center;'>🏢 Employee Attrition Prediction System</h1>", unsafe_allow_html=True)
    st.write("")

    # Animated tagline
    placeholder = st.empty()
    tagline = "Predict • Analyze • Understand • Retain"
    anim = ""
    for ch in tagline:
        anim += ch
        placeholder.markdown(f"<h3 style='text-align:center; color:#bbbbbb;'>{anim}▌</h3>", unsafe_allow_html=True)
        time.sleep(0.02)
    placeholder.markdown(f"<h3 style='text-align:center; color:#bbbbbb;'>{tagline}</h3>", unsafe_allow_html=True)

    st.write("")

    # About section
    st.subheader("🌟 Why this tool matters")
    st.info("""
Employee turnover affects productivity, culture, and financial health.
This system empowers HR teams to:
- Spot early resignation risks
- Understand underlying causes
- Improve retention strategies
- Strengthen workforce planning
""")

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Let's Get Started")
        st.caption("Tell us who you are to personalize your journey.")
        name = st.text_input("Enter Your Name")

        if st.button("🚀 Begin"):
            st.session_state["entered"] = True
            st.session_state["username"] = name if name else "Guest"
            st.rerun()

    with col2:
        st.subheader("🔍 What powers this system?")
        st.success("""
✔ Machine Learning Model  
✔ HR Dashboard  
✔ Recommendation Engine  
✔ Bulk Prediction  
""")

    st.write("---")

    st.markdown("## ⭐ Platform Highlights")

    c1, c2, c3 = st.columns(3)
    c1.metric("🔮 Prediction", "Instant Insights")
    c2.metric("📊 Batch Analysis", "Bulk Processing")
    c3.metric("🎯 Smart Recommendations", "Action Plans")

    st.caption("✨ Designed by Shyam Kumar ")


# ------------------------------------------------------
# MAIN PREDICTION PAGE
# ------------------------------------------------------
def main_app():
    if model is None:
        st.error("Model file missing: final_attrition_model.pkl")
        st.stop()

    # -----------------------------
    # HEADER
    # -----------------------------
    st.markdown("<h1 style='text-align:center;'>🔮 Single Employee Attrition Predictor</h1>", unsafe_allow_html=True)
    st.write("")
    st.caption("Use this tool to assess the attrition risk of one employee based on key HR factors.")

    st.write("---")

    # -----------------------------
    # LAYOUT
    # -----------------------------
    left, right = st.columns([1, 1])

    # =============================
    # LEFT CARD — INPUTS
    # =============================
    with left:
        st.markdown("""
        <div style="
            background-color:#1a1a1a;
            padding:18px;
            border-radius:12px;
            border:1px solid #333;
        ">
        <h3>🧾 Employee Details</h3>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        
        Age = st.number_input("👤 Age", 18, 65, 30)
        Income = st.number_input("💰 Monthly Income", 0, 200000, 5000)
        Dist = st.number_input("📍 Distance From Home (km)", 0, 100, 10)
        JobSat = st.slider("⭐ Job Satisfaction", 1, 4, 3)
        EnvSat = st.slider("🏢 Environment Satisfaction", 1, 4, 3)
        WorkYrs = st.number_input("📆 Total Working Years", 0, 50, 5)

        input_values = {
            "Age": Age,
            "MonthlyIncome": Income,
            "DistanceFromHome": Dist,
            "JobSatisfaction": JobSat,
            "EnvironmentSatisfaction": EnvSat,
            "TotalWorkingYears": WorkYrs
        }

        run_button = st.button("🚀 Run Prediction", use_container_width=True)

    # =============================
    # RIGHT CARD — OUTPUT
    # =============================
    with right:

        st.markdown("""
        <div style="
            background-color:#1a1a1a;
            padding:18px;
            border-radius:12px;
            border:1px solid #333;
        ">
        <h3>📊 Prediction Result</h3>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if not run_button:
            st.info("Fill in the details and click **Run Prediction** to proceed.")
            return

        # -----------------------------
        # MODEL PREDICTION
        # -----------------------------
        X = build_input(input_values)
        proba = float(model.predict_proba(X)[0][1])
        label = int(model.predict(X)[0])

        # Risk Badge
        if proba >= 0.66:
            badge = "🔴 **HIGH RISK**"
            color = "#FF4C4C"
        elif proba >= 0.33:
            badge = "🟠 **MEDIUM RISK**"
            color = "#FFA500"
        else:
            badge = "🟢 **LOW RISK**"
            color = "#4CAF50"

        st.markdown(
            f"<h2 style='color:{color}; text-align:center;'>{badge}</h2>",
            unsafe_allow_html=True
        )

        st.metric("Attrition Probability", f"{proba:.2%}")
        st.progress(proba)

        st.write("---")

        # -----------------------------
        # HR INSIGHT
        # -----------------------------
        st.subheader("🧠 HR Recommendation")

        if proba >= 0.66:
            st.markdown("""
            **This employee shows a high likelihood of attrition.**
            Consider:
            - Role satisfaction/feedback discussion  
            - Salary benchmarking  
            - Hybrid work if commute is long  
            - Internal mobility or upskilling  
            """)
        elif proba >= 0.33:
            st.markdown("""
            **Moderate attrition risk.**
            Recommended:
            - Monitor engagement over next cycle  
            - Provide recognition or small incentives  
            - Address work-life balance factors  
            """)
        else:
            st.markdown("""
            **Low attrition risk.**
            Maintain:
            - Good culture & clear communication  
            - Learning and development paths  
            - Positive reinforcement  
            """)

        st.write("---")

        # -----------------------------
        # INPUT SUMMARY
        # -----------------------------
        st.subheader("📌 Summary of Inputs")
        st.table(pd.DataFrame([input_values]).T.rename(columns={0: "Value"}))

# ------------------------------------------------------
# BATCH PREDICTION PAGE — FIXED
# ------------------------------------------------------
def batch_page():
    st.title("📂 Batch Attrition Prediction")
    st.write("Upload once, reuse across all pages — or override just for this batch.")

    df = None

    # If a global dataset is already loaded (from sidebar), offer to use it
    if "global_df" in st.session_state:
        st.success("Using dataset from sidebar upload.")
        use_global = st.checkbox("Use global dataset from sidebar", value=True)

        if use_global:
            df = st.session_state["global_df"].copy()

    # If not using global, allow local upload here
    if df is None:
        file = st.file_uploader("Upload Employee CSV", type=["csv"])
        if file is None:
            st.info("Either upload a CSV here, or upload once from the sidebar.")
            return

        try:
            df = pd.read_csv(file)
            # Store this as global as well so other pages can reuse it
            st.session_state["global_df"] = df
            st.success("File loaded and stored for use across pages.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

    # Check required columns
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"❌ Your file is missing required columns: {', '.join(missing)}")
        st.info("Please ensure these exact columns exist in your CSV:")
        st.write(FEATURES)
        return

    # Clean numeric features
    clean_df = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    clean_df = clean_df.fillna(clean_df.median())

    # Make predictions
    probs = model.predict_proba(clean_df)[:, 1]
    labels = (probs >= 0.5).astype(int)

    df["AttritionProbability"] = probs
    df["Prediction"] = labels
    df["RiskLabel"] = pd.cut(
        probs,
        bins=[0, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"]
    )

    st.write("---")

    st.subheader("📊 Batch Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Employees Processed", len(df))
    with c2:
        st.metric("High-Risk Count", int((df["RiskLabel"] == "High").sum()))
    with c3:
        st.metric("Avg Attrition Probability", f"{df['AttritionProbability'].mean():.2%}")

    st.write("---")

    st.subheader("👁️ Preview of Predictions (First 20 Records)")
    st.dataframe(df.head(20))

    st.write("---")

    st.download_button(
        label="⬇️ Download Full Prediction CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="batch_predictions.csv",
        mime="text/csv"
    )

    st.caption("Batch prediction completed successfully!")

# ------------------------------------------------------
# RECOMMENDATION ENGINE — FIXED
# ------------------------------------------------------
def recommendation_engine_page():
    st.title("🎯 Recommendation Engine")
    st.write("Generate actionable HR insights based on employee risk factors.")
    st.write("---")

    # -----------------------------
    # STEP 1: Ensure dataset exists
    # -----------------------------
    if "global_df" not in st.session_state:
        st.warning("⚠️ No dataset loaded. Upload a file using the sidebar.")
        return

    df = st.session_state["global_df"].copy()

    # -----------------------------
    # STEP 2: Check features
    # -----------------------------
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        st.error("❌ Missing required columns: " + ", ".join(missing))
        return

    st.subheader("📌 Dataset Summary")
    st.caption(f"Employees loaded: {len(df)}")
    st.write("---")

    # -----------------------------
    # STEP 3: Clean numeric features + prediction
    # -----------------------------
    clean_df = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    clean_df = clean_df.fillna(clean_df.median())

    probs = model.predict_proba(clean_df)[:, 1]
    labels = (probs >= 0.5).astype(int)

    df["AttritionProbability"] = probs
    df["RiskLabel"] = pd.cut(
        probs,
        bins=[0, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"]
    )

    df["Prediction"] = labels

    # -----------------------------
    # STEP 4: Generate recommendations
    # -----------------------------
    recommendations = []

    for _, row in df.iterrows():
        r = []

        if row["JobSatisfaction"] <= 2:
            r.append("Enhance job satisfaction through role redesign or feedback sessions.")
        if row["EnvironmentSatisfaction"] <= 2:
            r.append("Address work environment concerns; schedule 1-on-1 conversations.")
        if row["DistanceFromHome"] >= 20:
            r.append("Consider hybrid or remote work options to reduce commute stress.")
        if row["MonthlyIncome"] < df["MonthlyIncome"].median():
            r.append("Review salary or provide performance-based incentives.")
        if row["TotalWorkingYears"] <= 3:
            r.append("Provide training, onboarding support, or mentorship.")
        if row["RiskLabel"] == "High":
            r.append("⚠️ High-risk employee — Immediate retention plan advised.")

        if not r:
            r.append("Employee appears stable — continue engagement and growth initiatives.")

        recommendations.append("; ".join(r))

    df["HR_Recommendations"] = recommendations

    # -----------------------------
    # STEP 5: Filters
    # -----------------------------
    st.subheader("🔎 Filter Employees")
    filter_choice = st.radio(
        "Select risk level:",
        ["All", "High", "Medium", "Low"],
        horizontal=True
    )

    filtered_df = df.copy()
    if filter_choice != "All":
        filtered_df = filtered_df[filtered_df["RiskLabel"] == filter_choice]

    st.write("---")

    # -----------------------------
    # STEP 6: Display Recommendations Table with styling
    # -----------------------------
    st.subheader(f"🧠 HR Recommendations — {filter_choice} Risk Employees")

    styled_df = filtered_df.style.apply(
        lambda row: [
            "background-color: #4CAF50" if row["RiskLabel"] == "Low"
            else "background-color: #FFC107" if row["RiskLabel"] == "Medium"
            else "background-color: #F44336"
            for _ in row
        ],
        axis=1
    )

    st.dataframe(
        styled_df,
        height=450
    )

    st.write("---")

    # -----------------------------
    # STEP 7: Download button
    # -----------------------------
    st.download_button(
        "⬇️ Download Recommendations CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="recommendations.csv",
        mime="text/csv"
    )

    st.caption("Recommendation Engine | Powered by Machine Learning + Streamlit")

# ------------------------------------------------------
# DASHBOARD — FIXED
# ------------------------------------------------------
def dashboard_page():
    st.title("📊 HR Attrition Dashboard")
    st.write("A smart overview of your workforce risk distribution.")
    st.write("")

    # -----------------------------
    # STEP 1: Load Dataset
    # -----------------------------
    if "global_df" not in st.session_state:
        uploaded = st.file_uploader("Upload Employee CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a dataset to generate the HR dashboard.")
            return
        df = pd.read_csv(uploaded)
        st.session_state["global_df"] = df
    else:
        df = st.session_state["global_df"]

    # Copy to avoid modifying original
    temp_df = df.copy()

    # -----------------------------
    # STEP 2: Filters
    # -----------------------------
    st.subheader("🎛️ Filters")
    st.write("Refine the analysis based on employee characteristics.")

    # Department Filter
    if "Department" in temp_df.columns:
        dept_list = ["All"] + sorted(temp_df["Department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department", dept_list)
        if selected_dept != "All":
            temp_df = temp_df[temp_df["Department"] == selected_dept]

    # Age Filter
    if "Age" in temp_df.columns:
        age_min, age_max = int(temp_df["Age"].min()), int(temp_df["Age"].max())
        age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))
        temp_df = temp_df[temp_df["Age"].between(age_range[0], age_range[1])]

    # Distance Filter
    if "DistanceFromHome" in temp_df.columns:
        dmin, dmax = int(temp_df["DistanceFromHome"].min()), int(temp_df["DistanceFromHome"].max())
        dist_range = st.slider("Distance From Home (KM)", dmin, dmax, (dmin, dmax))
        temp_df = temp_df[temp_df["DistanceFromHome"].between(dist_range[0], dist_range[1])]

    # Job Satisfaction Filter
    if "JobSatisfaction" in temp_df.columns:
        js_min, js_max = 1, 4
        js_range = st.slider("Job Satisfaction (1–4)", js_min, js_max, (js_min, js_max))
        temp_df = temp_df[temp_df["JobSatisfaction"].between(js_range[0], js_range[1])]

    st.write("---")

    # -----------------------------
    # STEP 3: Clean numeric features + predict
    # -----------------------------
    clean_df = temp_df[FEATURES].apply(pd.to_numeric, errors="coerce")
    clean_df = clean_df.fillna(clean_df.median())

    temp_df["AttritionProbability"] = model.predict_proba(clean_df)[:, 1]

    temp_df["RiskLabel"] = pd.cut(
        temp_df["AttritionProbability"],
        bins=[0, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"]
    )

    # -----------------------------
    # STEP 4: KPIs
    # -----------------------------
    st.subheader("📌 Workforce Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("👥 Total Employees", len(temp_df))

    with c2:
        st.metric("🔥 High Risk Count", int((temp_df["RiskLabel"] == "High").sum()))

    with c3:
        avg_prob = temp_df["AttritionProbability"].mean()
        st.metric("📉 Avg Attrition Probability", f"{avg_prob:.1%}")

    st.write("---")
    st.write("---")

    # -----------------------------
    # STEP 4.5: Department Insight Cards (Safe Version)
    # -----------------------------
    if "Department" in temp_df.columns:

        st.subheader("🏢 Department Risk Overview")

        dept_summary = (
            temp_df.groupby("Department")["RiskLabel"]
            .apply(lambda x: (x == "High").mean())  # % high risk
            .sort_values(ascending=False)
        )

        cols = st.columns(3)
        idx = 0

        for dept, risk_pct in dept_summary.items():
            col = cols[idx % 3]
            with col:
                st.container().markdown(
                    f"""
                    **📌 {dept}**  
                    High-Risk Employees: **{risk_pct:.1%}**
                    """,
                    unsafe_allow_html=True
                )
            idx += 1

    st.write("---")

    
    # -----------------------------
    # STEP 5: Charts (Improved Colors & Layout)
    # -----------------------------
    st.subheader("📈 Attrition Risk Visual Insights")

    chart_left, chart_right = st.columns(2)

    # PIE CHART (Better Colors)
    with chart_left:
        risk_counts = temp_df["RiskLabel"].value_counts()

        fig, ax = plt.subplots(figsize=(4, 4))
        colors = ["#4CAF50", "#FFC107", "#F44336"]  # green, amber, red

        ax.pie(
            risk_counts,
            labels=risk_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors
        )
        ax.axis("equal")
        ax.set_title("Risk Distribution", color="#f2f2f2")
        st.pyplot(fig)

    # BAR CHART (Consistent Colors)
    with chart_right:
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        risk_counts.plot(kind="bar", ax=ax2, color=colors)
        ax2.set_ylabel("Employees")
        ax2.set_title("Risk Level Count", color="#f2f2f2")
        ax2.tick_params(colors="#f2f2f2")
        st.pyplot(fig2)

    st.write("---")

    # -----------------------------
    # STEP 6: Satisfaction Heatmap
    # -----------------------------
        # -----------------------------
    # STEP 6: Satisfaction Heatmap (No Seaborn Version)
    # -----------------------------
    st.subheader("🔥 Satisfaction Heatmap (Job vs Environment)")

    if "JobSatisfaction" in temp_df.columns and "EnvironmentSatisfaction" in temp_df.columns:

        heat_df = temp_df.copy()
        heat_df["JS"] = heat_df["JobSatisfaction"]
        heat_df["ES"] = heat_df["EnvironmentSatisfaction"]

        pivot = pd.pivot_table(
            heat_df,
            values="AttritionProbability",
            index="JS",
            columns="ES",
            aggfunc="mean"
        )

        fig3, ax3 = plt.subplots(figsize=(6, 4))

        cax = ax3.imshow(pivot, cmap="inferno", aspect="auto")
        fig3.colorbar(cax)

        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels(pivot.columns)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(pivot.index)

        ax3.set_xlabel("Environment Satisfaction")
        ax3.set_ylabel("Job Satisfaction")
        ax3.set_title("Attrition Probability by Satisfaction Levels", color="#f2f2f2")

        st.pyplot(fig3)

    else:
        st.info("Heatmap unavailable — dataset missing satisfaction columns.")

    st.write("---")


    # -----------------------------
    # STEP 7: Dashboard Polish — Layout & Shadows
    # -----------------------------
    st.markdown("""
    <style>
    /* Panels padding & shadow */
    div[data-testid="stMetric"] {
        background-color: #1c1c1c !important;
        padding: 18px !important;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 0 10px #0005;
    }

    /* Chart containers */
    .plot-container {
        background-color: #181818 !important;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # -----------------------------
    # STEP 6: Top 10 High-Risk Employees
    # -----------------------------
    st.subheader("🔥 Top 10 Highest-Risk Employees")

    top10 = temp_df.sort_values("AttritionProbability", ascending=False).head(10)[
        ["AttritionProbability", "RiskLabel"] + FEATURES
    ]

    st.dataframe(top10)

    st.write("---")
    st.caption("HR Dashboard |")


# ------------------------------------------------------
# SIDEBAR UPLOAD
# ------------------------------------------------------
st.sidebar.markdown("### Upload Dataset (One Time)")
uploaded_global = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="global")

if uploaded_global is not None:
    st.session_state["global_df"] = pd.read_csv(uploaded_global)
    st.sidebar.success("Dataset stored.")

# ------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------
if "entered" not in st.session_state:
    st.session_state["entered"] = False

if not st.session_state["entered"]:
    landing_page()
else:
    page = st.sidebar.radio(
        "Navigation",
        ["Main", "Batch Prediction", "Recommendation Engine", "Dashboard"]
    )

    if page == "Main": main_app()
    elif page == "Batch Prediction": batch_page()
    elif page == "Recommendation Engine": recommendation_engine_page()
    elif page == "Dashboard": dashboard_page()
