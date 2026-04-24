"""
STEP 11: Streamlit Deployment — EduPulse AI Dashboard
======================================================
Premium, role-aware student intelligence platform.
Auto-trains models on first deploy (Streamlit Cloud compatible).
"""
import sys, os

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EduPulse AI", page_icon="🧠", layout="wide")

# ── Boot: auto-train on first run ──────────────────────────────────────
@st.cache_resource(show_spinner="🚀 Training AI models (first run only — ~30s)...")
def boot():
    os.chdir(ROOT)
    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.joblib")

    if not os.path.exists(model_path):
        from src.data_processing import generate_student_data, preprocess_data
        from src.feature_engineering import create_features
        from src.model_training import train_all_models, save_model, get_feature_importance
        import json, joblib

        df = generate_student_data(3000, seed=42)
        df = create_features(df)
        df_proc, scaler, encoders, feat_cols = preprocess_data(df)

        X = df_proc[feat_cols].values
        y = df_proc["final_score"].values
        results, best_model, best_name, all_models, _, _, _, _ = train_all_models(X, y, feat_cols)

        save_model(best_model, os.path.join(models_dir, "best_model.joblib"))
        joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
        joblib.dump(encoders, os.path.join(models_dir, "encoders.joblib"))
        with open(os.path.join(models_dir, "feature_cols.json"), "w") as f:
            json.dump(feat_cols, f)
        with open(os.path.join(models_dir, "results.json"), "w") as f:
            json.dump(results, f, default=str)
        fi = get_feature_importance(best_model, feat_cols)
        if not fi.empty:
            fi.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)

    import joblib, json
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    encoders = joblib.load(os.path.join(models_dir, "encoders.joblib"))
    with open(os.path.join(models_dir, "feature_cols.json")) as f:
        feat_cols = json.load(f)
    results = {}
    rp = os.path.join(models_dir, "results.json")
    if os.path.exists(rp):
        with open(rp) as f:
            results = json.load(f)
    return model, scaler, encoders, feat_cols, results

try:
    MODEL, SCALER, ENCODERS, FEAT_COLS, RESULTS = boot()
except Exception as e:
    st.error(f"⚠️ Init failed: {e}")
    import traceback
    with st.expander("Traceback"): st.code(traceback.format_exc())
    st.stop()

# ── Predict helper ──
def predict(input_data: dict):
    from src.feature_engineering import create_features
    from src.data_processing import preprocess_data
    df = pd.DataFrame([input_data])
    df = create_features(df)
    # Encode categoricals
    for col, le in ENCODERS.items():
        if col in df.columns:
            try: df[col] = le.transform(df[col])
            except ValueError: df[col] = 0
    # Ensure all feature columns exist
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0
    X = df[FEAT_COLS].values
    score = float(np.clip(MODEL.predict(X)[0], 0, 100))
    return score

def get_shap_factors(input_data: dict):
    try:
        import shap
        from src.feature_engineering import create_features
        df = pd.DataFrame([input_data])
        df = create_features(df)
        for col, le in ENCODERS.items():
            if col in df.columns:
                try: df[col] = le.transform(df[col])
                except: df[col] = 0
        for c in FEAT_COLS:
            if c not in df.columns:
                df[c] = 0
        X = df[FEAT_COLS].values
        explainer = shap.Explainer(MODEL, feature_names=FEAT_COLS)
        sv = explainer(X)
        vals = sv.values[0]
        pairs = sorted(zip(FEAT_COLS, vals), key=lambda x: abs(x[1]), reverse=True)[:8]
        return [{"feature": f, "impact": round(float(v), 3),
                 "direction": "positive" if v > 0 else "negative"} for f, v in pairs]
    except:
        return []

# ── CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{font-family:'Inter',sans-serif}
.main,.stApp{background:linear-gradient(135deg,#0f0c29,#1a1a3e,#24243e)}
.metric-card{background:rgba(255,255,255,.06);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:24px;text-align:center;transition:transform .3s,box-shadow .3s}
.metric-card:hover{transform:translateY(-4px);box-shadow:0 12px 40px rgba(99,102,241,.3)}
.metric-value{font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.metric-label{font-size:.85rem;color:rgba(255,255,255,.6);text-transform:uppercase;letter-spacing:1px;margin-top:4px}
.hero-title{font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center}
.hero-sub{text-align:center;color:rgba(255,255,255,.5);font-size:1.1rem;margin-top:4px}
.risk-Low{color:#22c55e;font-weight:700}.risk-Medium{color:#eab308;font-weight:700}.risk-High{color:#ef4444;font-weight:700}
.rec-card{background:rgba(255,255,255,.04);border-left:4px solid #818cf8;border-radius:8px;padding:12px 16px;margin:8px 0}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e1b4b,#312e81)}
</style>""", unsafe_allow_html=True)

# ── Sidebar Inputs ──
st.sidebar.markdown('<p style="text-align:center;font-size:2rem">🧠</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="text-align:center;font-weight:700;font-size:1.3rem;color:#c084fc">EduPulse AI</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Student Profile")

study = st.sidebar.slider("📚 Study Hours/Week", 1, 40, 12)
attendance = st.sidebar.slider("🏫 Attendance %", 30, 100, 78)
assignments = st.sidebar.slider("📝 Assignments %", 10, 100, 72)
gpa = st.sidebar.slider("📈 Previous GPA", 0.0, 10.0, 6.5, 0.1)
sleep = st.sidebar.slider("😴 Sleep Hours", 3.0, 11.0, 7.0, 0.5)
social = st.sidebar.slider("📱 Social Media Hrs/Day", 0.0, 12.0, 3.0, 0.5)
part_time = st.sidebar.selectbox("💼 Part-time Job", [0, 1], format_func=lambda x: "Yes" if x else "No")
stress = st.sidebar.slider("😰 Stress Level", 1, 10, 5)
motivation = st.sidebar.slider("🎯 Motivation", 1, 10, 6)
extra = st.sidebar.slider("🏀 Extracurricular Hrs", 0.0, 10.0, 2.0, 0.5)
gender = st.sidebar.selectbox("👤 Gender", ["Male", "Female"])
income = st.sidebar.selectbox("💰 Family Income", ["Medium", "Low", "High"])
parent_edu = st.sidebar.selectbox("🎓 Parent Education", ["Bachelor", "High School", "Master", "PhD"])
internet = st.sidebar.selectbox("🌐 Internet Quality", ["Good", "Average", "Poor"])

input_data = {
    "gender": gender, "family_income": income, "parent_education": parent_edu,
    "internet_quality": internet, "study_hours_per_week": float(study),
    "attendance_rate": float(attendance), "assignments_completed": float(assignments),
    "previous_gpa": gpa, "sleep_hours": sleep, "social_media_hours": social,
    "part_time_job": part_time, "extracurricular_hours": extra,
    "stress_level": float(stress), "motivation_level": float(motivation),
}

# ═══ MAIN DASHBOARD ═══
st.markdown('<h1 class="hero-title">🧠 EduPulse AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI-Powered Student Intelligence & Performance Optimization</p>', unsafe_allow_html=True)
st.markdown("")

tab1, tab2, tab3 = st.tabs(["🎓 Prediction & Insights", "📊 Model Performance", "📈 EDA"])

# ── TAB 1: Prediction ──
with tab1:
    with st.spinner("🤖 Analyzing..."):
        score = predict(input_data)
        from src.evaluation import classify_performance
        category = classify_performance(score)
        factors = get_shap_factors(input_data)
        from src.recommendation import generate_recommendations
        recs = generate_recommendations(input_data, score, factors)

    # KPI Cards
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{score:.1f}</div><div class="metric-label">Predicted Score</div></div>', unsafe_allow_html=True)
    grade = "A+" if score>=93 else "A" if score>=85 else "B+" if score>=77 else "B" if score>=70 else "C" if score>=60 else "D" if score>=50 else "F"
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{grade}</div><div class="metric-label">Grade</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value risk-{category}">{category}</div><div class="metric-label">Performance</div></div>', unsafe_allow_html=True)
    st.markdown("")

    left, right = st.columns(2)
    with left:
        st.markdown("### 🔍 SHAP Feature Impact")
        if factors:
            colors = ["#22c55e" if f["direction"]=="positive" else "#ef4444" for f in factors]
            fig = go.Figure(go.Bar(x=[f["impact"] for f in factors], y=[f["feature"] for f in factors], orientation='h', marker_color=colors))
            fig.update_layout(template="plotly_dark", height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP analysis not available.")

    with right:
        st.markdown("### 💡 Personalized Recommendations")
        if recs:
            for r in recs[:6]:
                bc = {"Critical":"#ef4444","High":"#f97316","Medium":"#eab308","Low":"#22c55e"}.get(r["priority"],"#818cf8")
                st.markdown(f'<div class="rec-card" style="border-left-color:{bc}"><strong>{r["category"]}: {r["message"]}</strong><br/><small style="color:rgba(255,255,255,.4)">Expected: {r["expected_impact"]} | Priority: {r["priority"]}</small></div>', unsafe_allow_html=True)
        else:
            st.success("🌟 Excellent! Keep up the great work.")

    # What-If
    st.markdown("---")
    st.markdown("### 🔄 What-If Simulation")
    wc1,wc2,wc3 = st.columns(3)
    ws = wc1.slider("Sim Study Hrs", 1, 40, study, key="ws")
    wsl = wc2.slider("Sim Sleep Hrs", 3.0, 11.0, sleep, 0.5, key="wsl")
    wsm = wc3.slider("Sim Social Media", 0.0, 12.0, social, 0.5, key="wsm")
    if ws != study or wsl != sleep or wsm != social:
        sim_data = {**input_data, "study_hours_per_week": float(ws), "sleep_hours": wsl, "social_media_hours": wsm}
        sim_score = predict(sim_data)
        delta = sim_score - score
        m1,m2,m3 = st.columns(3)
        m1.metric("Current", f"{score:.1f}")
        m2.metric("Simulated", f"{sim_score:.1f}", delta=f"{delta:+.1f}")
        m3.metric("Change", f"{delta:+.1f} pts", delta="Improved" if delta > 0 else "Declined")

# ── TAB 2: Model Performance ──
with tab2:
    st.markdown("### 📊 Model Comparison (STEP 7)")
    if RESULTS:
        rows = [{"Model": n, "R²": m["r2"], "RMSE": m["rmse"], "MAE": m["mae"], "CV R²": m["cv_r2_mean"]}
                for n, m in RESULTS.items()]
        rdf = pd.DataFrame(rows).sort_values("R²", ascending=False)
        st.dataframe(rdf, use_container_width=True)

        fig = px.bar(rdf, x="Model", y="R²", color="R²", color_continuous_scale="Viridis",
                     title="Model R² Score Comparison")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(rdf, x="Model", y=["RMSE", "MAE"], barmode="group",
                      color_discrete_sequence=["#ef4444", "#eab308"], title="Error Comparison")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🏆 Feature Importance")
    fi_path = os.path.join(ROOT, "models", "feature_importance.csv")
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        fig = px.bar(fi.head(15), x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale="Plasma")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: EDA ──
with tab3:
    st.markdown("### 📈 Exploratory Data Analysis (STEP 5)")
    from src.data_processing import generate_student_data
    @st.cache_data
    def load_eda_data():
        return generate_student_data(1000, seed=99)
    eda_df = load_eda_data()

    st.markdown("#### Score Distribution")
    fig = px.histogram(eda_df, x="final_score", nbins=30, color_discrete_sequence=["#818cf8"],
                       title="Distribution of Final Scores")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("#### Attendance vs Score")
        fig = px.scatter(eda_df, x="attendance_rate", y="final_score", color="performance_category",
                         color_discrete_map={"Low":"#ef4444","Medium":"#eab308","High":"#22c55e"}, opacity=0.6)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with ec2:
        st.markdown("#### Study Hours vs Score")
        fig = px.scatter(eda_df, x="study_hours_per_week", y="final_score", color="performance_category",
                         color_discrete_map={"Low":"#ef4444","Medium":"#eab308","High":"#22c55e"}, opacity=0.6)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Correlation Heatmap")
    num_cols = eda_df.select_dtypes(include=[np.number]).columns.tolist()
    corr = eda_df[num_cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto", title="Feature Correlation Matrix")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Performance Category Breakdown")
    perf = eda_df["performance_category"].value_counts().reset_index()
    perf.columns = ["Category", "Count"]
    fig = px.pie(perf, names="Category", values="Count", hole=0.4,
                 color="Category", color_discrete_map={"Low":"#ef4444","Medium":"#eab308","High":"#22c55e"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center;color:rgba(255,255,255,.3);font-size:.8rem">EduPulse AI v2.0 — XGBoost · SHAP · Streamlit | Built for Excellence</div>', unsafe_allow_html=True)
