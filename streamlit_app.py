"""EduPulse AI — Cloud-ready Streamlit Dashboard. Auto-trains on first deploy."""
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go

st.set_page_config(page_title="EduPulse AI", page_icon="🧠", layout="wide")

@st.cache_resource(show_spinner="🚀 Initializing AI engine (first run only)...")
def boot_engine():
    os.chdir(ROOT)
    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    if not os.path.exists(os.path.join(models_dir, "best_regression_model.joblib")):
        from src.training.training_pipeline import run_training_pipeline
        run_training_pipeline(n_students=3000, seed=42, use_mlflow=False, output_dir=models_dir)
    from src.inference.inference_engine import InferenceEngine
    engine = InferenceEngine(models_dir)
    engine.load()
    return engine

try:
    ENGINE = boot_engine()
except Exception as e:
    st.error(f"⚠️ Engine failed: {e}")
    import traceback
    with st.expander("Full traceback"): st.code(traceback.format_exc())
    st.stop()

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{font-family:'Inter',sans-serif}
.main,.stApp{background:linear-gradient(135deg,#0f0c29,#1a1a3e,#24243e)}
.metric-card{background:rgba(255,255,255,.06);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:24px;text-align:center;transition:transform .3s,box-shadow .3s}
.metric-card:hover{transform:translateY(-4px);box-shadow:0 12px 40px rgba(99,102,241,.3)}
.metric-value{font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.metric-label{font-size:.85rem;color:rgba(255,255,255,.6);text-transform:uppercase;letter-spacing:1px;margin-top:4px}
.hero-title{font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:0}
.hero-sub{text-align:center;color:rgba(255,255,255,.5);font-size:1.1rem;margin-top:4px}
.risk-critical{color:#ef4444;font-weight:700}.risk-high{color:#f97316;font-weight:700}
.risk-moderate{color:#eab308;font-weight:700}.risk-low{color:#22c55e;font-weight:700}
.risk-excellent{color:#06b6d4;font-weight:700}
.rec-card{background:rgba(255,255,255,.04);border-left:4px solid #818cf8;border-radius:8px;padding:12px 16px;margin:8px 0}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e1b4b,#312e81)}
</style>""", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.markdown('<p style="text-align:center;font-size:2rem">🧠</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="text-align:center;font-weight:700;font-size:1.3rem;color:#c084fc">EduPulse AI</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")
role = st.sidebar.selectbox("👤 Select Role", ["Student", "Teacher", "Admin"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Student Input")
study_hours = st.sidebar.slider("📚 Study Hours/Day", 0.0, 14.0, 4.5, 0.5)
attendance = st.sidebar.slider("🏫 Attendance %", 30, 100, 78)
assignments = st.sidebar.slider("📝 Assignments %", 10, 100, 72)
previous_gpa = st.sidebar.slider("📈 Previous GPA", 0.0, 10.0, 6.5, 0.1)
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 3.0, 11.0, 7.0, 0.5)
social_media = st.sidebar.slider("📱 Social Media Hrs", 0.0, 12.0, 3.0, 0.5)
part_time = st.sidebar.selectbox("💼 Part-time Job", [0, 1], format_func=lambda x: "Yes" if x else "No")
stress = st.sidebar.slider("😰 Stress Level", 1, 10, 5)
motivation = st.sidebar.slider("🎯 Motivation Level", 1, 10, 6)
week = st.sidebar.slider("📅 Semester Week", 1, 16, 8)
internet = st.sidebar.selectbox("🌐 Internet Quality", ["good", "average", "poor"])
income = st.sidebar.selectbox("💰 Family Income", ["medium", "low", "high"])
education = st.sidebar.selectbox("🎓 Parent Education", ["graduate", "school", "postgraduate"])

student_data = {"student_id": "STU_LIVE", "study_hours": study_hours, "attendance": float(attendance),
    "assignments_completed": float(assignments), "previous_gpa": previous_gpa,
    "sleep_hours": sleep_hours, "social_media_hours": social_media, "part_time_job": part_time,
    "extracurricular_hours": 2.0, "stress_level": float(stress), "motivation_level": float(motivation),
    "internet_quality": internet, "family_income": income, "parent_education": education, "week": week}

# ═══ STUDENT VIEW ═══
if role == "Student":
    st.markdown('<h1 class="hero-title">🧠 EduPulse AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">AI-Powered Student Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown("")
    with st.spinner("🤖 AI analyzing..."):
        result = ENGINE.predict_student(student_data, use_shap=True)
    if result:
        s, r, g = result["predicted_score"], result["risk_level"], result["grade"]
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{s:.1f}</div><div class="metric-label">Predicted Score</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{g}</div><div class="metric-label">Grade</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value risk-{r}">{r.upper()}</div><div class="metric-label">Risk Level</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-value">{result.get("confidence","N/A").upper()}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
        st.markdown("")
        left, right = st.columns(2)
        with left:
            st.markdown("### 🔍 Top Influencing Factors")
            factors = result.get("top_factors", [])
            if factors:
                colors = ["#22c55e" if f["direction"]=="positive" else "#ef4444" for f in factors]
                fig = go.Figure(go.Bar(x=[f["impact"] for f in factors], y=[f["feature"] for f in factors], orientation='h', marker_color=colors))
                fig.update_layout(template="plotly_dark", height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=30))
                st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("### 💡 AI Recommendations")
            for rec in result.get("recommendations", [])[:6]:
                bc = {"critical":"#ef4444","high":"#f97316","medium":"#eab308","low":"#22c55e"}.get(rec["priority"],"#818cf8")
                st.markdown(f'<div class="rec-card" style="border-left-color:{bc}"><strong>{rec.get("icon","")} {rec["message"]}</strong><br/><small style="color:rgba(255,255,255,.4)">Impact: {rec.get("expected_impact","")} | {rec["priority"].upper()}</small></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🔄 What-If Simulation")
        sc1,sc2,sc3 = st.columns(3)
        sim_study = sc1.slider("Sim Study Hrs", 0.0, 14.0, study_hours, 0.5, key="ss")
        sim_sleep = sc2.slider("Sim Sleep Hrs", 3.0, 11.0, sleep_hours, 0.5, key="sl")
        sim_social = sc3.slider("Sim Social Media", 0.0, 12.0, social_media, 0.5, key="sm")
        changes = {"study_hours": sim_study, "sleep_hours": sim_sleep, "social_media_hours": sim_social}
        if any(changes[k] != student_data[k] for k in changes):
            wr = ENGINE.what_if_simulation(student_data, changes)
            if wr:
                m1,m2,m3 = st.columns(3)
                m1.metric("Current", f"{wr['original_score']:.1f}")
                m2.metric("Simulated", f"{wr['simulated_score']:.1f}", delta=f"{wr['score_delta']:+.1f}")
                m3.metric("Risk", wr["simulated_risk"].upper(), delta=f"from {wr['original_risk'].upper()}")

# ═══ TEACHER VIEW ═══
elif role == "Teacher":
    st.markdown('<h1 class="hero-title">📊 Class Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Teacher Dashboard — Identify At-Risk Students</p>', unsafe_allow_html=True)
    @st.cache_data(ttl=300)
    def get_class_data():
        from src.data.data_generator import generate_student_data
        df = generate_student_data(200, seed=42)
        preds = []
        for _, row in df.iterrows():
            try: preds.append(ENGINE.predict_student(row.to_dict(), use_shap=False))
            except: preds.append({"predicted_score":50,"risk_level":"moderate","grade":"C"})
        df["predicted_score"] = [p["predicted_score"] for p in preds]
        df["risk_level"] = [p["risk_level"] for p in preds]
        df["grade"] = [p["grade"] for p in preds]
        return df
    with st.spinner("📊 Analyzing class..."): cdf = get_class_data()
    if not cdf.empty:
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(cdf)}</div><div class="metric-label">Total Students</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{cdf["predicted_score"].mean():.1f}</div><div class="metric-label">Avg Score</div></div>', unsafe_allow_html=True)
        ar = len(cdf[cdf["risk_level"].isin(["critical","high"])])
        c3.markdown(f'<div class="metric-card"><div class="metric-value risk-high">{ar}</div><div class="metric-label">At-Risk</div></div>', unsafe_allow_html=True)
        tp = len(cdf[cdf["predicted_score"]>=80])
        c4.markdown(f'<div class="metric-card"><div class="metric-value risk-excellent">{tp}</div><div class="metric-label">Top Performers</div></div>', unsafe_allow_html=True)
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Score Distribution")
            fig = px.histogram(cdf, x="predicted_score", nbins=20, color_discrete_sequence=["#818cf8"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 🎯 Risk Distribution")
            rc = cdf["risk_level"].value_counts().reset_index(); rc.columns=["Risk","Count"]
            cm = {"critical":"#ef4444","high":"#f97316","moderate":"#eab308","low":"#22c55e","excellent":"#06b6d4"}
            fig = px.pie(rc, names="Risk", values="Count", color="Risk", color_discrete_map=cm, hole=0.4)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📈 Attendance vs Score")
        fig = px.scatter(cdf, x="attendance", y="predicted_score", color="risk_level", size="study_hours", color_discrete_map=cm, opacity=0.7)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 🚨 At-Risk Students")
        ard = cdf[cdf["risk_level"].isin(["critical","high"])][["student_id","predicted_score","risk_level","study_hours","attendance","stress_level","grade"]].sort_values("predicted_score").head(20)
        st.dataframe(ard, use_container_width=True, height=300)

# ═══ ADMIN VIEW ═══
elif role == "Admin":
    st.markdown('<h1 class="hero-title">⚙️ System Admin</h1>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown('<div class="metric-card"><div class="metric-value" style="font-size:1.5rem">🟢 Online</div><div class="metric-label">Status</div></div>', unsafe_allow_html=True)
    mt = type(ENGINE.model).__name__ if ENGINE.model else "N/A"
    c2.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.2rem">{mt}</div><div class="metric-label">Model</div></div>', unsafe_allow_html=True)
    fc = len(ENGINE.feature_names) if ENGINE.feature_names else "N/A"
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{fc}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-card"><div class="metric-value">v2.0</div><div class="metric-label">Version</div></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("### 📊 Model Performance")
    import json
    rp = os.path.join(ROOT, "models", "training_results.json")
    if os.path.exists(rp):
        with open(rp) as f: res = json.load(f)
        if "regression" in res:
            rows = [{"Model":n, "R²":round(m.get("r2",0),4), "RMSE":round(m.get("rmse",0),4), "MAE":round(m.get("mae",0),4)} for n,m in res["regression"].items()]
            pdf = pd.DataFrame(rows).sort_values("R²", ascending=False)
            st.dataframe(pdf, use_container_width=True)
            fig = px.bar(pdf, x="Model", y="R²", color="R²", color_continuous_scale="Viridis")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🏆 Feature Importance")
    try:
        md = os.path.join(ROOT, "models")
        fif = [f for f in os.listdir(md) if f.startswith("feature_importance")]
        if fif:
            fi = pd.read_csv(os.path.join(md, fif[0]))
            fig = px.bar(fi.head(15), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Plasma")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    except Exception: st.info("Feature importance not available yet.")
    st.markdown("### 🔄 Retrain")
    if st.button("🚀 Retrain Models", type="primary"):
        with st.spinner("Training..."):
            from src.training.training_pipeline import run_training_pipeline
            r = run_training_pipeline(3000, 42, use_mlflow=False, output_dir=os.path.join(ROOT,"models"))
            st.success(f"Done! Best: {r['best_regression_model']} in {r['elapsed_seconds']}s")
            st.cache_resource.clear(); st.rerun()

st.markdown("---")
st.markdown('<div style="text-align:center;color:rgba(255,255,255,.3);font-size:.8rem">EduPulse AI v2.0 — XGBoost, SHAP, Streamlit</div>', unsafe_allow_html=True)
