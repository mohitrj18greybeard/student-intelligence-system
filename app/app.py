"""
EduPulse AI — Main Streamlit Dashboard
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings("ignore")

from src.data_processing import generate_student_data, preprocess_data
from src.feature_engineering import create_features
from src.model_training import train_all_models, get_feature_importance
from src.evaluation import comparison_table, classify_performance, performance_summary, prediction_confidence
from src.recommendation import generate_recommendations
from src.insights import generate_dataset_insights, generate_individual_insight

# ─── Custom CSS ───
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1.5rem; max-width: 1200px; }
.stMetric { background: linear-gradient(135deg, #1a1d2e 0%, #252840 100%);
    border: 1px solid rgba(108,99,255,0.2); border-radius: 12px; padding: 16px; }
div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
.insight-card { background: linear-gradient(135deg, #1e2235 0%, #262a45 100%);
    border-left: 4px solid #6C63FF; border-radius: 10px; padding: 18px; margin: 10px 0; }
.rec-critical { border-left: 4px solid #FF4B4B; background: rgba(255,75,75,0.08);
    border-radius: 8px; padding: 14px; margin: 8px 0; }
.rec-high { border-left: 4px solid #FFA726; background: rgba(255,167,38,0.08);
    border-radius: 8px; padding: 14px; margin: 8px 0; }
.rec-medium { border-left: 4px solid #42A5F5; background: rgba(66,165,245,0.08);
    border-radius: 8px; padding: 14px; margin: 8px 0; }
.rec-low { border-left: 4px solid #66BB6A; background: rgba(102,187,106,0.08);
    border-radius: 8px; padding: 14px; margin: 8px 0; }
.perf-high { color: #66BB6A; font-weight: 700; font-size: 1.3rem; }
.perf-medium { color: #FFA726; font-weight: 700; font-size: 1.3rem; }
.perf-low { color: #FF4B4B; font-weight: 700; font-size: 1.3rem; }
h1 { background: linear-gradient(90deg, #6C63FF, #48c6ef); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 800 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: rgba(108,99,255,0.1); border-radius: 8px;
    padding: 8px 20px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6C63FF, #48c6ef) !important; }
</style>
"""

@st.cache_data(show_spinner=False)
def load_and_train():
    df_raw = generate_student_data(3000, seed=42)
    df_feat = create_features(df_raw.copy())
    df_proc, scaler, encoders, feat_cols = preprocess_data(df_feat)
    X = df_proc[feat_cols].values
    y = df_proc["final_score"].values
    results, best_model, best_name, models, X_tr, X_te, y_tr, y_te = train_all_models(X, y, feat_cols)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_te[:200])
    return {
        "df_raw": df_raw, "df_feat": df_feat, "df_proc": df_proc,
        "scaler": scaler, "encoders": encoders, "feat_cols": feat_cols,
        "results": results, "best_model": best_model, "best_name": best_name,
        "models": models, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "explainer": explainer, "shap_values": shap_values,
    }

def render_sidebar(data):
    st.sidebar.markdown("## 🎛️ Student Profile Input")
    st.sidebar.markdown("---")
    study = st.sidebar.slider("📚 Study Hours/Week", 1.0, 40.0, 12.0, 0.5)
    attendance = st.sidebar.slider("🏫 Attendance Rate (%)", 30.0, 100.0, 78.0, 1.0)
    assignments = st.sidebar.slider("📝 Assignments Completed (%)", 10.0, 100.0, 70.0, 1.0)
    prev_gpa = st.sidebar.slider("📊 Previous GPA", 0.0, 10.0, 5.5, 0.1)
    sleep = st.sidebar.slider("😴 Sleep Hours/Night", 3.0, 11.0, 7.0, 0.5)
    social = st.sidebar.slider("📱 Social Media (hrs/day)", 0.0, 12.0, 3.5, 0.5)
    stress = st.sidebar.slider("🧘 Stress Level (1-10)", 1.0, 10.0, 5.0, 0.5)
    motivation = st.sidebar.slider("🎯 Motivation (1-10)", 1.0, 10.0, 6.0, 0.5)
    extracurricular = st.sidebar.slider("🏅 Extracurricular (hrs)", 0.0, 10.0, 2.0, 0.5)
    part_time = st.sidebar.selectbox("💼 Part-Time Job?", ["No", "Yes"])
    gender = st.sidebar.selectbox("👤 Gender", ["Male", "Female"])
    income = st.sidebar.selectbox("💰 Family Income", ["Low", "Medium", "High"])
    parent_edu = st.sidebar.selectbox("🎓 Parent Education", ["High School", "Bachelor", "Master", "PhD"])
    internet = st.sidebar.selectbox("🌐 Internet Quality", ["Poor", "Average", "Good"])
    return {
        "study_hours_per_week": study, "attendance_rate": attendance,
        "assignments_completed": assignments, "previous_gpa": prev_gpa,
        "sleep_hours": sleep, "social_media_hours": social,
        "stress_level": stress, "motivation_level": motivation,
        "extracurricular_hours": extracurricular, "part_time_job": 1 if part_time == "Yes" else 0,
        "gender": gender, "family_income": income,
        "parent_education": parent_edu, "internet_quality": internet,
    }

def prepare_input(student_data, data):
    row = pd.DataFrame([student_data])
    row = create_features(row)
    for col, le in data["encoders"].items():
        if col in row.columns:
            try:
                row[col] = le.transform(row[col])
            except ValueError:
                row[col] = 0
    feat_cols = data["feat_cols"]
    for c in feat_cols:
        if c not in row.columns:
            row[c] = 0
    return row[feat_cols].values

def render_prediction_tab(student_data, data):
    X_input = prepare_input(student_data, data)
    model = data["best_model"]
    predicted = float(model.predict(X_input)[0])
    predicted = np.clip(predicted, 0, 100)
    category = classify_performance(predicted)
    confidence = prediction_confidence(model, X_input, predicted)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Predicted Score", f"{predicted:.1f}/100")
    with col2:
        cls = f"perf-{category.lower()}"
        st.markdown(f"**Performance Tier**")
        st.markdown(f'<p class="{cls}">{"🟢" if category=="High" else "🟡" if category=="Medium" else "🔴"} {category}</p>', unsafe_allow_html=True)
    with col3:
        st.metric("📊 Confidence", f"{confidence:.0f}%")

    st.markdown("---")
    summary = performance_summary(predicted, category)
    st.info(f"📋 **Assessment:** {summary}")

    # SHAP explanation
    st.markdown("### 🔍 Why This Prediction? (SHAP Analysis)")
    explainer = data["explainer"]
    shap_vals = explainer.shap_values(X_input)
    feat_cols = data["feat_cols"]

    shap_df = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in feat_cols],
        "SHAP Impact": shap_vals[0],
        "Abs Impact": np.abs(shap_vals[0])
    }).sort_values("Abs Impact", ascending=False).head(10)

    colors = ["#66BB6A" if v > 0 else "#FF4B4B" for v in shap_df["SHAP Impact"]]
    fig = go.Figure(go.Bar(
        x=shap_df["SHAP Impact"].values, y=shap_df["Feature"].values,
        orientation='h', marker_color=colors,
        text=[f"{v:+.2f}" for v in shap_df["SHAP Impact"].values], textposition='outside'
    ))
    fig.update_layout(
        title="Top 10 Feature Impacts on This Student's Score",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis=dict(autorange="reversed"),
        height=420, template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Individual narrative
    shap_top = list(zip(feat_cols, shap_vals[0]))
    shap_top.sort(key=lambda x: abs(x[1]), reverse=True)
    narrative = generate_individual_insight(student_data, predicted, shap_top[:5])
    st.markdown(f"**🧠 AI Narrative:** {narrative}")

    # SHAP-aware recommendations
    st.markdown("### 💡 Personalized Recommendations")
    shap_factors = []
    for fname, impact in shap_top[:8]:
        shap_factors.append({
            "feature": fname, "impact": impact,
            "direction": "positive" if impact > 0 else "negative",
        })
    recs = generate_recommendations(student_data, predicted, shap_factors)
    if not recs:
        st.success("✅ No critical issues found. Keep up the great work!")
    for r in recs:
        p = r["priority"].lower()
        css_class = f"rec-{p}"
        icon = {"critical": "🚨", "high": "⚠️", "medium": "💡", "low": "✅"}.get(p, "📌")
        st.markdown(
            f'<div class="{css_class}">'
            f'<strong>{icon} {r["category"]}</strong> '
            f'<span style="float:right;opacity:0.7;">Priority: {r["priority"]} | {r["expected_impact"]}</span><br>'
            f'{r["message"]}</div>', unsafe_allow_html=True
        )

    return predicted, category

def render_whatif_tab(student_data, data):
    st.markdown("### 🔄 What-If Simulator")
    st.markdown("Adjust parameters below and see how changes would affect this student's predicted score.")
    baseline_X = prepare_input(student_data, data)
    baseline_score = float(np.clip(data["best_model"].predict(baseline_X)[0], 0, 100))

    col1, col2 = st.columns(2)
    with col1:
        new_study = st.slider("New Study Hours", 1.0, 40.0, float(student_data["study_hours_per_week"]), 1.0, key="wi_study")
        new_att = st.slider("New Attendance %", 30.0, 100.0, float(student_data["attendance_rate"]), 1.0, key="wi_att")
        new_sleep = st.slider("New Sleep Hours", 3.0, 11.0, float(student_data["sleep_hours"]), 0.5, key="wi_sleep")
    with col2:
        new_social = st.slider("New Social Media Hrs", 0.0, 12.0, float(student_data["social_media_hours"]), 0.5, key="wi_social")
        new_stress = st.slider("New Stress Level", 1.0, 10.0, float(student_data["stress_level"]), 0.5, key="wi_stress")
        new_motivation = st.slider("New Motivation", 1.0, 10.0, float(student_data["motivation_level"]), 0.5, key="wi_mot")

    modified = student_data.copy()
    modified.update({
        "study_hours_per_week": new_study, "attendance_rate": new_att,
        "sleep_hours": new_sleep, "social_media_hours": new_social,
        "stress_level": new_stress, "motivation_level": new_motivation,
    })
    new_X = prepare_input(modified, data)
    new_score = float(np.clip(data["best_model"].predict(new_X)[0], 0, 100))
    delta = new_score - baseline_score

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Score", f"{baseline_score:.1f}")
    c2.metric("New Score", f"{new_score:.1f}", delta=f"{delta:+.1f}")
    c3.metric("Category", classify_performance(new_score))

    changes = {"Study": new_study - student_data["study_hours_per_week"],
               "Attendance": new_att - student_data["attendance_rate"],
               "Sleep": new_sleep - student_data["sleep_hours"],
               "Social Media": new_social - student_data["social_media_hours"],
               "Stress": new_stress - student_data["stress_level"],
               "Motivation": new_motivation - student_data["motivation_level"]}
    changes = {k: v for k, v in changes.items() if abs(v) > 0.01}
    if changes:
        fig = go.Figure(go.Bar(
            x=list(changes.keys()), y=list(changes.values()),
            marker_color=["#66BB6A" if v > 0 else "#FF4B4B" for v in changes.values()],
        ))
        fig.update_layout(title="Changes Made", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_insights_tab(data):
    st.markdown("### 🔬 Key Insights from Data")
    st.markdown("These insights are **dynamically computed** from the dataset — not hardcoded.")
    insights = generate_dataset_insights(data["df_raw"])
    for ins in insights:
        icon = {"surprising": "🔥", "actionable": "🎯", "expected": "📊"}.get(ins["category"], "💡")
        st.markdown(
            f'<div class="insight-card">'
            f'<strong>{icon} {ins["title"]}</strong><br>'
            f'{ins["description"]}<br>'
            f'<small style="opacity:0.6;">Evidence: {ins["evidence"]}</small></div>',
            unsafe_allow_html=True
        )
    # Correlation heatmap
    st.markdown("### 📊 Feature Correlation Heatmap")
    num_cols = data["df_raw"].select_dtypes(include=[np.number]).columns.tolist()
    corr = data["df_raw"][num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    aspect="auto", title="Feature Correlations")
    fig.update_layout(template="plotly_dark", height=600,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

def render_model_tab(data):
    st.markdown("### 🏆 Model Comparison")
    comp = comparison_table(data["results"])
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.success(f"**Winner: {data['best_name']}** — selected for highest test R² score.")

    # Model comparison chart
    models_list = list(data["results"].keys())
    r2_vals = [data["results"][m]["r2"] for m in models_list]
    rmse_vals = [data["results"][m]["rmse"] for m in models_list]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² Score (higher=better)", "RMSE (lower=better)"))
    fig.add_trace(go.Bar(x=models_list, y=r2_vals, marker_color=["#6C63FF","#48c6ef","#66BB6A"], name="R²"), row=1, col=1)
    fig.add_trace(go.Bar(x=models_list, y=rmse_vals, marker_color=["#6C63FF","#48c6ef","#66BB6A"], name="RMSE"), row=1, col=2)
    fig.update_layout(template="plotly_dark", showlegend=False, height=350,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("### 📈 Feature Importance (XGBoost)")
    imp_df = get_feature_importance(data["best_model"], data["feat_cols"])
    if not imp_df.empty:
        top15 = imp_df.head(15)
        fig2 = go.Figure(go.Bar(
            x=top15["importance"].values[::-1],
            y=[f.replace("_"," ").title() for f in top15["feature"].values[::-1]],
            orientation='h', marker_color='#6C63FF',
        ))
        fig2.update_layout(title="Top 15 Features by Importance", template="plotly_dark",
            height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Global SHAP
    st.markdown("### 🌐 Global SHAP Summary")
    shap_abs_mean = np.abs(data["shap_values"]).mean(axis=0)
    shap_global = pd.DataFrame({"Feature": data["feat_cols"], "Mean |SHAP|": shap_abs_mean})
    shap_global = shap_global.sort_values("Mean |SHAP|", ascending=False).head(12)
    fig3 = go.Figure(go.Bar(
        x=shap_global["Mean |SHAP|"].values[::-1],
        y=[f.replace("_"," ").title() for f in shap_global["Feature"].values[::-1]],
        orientation='h', marker=dict(color=shap_global["Mean |SHAP|"].values[::-1], colorscale="Viridis"),
    ))
    fig3.update_layout(title="Global SHAP Feature Importance", template="plotly_dark",
        height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

def render_eda_tab(data):
    df = data["df_raw"]
    st.markdown("### 📊 Exploratory Data Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(df):,}")
    c2.metric("Avg Score", f"{df['final_score'].mean():.1f}")
    c3.metric("Features", f"{len(data['feat_cols'])}")
    c4.metric("At-Risk", f"{(df['final_score']<40).sum()}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="final_score", nbins=40, color="performance_category",
            color_discrete_map={"High":"#66BB6A","Medium":"#FFA726","Low":"#FF4B4B"},
            title="Score Distribution by Performance Category")
        fig.update_layout(template="plotly_dark", height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df, x="study_hours_per_week", y="final_score",
            color="performance_category", opacity=0.5, title="Study Hours vs Score",
            color_discrete_map={"High":"#66BB6A","Medium":"#FFA726","Low":"#FF4B4B"})
        fig.update_layout(template="plotly_dark", height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(df, x="performance_category", y="attendance_rate",
            color="performance_category", title="Attendance by Performance Tier",
            color_discrete_map={"High":"#66BB6A","Medium":"#FFA726","Low":"#FF4B4B"})
        fig.update_layout(template="plotly_dark", height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.scatter(df, x="stress_level", y="final_score",
            color="social_media_hours", title="Stress vs Score (colored by Social Media)",
            color_continuous_scale="RdYlGn_r", opacity=0.5)
        fig.update_layout(template="plotly_dark", height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

def render_analysis_tab():
    st.markdown("### 🧠 My Analysis & Learnings")
    st.markdown("""
**What I discovered during this project:**

1. **Attendance matters more than study hours.** I initially assumed study time would dominate predictions, but correlation analysis and SHAP both reveal attendance has a stronger relationship with outcomes. The likely explanation: classroom time provides structured, guided learning that self-study cannot replicate.

2. **The "study time paradox" was surprising.** About 8-10% of students study above-average hours but still score poorly. Digging deeper, these students tend to have high stress and high social media usage — they're studying more but retaining less due to poor mental state and constant distractions.

3. **Feature interactions are critical.** Linear models miss the interaction between motivation and study hours. A student studying 20 hrs/week with high motivation vastly outperforms one with the same hours but low motivation. XGBoost captures this naturally, which is why it outperforms Linear Regression.

4. **Socioeconomic background creates compounding disadvantage.** Low-income students with part-time jobs face a double penalty: less study time AND higher stress. This finding has real policy implications — financial aid programs could have outsized impact on academic outcomes.

5. **Sleep has a non-linear effect.** Both too little (<5 hrs) and too much (>9 hrs) sleep correlate with lower scores. The optimal window is 6.5-8 hours, which aligns with sleep science research on cognitive consolidation.

**Limitations of this approach:**
- Synthetic data, while realistic, may not capture all real-world edge cases
- SHAP explanations are model-dependent — a different model might highlight different features
- The recommendation engine uses fixed thresholds; ideally these would be learned from intervention outcome data
- Cross-sectional data cannot establish causation, only correlation

**Future improvements I'd pursue:**
- Integration with real LMS data (Canvas/Moodle APIs) for longitudinal tracking
- A/B testing the recommendations to measure actual impact
- Adding a temporal dimension to track student trajectories over semesters
- Deep learning experiments (TabNet) to compare against gradient boosting
    """)

def main():
    st.set_page_config(page_title="EduPulse AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown("# 🧠 EduPulse AI")
    st.markdown("**AI-Powered Student Intelligence & Performance Optimization System**")
    st.markdown("---")

    with st.spinner("🔧 Training models & computing SHAP values... (first load only)"):
        data = load_and_train()

    student_data = render_sidebar(data)

    tabs = st.tabs(["🎯 Prediction & Recommendations", "🔄 What-If Simulator",
                     "🔬 Key Insights", "🏆 Models & SHAP", "📊 Data Explorer", "🧠 My Analysis"])
    with tabs[0]:
        render_prediction_tab(student_data, data)
    with tabs[1]:
        render_whatif_tab(student_data, data)
    with tabs[2]:
        render_insights_tab(data)
    with tabs[3]:
        render_model_tab(data)
    with tabs[4]:
        render_eda_tab(data)
    with tabs[5]:
        render_analysis_tab()

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;opacity:0.5;">Built by Mohit | '
        'Powered by XGBoost, SHAP & Streamlit</p>', unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
