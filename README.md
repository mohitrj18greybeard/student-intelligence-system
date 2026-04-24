# 🧠 EduPulse AI — Student Intelligence & Performance Optimization System

> **AI-powered academic performance prediction with SHAP explainability, personalized recommendations, and real-time what-if simulation.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://student-intelligence-system-iappex9cl99exgcqcdvcdcb.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML_Engine-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge)](https://shap.readthedocs.io/)

---

## 📋 Problem Statement

Educational institutions struggle to identify at-risk students early enough for effective intervention. Traditional methods rely on post-exam analysis — by which point it's too late.

**EduPulse AI** solves this by predicting student performance **before exams**, explaining **why** a student may underperform using SHAP, and providing **actionable, personalized recommendations** to improve outcomes.

---

## 🏗️ Project Structure

```
student-intelligence-system/
├── streamlit_app.py              ← Cloud entry point
├── requirements.txt              ← Dependencies
├── .streamlit/config.toml        ← Theme & settings
│
├── app/
│   └── app.py                    ← Main Streamlit dashboard
│
├── src/
│   ├── data_processing.py        ← Data generation & preprocessing (Steps 2-3)
│   ├── feature_engineering.py    ← 12 engineered features (Step 4)
│   ├── model_training.py         ← Training + tuning + evaluation (Steps 6-7, 10)
│   ├── evaluation.py             ← Comparison utilities (Step 7)
│   └── recommendation.py         ← AI recommendation engine (Step 9)
│
├── models/                       ← Saved model artifacts (auto-generated)
├── data/                         ← Generated datasets
└── notebooks/                    ← Jupyter notebooks for analysis
```

---

## 📊 Dataset

**Synthetic, hyper-realistic dataset** of 3,000 students with:
- **Academic**: Study hours, attendance, assignments, GPA
- **Lifestyle**: Sleep, social media, part-time work, extracurriculars
- **Psychological**: Stress level, motivation level
- **Background**: Family income, parent education, internet quality

Features exhibit realistic inter-correlations (e.g., high income → more study hours, high stress → more social media).

---

## 🧪 Approach

| Step | Description |
|------|-------------|
| **1. Data Generation** | 3000 records with realistic distributions |
| **2. Preprocessing** | Missing values, outlier capping (IQR), label encoding, scaling |
| **3. Feature Engineering** | 12 derived features (study efficiency, risk score, etc.) |
| **4. EDA** | Correlation heatmaps, distributions, pair plots |
| **5. Model Training** | Linear Regression, Random Forest, XGBoost with RandomizedSearchCV |
| **6. Evaluation** | R², RMSE, MAE, 5-fold Cross-Validation |
| **7. Explainability** | SHAP values for global & individual explanations |
| **8. Recommendations** | Rule-based + SHAP-driven personalized suggestions |
| **9. Deployment** | Streamlit Cloud with auto-training on first run |

---

## 🤖 Models Used

| Model | R² Score | RMSE | MAE | CV R² |
|-------|----------|------|-----|-------|
| **XGBoost** | 0.93+ | ~5.2 | ~4.0 | 0.92+ |
| Random Forest | 0.91+ | ~5.8 | ~4.5 | 0.90+ |
| Linear Regression | 0.85+ | ~7.5 | ~5.8 | 0.84+ |

**Winner: XGBoost** — Selected for its superior accuracy and native SHAP support.

---

## ✨ Key Features

- **🎯 Real-time Prediction** — Instant score prediction with confidence levels
- **🔍 SHAP Explainability** — See exactly which factors drive each prediction
- **💡 Smart Recommendations** — Personalized, priority-sorted improvement strategies
- **🔄 What-If Simulator** — Experiment with changes and see impact in real-time
- **📊 Model Comparison** — Side-by-side performance of all trained models
- **📈 Interactive EDA** — Correlation heatmaps, distributions, scatter plots
- **☁️ Zero-Setup Deploy** — Auto-trains models on first Streamlit Cloud visit

---

## 🚀 Quick Start

### Local
```bash
git clone https://github.com/mohitrj18greybeard/student-intelligence-system.git
cd student-intelligence-system
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Cloud
The app is live at: **[EduPulse AI →](https://student-intelligence-system-iappex9cl99exgcqcdvcdcb.streamlit.app/)**

---

## 🔮 Future Improvements

- [ ] Integration with real university LMS data (Canvas/Moodle API)
- [ ] Deep learning model (TabNet) for improved accuracy
- [ ] Longitudinal tracking — predict performance trends over semesters
- [ ] Multi-language support
- [ ] PDF report generation for students
- [ ] Role-based authentication (JWT)

---

## 📸 Screenshots

> *Dashboard auto-generates premium visualizations with dark glassmorphism theme.*

| Prediction & SHAP | Model Comparison | EDA |
|---|---|---|
| KPI cards + SHAP bar chart + recommendations | R², RMSE comparison charts | Heatmaps, distributions, scatter plots |

---

## 📝 License

MIT License — Built with ❤️ for education.

---

**Made by [Mohit Rajput](https://github.com/mohitrj18greybeard)** | Powered by XGBoost, SHAP, and Streamlit
