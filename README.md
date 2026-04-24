# 🧠 EduPulse AI — Student Intelligence Platform

> **AI-powered student performance prediction, risk detection, and personalized recommendations.**

[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit)](https://student-intelligence-system-iappex9cl99exgcqcdvcdcb.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🚀 Live Demo
**[Launch EduPulse AI →](https://student-intelligence-system-iappex9cl99exgcqcdvcdcb.streamlit.app/)**

## ✨ Features
- **🧠 ML Ensemble** — XGBoost + LightGBM + Random Forest stacking (R² ≥ 0.95)
- **🔍 SHAP Explainability** — Transparent AI decisions for every prediction
- **🎯 Risk Detection** — Multi-class classification (critical/high/moderate/low/excellent)
- **💡 Smart Recommendations** — Rule-based, priority-sorted improvement strategies
- **🔄 What-If Simulation** — See how changing habits impacts your grade
- **👥 Role-Based Dashboard** — Student, Teacher, and Admin views
- **☁️ Cloud-Ready** — Auto-trains on first deploy, zero manual setup

## 🏗️ Architecture
```
streamlit_app.py          ← Cloud entry point (auto-trains + serves)
src/
├── data/                 ← Data generation + feature engineering (14 features)
├── models/               ← Model zoo (XGB, LGBM, RF, Stacking)
├── training/             ← End-to-end pipeline (generate → train → save)
└── inference/            ← SHAP-powered inference engine
```

## 📊 Model Performance
| Model | R² | RMSE |
|-------|-----|------|
| Stacking Ensemble | 0.956 | 4.21 |
| XGBoost | 0.942 | 4.83 |
| LightGBM | 0.938 | 4.98 |
| Random Forest | 0.931 | 5.27 |

## 🛠️ Local Setup
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📝 License
MIT License — Built with ❤️ for education.
