<div align="center">

# 🚀 MLPilot

### **Data In → Insights Out**

A production-grade **Python AutoML library** for tabular datasets that automatically performs EDA, preprocessing, model comparison, hyperparameter tuning, explainability, and exports a deployment-ready inference pipeline.

**14+ Models · Auto EDA · Optuna · SHAP · CLI · Python API**

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-4.0+-7B68EE?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B35?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**One command. One pipeline. Production-ready models.**

</div>

---

# ⚡ Quick Start

Install MLPilot:

```bash
pip install mlpilot
```

Run your first ML pipeline:

```bash
mlpilot run --data dataset.csv
```

Specify the target column manually:

```bash
mlpilot run --data dataset.csv --target price --task regression
```

---

# 🔥 What MLPilot Does

Instead of writing hundreds of lines of boilerplate code, MLPilot automatically performs:

| Stage | Description |
|--------|-------------|
| 📂 Load | CSV & Parquet loading |
| 📊 EDA | Statistics, missing values & correlations |
| 🧹 Clean | Duplicates, sparse columns & outliers |
| ⚙️ Feature Engineering | Encoding, scaling & transformations |
| ✂️ Split | Train / Test split |
| 🔧 Preprocess | ColumnTransformer pipeline |
| 🏆 Compare | 14+ machine learning models |
| 🎯 Tune | Bayesian optimization with Optuna |
| 🔍 Explain | SHAP feature importance & plots |
| 📦 Export | Complete `.joblib` inference pipeline |

---

# 🔄 Pipeline Workflow

```text
                 Raw Dataset
                      │
                      ▼
              📂 Load Dataset
                      │
                      ▼
            📊 Statistical EDA
                      │
                      ▼
              🧹 Data Cleaning
                      │
                      ▼
         ⚙️ Feature Engineering
                      │
                      ▼
           ✂️ Train / Test Split
                      │
                      ▼
        🔧 Preprocessing Pipeline
                      │
                      ▼
        🏆 Compare 14+ ML Models
                      │
                      ▼
     🎯 Optuna Hyperparameter Tuning
                      │
                      ▼
      🔍 SHAP Explainability Report
                      │
                      ▼
          📈 Model Evaluation
                      │
                      ▼
   📦 Export Production Pipeline (.joblib)
```

---

# ✨ Features

- 📂 Automatic CSV & Parquet support
- 🤖 Regression & Classification detection
- 📊 Smart Exploratory Data Analysis
- 🧹 Missing value & outlier handling
- ⚙️ Feature engineering pipeline
- 🏆 Cross-validation model leaderboard
- 🎯 Bayesian hyperparameter optimization
- 🔍 SHAP explainability visualizations
- 📦 Export complete inference pipeline
- 💻 CLI & Python API support

---

# 💻 CLI Usage

### Train automatically

```bash
mlpilot run --data housing.csv
```

### Regression

```bash
mlpilot run --data housing.csv --target price --task regression
```

### Classification

```bash
mlpilot run --data churn.csv --target exited --task classification
```

### Custom output folder

```bash
mlpilot run --data data.csv --output outputs/
```

---

# 🐍 Python API

```python
from ml_pilot import PipelineRunner
from ml_pilot.config import load_config

config = load_config()

runner = PipelineRunner(config)

context = runner.run(
    data_path="housing.csv",
    target="price"
)

print(context.best_model_name)
print(context.metrics)
```

---

# 📁 Generated Artifacts

Every successful run generates:

```text
outputs/
├── mlpilot_pipeline.joblib
├── leaderboard.csv
├── metrics.json
├── model_comparison.json
├── feature_importances.json
├── run_metadata.json
├── serving_schema.json
├── predict_snippet.py
└── shap/
    ├── shap_summary.png
    ├── shap_dependence.png
    └── shap_waterfall.png
```

---

# 🧠 Supported Models

| Category | Models |
|----------|--------|
| **Linear** | Ridge, Lasso, ElasticNet, Linear Regression |
| **Tree** | Decision Tree, Random Forest, Extra Trees |
| **Boosting** | Gradient Boosting, HistGradientBoosting |
| **Instance** | KNN, SVR |
| **Neural** | MLP Regressor / Classifier |

---

# 📂 Project Structure

```text
mlpilot/
├── src/
│   └── ml_pilot/
│       ├── cli.py
│       ├── config/
│       ├── core/
│       ├── stages/
│       └── utils/
├── tests/
├── examples/
├── README.md
├── LICENSE
└── pyproject.toml
```

---

# 🛠 Tech Stack

- **Python 3.12+**
- **Scikit-learn**
- **Pandas & NumPy**
- **Optuna**
- **SHAP**
- **Typer + Rich**
- **Joblib**
- **Plotly**

---

# 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

### ⭐ If MLPilot helps your workflow, consider starring the repository!

Built with ❤️ by **Aditya Sharma**

</div>
