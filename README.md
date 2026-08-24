<div align="center">

# 🚀 MLPilot

### **Data In → Insights Out**

A production-grade **Python Machine Learning Library** for tabular datasets that automatically performs EDA, preprocessing, model comparison, hyperparameter tuning, explainability, and exports a deployment-ready inference pipeline.

**14+ Models · Auto EDA · Optuna · SHAP · CLI · Python API**

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-4.0+-7B68EE?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B35?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**One command. One pipeline. Production-ready models.**

</div>

---

# ⚡ Quick Start

Install MLPilot:

```bash
pip install mlpilotx
```

Run your first ML pipeline:

```bash
mlpilot run --data examples/USA_Housing.csv
```

Specify the target column manually:

```bash
mlpilot run --data examples/USA_Housing.csv --target price
```

---

# 🔥 What MLPilot Does

Instead of writing hundreds of lines of boilerplate code, MLPilot automatically performs:

| Stage | Description |
|--------|-------------|
| 📂 Load | CSV loading & validation |
| 📊 EDA | Statistics, missing values & correlations |
| 🧹 Clean | Duplicates, sparse columns & outlier handling |
| ⚙️ Feature Engineering | Encoding, scaling & transformations |
| ✂️ Split | Train / Test split |
| 🔧 Preprocess | Scikit-learn preprocessing pipeline |
| 🏆 Compare | 14+ Machine Learning models |
| 🎯 Tune | Bayesian optimization with Optuna |
| 🔍 Explain | SHAP feature importance & visualizations |
| 📦 Export | Deployment-ready `.joblib` pipeline |

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

- 📂 Automatic CSV support
- 🤖 Automatic Regression & Classification detection
- 📊 Smart Exploratory Data Analysis
- 🧹 Missing value & outlier handling
- ⚙️ Feature engineering pipeline
- 🏆 Cross-validation model leaderboard
- 🎯 Bayesian hyperparameter optimization
- 🔍 SHAP explainability visualizations
- 📦 Export complete inference pipeline
- 💻 Rich CLI & Python API support

---

# 💻 CLI Usage

### Automatic Training

```bash
mlpilot run --data examples/heart.csv
```

### Regression

```bash
mlpilot run --data examples/USA_Housing.csv --target price
```

### Classification

```bash
mlpilot run --data examples/patient_adherence_dataset.csv --target adherence
```

### Custom Output Folder

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
    data_path="examples/USA_Housing.csv",
    target="price"
)

print(context.best_model_name)
print(context.metrics)
```

---

# 📁 Example Datasets

MLPilot includes ready-to-use datasets inside the `examples/` folder.

| Dataset | Task |
|----------|------|
| `USA_Housing.csv` | Regression |
| `insurance.csv` | Regression |
| `heart.csv` | Classification |
| `patient_adherence_dataset.csv` | Classification |
| `Student_performance_data.csv` | Classification |
| `Food_Delivery_Times.csv` | Regression |
| `Exam_Score_Prediction.csv` | Regression |
| `taxi_trip_pricing.csv` | Regression |
| `personality_synthetic_dataset.csv` | Classification |

Example:

```bash
mlpilot run --data examples/insurance.csv --target charges
```

---

# 📦 Generated Artifacts

Every successful run generates:

```text
mlpilot_artifacts/
├── mlpilot_pipeline.joblib
├── leaderboard.csv
├── metrics.json
├── model_comparison.json
├── feature_importances.json
├── eda_report.json
├── run_metadata.json
├── serving_schema.json
├── predict_snippet.py
├── DEPLOY.md
└── shap/
    ├── shap_summary.png
    ├── shap_dependence_*.png
    └── shap_waterfall.png
```

---

# 🧠 Supported Models

| Category | Models |
|----------|--------|
| **Linear** | Linear Regression, Ridge, Lasso, ElasticNet |
| **Tree** | Decision Tree, Random Forest, Extra Trees |
| **Boosting** | Gradient Boosting, HistGradientBoosting |
| **Instance** | KNN, SVR |
| **Neural** | MLP |
| **Classification** | Logistic Regression, SGD, Linear SVC, Passive Aggressive |

---

# 🧪 Edge Case Testing

MLPilot includes dedicated validation datasets.

```text
tests/
└── edge_cases/
    ├── empty.csv
    ├── one_row.csv
    ├── all_null.csv
    ├── duplicate_col.csv
    ├── target_missing.csv
    ├── only_numeric.csv
    └── only_categorical.csv
```

Run an edge-case test:

```bash
mlpilot run --data tests/edge_cases/empty.csv
```

---

# 📂 Project Structure

```text
MLPilot/
├── configs/
│   └── default.yaml
├── examples/
│   ├── USA_Housing.csv
│   ├── insurance.csv
│   ├── heart.csv
│   └── ...
├── src/
│   └── ml_pilot/
│       ├── cli.py
│       ├── config/
│       ├── core/
│       ├── stages/
│       └── utils/
├── tests/
│   ├── edge_cases/
│   ├── test_load.py
│   ├── test_pipeline_smoke.py
│   └── ...
├── LICENSE
├── README.md
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