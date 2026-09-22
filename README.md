# Veritas AI - Fake News Detection System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen?style=for-the-badge&logo=render&logoColor=white)](https://fake-news-detection-2fpr.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![ML Engine](https://img.shields.io/badge/ML-Scikit--Learn%201.6-orange.svg)](https://scikit-learn.org/)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-99.2%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🚀 **Live Web Application**: [https://fake-news-detection-2fpr.onrender.com](https://fake-news-detection-2fpr.onrender.com/)

A full-stack, machine learning-powered misinformation intelligence platform that analyzes news articles, headlines, and claims in real-time using an ensemble of four NLP classification algorithms.

---

## Key Features

- **Multi-Model Consensus Engine**: Simultaneously evaluates content across 4 machine learning classifiers:
  1. **Logistic Regression** (97.6% accuracy)
  2. **Decision Tree** (98.9% accuracy)
  3. **Random Forest** (98.8% accuracy)
  4. **Gradient Boosting** (99.2% accuracy)
- **Instant Pre-trained Inference**: Models and vectorizer are pre-trained and serialized via `joblib`, loading in < 1 second.
- **Modern Glassmorphic Web Dashboard**: Sleek dark-mode interface with live consensus badges, confidence meters, agreement ratios, and token preview.
- **Flexible CLI Runner**: Supports one-off predictions, interactive terminal mode, and evaluation on held-out test datasets.
- **RESTful API**: Easily integrate prediction endpoints into external apps (`POST /api/predict`).
- **Production Ready**: Includes `Procfile`, `render.yaml`, `Dockerfile`, and WSGI support for zero-config cloud deployment.

---

## Model Benchmark & Evaluation

Trained and evaluated on the Reuters & Politifact dataset (44,898 total articles):

| Classifier | Model Type | Test Accuracy | Manual Test Set (20 Articles) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear margin classifier | 97.64% | 100% (20/20) |
| **Decision Tree** | Gini split tree | 98.92% | 100% (20/20) |
| **Random Forest** | 60-tree Bagging Ensemble | 98.80% | 100% (20/20) |
| **Gradient Boosting** | Sequential Boosted Trees | 99.24% | 100% (20/20) |
| **Ensemble Consensus** | Majority Vote | **99.5%** | **100% (20/20)** |

---

## Quick Start (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/Arman-Kumar01/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web App
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## CLI Usage

### Evaluate Pre-trained Models
```bash
python run_project.py
```

### Predict Any Custom News Article
```bash
python run_project.py --predict "BREAKING: NASA confirms discovery of liquid water oceans on Europa"
```

### Interactive Terminal Mode
```bash
python run_project.py --interactive
```

### Retrain Models
```bash
# Retrain on 10,000 balanced samples
python run_project.py --train --samples 10000

# Retrain on full 44,898 dataset
python run_project.py --train --full
```

---

## API Documentation

### `POST /api/predict`
Analyzes input text and returns model consensus and individual predictions.

**Request Body:**
```json
{
  "text": "Pope Francis has endorsed Donald Trump for president in shocking announcement."
}
```

**Response:**
```json
{
  "consensus": "Fake News",
  "confidence_percent": 100.0,
  "confidence_ratio": "4/4",
  "votes": { "fake": 4, "true": 0 },
  "models": {
    "LR": { "name": "Logistic Regression", "label": "Fake News", "prediction": 0 },
    "DT": { "name": "Decision Tree", "label": "Fake News", "prediction": 0 },
    "RF": { "name": "Random Forest", "label": "Fake News", "prediction": 0 },
    "GB": { "name": "Gradient Boosting", "label": "Fake News", "prediction": 0 }
  }
}
```

### `GET /api/stats`
Returns benchmark accuracies and configuration for active models.

---

## Cloud Deployment

### Option A: Render (Recommended - Free Tier)
1. Push this repository to your GitHub account.
2. Sign in to [Render](https://dashboard.render.com/).
3. Click **New +** > **Blueprint** (or **Web Service**).
4. Connect this GitHub repository.
5. Render detects `render.yaml` or `Procfile` and deploys automatically!

### Option B: Railway
1. Go to [Railway](https://railway.app/).
2. Select **Deploy from GitHub repo**.
3. Railway detects `requirements.txt` and `Procfile` and deploys automatically.

### Option C: Docker Container
```bash
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

---

## Project Structure

```
├── app.py                      # Flask web server & REST API
├── fake_news_detector.py       # Core ML engine (cleaning, vectorizer, models, inference)
├── run_project.py              # CLI runner & interactive tester
├── requirements.txt            # Python dependencies
├── Procfile                    # Cloud WSGI process definition
├── render.yaml                 # Render Blueprint configuration
├── Dockerfile                  # Container deployment build
├── saved_models/               # Serialized pre-trained models (.joblib)
│   ├── vectorizer.joblib
│   ├── model_lr.joblib
│   ├── model_dt.joblib
│   ├── model_rf.joblib
│   ├── model_gb.joblib
│   └── metrics.joblib
├── templates/
│   └── index.html              # Glassmorphic web dashboard
├── static/
│   ├── style.css               # Styling & dark-mode design system
│   └── app.js                  # Frontend async reactive logic
├── Datasets/
│   └── datasets.zip            # Original dataset archive
└── manual_testing.csv          # Held-out 20 validation articles
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
