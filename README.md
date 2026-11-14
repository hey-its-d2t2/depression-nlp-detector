# Depression NLP Detector

**Early Detection of Depression from Text Using NLP and Machine Learning**

---

**Author:** Deepak Singh
**Email:** todeepaksingh.com@gmail.com
**Copyright:** © 2026 Deepak Singh. All rights reserved.

> This project is developed for portfolio and research demonstration purposes. Unauthorized commercial use, redistribution, or modification without explicit permission from the author is not permitted.

---

> **Medical Disclaimer:** This tool is for research and educational purposes only. It is NOT a substitute for professional mental health evaluation. If you or someone you know is struggling, please seek help from a qualified mental health professional.

---

## Table of Contents

1. [About This Project](#about-this-project)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Libraries and Techniques Used](#libraries-and-techniques-used)
5. [Installation Guide](#installation-guide)
6. [API Keys and URL Configuration](#api-keys-and-url-configuration)
7. [How to Use](#how-to-use)
8. [API Reference](#api-reference)
9. [Running Tests](#running-tests)
10. [Training Your Own Model](#training-your-own-model)
11. [Troubleshooting](#troubleshooting)

---

## About This Project

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze text and identify linguistic patterns associated with depression. It processes written language (social media posts, diary entries, text messages) and predicts whether the content shows signs of depressive language.

### What it can do

- Analyze a single piece of text and return a **depression risk score**
- Process **batches of text** (up to 100 at once)
- Provide **confidence scores** so you know how certain the model is
- Serve predictions through a **REST API** (web service)
- Fully scriptable from the command line

### Why it matters

Depression affects 1 in 5 people globally. Many people express distress in writing long before seeking help. NLP tools that analyze language patterns can assist:
- Mental health researchers studying linguistic markers of depression
- Early warning systems that flag high-risk content for human review
- Educational demonstrations of AI in healthcare

---

## How It Works

The system uses a **two-channel feature approach** — combining word patterns with psychological language features:

### Channel 1: TF-IDF Text Features
Converts text into numerical vectors based on which words and character sequences appear. Words like "hopeless", "worthless", "empty" score high in depression prediction.

- **Word n-grams (1 to 3 words):** Captures phrases like "feel hopeless", "can't sleep"
- **Character n-grams (3 to 5 characters):** Catches word fragments like "worth" in "worthless"

### Channel 2: Psychological Features (LIWC-inspired)
Extracts 21 hand-crafted features based on clinical research:

| Feature | Why It Matters |
|---------|---------------|
| First-person pronoun ratio ("I", "me", "my") | Depressed people use more self-referential language |
| Negative emotion word density | Hopeless, worthless, empty, broken |
| Cognitive distortion markers | "Always", "never", "nothing", "everyone" (black-and-white thinking) |
| Death-related word count | Direct and indirect references |
| Social process word density | Lower = more isolated |
| VADER sentiment scores | Automated positive/negative/neutral sentiment |
| Negation density | "can't", "won't", "not" |
| Past vs future tense ratio | Depression correlates with past-focused language |

### Classification
Both feature channels are combined and fed to a **Logistic Regression** classifier (trained with class-balanced weighting). The model outputs a probability score between 0 and 1.

---

## Project Structure

```
depression-nlp-detector/
|
+-- src/depression_detector/        # Main source package
|   +-- api/                        # REST API (FastAPI)
|   |   +-- main.py                 # App factory + startup
|   |   +-- routes/
|   |   |   +-- predict.py          # /predict and /predict/batch endpoints
|   |   |   +-- health.py           # /health and /info endpoints
|   |   +-- schemas/                # Request/response data models
|   +-- data/                       # Data handling
|   |   +-- dataset.py              # Dataset class (load, split, generate synthetic data)
|   |   +-- preprocessing.py        # Text cleaning pipeline
|   |   +-- data_loader.py          # Load from CSV, HuggingFace, JSON, Parquet
|   +-- features/                   # Feature engineering
|   |   +-- text_features.py        # TF-IDF extractor
|   |   +-- psychological_features.py  # 21 clinical linguistic features
|   +-- models/                     # ML models
|   |   +-- traditional/
|   |   |   +-- sklearn_classifier.py  # Logistic Regression / SVM / Random Forest
|   |   +-- deep_learning/
|   |       +-- bert_classifier.py  # BERT fine-tuning (optional, requires GPU)
|   +-- training/
|   |   +-- trainer.py              # Training loop, cross-validation, oversampling
|   +-- evaluation/
|   |   +-- metrics.py              # Precision, Recall, F1, AUC-ROC, threshold sweep
|   +-- utils/
|       +-- config.py               # Configuration management
|       +-- logger.py               # Structured logging (loguru)
|
+-- scripts/                        # Command-line tools
|   +-- train.py                    # Train a model
|   +-- evaluate.py                 # Evaluate a saved model
|   +-- predict.py                  # Make predictions from CLI
|
+-- tests/                          # Test suite (43 tests)
|   +-- unit/                       # Unit tests for each module
|   +-- integration/                # API endpoint tests
|
+-- models/saved/                   # Trained model files (.pkl)
+-- data/raw/                       # Raw training data (CSV)
+-- reports/                        # Evaluation reports (JSON)
|
+-- pyproject.toml                  # Package configuration (uv / pip)
+-- requirements.txt                # Python dependencies
+-- README.md                       # This file
```

---

## Libraries and Techniques Used

### Core ML / NLP
| Library | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | 1.3+ | TF-IDF vectorizer, Logistic Regression, SVM, Random Forest, model evaluation |
| **numpy** | 1.24+ | Numerical arrays and matrix operations |
| **nltk** | 3.8.x | Tokenization, lemmatization, stopword removal |
| **vaderSentiment** | 3.3+ | Rule-based sentiment analysis (great for social media text) |
| **imbalanced-learn** | 0.11+ | Random oversampling to handle class imbalance |

### Data Handling
| Library | Version | Purpose |
|---------|---------|---------|
| **polars** | 1.0+ | Fast DataFrame library for reading/writing CSV/Parquet files |
| **joblib** | 1.3+ | Saving and loading trained model files |

### API / Web Service
| Library | Version | Purpose |
|---------|---------|---------|
| **FastAPI** | 0.104+ | High-performance REST API framework |
| **uvicorn** | 0.24+ | ASGI web server to run the API |
| **pydantic** | 2.5+ | Data validation and serialization |
| **pydantic-settings** | 2.1+ | Configuration from environment variables |
| **httpx** | 0.25+ | HTTP client for API integration tests |

### Logging and Visualization
| Library | Version | Purpose |
|---------|---------|---------|
| **loguru** | 0.7+ | Beautiful structured logging |
| **matplotlib** | 3.8+ | Plotting confusion matrices and ROC curves |
| **seaborn** | 0.13+ | Statistical visualization |

### Package Management
- **uv** — Fast Python package installer and virtual environment manager (10-100x faster than pip)

### ML Techniques Applied
- **TF-IDF with word and character n-grams** — Converts text to numerical features
- **LIWC-inspired psychological feature extraction** — Clinical linguistic signals based on peer-reviewed research
- **Feature fusion** — Combines sparse TF-IDF matrix with dense psychological features (scipy sparse + dense)
- **Stratified train/validation/test split** — Preserves class distribution across all splits
- **5-fold cross-validation** — Robust model evaluation, reduces overfitting risk
- **Class-balanced training** — Handles unbalanced datasets fairly (`class_weight="balanced"`)
- **Threshold sweep** — Find the optimal classification threshold for precision/recall trade-off
- **VADER sentiment analysis** — Domain-appropriate sentiment for informal/social media text

---

## Installation Guide

### Prerequisites

- **Python 3.10 or higher** (tested on Python 3.15)
- **Windows 10/11**, macOS, or Linux
- At least **2 GB free disk space**
- Internet connection (for first-time package download)

### Step 1: Install uv (package manager)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, close and reopen your terminal.

### Step 2: Navigate to the project folder

```bash
cd depression-nlp-detector
```

### Step 3: Create a virtual environment and install all dependencies

```bash
uv venv
uv pip install -e .
```

This automatically:
- Creates an isolated `.venv` folder inside the project
- Installs all required Python libraries (listed in `pyproject.toml`)
- Makes the `depression_detector` package importable

### Step 4: (First run only) Download NLTK language data

**Windows:**
```powershell
.venv\Scripts\python.exe -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

**macOS / Linux:**
```bash
.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

> NLTK downloads happen automatically on first use if you skip this step.

---

## API Keys and URL Configuration

This section is **important if you are deploying this project on a server** or integrating it into another application. By default everything runs locally on your own machine — no keys or external URLs are needed just to try it out.

---

### 1. HuggingFace API Token (optional — only for private datasets)

Required only if you want to load a **private** HuggingFace dataset during training.

**How to get your token:**
1. Create a free account at [https://huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens → New token**
3. Copy your token (starts with `hf_...`)

**How to set it:**

Windows:
```powershell
set HF_TOKEN=hf_your_token_here
```

macOS / Linux:
```bash
export HF_TOKEN=hf_your_token_here
```

Then run training as normal:
```bash
python scripts/train.py --hf-dataset "username/private-dataset"
```

> Public HuggingFace datasets do NOT require a token.

---

### 2. Changing the API Server URL (for deployment or sharing with others)

By default the API listens on `http://127.0.0.1:8000` — this is **only accessible from your own machine**.

**To make it accessible to other people on your network or the internet:**

```bash
# Allow connections from any device on your local network
uvicorn depression_detector.api.main:app --host 0.0.0.0 --port 8000

# Use a custom port (e.g. if 8000 is taken)
uvicorn depression_detector.api.main:app --host 0.0.0.0 --port 9000
```

Once running with `--host 0.0.0.0`, other users can reach the API at:
```
http://<your-machine-ip>:8000/api/v1/predict
```

To find your machine's IP address:
- **Windows:** run `ipconfig` in a terminal, look for "IPv4 Address"
- **macOS / Linux:** run `ifconfig` or `ip addr`

---

### 3. Deploying to a Cloud Server (AWS / Azure / Google Cloud / Railway / Render)

When hosting on a cloud server, update the base URL in any client applications from:
```
http://127.0.0.1:8000
```
to your server's public address, for example:
```
https://depression-api.yourserver.com
```

**Example `.env` file for a deployed setup** (create this file in the project root):
```env
# Server binding
APP_HOST=0.0.0.0
APP_PORT=8000

# Optional: HuggingFace token for private datasets
HF_TOKEN=hf_your_token_here

# Optional: set log level (DEBUG, INFO, WARNING)
APP__LOG_LEVEL=INFO
```

Load it before starting the server:

Windows:
```powershell
Get-Content .env | ForEach-Object { if ($_ -match "^(\w+)=(.+)$") { [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }
uvicorn depression_detector.api.main:app --host 0.0.0.0 --port 8000
```

macOS / Linux:
```bash
export $(cat .env | grep -v "#" | xargs)
uvicorn depression_detector.api.main:app --host 0.0.0.0 --port 8000
```

---

### 4. CORS (Cross-Origin Resource Sharing) — for web app integration

If you are calling the API from a web browser or a frontend app hosted on a different domain, you need to enable CORS. Edit `src/depression_detector/api/main.py` and add your frontend's URL to the `allow_origins` list:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-app.com", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

For quick testing only (not recommended for production):
```python
allow_origins=["*"]  # allows ALL origins
```

---

### Quick Reference: What needs a key vs what does not

| Feature | Needs API Key? | Notes |
|---------|---------------|-------|
| Local predictions (CLI) | No | Works fully offline |
| Local API server | No | Runs on your own machine |
| Public HuggingFace datasets | No | Free, no login needed |
| Private HuggingFace datasets | Yes | `HF_TOKEN` environment variable |
| Sharing API with others on LAN | No | Change `--host` to `0.0.0.0` |
| Cloud deployment | No key, but need a server | Update the URL to your server's address |

---

## How to Use

### Step 1: Train the Model (Required before first use)

**Windows:**
```powershell
.venv\Scripts\python.exe scripts\train.py
```

**macOS / Linux:**
```bash
.venv/bin/python scripts/train.py
```

This generates 4000 synthetic training samples, trains a Logistic Regression model, runs 5-fold cross-validation, and saves the model to `models/saved/`.

**Expected output:**
```
TRAINING COMPLETE
CV F1:  0.9996 +/- 0.0007
TEST F1: 1.0000
Model saved to models\saved\traditional_logistic_regression.pkl
```

---

### Option A: Command Line Prediction

**Single text prediction:**

Windows:
```powershell
.venv\Scripts\python.exe scripts\predict.py --text "I feel so hopeless and empty every day."
```

macOS / Linux:
```bash
.venv/bin/python scripts/predict.py --text "I feel so hopeless and empty every day."
```

**Output:**
```
[DEPRESSED] confidence=0.996 | I feel so hopeless and empty every day.
```

**Predict from a text file** (one entry per line):
```bash
python scripts/predict.py --file my_texts.txt
```

**Adjust sensitivity** (lower threshold = more sensitive, more false positives):
```bash
python scripts/predict.py --text "I feel empty inside." --threshold 0.4
```

---

### Option B: REST API (Web Service)

**Start the API server:**

Windows:
```powershell
.venv\Scripts\python.exe -m uvicorn depression_detector.api.main:app --host 127.0.0.1 --port 8000
```

macOS / Linux:
```bash
.venv/bin/python -m uvicorn depression_detector.api.main:app --host 127.0.0.1 --port 8000
```

**Interactive docs (Swagger UI):**
Open your browser and go to: **http://127.0.0.1:8000/docs**

You will see a visual interface where you can test all API endpoints by clicking "Try it out."

---

## API Reference

### Health Check
```
GET http://127.0.0.1:8000/health
```
Response: `{"status": "healthy", "timestamp": "2026-11-14T10:00:00"}`

---

### Single Text Prediction
```
POST http://127.0.0.1:8000/api/v1/predict
Content-Type: application/json

{
  "text": "I feel so hopeless every day. Nothing matters anymore."
}
```

**Response:**
```json
{
  "label": 1,
  "label_text": "depressed",
  "confidence": 0.996,
  "probabilities": {
    "non_depressed": 0.004,
    "depressed": 0.996
  },
  "threshold_used": 0.5,
  "model_name": "traditional_logistic_regression",
  "disclaimer": "This tool is for research purposes only and is NOT a substitute for professional mental health evaluation."
}
```

---

### Batch Prediction (up to 100 texts at once)
```
POST http://127.0.0.1:8000/api/v1/predict/batch
Content-Type: application/json

{
  "texts": [
    "I feel hopeless and empty.",
    "Had a great day with friends!"
  ]
}
```

**Response:**
```json
{
  "results": [
    {"label_text": "depressed", "confidence": 0.994, ...},
    {"label_text": "non-depressed", "confidence": 0.999, ...}
  ],
  "total": 2,
  "model_name": "traditional_logistic_regression"
}
```

---

### Model Information
```
GET http://127.0.0.1:8000/info
```

---

## Understanding the Results

| Output | Meaning |
|--------|---------|
| `label: 0` / `non-depressed` | Text shows no strong signs of depressive language |
| `label: 1` / `depressed` | Text contains linguistic patterns associated with depression |
| `confidence: 0.9+` | Model is very confident in this prediction |
| `confidence: 0.5 to 0.7` | Model is uncertain — treat with caution |

**Important:** A "depressed" label means the **text contains depressive language patterns**, not that the person is clinically depressed. Always defer to trained mental health professionals for diagnosis.

---

## Running Tests

Run the full test suite (43 tests: unit tests + API integration tests):

**Windows:**
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

**macOS / Linux:**
```bash
.venv/bin/python -m pytest tests/ -v
```

**Expected result:** `43 passed`

---

## Training Your Own Model

### Using Synthetic Data (default, no external data needed)

```bash
# Train with more samples for better coverage
python scripts/train.py --n-samples 4000

# Use a different classifier
python scripts/train.py --classifier svm
python scripts/train.py --classifier random_forest
python scripts/train.py --classifier ensemble  # combines LR + SVM + RF

# Skip cross-validation (faster training)
python scripts/train.py --no-cv
```

### Using Your Own CSV Data

Prepare a CSV file with two columns:
- `text` — the written content to analyze
- `label` — 0 for non-depressed, 1 for depressed

```bash
python scripts/train.py --data path/to/your/data.csv
```

### Using a HuggingFace Dataset

```bash
python scripts/train.py --hf-dataset "USERNAME/depression-dataset"
```
> First install: `uv pip install datasets`

### Evaluate a Trained Model

```bash
python scripts/evaluate.py --model models/saved/traditional_logistic_regression.pkl --data path/to/test.csv
```

---

## Troubleshooting

### "No model found" error when starting the API
The model file is missing. Run training first:
```bash
python scripts/train.py
```

### "NLTK resource not found" error
Download NLTK data manually:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Installation fails on Windows with DLL errors
Use binary-only installation (skips C extension compilation):
```bash
uv pip install -e . --no-build
```

### Model predicts incorrectly on natural text
The default model is trained on synthetic data. For better real-world accuracy, train with real datasets from sources like:
- Reddit mental health communities (available on HuggingFace as `ShreyaR/DepressionDetection`)
- CLPsych shared task datasets

### Port 8000 already in use
Start the server on a different port:
```bash
uvicorn depression_detector.api.main:app --port 8001
```

---

## For Developers: Advanced Configuration

All settings can be overridden via environment variables using double-underscore notation:

```bash
# Windows
set APP__DATA__TEST_SIZE=0.15
set APP__MODEL__USE_BERT=false

# macOS / Linux
export APP__DATA__TEST_SIZE=0.15
export APP__MODEL__USE_BERT=false
```

See `src/depression_detector/utils/config.py` for all available settings.

---

---

## Author

**Deepak Singh**
Email: todeepaksingh.com@gmail.com

© 2026 Deepak Singh. All rights reserved.
Built with Python, FastAPI, scikit-learn, and NLTK. Package management with [uv](https://docs.astral.sh/uv/).
