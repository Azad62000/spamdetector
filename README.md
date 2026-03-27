# Email Spam Detection Classifier

## 🚀Live Demo: 
https://spamdetector-k40j.onrender.com

A machine learning-powered web application for classifying emails as spam or ham (non-spam) using natural language processing techniques. The system achieves high accuracy through multiple ML models and provides an intuitive web interface for real-time predictions.

## Production Summary (JSON)

```json
{
  "project_title": "Email Spam Detection – Production Ready ML API",
  "final_decision": {
    "selected_model": "Support Vector Machine (Linear SVM)",
    "reason": [
      "Best F1-score and accuracy among all evaluated models",
      "High recall ensures minimal missed spam emails",
      "Handles high-dimensional TF-IDF features effectively",
      "More robust decision boundary than Naive Bayes"
    ]
  },
  "model_comparison_summary": {
    "models_evaluated": [
      "Naive Bayes",
      "Logistic Regression",
      "Support Vector Machine",
      "Random Forest",
      "K-Nearest Neighbors"
    ],
    "evaluation_metrics": [
      "Accuracy",
      "Precision",
      "Recall",
      "F1-Score",
      "ROC-AUC"
    ],
    "best_performing_model": "Support Vector Machine"
  },
  "training_strategy": {
    "where_training_happens": "Local machine / notebook environment",
    "training_frequency": "One-time or offline retraining",
    "outputs": [
      "model.pkl",
      "vectorizer.pkl"
    ],
    "reason": "Training in production increases startup time, memory usage, and risk of deployment failure"
  },
  "data_processing_pipeline": {
    "text_cleaning_steps": [
      "Lowercasing",
      "Remove punctuation and numbers",
      "Remove HTML tags",
      "Tokenization",
      "Stopword removal",
      "Stemming or Lemmatization"
    ],
    "feature_extraction": {
      "method": "TF-IDF",
      "configuration": {
        "max_features": 3000,
        "ngram_range": [1, 2]
      }
    }
  },
  "production_architecture": {
    "api_framework": "FastAPI",
    "server": "Uvicorn",
    "api_role": "Inference only (prediction)",
    "training_code_usage": "Not imported in production"
  },
  "deployment_configuration": {
    "platform": "Render",
    "runtime": "Python",
    "start_command": "uvicorn app:app --host 0.0.0.0 --port $PORT",
    "containerization": {
      "docker_usage": "Optional (Render can auto-build)",
      "training_dependencies": "Excluded from production image"
    }
  },
  "dependencies_management": {
    "production_requirements": [
      "fastapi",
      "uvicorn",
      "scikit-learn",
      "pandas",
      "numpy",
      "nltk",
      "joblib",
      "shap",
      "lime",
      "matplotlib",
      "seaborn"
    ],
    "excluded_from_production": [
      "training-only utilities"
    ]
  },
  "common_mistakes_avoided": [
    "Training model during application startup",
    "Using gunicorn with FastAPI",
    "Using Django-style wsgi entrypoint",
    "Including heavy plotting libraries in production"
  ],
  "final_status": {
    "model": "Trained and serialized",
    "api": "Running with FastAPI + Uvicorn",
    "deployment": "Successful on Render",
    "readiness": "Production-ready"
  }
}
```

## 🚀 Features

- **Advanced Explainable AI (XAI)**: Understand *why* an email is spam with top keyword highlighting from TF-IDF weights.
- **Dynamic Threshold Tuning**: Control classification sensitivity with a tunable confidence threshold (0.0 - 1.0).
- **Zero-JS Server-Side UI**: Robust, SEO-friendly interface powered by FastAPI + Jinja2 templates (works without client-side JavaScript).
- **Real-Time Monitoring**: Integrated production logging (`app.log`) tracking confidence, labels, and request metadata.
- **Multi-Model Evaluation**: Trains and compares 5 different ML algorithms (SVM, Naive Bayes, etc.).
- **High Accuracy**: SVM model achieves 98.6% test accuracy.
- **Web Interface**: Clean, responsive UI with text input and file upload.
- **Docker Ready**: Containerized for easy deployment.

## 📊 Dataset

**Source**: SMS Spam Collection Dataset (publicly available)
- **Total Samples**: 5,572 emails
- **Spam Emails**: 747 (13.4%)
- **Ham Emails**: 4,825 (86.6%)
- **Format**: CSV with columns `v1` (label) and `v2` (text)

The dataset shows class imbalance, which is handled through appropriate evaluation metrics rather than oversampling to maintain real-world distribution.

## 🔧 Data Preprocessing Pipeline

### Step-by-Step Processing:

1. **Lowercasing**: Convert all text to lowercase for uniformity
2. **HTML Tag Removal**: Strip HTML content using regex `<[^>]+>`
3. **Punctuation & Number Removal**: Eliminate special characters and digits using regex `[^a-z\s]`
4. **Tokenization**: Split text into individual words
5. **Stopword Removal**: Remove common English stopwords using NLTK
6. **Stemming**: Reduce words to base form using Porter Stemmer

### Feature Engineering:

- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency
- **Parameters**: max_features=3000, ngram_range=(1,2)
- **Stop Words**: English stopwords removed
- **Preprocessing**: Custom clean_text function applied

## 🤖 Model Training & Evaluation

### Models Evaluated:
1. **Naive Bayes** (MultinomialNB)
   - Accuracy: 97.4%
   - Precision: 99.2%
   - Recall: 81.3%
   - F1-Score: 89.3%

2. **Logistic Regression**
   - Accuracy: 97.3%
   - Precision: 100%
   - Recall: 79.9%
   - F1-Score: 88.8%

3. **Support Vector Machine** (SVM)
   - Accuracy: 98.6%
   - Precision: 97.8%
   - Recall: 91.3%
   - F1-Score: 94.4%
   - **Selected as Best Model**

4. **Random Forest**
   - Accuracy: 97.8%
   - Precision: 100%
   - Recall: 83.9%
   - F1-Score: 91.2%

5. **K-Nearest Neighbors**
   - Accuracy: 92.0%
   - Precision: 98.4%
   - Recall: 40.9%
   - F1-Score: 57.8%

### Evaluation Metrics:

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under Receiver Operating Characteristic curve

## 🧠 Model Interpretability (SHAP & LIME)

To ensure the model's decisions are transparent and trustworthy, we integrated **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**.

### 1. SHAP (Global Interpretability)
SHAP provides a global view of which features (words) are most important across the entire dataset.
- **SHAP Summary Plot**: Shows the top 20 words that most strongly influence the model's spam vs. ham classification.
- **Insight**: Words like "free", "claim", "win", and "txt" consistently drive spam predictions, while personal greetings or common conversational words drive ham predictions.

![SHAP Summary](artifacts/shap_summary.png)

### 2. LIME (Local Interpretability)
LIME explains individual predictions by perturbing the input and seeing how the prediction changes.
- **LIME Local Explanation**: Visualizes why a specific email was classified as spam.
- **Insight**: Highlights exactly which words in a given text (e.g., "prize", "urgent") pushed the probability towards "spam".

![LIME Explanation](artifacts/lime_explanation.png)

## 🖥️ Web Application

### Backend (FastAPI):
- **Framework**: FastAPI with Uvicorn server
- **Templating**: Jinja2 for server-side HTML rendering (no-JS core functionality)
- **Monitoring**: Production logging to `app.log`
- **XAI Engine**: Custom TF-IDF based feature importance extractor
- **Endpoints**:
  - `GET /`: Serves the main page with state support
  - `POST /predict_form`: Classifies text input (Server-side UI)
  - `POST /predict_file_form`: Classifies uploaded files (Server-side UI)
  - `POST /predict`: JSON API for text classification
  - `POST /predict_file`: JSON API for file classification
  - `GET /health`: Health check endpoint

### Frontend:
- **HTML5/CSS3**: Modern responsive design with Jinja2 dynamic blocks
- **Accessibility**: Standard HTML forms for robust user interaction
- **Performance**: Zero-JS core; page reloads provide consistent state and results
- **Features**:
  - Tabbed interface (Text Input / File Upload)
  - Interactive probability visualization
  - Clickable example email cards
  - Keyword-based prediction explanations

## 🛠️ Installation & Usage

### Prerequisites:
- Python 3.8+
- pip package manager
- Web browser

### Local Setup:

1. **Clone/Download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**:
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

4. **Train the models**:
   ```bash
   python train.py
   ```
   This creates:
   - `models/` directory with all trained models
   - `artifacts/` directory with metrics and visualizations

5. **Start the web application**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   Open `http://localhost:8000` in your browser

### Docker Deployment:

1. **Build the Docker image**:
   ```bash
   docker build -t spam-detector .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 spam-detector
   ```

3. **Access**: `http://localhost:8000`

## 📈 Analysis & Results

### Key Findings:

1. **Classical ML outperforms**: Traditional algorithms like SVM perform excellently for text classification without needing transformers
2. **TF-IDF sufficiency**: Term frequency-inverse document frequency captures spam patterns effectively
3. **SVM superiority**: Best balance of precision (97.8%) and recall (91.3%) with F1-score of 94.4%
4. **No overfitting**: Test accuracy (98.6%) close to train accuracy (99.9%), indicating good generalization

### Business Impact:

- **Reduced spam leakage**: High recall minimizes missed spam emails
- **Improved user experience**: Fast, real-time classification
- **Scalable solution**: Lightweight model suitable for production deployment

### Model Performance Comparison:

```
Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC
--------------------|----------|-----------|--------|----------|--------
SVM (Best)         | 98.6%   | 97.8%    | 91.3% | 94.4%   | 98.5%
Naive Bayes        | 97.4%   | 99.2%    | 81.3% | 89.3%   | 98.5%
Logistic Regression| 97.3%   | 100%     | 79.9% | 88.8%   | 98.3%
Random Forest      | 97.8%   | 100%     | 83.9% | 91.2%   | 98.0%
KNN                | 92.0%   | 98.4%    | 40.9% | 57.8%   | 83.9%
```

## 🧠 Technical Decisions & Thinking

### Why SVM?
- **Best F1-score**: Balances precision and recall optimally
- **High accuracy**: 98.6% on test set
- **Probabilistic output**: Provides confidence scores via CalibratedClassifierCV
- **Text classification**: SVMs excel at high-dimensional sparse data like TF-IDF vectors

### Why TF-IDF over Word Embeddings?
- **Interpretability**: TF-IDF weights are understandable and explainable
- **Lightweight**: No complex neural network training required
- **Effectiveness**: Captures spam-specific keywords and phrases
- **Performance**: Fast inference suitable for real-time applications

### Why Multiple Models?
- **Comparison**: Allows evaluation of different approaches
- **Robustness**: Ensures best model selection through data-driven approach
- **Flexibility**: All models saved for potential future use

### Why FastAPI over Flask/Django?
- **Performance**: Asynchronous capabilities for high-throughput
- **Type hints**: Better code documentation and IDE support
- **Auto docs**: Built-in API documentation
- **Modern**: Python 3.6+ async/await support

### Why Separate Frontend?
- **Separation of concerns**: Backend focuses on ML, frontend on UX
- **Caching**: Static files can be cached independently
- **Scalability**: Frontend can be served by CDN or separate server
- **Technology choice**: Allows different tech stacks if needed

### Why Docker?
- **Reproducibility**: Ensures consistent environment across systems
- **Deployment**: Easy to deploy on cloud platforms (Render, Heroku, etc.)
- **Isolation**: Dependencies don't conflict with system packages

## 📁 Project Structure

```
email-spam-detector/
├── app.py                 # FastAPI application
├── train.py              # Model training script
├── utils.py              # Text preprocessing utilities
├── index.html            # Main web page
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── spam.csv              # Dataset
├── static/
│   ├── styles.css        # CSS styling
│   └── script.js         # Frontend JavaScript
├── models/               # Trained ML models
├── artifacts/            # Metrics and visualizations
└── README.md             # This file
```

## 🔄 API Usage

### API JSON Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your email text here", "threshold": 0.5}'
```

### Form Post Request (Web Interface Mode):
```bash
curl -X POST "http://localhost:8000/predict_form" \
     -d "text=Your email text here&threshold=0.5"
```

### File Upload Classification:
```bash
curl -X POST "http://localhost:8000/predict_file" \
     -F "file=@email.txt"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using Python, FastAPI, Scikit-learn, and modern web technologies.**
