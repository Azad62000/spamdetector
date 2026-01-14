# Email Spam Detection Classifier

A machine learning-powered web application for classifying emails as spam or ham (non-spam) using natural language processing techniques. The system achieves high accuracy through multiple ML models and provides an intuitive web interface for real-time predictions.

## 🚀 Features

- **Multi-Model Evaluation**: Trains and compares 5 different ML algorithms
- **High Accuracy**: SVM model achieves 98.6% test accuracy
- **Web Interface**: Clean, responsive UI with text input and file upload
- **Real-time Predictions**: Instant classification with probability visualization
- **Docker Ready**: Containerized for easy deployment
- **Comprehensive Analysis**: Detailed metrics and visualizations

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

## 🖥️ Web Application

### Backend (FastAPI):
- **Framework**: FastAPI with Uvicorn server
- **Endpoints**:
  - `GET /`: Serves the main HTML page
  - `POST /predict`: Classifies text input
  - `POST /predict_file`: Classifies uploaded files
  - `GET /health`: Health check endpoint

### Frontend:
- **HTML5**: Semantic markup with accessibility features
- **CSS3**: Modern styling with gradients, animations, and responsive design
- **JavaScript**: AJAX requests for real-time predictions
- **Features**:
  - Tabbed interface (Text Input / File Upload)
  - Dynamic probability visualization
  - Example email cards for testing
  - Mobile-responsive design

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

### Text Classification:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your email text here"}'
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
