# Sentiment Analysis of Satu Sehat Mobile App Review

## 📖 **Introduction**  
This project was inspired by the increasing volume of user concerns and discussions surrounding the SatuSehat Mobile application, particularly in relation to its usability, system stability, and overall performance. To better understand public perception and common issues, a sentiment analysis was conducted on user reviews collected from the Google Play Store. This sentiment analysis specifically focuses on **Indonesian-language** reviews, providing insights into how local users perceive and respond to the application.

*SatuSehat Mobile* is a public health application developed by the Indonesian Ministry of Health. It serves as a centralized platform for accessing individual health records and national health services, including COVID-19 vaccination status, medical history, electronic prescriptions, and digital health certificates. Originally known as *PeduliLindungi*, *SatuSehat* has evolved to play a key role in Indonesia's digital health transformation, connecting users, healthcare providers, and government services through a unified system.

## 🎯 **Workflow**  
The workflow is divided into three main phases:

- **Data Collection** – Scraping real user reviews from the Google Play platform.
- **Sentiment Exploration & Visualization** – Interpreting sentiment trends and presenting them visually to uncover user sentiment patterns.
- **Sentiment Prediction** – Leveraging Machine Learning and Deep Learning techniques to classify reviews into categories such as Positive, Neutral, or Negative.

By applying Natural Language Processing (NLP), this project enables deeper insights into user experiences and provides actionable knowledge to help improve application development and user satisfaction.

## 🛠️ **Technologies Used**
This project utilizes various technologies and Python libraries to support data collection, text processing, model training, and inference:

### 🧑‍💻 Language and Platform
- Python – The primary programming language used for the project.
- Google Colab – For experimentation, visualization, and documenting processes in a collaborative environment.
### 📚 Libraries and Frameworks
- Pandas & NumPy – For data manipulation, cleaning, and numerical operations.
- Matplotlib & Seaborn –  For creating data visualizations and displaying model evaluation results.
- Scikit-learn – For model evaluation, splitting data, and additional preprocessing tasks.
- TensorFlow & Keras – Used for building and training deep learning models (LSTM).
- Google Play Scraper (google-play-scraper) – To scrape reviews of the SatuSehat app from the Google Play Store.
- Sastrawi – A library specifically designed for preprocessing Indonesian text (e.g., stopword removal, stemming).

## 📊 Machine Learning Models & Performance
I experimented with three different models for sentiment classification, each with a unique feature extraction approach and training setup. Below is a summary of the models and their respective performance metrics:

| Model                                      | Accuracy  | F1-Score | Split Ratio | Feature Type    |
|--------------------------------------------|-----------|----------|-------------|-----------------|
| **1. LSTM + Embedding (80/20)**            | **99.24%**| **99.18%**| 80/20       | Word Embedding  |
| **2. SVM + TF-IDF (80/20)**                | **98.28%**| **98.09%**| 80/20       | TF-IDF          |
| **3. Logistic Regression + TF-IDF (80/20)**| **94.78%**| **94.06%**| 80/20       | TF-IDF          |

### **Model Descriptions**
1. **LSTM + Embedding**: This model utilizes a Long Short-Term Memory (LSTM) network with word embeddings to capture the sequential nature of text. It achieved the highest accuracy and F1-score.
2. **SVM + TF-IDF**: This model uses a Support Vector Machine (SVM) classifier with TF-IDF features. It performed very well, coming close to the LSTM model in accuracy.
3. **Logistic Regression + TF-IDF**: This model uses Logistic Regression with TF-IDF features. Although it performed slightly lower than the other two models, it still provided solid results for sentiment classification.

## ⚙️ How to Run
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/FR21/SatuSehatApp_SentimentAnalysis.git
2. Install the required dependencies using requirements.txt:
   ```bash
   pip install -r requirements.txt
3. Scrape the data from Google Play Store using the scraper script.
4. Open PelatihanModel_FelixRafael.ipynb to train and evaluate the sentiment analysis models.
5. To test or classify new data, open and run Inference_FelixRafael.ipynb.

## 📁 Project Structure
```sh
Submission/
├── Dataset/
│   └── ulasan_satusehat.csv                    
├── Models/
│   ├── logreg_model.pkl                             
│   ├── lstm_sentiment_model.keras                           
│   ├── svm_model.pkl                          
│   ├── tfidf_vectorizer.pkl
│   └── tokenizer_lstm.pkl
├── Notebooks/
│   ├── Inference_FelixRafael.ipynb                           
│   ├── PelatihanModel_FelixRafael.ipynb                         
│   └── ScrapingData_FelixRafael.ipynb                                    
├── README.md
└── requirements.txt                          
```
