# Fake-News-Detection
 This project builds a Fake News Detection system using NLP and Machine Learning. News articles are preprocessed using stopword removal and stemming, then converted into numerical features using TF-IDF. A Logistic Regression model is trained to classify articles as Real (0) or Fake (1), achieving accurate predictions on test data.
Fake news detection is an important problem in the digital era. This project builds a supervised machine learning model that classifies news articles as:

0 → Real News

1 → Fake News

The system processes textual data, extracts meaningful features, and trains a Logistic Regression classifier to make predictions.

📂 Dataset Description

The dataset contains the following features:

Column Name	Description
id	Unique identifier for each article
title	Title of the news article
author	Author of the article
text	Full article content (may be incomplete)
label	Target variable (0 = Real, 1 = Fake)
🛠️ Tech Stack

Python

NumPy

Pandas

NLTK

Scikit-learn

Regex (re)

🔄 Project Workflow
1️⃣ Data Preprocessing

Loaded dataset using Pandas

Handled missing values

Combined title and author into a new feature (content)

Removed special characters and numbers

Converted text to lowercase

Removed English stopwords using NLTK

Applied Porter Stemming

2️⃣ Feature Engineering

Used TF-IDF (Term Frequency – Inverse Document Frequency) to convert text data into numerical vectors.

TF: Measures word frequency

IDF: Reduces importance of common words

3️⃣ Train-Test Split

80% Training Data

20% Testing Data

Stratified split to maintain label balance

4️⃣ Model Training

Trained a Logistic Regression classifier on extracted features.

5️⃣ Model Evaluation

Training Accuracy

Testing Accuracy

Built a predictive system for new samples

📊 Model Performance

The model is evaluated using:

Accuracy Score (Training Data)

Accuracy Score (Testing Data)

You can further improve evaluation using:

Confusion Matrix

Precision & Recall

F1 Score

🚀 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

2️⃣ Install dependencies
pip install numpy pandas nltk scikit-learn

3️⃣ Download NLTK stopwords
import nltk
nltk.download('stopwords')

4️⃣ Add Dataset

Place train.csv inside the project directory.

5️⃣ Run the script
python main.py

🧪 Example Prediction
X_new = X_test[3]
prediction = model.predict(X_new)

if prediction[0] == 0:
    print("The news is real")
else:
    print("The news is fake")

📁 Project Structure
fake-news-detection/
│
├── train.csv
├── main.py
├── README.md
└── requirements.txt

🔮 Future Improvements

Implement advanced models (Random Forest, XGBoost)

Use Deep Learning models (LSTM, BERT)

Deploy as a web application (Flask / Streamlit)

Add real-time news API integration

Improve preprocessing using Lemmatization

📚 Learning Outcomes

Text preprocessing using NLP

Feature extraction using TF-IDF

Binary classification using Logistic Regression

Model evaluation techniques

Building a predictive ML system

🤝 Contributing

Contributions are welcome!
Feel free to fork this repository and submit a pull request.

📜 License

This project is open-source and available under the MIT License.
