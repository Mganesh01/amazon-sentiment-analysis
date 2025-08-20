# amazon-sentiment-analysis

# 📝 Sentiment Analysis on Amazon Product Reviews

## 📖 Project Overview
This project applies **Natural Language Processing (NLP)** techniques to analyze customer reviews from the [Amazon Product Reviews dataset](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews).  
The goal is to classify reviews into **Positive, Negative, or Neutral sentiment**, helping businesses better understand customer feedback and improve decision-making.

---

## 🎯 Objectives
- Preprocess raw customer review text (cleaning, tokenization, stopword removal).
- Perform **Exploratory Data Analysis (EDA)** to understand review distribution.
- Apply **TF-IDF vectorization** to transform text into numerical features.
- Train ML models (Logistic Regression, Random Forest, Naïve Bayes).
- Evaluate models using accuracy, precision, recall, and F1-score.
- Visualize insights such as most common positive/negative words.

---

## 📂 Dataset
- **Source:** [Amazon Product Reviews on Kaggle](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)
- Contains:
  - `reviews` → Customer review text  
  - `sentiment` → Sentiment label (`Positive`, `Negative`, `Neutral`)  
  - Additional metadata (ratings, product info, etc.)

---

## 🛠️ Tech Stack
- **Python**
- **Libraries:**
  - `pandas`, `numpy` → Data handling
  - `matplotlib`, `seaborn`, `wordcloud` → Visualization
  - `scikit-learn` → ML models & evaluation
  - `nltk` / `spacy` → NLP preprocessing

---

## 📊 Workflow
1. **Data Preprocessing**
   - Remove punctuation, numbers, and special characters.
   - Convert text to lowercase.
   - Remove stopwords.
   - Tokenize and lemmatize words.

2. **Exploratory Data Analysis (EDA)**
   - Sentiment distribution (positive/negative/neutral).
   - Word clouds for most frequent terms by sentiment.

3. **Feature Engineering**
   - Convert text into numerical form using **TF-IDF**.

4. **Modeling**
   - Train baseline models: Logistic Regression, Naïve Bayes, Random Forest.
   - Evaluate performance metrics.

5. **Evaluation**
   - Compare models on accuracy, precision, recall, F1-score.
   - Confusion matrix visualization.

---

## 📈 Expected Results
- Identify the **overall sentiment distribution** of Amazon reviews.
- Achieve good accuracy using **Logistic Regression with TF-IDF**.
- Highlight key words influencing positive/negative sentiment.
- Provide insights into customer opinions for businesses.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/amazon-sentiment-analysis.git
   cd amazon-sentiment-analysis
