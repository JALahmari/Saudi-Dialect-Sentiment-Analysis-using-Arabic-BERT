# Saudi Dialect Sentiment Analysis using Arabic BERT

## Project Overview

This project implements a sentiment analysis system for **Saudi dialect tweets** using both classical machine learning and transformer-based models. The goal is to classify tweets as **positive or negative** while addressing the linguistic challenges of dialectal Arabic commonly used on social media.

The project compares a traditional **TF-IDF + Logistic Regression** model with a **fine-tuned Arabic BERT transformer model** to evaluate which approach performs better on Saudi dialect text.

## Dataset

The project uses the **SS2030 Saudi Sentiment Dataset**, which contains approximately **4,000 Saudi dialect tweets** collected from Twitter.

Dataset characteristics:

* Total tweets: ~4,000
* Positive tweets: ~2,500
* Negative tweets: ~1,500
* Average tweet length: 14–16 words
* Language: Saudi Arabic dialect

Each tweet is labeled as:

* **1 → Positive**
* **0 → Negative**

## Preprocessing

Since Twitter data is noisy, several preprocessing steps were applied:

* Remove URLs
* Remove user mentions (@username)
* Remove hashtags (#topic)
* Remove non-Arabic characters
* Normalize whitespace

Example:

Before:
RT @user123: حقوق المرأة 💚💚💚 https://t.co/xyz123

After:
حقوق المرأة

## Model Development

### 1. Logistic Regression (Baseline)

A classical machine learning pipeline was built using:

* **TF-IDF vectorization**
* **Logistic Regression classifier**

This model serves as a baseline to evaluate the effectiveness of traditional NLP techniques on dialectal Arabic.

### 2. Arabic BERT (Transformer Model)

The main model used in this project is a fine-tuned **Arabic BERT model (asafaya/bert-base-arabic)**.

Training configuration:

* Batch size: 16
* Learning rate: 2e-5
* Epochs: 2
* Sequence length: 128
* Optimizer: AdamW

Training was performed on **Kaggle GPU (T4)**.

## Results

Two models were evaluated on the test dataset.

| Model               | Accuracy | Macro F1 |
| ------------------- | -------- | -------- |
| Logistic Regression | 82.02%   | 0.81     |
| Arabic BERT         | 85.54%   | 0.85     |

The **Arabic BERT model outperformed the traditional machine learning approach**, demonstrating the advantage of transformer-based models in capturing contextual and dialectal patterns in Saudi Arabic tweets.

## Technologies Used

* Python
* PyTorch
* Transformers (HuggingFace)
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* TF-IDF Vectorization


## Future Work

Possible improvements for this project include:

* Adding a **neutral sentiment class**
* Testing more advanced Arabic models such as **MARBERT or AraBERTv2**
* Expanding the dataset with more Saudi dialect sources
* Applying **explainability techniques (SHAP, LIME)** to analyze model decisions
* Studying differences between Saudi dialect regions (Najdi, Hijazi, Gulf)

## Author

**Joud Alahmari**
Artificial Intelligence Student
Imam Abdulrahman Bin Faisal University

LinkedIn:
https://www.linkedin.com/in/joud-alahmari-52607130a

GitHub:
https://github.com/JALahmari
