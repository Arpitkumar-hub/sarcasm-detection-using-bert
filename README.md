# Sarcasm Detection in News Headlines using BERT

A Natural Language Processing (NLP) project that detects sarcasm in news headlines using a fine-tuned BERT (bert-base-uncased) model.

**Binary NLP Classifier** — Detects whether a news headline is Sarcastic or Genuine.

**Accuracy:** 93%+

**F1 Score:** 0.93+

---

## Overview

Sarcasm Detection in News Headlines is a Natural Language Processing (NLP) project that uses a fine-tuned BERT (bert-base-uncased) model to classify news headlines as Sarcastic or Genuine. The model understands contextual relationships between words and achieves high classification performance on the Kaggle Sarcasm Headlines Dataset.

The project includes a complete machine learning pipeline consisting of data preprocessing, model training, evaluation, inference, and a Gradio-based web interface for real-time predictions.

---

## Features

* Fine-tuned BERT (bert-base-uncased) model
* Binary text classification
* Sarcastic vs Genuine headline prediction
* Real-time prediction interface using Gradio
* Confidence score generation
* High-performance NLP model
* End-to-end machine learning pipeline

---

## Dataset

**Dataset:** News Headlines Dataset for Sarcasm Detection (Kaggle)

The dataset contains news headlines labeled as:

* 0 → Genuine
* 1 → Sarcastic

### Preprocessing Steps

* Text cleaning
* Lowercasing
* Removing unwanted characters
* Tokenization using BERT Tokenizer
* Train-validation-test split

---

## Tech Stack

### Programming Language

* Python

### Deep Learning & NLP

* PyTorch
* Hugging Face Transformers
* BERT (bert-base-uncased)

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Deployment & Interface

* Gradio

### Version Control

* Git
* GitHub

---

## Project Workflow

Dataset

↓

Text Preprocessing

↓

BERT Tokenization

↓

Fine-Tuned BERT Model

↓

Binary Classification

↓

Prediction Generation

↓

Gradio Interface

↓

Final Output

---

## Model Performance

| Metric   | Score |
| -------- | ----- |
| Accuracy | 93%+  |
| F1 Score | 0.93+ |

---

## Project Structure

sarcasm_detector/

├── preprocess.py        # Text preprocessing pipeline

├── dataset.py           # Dataset loading and DataLoader creation

├── model.py             # BERT classification model

├── train.py             # Model training script

├── evaluate.py          # Model evaluation script

├── inference.py         # Prediction and inference logic

├── app.py               # Gradio web interface

├── notebook.py          # Experimental notebook code

├── requirements.txt     # Project dependencies

├── README.md            # Project documentation

├── Sarcasm_Headlines_Dataset.json

├── Sarcasm_Headlines_Dataset_v2.json

└── saved_model/         # Fine-tuned BERT model files

---

## Running the Project

### Install Dependencies

pip install -r requirements.txt

### Train Model

python train.py

### Evaluate Model

python evaluate.py

### Launch Web Interface

python app.py --model_dir ./saved_model

---

## Example Prediction

### Input

Scientists Discover Correlation Between Ice Cream Sales and Drowning

### Output

Prediction: Sarcastic

Confidence: 96%

---

## Skills Demonstrated

* Natural Language Processing (NLP)
* Deep Learning
* Transformer Models
* BERT Fine-Tuning
* Text Classification
* Model Evaluation
* Data Preprocessing
* PyTorch
* Gradio Deployment
* Git & GitHub

---

## Future Scope

* Multilingual sarcasm detection
* Social media sentiment analysis
* Real-time API deployment
* Mobile application integration
* Advanced explainable AI techniques

---

## Author

Arpit Kumar

B.Tech Computer Science Engineering

COER University
