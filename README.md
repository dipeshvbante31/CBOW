# CBOW Word Embedding with PyTorch

A simple **Continuous Bag of Words (CBOW)** implementation built from scratch using **PyTorch**.

This project demonstrates how word embeddings can be learned from text using the CBOW approach. The model learns word representations by using surrounding context words to predict a target word.

## 📌 Project Overview

**CBOW (Continuous Bag of Words)** is one of the architectures used in Word2Vec.

The basic idea is:

```text
Context Words
     ↓
Word Embeddings
     ↓
Average Embeddings
     ↓
Linear Layer
     ↓
Predict Target Word
```

For example:

```text
Context:  [i, love, learning, is]
                    ↓
                CBOW Model
                    ↓
              Target: machine
```

The model learns the relationships between words during training.

---

## 🚀 Features

* Text preprocessing and tokenization
* Automatic vocabulary creation
* Word-to-index and index-to-word mapping
* CBOW training data generation
* Word embedding layer using PyTorch
* Context embedding averaging
* Target word prediction
* Cross-entropy loss
* Adam optimizer
* Backpropagation
* 50-dimensional word embeddings
* Cosine similarity for finding similar words
* 1000 training epochs

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Neural Networks
* Natural Language Processing (NLP)
* Word Embeddings
* CBOW / Word2Vec concept
* Cosine Similarity

---

## 📂 Project Structure

```text
CBOW-Word-Embedding/
│
├── cbow.py
├── README.md
└── requirements.txt
```

> Replace `cbow.py` with the actual name of your Python file if it is different.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/CBOW-Word-Embedding.git
```

### 2. Navigate to the project

```bash
cd CBOW-Word-Embedding
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
so
```

