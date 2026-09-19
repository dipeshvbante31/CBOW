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
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📋 Requirements

Create a `requirements.txt` file containing:

```text
torch
```

You can install it directly with:

```bash
pip install torch
```

---

## ▶️ How to Run

Run the Python file:

```bash
python cbow.py
```

The program will:

1. Create the vocabulary.
2. Generate CBOW training examples.
3. Create the neural network.
4. Train the model for 1000 epochs.
5. Display the learned embedding for `learning`.
6. Find words similar to `learning`.

---

## 🧠 How CBOW Works

CBOW predicts a **target word from its surrounding context words**.

For example:

```text
I love machine learning
```

If `machine` is the target word:

```text
Context Words:
love
learning

        ↓

      CBOW

        ↓

Target:
machine
```

The model repeatedly performs this process during training.

---

## 🔢 Model Architecture

The model contains two main layers.

### 1. Embedding Layer

```python
self.embeddings = nn.Embedding(
    vocab_size,
    embedding_dim
)
```

This converts word indexes into numerical vectors.

The project uses:

```python
embedding_dim = 50
```

Therefore, each word is represented by a **50-dimensional vector**.

---

### 2. Linear Layer

```python
self.linear = nn.Linear(
    embedding_dim,
    vocab_size
)
```

The averaged context embedding is passed through the linear layer to predict the target word.

---

## ⚙️ Training Configuration

| Parameter           |            Value |
| ------------------- | ---------------: |
| Embedding Dimension |               50 |
| Window Size         |                2 |
| Epochs              |             1000 |
| Learning Rate       |             0.01 |
| Optimizer           |             Adam |
| Loss Function       | CrossEntropyLoss |

---

## 📊 Training Process

For every training example:

```text
Context words
     ↓
Convert words → indexes
     ↓
Embedding layer
     ↓
Get word vectors
     ↓
Average vectors
     ↓
Linear layer
     ↓
Predict target
     ↓
Calculate loss
     ↓
Backpropagation
     ↓
Update weights
```

The process is repeated for **1000 epochs**.

The loss is printed every 100 epochs.

Example:

```text
Epoch 100/1000, Loss: ...
Epoch 200/1000, Loss: ...
Epoch 300/1000, Loss: ...
...
Epoch 1000/1000, Loss: ...
```

---

## 🔍 Getting Word Embeddings

The project extracts the learned embedding for:

```text
learning
```

using:

```python
learning_vector = model.embeddings.weight[
    learning_index
].detach()
```

The result is a vector containing **50 numerical values**.

Example format:

```text
tensor([
    0.12,
   -0.45,
    0.78,
    ...
])
```

The exact values will vary between runs because neural-network weights are randomly initialized.

---

## 📐 Finding Similar Words

The project uses **cosine similarity** to compare word vectors.

```python
torch.cosine_similarity(...)
```

Cosine similarity measures how similar the directions of two vectors are.

Conceptually:

```text
Similarity ≈ 1
      ↓
Very similar vector direction

Similarity ≈ 0
      ↓
Less related

Similarity ≈ -1
      ↓
Opposite direction
```

The program compares `learning` with the other words in the vocabulary and returns the top 5 most similar words.

---

## ⚠️ Important Limitation

This project uses a **very small training dataset**.

```text
I love natural language processing
I love machine learning
...
```

Because the dataset is small, the resulting embeddings are mainly useful for **learning and demonstrating the CBOW algorithm**.

They should not be considered high-quality general-purpose word embeddings.

For real NLP applications, Word2Vec models are usually trained on much larger corpora.

---

## 🎯 Learning Objectives

This project helps demonstrate:

* What word embeddings are
* How CBOW works
* How Word2Vec-style training works
* How an embedding layer works
* How context words can predict a target word
* How neural networks learn embeddings
* How backpropagation updates embeddings
* How cosine similarity can compare word vectors

---

## 🔮 Future Improvements

Possible improvements include:

* Train on a much larger text corpus
* Add text preprocessing
* Remove stop words
* Add punctuation handling
* Implement CBOW with negative sampling
* Implement Skip-gram
* Visualize embeddings using PCA
* Visualize embeddings using t-SNE
* Compare CBOW with Gensim Word2Vec
* Save and load trained embeddings
* Build a small NLP similarity application

---

## 📚 Concept

The project is based on the **Word2Vec CBOW architecture**, where surrounding words are used to predict a target word.

```text
          Context Words
        /      |       \
       /       |        \
      i       love     learning
       \       |        /
        \      |       /
         ↓     ↓      ↓
       Embedding Layer
              ↓
       Average Vectors
              ↓
         Linear Layer
              ↓
       Target Prediction
```

---

## 👨‍💻 Author

**Dipesh Bante**

Computer Science Engineering
Interested in:

* Data Science
* Machine Learning
* Artificial Intelligence
* Natural Language Processing
* Python

---

## ⭐ If You Found This Useful

Give the repository a ⭐ on GitHub and feel free to explore the implementation.
