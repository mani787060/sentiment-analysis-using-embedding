# Sentiment Analysis Using Embedding

## Overview

This project demonstrates the fundamentals of **Sentiment Analysis using Word Embeddings** with a small custom text dataset.

The main focus is understanding how words represented as integer IDs can be transformed into **dense vector representations using an Embedding layer**. These learned representations allow neural networks to capture useful relationships between words instead of treating each word as only an independent integer.

---

## Objective

The objectives of this project are to:

* Understand the concept of word embeddings.
* Convert text data into numerical sequences.
* Learn how the Keras `Embedding` layer works.
* Understand how words are represented as dense vectors.
* Prepare textual data for neural network-based sentiment analysis.
* Understand the difference between integer encoding and embeddings.

---

## Dataset

The project uses a small custom collection of text documents:

```text
go india
india india
hip hip hurray
jeetega bhai jeetega india jeetega
bharat mata ki jai
kohli kohli
sachin sachin
dhoni dhoni
modi ji ki jai
inquilab zindabad
```

These short sentences are used to demonstrate how textual data can be converted into numerical representations and then processed through an embedding layer.

---

## What Is Word Embedding?

An embedding represents a word as a **dense numerical vector** instead of simply assigning it an integer ID.

For example:

```text
Word
 ↓
Integer Encoding
 ↓
Embedding Layer
 ↓
Dense Vector
```

Instead of treating:

```text
india → 5
kohli → 8
```

as meaningful numerical values, the embedding layer learns vector representations for these words.

For example, conceptually:

```text
india → [0.21, -0.14, 0.63, ...]
kohli → [0.45,  0.08, -0.31, ...]
```

The actual vectors are learned during model training.

---

## Integer Encoding vs Embedding

Integer encoding and embedding are related but different steps.

### Integer Encoding

Converts words into integer IDs:

```text
word → integer
```

Example:

```text
india → 5
```

The number `5` does not contain semantic information.

### Embedding

Converts those integer IDs into learned dense vectors:

```text
integer ID → dense vector
```

The embedding layer learns representations that can capture useful patterns from the data.

---

## How the Embedding Layer Works

The general workflow is:

```text
Raw Text
   ↓
Tokenization
   ↓
Integer Encoding
   ↓
Integer Sequences
   ↓
Embedding Layer
   ↓
Dense Word Vectors
   ↓
Neural Network
   ↓
Sentiment Prediction
```

The `Embedding` layer maintains a trainable matrix where each integer ID corresponds to a vector.

During training, these vectors are updated so that the model can learn useful representations from the available data.

---

## Key Concepts Covered

### 1. NLP

Natural Language Processing allows machine learning models to work with human language.

### 2. Tokenization

Text is divided into smaller units called tokens, such as words.

```text
"go india"
    ↓
["go", "india"]
```

### 3. Integer Encoding

Each word is assigned an integer ID so that the text can be represented numerically.

### 4. Word Embedding

Integer IDs are mapped to dense vectors using an embedding layer.

### 5. Embedding Dimension

The embedding dimension determines the number of values used to represent each word.

For example:

```text
Embedding dimension = 8

word → [v1, v2, v3, v4, v5, v6, v7, v8]
```

### 6. Trainable Representations

Embedding vectors are learned and updated during training rather than manually assigned.

---

## Why Use Embeddings?

Integer IDs alone do not express relationships between words.

For example:

```text
india → 3
kohli → 7
```

There is no meaningful relationship between `3` and `7`.

An embedding layer instead learns a vector representation for each word, allowing the model to learn useful patterns from the training data.

This makes embeddings an important concept in NLP and deep learning.

---

## Implementation Concept

A Keras embedding layer can be created using:

```python
from keras.layers import Embedding

Embedding(
    input_dim=vocabulary_size,
    output_dim=embedding_dimension
)
```

Where:

* `input_dim` → size of the vocabulary
* `output_dim` → number of dimensions in each word vector

The exact values depend on the implementation in the notebook.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* Natural Language Processing
* Word Embeddings
* Deep Learning

---

## Key Learnings

Through this project, I learned:

* How text is converted into numerical sequences.
* Why integer encoding alone is not enough to represent semantic information.
* What word embeddings are.
* How the Keras `Embedding` layer works.
* How embedding vectors are learned during training.
* The difference between integer encoding and dense vector representations.
* Why embeddings are important for NLP and sentiment-analysis models.

---

## Future Improvements

This project can be extended by:

* Adding a complete sentiment classification model.
* Using a larger text dataset such as IMDB.
* Adding sequence padding.
* Combining embeddings with SimpleRNN.
* Experimenting with LSTM and GRU.
* Visualizing learned word embeddings.
* Comparing different embedding dimensions.
* Comparing trainable embeddings with pre-trained embeddings.

---

## Conclusion

This project provides a practical introduction to **word embeddings for NLP**.

It builds upon the basic idea of integer encoding by showing how integer IDs can be transformed into meaningful dense vector representations using an Embedding layer. Understanding this concept provides an important foundation for building more advanced NLP systems using **RNNs, LSTMs, GRUs, Transformers, and modern language models**.
