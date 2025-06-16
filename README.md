# Next Word Prediction using Enhanced Markov Chains

## Project Overview

This project implements a robust **Next Word Prediction** system using an **Enhanced Markov Chain model**. The system is designed to predict the most probable next words given a seed input by analyzing a corpus of text data. It leverages probabilistic language modeling and beam search techniques to enhance prediction accuracy and robustness.

## Key Features

* **Multi-order Markov Chains:** Supports configurable Markov chain orders (default is 3), capturing varying levels of context.
* **Smoothing Techniques:** Applies additive (Laplace) smoothing to handle unseen word transitions and avoid zero probabilities.
* **Beam Search Decoding:** Utilizes beam search with adjustable width to generate more coherent and higher-probability word sequences.
* **Custom Preprocessing:** Includes text normalization to handle noisy input files and unify formatting.
* **Scalability Handling:** Designed to issue warnings for large datasets and prevent memory-related crashes.
* **Fallback Mechanisms:** Introduces graceful fallbacks with uniform probability sampling when input states are not found in training data.

## Input Requirements

* A set of `.txt` files containing natural language text data.
* Cleaned text with consistent spacing and encoding (handled by preprocessing pipeline).

## Output

* For a given input phrase (seed text), the system predicts the next word(s) based on learned probabilities.
* Supports flexible word generation lengths.

## Usage Flow

1. Input text files are preprocessed and concatenated into a single dataset.
2. An enhanced Markov Chain model is trained on this dataset.
3. The user provides a seed phrase, and the model returns the most likely next words.
4. A series of test seed sentences are evaluated to showcase the model’s performance and generalization.
