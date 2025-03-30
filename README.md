## Next Word Prediction Model

This repository contains an LSTM-based Next Word Prediction model that uses deep learning techniques to predict the next word(s) in a sequence. The model has been optimized for better accuracy and efficiency.

---

### Features
- **Data Preprocessing**: Text cleaning, tokenization, and sequence generation.
- **Bidirectional LSTM Model**: Uses an advanced architecture for context-aware predictions.
- **Hyperparameter Optimization**: Improved embedding size, learning rate adjustments, and early stopping.
- **Interactive Testing**: Real-time prediction of the next three words.

---

## 1. Data Preprocessing
Before training the model, the text is preprocessed to ensure it is clean and structured for learning.

### Steps:
1. **Read and clean the text**: Remove unwanted characters while keeping punctuation.
2. **Tokenization**: Convert text into sequences of integers.
3. **N-gram sequence creation**: Use the last 3 words to predict the next one.
4. **One-hot encoding**: Convert target outputs into categorical format.

---

## 2. Model Architecture
The model is built using a **Bidirectional LSTM** to capture both past and future context. The architecture consists of:
- **Embedding Layer**: Converts words into dense vectors of fixed size.
- **Two Bidirectional LSTMs**: Enhances learning with forward and backward information flow.
- **Dense Layers**: Extracts relevant features and predicts the next word.

### Code:
```python
# Model Definition
model = Sequential([
    Embedding(vocab_size, 50, input_length=sequence_length),
    Bidirectional(LSTM(500, return_sequences=True)),
    Bidirectional(LSTM(500)),
    Dense(500, activation="relu"),
    Dense(vocab_size, activation="softmax")
])
```

---

## 3. Training the Model
The model is compiled using **categorical cross-entropy loss** and optimized using the Adam optimizer. To enhance training efficiency:
- **ReduceLROnPlateau**: Lowers learning rate if loss stops improving.
- **EarlyStopping**: Stops training if no progress is detected.

---

## 4. Interactive Prediction
Once trained, the model can predict the next three words based on user input.

---

## 5. Model Saving and Deployment
After training, the model is saved for later use.

### Code:
```python
# Save Model and Tokenizer
model.save("nextword_improved.keras")
pickle.dump(tokenizer, open('tokenizer_improved.pkl', 'wb'))
```

---

## 6. How to Run
### Install Dependencies
```bash
pip install numpy tensorflow keras
```
---

## 7. Contributors
If you would like to contribute to this project, feel free to open a pull request!

