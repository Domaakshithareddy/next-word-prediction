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

### Code:
```python
# Preprocessing Function
def preprocess_text(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        text = file.read()
    text = text.replace('\n', ' ').replace('\r', '').replace('\ufeff', '')
    return text
```

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

### Code:
```python
# Training Setup
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)

# Model Training
history = model.fit(X, y, epochs=150, batch_size=128, validation_split=0.2, callbacks=[reduce_lr, early_stopping])
```

---

## 4. Interactive Prediction
Once trained, the model can predict the next three words based on user input.

### Code:
```python
# Interactive Testing
print("\nTesting the model:")
while True:
    text = input("Enter your line (or 'stop' to exit): ").strip()
    if text.lower() == "stop":
        print("Exiting...")
        break
    try:
        predict_next_words(model, tokenizer, text, top_k=3, temperature=0.8, num_words=3)
    except Exception as e:
        print(f"Error: {e}")
```

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
### Train the Model
```bash
python train.py
```
### Test the Model
```bash
python test.py
```

---

## 7. Future Improvements
- Integration with **pre-trained word embeddings** (GloVe, Word2Vec) for better contextual predictions.
- Fine-tuning hyperparameters using **Bayesian Optimization**.
- Deploying as a **web service** using Flask or FastAPI.

---

## 8. Contributors
If you would like to contribute to this project, feel free to open a pull request!

---

## 9. License
This project is released under the MIT License.
