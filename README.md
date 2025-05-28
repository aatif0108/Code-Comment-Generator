

```markdown
# 🧠 Code Comment Generator

Automatically generate human-like comments for Python functions using deep learning models. This project includes both an **LSTM with Attention** and a **SimpleRNN** model for comparison.

---

## 📌 Overview

This tool reads code snippets and predicts descriptive comments, trained on a labeled dataset of Python functions. It supports:

- 🧠 LSTM with attention mechanism
- ⚡ SimpleRNN baseline
- ✍️ Interactive user input for real-time predictions
- 📊 Accuracy and validation visualizations

---

## 📁 Project Files

```

.
├── dataset.txt                      # Labeled dataset of Python code and comments
├── lstm\_code.py                     # LSTM + Attention model script
├── rnn\_code.txt                     # SimpleRNN model script
├── accuracy.png                     # Training accuracy plot
├── validation\_accuracy.png          # Validation accuracy plot
├── code\_tokenizer.pkl               # Pickled tokenizer for code
├── comment\_tokenizer.pkl            # Pickled tokenizer for comments
├── max\_lengths.pkl                  # Pickled max lengths for sequences
├── code\_comment\_attention.keras     # Trained LSTM model
├── simple\_rnn\_code\_comment.keras    # Trained SimpleRNN model
└── README.md                        # Project documentation

```

---

## 📊 Dataset Format

Each sample in the dataset looks like this:

```

Code: def reverse\_string(s): return s\[::-1]

# Reverse a string

````

The script uses regex to extract code-comment pairs from `dataset.txt`.

---

## 🧠 Model Architectures

### ✅ LSTM + Attention (`lstm_code.py`)
- Encoder-Decoder with Attention
- Embedding layers
- LSTM units (256)
- Dot-product attention
- Dense output with Softmax

### ✅ SimpleRNN (`rnn_code.txt`)
- Simpler encoder-decoder setup
- Basic SimpleRNN layers
- No attention mechanism

---

## 🚀 Setup Instructions

### 🔧 1. Install Dependencies

```bash
pip install tensorflow scikit-learn matplotlib
````

### 🏋️ 2. Train a Model

```bash
# Train LSTM model
python lstm_code.py

# Or, train SimpleRNN model
python rnn_code.txt
```

* The models will be saved as `.keras` files.
* Tokenizers and max lengths will be saved as `.pkl`.

---

## 💬 Inference Example

Once training is complete, run the script and input your code:

```
Enter your Python code snippet (or type 'exit' to quit):
def add(a, b): return a + b

Generated Comment: perform + operation on two numbers
```

---

## 📈 Performance

* `accuracy.png` and `validation_accuracy.png` visualize model performance.
* LSTM with Attention shows improved generalization and accuracy compared to SimpleRNN.

---

## 🔮 Future Improvements

* Integrate Transformer-based models (e.g., GPT, BERT)
* Handle multi-line and complex code structures
* Add support for Java, C++, and other languages
* Build a web-based interactive frontend
---

## 🙏 Acknowledgments

* TensorFlow & Keras
* Scikit-learn
* Python community



