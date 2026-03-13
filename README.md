# IMDB Sentiment Analysis

This project performs binary sentiment classification on IMDB movie reviews using a Simple RNN model built with TensorFlow/Keras, and serves predictions through a Streamlit web app.

## Project Overview

The repository includes:

- Notebook-based experimentation and training
- A trained model for immediate inference
- A Streamlit UI for interactive review classification

The target output is one of two classes:

- Positive
- Negative

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- Jupyter Notebook

## Model Summary

Based on the training notebook, the model pipeline is:

- Dataset: `tensorflow.keras.datasets.imdb`
- Vocabulary size: `10000`
- Max sequence length: `500`
- Architecture:
  - `Embedding(10000, 128)`
  - `SimpleRNN(128, activation='relu')`
  - `Dense(1, activation='sigmoid')`
- Compilation:
  - Optimizer: `adam`
  - Loss: `binary_crossentropy`
  - Metric: `accuracy`
- Training defaults:
  - Epochs: `10`
  - Batch size: `32`
  - Validation split: `0.2`
  - Callback: `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`

## Repository Structure

```text
.
|-- app.py
|-- Embeddings.ipynb
|-- Prediction.ipynb
|-- SimpleRNN.ipynb
|-- requirements.txt
|-- simple_rnn_imdb.keras
`-- simple_rnn_imdb.h5
```

## Setup

### 1. Create a Virtual Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

After startup, open the local URL printed in the terminal (usually `http://localhost:8501`).

## How Inference Works

1. Input review text is lowercased and split into words.
2. Words are mapped to IMDB word indices (unknown words mapped to fallback token).
3. Sequence is padded to length `500`.
4. The trained model predicts a score in `[0, 1]`.
5. Score > `0.5` is labeled Positive; otherwise Negative.

## Notebook Workflow

- `Embeddings.ipynb`: introductory embedding and tokenization experiments
- `SimpleRNN.ipynb`: dataset loading, preprocessing, training, and model save
- `Prediction.ipynb`: model loading and standalone prediction function

Recommended order:

1. Run `Embeddings.ipynb` to understand text-to-vector concepts.
2. Run `SimpleRNN.ipynb` to train or retrain the classifier.
3. Run `Prediction.ipynb` for notebook-based inference checks.
4. Launch `app.py` for interactive usage.

## Troubleshooting

- Model load error (`No file or directory found`): ensure `simple_rnn_imdb.keras` is in the project root.
- TensorFlow install issues on latest Python: use Python `3.10` or `3.11` and reinstall dependencies.
- Streamlit port conflict: run `streamlit run app.py --server.port 8502`.

## Next Improvements

- Add automated tests for preprocessing and prediction logic.
- Add model evaluation metrics and confusion matrix output.
- Add reproducible training configuration (seed, config file, and experiment tracking).
