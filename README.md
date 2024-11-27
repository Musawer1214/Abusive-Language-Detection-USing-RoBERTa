# Abusive Language Detection Using RoBERTa

This repository contains code for training and deploying an abusive language detection model using the [RoBERTa](https://huggingface.co/roberta-base) transformer model. The model identifies abusive or toxic comments in English text and is designed to be deployed on [Hugging Face Spaces](https://huggingface.co/spaces) using a Gradio interface.

![Repository Structure](image.png)

## Table of Contents

- [Project Overview](#project-overview)
- [Setup](#setup)
- [Training](#training)
- [Deployment](#deployment)
- [Files and Directory Structure](#files-and-directory-structure)
- [License](#license)

## Project Overview

This project uses a pre-trained RoBERTa model fine-tuned on a dataset of abusive language comments. The goal is to classify comments as abusive or non-abusive. The training is implemented in a Jupyter notebook, and deployment is handled by a Flask-based API in `app.py` to integrate with Hugging Face Spaces.

### Dataset

The dataset used for training includes text labeled for various types of toxicity, including:
- Toxic
- Severe toxic
- Obscene
- Threat
- Insult
- Identity hate

The dataset is cleaned and preprocessed before being used for fine-tuning the RoBERTa model.
The dataset can be found on https://www.kaggle.com/datasets/musawerhussain/abusive-language-detection-in-speech

## Setup

### Prerequisites

- Python 3.7 or higher
- `pip` for installing dependencies

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Musawer1214/Abusive-Language-Detection-USing-RoBERTa.git
   cd Abusive-Language-Detection-USing-RoBERTa
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

The training process is outlined in the `abusive-language-detection.ipynb` notebook. It includes:

1. Loading and preprocessing the dataset.
2. Splitting the dataset into training and validation sets.
3. Fine-tuning the RoBERTa model on the abusive language detection dataset.
4. Evaluating the model's performance on the validation set.

### Running the Notebook

To run the training notebook:

1. Open `abusive-language-detection.ipynb` in Jupyter or any compatible environment.
2. Execute the cells sequentially to preprocess the data, fine-tune the model, and evaluate it.

## Deployment

The model is deployed using a Flask API defined in `app.py`. This script uses Gradio to create a web-based interface that allows users to input text and get predictions.

### Running the Deployment Script

1. Ensure the model is saved as `model.safetensors`.
2. Run `app.py` using:
   ```bash
   python app.py
   ```
3. Access the web interface through the provided link to test the model.

## Files and Directory Structure

- `app.py`: Flask app for deploying the model on Hugging Face Spaces.
- `abusive-language-detection.ipynb`: Jupyter notebook for training the model.
- `config.json`: Configuration file for model parameters.
- `model.safetensors`: The fine-tuned model in safe tensor format.
- `requirements.txt`: List of Python dependencies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
