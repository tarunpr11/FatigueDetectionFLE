# Federated Learning for Privacy-Preserving Fatigue Detection

This repository contains the implementation of a privacy-preserving fatigue detection system using Federated Learning (FL). The project demonstrates the feasibility of training a complex, multi-modal deep learning model on distributed data without centralizing sensitive user information.

## 📜 Problem Statement

Traditional machine learning systems for detecting human fatigue require centralizing vast amounts of sensitive, personal health data from wearable sensors. This approach creates significant **privacy risks**, security vulnerabilities, and challenges with data regulations (like GDPR), making it difficult to implement these life-saving systems at scale. This project addresses the critical need for a fatigue detection system that is both highly effective and fundamentally private.

## 🎯 Project Scope

This project designs, builds, and evaluates a **privacy-preserving fatigue detection system** using Federated Learning. The scope includes:
1.  **Developing a multi-modal deep learning model** that can predict fatigue by intelligently fusing data from multiple wearable sensors.
2.  **Integrating advanced techniques**, including LSTMs, attention mechanisms, and domain adaptation, to improve accuracy and generalization across different individuals.
3.  **Implementing the system within a Federated Learning framework** to train a robust global model without raw data ever leaving the local device.
4.  **Simulating and evaluating the system's performance** on a real-world dataset, measuring its final accuracy on a held-out test set.

## 🏗️ Architecture Overview

The core of this project is a Federated Multi-Modal Attention Architecture.

1.  **Multi-Modal Feature Extraction:** The model processes five sensor modalities (HR, IBI, ACC, EDA, Temp). Each modality has a dedicated **LSTM** network to learn temporal patterns and produce a feature embedding.
2.  **Cross-Modal Attention:** An attention mechanism intelligently fuses the embeddings from all modalities, creating a single, rich feature vector that represents the most salient information.
3.  **Domain Adaptation:** A **Gradient Reversal Layer (GRL)** is used in an adversarial scheme to force the model to learn person-agnostic features, enhancing its ability to generalize to new, unseen users.
4.  **Federated Learning:** The entire model is trained using the **Flower** framework. Clients train the model on their local private data, and only the model weight updates are sent to a central server for aggregation via **Federated Averaging (FedAvg)**.

## 📂 Repository Structure
.
├── Centralized.ipynb        # Notebook for the baseline (non-private) model.
├── FLE_Implementation.ipynb   # Notebook with the full Federated Learning simulation.
└── README.md                # This readme file.

## 🚀 Getting Started

This project is designed to be run in a Google Colab environment.

### Prerequisites

You will need the following libraries:
- Python 3.10+
- PyTorch
- Flower (`flwr`)
- Pandas
- Scikit-learn

All required libraries are installed by the notebooks themselves.

### Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/tarunpr11/FatigueDetectionFLE.git
    ```
2.  **Dataset:** This project uses the `final_feature_label_dataset_normalized.csv` from the FatigueSet dataset. Due to its size and privacy, it is not included in this repository.
    * Download the dataset - https://drive.google.com/drive/folders/1FUdlk9iPSbzP7X7csZw5TSx8sPo9Ecbx?usp=sharing
    * In your Google Drive, create the folder structure: `My Drive/Fatigue_Set/`.
    * Upload the `final_feature_label_dataset_normalized.csv` file to this folder.
3.  **Mount Google Drive:** When you open the notebooks in Google Colab, you will be prompted to mount your Google Drive. Please authorize it to allow the notebooks to access the dataset.

### Running the Notebooks

1.  **`Centralized.ipynb` (Baseline Model)**
    * Open the notebook in Google Colab.
    * Ensure your Google Drive is mounted and the dataset is in the correct path.
    * The code for centralized model with and without the novelty is in the bottom of the notebook

2.  **`FLE_Implementation.ipynb` (Federated Model)**
    * This is the main notebook for the project.
    * Open it in Google Colab and ensure your Drive is mounted.
    * Run all code cells under FL with novelty. It will automatically:
        1.  Prepare and split the data for all clients and the test set.
        2.  Define the server and client logic.
        3.  Run the full 15-round federated learning simulation.
        4.  Report the final, conclusive accuracy on the held-out test set.

## 📊 Results

The performance of four different modeling approaches was evaluated on the held-out test set. The models include a traditional **Centralized** approach and our privacy-preserving **Federated** approach, both with and without the novel **Domain Adaptation** feature. The results for predicting Physical and Mental fatigue are detailed below.

| Metric         | Centralized (with Novelty) | Centralized (without Novelty) | **Federated (with Novelty)** | Federated (without Novelty) |
| :------------- | :------------------------: | :---------------------------: | :--------------------------: | :-------------------------: |
| **Physical Fatigue**                                                                                                                     |
| RMSE           |           0.1128           |            0.1777             |          **0.1318**          |           0.1889            |
| MAE            |           0.0878           |            0.1306             |          **0.1002**          |           0.1492            |
| R² Score       |           0.6082           |            0.0286             |          **0.6257**          |           0.2319            |
| **Mental Fatigue**                                                                                                                       | 
| RMSE           |           0.1671           |            0.1678             |          **0.1410**          |           0.2079            |
| MAE            |           0.1474           |            0.1571             |          **0.1062**          |           0.1662            |
| R² Score       |           0.1043           |            0.0973             |          **0.6530**          |           0.2467            |



## 🔭 Future Work

Future work will focus on advancing this model's real-world applicability by enhancing its generalization, efficiency, and privacy. Key priorities include exploring more sophisticated domain adaptation techniques, investigating model compression for on-device feasibility, and integrating stronger privacy guarantees like differential privacy. Validating the framework on real-world, on-device datasets will be a critical next step to prove its practical effectiveness.
