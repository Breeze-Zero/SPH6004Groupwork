# SPH6004 Project 2 - Group 3 🚀

## ✨ Overview

Welcome to the repository for Group 3's Project 2 for the SPH6004 module! This project focuses on MIMIC-CXR.

We leverage state-of-the-art techniques in Deep Learning, NLP, Image Analysis, etc. to build and evaluate models capable of extracting clinical insights.

## 📂 Code Structure

The project is organized as follows:

```
.
├── Code_Liu/         # Individual contribution/exploration space for Liu
├── Code_Zhang/       # Individual contribution/exploration space for Zhang
├── configs/          # Configuration files for experiments
├── dataset/          # Placeholder for datasets (or scripts to download/prepare them)
├── logs/             # General logging output
├── lightning_logs/   # Logs specific to PyTorch Lightning training runs
├── models/           # Saved model checkpoints
├── result/           # Output files, predictions, evaluation metrics
├── script/           # Utility scripts (e.g., data processing, helper functions)
├── utils/            # Utility modules and helper functions used across the project
├── .gitignore        # Specifies intentionally untracked files that Git should ignore
├── check_files.py    # Script likely used for data validation or checks
├── clinicalbert.ipynb # Jupyter notebook for exploring/using ClinicalBERT
├── eval.py           # Script for evaluating model performance
├── requirement.txt   # Project dependencies
├── test.py           # Script for running model inference/testing
├── train.py          # Main script for training models (potentially combined modalities)
├── train_img_emb.py  # Script specifically for training image embedding models
├── train_img_text_emb.py # Script specifically for training joint image-text embedding models
└── README.md         # You are here!
```

## ⚙️ Setup & Dependencies

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd SPH6004Groupwork-master
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirement.txt
    ```

## 🚀 How to Run

1.  **Data Preparation:**


2.  **Training:**
    *   To train the image embedding model:
        ```bash
        python train_img_emb.py --config [path/to/image_config.yaml] # Example
        ```
    *   To train the joint image-text model:
        ```bash
        python train_img_text_emb.py --config [path/to/joint_config.yaml] # Example
        ```
    *   To train other models (referencing `train.py`):
        ```bash
        python train.py --config [path/to/main_config.yaml] # Example
        ```
    *   Training logs will be saved in `logs/` and `lightning_logs/`.

3.  **Evaluation:**
    *   Run the evaluation script, likely pointing to a saved model checkpoint:
        ```bash
        python eval.py --checkpoint models/[your_model_checkpoint.ckpt] --data [path/to/test_data] # Example
        ```
    *   Results will be stored in the `result/` directory.

4.  **Testing/Inference:**
    *   Use `test.py` for running predictions on new data:
        ```bash
        python test.py --checkpoint models/[your_model_checkpoint.ckpt] --input [path/to/input_data] # Example
        ```



## 🧑‍💻 Team - Group 3


## 🙏 Acknowledgements

*   SPH6004 Teaching Team

---
