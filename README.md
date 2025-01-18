
# Skip-Gram Model

This repository contains an implementation of the Skip-Gram model for learning word embeddings using PyTorch. The code also evaluates the learned embeddings on the WordSim-353 dataset and provides an optional visualization of the embeddings.

----------

## Installation
    
1. Run from the root to install dependences:
    ```bash
    pip install -r requirements.txt
    ```
    This installs all necessary dependencies to run the scripts.
    

----------

## Usage

### Training the Model

To train the Skip-Gram model, run the `main.py` script:

```bash
python main.py
```

-   Ensure that the `data` folder contains the `text8.zip` before running `main.py`.
-   The training script will save the model and vocabulary in the `data` folder by default.

### Evaluating the Model

To evaluate the trained embeddings using the WordSim-353 dataset, run the `evaluation.py` script:

```bash
python evaluation.py
```

-   Make sure the following files are in the `data` folder before evaluation:
    -   Trained model file (e.g., `skipgram_model.pth`)
    -   Vocabulary file (e.g., `vocabulary.pkl`)
    -   WordSim-353 dataset file (`combined.csv`)

The script will:

-   Compute Spearman's rank correlation between the model's similarity scores and human-annotated scores.
-   Visualize the embeddings of selected words using PCA and save the plot into `visualization.png`.

### Visualization

The `evaluation.py` script includes an optional visualization step. It generates a 2D scatter plot of embeddings for the following words:

-   king, queen, prince, princess, aunt, uncle, daughter, son
-   paris, france, london, england
-   apple, potato, mango, fruit
-   lion, wolf, tiger, elephant
-   car, truck, vehicle, bus
-   neptune, saturn, pluto, earth


----------

## Project Structure

```
.
├── data
│   ├── text8.zip               # Text8 corpus (downloaded separately)
│   ├── combined.csv            # WordSim-353 dataset
├── main.py                     # Script for training the Skip-Gram model
├── evaluation.py               # Script for evaluation and visualization
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)

```

----------

## Notes

-   For data preprocessing, ensure the Text8 corpus is unzipped and preprocessed as per the instructions in the project.
-   Modify hyperparameters (e.g., embedding size, batch size, learning rate) directly in `main.py` as needed.