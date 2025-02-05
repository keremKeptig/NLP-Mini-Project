# Dependency Parser with BiLSTM and Bilinear Scoring

This project implements a dependency parser using a BiLSTM model with bilinear scoring for arc prediction. The dataset format follows Universal Dependencies (`.conllu` files).

## Requirements
Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
torch
torchvision
tqdm
scikit-learn
conllu
```

## How to Run the Code

### 1. Prepare Dataset
Ensure you have a dataset containing sentences with annotations **Universal Dependencies (.conllu) file**. Update the path to the file `data_file` in the script:

```python
# Modify this path in the script
data_file = "./en_ewt-ud-train.conllu"
```

### 2. Run the Script
Execute the following command:

```bash
python parser.py
```

### 3. Training and Evaluation
- The script trains the model and validate performance.
- The training process is monitored using a `tqdm` progress bar, and the average loss is printed after each epoch.
- The model is evaluated on the test set.
- Sample predictions are displayed after training.


## Notes
- You can modify hyperparameters in the script for better performance.

