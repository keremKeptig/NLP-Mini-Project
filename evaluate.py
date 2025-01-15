import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Function to load WordSim-353 dataset
def load_wordsim_353(data_folder="data", filename="combined.csv"):
    filepath = os.path.join(data_folder, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"The WordSim-353 file {filepath} was not found. Please download it and place it in {data_folder}."
        )

    word_pairs = []
    human_scores = []

    with open(filepath, "r") as f:
        next(f)  # Skip the header
        for line in f:
            w1, w2, score = line.strip().split(",")
            word_pairs.append((w1, w2))
            human_scores.append(float(score))

    return word_pairs, human_scores


# Function to compute cosine similarity
def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))


def visualize_embeddings(
    model_embeddings, vocab, words_to_visualize, reduction_method="pca"
):
    print("Starting visualization...")
    word_indices = [vocab[word] for word in words_to_visualize if word in vocab]
    embeddings_to_plot = model_embeddings[word_indices].detach().numpy()

    if reduction_method == "pca":
        reducer = PCA(n_components=2)
    elif reduction_method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid reduction method. Choose 'pca' or 'tsne'.")

    reduced_embeddings = reducer.fit_transform(embeddings_to_plot)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words_to_visualize):
        if word in vocab:
            plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label=word)
            plt.text(
                reduced_embeddings[i, 0] + 0.01,
                reduced_embeddings[i, 1] + 0.01,
                word,
                fontsize=9,
            )

    plt.title("Word Embeddings Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


# Main evaluation script
def evaluate_model(
    model_path, vocab_path, data_folder="data", wordsim_filename="combined.csv"
):
    # Load the model
    print("Loading the model...")

    # Check and load the vocabulary
    if vocab_path.endswith(".pkl"):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = torch.load(vocab_path)  # Use torch.load for .pth or .pt formats

    vocab_size = len(vocab)
    embedding_dim = 100  # Ensure this matches your model's dimension

    # Load the embeddings directly from the saved model file
    model_embeddings = torch.load(model_path)["input_embeddings.weight"]

    # Load the WordSim-353 dataset
    print("Loading WordSim-353 dataset...")
    word_pairs, human_scores = load_wordsim_353(
        data_folder=data_folder, filename=wordsim_filename
    )

    # Prepare to compute cosine similarities
    model_word_to_index = {word: idx for idx, word in enumerate(vocab.keys())}

    computed_similarities = []
    valid_human_scores = []

    print("Computing similarities...")
    for (w1, w2), human_score in zip(word_pairs, human_scores):
        if w1 in model_word_to_index and w2 in model_word_to_index:
            idx1 = model_word_to_index[w1]
            idx2 = model_word_to_index[w2]

            embedding1 = model_embeddings[idx1]
            embedding2 = model_embeddings[idx2]

            similarity = cosine_similarity(embedding1, embedding2).item()
            computed_similarities.append(similarity)
            valid_human_scores.append(human_score)
        else:
            # Skip pairs where one or both words are not in the vocabulary
            print(
                f"Skipping pair ({w1}, {w2}) as one or both words are not in the vocabulary."
            )

    # Calculate Spearman's rank correlation
    print("Calculating Spearman's rank correlation...")
    correlation, _ = spearmanr(computed_similarities, valid_human_scores)

    # Report the result
    print(f"Spearman's rank correlation coefficient: {correlation:.4f}")

    words_to_visualize = [
        "king",
        "queen",
        "prince",
        "princess",
        "aunt",
        "uncle",
        "daughter",
        "son",
        "paris",
        "france",
        "london",
        "england",
        "apple",
        "potato",
        "mango",
        "fruit",
        "lion",
        "wolf",
        "tiger",
        "elephant",
        "car",
        "truck",
        "vehicle",
        "bus",
        "neptune",
        "saturn",
        "pluto",
        "earth",
    ]
    visualize_embeddings(model_embeddings, model_word_to_index, words_to_visualize)

    return correlation


# Example usage
if __name__ == "__main__":
    data_folder = "data"
    model_path = os.path.join(data_folder, "skipgram_model.pth")
    vocab_path = os.path.join(data_folder, "vocabulary.pkl")
    wordsim_filename = "combined.csv"

    spearman_correlation = evaluate_model(
        model_path,
        vocab_path,
        data_folder=data_folder,
        wordsim_filename=wordsim_filename,
    )
    print(
        f"Evaluation completed. Spearman's rank correlation: {spearman_correlation:.4f}"
    )
