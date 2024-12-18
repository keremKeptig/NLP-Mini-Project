import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import zipfile
import torch.nn.functional as F


def extract_first_n_words(data_folder='data', zip_filename='text8.zip', output_filename='text8_20m.txt', n_words=2000000):
    output_path = os.path.join(data_folder, output_filename)
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping extraction.")
        return

    zip_path = os.path.join(data_folder, zip_filename)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The dataset file {zip_path} does not exist. Please ensure it is in the {data_folder} folder.")


    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder)


    text8_path = os.path.join(data_folder, 'text8')
    with open(text8_path, 'r') as f:
        text = f.read()


    words = text.split()


    selected_words = words[:n_words]


    with open(output_path, 'w') as f:
        f.write(' '.join(selected_words))

    print(f'Saved first {n_words} words to {output_path}')


def build_vocabulary(input_filename='text8_20m.txt', data_folder='data', vocab_size=60000):
    vocab_path = os.path.join(data_folder, 'vocabulary.pkl')
    if os.path.exists(vocab_path):
        print(f"Vocabulary already exists at {vocab_path}, loading it.")
        with open(vocab_path, 'rb') as f:
            vocabulary = pickle.load(f)
        return vocabulary

    input_path = os.path.join(data_folder, input_filename)


    with open(input_path, 'r') as f:
        text = f.read()


    words = text.split()


    word_counts = Counter(words)
    print('Total unique words:', len(word_counts))


    most_common = word_counts.most_common(vocab_size - 1)  # Reserve one spot for <UNK>
    vocabulary = {word: count for word, count in most_common}


    vocabulary['<UNK>'] = sum(count for word, count in word_counts.items() if word not in vocabulary)

    print('Vocabulary size (including <UNK>):', len(vocabulary))


    with open(vocab_path, 'wb') as f:
        pickle.dump(vocabulary, f)
    print(f"Vocabulary saved to {vocab_path}")

    return vocabulary


def assign_indices(vocabulary, data_folder='data'):
    indices_path = os.path.join(data_folder, 'word_indices.pkl')
    if os.path.exists(indices_path):
        print(f"Word indices already exist at {indices_path}, loading them.")
        with open(indices_path, 'rb') as f:
            word_to_index, index_to_word = pickle.load(f)
        return word_to_index, index_to_word


    word_to_index = {word: idx for idx, (word, _) in enumerate(vocabulary.items())}
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    with open(indices_path, 'wb') as f:
        pickle.dump((word_to_index, index_to_word), f)
    print(f"Word indices saved to {indices_path}")

    return word_to_index, index_to_word


def convert_to_indices(input_filename='text8_20m.txt', output_filename='text8_indices.txt', data_folder='data', word_to_index=None):
    indices_npy_path = os.path.join(data_folder, 'text8_indices.npy')
    if os.path.exists(indices_npy_path):
        print(f"Indices file {indices_npy_path} already exists, skipping conversion.")
        return

    input_path = os.path.join(data_folder, input_filename)
    output_path = os.path.join(data_folder, output_filename)


    with open(input_path, 'r') as f:
        text = f.read()


    words = text.split()


    indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in words]


    word_frequencies = Counter(words)
    top_words = dict(word_frequencies.most_common(60000 - 1))  # Reserve one spot for <UNK>
    unk_count = sum(count for word, count in word_frequencies.items() if word not in top_words)
    top_words['<UNK>'] = unk_count


    with open(output_path, 'w') as f:
        for word, count in sorted(top_words.items(), key=lambda item: -item[1]):
            f.write(f'"{word}": {count}\n')

    print(f'Converted text saved to {output_path}')


    np.save(indices_npy_path, np.array(indices))
    print(f"Indices saved to {indices_npy_path}")


def generate_skip_gram_pairs(data_folder='data', window_size=2, index_to_word=None):
    pairs_path = os.path.join(data_folder, 'skip_gram_pairs.txt')
    indices_path = os.path.join(data_folder, 'text8_indices.npy')

    if os.path.exists(pairs_path):
        print(f"Skip-gram pairs file {pairs_path} already exists, skipping generation.")
        return

    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"{indices_path} not found. Please run convert_to_indices first.")

    indices = np.load(indices_path)


    skip_gram_pairs = []
    for i in range(len(indices)):
        center_word = indices[i]
        start = max(i - window_size, 0)
        end = min(i + window_size + 1, len(indices))
        for j in range(start, end):
            if j != i:
                context_word = indices[j]
                skip_gram_pairs.append((center_word, context_word))


    with open(pairs_path, 'w') as f:
        for center, context in skip_gram_pairs:
            f.write(f'{center},{context}\n')


    if index_to_word:
        for center, context in skip_gram_pairs[:20]:  # Display only the first 20 pairs
            print(f"({center}, {context}) -> ('{index_to_word[center]}', '{index_to_word[context]}')")

    print(f"Skip-gram pairs saved to {pairs_path}")


def prepare_negative_sampling_distribution(vocabulary, data_folder='data', output_filename='smoothed_distribution.txt'):
    dist_path = os.path.join(data_folder, 'smoothed_distribution.pkl')
    if os.path.exists(dist_path):
        print(f"Smoothed distribution already exists at {dist_path}, loading it.")
        with open(dist_path, 'rb') as f:
            smoothed_unigram_distribution = pickle.load(f)
        return smoothed_unigram_distribution


    freqs = list(vocabulary.values())
    total_count = sum(freqs)


    unigram_distribution = {word: freq / total_count for word, freq in vocabulary.items()}


    alpha = 0.75  # α = 3/4
    smoothed_values = [prob ** alpha for prob in unigram_distribution.values()]
    normalization_factor = sum(smoothed_values)
    smoothed_unigram_distribution = {
        word: (unigram_distribution[word] ** alpha) / normalization_factor
        for word in vocabulary.keys()
    }


    output_path = os.path.join(data_folder, output_filename)
    with open(output_path, 'w') as f:
        for word, probability in sorted(smoothed_unigram_distribution.items(), key=lambda item: -item[1]):
            f.write(f'"{word}": {probability}\n')


    with open(dist_path, 'wb') as f:
        pickle.dump(smoothed_unigram_distribution, f)

    print(f"Smoothed unigram distribution saved to {output_path}")
    return smoothed_unigram_distribution


def log_sigmoid(x):

    return -F.softplus(-x)


class SkipGramDataset(Dataset):
    def __init__(self, data_folder='data', word_to_index=None, smoothed_distribution=None, negative_samples=5):
        self.data_folder = data_folder
        self.word_to_index = word_to_index
        self.negative_samples = negative_samples


        pairs_file = os.path.join(data_folder, 'skip_gram_pairs.txt')
        with open(pairs_file, 'r') as f:
            lines = f.readlines()

        self.pairs = []
        for line in lines:
            center, context = line.strip().split(',')
            center = int(center)
            context = int(context)
            self.pairs.append((center, context))

        words = list(smoothed_distribution.keys())
        self.word_indices = [word_to_index[w] for w in words]
        self.word_probs = torch.tensor([smoothed_distribution[w] for w in words], dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]

        neg_indices = torch.multinomial(self.word_probs, self.negative_samples, replacement=True)
        neg_samples = [self.word_indices[i] for i in neg_indices]

        return center, context, torch.tensor(neg_samples, dtype=torch.long)


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)


        nn.init.uniform_(self.input_embeddings.weight, a=-0.5/embedding_dim, b=0.5/embedding_dim)
        nn.init.uniform_(self.output_embeddings.weight, a=-0.5/embedding_dim, b=0.5/embedding_dim)

    def forward(self, center_words, context_words, negative_words):
        center_embeds = self.input_embeddings(center_words)  # (batch_size, embedding_dim)
        context_embeds = self.output_embeddings(context_words)  # (batch_size, embedding_dim)
        negative_embeds = self.output_embeddings(negative_words)  # (batch_size, negative_samples, embedding_dim)

        positive_score = torch.sum(center_embeds * context_embeds, dim=1)
        negative_score = torch.bmm(negative_embeds, center_embeds.unsqueeze(2)).squeeze(2)

        return positive_score, negative_score

    def loss(self, positive_score, negative_score):
        pos_loss = -torch.mean(log_sigmoid(positive_score))
        neg_loss = -torch.mean(torch.sum(log_sigmoid(-negative_score), dim=1))
        return pos_loss + neg_loss


if __name__ == '__main__':
    data_folder = 'data'


    extract_first_n_words(data_folder=data_folder)


    vocab = build_vocabulary(data_folder=data_folder)


    word_to_index, index_to_word = assign_indices(vocab, data_folder=data_folder)


    convert_to_indices(data_folder=data_folder, word_to_index=word_to_index)


    generate_skip_gram_pairs(data_folder=data_folder, window_size=2, index_to_word=index_to_word)


    smoothed_distribution = prepare_negative_sampling_distribution(vocab, data_folder=data_folder)
    print("Smoothed unigram distribution prepared.")




    embedding_dim = 100
    negative_samples = 5
    batch_size = 512
    epochs = 2
    lr = 0.001

    dataset = SkipGramDataset(data_folder=data_folder,
                              word_to_index=word_to_index,
                              smoothed_distribution=smoothed_distribution,
                              negative_samples=negative_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model_path = os.path.join(data_folder, 'skipgram_model.pth')
    if os.path.exists(model_path):
        print("Loading existing model weights...")
        model = SkipGramModel(vocab_size=len(vocab), embedding_dim=embedding_dim)
        model.load_state_dict(torch.load(model_path))
    else:
        model = SkipGramModel(vocab_size=len(vocab), embedding_dim=embedding_dim)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    print("Starting training...")
max_batches = 999999  # Set the maximum number of batches to process per epoch

for epoch in range(epochs):
    total_loss = 0.0
    for i, (center, context, negatives) in enumerate(dataloader):
        center = center.long()
        context = context.long()
        negatives = negatives.long()

        optimizer.zero_grad()
        positive_score, negative_score = model(center, context, negatives)
        loss = model.loss(positive_score, negative_score)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        if i + 1 >= max_batches:
            print(f"Reached maximum batch limit ({max_batches}) for this epoch.")
            break  # Exit the loop early

    avg_loss = total_loss / min(len(dataloader), max_batches)  # Adjust for fewer batches
    print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")


    torch.save(model.state_dict(), model_path)
    print(f"Model state_dict saved to {model_path}")




    vocab_path = os.path.join(data_folder, 'vocabulary.pkl')
    indices_path = os.path.join(data_folder, 'word_indices.pkl')
    dist_path = os.path.join(data_folder, 'smoothed_distribution.pkl')



