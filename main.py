from collections import Counter
import numpy as np
import os
import zipfile

def extract_first_n_words(data_folder='data', zip_filename='text8.zip', output_filename='text8_20m.txt', n_words=2000000):
    zip_path = os.path.join(data_folder, zip_filename)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The dataset file {zip_path} does not exist. Please ensure it is in the {data_folder} folder.")

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder)

    # Read the text8 file
    text8_path = os.path.join(data_folder, 'text8')
    with open(text8_path, 'r') as f:
        text = f.read()

    # Split into words
    words = text.split()

    # Take the first n_words
    selected_words = words[:n_words]

    # Write to the output file
    output_path = os.path.join(data_folder, output_filename)
    with open(output_path, 'w') as f:
        f.write(' '.join(selected_words))

    print(f'Saved first {n_words} words to {output_path}')

def build_vocabulary(input_filename='text8_20m.txt', data_folder='data', vocab_size=60000):
    input_path = os.path.join(data_folder, input_filename)

    # Read the processed text file
    with open(input_path, 'r') as f:
        text = f.read()

    # Split into words
    words = text.split()

    # Count word frequencies
    word_counts = Counter(words)
    print('Total unique words:', len(word_counts))

    # Select the most frequent words
    most_common = word_counts.most_common(vocab_size - 1)  # Reserve one spot for <UNK>
    vocabulary = {word: count for word, count in most_common}

    # Add <UNK> token to the vocabulary
    vocabulary['<UNK>'] = sum(count for word, count in word_counts.items() if word not in vocabulary)

    print('Vocabulary size (including <UNK>):', len(vocabulary))
    return vocabulary

def assign_indices(vocabulary):
    # Assign a unique index to each word
    word_to_index = {word: idx for idx, (word, _) in enumerate(vocabulary.items())}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return word_to_index, index_to_word

def convert_to_indices(input_filename='text8_20m.txt', output_filename='text8_indices.txt', data_folder='data', word_to_index=None):
    input_path = os.path.join(data_folder, input_filename)
    output_path = os.path.join(data_folder, output_filename)

    # Read the processed text file
    with open(input_path, 'r') as f:
        text = f.read()

    # Split into words
    words = text.split()

    # Convert words to indices
    indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in words]

    # Create a word frequency dictionary with cutoff at 60k words
    word_frequencies = Counter(words)
    top_words = dict(word_frequencies.most_common(60000 - 1))  # Reserve one spot for <UNK>
    unk_count = sum(count for word, count in word_frequencies.items() if word not in top_words)
    top_words['<UNK>'] = unk_count

    # Save words with their frequencies (sorted by frequency) to a file
    with open(output_path, 'w') as f:
        for word, count in sorted(top_words.items(), key=lambda item: -item[1]):
            f.write(f'"{word}": {count}\n')

    print(f'Converted text saved to {output_path}')

    # Save indices for later use (adding this line as new code)
    np.save(os.path.join(data_folder, 'text8_indices.npy'), np.array(indices))

def generate_skip_gram_pairs(data_folder='data', window_size=2, index_to_word=None):
    # Load the indices we saved previously
    indices_path = os.path.join(data_folder, 'text8_indices.npy')
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"{indices_path} not found. Please run convert_to_indices first.")

    indices = np.load(indices_path)

    # Generate skip-gram pairs
    skip_gram_pairs = []
    for i in range(len(indices)):
        center_word = indices[i]
        # context window: from i - window_size to i + window_size, excluding i
        start = max(i - window_size, 0)
        end = min(i + window_size + 1, len(indices))
        for j in range(start, end):
            if j != i:
                context_word = indices[j]
                skip_gram_pairs.append((center_word, context_word))

    # Save skip-gram pairs to a text file for readability
    readable_output_path = os.path.join(data_folder, 'skip_gram_pairs.txt')
    with open(readable_output_path, 'w') as f:
        for center, context in skip_gram_pairs:
            f.write(f'{center},{context}\n')

    # Print skip-gram pairs with words if index_to_word is provided
    if index_to_word:
        for center, context in skip_gram_pairs[:20]:  # Display only the first 20 pairs
            print(f"({center}, {context}) -> ('{index_to_word[center]}', '{index_to_word[context]}')")

    print(f"Skip-gram pairs saved to {readable_output_path}")

###########################################################
# New Logic: Preparing for Negative Sampling
###########################################################

def prepare_negative_sampling_distribution(vocabulary, data_folder='data', output_filename='smoothed_distribution.txt'):
    # Extract words and their frequencies from the vocabulary.
    # Vocabulary is a dict: {word: frequency}
    freqs = list(vocabulary.values())
    total_count = sum(freqs)

    # Compute the unigram distribution U(w) = freq(w)/N
    unigram_distribution = {word: freq / total_count for word, freq in vocabulary.items()}

    # Compute the smoothed unigram distribution
    alpha = 0.75  # α = 3/4
    # Raise each probability to the power of α
    smoothed_values = [prob ** alpha for prob in unigram_distribution.values()]
    normalization_factor = sum(smoothed_values)
    smoothed_unigram_distribution = {
        word: (unigram_distribution[word] ** alpha) / normalization_factor
        for word in vocabulary.keys()
    }

    # Save the smoothed distribution to a file in human-readable format
    output_path = os.path.join(data_folder, output_filename)
    with open(output_path, 'w') as f:
        for word, probability in sorted(smoothed_unigram_distribution.items(), key=lambda item: -item[1]):
            f.write(f'"{word}": {probability}\n')

    print(f"Smoothed unigram distribution saved to {output_path}")
    return smoothed_unigram_distribution

if __name__ == '__main__':
    data_folder = 'data'

    # Step 1: Preprocess the text
    extract_first_n_words(data_folder=data_folder)

    # Step 2: Build the vocabulary
    vocab = build_vocabulary(data_folder=data_folder)

    # Step 3: Assign indices to the vocabulary
    word_to_index, index_to_word = assign_indices(vocab)

    # Step 4: Convert the text to a sequence of word frequencies
    convert_to_indices(data_folder=data_folder, word_to_index=word_to_index)

    # Step 5 (New): Generate skip-gram training data
    generate_skip_gram_pairs(data_folder=data_folder, window_size=2, index_to_word=index_to_word)

    # Step 6 (New): Prepare for Negative Sampling
    smoothed_distribution = prepare_negative_sampling_distribution(vocab, data_folder=data_folder)
    print("Smoothed unigram distribution prepared.")
