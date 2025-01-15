import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from conllu import parse_incr
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
import collections

# -----------------------------
# HYPERPARAMETERS (You can play with these)
# -----------------------------
EMBEDDING_DIM = 128    # Size of the word embedding vectors
HIDDEN_DIM = 256       # Size of the hidden dimension in the LSTM and MLP layers
BATCH_SIZE = 512       # Number of samples per training batch
EPOCHS = 3             # Training epochs
LEARNING_RATE = 0.001  # Initial learning rate

# Detect GPU (CUDA) or default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# DATASET CLASS (WORD-LEVEL) + VOCAB BUILDING
# ------------------------------------------------------------------
class DependencyDataset(Dataset):
    """
    Reads a .conllu file and:
      - Collects (words, heads) pairs for each valid sentence.
      - Adjusts heads so that:
         - head = 0 (root) -> -1 (ignored by CrossEntropyLoss)
         - head = k (1..T) -> (k-1) (0-based index)
    """
    def __init__(self, data_file, word2idx=None, build_vocab=False):
        self.sentences = []
        self.heads = []
        self.word2idx = word2idx
        self.build_vocab = build_vocab
        self.word_counter = collections.Counter() if build_vocab else None
        self._load_data(data_file)
        if self.build_vocab:
            # Reserve index 0 for <PAD>, 1 for <UNK>
            self.word2idx = {"<PAD>": 0, "<UNK>": 1}
            for w, _ in self.word_counter.most_common():
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.word2idx)

    def _load_data(self, data_file):
        with open(data_file, encoding="utf-8") as f:
            for sentence in parse_incr(f):
                tokens = [token["form"] for token in sentence]
                heads_raw = [token["head"] for token in sentence]

                if any(h is None for h in heads_raw):
                    continue

                if self.build_vocab:
                    self.word_counter.update(tokens)

                shifted_heads = []
                for h in heads_raw:
                    if h == 0:
                        shifted_heads.append(-1)   # root -> ignore index
                    else:
                        shifted_heads.append(h - 1)  # 1-based -> 0-based

                self.sentences.append(tokens)
                self.heads.append(shifted_heads)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        heads = self.heads[idx]

        indexed_tokens = []
        for w in tokens:
            if self.word2idx is not None:
                indexed_tokens.append(self.word2idx.get(w, self.word2idx["<UNK>"]))
            else:
                indexed_tokens.append(0)

        indexed_tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long)
        heads_tensor = torch.tensor(heads, dtype=torch.long)
        return indexed_tokens_tensor, heads_tensor


# ------------------------------------------------------------------
# COLLATE FUNCTION FOR DATALOADER
# ------------------------------------------------------------------
def collate_fn(batch):
    """
    Pad sequences of different lengths.
    For word indices, we use 0 (PAD).
    For heads, we use -1 (ignored by the loss).
    """
    indexed_list = [item[0] for item in batch]
    heads_list = [item[1] for item in batch]

    padded_inputs = pad_sequence(indexed_list, batch_first=True, padding_value=0)
    padded_heads = pad_sequence(heads_list, batch_first=True, padding_value=-1)

    return {"input_ids": padded_inputs}, padded_heads


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from conllu import parse_incr
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
import collections

# -----------------------------
# HYPERPARAMETERS (You can play with these)
# -----------------------------
EMBEDDING_DIM = 128    # Size of the word embedding vectors
HIDDEN_DIM = 256       # Size of the hidden dimension in the LSTM and MLP layers
BATCH_SIZE = 512       # Number of samples per training batch
EPOCHS = 3             # Training epochs
LEARNING_RATE = 0.001  # Initial learning rate

# Detect GPU (CUDA) or default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# MODEL: BILINEAR PARSER
# ------------------------------------------------------------------
class BilinearParser(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(BilinearParser, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.dependent_mlp = nn.Linear(hidden_dim * 2, hidden_dim)
        self.head_mlp = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)  # [B, T, D]
        lstm_out, _ = self.lstm(x)  # [B, T, 2*H]
        dependents = self.dependent_mlp(lstm_out)  # [B, T, H]
        heads = self.head_mlp(lstm_out)  # [B, T, H]

        B, T, H = dependents.size()
        scores = torch.zeros(B, T, T, device=x.device)

        for i in range(T):
            # Extract the dependent representation for token `i`: [B, H]
            dependent_rep = dependents[:, i, :]  # [B, H]

            # Expand dependent_rep to match heads: [B, 1, H] -> Broadcastable with [B, T, H]
            dependent_rep = dependent_rep.unsqueeze(1)  # [B, 1, H]

            # Compute scores for all possible heads for token `i`: [B, T]
            scores[:, i, :] = self.bilinear(dependent_rep, heads).squeeze(1)
        return scores


# ------------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------------
if __name__ == "__main__":
    data_file = r"C:\Users\userpc\Desktop\first-semester-JMU\natural-language\mini-project\mini2\en_ewt-ud-train.conllu"

    print("Building vocabulary from dataset...")
    vocab_builder_ds = DependencyDataset(data_file, build_vocab=True)
    word2idx = vocab_builder_ds.word2idx
    print(f"Vocabulary size: {len(word2idx)}")

    dataset = DependencyDataset(data_file, word2idx=word2idx, build_vocab=False)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    vocab_size = len(word2idx)
    model = BilinearParser(EMBEDDING_DIM, HIDDEN_DIM, vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs["input_ids"].to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

    print("Evaluating on test set...")
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs["input_ids"].to(device)
            targets = targets.to(device)

            logits = model(inputs)
            preds = torch.argmax(logits, dim=2).view(-1).cpu().tolist()
            targets = targets.view(-1).cpu().tolist()

            valid_indices = [i for i, t in enumerate(targets) if t != -1]
            test_preds.extend([preds[i] for i in valid_indices])
            test_targets.extend([targets[i] for i in valid_indices])

    test_acc = accuracy_score(test_targets, test_preds)
    print(f"Test Accuracy: {test_acc:.4f}")

    idx2word = {v: k for k, v in word2idx.items()}
    for i in range(2):
        tokens_tensor, heads_tensor = test_dataset[i]
        input_ids = tokens_tensor.unsqueeze(0).to(device)
        logits = model(input_ids)
        preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
        words = [idx2word[idx.item()] for idx in tokens_tensor]
        print(f"Sentence: {words}")
        print(f"True Heads: {heads_tensor.tolist()}")
        print(f"Predicted Heads: {preds.tolist()}")
        print("----")


