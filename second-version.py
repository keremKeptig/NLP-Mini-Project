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
LEARNING_RATE = 0.01  # Initial learning rate

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
    Collate function to pad sequences of different lengths.
    Pads word indices with 0 (for <PAD>) and heads with -1 (ignored by the loss).
    """
    tokens_list = [item[0] for item in batch]
    heads_list = [item[1] for item in batch]

    # Pad sequences to the same length
    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=0)
    padded_heads = pad_sequence(heads_list, batch_first=True, padding_value=-1)

    return padded_tokens, padded_heads


# ------------------------------------------------------------------
# MODEL: BILINEAR PARSER (TODO)  
# ------------------------------------------------------------------


class BilinearParser(nn.Module):
    """
    Bilinear Parser for Dependency Parsing.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(BilinearParser, self).__init__()
        
        # Embedding layer to convert word indices to embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM layer with 2 layers and bidirectional encoding
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            bidirectional=True, 
                            batch_first=True)
        
        # MLP layers for dependent and head representations
        self.dependent_mlp = nn.Linear(hidden_dim * 2, hidden_dim)
        self.head_mlp = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Bilinear layer to score dependent-head pairs
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)

    def forward(self, x):
        """
        Forward pass to compute arc scores.

        :param x: [B, T] tensor of word indices
        :return: [B, T, T+1] arc scores
        """
        batch_size, seq_len = x.size()

        # Step 1: Embed the input word indices
        embeddings = self.embedding(x)  # Shape: [B, T, embedding_dim]

        # Step 2: Pass embeddings through BiLSTM
        lstm_out, _ = self.lstm(embeddings)  # Shape: [B, T, hidden_dim*2]

        # Step 3: Compute dependent and head representations
        dependents = self.dependent_mlp(lstm_out)  # Shape: [B, T, hidden_dim]
        heads = self.head_mlp(lstm_out)  # Shape: [B, T, hidden_dim]

        # Step 4: Compute arc scores
        # Initialize score tensor: [B, T, T+1]
        scores = torch.zeros(batch_size, seq_len, seq_len + 1, device=x.device)

        for b in range(batch_size):
            for i in range(seq_len):
                # Extract dependent representation for token i
                dependent = dependents[b, i].unsqueeze(0)  # Shape: [1, hidden_dim]
                
                # Expand head representations for token i
                head_reps = heads[b]  # Shape: [T, hidden_dim]
                
                # Compute scores for token i against all potential heads
                score = self.bilinear(dependent, head_reps)  # Shape: [T, 1]
                
                # Save scores (append a dummy score for ROOT at the end)
                scores[b, i, :-1] = score.squeeze(-1)  # Shape: [T]
                scores[b, i, -1] = 0.0  # ROOT dummy score

        return scores



# ------------------------------------------------------------------
# DUMMY MODEL DEFINITION (Replace with your actual model)
# ------------------------------------------------------------------
class DummyDependencyModel(nn.Module):
    """
    A simple placeholder model with:
    - An embedding layer
    - An LSTM
    - A Linear output layer
    Just for demonstration.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # for example

    def forward(self, x):
        """
        x is assumed to be of shape (batch_size, seq_length).
        Returns logits of shape (batch_size, seq_length, vocab_size).
        """
        x = self.embed(x)  # <--- IndexError occurs if x has out-of-range indices
        x, _ = self.lstm(x)
        logits = self.fc(x)
        return logits


# ------------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters (Add or adjust as needed)
    BATCH_SIZE = 2
    EPOCHS = 2

    # Path to your UD file (update path if needed):
    data_file = r"C:\Users\userpc\Desktop\first-semester-JMU\natural-language\mini-project\mini2\en_ewt-ud-train.conllu"


    # Build vocabulary
    print("Building vocabulary from dataset...")
    vocab_builder_ds = DependencyDataset(data_file, build_vocab=True)
    word2idx = vocab_builder_ds.word2idx
    print(f"Vocabulary size: {len(word2idx)}")

    # Load final dataset using the built vocabulary
    dataset = DependencyDataset(data_file, word2idx=word2idx, build_vocab=False)

    # Split data (Placeholder: if dataset is small, just keep them all in train)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"Dataset size: {dataset_size}")
    print(f"Train/Val/Test sizes: {train_size}/{val_size}/{test_size}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                             collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # Initialize Model, Optimizer, Loss
    # ------------------------------------------------------------------
    vocab_size = len(word2idx)
    model = DummyDependencyModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Specify ignore_index=-1

    # ---------------------------------------------------------
    # TRAINING LOOP (TODO)
    # ---------------------------------------------------------
    print("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(EPOCHS):
        # set model to training mode
        model.train()
        total_loss = 0.0

        for inputs, heads in train_loader:
            # move inputs and heads to device
            inputs, heads = inputs.to(device), heads.to(device)

            # forward pass
            logits = model(inputs)

            # For CrossEntropyLoss, logits need shape (N, C, ...) => do transpose
            loss = criterion(logits.transpose(1, 2), heads)

            # zero out previous gradients
            optimizer.zero_grad()

            # backprop + optimizer step
            loss.backward()
            optimizer.step()

            # accumulate total_loss
            total_loss += loss.item()

        # compute average loss for logging
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_loss:.4f}")

    # ---------------------------------------------------------
    # (optional) Validation Loop (TODO)
    # ---------------------------------------------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, heads in val_loader:
            inputs, heads = inputs.to(device), heads.to(device)
            logits = model(inputs)
            val_batch_loss = criterion(logits.transpose(1, 2), heads)
            val_loss += val_batch_loss.item()

    val_loss /= len(val_loader) if len(val_loader) > 0 else 1
    print(f"Validation Loss: {val_loss:.4f}")

    # ---------------------------------------------------------
    # TEST EVALUATION LOOP (TODO)
    # ---------------------------------------------------------
    print("Evaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, heads in test_loader:
            inputs, heads = inputs.to(device), heads.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=-1)

            correct += (preds == heads).sum().item()
            total += heads.numel()

    test_acc = correct / total if total != 0 else 0
    print(f"Test Accuracy: {test_acc:.4f}")

    # ---------------------------------------------------------
    # SAMPLE PREDICTIONS
    # ---------------------------------------------------------
    idx2word = {v: k for k, v in word2idx.items()}
    for i in range(min(2, len(test_dataset))):
        tokens_tensor, heads_tensor = test_dataset[i]
        input_ids = tokens_tensor.unsqueeze(0).to(device)
        logits = model(input_ids)
        preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
        words = [idx2word[idx.item()] for idx in tokens_tensor]

        print(f"Sentence: {words}")
        print(f"True Heads: {heads_tensor.tolist()}")
        print(f"Predicted Heads: {preds.tolist()}")
        print("----")

    correct_unlabeled = 0
    total_unlabeled = 0
    for inputs, heads in test_loader:
        inputs, heads = inputs.to(device), heads.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=-1)  # Predicted heads
        mask = heads != -1  # Ignore padding positions
        correct_unlabeled += (preds[mask] == heads[mask]).sum().item()
        total_unlabeled += mask.sum().item()

    uas = correct_unlabeled / total_unlabeled if total_unlabeled > 0 else 0
    print(f"Test UAS: {uas:.4f}")

