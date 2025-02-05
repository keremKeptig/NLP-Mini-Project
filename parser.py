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
EMBEDDING_DIM = 128  # Size of the word embedding vectors
HIDDEN_DIM = 512  # Size of the hidden dimension in the LSTM and MLP layers
BATCH_SIZE = 32  # Number of samples per training batch
EPOCHS = 2  # Training epochs
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
                        shifted_heads.append(-1)  # root -> ignore index
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

    return {"input_ids": padded_inputs, "heads": padded_heads}


# ------------------------------------------------------------------
# MODEL: BILINEAR PARSER
# ------------------------------------------------------------------
class BilinearParser(nn.Module):
    """
    Implement the bilinear parser.

    Requirements:
      - An embedding layer for word indices.
      - A BiLSTM for contextual encoding (2 layers, bidirectional).
      - Two MLP (nn.Linear) layers to produce 'dependent' and 'head' representations for each token.
      - A bilinear function (nn.Bilinear) that scores every possible (dependent, head) pair.

    Input shape: [B, T] word indices
    Output shape: [B, T, T+1] arc scores (the extra column is for the dummy root)
    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.0
    ):
        super(BilinearParser, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # BiLSTM layer (2 layers, bidirectional)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # MLPs for dependent and head representations.
        # Since LSTM is bidirectional, its output dimension is 2 * hidden_dim.
        self.dep_mlp = nn.Linear(2 * hidden_dim, hidden_dim)
        self.head_mlp = nn.Linear(2 * hidden_dim, hidden_dim)

        # Activation function
        self.activation = nn.ReLU()

        # Bilinear layer for scoring.
        # Note: nn.Bilinear by default produces an output of shape [B, T, 1]
        #       but we extract the weight for our computations.
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)

        # Define a learnable dummy root parameter.
        # Its shape is [1, hidden_dim]. This will be expanded for each batch.
        self.root = nn.Parameter(torch.randn(1, hidden_dim))

    def forward(self, x):
        """
        :param x: [B, T] tensor of word indices
        :return:  [B, T, T+1] arc scores where score[b, i, j] is how likely
                  it is for token i to have head j in batch b.
                  The extra column (j = T) corresponds to the dummy root.
        """
        # 1. Get embeddings: [B, T, EMBEDDING_DIM]
        embed = self.embedding(x)

        # 2. Pass through BiLSTM: output shape [B, T, 2*HIDDEN_DIM]
        lstm_out, _ = self.lstm(embed)

        # 3. Compute dependent and head representations using MLPs.
        dep = self.activation(self.dep_mlp(lstm_out))  # shape: [B, T, HIDDEN_DIM]
        head = self.activation(self.head_mlp(lstm_out))  # shape: [B, T, HIDDEN_DIM]

        # 4. Compute bilinear scores.
        #    We want: score[b, i, j] = dep[b, i]^T * W * head[b, j]
        #    Extract W from the nn.Bilinear layer.
        W = self.bilinear.weight.squeeze(0)  # shape: [HIDDEN_DIM, HIDDEN_DIM]
        # Compute intermediate representations: [B, T, HIDDEN_DIM]
        intermediate = torch.matmul(dep, W)

        # 5. Compute scores for tokens (non-root heads): [B, T, T]
        token_scores = torch.bmm(intermediate, head.transpose(1, 2))

        # 6. Compute scores for the dummy root head.
        #    Expand the dummy root parameter for each example in the batch.
        B = x.size(0)
        # Expand self.root: [B, 1, HIDDEN_DIM]
        dummy_root = self.root.expand(B, -1, -1)
        # Compute scores with the dummy root: [B, T, 1]
        root_scores = torch.bmm(intermediate, dummy_root.transpose(1, 2))

        # 7. Concatenate the token and dummy root scores along the head dimension.
        #    This results in scores of shape [B, T, T+1]
        scores = torch.cat([token_scores, root_scores], dim=-1)

        return scores


# ------------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Path to your UD file (update path if needed):
    data_file = "./en_ewt-ud-train.conllu"

    # Build vocabulary
    print("Building vocabulary from dataset...")
    vocab_builder_ds = DependencyDataset(data_file, build_vocab=True)
    word2idx = vocab_builder_ds.word2idx
    print(f"Vocabulary size: {len(word2idx)}")

    # Load final dataset using the built vocabulary
    dataset = DependencyDataset(data_file, word2idx=word2idx, build_vocab=False)

    # Split data
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
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Initialize Model, Optimizer, Loss
    vocab_size = len(word2idx)
    # Note: The constructor expects (vocab_size, embedding_dim, hidden_dim, ...)
    model = BilinearParser(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # We ignore targets that are -1 (the root and padded positions)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------
    print("Starting training...")
    with tqdm(
        total=EPOCHS, desc="Training Progress", unit="epoch", disable=False
    ) as pbar:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                inputs = batch["input_ids"].to(device)  # [B, T]
                targets = batch["heads"].to(device)  # [B, T]

                optimizer.zero_grad()
                logits = model(inputs)  # [B, T, T+1] arc scores

                # Reshape logits and targets for loss computation.
                B, T, _ = logits.shape
                logits_flat = logits.view(B * T, T + 1)
                targets_flat = targets.view(B * T)

                loss = criterion(logits_flat, targets_flat)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            pbar.set_postfix({"Epoch": epoch, "Average Loss": f"{avg_loss:.4f}"})
            pbar.update(1)

        # ---------------------------------------------------------
        # (Optional) Validation Loop
        # ---------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device)  # [B, T]
                targets = batch["heads"].to(device)
                logits = model(inputs)
                B, T, _ = logits.shape
                logits_flat = logits.view(B * T, T + 1)
                targets_flat = targets.view(B * T)
                loss = criterion(logits_flat, targets_flat)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        print(f"Epoch {epoch}: Average Validation Loss = {avg_val_loss:.4f}")

    # ---------------------------------------------------------
    # TEST EVALUATION LOOP
    # ---------------------------------------------------------
    print("Evaluating on test set...")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["heads"].to(device)  # [B, T]
            logits = model(inputs)  # [B, T, T+1]
            # Get predicted head indices per token.
            preds = torch.argmax(logits, dim=2)  # [B, T]

            # Flatten predictions and targets, but only consider positions where target != -1.
            mask = targets != -1
            all_preds.extend(preds[mask].cpu().numpy())
            all_targets.extend(targets[mask].cpu().numpy())

    test_acc = accuracy_score(all_targets, all_preds)
    print(f"Test Accuracy: {test_acc:.4f}")

    # ---------------------------------------------------------
    # SAMPLE PREDICTIONS
    # ---------------------------------------------------------
    # Manually inspect a couple of samples from the test dataset.
    idx2word = {v: k for k, v in word2idx.items()}
    for i in range(2):
        tokens_tensor, heads_tensor = test_dataset[i]
        input_ids = tokens_tensor.unsqueeze(0).to(device)  # [1, T]
        logits = model(input_ids)
        preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
        words = [idx2word[idx.item()] for idx in tokens_tensor]
        print(f"Sentence: {words}")
        print(f"True Heads: {heads_tensor.tolist()}")
        print(f"Predicted Heads: {preds.tolist()}")
        print("----")
