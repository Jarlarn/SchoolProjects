import numpy as np
from collections import Counter


# ============================================================
# Step 1: Load the data
# ============================================================


def load_data(filepath):
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                labels.append(int(parts[0]))
                texts.append(parts[1])
    return texts, labels


train_texts, train_labels = load_data("data/ReviewBaseTraining.txt")
val_texts, val_labels = load_data("data/ReviewBaseValidation.txt")
test_texts, test_labels = load_data("data/ReviewBaseTest.txt")

print(
    f"Training: {len(train_texts)}, Validation: {len(val_texts)}, Test: {len(test_texts)}"
)


# ============================================================
# Step 2: N-gram feature extraction
# ============================================================


def get_ngrams(text, n_max):
    """Extract all n-grams (1 up to n_max) from a space-separated text."""
    words = text.split()
    ngrams = []
    for n in range(1, n_max + 1):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i : i + n]))
    return ngrams


def build_vocabulary(texts, n_max, c_min):
    """Build a vocabulary of n-grams that appear at least c_min times."""
    counts = Counter()
    for text in texts:
        ngrams = get_ngrams(text, n_max)
        counts.update(ngrams)

    # Only keep n-grams with count >= c_min
    vocab = {}
    idx = 0
    for ngram, count in counts.items():
        if count >= c_min:
            vocab[ngram] = idx
            idx += 1
    return vocab


def texts_to_sparse_features(texts, vocab, n_max):
    """Convert texts to a list of sparse feature dicts {feature_index: count}."""
    sparse_data = []
    for text in texts:
        features = {}
        ngrams = get_ngrams(text, n_max)
        for ngram in ngrams:
            if ngram in vocab:
                idx = vocab[ngram]
                features[idx] = features.get(idx, 0) + 1
        sparse_data.append(features)
    return sparse_data


# ============================================================
# Step 3: Perceptron model (sparse)
# ============================================================


class LinearPerceptron:
    def __init__(self, num_features, max_iter=100, lr=1.0):
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        self.max_iter = max_iter
        self.lr = lr

    def predict_one(self, x_sparse):
        score = self.bias
        for idx, val in x_sparse.items():
            score += self.weights[idx] * val
        return 1 if score > 0 else 0

    def predict(self, X_sparse):
        return np.array([self.predict_one(x) for x in X_sparse])

    def fit(self, X_sparse, y):
        y = np.array(y)
        for epoch in range(self.max_iter):
            errors = 0
            for i in range(len(X_sparse)):
                pred = self.predict_one(X_sparse[i])
                if pred != y[i]:
                    update = self.lr * (y[i] - pred)
                    for idx, val in X_sparse[i].items():
                        self.weights[idx] += update * val
                    self.bias += update
                    errors += 1
            if errors == 0:
                break


def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


# ============================================================
# Step 4: Train with different n_max and c_min values
# ============================================================

n_max_values = [1, 2, 3]
c_min_values = [1, 3, 10, 100]

best_models = {}

for n_max in n_max_values:
    best_val_acc = 0
    best_c_min = None
    best_model = None
    best_vocab = None

    print(f"\n{'=' * 50}")
    print(f"n_max = {n_max}")
    print(f"{'=' * 50}")

    for c_min in c_min_values:
        vocab = build_vocabulary(train_texts, n_max, c_min)
        num_features = len(vocab)

        X_train = texts_to_sparse_features(train_texts, vocab, n_max)
        X_val = texts_to_sparse_features(val_texts, vocab, n_max)

        model = LinearPerceptron(num_features, max_iter=100)
        model.fit(X_train, train_labels)

        val_preds = model.predict(X_val)
        val_acc = accuracy(val_labels, val_preds)

        print(
            f"  c_min={c_min:>3d} | features: {num_features:>6d} | Val Accuracy: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_c_min = c_min
            best_model = model
            best_vocab = vocab

    best_models[n_max] = (best_model, best_vocab, best_c_min, best_val_acc)


# ============================================================
# Step 5: Evaluate best models on test set
# ============================================================

print(f"\n{'=' * 50}")
print("Test set results (best model per n_max)")
print(f"{'=' * 50}")

for n_max in n_max_values:
    model, vocab, c_min, val_acc = best_models[n_max]
    X_test = texts_to_sparse_features(test_texts, vocab, n_max)
    test_preds = model.predict(X_test)
    test_acc = accuracy(test_labels, test_preds)

    print(
        f"  n_max={n_max} | best c_min={c_min:>3d} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}"
    )
