import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------------
# 1. Training data
# -----------------------------------

text = """
I love natural language processing
I love machine learning
natural language processing is interesting
machine learning is powerful
I like deep learning
deep learning is interesting
"""

# Convert text to lowercase and tokenize
words = text.lower().split()

# Create vocabulary
vocab = sorted(set(words))

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

vocab_size = len(vocab)

print("Vocabulary:")
print(word_to_idx)
print("Vocabulary size:", vocab_size)


# -----------------------------------
# 2. Create CBOW training data
# -----------------------------------

window_size = 2

training_data = []

for i in range(window_size, len(words) - window_size):

    context = (
        words[i - window_size:i]
        + words[i + 1:i + window_size + 1]
    )

    target = words[i]

    context_indices = [word_to_idx[word] for word in context]
    target_index = word_to_idx[target]

    training_data.append(
        (context_indices, target_index)
    )

print("\nExample training data:")
for context, target in training_data[:5]:
    print(
        [idx_to_word[i] for i in context],
        "->",
        idx_to_word[target]
    )


# -----------------------------------
# 3. CBOW Model
# -----------------------------------

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # Word embedding layer
        self.embeddings = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Linear layer for prediction
        self.linear = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, context_words):

        # Get embeddings for context words
        embedded = self.embeddings(context_words)

        # Average context embeddings
        context_embedding = embedded.mean(dim=0)

        # Predict target word
        output = self.linear(context_embedding)

        return output


# -----------------------------------
# 4. Create model
# -----------------------------------

embedding_dim = 50

model = CBOW(
    vocab_size,
    embedding_dim
)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# -----------------------------------
# 5. Train the model
# -----------------------------------

epochs = 1000

for epoch in range(epochs):

    total_loss = 0

    for context, target in training_data:

        context_tensor = torch.tensor(
            context,
            dtype=torch.long
        )

        target_tensor = torch.tensor(
            [target],
            dtype=torch.long
        )

        # Forward pass
        output = model(context_tensor)

        # Calculate loss
        loss = loss_function(
            output.unsqueeze(0),
            target_tensor
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss:.4f}"
        )


# -----------------------------------
# 6. Get word embeddings
# -----------------------------------

print("\nEmbedding for 'learning':")

learning_index = word_to_idx["learning"]

learning_vector = model.embeddings.weight[
    learning_index
].detach()

print(learning_vector)


# -----------------------------------
# 7. Find similar words
# -----------------------------------

def similar_words(word, top_n=5):

    word_index = word_to_idx[word]

    word_vector = model.embeddings.weight[
        word_index
    ]

    similarities = []

    for other_word, other_index in word_to_idx.items():

        if other_word == word:
            continue

        other_vector = model.embeddings.weight[
            other_index
        ]

        similarity = torch.cosine_similarity(
            word_vector.unsqueeze(0),
            other_vector.unsqueeze(0)
        )

        similarities.append(
            (other_word, similarity.item())
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:top_n]


print("\nWords similar to 'learning':")

for word, score in similar_words("learning"):
    print(f"{word}: {score:.4f}")
