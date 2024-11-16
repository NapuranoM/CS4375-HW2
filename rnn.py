import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.num_layers = 1
        self.rnn = nn.RNN(input_dim, h, self.num_layers, nonlinearity='tanh', batch_first=False)
        self.W = nn.Linear(h, 5)  # Number of classes (stars from 1 to 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # Obtain hidden layer representation
        outputs, hidden = self.rnn(inputs)
        # Obtain output layer representations
        outputs = self.W(outputs)
        # Sum over outputs
        sum_outputs = torch.sum(outputs, dim=0)
        # Obtain probability distribution
        predicted_vector = self.softmax(sum_outputs)
        return predicted_vector

def load_data(train_data, val_data):
    with open(train_data, 'r', encoding='utf-8') as training_f:
        training = json.load(training_f)
    with open(val_data, 'r', encoding='utf-8') as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    return tra, val

def preprocess_text(text):
    # Remove punctuation and split
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).split()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    # Load pre-trained word embeddings
    print("========== Loading word embeddings ==========")
    # Assume word_embedding.pkl is a dictionary mapping words to embedding vectors
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    # Ensure 'unk' token exists in word_embedding
    if 'unk' not in word_embedding:
        print("'unk' token not found in word_embedding. Adding it.")
        embedding_dim = len(next(iter(word_embedding.values())))
        word_embedding['unk'] = np.zeros(embedding_dim)
    else:
        embedding_dim = len(word_embedding['unk'])

    # Initialize model
    model = RNN(embedding_dim, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with early stopping condition
    print("========== Starting training ==========")
    stopping_condition = False
    epoch = 0
    best_validation_accuracy = 0
    patience = 2  # Number of epochs to wait before stopping if no improvement
    epochs_no_improve = 0

    while not stopping_condition and epoch < args.epochs:
        model.train()
        random.shuffle(train_data)
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        minibatch_size = 16
        N = len(train_data)

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                idx = minibatch_index * minibatch_size + example_index
                if idx >= N:
                    break
                input_words, gold_label = train_data[idx]
                input_words = preprocess_text(" ".join(input_words))

                # Build vectors with checks
                vectors = []
                for word in input_words:
                    embedding = word_embedding.get(word.lower(), word_embedding.get('unk'))
                    if embedding is None:
                        # This should not happen since we ensured 'unk' exists
                        print(f"Warning: Embedding for word '{word}' is None")
                        continue
                    vectors.append(embedding)

                # Check if vectors is not empty
                if len(vectors) == 0:
                    # Skip this sample
                    continue

                # Convert vectors to numpy array
                vectors = np.array(vectors)

                # Convert numpy array to tensor
                vectors = torch.tensor(vectors, dtype=torch.float32).view(len(vectors), 1, -1)

                # Proceed with forward pass and loss computation
                output = model(vectors)
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)

                # Accumulate loss and update counts
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

                correct += int(predicted_label == gold_label)
                total += 1

            # Update model parameters
            if loss is not None:
                loss = loss / minibatch_size
                loss.backward()
                optimizer.step()

        training_accuracy = correct / total if total > 0 else 0
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {:.4f}".format(epoch + 1, training_accuracy))
        print("Training time for this epoch: {:.2f} seconds".format(time.time() - start_time))

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_words, gold_label in tqdm(valid_data):
                input_words = preprocess_text(" ".join(input_words))
                vectors = []
                for word in input_words:
                    embedding = word_embedding.get(word.lower(), word_embedding.get('unk'))
                    if embedding is None:
                        continue
                    vectors.append(embedding)

                if len(vectors) == 0:
                    continue

                vectors = np.array(vectors)
                vectors = torch.tensor(vectors, dtype=torch.float32).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1

        validation_accuracy = correct / total if total > 0 else 0
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {:.4f}".format(epoch + 1, validation_accuracy))

        # Early stopping
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            epochs_no_improve = 0
            # Optionally, save the best model
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            stopping_condition = True

        epoch += 1

    print("Training completed. Best validation accuracy: {:.4f}".format(best_validation_accuracy))

    # Optional: Evaluate on test data if provided
    if args.test_data:
        print("========== Evaluating on Test Data ==========")
        with open(args.test_data, 'r', encoding='utf-8') as test_f:
            test = json.load(test_f)
        test_data = []
        for elt in test:
            test_data.append((elt["text"].split(), int(elt["stars"] - 1)))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_words, gold_label in tqdm(test_data):
                input_words = preprocess_text(" ".join(input_words))
                vectors = []
                for word in input_words:
                    embedding = word_embedding.get(word.lower(), word_embedding.get('unk'))
                    if embedding is None:
                        continue
                    vectors.append(embedding)

                if len(vectors) == 0:
                    continue

                vectors = np.array(vectors)
                vectors = torch.tensor(vectors, dtype=torch.float32).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1

        test_accuracy = correct / total if total > 0 else 0
        print("Test accuracy: {:.4f}".format(test_accuracy))
