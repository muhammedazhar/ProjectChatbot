# Importing the necessary libraries
import nltk
# Download NLTK data to the current directory
nltk.download('punkt', download_dir='.')
# import random
import json
import torch
import torch.nn as nn
import numpy as np
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem

# Opening intents JSON file and loading it into a variable
try:
    with open('./training_data/intents.json', 'r') as f:
        intents = json.load(f)
except json.decoder.JSONDecodeError:
    print('Error: JSON file is not valid.')

# Initializing empty lists for words, tags, and xy pairs
all_words = []
tags = []
xy = []

# Loop through each sentence in the intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # Add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = tokenize(pattern)
        # Add to our words list
        all_words.extend(w)
        # Add to xy pair
        xy.append((w, tag))

# Stem and lower each word in the all_words list
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Remove duplicates and sort the all_words and tags list
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Creating training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # Y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(f'Input size = {input_size}, Output size = {output_size}')

# Creating a ChatDataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # We can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
dataset = ChatDataset()

# Create a PyTorch DataLoader to handle batches during training
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Set the device for training to CUDA if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the NeuralNet model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Checks if the current epoch is a multiple of 100 and prints the loss value.    
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}],\n[==============================] Loss: {loss.item():.4f}')

# Prints the final loss value after the training process is complete.
print(f'final loss: {loss.item():.4f}')

# Create a dictionary containing the trained model's state, input size, hidden size, output size, all words, and tags
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

# Save the trained model's data dictionary to a file named "data.pth"
FILE = "data.pth"
torch.save(data, FILE)

# Print a message indicating that the training is complete and the file is saved
print(f'Training complete. File has saved to {FILE}')
print('Model has sucessfully trained')