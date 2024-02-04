# Importing the necessary libraries
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Check if GPU is available, use CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Opening intents JSON file and loading it into a variable
try:
    with open('intents.json', 'r') as f:
        intents = json.load(f)
except json.decoder.JSONDecodeError:
    print('Error: JSON file is not valid.')

# Load preprocessed data from pth file
FILE = "data.pth"
data = torch.load(FILE)

# Get input, hidden and output size, all words, and tags from data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Create a NeuralNet instance, move it to device, and load its state
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)

# Set the model to evaluation mode
model.eval()

# Set the bot name
bot_name = "Jarvis"

# Define a function to get the bot's response to a message
def get_response(msg):
    # Tokenize the message and convert it to a bag of words
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)

    # Convert the bag of words to a tensor and move it to device
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get the output of the model and its predicted tag
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Check if the predicted tag has high enough probability
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        # Randomly choose a response from the corresponding intent
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    # If the predicted tag has low probability or is not recognized, return a default response
    return "I'm sorry, I didn't understand what you said. Could you please rephrase your question?"

# Main function to start the chat session
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        user_message = input("You: ")
        if user_message == "quit":
            print("Thanks for chatting! Talk to you later.")
            break
        else:
            bot_response = get_response(user_message)
            print("Chatbot: ", bot_response)