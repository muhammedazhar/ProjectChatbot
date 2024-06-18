from flask import Flask, render_template, request, jsonify
import json
import datetime

from chatbot import get_response

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    
    # Create a new chat history object
    chat_history = {
        'user_message': text,
        'bot_response': response,
        'timestamp': str(datetime.datetime.now())
    }
    
    # Open the chat history JSON file in append mode
    with open('../Docs/chat_history.json', 'a') as file:
        # Write the chat history object to the file
        json.dump(chat_history, file)
        file.write('\n') # Add a new line for each chat history entry
    
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)