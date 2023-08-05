import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import random


nltk.download('punkt')
nltk.download('wordnet')


intent_response_pairs = {
    "hello": ["Hi there!", "Hello!", "Hey!"],
    "how_are_you": ["I'm doing great, thank you!", "I'm just a bot, but thanks for asking!"],
    "goodbye": ["Goodbye!", "See you later!", "Bye!"],
    "default": ["I'm sorry, I don't understand.", "Can you rephrase that?", "I'm still learning!"],
}


def get_intent(user_input):
    # Tokenize the user input
    tokens = word_tokenize(user_input.lower())

    # Define some patterns to match intents
    hello_patterns = ["hello", "hi", "hey", "hi there"]
    how_are_you_patterns = ["how are you", "how's it going", "what's up"]
    goodbye_patterns = ["goodbye", "bye", "see you later"]

    # Check for matching patterns to determine the intent
    if any(word in tokens for word in hello_patterns):
        return "hello"
    elif any(word in tokens for word in how_are_you_patterns):
        return "how_are_you"
    elif any(word in tokens for word in goodbye_patterns):
        return "goodbye"
    else:
        return "default"

# Function to get a random response based on the intent
def get_response(intent):
    if intent in intent_response_pairs:
        return random.choice(intent_response_pairs[intent])
    else:
        return random.choice(intent_response_pairs["default"])

# Main function to run the chatbot
def main():
    print("Bot: Hi, I'm a simple chatbot. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break

        intent = get_intent(user_input)
        response = get_response(intent)
        print("Bot:", response)

if __name__ == "__main__":
    main()
