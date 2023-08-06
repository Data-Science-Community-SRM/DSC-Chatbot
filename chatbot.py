import random
import json
import nltk
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nltk.download("punkt")
nltk.download("stopwords")

def load_dataset(file_path):
    with open(file_path, "r") as file:
        dataset = json.load(file)
    return dataset

def preprocess_data(dataset):
    texts = []
    labels = []
    
    for intent in dataset["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern.lower())
            labels.append(intent["tag"])
    
    return texts, labels

def create_word_embeddings(texts):
    tokenized_texts = [nltk.word_tokenize(text) for text in texts]
    model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, sg=1)
    return model

def create_input_data(texts, word_embeddings_model):
    tokenized_texts = [nltk.word_tokenize(text) for text in texts]
    input_data = []
    for tokens in tokenized_texts:
        embeddings = [word_embeddings_model.wv[token] for token in tokens if token in word_embeddings_model.wv]
        if embeddings:
            input_vector = np.mean(embeddings, axis=0)
            input_data.append(input_vector)
    return np.array(input_data)


from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def create_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, activation="relu", input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_dim, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def get_response(dataset, model, word_embeddings_model, user_input):
    user_input = user_input.lower()
    user_input_embedding = [word_embeddings_model.wv[token] for token in nltk.word_tokenize(user_input) if token in word_embeddings_model.wv]
    if not user_input_embedding:
        return "I'm sorry, I don't understand. Can you rephrase that?"

    user_input_vector = np.mean(user_input_embedding, axis=0)
    user_input_vector = user_input_vector.reshape(1, -1)

    predicted_class_index = np.argmax(model.predict(user_input_vector), axis=-1)
    predicted_class = unique_labels[predicted_class_index[0]]  # Use unique_labels here

    for intent in dataset["intents"]:
        if intent["tag"] == predicted_class:
            response = random.choice(intent["responses"])
            return response

    return "I'm sorry, I don't understand. Can you rephrase that?"


if __name__ == "__main__":
    dataset = load_dataset("intents.json")
    texts, labels = preprocess_data(dataset)

    word_embeddings_model = create_word_embeddings(texts)
    input_data = create_input_data(texts, word_embeddings_model)

    # Convert labels to one-hot encoded vectors
    unique_labels = np.unique(labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_index[label] for label in labels])
    y_onehot = np.eye(len(unique_labels))[y_encoded]

    model = create_model(input_data.shape[1], y_onehot.shape[1])

    # Train the model
    model.fit(input_data, y_onehot, epochs=50, batch_size=8)

    print("Chatbot: Hi there! How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("Chatbot: Goodbye! Have a great day!")
            break

        response = get_response(dataset, model, word_embeddings_model, user_input)
        print("Chatbot:", response)
