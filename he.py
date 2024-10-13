import numpy as np
import tensorflow as tf
import hashlib
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



import tensorflow as tf
import numpy as np

# encoder
def encode_sentence(sentence, output_size=128):
    ascii_values = [ord(char) for char in sentence]
    if len(ascii_values) < output_size:
        ascii_values += [0] * (output_size - len(ascii_values))
    elif len(ascii_values) > output_size:
        ascii_values = ascii_values[:output_size]
    normalized = np.array(ascii_values)
    return normalized.tolist()

#decoder
def decode_sentence(encoded_array):
    denormalized = (np.array(encoded_array) ).astype(int)
    chars = [chr(value) if value > 0 else '' for value in denormalized]
    return ''.join(chars).strip()

# Dataset Loader
def load_qa_data(file_path):
    questions, answers = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                questions.append(lines[i].strip())
                answers.append(lines[i + 1].strip())
    return questions, answers

# model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(128,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256,activation='relu'),
         tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256,activation='relu'),
         tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256,activation='relu'),
         tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128)
    ])
    model.compile(optimizer='adamax', loss='huber',metrics=['accuracy'])
    return model

# reaining function
def train_qa_model(file_path, epochs=50, batch_size=32):
    # Load data
    questions, answers = load_qa_data(file_path)
    X = np.array([encode_sentence(q) for q in questions])
    y = np.array([encode_sentence(a) for a in answers])

    # train
    model = create_model()
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    return model

# model eval
def evaluate_model(model, questions, answers):
    correct = 0
    total = len(questions)

    for question, true_answer in zip(questions, answers):
        input_encoded = np.array([encode_sentence(question)])
        output_encoded = model.predict(input_encoded)
        predicted_answer = decode_sentence(output_encoded[0])

        if predicted_answer.lower() == true_answer.lower():
            correct += 1

    accuracy = correct / total
    print(f"Model Accuracy: {accuracy:.2f}")




epochs = 50000
batch_Size = 4
file_path = 'questions_answers.txt'
trained_model = train_qa_model(file_path,epochs,batch_Size)

# calling eval
questions, answers = load_qa_data(file_path)
evaluate_model(trained_model, questions, answers)

trained_model.save('biggmodel.h5')
#testing
test_question = "What is the currency of Japan?"
encoded_question = np.array([encode_sentence(test_question)])
encoded_answer = trained_model.predict(encoded_question)
decoded_answer = decode_sentence(encoded_answer[0])
print(f"Q: {test_question}")
print(f"A: {decoded_answer}")

