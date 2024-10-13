from tensorflow.keras.models import load_model
import numpy as np
# Load the .h5 model file
model = load_model('E:/Llama/tinu/my_model_2.h5')

# Check the model summary to verify the structure
model.summary()


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

test_question = "What is the smallest prime number?"
encoded_question = np.array([encode_sentence(test_question)])
encoded_answer = model.predict(encoded_question)
decoded_answer = decode_sentence(encoded_answer[0])
print(f"Q: {test_question}")
print(f"A: {decoded_answer}")