import torch
import numpy as np
from static_variables import VOCAB_SIZE, char_to_int, int_to_char

def char_to_onehot(char):
    one_hot = torch.zeros(1, VOCAB_SIZE)
    one_hot[0, char_to_int[char]] = 1
    return one_hot

def onehot_to_char(one_hot):
    argmax = torch.argmax(one_hot).item()
    return int_to_char[argmax]

def generate_name(model, temperature=0.5):
    with torch.no_grad():
        hidden = model.initHidden()
        name = ''
        char = '<S>'
        while char != '<E>':
            input_tensor = char_to_onehot(char)
            output, hidden = model(input_tensor, hidden)

            # Adjust the probabilities with temperature
            adj_outputs = output[0].numpy() / temperature
            # Apply softmax to get the probability distribution
            probs = np.exp(adj_outputs) / np.sum(np.exp(adj_outputs))
            
            # Sample a character index based on the probabilities
            char_idx = np.random.choice(len(probs), p=probs)
            char = int_to_char[char_idx]
            if char != '<E>': name += char
    return name