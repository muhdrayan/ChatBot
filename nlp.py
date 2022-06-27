"""
Made by Muhammed Rayan Savad
Admission Number: 08202
Email Address: 08202@bpsdoha.edu.qa
Class: IX - K, 2022-23
"""

# NOTE: I have commented what everything does in this file so that it is better understood what I am trying to do
# Hope you like it! 

# NOTE: There are some modules that have to be imported before running this script
# please find everything to be installed (using pip) in requirements.txt

# This file is for the NLP Processes which have to be done so that the computer can understand our language
# After NLP we will feed it to the Nueral Network

# Importing nltk for NLP
import nltk

# Importing the stemmer for NLP
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() # simplifying words to its basic form (cleaned -> clean)

# Importing other modules
import json

# Necessary download
nltk.download('punkt')

def nlp():
    # Opening our intents file which contains all the training data
    with open('Data/intents.json') as f:
        intents = json.load(f)

    words, documents, categories = [], [], [] # Creating lists to store all of them
    ignore = ['?', '!']

    # Looping through all the intents
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern) # Tokenization or splitting
            words.extend(word)

            # Adding it to the document list
            documents.append((word, intent['tag']))
        categories.append(intent['tag'])

    # Stemming process 
    words = [stemmer.stem(word.lower()) for word in words if word not in ignore] # List Comprehension Technique

    # We now convert it to set (to remove duplicates as sets contain only unique) and then back to list
    words = list(set(words))

    # Then we sort it
    words.sort()
    """
    # Result checking
    print(f'Documents: {len(documents)}')
    print(f'Categories: {len(categories)} -> {", ".join(categories)}')
    print(f'Unique Words: {len(words)} -> {", ".join(words)}')

    """

    return words, categories, documents
