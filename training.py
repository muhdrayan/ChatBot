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

# In this file we are going to create the Nueral Network Model for the Chat Bot


# I am going to use TensorFlow and TFLearn here
import tensorflow as tf
import numpy as np
import tflearn
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import pyttsx3

# Importing pickle for saving training dataset at the end
import pickle

# Importing my other file, nlp.py, so that we can use nlp() function
from nlp import nlp

# Getting the words, categories, and documents
words, categories, documents = nlp()

# Creating the input and output lists
training, output, output_empty = [], [], [0] * len(categories)

# Now we are going to make the training set
for document in documents:
    bag = [] # Our bag of words
    pattern_words = [stemmer.stem(word.lower()) for word in document[0]] # Retrieving the words and then stemming them

    # Adding 0s and 1s to the bag
    for word in words:
        if word in pattern_words: bag.append(1)
        else: bag.append(0)

    # Now we have set the output to 1 for current tag and 0 for other sets
    row_output = list(output_empty)
    row_output[categories.index(document[1])] = 1

    # Now we append to the training bag
    training.append([bag, row_output])

# Now that we created the training set, we can convert to numpy array after shuffling for randomness
random.shuffle(training)
training = np.array(training)

# Splitting the training set into features and labels
x_train = list(training[:,0])
y_train = list(training[:,1])

# Resetting the graph
tf.compat.v1.reset_default_graph()

# Nueral Network Input Layer (here length of x set is number of input nodes)
nueral = tflearn.input_data(shape=[None, len(x_train[0])])

# Making 2 hidden layers with 10 nodees
nueral = tflearn.fully_connected(nueral, 10)
nueral = tflearn.fully_connected(nueral, 10)

# Output layer
nueral = tflearn.fully_connected(nueral, len(y_train[0]), activation='softmax')
nueral = tflearn.regression(nueral)

# Making the model and tensorboard
model = tflearn.DNN(nueral, tensorboard_dir='tflearn_logs')

# Starting the training process
model.fit(x_train, y_train, n_epoch=1000, batch_size=8, show_metric=False)

# Saving the model
model.save('Models/model.tflearn')

# Pickling / Storing our data structures
pickle.dump( {'words':words, 'categories':categories, 'x_train':x_train, 'y_train':y_train}, open( "Data/training_data", "wb" ))

# Restoring all the data structures
data = pickle.load( open( "Data/training_data", "rb" ) )
words = data['words']
categories = data['categories']
x_train = data['x_train']
y_train = data['y_train']

with open('Data/intents.json') as f:
    intents = json.load(f)

# Loading the saved model
model.load('Models/model.tflearn')

def clean_up_sentence(sentence):
    # tokenizing the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stemming each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# returning bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    # generating bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

ERROR_THRESHOLD = 0.30
def classify(sentence, show_details=False):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words, show_details)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((categories[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence, show_details)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return random.choice(i['responses'])

            results.pop(0)

engine = pyttsx3.init()
while True:
    q = input("You: ")
    if q == '-1': exit()
    answer = response(q)
    print(f"Robo: {answer}")
    engine.say(answer)
    engine.runAndWait()
