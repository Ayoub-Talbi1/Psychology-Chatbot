import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.legacy import SGD as LegacySGD

nltk.download('punkt')  # Download the 'punkt' tokenizer data if not already downloaded

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))  # set eliminates duplicates
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))  # Corrected the variable name

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Ensure that bag has a consistent length equal to the number of unique words
    bag = [1 if word in word_patterns else 0 for word in words]
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

max_bag_length = max(len(row[0]) for row in training)
max_output_length = len(training[0][1])  # Output should always have the same length

# Zero-pad the bags and output_rows to have the same maximum length
bags = []
outputs = []

# Iterate through training data and separate bags and outputs
for item in training:
    bags.append(item[0])
    outputs.append(item[1])

# Convert bags and outputs to NumPy arrays
train_x = np.array(bags)
train_y = np.array(outputs)


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = LegacySGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)  # Corrected 'VERBOSE' to 'verbose'
model.save('chatbotmodel.h5')
print('DONE!')
