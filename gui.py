# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# restore all of our data structures
import pickle

data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json

with open('intents.json') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))

# load our saved model
model.load('./model.tflearn')

# create a data structure to hold user context
context = {}
basket = []
helper = []
addBasket=False
events = [['dinner_krimi', 101], ['new_york', 31.95], ['glanz_auf_dem_vulkan', 37.25], ['future_palace', 17], ['goetz_widmann', 18.45], ['ehrlich_brothers', 45]]

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    global addBasket
    results = classify(sentence)
    if not results:
        results.append(('noanswer', 1.0000))
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if i['tag'] == 'basket' and not addBasket:
                            return "To add a event to your basket, first ask\n for more information about the event by entering\n the name of the event."
                        if i['tag'] == 'basket' and addBasket:
                            basket.extend(helper)
                            helper.clear()
                            addBasket=False
                        else:
                            addBasket = False
                            helper.clear()
                        for y in events:
                            if i['tag'] == y[0]:
                                helper.append(y[1])
                                addBasket = True
                        if i['tag'] == 'checkout':
                            total = 0
                            for li in basket:
                                total = total + li
                            sentence = "You have to pay a total of {} Euro"
                            return sentence.format(total)
                        # a random response from the intent
                        return random.choice(i['responses'])

            results.pop(0)


#Creating GUI with tkinter
import tkinter
from tkinter import *
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = response(msg)

        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("500x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
res = "##General Information##\n" \
          " Please type in lower case english \n" \
          " This Bot suggest events for you.\n" \
          " You can enter something like 'i would like to see a\n" \
          " concert' and it will find a concert for you.\n" \
          " To show more information please enter the name of\n" \
          " the event. You can then add one Ticket to your\n" \
          " basket by entering something like 'add to basket'.\n" \
          " There is no 'back' functionality. So if you want to\n" \
          " add another event you have to either enter the\n" \
          " name or you can browse by category."
ChatLog.insert(END, "Bot: " + res + '\n\n')
ChatLog.config(state=DISABLED)
ChatLog.yview(END)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="125", height="5", font="Arial")

#Place all components on the screen
scrollbar.place(x=476,y=6, height=386)
ChatLog.place(x=6,y=6, height=400, width=470)
EntryBox.place(x=128, y=401, height=90, width=347)
SendButton.place(x=6, y=401, height=90)

base.mainloop()