import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import gensim
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

import time
import re 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import load_model

# =============================================================================
# Parameters
# =============================================================================
stem = True
lemma = True

# =============================================================================
# Data Import 
# =============================================================================
df = pd.read_csv(
    r"tweets.csv",
    encoding = "ISO-8859-1",
    names = ["sentiment", "ids", "date", "flag", "user", "text"]
)
df = df.drop(["ids", "date", "flag", "user"], axis = 1)
df['sentiment'] = df['sentiment'].replace(4,1)
df= df.sample(n=10000,random_state=1).reset_index(drop = True)

# =============================================================================
# Preprocess Text
# =============================================================================
stemmer = SnowballStemmer("english")
stop_words = stopwords.words("english")    
lemmatized = WordNetLemmatizer()

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat', ':))': 'smile','(:': 'smile'}

def preprocess(tweet, stem = stem, lemma = lemma):
    
    processed = []    
    tweet = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+",' ',tweet)
    tweet = tweet.lower()
    for emoji in emojis.keys():
        tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        
    for word in tweet.split():
        if word not in stop_words:
            if len(word)>1:
                if lemma and not stem:
                    word = lemmatized.lemmatize(word)
                    processed.append(word)
                if stem and not lemma:
                    processed.append(stemmer.stem(word))
                if lemma and stem:
                    word = lemmatized.lemmatize(word)
                    processed.append(stemmer.stem(word))
                if not lemma and not stem:
                    processed.append(word)
                        
    return " ".join(processed)


df1 = []

pd.DataFrame(columns = ['text'])
for i in df['text']:
    df1.append(preprocess(i))



df1 = pd.DataFrame(df1, columns=['text'])
df1['sentiment'] = df['sentiment']
df1 = df1[["sentiment", "text"]]


df1_train, df1_test = train_test_split(df1, test_size=0.1, random_state=1)

all_words = []
for line in df1_train['text']:
    all_words.append(line.split())

w2v_model = gensim.models.word2vec.Word2Vec(size=300, 
                                            window=7, 
                                            min_count=10, 
                                            workers=8)

w2v_model.build_vocab(all_words)

words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)

w2v_model.train(all_words, total_examples=len(all_words), epochs=32)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df1_train.text)

vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)

x_train = pad_sequences(tokenizer.texts_to_sequences(df1_train.text), maxlen=300)
x_test = pad_sequences(tokenizer.texts_to_sequences(df1_test.text), maxlen=300)

y_train = np.array(df1_train['sentiment']).reshape(-1,1)
y_test = np.array(df1_test['sentiment']).reshape(-1,1)

# =============================================================================
# # Confusion Matrix
# =============================================================================

def cfmat(model, model_name):
    
    y_pred = model.predict(x_test)
    
    print(classification_report(y_test, np.round(y_pred,0)))
    print(accuracy_score(y_test, np.round(y_pred,0)))
    cf_matrix = confusion_matrix(y_test, np.round(y_pred,0))

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title (f"Confusion Matrix: {model_name}", fontdict = {'size':18}, pad = 20)
    
    plt.figure()
    return 

# =============================================================================
# # Create LSTM Model
# =============================================================================

# w2v_model.wv[word] contains all the correlations between 
# different words with each other
# The embedding layer is essentially a map of correlations
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
  if word in w2v_model.wv:
    embedding_matrix[i] = w2v_model.wv[word]
    
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False)

# Run the model
LSTM_Model = Sequential()
LSTM_Model.add(embedding_layer)
LSTM_Model.add(Dropout(0.5))
LSTM_Model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
LSTM_Model.add(Dense(1, activation='sigmoid'))

LSTM_Model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

callbacks = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0
)

# Train LSTM Model
t = time.time()
LSTM_History = LSTM_Model.fit(x_train, y_train,
                    batch_size=10,
                    epochs=4,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)

print("Text Preprocessing complete.")
print(f"Time Taken: {round(time.time()-t)} seconds")

# Plot Accuracy and Loss graphs
acc = LSTM_History.history['accuracy']
val_acc = LSTM_History.history['val_accuracy']
loss = LSTM_History.history['loss']
val_loss = LSTM_History.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()

cfmat(LSTM_Model)

# =============================================================================
# # Create other models
# =============================================================================

BERNOULLI_Model = BernoulliNB(alpha = 1)
BERNOULLI_Model.fit(x_train, y_train)

SVC_Model = LinearSVC()
SVC_Model.fit(x_train, y_train)

LR_Model = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LR_Model.fit(x_train, y_train)

cfmat(BERNOULLI_Model, "Bernoulli")
cfmat(SVC_Model, "Support Vector Classifier")
cfmat(LR_Model, "Logistic Regression")
cfmat(LSTM_Model, "Logistic Regression")

def predict(model, text):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=300)
    # Predict
    score = model.predict(x_test)[0]
    # Decode sentiment
    if score >= 0.5:
        label = "Positive"
    else:
        label = "Negative"
    
    return {"label": label, 
            "score": float(score)}

# Predict using the LSTM Model
predict(LSTM_Model,"Happy time")

# Save Models
LSTM_Model.save("model.h5")
w2v_model.save("model.w2v")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"), protocol=0)
pickle.dump(LR_Model, open("LR_Model.pickle", "wb"), protocol=0)
pickle.dump(BERNOULLI_Model, open("BERNOULLI_Model.pickle", "wb"), protocol=0)
pickle.dump(SVC_Model, open("SVC_Model.pickle", "wb"), protocol=0)
