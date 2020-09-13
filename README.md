This is a sentiment analysis on 1.6 million tweets from the year 2009, based on the Sentiment140 dataset and using Python 3.8.

## Formatting the Data

In order to streamline the tweets for effective tokenization, a number of things were removed from the data. They are listed below in no particular order:

- Non alphabets, 
- URLs, 
- Stopwords, and 
- Emojis

In this example, we have also lemmatized (grouping similar words) and stemmed (suffix trimming) the data but that option is up to the user.

## Vectorize & Tokenize

Next up, we run Word2Vec on the training data. This gives us a multi-dimensional vector map between the entire vocabulary set, clumping similar words closer to each other. This map is later used as the embedding layer in the LSTM model. 

To complete the preparation of the data, we tokenize the tweets into a computer readable format using the Tokenizer package within Keras with the maximum word length per tweet capped at 300 words.

## Models

We have used four different models in this situation, in order to be fully flexible and assess the strengths of all possible techniques available. These models include the LSTM, Bernoulli, Logistic Regression and Support Vector Classifier. Default parameters were used for nearly all models. A thing to note is that unlike the other three, the LSTM model returns values between 0 and 1. Therefore, during classification, a simple rounding is used to display the output consistently with the other models. The confusion matrices for all four are given below:

Upon observation, all the models seem to give decent results, with the best performance displayed by the LSTM and Support Vector Classifier. A simple way to increase performance in a Natural Language Processing (NLP) is to just take a larger corpus as the training set. 
