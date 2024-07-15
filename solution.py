import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, Dense

from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
nltk.download('stopwords')
nltk.download('punkt')

# Load file
df= pd.read_csv(r"amazon_reviews.csv")
df.head()

def show_df_info(df):
  print("Data Inforamation:\n")
  df.info()
  # Count the occurrences of each value in the 'sentiments' column
  value_counts = df['sentiments'].value_counts()
  print("\nSentiments Value Counts:\n")
  print(value_counts)

show_df_info(df)

def preprocess_df(df):
  # Drop cleaned_review_length and review_score
  df = df.drop(['cleaned_review_length', 'review_score'],axis=1)

  # Drop rows with missing values
  df = df.dropna()

  # Convert sentiments to numerical values
  df['sentiments'] = df['sentiments'].map({ 'negative': 0, 'neutral':1, 'positive': 2})

  return df

df = preprocess_df(df)

def preprocess_review(review):

    # Convert to lowercase
    review = review.lower()

    # Tokenize
    tokens = word_tokenize(review)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove empty tokens
    tokens = [token for token in tokens if token != '']

    # Join the tokens with spaces in between to form a single processed text string
    processed_review= ' '.join(tokens)

    return processed_review

df['cleaned_review'].apply(preprocess_review)

# Show data after preperations
show_df_info(df)

def spliting_dataset(df,test_size,random_state):
  # Splitting the dataset

  x=df['cleaned_review']
  y=df['sentiments']
  # Splitting the dataset into the Training set and Test set
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-test_size, random_state=random_state)
  return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = spliting_dataset(df,0.8,42)

def build_tokenizer(x_train,max_words):
  tokenizer = Tokenizer(num_words=max_words)
  tokenizer.fit_on_texts(x_train)
  return tokenizer

tokenizer = build_tokenizer(x_train,1000)

def word_embedding(x_train,x_test,max_words,max_len):
    # Convert the text data to sequences of integers
    tokenizer = build_tokenizer(x_train,max_words)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    # Pad the sequences to a fixed length
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)
    return x_train, x_test

x_train,x_test = word_embedding(x_train,x_test,1000,100)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

def simple_rnn(x_train, y_train, x_test, y_test, max_words, max_len, epochs, print_flag):
  # Define SimpleRNN model
  simple_rnn_model = Sequential([
      Embedding(max_words, 64, input_length=max_len),
      SimpleRNN(32),
      Dense(3, activation='softmax')
  ])

  # Compile the model
  simple_rnn_model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

  # Train the model
  simple_rnn_model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

  # Evaluate the model on testing data
  loss, accuracy = simple_rnn_model.evaluate(x_test, y_test)
  if print_flag:
    print("\nSimpleRNN Model Accuracy:", accuracy*100,"%")

  return simple_rnn_model

simple_rnn_model = simple_rnn(x_train, y_train, x_test, y_test, 1000, 100, 10, True)

def lstm(x_train, y_train, x_test, y_test, max_words, max_len, epochs, print_flag):
  # Define LSTM model
  lstm_model = Sequential([
      Embedding(max_words, 64, input_length=max_len),
      LSTM(32),
      Dense(3, activation='softmax')
  ])

  # Compile the model
  lstm_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

  # Train the model
  lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

  # Evaluate the model on testing data
  loss, accuracy = lstm_model.evaluate(x_test, y_test)
  if print_flag:
    print("LSTM Model Accuracy:", accuracy*100,"%")
  return lstm_model

lstm_model = lstm(x_train, y_train, x_test, y_test, 1000, 100, 10, True)

def predict_embedding_review(review, max_words, max_len):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

def predict(review, model):
    preprocessed_review = preprocess_review(review)
    embedding_review = predict_embedding_review(preprocessed_review,1000,100)
    prediction_value = model.predict(embedding_review)
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_class = sentiment_mapping[np.argmax(prediction_value)]
    return predicted_class

def predict_input():
  review = input("Enter your review: ")
  predicted_class_lstm = predict(review, lstm_model)
  predicted_class_simple_rnn = predict(review, simple_rnn_model)
  print("Predicted sentiment for the input using LSTM Model: ", predicted_class_lstm)
  print("Predicted sentiment for the input using Simple RNN Model: ", predicted_class_simple_rnn)

predict_input()

# Define parameters for experimentation
params = {
    'splitting_ratio': [0.6, 0.7, 0.9],
    'sequence_padding_length': [500, 1000],
    'epochs': [5, 10],
}

def train_with_different_param(params):
  # Dictionary to store accuracy for each parameter combination
  accuracy_results = {}
  for ratio in params['splitting_ratio']:
      for length in params['sequence_padding_length']:
          for epoch in params['epochs']:
              x_train, x_test, y_train, y_test= spliting_dataset(df=df,random_state=42, test_size=1-ratio)

              x_train, x_test = word_embedding(x_train=x_train,x_test=x_test,max_words=length,max_len=100)

              simple_rnn_model = simple_rnn(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, max_words=length, max_len=100, epochs=epoch,print_flag= False)
              simple_rnn_loss, simple_rnn_accuracy = simple_rnn_model.evaluate(x_test, y_test)

              lstm_model = lstm(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, max_words=length, max_len=100, epochs=epoch, print_flag=False)
              lstm_loss, lstm_accuracy = lstm_model.evaluate(x_test, y_test)

              accuracy_results[(ratio,length,epoch)] = ( simple_rnn_accuracy*100,lstm_accuracy*100)

  return accuracy_results


# Train models with different parameters
accuracy_results = train_with_different_param(params)

pip install tabulate

from tabulate import tabulate

# Function to display model summaries and accuracies

def show_model_summary(accuracy_results):
    _simple_rnn = []
    _lstm = []
    _ratio = []
    _length = []
    _epoch = []
    for (ratio, length, epoch), (simple_rnn, lstm) in accuracy_results.items():
        _simple_rnn.append(simple_rnn)
        _lstm.append(lstm)
        _ratio.append(ratio)
        _length.append(length)
        _epoch.append(epoch)

    summary_df = pd.DataFrame({
        "Splitting Ratio": _ratio,
        'Sequence Padding Length': _length,
        "Epoch Size": _epoch,
        'SimpleRNN': _simple_rnn,
        'LSTM': _lstm
    })

    # Find the best row accuracy for SimpleRNN and LSTM
    best_simple_rnn_accuracy = summary_df['SimpleRNN'].max()
    best_lstm_accuracy = summary_df['LSTM'].max()

    # Add a column to indicate if the row has the best accuracy for each model
    summary_df['Best SimpleRNN'] = ['Best' if acc == best_simple_rnn_accuracy else '____' for acc in summary_df['SimpleRNN']]
    summary_df['Best LSTM'] = ['Best' if acc == best_lstm_accuracy else '____' for acc in summary_df['LSTM']]

    print("Model Summary:")
    print(tabulate(summary_df, headers='keys', tablefmt='pretty'))
# Display model summaries and accuracies in DataFrame
show_model_summary(accuracy_results)

