!#/apps/anaconda3/bin/python

# Loading packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
from numpy import asarray
from numpy import savetxt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Helper functions
def replace_contraction(text):
    contraction_patterns = [(r'won\'t', 'will not'),
                            (r'can\'t', 'can not'),
                            (r'i\'m', 'i am'),
                            (r'ain\'t', 'is not'),
                            (r'(\w+)\'ll', '\g<1> will'),
                            (r'(\w+)n\'t', '\g<1> not'),
                            (r'(\w+)\'ve', '\g<1> have'),
                            (r'(\w+)\'s', '\g<1> is'),
                            (r'(\w+)\'re', '\g<1> are'),
                            (r'(\w+)\'d', '\g<1> would'),
                            (r'&', 'and'),
                            (r'dammit', 'damn it'),
                            (r'dont', 'do not'),
                            (r'wont', 'will not')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text


def replace_links(text, filler=' '):
        text = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
                      filler, text).strip()
        return text

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def str_len(text):
    return len(text.split())

def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = replace_contraction(text)
    text = replace_links(text, "link")
    text = remove_numbers(text)
    text = re.sub(r'[,!@#$%^&*)(|/><";:.?\'\\}{]',"",text)
    text = text.lower()
    return text

def loading_training_set(df_train_path, subset = 0.1, balanced = False):
    # Loading and processing the sentence-level dataset
    df_train = pd.read_csv(df_train_path)
    df_train_dyspnea = df_train[['Note', 'Dyspnea (# of simclins)']]
    df_train_dyspnea['Dyspnea (# of simclins)'] = df_train_dyspnea['Dyspnea (# of simclins)'].fillna(0.0)
    df_train_dyspnea['dyspnea'] = np.where(df_train_dyspnea['Dyspnea (# of simclins)'] > 0.0, 1, 0)
    df_train_dyspnea = df_train_dyspnea[['Note', 'dyspnea']].reset_index()
    df_train_dyspnea = df_train_dyspnea.drop('index', axis = 1)
    # Remove rows where 'Note' is empty
    df_train_dyspnea = df_train_dyspnea[pd.notnull(df_train_dyspnea['Note'])]
    df_train_dyspnea['sent_len'] = df_train_dyspnea['Note'].apply(str_len)
    # Clip the length of 'Note' to 35 words max.
    df_train_dyspnea = df_train_dyspnea[df_train_dyspnea['sent_len'] < 35]
    # Subset
    df_train_dyspnea = df_train_dyspnea.sample(frac = subset, random_state = 2019)
    if balanced:
    # Balance the training set.
        df_pos = df_train_dyspnea[df_train_dyspnea['dyspnea'] == 1]
        df_neg = df_train_dyspnea[df_train_dyspnea['dyspnea'] == 0].sample(n = df_pos.shape[0], random_state = 2019)
        df_train_dyspnea = pd.concat([df_pos, df_neg])
        df_train_dyspnea = df_train_dyspnea.reset_index()
        df_train_dyspnea.drop('index', inplace = True, axis = 1)
    # Final processing
    df_train_dyspnea['Note'] = df_train_dyspnea['Note'].apply(cleanText)
    df_train_dyspnea['Note'] = df_train_dyspnea['Note'].str.replace('\s+', ' ', regex = True)
    return df_train_dyspnea

def loading_test_set(df_test_path, balanced = False):
    # Loading and processing the sentence-level dataset
    df = pd.read_csv(df_test_path)
    for i in range(1, 5):
        df['dyspnea_' + str(i)] = np.where(df['Category ' + str(i)] == 'Dyspnea', 1, 0)
    df = df[['Note', 'dyspnea_1', 'dyspnea_2', 'dyspnea_3', 'dyspnea_4']]
    df['dyspnea'] = df[['dyspnea_1', 'dyspnea_2', 'dyspnea_3', 'dyspnea_4']].sum(axis = 1)
    df['dyspnea'] = np.where(df['dyspnea'] > 0, 1, 0)
    df = df[['Note', 'dyspnea']]
    # Remove rows where 'Note' is empty
    df = df[pd.notnull(df['Note'])]
    df['sent_len'] = df['Note'].apply(str_len)
    # Clip the length of 'Note' to 35 words max.
    df = df[df['sent_len'] < 35]
    if balanced:
        df_pos = df[df['dyspnea'] == 1]
        df_neg = df[df['dyspnea'] == 0].sample(n = df_pos.shape[0], random_state = 2019)
        df = pd.concat([df_pos, df_neg])
        df = df.reset_index()
        df.drop('index', inplace = True, axis = 1)
    df['Note'] = df['Note'].apply(cleanText)
    df['Note'] = df['Note'].str.replace('\s+', ' ', regex = True)
    return df

def train_elmo(elmo_module_path, elmo_model_path, epochs=20, batch_size=256):
    embed = hub.Module(elmo_module_path)
    def ELMoEmbedding(x):
        return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
    def build_model():
        input_text = Input(shape=(1,), dtype="string")
        embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
        dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(embedding)
        pred = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[input_text], outputs=pred)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model
    model_elmo_dyspnea = build_model()
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        history = model_elmo_dyspnea.fit(X_train, y_train,
                                         epochs=epochs,
                                         batch_size=batch_size,
                                         validation_split = 0.2)
        model_elmo_dyspnea.save_weights(elmo_model_path)
    return history

def plot_training(history, model_plot_path):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize = (6, 6))
    plt.plot(epochs, acc, 'g', label='Training Acc')
    plt.plot(epochs, val_acc, 'b', label='Validation Acc')
    plt.title('Training and validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(model_plot_path, bbox_inches = 'tight', dpi = 500)

def run_prediction(df_test, elmo_model_path, elmo_predict_path):
    df_test_text = df_test['Note'].to_list()
    test_text_to_pred = np.array(df_test_text, dtype=object)[:, np.newaxis]
    embed = hub.Module(elmo_module_path)
    def ELMoEmbedding(x):
        return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
    def build_model():
        input_text = Input(shape=(1,), dtype="string")
        embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
        dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(embedding)
        pred = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[input_text], outputs=pred)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model
    model_elmo_dyspnea = build_model()
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        model_elmo_dyspnea.load_weights(elmo_model_path)
        predicts = model_elmo_dyspnea.predict(test_text_to_pred)
        savetxt(elmo_predict_path, predicts, delimiter=',')
    return predicts

def get_confusion_matrix(predicts, df_test):
    predicts = np.array(predicts)
    predicts = np.where(predicts > 0.5, 1, 0)
    y_test = np.array(df_test['dyspnea'])
    return confusion_matrix(y_test, predicts), f1_score(y_test, predicts, average = 'micro')

if __name__=="__main__":
    # Directories
    os.chdir("/work/han2114")
    df_train_path = os.getcwd() + "/labeled-data-2019-07-18_14-22.csv"
    df_test_path = os.getcwd() + "/gold_standard_HF_150.csv"
    elmo_module_path = os.getcwd() + "/module/module_elmo3"
    elmo_model_path_unbalanced = os.getcwd() + "/model_elmo_weights_dyspnea_sentences_unbalanced.h5"
    model_plot_path_unbalanced = os.getcwd() + "/model_elmo_weights_dyspnea_sentences_unbalanced.png"
    elmo_predict_path_unbalanced = os.getcwd() + "/predicts_unbalanced.csv"
    # Prepare training set
    df_train = loading_training_set(df_train_path, subset = 0.1)
    X_train = np.array(df_train["Note"])
    y_train = np.array(df_train["dyspnea"])
    # Prepare test set
    df_test = loading_test_set(df_test_path)
    X_test = np.array(df_test["Note"])
    y_test = np.array(df_test["dyspnea"])
    # Training Elmo
    history = train_elmo(elmo_module_path, elmo_model_path_unbalanced, epochs=5)
    plot_training(history, model_plot_path_unbalanced)
    # Running predictions
    predicts = run_prediction(df_test, elmo_model_path_unbalanced, elmo_predict_path_unbalanced)
    get_confusion_matrix(predicts, df_test)
