import numpy as np
import pandas as pd
from model import DCNN
from utils import clean_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from pickle import dump

# Rename it if you don't want to overwrite existing model!
MODEL_PATH = "saved/sms_classification.hdf5"
TOKENIZER_PATH = "saved/sms_tokenizer"

save_model = True

with open("config.json", "rb") as f: config = json.load(f)

df = pd.read_csv("train.csv")
df["text"] = df["sms"].apply(lambda txt: clean_text(txt))
train_inputs, train_labels = df["text"], df["label"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_inputs)
train_inputs = tokenizer.texts_to_sequences(train_inputs)
MAX_LENGTH = int(max([len(txt) for txt in train_inputs]))
config["MAX_LENGTH"] = MAX_LENGTH
train_inputs = pad_sequences(train_inputs, maxlen=MAX_LENGTH, padding="post")
VOCAB_SIZE = len(tokenizer.word_index) + 1

dcnn = DCNN(vocab_size=VOCAB_SIZE, n_filters=config["N_FILTERS"], n_classes=2,
            embedd_dim=config["EMBEDD_DIM"], dropout=0.1,
            n_units=config["N_UNITS"], maxlen=MAX_LENGTH, training=True)

dcnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

dcnn.fit(train_inputs, train_labels, batch_size=config["BATCH_SIZE"], epochs=config["EPOCHS"])

if save_model:
    with open('config.json', 'w') as f: json.dump(config, f)
    dcnn.save_weights(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f: dump(tokenizer, f)
    print("Model saved.")
