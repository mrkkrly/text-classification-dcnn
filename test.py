import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import DCNN
from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text

MODEL_PATH = "saved/sms_classification.hdf5"
TOKENIZER_PATH = "saved/sms_tokenizer"
with open(TOKENIZER_PATH, "rb") as f: tokenizer = load(f)
with open("config.json", "rb") as f: config = json.load(f)
VOCAB_SIZE = len(tokenizer.word_index) + 1

df = pd.read_csv("test.csv")
df["text"] = df["sms"].apply(lambda txt: clean_text(txt))
test_inputs, test_labels = df["text"], df["label"]
test_inputs = pad_sequences(tokenizer.texts_to_sequences(test_inputs), maxlen=config["MAX_LENGTH"], padding="post")

dcnn = DCNN(vocab_size=VOCAB_SIZE, n_filters=config["N_FILTERS"], n_classes=2,
            embedd_dim=config["EMBEDD_DIM"], dropout=0.1,
            n_units=config["N_UNITS"], maxlen=config["MAX_LENGTH"], training=False)

dcnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
_ = dcnn(test_inputs)
dcnn.load_weights(MODEL_PATH)
predictions = np.round(dcnn(test_inputs))
print(classification_report(test_labels, predictions))
