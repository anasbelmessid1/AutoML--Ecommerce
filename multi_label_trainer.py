#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sayakpaul/A-B-testing-with-Machine-Learning/blob/master/multi_label_trainer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Imports

# In[ ]:


from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from ast import literal_eval
import pandas as pd


# ## Read data and perform basic EDA

# In[ ]:


arxiv_data = pd.read_csv(
    "https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv"
)
arxiv_data.head()


# In[ ]:


print(f"There are {len(arxiv_data)} rows in the dataset.")


# In[ ]:


total_duplicate_titles = sum(arxiv_data["titles"].duplicated())
print(f"There are {total_duplicate_titles} duplicate titles.")


# In[ ]:


arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]
print(f"There are {len(arxiv_data)} rows in the deduplicated dataset.")


# In[ ]:


# There are some terms with occurence as low as 1.
sum(arxiv_data["terms"].value_counts() == 1)


# In[ ]:


# How many unique terms?
arxiv_data["terms"].nunique()


# In[ ]:


# Filtering the rare terms.
arxiv_data_filtered = arxiv_data.groupby("terms").filter(lambda x: len(x) > 1)
arxiv_data_filtered.shape


# ## Convert the string labels to list of strings. 
# 
# The initial labels are represented as raw strings. Here we make them `List[str]` for a more compact representation. 

# In[ ]:


arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(
    lambda x: literal_eval(x)
)
arxiv_data_filtered["terms"].values[:5]


# ## Stratified splits because of class imbalance

# In[ ]:


test_split = 0.1

# Initial train and test split.
train_df, test_df = train_test_split(
    arxiv_data_filtered,
    test_size=test_split,
    stratify=arxiv_data_filtered["terms"].values,
)

# Splitting the test set further into validation
# and new test sets.
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)

print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")


# ## Multi-label binarization

# In[ ]:


mlb = MultiLabelBinarizer()
mlb.fit_transform(train_df["terms"])
mlb.classes_


# ## Data preprocessing and `tf.data.Dataset` objects
# 
# Get percentile estimates of the sequence lengths. 

# In[ ]:


train_df["summaries"].apply(lambda x: len(x.split(" "))).describe()


# Notice that 50% of the abstracts have a length of 158. So, any number near that is a good enough approximate for the maximum sequence length. 

# In[ ]:


max_seqlen = 150
batch_size = 128


def unify_text_length(text, label):
    unified_text = tf.strings.substr(text, 0, max_seqlen)
    return tf.expand_dims(unified_text, -1), label


def make_dataset(dataframe, train=True):
    label_binarized = mlb.transform(dataframe["terms"].values)
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["summaries"].values, label_binarized)
    )
    if train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(unify_text_length).cache()
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# In[ ]:


train_dataset = make_dataset(train_df)
validation_dataset = make_dataset(val_df, False)
test_dataset = make_dataset(test_df, False)


# ## Dataset preview

# In[ ]:


text_batch, label_batch = next(iter(train_dataset))

for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text[0]}")
    print(f"Label(s): {mlb.inverse_transform(label)[0]}")
    print(" ")


# ## Vocabulary size for vectorization

# In[ ]:


train_df["total_words"] = train_df["summaries"].str.split().str.len()
vocabulary_size = train_df["total_words"].max()
print(f"Vocabulary size: {vocabulary_size}")


# ## Create model with `TextVectorization`

# In[ ]:


text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
)

with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))


def make_model():
    shallow_mlp_model = keras.Sequential(
        [
            keras.Input(shape=(), dtype=tf.string),
            text_vectorizer,
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(len(mlb.classes_), activation="softmax"),
        ]
    )
    return shallow_mlp_model


# With the CPU placement, we run into: 
# 
# ```
# (1) Invalid argument: During Variant Host->Device Copy: non-DMA-copy attempted of tensor type: string
# ```

# In[ ]:


shallow_mlp_model = make_model()
shallow_mlp_model.summary()


# ## Train the model

# In[ ]:


epochs = 20

shallow_mlp_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
)

shallow_mlp_model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)


# ## Evaluate the model

# In[ ]:


_, categorical_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")

