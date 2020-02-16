import tensorflow as tf
import numpy as np
import pandas as pd
import requests
from datetime import datetime

try:
	answer = int(input('1. All data (~ 750 MB)\n2. 10 percent of data'))
	if answer != 1 or answer != 2:
		exit(1)
except:
	exit(1)

if answer == 1:
	data_url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names'
else:
	data_url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'

attack_types, *features = requests.get(data_url).text.split('\n')[:-1]
attack_types = attack_types.split(',')
attack_types[-1] = attack_types[-1][:-1]
features_types_dict = {f.split(':')[0]: f.split(':')[1][1:-1] for f in features}
features = list(features_types_dict.keys())
features_types_dict['dst_host_srv_rerror_rate'] = 'continuous'


buf = requests.get('http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types').text
buf = buf.split('\n')[:-2]
target_classes = {
    'normal': 0,
    'u2r': 1,
    'r2l': 2,
    'probe': 3,
    'dos': 4
}
attack_types_dict = {line.split()[0]: line.split()[1] for line in buf}
attack_types_dict['normal'] = 'normal'
attack_types_dict

url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
df = pd.read_csv(url, header=None)

df[41] = [i[:-1] for i in df[41].values]

for i in (1, 2, 3):
    classes_list = list(dict(df[i].value_counts()).keys())
    classes = {c: i for i, c in enumerate(classes_list)}
    df[i] = df[i].apply(lambda x: classes[x])
df[41] = df[41].apply(lambda x: target_classes[attack_types_dict[x]])

for i in (0, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 22, 31, 32):
    max_ = df[i].max()
    df[i] = df[i].apply(lambda x: x / max_)

10_percent_num = len(df) / 10

df = df.sample(frac=1)
test_df = df[:10_percent_num]
val_df = df[10_percent_num:10_percent_num * 2]
df = df[10_percent_num * 2:]

target = df.pop(41)
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

val_target = val_df.pop(41)
val_dataset = tf.data.Dataset.from_tensor_slices((val_df.values, val_target.values))

test_target = test_df.pop(41)
test_dataset = tf.data.Dataset.from_tensor_slices((test_df.values, test_target.values))

BATCH_SIZE = 64

train_dataset = dataset.shuffle(len(df)).batch(BATCH_SIZE)
val_dataset = val_dataset.shuffle(len(val_df)).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_df)).batch(BATCH_SIZE)

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(41, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

model = get_compiled_model()
epochs = 40
history = model.fit_generator(
    train_dataset,
    steps_per_epoch=int(np.ceil(len(df) / (epochs * float(BATCH_SIZE)))) - 2,
    epochs=epochs,
    validation_data=val_dataset,
    validation_steps=int(np.ceil(len(val_df) / (epochs * float(BATCH_SIZE)))) - 2
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print('Accuracy on test dataset:', test_accuracy)