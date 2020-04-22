import os
import requests
import shutil
import json
import gzip
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


NUM_WORKERS = 2
ip1 = input('Enter ip 1: ')
ip2 = input('Enter ip 2: ')
IP_ADDRS = [ip1, ip2]
port1 = int(input('Enter port 1: '))
port2 = int(input('Enter port 2: '))
PORTS = [port1, port2]
index = int(input('Enter worker index: '))

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(NUM_WORKERS)],
        'ps': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(NUM_WORKERS)]
    },
    'task': {
        'type': 'worker',
        'index': index
    }
})

choose = input('Strategy:\n1. MultiWorkerMirroredStrategy\n2. ParameterServerStrategy')

if choose == 1:
	strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
else:
	strategy = tf.distribute.experimental.ParameterServerStrategy()


# Get features types dict and features list
content = requests.get('http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names').text
buf, *features = content.split('\n')[:-1]
attack_types = buf.split(',')
attack_types[-1] = attack_types[-1][:-1]
features_types_dict = {f.split(':')[0]: f.split(':')[1][1:-1] for f in features}
features = list(features_types_dict.keys())

# Get attack types dict
content = requests.get('http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types').text
buf = content.split('\n')[:-2]
target_classes = {
    'normal': 0,
    'u2r': 1,
    'r2l': 2,
    'probe': 3,
    'dos': 4
}
attack_types_dict = {line.split()[0]: line.split()[1] for line in buf}
attack_types_dict['normal'] = 'normal'

#Get data
_URL = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="kddcup.data.gz",
                                   extract=True)

# Extract zip file with data
data_file = 'kddcup_data'

with gzip.open(zip_file, 'rb') as f_in:
    with open(data_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load data into df
data = pd.read_csv(data_file, 
                      header=None, 
                      names=features + ['label'])
data['label'] = [i[:-1] for i in data['label'].values]

# Unnormalized numerical features
num_attrs = []
for i in [0, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 22, 31, 32]:
    num_attrs.append(features[i])

# Categorical features
cat_attrs = []
for i in [1, 2, 3]:
    cat_attrs.append(features[i])

# Detect small and numerous classes
numerous_classes = ['smurf', 'neptune', 'normal', 'satan', 'ipsweep', 'portsweep']
small_classes = ['nmap', 'back', 'warezclient', 'teardrop']

# Form training, test and validation dataframes
train_df, test_df, val_df = data[1:2], data[2:3], data[3:4]

TRAIN_NUM = 2000
TEST_NUM = 200

for cl in numerous_classes:
    train_df = train_df.merge(data[data['label']==cl][:TRAIN_NUM], how='outer')
    test_df = test_df.merge(data[data['label']==cl][TRAIN_NUM:TRAIN_NUM+TEST_NUM], how='outer')
    val_df = val_df.merge(data[data['label']==cl][TRAIN_NUM+TEST_NUM:TRAIN_NUM+TEST_NUM+TEST_NUM], how='outer')

for cl in small_classes:
    TRAIN_NUM = round(len(data[data['label']==cl]) * 0.8)
    TEST_NUM = round(len(data[data['label']==cl]) * 0.1)
    train_df = train_df.merge(data[data['label']==cl][:TRAIN_NUM], how='outer')
    test_df = test_df.merge(data[data['label']==cl][TRAIN_NUM:TRAIN_NUM+TEST_NUM], how='outer')
    val_df = val_df.merge(data[data['label']==cl][TRAIN_NUM+TEST_NUM:TRAIN_NUM+TEST_NUM+TEST_NUM], how='outer')
    
print('train_df: ', len(train_df))
print('test_df:  ', len(test_df)) 
print('val_df:   ', len(val_df))


train_data = train_df.copy()
cat_encoder = LabelBinarizer()
scaler = MinMaxScaler()
for attr in cat_attrs:
    train_data[attr] = cat_encoder.fit_transform(train_data[attr].values.reshape(-1, 1))
for attr in num_attrs:
    train_data[attr] = scaler.fit_transform(train_data[attr].values.reshape(-1, 1))


rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(train_data.drop('label', axis=1), train_data['label'])    

BATCH_SIZE = 32
BIAS = 0.02

bias=BIAS
batch_size=BATCH_SIZE

cat_encoder = LabelBinarizer()
scaler = MinMaxScaler()
for attr in cat_attrs:
    cat_encoder.fit(data[attr].values.reshape(-1, 1))
    train_df[attr] = cat_encoder.transform(train_df[attr].values.reshape(-1, 1))
    val_df[attr] = cat_encoder.transform(val_df[attr].values.reshape(-1, 1))
for attr in num_attrs:
    scaler.fit(data[attr].values.reshape(-1, 1))
    train_df[attr] = scaler.fit_transform(train_df[attr].values.reshape(-1, 1))
    val_df[attr] = scaler.fit_transform(val_df[attr].values.reshape(-1, 1))
            
d = dict(zip(features, rnd_clf.feature_importances_)) # dict(feature: feature_importance)
train_df = train_df.drop(list(filter(lambda x: d[x] < bias, d)), axis=1) # drop unimportance features

train_df['label'] = train_df['label'].apply(lambda x: target_classes[attack_types_dict[x]]) # 10 classes > 4 main classes
train_df['label'], _ = train_df['label'].factorize()
        
train_labels = train_df.pop('label')
train_dataset = tf.data.Dataset.from_tensor_slices((train_df.values, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(train_df))
train_dataset = train_dataset.batch(batch_size)

val_df = val_df.drop(list(filter(lambda x: d[x] < bias, d)), axis=1) # drop unimportance features

val_df['label'] = val_df['label'].apply(lambda x: target_classes[attack_types_dict[x]]) # 10 classes > 4 main classes
val_df['label'], _ = val_df['label'].factorize()
        
val_labels = val_df.pop('label')
val_dataset = tf.data.Dataset.from_tensor_slices((val_df.values, val_labels))
val_dataset = val_dataset.shuffle(buffer_size=len(val_df))
val_dataset = val_dataset.batch(batch_size)


with strategy.scope():
    # Compile and fit model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(21, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    epochs = 30
    history = model.fit(train_dataset, 
        validation_data=val_dataset, 
        use_multiprocessing=True, 
        epochs=epochs,
        callbacks=[tensorboard_callback]
    )
