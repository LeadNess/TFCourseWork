import os
import requests
import json
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


NUM_WORKERS = 2
ip1 = '192.168.1.37'
ip2 = '192.168.1.72'
IP_ADDRS = [ip1, ip2]
port1 = 12345
port2 = 12345
PORTS = [port1, port2]
index = 1

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

choose = int(input('1. MultiWorkerMirroredStrategy\n2. ParameterServerStrategy\n'))

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

train_df = pd.read_csv(r'data\train_df.csb')
val_df = pd.read_csv(r'data\val_df.csv')

num_attrs = []
for i in [0, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 22, 31, 32]:
    num_attrs.append(features[i])


# Categorical features
cat_attrs = []
for i in [1, 2, 3]:
    cat_attrs.append(features[i])


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
    cat_encoder.fit(train_data[attr].values.reshape(-1, 1))
    train_df[attr] = cat_encoder.transform(train_df[attr].values.reshape(-1, 1))
    val_df[attr] = cat_encoder.transform(val_df[attr].values.reshape(-1, 1))
for attr in num_attrs:
    scaler.fit(train_data[attr].values.reshape(-1, 1))
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
    )

