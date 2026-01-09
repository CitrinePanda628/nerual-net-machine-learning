import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from sklearn.metrics import classification_report

print("main.py")

TF_ENABLE_ONEDNN_OPTS=0

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic+gamma+telescope/magic04.data", names=cols)

df["class"] = (df["class"] == "g").astype(int)

#print(df.head())

# for label in cols[:-1]:
#     plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
#     plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
#     plt.title(label)
#     plt.ylabel("Probability")
#     plt.xlabel("label")
#     plt.legend()
#     plt.show()

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


# print((df["class"] == 1).sum())
# print((df["class"] == 0).sum())

def scale_data( dataframe, oversample = False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y,(-1,1))))

    return data, X, y


train, X_train, y_train = scale_data(train, oversample=True)
valid, X_valid, y_valid = scale_data(valid, oversample=False)
test, X_test, y_test = scale_data(test, oversample=False)

# print((y_train == 1).sum())
# print((y_train == 0).sum())

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle('Training History')
    plt.show()

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),            tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
    history = nn_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1
        )

    return nn_model, history
    
least_val_loss = float('inf')
least_loss_model = None

model, history = train_model(X_train, y_train, 32, 0.2, 0.001, 32, 20)
plot_history(history)
val_loss = model.evaluate(X_valid, y_valid)[0]
if val_loss < least_val_loss:
    least_val_loss = val_loss
    least_loss_model = model

y_pred = least_loss_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

model.save("nn_model.keras")

model.save_weights("nn_weights.weights.h5")

json_string = model.to_json()

json_string = model.to_json()
with open("nn.json", "w") as json_file:
    json_file.write(json_string)

print(model.summary())

print(tf.config.list_physical_devices('GPU'))
print(classification_report(y_test, y_pred))    