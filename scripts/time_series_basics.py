from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


data = [(1,28),  (2,17),  (3,92),  (4,41),  (5,9),   (6,87),  (7,54), (8,3),   (9,78),  (10,67),
        (11,1),  (12,67), (13,78), (14,3), (15,55), (16,86), (17,8),  (18,42), (19,92), (20,17),
        (21,29), (22,94), (23,28), (24,18), (25,93), (26,40), (27,9),  (28,87), (29,53), (30,3),
        (31,79), (32,66), (33,1),  (34,68), (35,77), (36,3),  (37,56), (38,86), (39,8),  (40,43),
        (41,92), (42,16), (43,30), (44,94), (45,27), (46,19), (47,93), (48,39), (49,10), (50,88),
        (51,53), (52,4),  (53,80), (54,65), (55,1),  (56,69), (57,77), (58,3),  (59,57), (60,86)]

features = []
labels = []

from matplotlib.pyplot import figure

figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')

for i in data:
    features.append(i[0])
    labels.append(i[1])

plot = plt.plot(features, labels)
plt.locator_params(axis='x', nbins=120)
plt.show()

import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(labels, freq=22)

figure(num=None, figsize=(13, 8), dpi=80, facecolor='w', edgecolor='k')
fig = res.plot()
fig.show()


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X, y = split_sequence(labels[:-20], 3)

X = X.reshape((X.shape[0], X.shape[1], 1))
print(X)
X_t, y_t = split_sequence(labels[-21:-10], 3)
X_t = X_t.reshape((X_t.shape[0], X_t.shape[1], 1))
print(X_t)

X_test, y_test = split_sequence(labels[-11:], 3)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_test = np.array(X_test, dtype=float)
print(X_test)
model = keras.Sequential()
model.add(keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(3,1)))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(30, activation='relu'))
model.add(keras.layers.Dense(1))

print(model.summary())
model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, epochs=200, batch_size=4, validation_data = (X_t, y_t), verbose=1)

pred = model.predict(X_test)
print(np.array(pred, dtype=int))
print(y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

