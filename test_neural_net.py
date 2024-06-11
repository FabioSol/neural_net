import numpy as np

from neural_net.models.sequential import Sequential
from neural_net.layers.dense import Dense
from neural_net.functions.activations.sigmoid import Sigmoid
import pandas as pd

data = pd.DataFrame({
    "x1": [0, 0, 0, 0, 1, 1],
    "x2": [0, 0, 1, 1, 0, 1],
    "x3": [0, 0, 1, 1, 0, 1],
    "x4": [0, 1, 0, 1, 0, 0],
    "d1": [0, 1, 1, 0, 1, 1],
    "d2": [0, 1, 1, 1, 0, 0]
})

X = data.loc[:, ["x1", "x2", "x3", "x4"]].values
Y = data.loc[:, ["d1", "d2"]].values

activation_fun =  Sigmoid()
np.random.seed(42)
model = Sequential([
    Dense(4, 15,activation_fun),
    Dense(15, 15,activation_fun),
    Dense(15, 2,activation_fun),
], lr=0.1, tol=1e-3)

model.fit(X, Y)
y_pred =np.round(model.predict(X))

print(all([i==e for i, e in zip(y_pred.flatten(),np.array(Y).flatten())]))
