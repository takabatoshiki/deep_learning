# 1) Fashion MNIST Data Download
from dlt.utils import plot_confusion_matrix
from dlt.utils import plot_prediction
from dlt.fashion_mnist import load_data

data = load_data()

from dlt.utils import plot_examples
plot_examples(data, fname="examples.png")

from dlt.utils import plot_distribution_data
plot_distribution_data(Y = data.train_labels, dataset_name = "y_train", classes = data.classes, fname= "y_prop.png")

# 2) Keras Data Preparation
input_size = 28 * 28
X_train = data.train_images.reshape(60000, input_size)
X_test = data.test_images.reshape(10000, input_size)

X_train = X_train/255.0
X_test = X_test/255.0


from keras.utils import to_categorical
y_train = to_categorical(data.train_labels)
y_test = to_categorical(data.test_labels)


# 3) Keras Architechture Design
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units = 256, activation = "relu", input_dim = input_size))
model.add(Dense(units = 64, activation = "relu"))
model.add(Dense(units = 10, activation = "softmax"))

model.summary()
model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics=["accuracy"])

# 4) Model Learning
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

res = model.fit(x = X_train, y = y_train, epochs = 100, batch_size = 500, validation_split = 0.1, verbose = 0)
y_pred = model.predict(X_test)
y_pred[0]

y_pred_classes = model.predict_classes(X_test)
y_pred_classes[0]

y_pred_classes, data.test_labels.reshape(10000)

import numpy as np
np.where(y_pred_classes != data.test_labels.reshape(10000))

plot_prediction(Yp=y_pred[0], X=data.test_images[0], y=data.test_labels[0],classes = data.classes, top_n = False, fname = "prediction_for_idx0.png")

plot_prediction(Yp=y_pred[12], X=data.test_images[12], y=data.test_labels[12],classes=data.classes, top_n=False, fname="prediction_for_idx2.png")

# 5) Model Evaluation
from dlt.utils import plot_loss_and_accuracy

plot_loss_and_accuracy(res, fname = "model_eval.png")
model.evaluate(X_test, y_test, batch_size=128)

plot_confusion_matrix(test_labels=data.test_labels, y_pred=y_pred_classes,classes = data.classes, title = "confusion matrix", fname = "confusion_matrix.png")
