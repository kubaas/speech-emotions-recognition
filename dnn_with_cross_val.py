import os

from keras.utils import plot_model

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import seaborn as sn
import tensorflow as tf


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# define 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# FILES

# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f1_eGemaps.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f1_COMPARE.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f10eGemaps.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f10COMPARE.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/TESS_sep_speaker/f1_eGemaps.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS_sep_speaker/f1_Compare.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f2.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f3.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f1_ANHS.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f3_ANHS.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f1_equal_SMOTE.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f3_equal_SMOTE.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f30.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f31.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f10.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f11.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f10_SMOTE.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f11_SMOTE.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f1_reduced_CfsSub_LFS.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f1_reduced_InfoGain_Ranker.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f1_reduced_Chi_Rank.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f1_reduced_PCA_Ranker.csv')


# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f3_reduced_Cfs_LFS.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f3_reduced_InfoG_Rank.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f3_reduced_Chi_Rank.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f10_reduced_Cfs_LFS.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f10_reduced_InfoG_Rank.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f10_reduced_Chi_Rank.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f10_reduced_PCA_Rank.csv')


# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f11_reduced_Chi_Rank.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f11_reduced_InfoG_Rank.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f11_reduced_Cfs_LFS.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f10_ANHS.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/RAVDESS_ONLY_SPEECH??/f11_ANHS.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f30.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f30_reduced_Cfs_LFS.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f30_reduced_Chi_Rank.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f30_reduced_InfoG_Rank.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f30_reduced_PCA_Rank.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f31.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f31_reduced_Cfs_LFS.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f31_reduced_Chi_Rank.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f31_reduced_InfoG_Rank.csv')

# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f30_ANHS.csv')
# data = pd.read_csv('/home/jakub/Dokumenty/TESS/f31_ANHS.csv')


############ Predicitons

# data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB_Norm/f11.csv')
data = pd.read_csv('/home/jakub/Dokumenty/EMO_DB/f1_noNorm.csv')

model_name = 'EmoDB_f11_NORM'

y = data.emotion
X = data.drop('emotion', axis=1)

n_cols = X.shape[1]
cvscores = []
cvscores2 = []
for train, test in kfold.split(X, y):
    # n_neurons = 256
    n_neurons = 2048

    # n_epochs = 500
     n_epochs = 80


    X_train = X.iloc[train]
    X_test = X.iloc[test]

    y[train], y[test] = np.array(y[train]), np.array(y[test])
    label_encoder = LabelEncoder()
    integer_encoded1, integer_encoded2 = label_encoder.fit_transform(y[train]), label_encoder.fit_transform(y[test])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded1 = integer_encoded1.reshape(len(integer_encoded1), 1)
    integer_encoded2 = integer_encoded2.reshape(len(integer_encoded2), 1)
    y_train, y_test = onehot_encoder.fit_transform(integer_encoded1), onehot_encoder.fit_transform(integer_encoded2)


    model = Sequential()
    model.add(Dropout(0.5, input_shape=(n_cols,)))
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    opt = keras.optimizers.SGD(learning_rate=0.01, nesterov=True, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    history = model.fit(X_train, y_train, batch_size=20, epochs=n_epochs, validation_data=(X_test, y_test))
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)

    cvscores.append(scores[1] * 100)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

print(cvscores)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

save_dir = os.path.join(os.getcwd(), 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name + '.h5')
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# import json
model_json = model.to_json()
with open(model_name + '.json', "w") as json_file:
    json_file.write(model_json)