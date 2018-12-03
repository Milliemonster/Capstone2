import numpy as np
np.random.seed(42)
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from image_process_cs2 import data_preprocess
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

def create_model():
    nb_classes = 3
    img_rows, img_cols = 60, 60
    nb_filters = 24
    pool_size = (3, 3)
    input_shape = (60, 60, 3)

    model = Sequential()

    model.add(Conv2D(nb_filters, (4, 4),padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (4, 4), padding='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.3)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Conv2D(nb_filters, (1, 5), padding='valid')) #2nd conv. layer (keep layer)
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (5, 1), padding='valid')) #2nd conv. layer (keep layer)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.3)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten())
    print('Model flattened out to ', model.output_shape)

    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

def generate_data(train_directory, validation_directory, test_directory):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True
        )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )

    train_generator = train_datagen.flow_from_directory(
        directory=train_directory,
        #save_to_dir = "japanese_beetle/train",
        target_size=(60, 60),
        color_mode="rgb",
        batch_size=16,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=validation_directory,
        target_size=(60, 60),
        color_mode="rgb",
        batch_size=138,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_directory,
        target_size=(60, 60),
        color_mode="rgb",
        batch_size=65, #number that divides test set evenly
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    return train_generator, test_generator, validation_generator

def make_analysis(generator):
    test_X = generator[0][0]
    test_y = generator.classes

    predicted_y = model.predict_classes(test_X)
    probs = model.predict_proba(test_X).round(2)

    labels = np.vstack((test_y, predicted_y))
    results = np.hstack((probs, labels.T))

    classes = {0:'cucumber beetle' , 1: 'Japanese beetle', 2: 'ladybug'}

    score = balanced_accuracy_score(test_y, predicted_y)
    print(f'balanced accuracy score is {score}')

    wrong_indices = []

    for i, prediction in enumerate(predicted_y):
        if prediction != test_y[i]:
            wrong_indices.append(i)

    for index in wrong_indices:
        plt.imshow(test_X[index])
        plt.text(0.05, 0.95, f'I thought this was a {classes[predicted_y[index]]} \n but it was a {classes[test_y[index]]}', fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.show()


if __name__ == '__main__':
    train_directory = "../images/select/train"
    test_directory = "../images/select/holdout"
    validation_directory = "../images/select/validation"

    model = create_model()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])


    checkpointer = ModelCheckpoint(filepath='./tmp/weights.hdf5', verbose=1, save_best_only=True)

    tensorboard = TensorBoard(
                log_dir='logs/', histogram_freq=0, batch_size=50, write_graph=True, embeddings_freq=0)

    train_generator, test_generator, validation_generator = generate_data(train_directory, validation_directory, test_directory)

    load = input("Load saved weights? (y/n) ")

    if load.lower() == 'y':
        model.load_weights("./tmp/loss-199-weights.hdf5")
        print("weights loaded")
    elif load.lower() == 'n':
        model.fit_generator(train_generator,
                steps_per_epoch=100,
                epochs=10,
                validation_data=validation_generator,
                validation_steps=1, callbacks=[checkpointer, tensorboard])

    make_analysis(test_generator)
