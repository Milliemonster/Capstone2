
import pandas as pd
import numpy as np
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


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
    directory="japanese_beetle/train",
    save_to_dir = "japanese_beetle/train",
    #target_size=(224, 224),
    color_mode="rgb",
    batch_size=20,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = validation_datagen.flow_from_directory(
    directory="japanese_beetle/validation",
    #target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory="japanese_beetle/holdout",
    #target_size=(224, 224),
    color_mode="rgb",
    batch_size=1, #number that divides test set evenly
    class_mode=None,
    shuffle=False,
    seed=42
)

train_datagen.fit(train_generator)
