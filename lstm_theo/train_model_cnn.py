import tensorflow as tf
import numpy as np
import pickle

import helper_stuff as rhlp

#X_train, X_test, Y_train, Y_test = train_test_split(dataset_x, dataset_y, test_size = 0.24)

BATCH_SIZE = 8
EPOCH = 100
PATIENCE = 3
model_unique_name = 'norm_lm_xy_softmax_8-16-32_dense128LRELU_drop2final_' + str(BATCH_SIZE) + '_' + str(EPOCH) + '_' + str(PATIENCE) #name convention: (name)_batchsize_epoch
print(model_unique_name)

with open('dataset_split.pickle', 'rb') as handle:
    b = pickle.load(handle)

train_dataset = rhlp.SequenceImagesDataset(b['TRAIN'], BATCH_SIZE)
tests_dataset = rhlp.SequenceImagesDataset(b['TEST'],  BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(30, 468, 2)),

    tf.keras.layers.Conv2D(8, 4, activation = "LeakyReLU"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(16, 4, activation = "LeakyReLU"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, 4, activation = "LeakyReLU"),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="LeakyReLU"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer = 'adam',
              loss      = tf.keras.losses.CategoricalCrossentropy(),
              metrics   = ['accuracy'])


save_best_callback = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath        = './best_model_cnn_' + model_unique_name + '.h5', 
        save_best_only  = True, 
        monitor         = 'val_accuracy',
        mode            = 'max'),
    tf.keras.callbacks.CSVLogger('./history_cnn_' + model_unique_name + '.csv'),
    tf.keras.callbacks.EarlyStopping(
        patience        = PATIENCE
    )
]

model.fit(
        train_dataset,
        epochs = EPOCH,
        validation_data = tests_dataset,
        callbacks = save_best_callback
)

model.summary()

print("training Complete...")