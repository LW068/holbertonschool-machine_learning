#!/usr/bin/env python3
import tensorflow.keras as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class CustomLearningRateScheduler(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            current_rate = float(K.backend.get_value(self.model.optimizer.learning_rate))
            new_rate = current_rate * 0.1
            K.backend.set_value(self.model.optimizer.learning_rate, new_rate)
            print(f'Learning rate adjusted to: {new_rate}')

def preprocess_data(X, Y):
    """
    Function to pre-processes the data for your model
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    # Load the Cifar10 dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    # Load the pre-trained model
    base_model = K.applications.DenseNet121(include_top=False,
                                            weights='imagenet',
                                            input_shape=(224, 224, 3))

    # Add a global spatial average pooling layer
    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)

    # Add a fully-connected layer
    x = K.layers.Dense(1024, activation='relu')(x)

    # Add a logistic layer with 10 classes (CIFAR has 10 classes)
    predictions = K.layers.Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = K.models.Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model (should be done after setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    lr = CustomLearningRateScheduler()

    # Train the model on the new data for a few epochs
    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              validation_data=(x_test, y_test),
              epochs=20,
              verbose=1,
              callbacks=[es, mc, lr])

    # Save the model
    model.save('cifar10.h5')
