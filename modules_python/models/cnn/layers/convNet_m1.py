import tensorflow as tf


def conv(input_shape : tuple = (224, 224, 3), classes : int = 12):
    
    # creating the input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # première couche de convolution stride = (4,4), padding="valid", kernel=(11, 11), pol_size = (2,2), filters=96
    X = tf.keras.layers.Conv2D(
        filters=96, kernel_size=(11, 11), 
        kernel_initializer="glorot_uniform",
        strides=(4, 4), padding="valid"
        )(X)
    # fonction  d'activation 
    X = tf.keras.layers.ReLU()(X)
    # réduction de dimension par 2
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(2,2), strides=(1, 1)
        )(X)

    # deuxième couche de convolution stride = (1,1), padding="valid", kernel=(5,5), pol_size = (2,2), filters=256
    X = tf.keras.layers.Conv2D(
        filters=256, kernel_size=(5, 5), 
        kernel_initializer="glorot_uniform",
        strides=(1, 1), padding="valid"
        )(X)
    # fonction  d'activation 
    X = tf.keras.layers.ReLU()(X)
    # réduction de dimension par 2
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(1, 1), padding="valid"
        )(X)

    # troisième couche de convolution stride = (1,1), padding="valid", kernel=(5,5), pol_size = (1, 1), filters =384
    X = tf.keras.layers.Conv2D(
        filters=384, kernel_size=(3, 3), 
        kernel_initializer="glorot_uniform",
        strides=(1, 1), padding="valid"
        )(X)
    # fonction d'activation
    X = tf.keras.layers.ReLU()(X)
    # conversation de la taille de l'image padding = "same" ---> stride = (1, 1)
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(1, 1), strides=(1, 1), padding="same"
        )(X)

    # quatrième couche de convolution stride = (1,1), padding="valid", kernel=(1, 1), 
    # filters =384, pol_size = (1, 1), drop_out = 0.7
    X = tf.keras.layers.Conv2D(
        filters=384, kernel_size=(1, 1), 
        kernel_initializer="glorot_uniform",
        strides=(1, 1), padding="same"
        )(X)
    # fonction d'activation
    X = tf.keras.layers.ReLU()(X)
    # conversation de la taille de l'image padding = "same" ---> stride = (1, 1)
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(1, 1), strides=(1, 1) 
        )(X)
    # déconnection de 20 % des couches de façon random pour limiter le surapprentissage 
    X = tf.keras.layers.Dropout(rate=0.7)(X)

    # 5eme couche de convolution stride = (1,1), padding="valid", kernel=(1, 1), filters = 512, drop_out = 0.5
    X = tf.keras.layers.Conv2D(
        filters=512, kernel_size=(1, 1), 
        kernel_initializer="glorot_uniform",
        strides=(1, 1), padding="same"
        )(X)
    # fonction d'activation
    X = tf.keras.layers.ReLU()(X)
    # conversation de la taille de l'image padding = "same" ---> stride = (1, 1)
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(1, 1), strides=(1, 1) 
        )(X)
    # déconnection de 20 % des couches de façon random pour limiter le surapprentissage 
    X = tf.keras.layers.Dropout(rate=0.7)(X)

    # applatissement 
    X = tf.keras.layers.Flatten()

    # 1ere couche full connected, avec 4096 neurones et relu comme function d'activation
    X = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu)(X)
    # déconnection de 50 % de neurone de façon random pour limiter le surapprentissage 
    X = tf.keras.layers.Dropout(rate=0.5)(X)

    # couche de classification (12 classes)
    X = tf.keras.layers.Dense(units=classes, activation=tf.keras.activations.softmax)(X)

    # output 
    outputs = X

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model