import numpy as np 
import tensorflow as tf

def data_augmenter_v1(fill_mode : str = "nearest"):
    

    """
    Copyright : Iréné A. Essomba (c) 2023


    >>> fill_mode = ["reflect", "wrap", "constant","nearest"]

    """
    import tensorflow as tf  

    # initialize the sequence
    model = tf.keras.models.Sequential()
    # random flip vertical and horizontal 
    model.add(tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal_and_vertical', seed=1))
    # random rotation 
    model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.8, seed=1, fill_mode=fill_mode, fill_value=0.5))
    # random zoom
    model.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2, seed=1, fill_value=0.5, fill_mode='nearest'))
    # random normalisation
    model.add(tf.keras.layers.experimental.preprocessing.Normalization())

    return model

def data_augmenter_v2():
     

    """
    Copyright : Iréné A. Essomba (c) 2023



    * imgs is the image with (n, m, m, c) dimension 
    n the sample (m, m, c) the features of images

    * imgs is ndarray dtype 

    >>> imgs = np.random.randn(1000, 160, 160, 3)
    >>> imgs_flow = data_augmenter_v2(imgs = imgs)
    
    
    """
    #import tensorflow as tf 
    import tensorflow as tf 

    data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        rotation_range=20.0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range= 0.2,
        zoom_range=0.2, 
        horizontal_flip=True,
        vertical_flip=True,
        validation_split= 0.2
    )

    return data_aug