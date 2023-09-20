import numpy as np
import cv2
from modules_python.config.config import fg, init

def img_encoding(image : np.ndarray, mask : np.ndarray):
    img = image.copy()

    img[..., 0] = img[..., 0] * mask * 1
    img[..., 1] = img[..., 1] * mask * 1
    img[..., 2] = img[..., 2] * mask * 1

    return img

def Best_mask(img : np.ndarray, mask_1 : np.ndarray, mask_2 : np.ndarray):

    import tensorflow as tf 

    # distnace between encoding and identity
    image_1 = img_encoding(image=img, mask=mask_1)
    image_2 = img_encoding(image=img, mask=mask_2)

    dist1 = tf.reduce_sum( tf.square( tf.subtract(img, image_1) ) ) 
    dist2 = tf.reduce_sum( tf.square( tf.subtract(img, image_2) ) )

    if dist1 < dist2: return (mask_1, "mask1")
    else: return (mask_2, "mask2") 

def get_mask(img : np.ndarray, threshold : list = [10, 50] , radius: float = 2, method : str = 'numpy') -> np.ndarray:

    """
    Copyright : Iréné A. Essomba (c) 2023



    * img is an image of (m, m, c) dimension, c is the channel , (m, m) is the width and height 
    * threshold if a list of values of 2 dimension used to delimite the border of the image, threshold = [m1, m2] with m1 < m2
    * radius is positive float number
    * method is string value that takes 2 values : numpy and where is the type of calculation

    *--------------------------------------------------------------------------------------------------

    >>> img = np.random.randn(160, 160, 3)
    >>> mask = get_mask(img = img, threshold=[10, 50], radius=2, method='numpy')
    >>> mask = get_mask(img = img, threshold=[10, 50], radius=2, method='where')

    """

    from skimage.morphology import closing
    from skimage.morphology import disk  

    # numpy method
    if method == "numpy": filter = ( img > threshold[0]  ) & ( img < threshold[1] ) 
    # where method
    else: filter = np.where( ( img > threshold[0]  ) & ( img < threshold[1] ), 1, 0)

    # create a disk
    S = disk(radius)
    # create filter by comparing the values close to S or inside the disk
    filter = closing(filter, S)

    # returning values
    return filter

def erorsion_and_dilation(img : np.ndarray, show : bool = False, shape=(3,3)) -> np.ndarray:
    """ 
    Copyright : Iréné A. Essomba (c) 2023


    * img is the image with (m, m, c) dimension 
    * show is a bool value initialized on false. used to plot curve 
    * shape is the kernel used to make errosion and dilation. it has (n, n) dimension 

    * -------------------------------------------------------------

    >>> img = np.random.randn(160, 160, 3)
    >>> new_img = errorsion_and_dilation(img = img, shape = (2, 2))

    """

    # module loading
    import matplotlib.pyplot as plt 
    from scipy import ndimage

    x_open = ndimage.binary_opening(img, structure=np.ones(shape))

    if show is True:
        plt.matshow(x_open, interpolation="nearest") 
        plt.axis("off")
        plt.show()
    else: pass

    return x_open

def float_to_int(imgs : np.ndarray, dtype : str = "int32") -> np.ndarray:

    """
    Copyright : Iréné A. Essomba (c) 2023



    * imgs is a ndarray type with (m, m, c) dimension 
    * dtype is the imgs data type 

    creating bins between [0, 255]
    then converting into integer types 

    >>> Examples

    >>> imgs = np.random.randn(100, 160, 160, 3)
    float_to_int(imgs = imgs, dtype = "int32")

    * The default value is int32

    >>>  float_to_int(imgs = imgs)

    """
    # intitializing 
    image       = imgs.copy() 
    dtype_lists = ['int8', 'int32', 'int64', 'int128', 'int256']

    # checking the type of image
    if type(imgs) == type(np.array([0])):
        # checking if we got the right dtype 
        if dtype in dtype_lists:  image = (image * 255).astype(dtype=dtype)
        else: 
            error = init.bold + fg.rbg(0, 255, 0) + f"{dtype}{fg.black_L} not in the list {fg.rbg(255, 75, 50)}{dtype_lists}" + init.reset
            print(error)
    else: 
        t = type(np.array([0]))
        error = init.bold + fg.rbg(0, 255, 0) + f"imgs {fg.white_L} is not {fg.rbg(255,0,0)}{t}" + init.reset
        print(error)
    
    return image

def change_bg(imgs : np.ndarray, upper_color : list, lower_color : list, dtype : str = "int32"):
    """
    Copyright : Iréné A. Essomba (c) 2023



    * imgs is a ndarray type with (n, m, m, c) dimension 
      n is the samples, (m, m, c) the features of image
    * upper_color is the maximal channel color and should hace (c, ) dimension 
    * lower_color is the maximal channel color and should hace (c, ) dimension 
    * dtype is the imgs data type

    to change the background color (BGC) we need to create a mask to fix the border limit 
    where the filter can be applied. To do that we will use the argument

    - upper_color 
    - lower color

    >>> lower_color = np.array([0, 0, 0]) 
    >>> upper_color = np.array([30, 30, 30])
    >>> mask = cv2.inRange(imgs, lower_color, upper_color)
    >>> dtype = ['int8', 'int32', 'int64', 'int128', 'int256']
    * ------------------------------------------------------------

    >>> imgs = np.random.randn(100, 160, 160, 3)
    >>> new_imgs = change_bg(imgs=imgs, upper_color=[30, 30, 30], lower_color=[0, 0, 0], dtype="int32")

    *-------------------------------------------------------------
    >>> new_imgs.shape = imgs.shape = (100, 160, 160, 3)

    """
    if type(imgs) == type(np.array([0])):
        IMGS = imgs.copy()
        shape = IMGS.shape
        if upper_color:
            if len(upper_color) == len(lower_color):
                if len(upper_color) == shape[-1]:
                    upper_color = np.array(upper_color).reshape((shape[-1], ))
                    lower_color = np.array(lower_color).reshape((shape[-1], ))
                    try:
                        prod = upper_color * lower_color

                        IMGS = float_to_int(IMGS, dtype=dtype)
                        for i in range(shape[0]):
                            mask = cv2.inRange(src=IMGS[i], lowerb=lower_color, upperb=upper_color)
                            IMGS[i, mask > 0] = [255, 255, 255]
                    except TypeError:
                        error = init.bold + fg.rbg(0, 255, 0) + f"data type error in {fg.rbg(255,0,0)}<<upper_color>> or <<lower_color>>" + init.reset
                        print(error)
                else:
                    error = init.bold + fg.rbg(0, 255, 0) + f"{fg.cyan_L}len(upper_color) != {shape[-1]}" + init.reset
                    print(error)
            else:
                error = init.bold + fg.rbg(0, 255, 0) + f"{fg.cyan_L}len(upper_color) != len(lower_color)" + init.reset
                print(error)
        else:
            error = init.bold + fg.rbg(0, 255, 0) + f"upper_color cannot be empty" + init.reset
            print(error)
    else: 
        t = type(np.array([0]))
        error = init.bold + fg.rbg(0, 255, 0) + f"imgs {fg.white_L} is not {fg.rbg(255,0,0)}{t}" + init.reset
        print(error)

    return IMGS