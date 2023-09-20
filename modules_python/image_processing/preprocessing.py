##########################################################################################
##########################################################################################
from skimage.color.adapt_rgb import each_channel, hsv_value, adapt_rgb                   #
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity        #
from skimage import transform, filters                                                   #
from skimage.color import rgba2rgb                                                       #
from skimage.morphology import disk                                                      #
from skimage.util import img_as_ubyte                                                    #
import numpy as np                                                                       #
import cv2                                                                               #
from skimage.morphology import closing, opening                                          #
from modules_python.config.config  import init, fg                                       #
from modules_python.image_processing.ImageSeg import ImageSegmentation                   #
from modules_python.image_processing.tools import  change_bg                             #
##########################################################################################
# Copyright : Iréné A. Essomba (c) 2023                                                  #
##########################################################################################

@adapt_rgb( each_channel )
def sobel_each( imgage )        :
    # sobel channel
    img = filters.sobel( imgage )
    return img

@adapt_rgb( hsv_value )
def sobel_hsv( image )          :
    # histogram value 
    img = filters.sobel( image )
    return img

def simple_gray( image )        :
    # channel n to 1
    return  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def GaussionBlur(image)         :
    # Gaussian Blur (GB)
    img = simple_gray(image)
    return cv2.GaussianBlur(src=img, ksize=(0,0), sigmaX=3)

def histogram( image, typ : str = "contrast" )  :
    # Contrast Stretching (CS)
    if typ =="contrast":
        p2, p98 = np.percentile(image, (2, 98))
        img_scale = rescale_intensity(image, in_range=(p2, p98))
        return img_scale

    #Grayscale / Histogram Equalization (HE)
    elif typ == "equal":
        img = simple_gray(image)
        img_equal = equalize_hist(img)
        return  img_equal 

    # ---- Grayscale / Contrast Limited Adaptive Histogram Equalization (CLAHE)  
    elif typ == "equal_adapt":
        img = simple_gray(image)
        img_adapteq = equalize_adapthist(img)
        return img_adapteq 
    
    #  ---- Grayscale / Local Histogram Equalization (LHE)
    else:
        footprint = disk(10)
        img = simple_gray( image )
        img = img_as_ubyte( img )
        img_eq_disk = filters.rank.equalize(img, footprint=footprint) / 255.0

        return  img_eq_disk 
    
def image_processing( image : np.ndarray, name : str = "RGB", reshape : tuple = (160, 160), add_contrast: bool = False):
    """
    Copyright : Iréné A. Essomba (c) 2023
    
    """
    # forme du maillaage 
    if reshape: width, height = reshape

    imgs            = image.copy()
    sobel           = False 
    filter          = None 

    # Conversion de rbga à rbg (channel = 4 ---> channel = 3)
    if imgs.shape[2] >= 4: 
        imgs    = rgba2rgb ( imgs )
        sobel   = True 

    # redimensionner l'image 
    if reshape is None: pass 
    else : imgs = transform.resize( image = imgs, output_shape=(width, height) )
    
    # rbg to histogram color 
    if   name == "RBG-HSV"      : filter = sobel_hsv( imgs ) 
    # histogram Contrast stretching
    elif name == "HIS-C"        : filter = histogram(imgs, typ = "contrast") 
    # hsitogram equalize 
    elif name == "HIS-EQ"       : filter = histogram(imgs, typ = "equal") 
    # histogram Adaptive Equalization
    elif name == "HIS-ADAPT"    : filter =  histogram(imgs, typ = "equal_adapt") 
    # histogram disk
    elif name == "HIS-DISK"     : filter =  histogram(imgs, typ = "disk") 
    # rgb to gary
    elif name == "SIMPLE_GRAY"  : filter = simple_gray( imgs ) 
    # gaussian blur
    elif name == "GAUSSIAN"     : filter =  GaussionBlur( imgs ) 
    # "RGR2-HVS"
    elif name == "RGR2-HVS"     : filter = cv2.cvtColor(imgs, cv2.COLOR_BGR2HSV)
    # "RGR2-LAB"
    elif name == "RGR2-LAB"     : 
        if add_contrast is False : filter = cv2.cvtColor(imgs, cv2.COLOR_BGR2LAB)
        else : 
            img_    =   histogram(imgs, typ = "contrast") 
            filter  = cv2.cvtColor(img_, cv2.COLOR_BGR2LAB)
    # rbg
    else                        : filter =  imgs

    return filter, sobel

def Size(image : np.ndarray)-> tuple:
    # initialize values
    width, height, channel = image.shape

    # compute the report 
    r       = width / height
    # compute the pixels of image
    pixel   = (width * height ) / (1024**2)

    # returning values 
    return r, pixel

def img_encoding(image : np.ndarray, mask : np.ndarray):
    img = image.copy()

    img[..., 0] = img[..., 0] * mask * 1
    img[..., 1] = img[..., 1] * mask * 1
    img[..., 2] = img[..., 2] * mask * 1

    return img

def filter_selection(
        img             : [list, np.ndarray], 
        figsize         : tuple  = (15, 4),  
        mul             : float  = 1.0,
        names           : list   = [], 
        select_index    : list   = [0],
        ylabel          : str    = "Intensité (Px)",
        bins            : int    = 20,
        rwidth          : float  = 0.2
        ):

    """
    Copyright : Iréné A. Essomba (c) 2023


    * img is the image with (n, m, m, 3) dimension, where n is the samples
    * figsize is a tuple used to create figures 
    * color_indexes is a list of size 3 used to set color in each plot
    * mul is numeric value
    * names is a list that contains the names of speces len(names) = n 
    
    *----------------------------------------------------------
    
    >>> img     = np.random.randn(3, 160, 160, 3)
    >>> names   = ["A", "B", "C"]
    >>> filter_selection(img = img, fisize = (8, 8), color_indexes = [20, 10,  6], names=names)
    
    """
    import matplotlib.pyplot as plt 
    import matplotlib.colors as mcolors

    # canaux 
    canaux = ["Luninosité", "Luninosité", "Luninosité"]
    error = None
    # uploading all python colors
    colors = ['darkred', "darkgreen", "darkblue"]
    #colors = list(mcolors.CSS4_COLORS.keys())
    # get the channel of the image
    #channel = img.shape[-1]
  
    # plotting image in function of the channel
    lenght = len(select_index)
    if lenght > 1 : fig, axes = plt.subplots(lenght, 3, figsize=figsize, sharey=True)
    elif lenght == 1 : fig, axes = plt.subplots(lenght, 3, figsize=figsize, sharey=True) 
    else: error = True  

    if error is None:
        if lenght > 1:
            for i in range(lenght): 
                ii =  select_index[i]
                channel = img[ii].shape[-1]
                for j in range(channel):
                    axes[i, j].hist(img[ii][:, :, j].ravel() * mul, bins=bins, color=colors[j], histtype="bar", 
                                    rwidth=rwidth ,density=False)
                    # title of image
                    if i == 0: axes[i, j].set_title(f"Canal {j}", fontsize="small", weight="bold", color=colors[j])
                    # set xlabel
                    if i == lenght-1 :axes[i, j].set_xlabel(canaux[j], weight="bold", fontsize='small', color=colors[j])
                    # set ylabel
                    #if j == 0 :  axes[i, j].set_ylabel(ylabel, weight="bold", fontsize='small', color=colors[j])
                    axes[i, j].set_ylabel(ylabel, weight="bold", fontsize='small', color=colors[j])
                    axes[i, j].legend(labels = [names[i]], fontsize='small', loc="best")
                else: pass

            else: pass
        else:
            
            for i in select_index:
                channel = img[i].shape[-1]
                for j in range(channel):
                    axes[j].hist( img[i][:, :, j].ravel(), bins=bins, color=colors[j], histtype="bar", 
                                    rwidth=rwidth ,density=False)
                   
                    axes[j].set_ylabel(ylabel, weight="bold", fontsize='small', color=colors[j])
                    axes[j].set_title(f"Canal {j}", fontsize="small", weight="bold", color=colors[j])
                    axes[j].set_xlabel(canaux[j], weight="bold",fontsize='small',color=colors[j])
                    axes[j].legend(labels = [names[i]], fontsize='small', loc="best")
                    
        plt.show()

def Semantic_Image_Plus_Data_Augment(
        imgs        : list, 
        srcs        : list,
        target      : list,
        feature_names : list,
        threshold   : list, 
        radius      : float = 3.0, 
        shape       : any   = None,
        axis        : int   = 1,
        dil_and_er  : bool  = False, 
        method      : str   = "numpy",
        color       : str   = "both",
        lower_color : list  = [0, 0, 0],
        upper_color : list  = [30, 30, 30],
        kernel      : tuple = (160, 160), 
        paths       : list  = [], 
        max_per_class : int = 3000
        ):
    
    """
    Copyright : Iréné A. Essomba (c) 2023



    The Semantic image segmentation is a powerfull tool used to create a mask for each objects loacted in th image .

    * img is an image of (m, m, c) dimension, c is the channel , (m, m) is the width and height 
    * threshold if a list of values of 2 dimension used to delimite the border of the image, threshold = [m1, m2] with m1 < m2
    * radius is a positive float number
    * shape used for dilation and errorion it takes two types of values : a None (for not reshaping ) and tuple = (l, l)
    * axis is a channel value and should lower than c
    * dil_and_err used to apply a second mask to the first created to go more in deep and extract and capture more details 
    * method is string value that takes 2 values : numpy and where is the type of calculation
    * src is the image transformed
    * upper_color is the maximal channel color and should hace (c, ) dimension 
    * lower_color is the maximal channel color and should hace (c, ) dimension
    * color = "both", "black" or "white"
    * kernel = (160, 160), (300, 300) etc....

    *-----------------------------------------------------------------------------
    
    >>> img = np.random.randn(160, 160, 3)
    >>> new_img = Semantic_Image_Plus_Data_Augment(img = img, threshold=[10, 100], radius=3, axis = 1, method='numpy'n color="both")

    * aplly the second filter on the first one

    >>> new_img = Semantic_Image_Plus_Data_Augment(img = img, threshold=[10, 100], radius=3, shape=(4, 4), dil_and_err = True, axis=1,method='where')
    """

    from modules_python.image_processing.data_aug import data_augmenter_v1

    index                       = 0
    MATRIX_BLACK                = []
    MATRIX_WHITE                = []
    TARGET                      = []
    FEATURE_NAMES               = []
    PATHS                       = []
    ID                          = []
    
    data_augmenter              = data_augmenter_v1()

    for j in range(len(imgs)):
        counter     = 0
        targ        = target[j]
        name        = feature_names[j]
        i           = 0
        if paths:   path = paths[j]
        else: path = None 

        if max_per_class > len(imgs[j]): 
            while counter <= max_per_class:            
                img         = imgs[j][i].astype('float32').copy()
                src         = srcs[j][i].astype('float32').copy()
                kernel      = img.shape 
        
                image_seg_  = ImageSegmentation(
                    img=img.copy(),
                    src=src.copy(),
                    threshold=threshold, 
                    radius=radius,
                    shape=shape,
                    axis=axis,
                    dil_and_err=dil_and_er,
                    method=method
                )
            
                if image_seg_ is not None:
                    image_seg = data_augmenter(image_seg_)
                    image_seg = image_seg.numpy().reshape(kernel)

                    if   color   == 'black'     : MATRIX_BLACK.append(image_seg)
                    elif color   == 'white'     :  
                        image_seg_b = image_seg.reshape((1, kernel[0], kernel[1], kernel[-1])) 
                        image_seg_b = change_bg(imgs=image_seg_b, lower_color=lower_color, upper_color=upper_color)
                        MATRIX_WHITE.append(image_seg_b[0])
                    elif color   == "both"      : 
                        MATRIX_BLACK.append(image_seg)
                        image_seg_b = image_seg.reshape((1, kernel[0], kernel[1], kernel[-1])) 
                        image_seg_b = change_bg(imgs=image_seg_b, lower_color=lower_color, upper_color=upper_color)
                        MATRIX_WHITE.append(image_seg_b[0])
                    
                    TARGET.append(targ)
                    FEATURE_NAMES.append(name)

                    if path is not None: 
                        path_ = path[i]
                        PATHS.append(path_)
                        ID.append(id(path_))
                    else: pass 

                    index   += 1
                    counter += 1

                    if i == len(imgs[j])-1 == 0: i = 0
                    else: i += 1

                else: break
        else: 
            error = fg(255, 0, 0)  + "max_per_class" + fg(255, 255, 255) + " is lower than " + \
            fg(255, 0, 255) + f" {len(imgs[j])} --> class {j} ---> name : {feature_names[j]}" + init.reset
            print(error)
            break

    if index > 0:
        indices_mel = np.arange(index)
        np.random.shuffle(indices_mel)

    if MATRIX_BLACK : MATRIX_BLACK = np.array(MATRIX_BLACK)[indices_mel]
    if MATRIX_WHITE : MATRIX_WHITE = np.array(MATRIX_WHITE)[indices_mel]
    if TARGET : TARGET = np.array(TARGET, dtype="int32")[indices_mel]
    if FEATURE_NAMES : FEATURE_NAMES = np.array(FEATURE_NAMES)[indices_mel]
    if PATHS : PATHS = np.array(PATHS)
    if ID: ID = np.array(ID, dtype="int64")

    data = {
        "X:black"       : MATRIX_BLACK,
        'X:white'       : MATRIX_WHITE, 
        "target"        : TARGET, 
        "id"            : index ,
        "kernel"        : kernel,
        "feature_names" : FEATURE_NAMES,
        "paths"         : PATHS,
        'ID'            : ID
        }

    return data.copy()