import numpy as np
from modules_python.image_processing.tools import get_mask,  erorsion_and_dilation, Best_mask
from modules_python.config.config import init, fg

def ImageSegmentation(
        img         : np.ndarray, 
        src         : np.ndarray,
        threshold   : list, 
        radius      : float = 3.0, 
        shape       : any   = None,
        axis        : int   = 1,
        dil_and_err : bool  = False, 
        method      : str   = "numpy",
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

    *-----------------------------------------------------------------------------
    
    >>> img = np.random.randn(160, 160, 3)
    >>> new_img = Semantic_image_segmentation(img = img, threshold=[10, 100], radius=3, axis = 1, method='numpy')

    * aplly the second filter on the first one

    >>> new_img = Semantic_image_segmentation(img = img, threshold=[10, 100], radius=3, shape=(4, 4), dil_and_err = True, axis=1,method='where')
    """

    if type(img) == type(np.array([0])) : 
        filter_img  = None 

        # shannel
        channel = img.shape[-1]
    
        # creating filter based on the image 
        filter = get_mask(img=img[:, :, axis], threshold=threshold, radius=radius, method=method)
        filter_er = filter.copy()

        if dil_and_err is True: 
            if shape:
                try:
                    np.array(shape).sum()
                    filter_er = erorsion_and_dilation(img=filter, shape=shape)
                except (TypeError, ValueError): 
                    error = init.bold + fg.rbg(0, 255, 0) + f"cannot apply the shape" + init.reset
                    print(error)
            else:
                error = init.bold + fg.rbg(0, 255, 0) + f"dil_and_err is set on true by shape = None" + init.reset
                print(error)
        else: pass 

        # make a copy
        filter_img = src.copy()

        # get the best filter 
        best_filter, _  = Best_mask(img=img.copy(), mask_1=filter, mask_2=filter_er)

        # make changes 
        for n in range(channel):
            filter_img[..., n] = filter_img[..., n] * best_filter * 1.0
         
    else: 
        t = type(np.array([0]))
        error = init.bold + fg.rbg(0, 255, 0) + f"img {fg.white_L} is not {fg.rbg(255,0,0)}{t}" + init.reset
        print(error)
    
    # returning values 
    return filter_img