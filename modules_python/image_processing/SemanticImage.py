import numpy as np
import random
import matplotlib.pyplot as plt 
from modules_python.config.config import fg, init
from modules_python.image_processing.tools import get_mask, change_bg, erorsion_and_dilation


def SemanticImage(
        data        : dict, 
        index       : int   = 10, 
        channel     : int   = 1, 
        threshold   : list  = [-50, -9.], 
        upper_color : list  = [30, 30, 30], 
        lower_color : list  = [0,0,0],
        legend      : list  = None,
        radius      : int   = 2,
        method      : str   = "numpy",
        bg          : str   = "white", 
        id_sel      : list  = [6, 7, 8, 9, 10, 11],
        deep_mask   : bool  = False,
        kernel      : tuple = (2, 2),
        dtype       : float = np.float32,
        figsize     : float = (12,8),
        nrow        : int   =  5,
        fig_name    : str   = "all_process.png",
        cmap        : str   = 'RdYlBu', 
        ):
    
    """
    ----------------------------------------------------------

    >>> Copyright : Iréné A. Essomba (c) 2023

    ----------------------------------------------------------

    The Semantic image  is a powerfull tool used to create a mask for each objects loacted in th image .

    * data : is a dictionary that contains the images 
    * threshold if a list of values of 2 dimension used to delimite the border of the image, threshold = [m1, m2] with m1 < m2
    * radius is a positive float number
    * channel is a channel value and should lower than c
    * method is string value that takes 2 values : numpy and where is the type of calculation
    * upper_color is the maximal channel color and should hace (c, ) dimension 
    * lower_color is the maximal channel color and should hace (c, ) dimension
    * bg = "black", "white" or "mask" or "all"
    * index : is an integer type used to select à particular image
    * legend is a list used for title images
    * kernel : is a tuple used to create a deep mask 
    * deep_mask : is a boolean value used to specify is deep mask should be applied
    * id_sel : is a list of species

    *-----------------------------------------------------------------------------
    
    """
    # increment and error
    idd, error = 0, None

    # select speces
    if id_sel:  
        if len(id_sel) == 6: pass 
        else: error = fg.rbg(255, 0, 255) + " len(id_sel) " + fg.rbg(255, 255, 255) + "!=" + fg.rbg(255, 0, 0) + " 6 " + init.reset
    else: id_sel = [6, 7, 8, 9, 10, 11]

    if error is None:
        # checking if legend exists
        if legend : pass 
        else: legend = id_sel.copy()

        if bg in ["white", "black", "mask"] : 
            fig, axes = plt.subplots(1, 6, figsize=figsize) 

            for j in id_sel:
                X           = data['X'][j][index].astype(dtype=dtype).copy()
                mask        = get_mask(img=X[..., channel], threshold=threshold, radius=radius, method=method)

                if mask is not None:
                    mask        = mask * 1.
                    img         = data['images'][j][index].astype(dtype=dtype).copy()
                    shape       = img.shape

                    if deep_mask is True : mask      = erorsion_and_dilation(mask, shape=kernel)
                    else: pass
                
                    img[..., 0] = img[..., 0] * mask * 1.
                    img[..., 1] = img[..., 1] * mask * 1.
                    img[..., 2] = img[..., 2] * mask * 1.

                    new_img = img.reshape((1, shape[0], shape[1], 3))
                    new_img = change_bg(imgs=new_img, lower_color=lower_color, upper_color=upper_color)

                    if bg == 'white': 
                        for i in range(1):   
                            axes[idd].imshow(new_img[0])
                            axes[idd].set_title(legend[j], fontsize="small")
                            axes[idd].axis("off")
                    
                    if bg == 'black': 
                        for i in range(1): 
                            axes[idd].axis("off")  
                            axes[idd].imshow(img)
                            axes[idd].set_title(legend[j], fontsize="small")

                    if bg == "mask":
                        for i in range(1): 
                            axes[idd].axis("off")  
                            axes[idd].imshow(mask)
                            axes[idd].set_title(legend[j], fontsize="small")

                    idd += 1
                else: break
            plt.show()

        elif bg == "all":
            error       = None 
            fig, axes   = plt.subplots(nrow, 6, figsize=figsize) 
            names       = ['images', "X", "mask", "black", "white"]

            for k in range(nrow):
                idd     = 0
                bg      = names[k]

                for j in id_sel:
                    if bg not in ['X', 'images']:
                        X           = data['X'][j][index].astype(dtype=dtype).copy()
                        mask        = get_mask(img=X[..., channel], threshold=threshold, radius=radius, method=method)
                        
                        if mask is not None:
                            mask        = mask * 1.
                            img         = data['images'][j][index].astype(dtype=dtype).copy()
                            shape       = img.shape

                            if deep_mask is True : mask      = erorsion_and_dilation(mask, shape=kernel)
                            else: pass
                        
                            img[..., 0] = img[..., 0] * mask * 1.
                            img[..., 1] = img[..., 1] * mask * 1.
                            img[..., 2] = img[..., 2] * mask * 1.

                            new_img = img.reshape((1, shape[0], shape[1], 3))
                            new_img = change_bg(imgs=new_img, lower_color=lower_color, upper_color=upper_color)
                        else:
                            error = True 
                            break 
                    
                    if error is None:
                        if bg == 'white': 
                            for i in range(1):   
                                axes[k, idd].imshow(new_img[0] )
                                axes[k, idd].axis("off")
                        
                        if bg == 'black': 
                            for i in range(1): 
                                axes[k, idd].axis("off")  
                                axes[k, idd].imshow(img)
                                
                        if bg == "mask":
                            for i in range(1): 
                                axes[k, idd].axis("off")  
                                axes[k, idd].imshow(mask)
                                
                        if bg == "X":
                            XX = data["X"][j][index] 
                            for i in range(1): 
                                axes[k, idd].axis("off")  
                                axes[k, idd].imshow(XX[..., 1], cmap = cmap, interpolation="nearest")
                                
                        if bg == "images" : 
                            XX = data[bg][j][index] 
                            for i in range(1): 
                                axes[k, idd].axis("off")  
                                axes[k, idd].imshow(XX)
                                
                        if k == 0 : axes[k, idd].set_title(legend[j], fontsize="small")
                        idd += 1
                    else : break

            if error is None:
                plt.savefig(f"./images/{fig_name}")
                plt.show()
        else : 
            error = fg.rbg(255, 0, 255) + " bg " + fg.rbg(255, 255, 255) + "not in" + fg.rbg(255, 0, 0) + " ['white', 'black', 'all'] " + init.reset
            print(error)
    else:  print(error)