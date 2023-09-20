import numpy as np
import random
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from modules_python.config.config import fg, init
def plot(
        data        : dict, 
        legend      : list  = None, 
        index       : int   = 10, 
        channel     : int   = 1, 
        fig_name    : str   = "out.png", 
        colors      : list  = None,
        type_img    : str   ='X',
        cmap        : str   = 'RdYlBu', 
        save        : bool  = False,
        nrow        : int   = 2,
        ncol        : int   = 6,
        figsize     : tuple = (12, 4),
        select_id   : list  = None, 
        ):
    """
    ----------------------------------------------------------

    >>> Copyright : Iréné A. Essomba (c) 2023

    ----------------------------------------------------------

    * data : is a dictionary that contains the images 
    * legend : is a list used for image's titles 
    * index : is an integer used to select a image in data 
    * channel : is an intger belongs to [0, 1, 2] used for colors representation
    * fig_name : is a string used to output and save image
    * colors : is a list used for title's colors
    * type_img : is a string used to select whitch category of images we want to plot it takes two values : ['X', 'images']
    * cmap : is a string for color mapping (2D dimensionnal plot)
    * nrow : is an integer used to specify the number of lines
    * ncol : is an integer used to specify the number of columns
    *----------------------------------------------------------------------------------------------------------------------------

    >>> plot(data=data, index = 1, fig_name = "out.png", channel=1, legend=None, colors=None)
    >>> plot(data=data, index = 1, fig_name = "out.png", channel=1, legend=None, colors=['red', 'g', 'm', 'g', 'k'])
    """

    # intialisation de l'incrément 
    idd  = 0

    if data:
        # verification des coleurs
        if colors is None: colors = random.sample(list(mcolors.CSS4_COLORS.keys()), 12)
        else: pass 
        # definition de la légende
        if legend is None: legend = [f'{x}' for x in range(12)]
        else: pass 

        try:
            if   type_img in ['X', 'images']:
                # création d'un canvas de 12 figures 2x6 représentant une espèce unique
                fig, axes = plt.subplots(nrow, ncol, figsize=figsize)  
                
                if nrow > 1:
                    for i in range(nrow):
                        for j in range(ncol):
                            X = data[type_img][idd]
                            # conversion dtype(object) ---> dtype(float32)
                            XX = X[index].astype("float")
                            # création de figures avec imshow
                            if type_img == 'X' : axes[i, j].imshow(XX[..., channel], cmap=cmap, interpolation="nearest")
                            else : axes[i, j].imshow(XX, interpolation="nearest")
                            # difinir un titre pour chaque plantes 
                            axes[i, j].set_title(legend[idd], fontsize='small', color=colors[idd], weight="bold")
                            # x_axis and y_axis off
                            axes[i, j].axis("off")
                            # incrémentation 
                            idd += 1
                else:
                    for j in range(ncol):
                        if    select_id is None: pass 
                        else: idd = select_id[j]

                        X = data[type_img][idd]
                        # conversion dtype(object) ---> dtype(float32)
                        XX = X[index].astype("float")
                        # création de figures avec imshow
                        if type_img == 'X' : axes[j].imshow(XX[..., channel], cmap=cmap, interpolation="nearest")
                        else : axes[j].imshow(XX, interpolation="nearest")
                        # difinir un titre pour chaque plantes 
                        axes[j].set_title(legend[idd], fontsize='small', color=colors[idd], weight="bold")
                        # x_axis and y_axis off
                        axes[j].axis("off")
                        # incrémentation 

                if nrow > 1 :
                    # saving figure in .png format 
                    if type_img == 'X' : plt.savefig("./images/out.png")
                    else : plt.savefig(f"./images/{fig_name}") if save is True else ""
                else: pass 
                plt.show()
            elif type_img == "both":
                if (ncol == 6) and (nrow == 2): 
                    fig, axes = plt.subplots(int( nrow * 2), ncol , figsize=figsize) 
                    k = 0
                    for type_ in ['images', 'X']:
                        idd = 0
                        for i in range(nrow):
                            for j in range(ncol):
                                X = data[type_][idd]
                                # conversion dtype(object) ---> dtype(float32)
                                XX = X[index].astype("float")
                                # création de figures avec imshow
                                if type_ == 'X' : axes[k, j].imshow(XX[..., channel], cmap=cmap, interpolation="nearest")
                                else : axes[k, j].imshow(XX, interpolation="nearest")
                                # difinir un titre pour chaque plantes 
                                axes[k, j].set_title(legend[idd], fontsize='small', color=colors[idd], weight="bold")
                                # x_axis and y_axis off
                                axes[k, j].axis("off")
                                # incrémentation 
                                idd += 1
                            k += 1
                    # saving figure in .png format 
                    plt.savefig(f"./images/{fig_name}") if save is True else ""
                    plt.show()
                else:
                    error = fg.rbg(255, 0, 255) + " nocl != 6 and nrow != 2 "  + init.reset
                    print(error)
            else: 
                error = fg.rbg(255, 0, 255) + " type_img " + fg.rbg(255, 255, 255) + "not in" + fg.rbg(255, 0, 255)+ " ['X', 'images', 'both'] " + init.reset
                print(error)
        except ValueError:
            error = fg.rbg(255, 0, 255) + " index " + fg.rbg(255, 255, 255) + "is out of range" + init.reset
            print(error)
    else: 
        error = fg.rbg(255, 0, 255) + " data " + fg.rbg(255, 255, 255) + "cannot be empty" + init.reset
        print(error)