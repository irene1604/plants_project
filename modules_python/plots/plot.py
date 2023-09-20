import numpy as np
import seaborn as sns 
import random
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def Pct(pct, size : int = 5539):
    value = (pct / size) * 100
    return  f"{np.round(value, 1)}%"

def hist_hist_plot(
        X       : list, 
        legend  : list,
        title   : list  = ['', ''], 
        xlabel  : list  = ['', ''],
        ylabel  : list  = ["", ""],
        box     : list  = [(0.5, 0.2, 0.5, 0.5), (0.5, 0.2, 0.5, 0.5)], 
        colors  : list  = [],
        figsize : tuple = None, 
        style   : int   = None,
        y_lim   : list  = [-5, 250],
        bins    : int   = 10,
        grille  : bool  = False, 
        bonding_box : bool = False,
        coord   : dict = {"x" : [[10, 250], [10, 250]], 
                        "y" : [[0, 245], [0, 245]], "xmin":[0, 0], "ymin":[0, 0],
                        "xmax" : [250, 250], "ymax" :  [245, 245]},
        annot   : bool = False, 
        text    : bool = False,
        s       : list = ['$range = [90, 250]$', '$range = [90, 250]$'],
        Types   : list = ["hist", "hist"]
        ):
    
    error = None 

    if bonding_box is True:
        x       = coord['x']
        y       = coord['y']
        xmin    = coord['xmin']
        xmax    = coord['xmax']
        ymin    = coord['ymin']
        ymax    = coord['ymax']
    else: pass 

    if grille is True : plt.grid()
    if style is not None : 
        av = list(plt.style.available)
        if style >= len(av) : pass 
        else: plt.style.use(av[style])
    if figsize is None: figsize = (10, 3)
    if colors: pass 
    else: colors = random.sample(list(mcolors.CSS4_COLORS.keys()), len(legend))

    sns.color_palette()
    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=True, sharey=True)

    for i in range(len(X)):
        if Types[i] == 'hist':
            axes[i].hist(X[i], bins=bins, histtype='bar', color = colors)
            axes[i].set_title(title[i], weight="bold", fontsize="small")
            axes[i].set_xlabel(xlabel[i],  weight="bold", fontsize="small")
            axes[i].set_ylabel(ylabel[i],  weight="bold", fontsize="small")
            axes[i].legend(legend, bbox_to_anchor=box[i])
            axes[i].set_ylim(y_lim)
        else:
            error = True 
            break

    if error is None:
        if bonding_box is True: 
            rect_box(axes=axes, x=x, y=y, xmin=xmin, xmax=xmax, 
                     ymin=ymin, ymax=ymax, text=text, annot=annot,s=s)
        else: pass

        plt.show()

        return [ax for ax in axes]
    else:
        print("ERROR")
        for i in range(len(X)):
            axes[i].remove()  
        return None

def hist_bar_plot(
        X       : list, 
        figsize : tuple = (12, 3), 
        Types   : list  = ["hist", "bar"], 
        colors  : list  = None, 
        box     : tuple = (0.5, 0.1, 0.5, 0.5),
        legend  : list  = [],
        bb_box  : dict  = {"x":1.5, "y":250},
        s       : str   = '$range = [0.1, 0.250]$\n$pixel=\dfrac{largeur * hauteur}{1024^2}\sim 0.25Mpx$',
        rot     : float = 10,
        titles  : list  = ["Histograme de Pixelisation", "Nombre de plantes par espèce"],
        xlabel  : list  = ['', ''],
        ylabel  : list  = ['', ''],
        y_lim   : list  = [[-5, 330], [-5, 720]],
        style   : int   = None,
        grille  : bool  = False,
        c       : str   = 'k',
        ls      : str   = '--',
        lw      : float = 2.,
        width   : float = 0.3,
        bins    : int   = 10,
        legends : list  = None,
        gama    : list  = None,
        encoding : bool = False,
        sort     : bool = False,
        rev    : bool = False,
        bar_bbox : tuple = (1.0, 0.5, 0.5, 0.5),
        sc       : str ="w"
        ):
    from sklearn.preprocessing import LabelEncoder 

    if grille is True : plt.grid()
    if style is not None : 
        av = list(plt.style.available)
        if style >= len(av) : pass 
        else: plt.style.use(av[style])
    if figsize is None: figsize = (10, 3)
    if colors: pass 
    else: colors = random.sample(list(mcolors.CSS4_COLORS.keys()), len(legend))

    sns.color_palette()

    fig, axes = plt.subplots(1, len(X), figsize=figsize, squeeze=True)

    for i in range(len(X)):
        if Types[i] == "hist":
            axes[i].hist(X[i], bins=bins, color=colors )
            axes[i].legend(legend, bbox_to_anchor=box, title = 'Légende')
            axes[i].text(x=bb_box['x'], y=bb_box['y'], s=s)
            
        elif Types[i] == "bar":
            if legends is None: pass 
            else:
                legend = legends
                colors  = colors + list( random.sample(list(mcolors.CSS4_COLORS.keys()), 6 ) )
                X[i]    = gama

            if sort is True: 
                if rev is False : id_sort = np.argsort(X[i])
                else : id_sort = list(reversed(np.argsort(X[i])))
                id_sort = np.array(id_sort)
            else: pass 
            
            labels = legend.copy()
            if encoding is True:
                legend = LabelEncoder().fit_transform(legend)
                if sort is True: 
                    legend  = legend[id_sort]
                    X[i]    = X[i][id_sort]
                    colors  = list(np.array(colors)[id_sort])
                else: pass
                labels = [f"[{x}] {labels[x]}" for x in legend]
            else:
                if sort is True: 
                    legend  = legend[id_sort]
                    X[i]    = X[i][id_sort]
                    colors  = np.array(colors)[id_sort]
                else: pass
            
            axes[i].bar(x=range(len(legend)), height=X[i], color=colors, width=width, label=labels)
            axes[i].legend(labels, bbox_to_anchor=bar_bbox, title = 'Légende', fontsize="small")
            axes[i].scatter(x=range(len(legend)), y=X[i], s=30, color=sc)
            axes[i].plot(X[i], color=c, ls = ls, lw=lw)
            axes[i].set_xticks(range(len(legend)), legend, rotation=rot, ha="center")
            
        axes[i].set_ylim(y_lim[i])
        axes[i].set_xlabel(xlabel[i],  weight="bold")
        axes[i].set_ylabel(ylabel[i],  weight="bold")
        axes[i].set_title(titles[i],  weight="bold")
        

    plt.show() 

def hist_pie_plot(
        X       : list, 
        figsize : tuple = (12, 3), 
        Types   : list  = ["hist", "pie"], 
        colors  : list  = None, 
        box     : tuple = (0.5, 0.1, 0.5, 0.5),
        legend  : list  = [],
        bb_box  : dict  = {"x":1.5, "y":250},
        s       : str   = '$range = [0.99, 1.0.2]$\n$frac=\dfrac{ largeur }{ hauteur} \sim 1$',
        titles  : list  = ["Ratio largeur\hauteur", ""],
        xlabel  : list  = ['', ''],
        ylabel  : list  = ['', ''],
        y_lim   : list  = [-5, 330],
        style   : int   = None,
        grille  : bool  = False,
        bins    : int   = 10,
        vline   : bool  = False,
        vh_lw   : float = 2.0, 
        radius  : float = 0.75,
        pctdistance : float = 0.88,
        x_lim   : list  = [0.95, 1.2],
        v_line  : float = 1.005, 
        Sobel_legends : list = None, 
        explode_id    : list = [],
        size    : int  = 5539
        ):

    if grille is True : plt.grid()
    if style is not None : 
        av = list(plt.style.available)
        if style >= len(av) : pass 
        else: plt.style.use(av[style])
    if figsize is None: figsize = (10, 3)
    if colors: pass 
    else: colors = random.sample(list(mcolors.CSS4_COLORS.keys()), len(legend))

    sns.color_palette()
    sum_ = sum(X[1])
    if sum_ != 0 : 
        N = len(X)
        fig, axes = plt.subplots(1, len(X),  figsize=figsize, squeeze=True)
    else: 
        N = 1
        fig, axes = plt.subplots(1, 1, figsize=figsize, squeeze=True)

    if N > 1:
        for i in range(N):
            if Types[i] == "hist":
                axes[i].hist(X[i], bins=bins, color=colors )
                axes[i].legend(legend, bbox_to_anchor=box)
                axes[i].text(x=bb_box['x'], y=bb_box['y'], s=s)
                if vline is True:
                    axes[i].vlines(x=v_line, ymax=y_lim[1], ymin=y_lim[0], lw = vh_lw, colors="k")
                
                axes[i].set_xlabel(xlabel[i],  weight="bold")
                axes[i].set_ylabel(ylabel[i],  weight="bold")
                axes[i].set_title(titles[i],  weight="bold")
                axes[i].set_ylim(y_lim)
                axes[i].set_xlim(x_lim)
            elif Types[i] == "pie":
                XX      = [p for p in X[i] if p != 0]
                colors  = [colors[X[i].index(p)] for p in XX]
                Sobel_legends = [Sobel_legends[X[i].index(p)] for p in XX]
                X[i]    = XX
                axes[i].pie(x=list(X[i]),
                    textprops=dict(c="w", weight="bold"), 
                    autopct=lambda pct : Pct(pct, size=size), 
                    pctdistance=pctdistance, colors=colors, radius=radius+0.4,
                    explode = [0.0 if p not in explode_id else 0.1 for p in range(len(X[i]))],
                    shadow=True, counterclock=True
                    )
            
                center_circle = plt.Circle((0, 0), radius=radius, fc='white')
                plt.gca().add_artist(center_circle)
                axes[i].legend(Sobel_legends, bbox_to_anchor=(0.4, 0.32, 0.5, 0.4), title=f"Canaux RGBA = {np.array(X[i]).astype('int')}")
    else:
        i = 0
        axes.hist(X[i], bins=bins, color=colors )
        axes.legend(legend, bbox_to_anchor=box)
        axes.text(x=bb_box['x'], y=bb_box['y'], s=s)
        if vline is True:
            axes.vlines(x=v_line, ymax=y_lim[1], ymin=y_lim[0], lw = vh_lw, colors="k")
        
        axes.set_xlabel(xlabel[i],  weight="bold")
        axes.set_ylabel(ylabel[i],  weight="bold")
        axes.set_title(titles[i],  weight="bold")
        axes.set_ylim(y_lim)
      
    plt.show()

def boxplot(X , figsize : tuple = (4, 4), 
        xlabel : str = 'box plot', ylabel : str = 'N. Plantes', 
        save : bool = False, name : str = "boxplots.png"
        ):
    
    plt.figure(figsize=figsize)
    plt.boxplot(X, showcaps=True, widths=0.5, labels=[xlabel], notch=True, 
                patch_artist=True, showmeans=True, manage_ticks=True, 
                )
    plt.ylabel(ylabel=ylabel)
    if save is True : plt.savefig(f"images/{name}")
    plt.show()
                          
def rect_box(
        axes    : list,
        x       : list,
        y       : list,
        xmin    : list,
        xmax    : list, 
        ymin    : list,
        ymax    : list,
        color   : str  = 'k',
        s       : list = ['$range = [90, 250]$', '$range = [90, 250]$'],
        text    : bool = False ,
        annot   : bool = False
        ):
    
    for i, ax in enumerate( axes ):
        ax.vlines(x=x[i][0], ymin=ymin[i], ymax=ymax[i], lw=2, colors=color)
        ax.vlines(x=x[i][1], ymin=ymin[i], ymax=ymax[i], lw=2, colors=color)

        ax.hlines(y=y[i][0], xmin=xmin[i], xmax=xmax[i], lw=2, colors=color)
        ax.hlines(y=y[i][1], xmin=xmin[i], xmax=xmax[i], lw=2, colors=color)

        if text is True:
            ax.text(x=xmax[i]+100, y=ymax[i]-70, s = s[i] )
        if annot is True:
            ax.annotate(text="bonding box", xy=(xmax[i], ymax[i]-40), arrowprops=dict(arrowstyle='->', color="k"), 
                     xytext=(xmax[i] + 150, ymax[i]-40), color="b", weight='bold' )
    
    plt.show() 