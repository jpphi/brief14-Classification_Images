import os
import numpy as np
from skimage.io import imread
from skimage.transform import rescale, resize
import copy
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def creatpkl(data, rep_donnees, rep_dossier, largeurimg= 80, hauteurimg= 80, recadrer= False, contraste= (0,0), ngris= False):
    
    data["description"] = f'Resized ({largeurimg}x{hauteurimg}) animal images in rgb'
    for el in rep_dossier:
        repertoire= fr"{rep_donnees}/{el}/"
        
        for file in os.listdir(repertoire):
            #print(file)
            
            if file[-3:] in {'jpg', 'png'}: # "file" est il du type jpg ou png? sinon on l'ignore "cat1060a.bmp" est ignoré
                if ngris: # on génère un fichier en noir et blanc
                    img = imread(os.path.join(repertoire, file),  as_gray= True)
                else: # On ne traite ici que les rgb
                    rgbok= False # À priori...!
                    img = imread(os.path.join(repertoire, file))
                    
                    # On ne prendra que les images en RGB
                    if len(img.shape) != 3: # La matrice doit être de dimension 3
                        print(f"{file} n'est pas en rgb (hauteur, largeur, 3). shape= {img.shape}")                
                    # La matrice doit être de dimension 3 avec la 3ème dimension = 3 (vérifié en 2 temps car risque de plantage sur shape[2]
                    #   si dimension matrice< 3)    
                    elif img.shape[2] != 3:
                        print(f"{file} n'est pas en rgb (hauteur, largeur, 3). shape= {img.shape}")
                    else:
                        rgbok= True

                if ngris or rgbok:
                    #print(f"On traite le fichier: {file}")

                    if recadrer:
                        img= decimg(img)

                    if contraste!= (0,0):
                        img= imgcont(img, seuilb= contraste[0], seuilh= contraste[1])

                    img = resize(img, (largeurimg, hauteurimg)) #[:,:,::-1]

                    data['label'].append(el[:-4]) # Le :-4 Permet d'enlever "Head" du nom du fichier et de ne concerver que le nom de l'animal
                    data['filename'].append(file)
                    data['data'].append(img)
            
            # Tagg de l'image elle même
            if ngris:
                nfich= f"animal_faces_{largeurimg}x{hauteurimg}px_carre_{recadrer}_ct_{contraste!=(0,0)}_nb.pkl"
            else:
                nfich= f"animal_faces_{largeurimg}x{hauteurimg}px_carre_{recadrer}_ct_{contraste!=(0,0)}.pkl"

    joblib.dump(data, nfich)

def decimg(img, affiche= False):
    """ docstring fonction decimg (découpe image)
    découpe l'image pour qu'elle soit en format carré.
    Valeur retournée: image au format carré.
    Paramètres:
    img: Image,
    affiche: paramètre d'affichage.
    """
    
    if img.shape[0] == img.shape[1]: #format carré pas de problème
        img_ret= img

    elif img.shape[0] < img.shape[1]: #format "paysage" redimensionner et couper !
        l= (img.shape[1] - img.shape[0])//2
        m= img.shape[1] - l
        img_ret= img[:,l:m].copy()

    else : # img.shape[0] > img.shape[1] => format "portrait" redimensionner et couper !
        l= (img.shape[0] - img.shape[1])//2
        m= img.shape[0] - l
        img_ret= img[l:m,:].copy()

    #print(img[:,l])

    # Création de l'espace d'affichage
    if affiche:
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

        axes[0].imshow(img, cmap='BrBG')
        axes[0].set_title("Image originale")

        axes[1].imshow(img_ret, cmap='BrBG')
        axes[1].set_title("Image découpée et centrée")

        axes[2].hist(img_ret.flatten(), bins=range(1,255)) # range(256) provoque un comportement étrange !!!!
        axes[2].set_title("Histogramme image découpée")
        #axes[0].set_xlim(0, 512)
        #axes[0].set_ylim(512, 0)
        plt.tight_layout()
        plt.show()

        print(f"Shape image originale: {img.shape}")
        print(f"Shape image découpée: {img_ret.shape}")
        
    return img_ret



def imgcont(img, affiche= False, seuilb= 30, seuilh= 220):
    """ docstring fonction deccont (contraste image)
    accentue le contraste de l'image.
    Valeur retournée: image contrastée.
    Paramètres:
    img: Image,
    affiche: paramètre d'affichage.
    seuilb: seuil bas
    seuilb: seuil haut
    
    Note: il serait souhaitable de retravailler les seuil et de tenir compte de la formule RGB -> niveau de gris (x*R + y*G + z*B)
    """
    imgct= copy.deepcopy(img)

    imgct[imgct<seuilb]= 0
    imgct[imgct>seuilh]= 255
    
    if affiche:
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))

        axes[0].imshow(img, cmap='BrBG')
        axes[0].set_title("Image originale")

        axes[1].hist(img.flatten(), bins=range(0,255))
        axes[1].set_title("Histogramme image originale")

        axes[2].imshow(imgct, cmap='BrBG')
        axes[2].set_title("Image découpée et centrée")

        axes[3].hist(imgct.flatten(), bins=range(1, 255)) # 0,255 ou (256) comportement étrange !!! valeur > 1200 pourquoi ? matrice OK
        axes[3].set_title("Histogramme image contrastée")

        #plt.tight_layout()
        plt.show()
        
    return imgct

def plot_bar(y, loc='left', relative=True):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5
     
    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]
     
    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'
         
    xtemp = np.arange(len(unique))
     
    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, unique, rotation=45)
    plt.xlabel('equipment type')
    plt.ylabel(ylabel_text)


def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()
 
    np.fill_diagonal(cmx_zero_diag, 0)
 
    fig, ax = plt.subplots(nrows=3)
    fig.set_size_inches(30, 20)
    [a.set_xticks(range(len(cmx)+1)) for a in ax]
    [a.set_yticks(range(len(cmx)+1)) for a in ax]
    #for a in ax:print(a) # c'est quoi les valeur de a ????
         
    im1 = ax[0].imshow(cmx, vmax=vmax1)
    ax[0].set_title('Valeur absolue.')
    ax[0].set_xlabel("Classe")
    
    im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
    ax[1].set_title('Valeur relative (%)')
    
    im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
    ax[2].set_title('% and 0 diagonal')
 
    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.tight_layout()
