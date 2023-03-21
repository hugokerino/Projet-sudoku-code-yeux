# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:03:57 2023

@author: hugob
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skt
from skimage.util import img_as_ubyte
import copy
from math import floor


import scipy.signal
from pathlib import Path
from skimage import io
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import transform
from skimage import filters
from matplotlib import pyplot as plt
from math import floor
import random

### Ryan functions ###
def iprint(img):
    plt.figure()
    plt.imshow(img,cmap='gray')

def take_first(array):
    return array[0][0]

def take_second(array):
    return array[0][1]

def hough_line_v1(img):
    minline = img.shape[0] // 10
    accu = img.shape[0] // 100
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    
    # Adaptative mean Thresholding 
    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    binary = cv2.bitwise_not(binary)
    
    # Perform Hough Transform on the edges images
    lines = cv2.HoughLinesP(binary, 1, np.pi/180,accu, minLineLength=minline, maxLineGap=10)
    
    return lines


def rotate(img,lines):
    center = tuple(map(lambda x: x/2, img.shape[:2]))
    max_ligne = 0 
    Ux=np.array((1,0))
    for line in lines :
        x2,y2,x1,y1 = line[0]
        u=np.array((x2-x1,y2-y1))
        d = np.linalg.norm(u)
        
        if(d > max_ligne):
            max_ligne = d
            
            if(np.dot(u,Ux)<0):
                angle=np.degrees(np.dot(-u,Ux)/(np.linalg.norm(u)*np.linalg.norm(Ux)))
            else:
                angle=np.degrees(np.dot(u,Ux)/(np.linalg.norm(u)*np.linalg.norm(Ux)))
            
            
            print(angle)
            angle=angle%90
            print(angle)

    rotated = skt.rotate(img,angle,center=center) # rotation in counterclockwise direction
    rotated=img_as_ubyte(rotated)
    return(rotated)


def resize(img,list_x): 
    ## Je peux faire un resize en prenant en compte la taille effective des case ou alors
    # en estimant la taille d'une case connaissant le lien taille grille taille en pixel.
    
    deltax = list_x[1:len(list_x)]-list_x[0:len(list_x)-1]
    meanx = np.mean(deltax)
    meanDx = int(np.mean( [i for i in deltax if i >= meanx ]))
    
    taille_case = meanDx
    facteur_resize = (50/taille_case)
    width = int(img.shape[1] * facteur_resize)
    height = int(img.shape[0] *facteur_resize)
    dim = (width,height)
    resized = cv2.resize(img,dim)

    return resized


def get_frame(lines):
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.degrees(np.arctan2( y1-y2,  x1-x2))
            if 89.9 < abs(int(angle)) < 90.1: # si Ligne verticales
                vertical_lines.append(line)
            if abs(int(angle)) < 0.1 or 179.9< abs(angle)<180.1: #si ligne horizontales
                horizontal_lines.append(line)
    
    
    len_v = len(vertical_lines)
    len_h = len(horizontal_lines)
    if(len_v*len_h == 0):
            print("Lignes verticales ou horizontales non détectés")
            exit(1)
            
    
    vertical_lines.sort(key=take_first)
    vertical_lines = np.array(vertical_lines)
    horizontal_lines.sort(key=take_second)
    horizontal_lines = np.array(horizontal_lines)
    
    tab_x = []
    tab_y = []
    
    
    # Pour ne prendre que les lignes intérieurs et extérieurs à chaque case
    tab_x.append(vertical_lines[0,0,0])  
    for j in range(len_v-2):
        
        if((vertical_lines[j+1,0,0]-vertical_lines[j,0,0])>2):
            # Si l'écart entre une ligne et la prochaine est plus grad que deux cela veut dire qu'on est
            # passé au cadre suivant, donc on peut les ajouter tous les deux au frames
            tab_x.append(vertical_lines[j,0,0])
            tab_x.append(vertical_lines[j+1,0,0])
    tab_x.append(vertical_lines[len_v-1,0,0])  
    
    tab_y.append(horizontal_lines[0,0,3])
    for j in range(len_h-2):
        if((horizontal_lines[j+1,0,3] - horizontal_lines[j,0,3])>2):
            tab_y.append(horizontal_lines[j,0,3])
            tab_y.append(horizontal_lines[j+1,0,3])
    
    tab_y.append(horizontal_lines[len_h-1,0,3])
    
    # Il faut éliminer les lignes abhérentes
    
    return([np.array(tab_x),np.array(tab_y)])


def delete_outliers(tab_x):
    #Supprime lignes abhérentes
    n=len(tab_x)
    mid = n/2    
    
    if(n<2):
        exit(1)
        
    deltax = (tab_x[1:n])-np.array(tab_x[0:n-1])
    meanx = np.mean(deltax)    
    deltax2 = [i for i in deltax if i > meanx]
    
    
    # Cas ou il n'ya pas d'outliers
    if(np.var(deltax2)<9):
        # Si la variance est trop faible, cela signifie qu'il n'ya que deux groupes homogènes à la base
        #donc pas  de valeurs abhérentes
        return tab_x
    
    meanx = np.mean(deltax2)    
    outliers = [i for i,v in enumerate(deltax) if v > meanx]
    
    outliers1 = [i for i in outliers if i < mid]
    outliers2 = [i for i in outliers if i > mid] 
    
    if(len(outliers1)==0):
        x=-1
    else:
        x = np.max( outliers1 )
        
    if(len(outliers2)==0):
        y=len(tab_x)-1
    else:
        y = np.min( outliers2 )

    return(tab_x[x+1:y+1])


def isole_case_v2(img,list_x,list_y):
    list_case = []
    
    deltax = list_x[1:len(list_x)]-list_x[0:len(list_x)-1]
    deltay = list_y[1:len(list_y)]-list_y[0:len(list_y)-1]
    
    meanx = np.mean(deltax)
    meany = np.mean(deltay)

    meandx = int(np.mean([i for i in deltax if i<meanx ])) 
    meanDx = int(np.mean( [i for i in deltax if i >= meanx ])) +1
    meandy = int(np.mean([i for i in deltay if i<meany ])) 
    meanDy = int(np.mean( [i for i in deltay if i >= meany ])) +1
    
    dim = 9 
    demi = 20
    xo,yo = list_x[0]+meandx+meanDx//2,list_y[0]+meandy+meanDy//2
    for j in range(dim):
        for i in range(dim):
            
            x= xo + (i+1)*meandx + i*meanDx
            y= yo + (j+1)*meandy + j*meanDy
                        
            case= [img[y-demi:y+demi,x-demi:x+demi,:],(x,y)]
            list_case.append(case)
            
    return list_case


def lecture_grille(img):
    img  = cv2.medianBlur(img, 5)
    lines = hough_line_v1(img)
    #rotated = rotate(img,lines)
    
    #lines = hough_line_v1(rotated)
    frame = get_frame(lines)
    list_x = delete_outliers(frame[0])

    resized = resize(img,list_x)
    #resized = cv2.medianBlur(resized,5)
    
    lines = hough_line_v1(resized)
    frame = get_frame(lines)
    list_x = delete_outliers(frame[0])
    list_y = delete_outliers(frame[1])

    list_case = isole_case_v2(resized,list_x,list_y)
    
    return list_case    
 

def print_list_images(listcase):
    plt.figure()
    for i in range(9):
        for j in range(9):
            plt.subplot(9,9,9*i+j+1)
            plt.axis("off")
            plt.imshow(listcase[9*i+j][0])



### Hugo functions ### 
def img_is_empty(img):
    if (np.mean(filters.sobel(img))) < 0.02 : return True # Regarde si l'image contient des HF
    else : return False

# Trouve l'orientation de l'image
def find_orientation(tab_num_to_reco, tab_num_model):
    possible_orientation = [0,90,180,270] # Differente orientations testées
    tab_orientation = []
    
    for i in range(tab_num_to_reco.shape[0]):
        for j in range(tab_num_to_reco.shape[1]):
            if (img_is_empty(tab_num_to_reco[i,j,:,:]) == True) : # Si image vide on passe à la suivante
                continue
            
            max_inter_corr = 0
            
            # Corrélation pour chaque img de tab_num_to_reco avec tab_num_model
            for k in range(len(tab_num_model)):
                inter_corr = scipy.signal.correlate(tab_num_to_reco[i,j,:,:],tab_num_model[k][0],mode = 'same',method='fft') 
                test = np.max(inter_corr)
                if test > max_inter_corr :
                    max_inter_corr = test
                    orientation = tab_num_model[k][2]
            tab_orientation.append(orientation)
        
    nbr_occurence = (tab_orientation.count(0),tab_orientation.count(90),tab_orientation.count(180),tab_orientation.count(270))
    orientation_f = possible_orientation[nbr_occurence.index(max(nbr_occurence))]
    
    return orientation_f



def number_recognition(tab_num_to_reco, tab_num_model, orientation):
    possible_orientation = [0,90,180,270]
    tab_to_return = np.zeros((tab_num_to_reco.shape[0],tab_num_to_reco.shape[1]))
    
    for i in range(tab_num_to_reco.shape[0]):
        for j in range(tab_num_to_reco.shape[1]):
            if (img_is_empty(tab_num_to_reco[i,j,:,:]) == True) : # Si image vide numéro trouvé égale 0
                tab_to_return[i,j] = 0
                continue
        
            max_inter_corr = 0
            
            for k in range(possible_orientation.index(orientation),len(tab_num_model),4):
                img = np.copy(tab_num_to_reco[i,j,:,:])
                img = (img- np.mean(img))/np.std(img)
                inter_corr = scipy.signal.correlate(img,tab_num_model[k][0],mode = 'same',method='fft') 
                #inter_corr = scipy.signal.correlate(tab_num_to_reco[i,j,:,:],tab_num_model[k][0],mode = 'same',method='fft') 
                test = np.max(inter_corr)
                if test > max_inter_corr :
                    num = tab_num_model[k][1]
                    max_inter_corr = test
                    
            tab_to_return[i,j] = num
            #print(f"({i},{j}) = {num}")
    return tab_to_return


# Ecriture dans un .txt des numéro trouvé dans tab_num_find
def write_sudoku(tab_num_find):
    sudoku_txt = open("sudoku.txt","w")
    
    for row in range(tab_num_find.shape[0]):
        for colum in range(tab_num_find.shape[1]):
            char = str(round(tab_num_find[row,colum]))+' '
            sudoku_txt.write(char) 
        sudoku_txt.write("\n")
    
    sudoku_txt.close()


def main_reco(tab_num_to_reco,orientation_initial):
    #Global parameters
    x_resize = 30
    y_resize = 30

    #Import data reference
    data = Path().cwd() / 'data' / 'data_train' / 'num_police2bis'
    k = 1
    tab_num = []
    for f in data.rglob("*.jpg"):
        img = rgb2gray(img_as_float(io.imread(f))) #Normalization
        img = (img - np.mean(img))/np.std(img) #Standardization
        tab_num.append((img,k,0)) # Reference images
        tab_num.append((transform.rotate(img,90),k,90))
        tab_num.append((transform.rotate(img,180),k,180))
        tab_num.append((transform.rotate(img,270),k,270))
        k += 1
    
    orientation_modif = find_orientation(tab_num_to_reco,tab_num)
    tab_num_find = number_recognition(tab_num_to_reco, tab_num, orientation_modif)
    write_sudoku(tab_num_find)
    
    return orientation_modif + orientation_initial

  
def plot_num_model(tab_model,i):
    plt.figure(i)
    plt.title("Model number")
    z = 1
    for k in range(0,36,4):
        plt.subplot(3,3,z)
        plt.imshow(tab_model[k][0])
        plt.title(str(k))
        z+=1
        
def plot_9nums_test(tab_num_test,i):
    len_ = len(tab_num_test)-1
    plt.figure(i)
    plt.title("Test number")
    for i in range(9):
        k = random.randint(0,len_)
        plt.subplot(3,3,i+1)
        plt.imshow(tab_num_test[k][0])


### Main function ###
plt.close("all")
x_resize = 40
y_resize = 40

img = cv2.imread("data/data_test/Sudoku/5.jpg")

iprint(img)
list_case = lecture_grille(img)
print_list_images(list_case)

#Import data test
x_resize = 40
y_resize = 40
tab_num_test = np.zeros((9,9,x_resize,y_resize))

for i in range(len(list_case)):
    img = rgb2gray(img_as_float(list_case[i][0]))
    x, y =  i//9, i%9 #Row, Column
    tab_num_test[x,y] = img 

plot_9nums_test(tab_num_test, 9)

result = main_reco(tab_num_test, 0)