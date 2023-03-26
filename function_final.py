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

take_first=lambda x:x[0][0]
take_second=lambda x:x[0][1]

def iprint(img):
    plt.figure()
    plt.imshow(img,cmap='gray')

def print_line(img,lines):
    img_copy = np.copy(img)
    for line in lines:
        for x2,y2,x1,y1 in line:
            cv2.line(img_copy,(x1,y1),(x2,y2),(255,0,0),1)
    iprint(img_copy)
    return(img_copy)

def hough_line(img):
    
    #image 1920*1080
    minline = img.shape[0] // 6
    accu = img.shape[0] // 30
        
    # Adaptative mean Thresholding 
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    binary = cv2.bitwise_not(binary)
    #iprint(binary)
    
    # Perform Hough Transform on the edges images
    lines = cv2.HoughLinesP(binary, 1, np.pi/180,accu, minLineLength=minline, maxLineGap=10)
    
    return lines

def rotate(img,lines):
    
    # Il faut que la grille soit au centre de l'image
    
    listlines = list(np.copy(lines))
    listlines.sort(key=take_first)    
    lines =np.array(listlines)

    center = tuple(map(lambda x: x/2, img.shape[:2]))
    
    x = center[0]
    mid=get_index(lines[:,0,0],x)+1
    
    i=0
    maxligne=0
    
    angle = 0
    while(abs(lines[mid+i][0][0]-x) < 150):
        i+=1
        line=lines[mid+i,0,:]
        x2,y2,x1,y1 = line
        u=np.array((x2-x1,y2-y1))
        d = np.linalg.norm(u)
        if(d > maxligne):
            maxligne = d
     
            angle=np.degrees(np.arctan2(u[1],u[0])) % 90
            angle2 = angle-90
            
            if(abs(angle2)<abs(angle)):
                angle=angle2
            
      
    rotated = skt.rotate(img,angle,center=center) # rotation in counterclockwise direction
    rotated=img_as_ubyte(rotated)
    return(rotated)
        
def get_frame(tabx):
    n = len(tabx)
    copy = [tabx[0]]
    for i in range(1,n-1):
        if(tabx[i]-tabx[i-1]>1):
            copy.append(tabx[i-1])
            copy.append(tabx[i])
            
    copy.append(tabx[n-1])
    copy=list(set(copy))
    copy.sort()
    
    return(np.array(copy))
        
def get_index(tab,elt):
    if(elt<tab[0]):
        return 0 
    for i in range(1,len(tab)-1):
        if(tab[i] < elt < tab[i+1]):
            return i
    return len(tab)-1
            
def get_case(img,lines):
    
    center = tuple(map(lambda x: x/2, img.shape[:2]))

    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.degrees(np.arctan2( y1-y2,  x1-x2))
            if 89 < abs(int(angle)) < 91: # si Ligne verticales
                vertical_lines.append(line)
            if abs(int(angle)) < 1 or 179< abs(angle)<181: #si ligne horizontales
                horizontal_lines.append(line)
    
    len_v,len_h = len(vertical_lines),len(horizontal_lines)
    if(len_v*len_h == 0):
            print("Lignes verticales ou horizontales non détectés")
            return ;
    
    vertical_lines.sort(key=take_first)
    horizontal_lines.sort(key=take_second)
    vertical_lines = np.array(vertical_lines)
    horizontal_lines = np.array(horizontal_lines)
    
    #imgcop=print_line(img, horizontal_lines)
    #print_line(imgcop, vertical_lines)
    
    tabx=get_frame(vertical_lines[:,0,0])
    taby=get_frame(horizontal_lines[:,0,3])  
    
    meanx = np.mean(tabx)
    meany = np.mean(taby)
    
    cv2.circle(img,(int(meanx),int(meany)),2,(255,0,0),1)

    
    case_x= np.array([i for i in tabx if meanx-220 < i < meanx+220])    
    len_v = len(case_x)
    deltax = case_x[1:len_v]-case_x[0:len_v-1]
    
    taille_case =int(np.ceil(np.mean([i for i in deltax if i > np.mean(deltax)]))) 
    taille_intercase = int(np.ceil(np.mean([i for i in deltax if i < np.mean(deltax)])))
    
    listcase=isole_case(img, taille_case, taille_intercase, tabx, taby)
    
    return(listcase)
    
   
def isole_case(img,taillecase,taille_intercase,tabx,taby):
    
    
    demi = 25
    taille_case_voulue = 40
    taille_case_actuel = 2*demi
    facteur_resize =  taille_case_voulue / taille_case_actuel

    listcase=[]
    
    deltax=tabx[1:len(tabx)]-tabx[0:len(tabx)-1]
    deltay=taby[1:len(taby)]-taby[0:len(taby)-1]
    
    """
    global echap_tabx
    echap_tabx = tabx
    
    global echap_deltay
    echap_deltay = deltay
    
    global echap_taby
    echap_taby = taby
    
    print("taille case : "+str(taillecase))
    """
    
    for i,dx in enumerate(deltax):
        if(taillecase-10<dx<taillecase+10):
            xo=tabx[i]+taillecase//2
            break
            
        
    
    for i,dy in enumerate(deltay):
        if(taillecase-10<dy<taillecase+10):
            yo=taby[i]+taillecase//2
            break
    
    #print("xo,yo : "+str(xo)+'\t'+str(yo)+'\n')
    #print(len(tabx))

    for j in range(9):
        for i in range(9):
            
            x= xo+i*taillecase+(i+1)*taille_intercase
            y= yo+j*taillecase+(j+1)*taille_intercase
            
            ix = get_index(tabx, x)
            iy = get_index(taby, y)
            #Probleme si on atteint le dernier indice
            
            if(ix<len(tabx)-1):
                x = (tabx[ix]+tabx[ix+1])//2
            if(iy<len(taby)-1):
                y = (taby[iy]+taby[iy+1])//2
            
            
            case = img_as_ubyte(skt.resize(img[y-demi:y+demi,x-demi:x+demi],(20,20),anti_aliasing=(True)))
            listcase.append(case)
            
            cv2.circle(img,(x,y),3,(255,0,0),2)
    
    return(listcase)
    
            
def list_print(tabcase):
    
    plt.figure()
    
    
    for i in range(9):
        for j in range(9):
            plt.subplot(9,9,9*i+j+1)
            plt.axis("off")
            plt.imshow(tabcase[9*i+j],cmap='gray')
                        
    
        
    
def lecture_grille(img):
    img = img_as_ubyte(rgb2gray(img))
    img=cv2.medianBlur(img,5)
    iprint(img)

    lines = hough_line(img)
    rotated = rotate(img,lines)
    lines_rot = hough_line(rotated)
    res = get_case(rotated, lines_rot)
    
    iprint(rotated)
    list_print(res)
    
    res=np.array(res)
    
    res=np.reshape(res[:,:,:],(9,9,20,20))
    #res=np.transpose(res,(1,0,2,3))
    
    return(res)


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
                    
            #Orientation 90 ET 270 a verifier
            #tab_to_return[i,j] = num
            if (orientation == 0):
                tab_to_return[i,j] = num
            elif (orientation == 90):
                tab_to_return[j,i-8] = num       
            elif (orientation == 180):
                tab_to_return[8-i,8-j] = num
            elif (orientation == 270):
                tab_to_return[8-j,i] = num    
            else:
                print("erreur orientation\n")
                
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
    x_resize = 20
    y_resize = 20

    #Import data reference
    data = Path().cwd()/ '..'/'Projet_sudoku'/ 'data' / 'data_train' / 'num_police2bis'
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
    
    
    plot_num_model(tab_num, 6)
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
x_resize = 20
y_resize = 20

img = cv2.imread("../Projet_sudoku/data/data_test/Sudoku/5.jpg")

iprint(img)
list_case = lecture_grille(img)
#iprint(list_case[0][1])

#Import data test
tab_num_test = np.zeros((9,9,x_resize,y_resize))

for i in range(list_case.shape[0]):
    for j in range(list_case.shape[1]): 
        img = img_as_float(list_case[i][0])
        #x, y =  i//9, i%9 #Row, Column
        tab_num_test[i,j] = img 

plot_9nums_test(tab_num_test, 7)
result = main_reco(tab_num_test, 0)

plt.figure(8)
plt.imshow(list_case[0][1])