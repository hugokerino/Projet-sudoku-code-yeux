import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skt
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import copy
from math import floor

import cv2


plt.close("all")

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


def test_grille(img):
    return


img_rot1 = cv2.imread("grille sudoku/4.jpg")
img_rot2 = cv2.imread("grille sudoku/6.jpg") #Problème

    
img1 = cv2.imread("grille sudoku/1.jpg")
img2 = cv2.imread("grille sudoku/2.jpg")
img3 = cv2.imread("grille sudoku/3.jpg") 
img4 = cv2.imread("grille sudoku/5.jpg") # Problème
img5 = cv2.imread("grille sudoku/7.jpg")
img6 = cv2.imread("grille sudoku/8.jpg") # Problème


res=lecture_grille(img3)
iprint(res[0][1])




