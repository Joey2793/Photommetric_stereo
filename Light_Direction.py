import numpy as np
import math
from scipy import misc
import matplotlib.pyplot as plt
import imageio
import matplotlib.image as mpimg
import pfmio
import cv2
import os
import netpbmfile


def find_centroid(gray_mask_array):
    '''
    param gray_mask_array:  gradyscaled masked ball array
    return: center of metal_ball
    assume: grayscale ball only contain 0 value and other?
    '''
    count = 0
    x = 0
    y = 0
    for index, value in np.ndenumerate(gray_mask_array):
        if value > 0:
            x = x + index[0]
            y = y + index[1]
            count +=1
    return (x/count,y/count)

def find_radius(gray_mask_array, centroid):
    r1 = 0
    r2 = 0
    for index, value in np.ndenumerate(gray_mask_array):
        if value > 0:
            r1 = max(r1,np.absolute(index[0]-centroid[0]))
            r2 = max(r2,np.absolute(index[1]-centroid[1]))
    #print(r1,r2)
    return r1
    #if r1 == r2:
    #    return r1
    #else:
    #    print("Wrong method to calcluate radius")
#test
#centroid = find_centroid(dir_ballgray1)
#print(centroid)
#print(find_radius(dir_ballgray1,centroid))





def find_shinypoint(gray_array,ball_mask_array):
    '''
    using dir_ball to mask orginal image
    :param gray_array:
    :param ball_mask_array:
    :return:
    '''
    temp_array = np.copy(gray_array)
    for i in range(len(ball_mask_array)):
        for j in range(len(ball_mask_array[0])):
            if ball_mask_array[i][j] == 0:
                temp_array[i][j] = 0
    #realarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #if (realarray == gray_array).all():
    #    print('tRUE')
    #else:
    #    print('false')
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(temp_array)
    #print("Max Gray value:",maxVal)
    return (maxLoc[1],maxLoc[0]) #x,y is reverse
    #for index,value in np.ndenumerate(temp_array):
    #    if value == 1:
    #        return (index[1],index[0])

def compute_light_intensity(gray_array, matte_mask_array):
    temp_array1 = np.copy(gray_array)
    for i in range(len(matte_mask_array)):
        for j in range(len(matte_mask_array[0])):
            if matte_mask_array[i][j] == 0:
                temp_array1[i][j] = 0
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(temp_array1)
    return maxVal



def compute_light_direction(imggray,dir_ballgray):
    '''
    
    :param imggray: grayscale img
    :param dir_ballgray: gray scale dir_ball
    :return: light_direction
    '''
    ball_centroid = find_centroid(dir_ballgray)
    #imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ball_shiny = find_shinypoint(imggray,dir_ballgray)
    ball_radius = find_radius(dir_ballgray,ball_centroid)

    nx = (ball_shiny[0]-ball_centroid[0])/ball_radius
    ny = (ball_shiny[1]-ball_centroid[1])/ball_radius
    nz = math.sqrt(1-nx*nx-ny*ny)

    N = np.array([nx,ny,nz])
    #print(N)
    R = np.array([0,0,1])
    #R = np.array([0,0,-1])
    L = (2*np.dot(N,R)*N - R)
    return L




def light_dir_ints(filefolder):
    '''

    :param filefolder: such as Apple or Pear
    :return:
    '''
    dir_1 = cv2.imread(os.path.join(filefolder,'mask_dir_1.png'))
    dir_2 = cv2.imread(os.path.join(filefolder,'mask_dir_2.png'))
    #mask_I = cv2.imread(os.path.join(filefolder,'mask_I.png'))
    #mask_Igray = cv2.cvtColor(mask_I, cv2.COLOR_BGR2GRAY)
    dir_gray1 = cv2.cvtColor(dir_1, cv2.COLOR_BGR2GRAY)
    dir_gray2 = cv2.cvtColor(dir_2, cv2.COLOR_BGR2GRAY)
    light = []
    #intensity = []
    #img_intensity = []
    for filename in os.listdir(filefolder):
        if filename.endswith('.pbm'):
            with open(os.path.join(filefolder, filename), encoding='utf-8', errors='ignore') as imgfile:
                img = pfmio.load_pfm(imgfile)
                img[np.isnan(img)] = 0
                #img = cv2.imread(filefolder+filename)
                imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # value : 0-1
                #img_intensity.append(imggray)
                light1 = compute_light_direction(imggray, dir_gray1)
                light2 = compute_light_direction(imggray, dir_gray1)
                light.append((np.array(light1) + np.array(light2)) / 2)
                #intensity.append(compute_light_intensity(imggray, mask_Igray))


    with open(os.path.join(filefolder, 'light_dir_output.txt'), 'w') as output_file:
        output_file.write('%d\n' % len(light))
        for l in light:
            output_file.write('%lf %lf %lf\n' % (l[0], l[1], l[2]))

    print('Done')


#light_dir_ints('Apple')

#light_dir_ints('Elephant')

#light_dir_ints('Pear')
