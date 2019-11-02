import pfmio
import numpy as np
import Light_Direction
import os
import cv2
from matplotlib import pyplot as plt
import scipy.sparse.linalg as sp
import scipy.sparse


#apple_mask = cv2.imread('Apple/mask.png')
#apple_mask_gray = cv2.cvtColor(apple_mask, cv2.COLOR_BGR2GRAY)


def imgtomatrix(filefolder):
    mask_I = cv2.imread(os.path.join(filefolder, 'mask_I.png'))
    mask_Igray = cv2.cvtColor(mask_I, cv2.COLOR_BGR2GRAY)
    mask_object = cv2.imread(os.path.join(filefolder,'mask.png'))
    mask_ogray = cv2.cvtColor(mask_object, cv2.COLOR_BGR2GRAY)
    imgmatrix = []
    light_intensity =[]

    for filename in os.listdir(filefolder):
        if filename.endswith('.pbm'):
            with open(os.path.join(filefolder, filename), encoding='utf-8', errors='ignore') as imgfile:
                img = pfmio.load_pfm(imgfile)
                img[np.isnan(img)] = 0
                imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # value : 0-1
                empty_img = np.zeros_like(imggray)
                for index,value in np.ndenumerate(mask_ogray):
                    if value !=0:
                        empty_img[index] = imggray[index]

                imgmatrix.append(empty_img)
                light_intensity.append(Light_Direction.compute_light_intensity(imggray, mask_Igray))


    #light_intensity = sum(intensity)/len(intensity)
    #light_intensity_matrix = np.array(intensity)
    print('img to matrix Done')
    return imgmatrix, light_intensity #img list and light_intensity list

#imglist_apple = imgtomatrix('Apple')[0]
#light_intensity_apple= imgtomatrix('Apple')[1]


#read light direction file: light_dir_ouput.txt
def read_lights_dir(lights_folder):
    lights_dir = []
    with open(os.path.join(lights_folder, 'light_dir_output.txt'),encoding='utf-8',errors='ignore') as lfile:
        for num, line in enumerate(lfile, 1):
            if num == 1:
                continue
            else:
                w = line.split()
                lights_dir.append([float(w[0]), float(w[1]), float(w[2])])
    print('read light file done')
    return lights_dir

#light_dir_list_apple = read_lights_dir('Apple')

#print(light_dir_list)

#plt.imshow(imglist[0])
#plt.show()
#print(len(f[1]))

def photometricstereo(img_list, lights_dir_list, light_intensity_list,object_mask_gray):
    len1 = img_list[0].shape
    shaper = (len1[0], len1[1], 3)
    G = np.zeros(shaper)
    N = np.zeros(shaper)
    normal = np.zeros(shaper)
    render = np.zeros(len1)
    k = np.zeros(len1)
    I = np.zeros(len(img_list))
    L = np.zeros((len(img_list),3))

    for i, value in np.ndenumerate(lights_dir_list):
        m = i[0]
        L[i] = value * light_intensity_list[m]

    for index,value in np.ndenumerate(object_mask_gray):
        if value != 0:
            for m in range(len(img_list)):
                img = img_list[m]
                I[m] = img[(index[0], index[1])]
            temp_matrix = np.dot(np.transpose(L),L)
            G[index] = np.linalg.multi_dot([np.linalg.inv(temp_matrix),np.transpose(L),I])
            k[index] = np.linalg.norm(G[index])
            N[index] = np.dot(1/k[index],G[index])
            normal[index][0] = (N[index][0] + 1) / 2
            normal[index][1] = (N[index][1] + 1) / 2
            normal[index][2] = (N[index][2] + 1) / 2
            render[index] = k[index]*np.dot(N[index],[0,0,1])

    print('k and N calculation done')
    return k,normal,render,N  #k: albedo, normal and N: normal

def depth_calculation(normal_matrix,mask_gray):#normal_matrix: 600*800*3 #mask_gray_matrix: img_mask
    len2 = mask_gray.shape
    size = len2[0]*len2[1]
    #X = scipy.sparse.coo_matrix((2*size, size),dtype=np.int8)
    #M = np.zeros((2*size, size))
    b = np.zeros((2*size))
    mask_position = np.zeros((size,2))
    index = np.array([ x for x in range(size)]).reshape(len2[0],len2[1])#0~(600*800-1)
    rowp = [] #row_position
    colp = [] #col_position
    data = [] #to construct M matrix
    for i,value in np.ndenumerate(index):
        mask_position[value,0] = i[0]  #img row
        mask_position[value,1] = i[1]   #img column

    for d in range(size):
        row = int(mask_position[d,0])
        col = int(mask_position[d,1])
        nx = normal_matrix[row,col,0]
        ny = normal_matrix[row,col,1]
        nz = normal_matrix[row,col,2]



        #if row <len2[0]-1 and col < len2[1] - 1:
        if nz != 0:
            lnz = normal_matrix[row - 1, col, 2]
            rnz = normal_matrix[row + 1, col, 2]
            tnz = normal_matrix[row, col - 1, 2]
            if col < len2[1] - 2:
                bnz = normal_matrix[row, col + 1, 2]
            else:
                bnz = 0

            if rnz != 0 and bnz != 0:
                colp.append(index[row+1, col])
                rowp.append(2*d)
                data.append(1)
                colp.append(index[row, col])
                rowp.append(2*d)
                data.append(-1)
                b[2 * d] = -nx / nz

                #M[2*d, index[row+1, col]] = 1
                #M[2*d, index[row, col]] = -1
                #b[2*d] = -nx/nz   #(x+1,y)
                colp.append(index[row, col+1])
                rowp.append(2 * d+1)
                data.append(1)
                colp.append(index[row,col])
                rowp.append(2 * d+1)
                data.append(-1)
                #M[2*d+1, index[row, col+1]]   = 1    #(x, y+1)
                #M[2*d+1, index[row,col]] = -1
                b[2*d+1] = -ny / nz

        #elif row == len2[0]-1 and col == len2[1]-1:
            elif rnz == 0 and bnz == 0:
                colp.append(index[row-1, col])
                rowp.append(2 * d)
                data.append(1)
                colp.append(index[row, col])
                rowp.append(2 * d)
                data.append(-1)
                #M[2 * d, index[row-1, col]] = 1
                #M[2 * d, index[row, col]] = -1
                b[2 * d] = nx / nz  # (x-1,y)
                colp.append(index[row, col - 1])
                rowp.append(2 * d+1)
                data.append(1)
                colp.append(index[row, col])
                rowp.append(2 * d+1)
                data.append(-1)
                #M[2 * d + 1, index[row, col - 1]] = 1  # (x, y-1)
                #M[2 * d + 1, index[row, col]] = -1
                b[2 * d + 1] = ny / nz

        #elif row == len2[0] - 1 and col != len2[1] - 1:
            elif bnz == 0 and rnz != 0:
                colp.append(index[row - 1, col])
                rowp.append(2 * d)
                data.append(1)
                colp.append(index[row, col])
                rowp.append(2 * d)
                data.append(-1)
                #M[2 * d, index[row - 1, col]] = 1
                #M[2 * d, index[row, col]] = -1
                b[2 * d] = nx / nz  # (x-1,y)

                colp.append(index[row, col + 1])
                rowp.append(2 * d + 1)
                data.append(1)
                colp.append(index[row, col])
                rowp.append(2 * d + 1)
                data.append(-1)
                #M[2 * d + 1, index[row, col + 1]] = 1  # (x, y+1)
                #M[2 * d + 1, index[row, col]] = -1
                b[2 * d + 1] = -ny / nz

        #elif row != len2[0] - 1 and col == len2[1] - 1:
            elif rnz == 0 and bnz != 0:
                colp.append(index[row +1, col])
                rowp.append(2 * d)
                data.append(1)
                colp.append(index[row, col])
                rowp.append(2 * d)
                data.append(-1)
                #M[2 * d, index[row + 1, col]] = 1
                #M[2 * d, index[row, col]] = -1
                b[2 * d] = -nx / nz  # (x+1,y)

                colp.append(index[row, col - 1])
                rowp.append(2 * d + 1)
                data.append(1)
                colp.append(index[row, col])
                rowp.append(2 * d + 1)
                data.append(-1)
                #M[2 * d + 1, index[row, col - 1]] = 1  # (x, y-1)
                #M[2 * d + 1, index[row, col]] = -1
                b[2 * d + 1] = ny / nz
    row1 = np.array(rowp)
    col1 = np.array(colp)
    data1 = np.array(data)
    M = scipy.sparse.coo_matrix((data1, (row1, col1)), shape=(2*size, size))
    print('M and b matrix finished process')

    #X = scipy.sparse.coo_matrix(M)
    Z = sp.lsqr(M,b)[0].reshape(len2)
    print('Z value done')

    return Z


def show_map(filefolder):
    mask = cv2.imread(os.path.join(filefolder,'mask.png'))
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    imglist = imgtomatrix(filefolder)[0]
    light_intensity = imgtomatrix(filefolder)[1]
    light_dir_list = read_lights_dir(filefolder)
    map = photometricstereo(imglist, light_dir_list, light_intensity, mask_gray)
    albedo_map = map[0]
    normal_map = map[1]
    render_map = map[2]

    N_map = map[3]
    depth_map = depth_calculation(N_map,mask_gray)



    plt.imshow(render_map, cmap='gray')
    plt.title(filefolder + ' Re-rendered Picture')
    plt.show()
    plt.imshow(albedo_map, cmap='gray')
    plt.title(filefolder + ' Albedo Map')
    plt.show()
    plt.imshow(normal_map)
    plt.title(filefolder + ' Normal Map')
    plt.show()

    plt.imshow(depth_map, cmap='gray')
    plt.title(filefolder + ' Depth Map')
    plt.show()

#show_map('Apple')
#show_map('Elephant')
#show_map('Pear')
