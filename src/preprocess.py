from scipy import misc
from skimage import color
import numpy as np
import os
import random
import pickle
import som as mySom

patch_sizes = [ [5,5] ]      #to be generated, sizes must be odd
directoryData = 'dataset/Opencountry'         #raw data
som_sizes = [[3,3]]  #soms to use
patches_per_img = 10000


def createDataSet (rawDataDir, patch_size, som):
    #Create the training set from a directory  with color images
    
    si, sj = patch_size[0], patch_size[1]
    
    cantFeatures = si*sj
    cantData = len(os.listdir(rawDataDir)) * patches_per_img
    
    X = np.zeros((cantData, cantFeatures))    #data
    y = np.zeros((cantData, 1), dtype=np.int)    #labels
    
    dataRow=0                      #count data rows
    for filename in os.listdir(rawDataDir):
        if filename.endswith(".jpg"):
            img = misc.imread(rawDataDir+'/'+filename)     #load image from file
            
            print ('        Processing image: ' + filename + ' ' + str(img.shape))
            
            imgLuv = color.rgb2luv(img)                     #transform the image to CIE LUV
            
            codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)

            #obtain the "patches" from each figure
            for n in range(0, patches_per_img):
              i = random.randrange(img.shape[0]-si)
              j = random.randrange(img.shape[1]-sj)
              #print ('    Processing patch: ' + str(j) + ', ' + str (i))
              subImg = imgLuv[i:i+si, j:j+sj, :]
              #print(subImg.shape)
              #misc.imsave('/tmp/parche'+str(x)+'_'+str(y)+'.png', color.luv2rgb(subImg))
              pixelUV = subImg[si//2, sj//2, 1:]                        # obtain the center pixel, only the U and V components
              pixelGroup = mySom.getBMU(som, pixelUV.reshape(1,-1), codebook) # get the group of the pixel (the Best Matching Unit of the SOM). For y NN
              #print (pixelGroup)
              
              patchL = subImg[:,:,0]                          # get the L components of the patch. For X NN
                      
              #print ('        Updating X...')
              X[dataRow] = patchL.reshape(1, si*sj)
              
              #print ('        Updating Y...')
              y[dataRow] = pixelGroup.reshape(1,-1)
                      
              #if X is None:
              #    X = patchL.reshape(1, si*sj)
              #else:
              #    print ('        Concatenating X...')
              #    X = np.concatenate((X, patchL.reshape(1, si*sj)))
                  
              #if y is None:
              #    y = pixelGroup.reshape(1,-1)
              #else:
              #    print ('        Concatenating y...')
              #    y = np.concatenate((y, pixelGroup.reshape(1,-1)))
              dataRow = dataRow+1

    return X, y


###################################################

#directorySom = 'dataset/Opencountry'            #som training images
#mapsize = [3,3]
#som = mySom.defineAndTrainSOM (directorySom, mapsize)

#somFile = open ('pkls/trainedSOM.pkl','wb') 
#pickle.dump (som, somFile)
#somFile.close()

for som_size in som_sizes:
    print ('Using som: [' + str(som_size[0]) + 'x' + str (som_size[1]) + ']')
    somFile = open ('pkls/trainedSOM' + str(som_size[0]) + 'x' + str (som_size[1])+'.pkl','rb')
    som = pickle.load (somFile)
    somFile.close()

    for patch_size in patch_sizes:
        print ('    With patch size: [' + str(patch_size[0]) + ',' + str (patch_size[1])+']')
        X, y = createDataSet (directoryData, patch_size, som)

        fileData = open ('pkls/data_patch' + str(patch_size[0]) + 'x' + str (patch_size[1])+'_som' + str(som_size[0]) + 'x' + str (som_size[1])+'.pkl','wb') 
        pickle.dump (X, fileData)
        fileData.close()

        fileLabels = open ('pkls/labels_patch' + str(patch_size[0]) + 'x' + str (patch_size[1])+'_som' + str(som_size[0]) + 'x' + str (som_size[1])+'.pkl','wb') 
        pickle.dump (y, fileLabels)
        fileLabels.close()

