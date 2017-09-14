from scipy import misc
from skimage import color
import numpy as np
import sompy
import os
import pickle
import random
from sompy.visualization.bmuhits import BmuHitsView

def createDatasetSom (trainingDir, ratio):
    dataset = None    #trainig data

    for filename in os.listdir(trainingDir):
        if filename.endswith(".jpg"):
            img = misc.imread(trainingDir+'/'+filename) #load image from file
            if __debug__:
                print ('Processing image: ' + filename)

            imgLuv = color.rgb2luv(img)                 #transform the image to CIE LUV
            imgUV = imgLuv[:,:,1:]                      #get the U and V components from image
            data = imgUV.reshape (img.shape[0]*img.shape[1], 2)  #convert to an array of UV pixels
            
            #reduce the size of the data set, sampling random
            reducedData = np.array (random.sample (data.tolist(), int(data.size*ratio)))



            if dataset is None:
                dataset = reducedData
            else:
                dataset = np.concatenate ((dataset, reducedData))

    return dataset


def defineAndTrainSOM (dataset, mapsize, rough_len=None, finetune_len=None):
    #Simple function to train the SOM. Returns the trained SOM
    
    """
    Create the network.
    It uses the dataset to get the input nerunons.
    Use varianze normalization ((data-mean)/standard dev).
    Use PCA to initialize weights.
    Use a gaussian function to define the neigbourhood.
    Currently the library only supports batch training.
    """
    som = sompy.SOMFactory.build(dataset, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='coloring')

    """
    Train the network.
    The SOM training consists in 2 phases: the rough and the finetune one.
    Rough organization of the map: large neighborhood, big initial value for 
    learning coefficient. Short training.
    Finetuning the map after rough organization phase. Small neighborhood, 
    learning coefficient is small already at the beginning. Long training.
    """
    som.train(n_job=1, verbose='info', train_rough_len=rough_len, train_finetune_len=finetune_len)

    return som

def getCodeword (som, pixelUV):
    #Returns the representative in the codebook for the given U,V values
    codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
    proj = som.project_data(pixelUV)
    return codebook[proj]
    
def getBMU (som, pixelUV, codebook):
  #Returns the index of the Best Matching Unit (winning neuron)
  best_dist=1000000
  best_index=0
  for i in range(len(codebook)):
            dist = np.linalg.norm(codebook[i] - pixelUV[0])
            if dist < best_dist:
                best_dist = dist
                best_index = i
    
  return np.array([best_index])

###Visualization Options###

##View the contribution of each component to the neurons
#view2D  = sompy.mapview.View2D(4,4,"Components Map",text_size=12)
#view2D.show(som, col_sz=4, which_dim="all", desnormalize=True)

##View the number of samples each neuron 
#vhts  = BmuHitsView(4,4,"Hits Map",text_size=12)
#vhts.show(som, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)

##Make clusters of neurons
#som.cluster(4)
#hits  = sompy.visualization.hitmap.HitMapView(4,4,"Clustering",text_size=12)
#a=hits.show(som)

####################################

###Transform image to codebook colors###

#imgGrey = imgLuv[:,:,0]
#codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
##proj = som.bmu_ind_to_xy(som.project_data(data))
#proj = som.project_data(data)
#newData = codebook[proj]
#newImg = np.concatenate((imgGrey.reshape(256*256,1),newData),1)
#newImg = newImg.reshape(256,256,3)
#misc.imsave('newImgRGB5x5.png',color.luv2rgb(newImg))

###################################################

###Usage Example###

#directory = 'dataset/trainSOM'  #train files
#size of the output neurons map.
#mapsize = [5,5]  #25 output neurons
#som = defineAndTrainSOM (directory, mapsize)

#fileObject = open ('pkls/trainedSOM.pkl','wb') 
#pickle.dump (som, fileObject)
#fileObject.close()

