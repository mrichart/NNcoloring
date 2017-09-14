import som as mySom
import pickle
import os.path
import numpy
from scipy import misc
from skimage import color

"""
Create a dataset of images from trainSOM.
Images are 256*256 pixels.
Select the 10% of the pixels of each image randomly.
"""
if os.path.exists('pkls/datasetSOM.pkl'):
    dataFile = open ('pkls/datasetSOM.pkl', 'rb')
    dataset = pickle.load (dataFile)
    dataFile.close()
else:
    print ('Creating dataset...')
    dataset = mySom.createDatasetSom ('dataset/Opencountry', 0.3)
    print ('Dataset of size: ' + str(dataset.shape))
    dataFile = open ('pkls/datasetSOM.pkl', 'wb')
    pickle.dump (dataset, dataFile)
    dataFile.close()
    

for mapsize in [[3,3],[5,5],[10,10],[20,20]]:
    print ('Training SOM with mapsize: ' + str(mapsize))
    som = mySom.defineAndTrainSOM (dataset, mapsize, 5, 10)
    fileObject = open ('pkls/trainedSOM'+str(mapsize[0])+'x'+str(mapsize[1])+'.pkl','wb') 
    pickle.dump (som, fileObject)
    fileObject.close()
    
    ####print SOM###
    size_x, size_y = mapsize[0], mapsize[1]
    codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
    L = numpy.zeros(size_x*size_y)
    L = L.reshape(-1,1)
    img = numpy.concatenate((L, codebook),1)
    img = img.reshape(size_x,size_y,3)
    img[:,:,0]=50
    misc.imsave('SOM' + str(size_x) +'x'+str(size_y)+'_L50.png',color.luv2rgb(img))
    
    
