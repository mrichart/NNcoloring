from scipy import misc
from skimage import color
import numpy
import sompy
import pickle

for mapsize in [[3,3],[5,5],[10,10],[20,20]]:
    somFile = open ('pkls/trainedSOM' + str(mapsize[0]) +'x'+str(mapsize[1])+'.pkl','rb')
    som = pickle.load (somFile)
    somFile.close()

    codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)

    L = numpy.zeros(mapsize[0]*mapsize[1])
    L = L.reshape(-1,1)

    img = numpy.concatenate((L, codebook),1)
    img = img.reshape(mapsize[0],mapsize[1],3)
    img[:,:,0]=50

    misc.imsave('SOM' + str(mapsize[0]) +'x'+str(mapsize[1])+'_L50.png',color.luv2rgb(img))

