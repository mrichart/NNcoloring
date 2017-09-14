import som as mySom
import sys
import pickle
import numpy as np
from scipy import misc
from skimage import color

if len(sys.argv) < 2:
    print ('A filename is needed')
    sys.exit(1) 

filename = sys.argv[1]
img = misc.imread(filename)
imgLUV = color.rgb2luv(img)   
imgL = imgLUV[:,:,0]
imgUV = imgLUV[:,:,1:]

for mapsize in [[2,2], [3,3], [5,5], [10,10]]:
    somFile = open ('pkls/trainedSOM'+str(mapsize[0])+'x'+str(mapsize[1])+'.pkl','rb') 
    som = pickle.load (somFile)
    somFile.close()    
    
    imgCode = mySom.getCodeword(som, imgUV.reshape(256*256,2))
    refImg = np.concatenate((imgL.reshape(256*256,1), imgCode),1)
    refImg = refImg.reshape(256,256,3)
    misc.imsave(filename.replace('.jpg','') + '_reference_' +str(mapsize[0])+'x'+str(mapsize[1])+'.png', color.luv2rgb(refImg))

