from scipy import misc
from skimage import color
import numpy as np
import pickle
import som as mySom 
import os


patch_size = [5,5]
som_size = [3,3]
image_dir = 'dataset/Opencountry'  # 'dataset/Opencountry'

pipeline = pickle.load (open ('pkls/pipeline_patch' + str(patch_size[0]) + 'x' + str (patch_size[1])+'_som' + str(som_size[0]) + 'x' + str (som_size[1])+'.pkl', 'rb'))
som = pickle.load (open ('pkls/trainedSOM' + str(som_size[0]) + 'x' + str (som_size[1])+'.pkl','rb'))


for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        img = misc.imread(image_dir+'/'+filename) #load image from file
        imgLuv = color.rgb2luv(img)   
        imgGrey = imgLuv[:,:,0]
        originalUV = imgLuv[:,:,1:]
        
        img_predict_size = [img.shape[0] - patch_size[0] + 1, img.shape[1] - patch_size[1] + 1]
        print ('Predicting for ', filename, img.shape, img_predict_size) 
        
        X_predict = np.zeros ((img_predict_size[0]*img_predict_size[1], patch_size[0]*patch_size[1]))

        pos = 0
        for j in range(0, img.shape[0]-patch_size[0]+1, 1):    # con patch 5x5: range(0, 252, 1)
            for i in range(0, img.shape[1]-patch_size[1]+1, 1):             
                #print ('    Processing patch: ' + str(j) + ', ' + str (i))
                patchL = imgGrey[j:j+patch_size[0],i:i+patch_size[1]].reshape (1, patch_size[0]*patch_size[1])
                X_predict[pos,:] = patchL
                pos += 1  

        #print (X_predict)
        y_predict = pipeline.predict(X_predict)

        imgUV = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)[y_predict]

        originalGroup = mySom.getCodeword(som, originalUV.reshape(img.shape[0]*img.shape[1],2))
        refImg = np.concatenate((imgGrey.reshape(img.shape[0]*img.shape[1],1), originalGroup),1)
        refImg = refImg.reshape(img.shape[0],img.shape[1],3)
        misc.imsave('out/'+filename + '.reference.png',color.luv2rgb(refImg))

        originalGroup = mySom.getCodeword(som, originalUV.reshape(img.shape[0]*img.shape[1],2))
        midgray = np.full((img.shape[0]*img.shape[1]), 50).reshape(img.shape[0]*img.shape[1],1)
        refImg = np.concatenate((    midgray   , originalGroup),1)
        refImg = refImg.reshape(img.shape[0],img.shape[1],3)
        misc.imsave('out/'+filename + '.reference_uv.png',color.luv2rgb(refImg))
            
        imgGrey = imgGrey[(patch_size[0]//2):(img.shape[0]-patch_size[0]//2), (patch_size[1]//2):(img.shape[1]-patch_size[1]//2)]     # con patch 5x5: [2:254,2:254]
        newImg = np.concatenate((imgGrey.reshape(img_predict_size[0]*img_predict_size[1],1), imgUV.reshape(img_predict_size[0]*img_predict_size[1],2)),1)
        newImg = newImg.reshape(img_predict_size[0],img_predict_size[1],3)
        misc.imsave('out/'+filename + '.colored.png',color.luv2rgb(newImg))

        imgGrey = imgGrey[(patch_size[0]//2):(img.shape[0]-patch_size[0]//2), (patch_size[1]//2):(img.shape[1]-patch_size[1]//2)]     # con patch 5x5: [2:254,2:254]
        midgray = np.full((img_predict_size[0]*img_predict_size[1]), 50).reshape(img_predict_size[0]*img_predict_size[1],1)
        newImg = np.concatenate((  midgray      , imgUV.reshape(img_predict_size[0]*img_predict_size[1],2)),1)
        newImg = newImg.reshape(img_predict_size[0],img_predict_size[1],3)
        misc.imsave('out/'+filename + '.colored_uv.png',color.luv2rgb(newImg))


#img = misc.imread('dataset/land515.jpg')
#imgLuv = color.rgb2luv(img)   
#imgGrey = imgLuv[:,:,0]
#originalUV = imgLuv[:,:,1:]


#X_predict = np.zeros ((252*252, 25))

#pos = 0
#for j in range(0, 252, 1):
#    for i in range(0, 252, 1):
#        
#        #print ('    Processing patch: ' + str(j) + ', ' + str (i))
#        patchL = imgGrey[j:j+5,i:i+5].reshape (1, 25)
#        X_predict[pos,:] = patchL
#        pos += 1  
#                
#print (X_predict)
#y_predict = pipeline.predict(X_predict)
##np.set_printoptions(threshold=np.nan)
#print (np.unique(y_predict.reshape(252*252)))

#somFile = open ('pkls/trainedSOM.pkl','rb') 
#som = pickle.load (somFile)
#somFile.close()

#imgUV = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)[y_predict]

#print (np.unique(imgUV.reshape(252*252,2)))

#print (som.codebook.matrix)


#originalGroup = mySom.getCodeword(som, originalUV.reshape(256*256,2))
#refImg = np.concatenate((imgGrey.reshape(256*256,1), originalGroup),1)
#refImg = refImg.reshape(256,256,3)
#misc.imsave('land515_reference.png',color.luv2rgb(refImg))
    
#imgGrey = imgGrey[2:254,2:254]
#newImg = np.concatenate((imgGrey.reshape(252*252,1), imgUV.reshape(63504,2)),1)
#newImg = newImg.reshape(252,252,3)
#misc.imsave('land515_colored.png',color.luv2rgb(newImg))


