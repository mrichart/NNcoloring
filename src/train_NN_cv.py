import pickle
import gc
import numpy as np
from sknn.mlp import Classifier, Convolution, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

patch_sizes = [[3,3]]   #[[3,3], [5,5], [11,11]]  #patches to use
som_sizes = [[3,3],[10,10],[20,20]]  #soms to use
training_sample_size = 50000
test_sample_size = 5000

def trainNN (X, y):
    print ('        NN X:' + str(X.shape) + ' y:' + str(y.shape))
    print ('        training_sample_size:' + str(training_sample_size) + ' test_sample_size:' + str(test_sample_size))
    # nn classifier definition
    
    #nn = Classifier(
    #    layers=[
    #        #Layer("Rectifier", units=30),
    #        #Layer("Rectifier", units=30),
    #        #Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
    #        Layer("Rectifier", units=10),
    #        Layer("Rectifier", units=5),
    #        Layer("Softmax")],
    #    learning_rate=0.03, #0.02,
    #    n_iter=1000)
        
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_sample_size, train_size=training_sample_size)
        
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier(
        hidden_layer_sizes=(500,500,),
        #random_state=1, 
        max_iter=1000,
        learning_rate_init = 0.001,
        verbose = True,
        ) #warm_start=False)
        
    #from sklearn.neural_network import MLPClassifier
    #nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                   hidden_layer_sizes=(15,), random_state=1)
        
    # scalling + classification
    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network', nn)])


    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    #print ('        Xtr:' + str(X.shape) + ' ytr:' + str(y.shape))
    X_train, y_train = X, np.ravel(y)

    #pipeline.fit(X_train, y_train)
    #for i in range(10000):
    #    pipeline.fit(X_train, y_train)
    #    if i%100 == 0:
    #        print(i)

    scores = cross_val_score(pipeline, X_train, y_train, cv=sss)

    #for train_index, test_index in sss.split(X_train, y_train):
    #    print("TRAIN:", train_index, "TEST:", test_index)
    #    _X_train, _X_test = X_train[train_index], X_train[test_index]
    #    _y_train, _y_test = y_train[train_index], y_train[test_index]
    #    pipeline.fit(_X_train, _y_train)
    #    print(pipeline.score(_X_test,_y_test ))
    #    return pipeline

    print (scores)
    
    return pipeline


for som_size in som_sizes:
    print ('Using som: [' + str(som_size[0]) + 'x' + str (som_size[1]) + ']')

    for patch_size in patch_sizes:
        print ('    With patch size: [' + str(patch_size[0]) + ',' + str (patch_size[1])+']')
        fileData = open ('pkls/data_patch' + str(patch_size[0]) + 'x' + str (patch_size[1])+'_som' + str(som_size[0]) + 'x' + str (som_size[1])+'.pkl','rb') 
        X = pickle.load (fileData)
        fileData.close()

        fileLabels = open ('pkls/labels_patch' + str(patch_size[0]) + 'x' + str (patch_size[1])+'_som' + str(som_size[0]) + 'x' + str (som_size[1])+'.pkl','rb') 
        y = pickle.load (fileLabels)
        fileLabels.close()
           
        #reducedData = np.array (random.sample (data.tolist(), int(data.size*ratio)))
        print ('    Original dataset: X:', X.shape, ' y:', y.shape)
        #np.random.shuffle(X)
        #np.random.shuffle(y)
        #X=X[0:training_sample_size,:]
        #y=y[0:training_sample_size,:]

        gc.collect()

        pipeline = trainNN(X, y)
        #pickle.dump(pipeline, open('pkls/pipeline_patch' + str(patch_size[0]) + 'x' + str (patch_size[1])+'_som' + str(som_size[0]) + 'x' + str (som_size[1])+'.pkl', 'wb'))
        
        X = None
        y = None
        pipeline = None
        gc.collect()

