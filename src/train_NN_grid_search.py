import pickle
import gc
import numpy as np
from sknn.mlp import Classifier, Convolution, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

patch_sizes = [[5,5]]   #[[3,3], [5,5], [11,11]]  #patches to use
som_sizes = [[3,3]]#,[10,10],[20,20]]  #soms to use
training_sample_size = 50000
test_sample_size = 5000

def trainNN (X, y):
    print ('        NN X:' + str(X.shape) + ' y:' + str(y.shape))
    print ('        training_sample_size:' + str(training_sample_size) + ' test_sample_size:' + str(test_sample_size))

    sss = StratifiedShuffleSplit(n_splits=10, test_size=test_sample_size, train_size=training_sample_size)
        
    nn = MLPClassifier(
        hidden_layer_sizes=(50), 
        #random_state=1, 
        max_iter=1000,
        learning_rate_init = 0.001,
        verbose = True,
        ) #warm_start=False)

    # scalling + classification
    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network', nn)])

    X_train, y_train = X, np.ravel(y)

    gs = GridSearchCV(nn, verbose=10, cv=sss, param_grid={
    'hidden_layer_sizes': [(10), (50), (100), (50,50), (100,100)]})
    gs.fit(X_train, y_train)
    
    print(gs.cv_results_)
    
    return gs.best_estimator_


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
           
        print ('    Original dataset: X:', X.shape, ' y:', y.shape)

        pipeline = trainNN(X, y)
        pickle.dump(pipeline, open('pkls/grid_patch' + str(patch_size[0]) + 'x' + str (patch_size[1])+'_som' + str(som_size[0]) + 'x' + str (som_size[1])+'.pkl', 'wb'))
        
