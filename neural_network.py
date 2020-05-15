# -*- coding: utf-8 -*-
"""

dans working_directory/wav se trouvent tout les fichiers wav
dans working_directory/img se trouvent tout les fichiers npy ET le fichier all.csv

"""


label_csv='all.csv'
working_directory='D:/LBRTI2202/all'
import ast
import pandas as pd
import numpy as np
from time import perf_counter
import keras
import sklearn
import os

import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from keras import Sequential

from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from keras.models import model_from_json

os.chdir(working_directory)    
    


""" essais batch"""

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(30,100), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        global test
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        test=list_IDs_temp
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



#commentaire : Je pense que l'erreur vient du fait qu'on définit la taille de y et X au préalable alors qu'on ne prend pas toute les données
#faudrait donc les remttre à la bonne taille
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_temp = np.empty((self.batch_size, *self.dim, self.n_channels))
        y_temp = np.empty((self.batch_size), dtype=int)
        #print("list_id_temps_" , list_IDs_temp)
        # Generate data
        cnt=0
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            a=0
            try :
                a= np.load(working_directory+"/img/" + ID + '.npy' ) [0:30,0:100].reshape(30,100,1)
            except  :
                print("\n seem a problem of dimension for the data " + str(ID) + "à la position" + str(i))  #enelver le premeir pass et laisser la deuxième partie pour observer les fichiers non traitées
            if type(a)!=type(0):  #donc si ta réussi à encoder la valeur
    
                X_temp[cnt,]= np.load(working_directory+"/img/" + ID + '.npy' ) [0:30,0:100].reshape(30,100,1) 
                # Store class
                y_temp[cnt] = self.labels[ID]
                cnt+=1
            #print("cnt :" ,cnt, "i : " , i, "ID: " , ID, "Y[i]:" , y[i])
        X=X_temp[0:cnt,:,:]
        y=y_temp[0:cnt]
        
        try :
                
            yy=keras.utils.to_categorical(y, num_classes=self.n_classes)
        except  :    
            print("y:",y,labels[ID])
        return X, yy


#Définitions des fonctions

def get_data_info(working_directory,label_csv):  #va chercher le csv et sort un dictionnaire ID label et un dictionnaire sur la partition
    label=pd.read_csv(label_csv).to_numpy()
    #rempalcer les nan par des 0.0
    a=label[:,1].astype('float64')
    label[:,1][np.isnan(a)] =0
    X=label.copy()
    #création des dictionnaire de labels
    labels={}
    for data in X:
        labels[data[0]]=data[1]
        
    
    #creation dictionnaire de partition
    partition={}
    X_trrain,X_testt=train_test_split(label[:,0])
    partition['train'] = X_trrain
    partition['validation'] = X_testt
    
    return labels,partition

def model_construction() :
    # Construct model 
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(30, 100,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(2, activation='softmax'))
    return model


def save_model (model,ID_model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(file=working_directory+'/model/'+str(ID_model)+".json", mode="w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filepath=working_directory+'/model/'+str(ID_model)+".h5")
    print("Saved model to disk")


 

def load_model(ID_model):# load json and create model
    json_file = open(working_directory+"/model/"+ID_model + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(working_directory+"/model/"+ID_model +".h5")
    print("Loaded model from disk")
    return loaded_model



def model_loaded_compile(ID_model): #si vous avez déja enregistré un modèle vous pourrez travailler avec celui-ci/
    loaded_model=load_model(ID_model)    
    loaded_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        
    # Matrice de confusion
    y_pred_classes = loaded_model.predict_classes(x_test,verbose=1)
    y_test_classes = (np.where(y_test == np.amax(y_test)))[1]
    cmtx = pd.DataFrame(
        sklearn.metrics.confusion_matrix(y_test_classes, y_pred_classes), 
        index=['true:no', 'true:yes'], 
        columns=['pred:no', 'pred:yes']
    )
    print(cmtx)
    tot = cmtx.at['true:no','pred:no']+cmtx.at['true:yes','pred:no']+cmtx.at['true:no','pred:yes']+cmtx.at['true:yes','pred:yes']
    print('Faux positif:%.2f%%' % (cmtx.at['true:no','pred:yes']/tot*100))
    print('Faux négatif:%.2f%%' % (cmtx.at['true:yes','pred:no']/tot*100))
    print('Vrai négatif:%.2f%%' % (cmtx.at['true:no','pred:no']/tot*100))
    print('Vrai positif:%.2f%%' % (cmtx.at['true:yes','pred:yes']/tot*100))
    
    print('%% oiseaux identifiés:%.2f%%' % (cmtx.at['true:yes','pred:yes']/(cmtx.at['true:yes','pred:yes']+cmtx.at['true:yes','pred:no'])*100))
    print('%% endroit sans oiseaux identifiés:%.2f%%' % (cmtx.at['true:no','pred:no']/(cmtx.at['true:no','pred:yes']+cmtx.at['true:no','pred:no'])*100))
    
    print('nbr sans oiseau/nbr oiseau:%.2f' % ((cmtx.at['true:no','pred:no']+cmtx.at['true:no','pred:yes'])/(cmtx.at['true:yes','pred:no']+cmtx.at['true:yes','pred:yes'])))
    #print(y_test)
    y_predicted = loaded_model.predict(x_test)
    #print(y_predicted)
    prob_true=np.sum(y_test*y_predicted, axis=1)
    #plt.scatter(y_test[:,1],prob_true,s=3)
    #plt.ylabel('probabilité prédite')
    #plt.xlabel('absence d''oiseaux (0) vs présence d''oiseaux (1)')
    #plt.show() 
    proba_oiseau_predicted_sup75 =[]
    proba_oiseau_predicted_around50 = []
    proba_oiseau_predicted_below25 = []
    
    y_proba_oiseau = y_predicted[:,1]
    
    idx75 = 0
    idx50 = 0
    idx25 =0
    
    idx = 0
    
    for val in y_proba_oiseau:
        
        if val < 0.25:
            proba_oiseau_predicted_below25.append([y_test[int(idx),1],val])
        elif val < 0.75:
            proba_oiseau_predicted_around50.append([y_test[int(idx),1],val])
        else:
            proba_oiseau_predicted_sup75.append([y_test[int(idx),1],val])
    
        idx = idx + 1
            
    nbr75_vrai = 0
    nbr75_faux = 0
    for idx,val in enumerate(proba_oiseau_predicted_sup75):
        if val[0] == 1:
            nbr75_vrai = nbr75_vrai +1
        else:
            nbr75_faux = nbr75_faux + 1
        
    print('nbr correct apres 75%% : %.2f%%' % (nbr75_vrai/(nbr75_vrai+nbr75_faux)*100))
    print('nbr données après 75%% : '+str(nbr75_vrai+nbr75_faux))
    
    
    
    
    
    
    nbr25_vrai = 0
    nbr25_faux = 0
    for idx,val in enumerate(proba_oiseau_predicted_below25):
        if val[0] == 0:
            nbr25_vrai = nbr25_vrai +1
        else:
            nbr25_faux = nbr25_faux + 1
        
    print('nbr correct sous 25%% : %.2f%%' % (nbr25_vrai/(nbr25_vrai+nbr25_faux)*100))
    print('nbr données sous 25%% : '+str(nbr25_vrai+nbr25_faux))
    
    nbr_quantile = 20
        
    repartition_avec_oiseau = []
    repartition_sans_oiseau = []
    
    echelle1 = []
    echelle2 = []
    
    for i in range(nbr_quantile):
        repartition_avec_oiseau.append(0)
        repartition_sans_oiseau.append(0)
        echelle1.append(100/nbr_quantile*(i) + (100/20/2))
        echelle2.append(100/nbr_quantile*(i) + (100/20/2)+1)
    
    idx = 0
    nbroiseau = 0
    nbrsansoiseau = 0
    for val in y_proba_oiseau:
        a = 0
        
        while (a +1)/nbr_quantile < val:
            a=a+1
        #print(str(a)+"  "+str(y_test[int(idx),0]))
        if y_test[int(idx),1] == 0:
            repartition_sans_oiseau[a] = repartition_sans_oiseau[a]+1
            nbrsansoiseau = nbrsansoiseau+1
        else:
            repartition_avec_oiseau[a] = repartition_avec_oiseau[a]+1
            nbroiseau = nbroiseau+1
        idx = idx + 1 
    
    fractionAvecOiseaux =repartition_avec_oiseau/(cmtx.at['true:yes','pred:no']+cmtx.at['true:yes','pred:yes']);
    fractionSansOiseaux = repartition_sans_oiseau/(cmtx.at['true:no','pred:no']+cmtx.at['true:no','pred:yes'])
    plt.bar(echelle1,fractionAvecOiseaux,label='avec oiseau',color=(1, 0, 0, 1))  
    plt.bar(echelle2,fractionSansOiseaux,label='sans oiseau',color=(0, 0, 1, 1))   
    plt.ylabel('Nombre de prédiction')
    plt.xlabel('Valeur de la prédiction')
    axes = plt.gca()
    axes.set_xlim([0,100])
    axes.set_ylim([0,max([max(fractionAvecOiseaux),max(fractionSansOiseaux)])])
    plt.legend() 
    plt.show() 
    
    plt.bar(echelle1,repartition_avec_oiseau,label='avec oiseau',color=(1, 0, 0, 1))  
    plt.bar(echelle2,repartition_sans_oiseau,label='sans oiseau',color=(0, 0, 1, 1))   
    plt.ylabel('Nombre de prédiction')
    plt.xlabel('Valeur de la prédiction')
    axes = plt.gca()
    axes.set_xlim([0,100])
    axes.set_ylim([0,max([max(repartition_avec_oiseau),max(repartition_sans_oiseau)])])
    plt.legend() 
    plt.show() 




#Déclaration des fonctions
#cette fonction prend le chemin d'accès ddu folder (img) contenant les fichier NPY et le fichier csv des label et ressort un data utilisable pour le train_test_split
def load_data(working_directory,label_csv):    
    actual_folder=os.getcwd()
    os.chdir(working_directory)
    label=pd.read_csv(label_csv).to_numpy()
    #label.sort(0)
    X=label.copy()
    Y=label[:,1]
    image=np.load(working_directory + '/img/55'+'.npy')  #J'ai mis ici le nom d'une image pour qu'il puisse recuperer les dimensiosn et creer les variables avant de toute les parcourir
    data=np.ndarray(shape=(len(X),image.shape[0]*image.shape[1]))
    cnt=0    
    t1_start = perf_counter()
    for i in X:
        image=np.load(str(working_directory + '/img/'+str(i[0])+'.npy'))
        obs_=image.reshape(30*100)
        data[cnt]=obs_
        cnt=cnt+1
    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:", 
                                        t1_stop-t1_start)
    os.chdir(actual_folder)
    return data,Y

def run_model(epoch,ID_model):
    # Parameters
    params = {'dim': (30,100),
              'batch_size': 64,
              'n_classes': 2,
              'n_channels': 1,
              'shuffle': True}

    # Datasets
    labels,partition = get_data_info(working_directory,'all.csv')# IDs

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)


    
    model=model_construction()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    

     # Display model architecture summary 
    model.summary()

    # Train model on dataset
    start = datetime.now()
    history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    #workers=2
                    epochs=epoch
                    )


    score = model.evaluate(validation_generator)
    accuracy = 100*score[1]

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    with open(working_directory+"/model/time_for_train"+str(ID_model)+'.txt', 'w') as f:
        f.write("balance %d" % duration.total_seconds())



    save_model(model,ID_model)
    return(model,partition,labels)




def test_split(data,Y):

        #chargement des données et prétraitement
    data,Y=load_data(working_directory,label_csv)



    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(Y)) 
    #séparation test,train et X,y
    X_train,X_test,y_train,y_test= train_test_split(
            data,yy,test_size=0.7, shuffle=False)    

    #remise en forme des DF
    x_test = X_test.reshape(3845,30, 100,1)
    x_train = X_train.reshape(3845,30, 100,1)
    return x_test,x_train,y_test,y_train



def x_test_creation(partition,labels) :
    
    
    
    
    
    y_test=np.ndarray(len(partition["validation"]))
    x_test=np.ndarray(shape=(len(partition["validation"]),30,100))
    cnt=1
    for i in partition["validation"]:
        y_test[cnt-1]=labels[i]
        x_test[cnt-1]=np.load(str(working_directory + '/img/'+str(i)+'.npy'))[0:30,0:100].reshape(30,100)
        cnt=cnt+1   
        print(i)
    x_test=x_test[:,:,:,np.newaxis]
    le = LabelEncoder()
    y_test = to_categorical(le.fit_transform(y_test))    
    
    return x_test,y_test
        
    #recuperation dans X_test des images 
    
    #recuperation dans y_test des labels
    

#ici c'est si tu veux créer un modele-----------------------------------------
# IDmodel="mettre_ton_code"
# model,partition,labels=run_model(0,IDmodel)
# save_model (model,IDmodel)
# np.save(file='validation', arr=partition["validation"],allow_pickle=True)
# np.save(file='train', arr=partition["train"],allow_pickle=True)

#ici c'est si tu veux travailler avec un modele existant-----------------------



labels,partition=get_data_info(working_directory,label_csv)


#test x_test et y_test
x_test,y_test=x_test_creation(partition,labels)





ID_model="model_3couches_45epoch"
model_loaded_compile(ID_model)





ID_model="Finale_model_4couches_45epoch"
model_loaded_compile(ID_model)






