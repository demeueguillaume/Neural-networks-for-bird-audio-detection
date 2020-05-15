import librosa
from time import perf_counter
import os
import numpy as np

#working_directory= "D:/LBRTI2202/warblrb10k_public_wav"
working_directory="D:/LBRTI2202/warblrb10k_public_wav"

def wav2mfcc_without_plot(file_path, max_len=249):
    wave, sr = librosa.load(file_path, mono=True,
                            sr=None)  # monaural / stereophonic sound (un seul channel/+ieurs channels)
    # sr=44.1khz
    mfcc = librosa.feature.mfcc(wave, sr, n_mfcc=40, n_fft=1764, hop_length=882)
    #print(file_path)
    return mfcc


#prend le chemin d'accÃ¨s en entrÃ©e et crÃ©e un dossier "img" contenant chacun des MMFC enregistrÃ© en .npy 
#MMFC est un nd.Array
def transformWav_to_MFFC(working_directory) :
    #cette partie transforme tout les wav en .npy
    
    t1_start = perf_counter()
    for i in os.listdir(working_directory+'/wav/') :   
        print("traiting :"+i)
        MFCC=wav2mfcc_without_plot(working_directory+'/wav/'+i)
        #np.save(working_directory+'/img/'+str(i)[:-4],MFCC)
        np.save('D:/LBRTI2202/newimg/img/'+str(i)[:-4],MFCC)
        
        
    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
    
    
transformWav_to_MFFC(working_directory)