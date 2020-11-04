# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:08:25 2020

@author: Dave
"""


import os
import xlrd
import pandas as pd
import numpy as np
from pathlib import Path
from NlpPreprocessing import NlpPreprocessing
from nltk.tokenize import word_tokenize 
import gensim
import gensim.downloader as api

home = os.path.expanduser('~')
homebase_path = Path(home)
country_folder = homebase_path / "Desktop" / "USA"
artists_tracks_folder = country_folder / "Got_Tracks"
results_path = homebase_path / "Desktop" / "Embedding_Classifiers" / "Results"

#%%
workbook_locations = xlrd.open_workbook(os.path.join(country_folder, 'USA Artists Locations.xlsx'))
sheet = workbook_locations.sheet_by_index(0)

states = ['California','New York']#,'Georgia','New Jersey','Virginia','Texas',
       #   'Pennsylvania','Ohio','Michigan','Louisiana','Illinois','Florida']

# Loop though the states and save in dict
dict_state_tracks = {}
min_number_tracks = 1e16
ThresholdNumberTracks = 10

lib_list = [ #"glove-twitter-25", "glove-twitter-50", "glove-twitter-100", "glove-twitter-200", 
            "glove-wiki-gigaword-50", "glove-wiki-gigaword-100", "glove-wiki-gigaword-200", "glove-wiki-gigaword-300"]#, 
           # "fasttext-wiki-news-subwords-300", "conceptnet-numberbatch-17-06-300", "word2vec-google-news-300" ]


for embed_lib in lib_list:
    model = api.load(embed_lib)
    embed_len = int(embed_lib.split('-')[-1])
    
    for num_first_words_2_use in [50, 100, 150, 200, 250, 300]:   # keep 100 first words  
        dict_artist_2_number = {}
        df_artist_total = None
        list_df_artist_tracks = []
        list_num_artist_tracks = []
        for i,state in enumerate(states):
            dict_artist_2_number[state] = i
            #   Setup folders and initialize variables
            df_state_tracks = None
            list_tracks = []
            list_state_name = []
            list_y_val = []
            
            # Get all tracks from state into one dataframe
            for rowx in range(sheet.nrows):
                if sheet.row_values(rowx)[2] == state:   
                    df_artists = pd.read_csv(os.path.join(artists_tracks_folder, sheet.row_values(rowx)[0] + '.csv'),encoding='latin1')
                    df_artists.columns = ['title','album','year','lyrics','urls']
                    if df_state_tracks is not None:
                        df_state_tracks = pd.concat([df_state_tracks, df_artists])
                    else:
                        df_state_tracks = df_artists         
         
            for ir in df_state_tracks.itertuples():
                clean_trac = ir[4]
                NlpPrep = NlpPreprocessing(ir[4])
                clean_track = NlpPrep.start_preprocess()
                list_tracks.append(clean_track) 
                list_state_name.append(state) 
                list_y_val.append(i) 
                                                    
            # Update minimum track number and store track lists into dict
            if len(list_tracks) > ThresholdNumberTracks and len(list_tracks) < min_number_tracks:
                min_number_tracks = len(list_tracks)
            if len(list_tracks) >= min_number_tracks:
                dict_state_tracks[state] = list_tracks
            
            zipped = zip(list_tracks, list_state_name, list_y_val)
            df_state_temp = pd.DataFrame(zipped)
            
            list_df_artist_tracks.append(df_state_temp)
            list_num_artist_tracks.append(len(df_state_temp))

        top_sort_index = np.argsort(list_num_artist_tracks)[::-1][:10]
        for df_idx in top_sort_index:
            df_state_temp = list_df_artist_tracks[df_idx]
            if len(df_state_temp) != 0:
                df_state_temp.columns = ['text','state','label'] 
                df_state_temp = df_state_temp.sample(frac=1, random_state=42)
                df_state_temp = df_state_temp[:7300]                   
                if df_artist_total is not None:
                    df_artist_total = pd.concat([df_artist_total, df_state_temp])
                else:
                    df_artist_total = df_state_temp  
    
            folder = embed_lib + '_' + str(num_first_words_2_use) 
            res_folder = results_path / folder
            Path(res_folder).mkdir(parents=True, exist_ok=True)
            
            text = df_artist_total['text']
            statename = df_artist_total['state']
            y_true = np.array(df_artist_total['label'], dtype=int)
 
        #%%
        # For reproducability of the results
        set_seed = 42
        np.random.seed(set_seed)
        # shuffle the data
        rndperm = np.random.permutation(df_artist_total.shape[0])
        
        #%%
        #   embed each lyric file
        array_length = num_first_words_2_use * embed_len # keep 20 first words and fasttext gives 300-d vectors
        embedding_features = pd.DataFrame()
        for document in text:
            # Saving the first 20 words of the document as a sequence
            words = document.split()
            np.random.shuffle(words)
            words = words[0:num_first_words_2_use]            
            
            # Retrieving the vector representation of each word and 
            # appending it to the feature vector 
            feature_vector = []
            for word in words:
                try:
                    feature_vector = np.append(feature_vector, 
                                               np.array(model[word]))
                except KeyError:
                    # In the event that a word is not included in our dictionary skip that word
                    pass
            # If the text has less then 20 words, fill remaining vector with zeros
            zeroes_to_add = array_length - len(feature_vector)
            feature_vector = np.append(feature_vector, 
                                       np.zeros(zeroes_to_add)
                                       ).reshape((1,-1))
            
            # Append the document feature vector to the feature table
            embedding_features = embedding_features.append( 
                                             pd.DataFrame(feature_vector)) 
        
        embedding_features_np = np.array(embedding_features)
        embed_cols = [ 'bed'+str(i) for i in range(embedding_features.shape[1]) ]
        df1 = pd.DataFrame(embedding_features_np)
        df1.reset_index(drop=True, inplace=True)
        df2 = pd.DataFrame(statename)
        df2.reset_index(drop=True, inplace=True)
        df3 = pd.DataFrame(y_true)
        df3.reset_index(drop=True, inplace=True)
        embedding_df = pd.concat([df1, df2, df3], axis=1)
        embedding_df.columns = embed_cols +[ 'state', 'label']
        embedding_df = embedding_df.loc[rndperm,:]

        #%%
        def scale_data(X_train, X_test):
            from sklearn.preprocessing import StandardScaler
            feature_scaler = StandardScaler()
            X_train = feature_scaler.fit_transform(X_train)
            X_test = feature_scaler.transform(X_test)
            return X_train, X_test
            
        #%%
#        import sklearn
        import copy as cp
        # Import necessary modules
        from sklearn.model_selection import train_test_split
#        from sklearn.metrics import mean_squared_error
#        from math import sqrt
        from sklearn.metrics import precision_score, recall_score
        
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_validate
                
        from sklearn.model_selection import KFold
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import LeavePOut
        from sklearn.model_selection import ShuffleSplit
        from sklearn.model_selection import StratifiedKFold
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis       
        
        scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}
        
        names = ["Logistic Regression", "Nearest Neighbors1", "Nearest Neighbors3", 
                 "Nearest Neighbors5", "Linear SVM", "RBF SVM", "Gaussian Process",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes", "QDA"]

        classifiers = [
            LogisticRegression(max_iter=500, n_jobs=-1),
            KNeighborsClassifier(1, n_jobs=-1),
            KNeighborsClassifier(3, n_jobs=-1),
            KNeighborsClassifier(5, n_jobs=-1),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        X_data = embedding_df[embed_cols].values
        y_labs = embedding_df['label'].values
        
        #%%
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            X_train_org, X_test_org, y_train, y_test = train_test_split(X_data, y_labs, test_size=0.30, random_state=set_seed)
            for scale in ['scaling', 'non_scaling']:
                X_train, X_test = scale_data(X_train_org, X_test_org)
                if scale == 'non_scaling':
                    X_train = cp.copy(X_train_org)  
                    X_test = cp.copy(X_test_org)  
        
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)            
                precision = precision_score( y_test, y_pred, average='macro')
                recall = recall_score( y_test, y_pred, average='micro')           
                acc = clf.score(X_test, y_test)
                output = 'Holdout_' + scale + '_clf_results.txt'
                with open(res_folder/output, "a") as file_sav:
                    file_sav.write(name+'\n')
                    file_sav.write(str(round((acc*100.0),ndigits=2)) + '     Accuracy' +'\n')
                    file_sav.write(str(round((precision*100.0),ndigits=2)) + '     Precision_macro' +'\n')
                    file_sav.write(str(round((recall*100.0),ndigits=2)) + '     Recall_micro' +'\n')

                #%%        
                for k in [3, 5, 10]:
                    kfold = KFold(n_splits=k,shuffle=True, random_state=set_seed)
                    pipeline = make_pipeline(StandardScaler(), clf)
                    if scale == 'non_scaling':
                        pipeline = make_pipeline(clf)
                                        
                    scores = cross_validate(pipeline, X_data, y_labs, scoring=scoring, cv=kfold, return_train_score=True)
                    acc_lst = [  round((elem*100.0),ndigits=2) for elem in scores['test_acc'] ]
                    prec_lst = [ round((elem*100.0),ndigits=2) for elem in scores['test_prec_macro'] ]
                    rec_lst = [ round((elem*100.0),ndigits=2) for elem in scores['test_rec_micro'] ]
                    
                    output = str(k)+'fold_' + scale + '_clf_results.txt'
                    with open(res_folder/output, "a") as file_sav:
                        file_sav.write(name+'\n')
                        file_sav.write(str(acc_lst) + '     ' + str(round((sum(acc_lst) / k),ndigits=2)) + '     Accuracy' +'\n')
                        file_sav.write(str(prec_lst) + '     ' + str(round((sum(prec_lst) / k),ndigits=2)) + '     Precision_macro' +'\n')
                        file_sav.write(str(rec_lst) + '     ' + str(round((sum(rec_lst) / k),ndigits=2)) + '     Recall_micro' +'\n')

                #%%        
                for k in [3, 5, 10]:
                    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=set_seed)
                    pipeline = make_pipeline(StandardScaler(), clf)
                    if scale == 'non_scaling':
                        pipeline = make_pipeline(clf)
                                        
                    scores = cross_validate(pipeline, X_data, y_labs, scoring=scoring, cv=skfold, return_train_score=True)
                    acc_lst = [  round((elem*100.0),ndigits=2) for elem in scores['test_acc'] ]
                    prec_lst = [ round((elem*100.0),ndigits=2) for elem in scores['test_prec_macro'] ]
                    rec_lst = [ round((elem*100.0),ndigits=2) for elem in scores['test_rec_micro'] ]
                    
                    output = str(k)+'skfold_' + scale + '_clf_results.txt'
                    with open(res_folder/output, "a") as file_sav:
                        file_sav.write(name+'\n')
                        file_sav.write(str(acc_lst) + '     ' + str(round((sum(acc_lst) / k),ndigits=2)) + '     Accuracy' +'\n')
                        file_sav.write(str(prec_lst) + '     ' + str(round((sum(prec_lst) / k),ndigits=2)) + '     Precision_macro' +'\n')
                        file_sav.write(str(rec_lst) + '     ' + str(round((sum(rec_lst) / k),ndigits=2)) + '     Recall_micro' +'\n')
                    
                #%%
# =============================================================================
#                 loocv = LeaveOneOut()
#                 pipeline = make_pipeline(StandardScaler(), clf)
#                 if scale == 'non_scaling':
#                     pipeline = make_pipeline(clf)
#                                     
#                 scores = cross_validate(pipeline, X_data, y_labs, scoring=scoring, cv=loocv, return_train_score=True)
#                 acc_lst = [  round((elem*100.0),ndigits=2) for elem in scores['test_acc'] ]
#                 prec_lst = [ round((elem*100.0),ndigits=2) for elem in scores['test_prec_macro'] ]
#                 rec_lst = [ round((elem*100.0),ndigits=2) for elem in scores['test_rec_micro'] ]
#                 
#                 output = 'loocv_' + scale + '_clf_results.txt'
#                 with open(res_folder/output, "a") as file_sav:
#                     file_sav.write(name+'\n')
#                     file_sav.write(str(round((sum(acc_lst) / k),ndigits=2)) + '     Accuracy' +'\n')
#                     file_sav.write(str(round((sum(prec_lst) / k),ndigits=2)) + '     Precision_macro' +'\n')
#                     file_sav.write(str(round((sum(rec_lst) / k),ndigits=2)) + '     Recall_micro' +'\n')
# =============================================================================
                
                #%%
                for nsplits in [10, 25, 50]:
                    kfold2 = ShuffleSplit(n_splits=nsplits, test_size=0.30, random_state=set_seed)    
                    pipeline = make_pipeline(StandardScaler(), clf)
                    if scale == 'non_scaling':
                        pipeline = make_pipeline(clf)
                        
                    scores = cross_validate(pipeline, X_data, y_labs, scoring=scoring, cv=kfold2, return_train_score=True)
                    acc_lst = [  round((elem*100.0),ndigits=2) for elem in scores['test_acc'] ]
                    prec_lst = [ round((elem*100.0),ndigits=2) for elem in scores['test_prec_macro'] ]
                    rec_lst = [ round((elem*100.0),ndigits=2) for elem in scores['test_rec_micro'] ]
                        
                    output = str(nsplits)+'repeatholdout_' + scale + '_clf_results.txt'
                    with open(res_folder/output, "a") as file_sav:
                        file_sav.write(name+'\n')
                        file_sav.write(str(round((sum(acc_lst) / nsplits),ndigits=2)) + '     Accuracy' +'\n')
                        file_sav.write(str(round((sum(prec_lst) / nsplits),ndigits=2)) + '     Precision_macro' +'\n')
                        file_sav.write(str(round((sum(rec_lst) / nsplits),ndigits=2)) + '     Recall_micro' +'\n')
 
    
#%%
   
    
    
