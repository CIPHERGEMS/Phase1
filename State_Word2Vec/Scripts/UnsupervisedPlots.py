# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 08:09:12 2020

@author: Dave
"""


# import libraries
import pandas as pd
import os
import xlrd
from pathlib import Path
import numpy as np
from random import randint
from NlpPreprocessing import NlpPreprocessing
import matplotlib.pyplot as plt
import gensim
import gensim.downloader as api
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

#%%
#   Functions
def cal_and_plot_UMAP(df, columns, path, name):      
      
    classes = list(np.unique(df['state']))
    unique_y = np.unique(df['label'])
    color = []
    n = len(classes)
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    
    #   2D
    embedding = umap.UMAP(n_neighbors=50, min_dist=0.3,
                                  n_components=2, random_state=42).fit_transform(df[columns].values)
    
    sns_plot = sns.scatterplot(
            x=embedding[:,0], y=embedding[:,1],
            hue=df['state'],
            palette=color,
            data=embedding,
            alpha=0.3, 
            s=5
    )
    box = sns_plot.get_position()
    sns_plot.set_position([box.x0, box.y0, box.width * 0.65, box.height]) # resize position
    
    # Put a legend to the right side
    sns_plot.legend(loc='center right', bbox_to_anchor=(1.65, 0.5), ncol=1)
    fig = sns_plot.get_figure()
    output = name + '_UMAPc1c2.png'
    fig.savefig(path/output, dpi=600)
    fig.clf()

    #   3D
    embedding = umap.UMAP(n_neighbors=50, min_dist=0.3,
                                  n_components=3, random_state=42).fit_transform(df[columns].values)

    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    for i,true_lab in enumerate(unique_y):
        lst = df.index[df['label'] == true_lab].tolist()
        ax.scatter(
            xs=embedding[lst,0], 
            ys=embedding[lst,1], 
            zs=embedding[lst,2], 
            c = color[i],
            s=5,
            label = list(dict_artist_2_number.keys())[list(dict_artist_2_number.values()).index(true_lab)]
        )
   
    ax.legend()
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    output = name + '_UMAPc1c2c3.png'
    plt.savefig(path/output, dpi=600)
    plt.clf()
    
def cal_and_plot_PCA(df, columns, path, name):
    #   PCA Embedding

    color = []
    n = len(np.unique(df['state']))
    unique_y = np.unique(df['label'])
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[columns].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    output = name + '_PCAc1c2.txt'
    with open(path/output, "a") as file_sav:
        file_sav.write('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    #   2D PCA
    sns_plot = sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="state",
            palette=color,
            #palette=sns.color_palette("hls", 5),
            data=df,
        #    legend="full",
            alpha=0.3,
            s=5
    )
    box = sns_plot.get_position()
    sns_plot.set_position([box.x0, box.y0, box.width * 0.65, box.height]) # resize position
    
    # Put a legend to the right side
    sns_plot.legend(loc='center right', bbox_to_anchor=(1.65, 0.5), ncol=1)
    fig = sns_plot.get_figure()
    output = name + '_PCAc1c2.png'
    fig.savefig(path/output, dpi=600)
    fig.clf()
    
    #   3D PCA
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    for i,true_lab in enumerate(unique_y):
        lst = df.index[df['label'] == true_lab].tolist()
        ax.scatter(
            xs=df.loc[lst,:]["pca-one"], 
            ys=df.loc[lst,:]["pca-two"], 
            zs=df.loc[lst,:]["pca-three"], 
            c = color[i],
            s=5,
            label = list(dict_artist_2_number.keys())[list(dict_artist_2_number.values()).index(true_lab)]
        )
   
    ax.legend()
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    output = name + '_PCAc1c2c3.png'
    plt.savefig(path/output, dpi=600)
    plt.clf()
    
def cal_and_plot_TSNE(df, columns, path, name):
   
    color = []
    n = len(np.unique(df['state']))
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[columns].values)
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    sns_plot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="state",
            palette=color,
            data=df,
            alpha=0.3,
            s=5
    )
    box = sns_plot.get_position()
    sns_plot.set_position([box.x0, box.y0, box.width * 0.65, box.height]) # resize position
    
    # Put a legend to the right side
    sns_plot.legend(loc='center right', bbox_to_anchor=(1.65, 0.5), ncol=1)
    fig = sns_plot.get_figure()
    output = name + '_TSNEc1c2.png'
    fig.savefig(path/output, dpi=600)
    fig.clf()
    
def cal_and_plot_PCA_TSNE(df, columns, path, name, pc_comp):
    
    color = []
    n = len(np.unique(df['state']))
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    
    pca = PCA(n_components=pc_comp)
    pca_result = pca.fit_transform(embedding_df[embed_cols].values)
    output = name + '_PCA'+str(pc_comp)+'_TSNEc1c2.txt'
    with open(path/output, "a") as file_sav:
        file_sav.write('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result)
    df['tsne-pca'+str(pc_comp)+'-one'] = tsne_pca_results[:,0]
    df['tsne-pca'+str(pc_comp)+'-two'] = tsne_pca_results[:,1]
    sns_plot = sns.scatterplot(
            x="tsne-pca"+str(pc_comp)+"-one", y="tsne-pca"+str(pc_comp)+"-two",
            hue="state",
            palette=color,
            data=df,
            alpha=0.3,
            s=5
    )
    box = sns_plot.get_position()
    sns_plot.set_position([box.x0, box.y0, box.width * 0.65, box.height]) # resize position
    
    # Put a legend to the right side
    sns_plot.legend(loc='center right', bbox_to_anchor=(1.65, 0.5), ncol=1)
    fig = sns_plot.get_figure()
    output = name + '_PCA'+str(pc_comp)+'_TSNEc1c2.png'
    fig.savefig(path/output, dpi=600)
    fig.clf()

#%%
# grab input folders
home = os.path.expanduser('~')
homebase_path = Path(home)
country_folder = homebase_path / "Desktop" / "USA"
artists_tracks_folder = country_folder / "Got_Tracks"
results_path = homebase_path / "Desktop" / "glove_results1"

workbook_locations = xlrd.open_workbook(os.path.join(country_folder, 'USA Artists Locations.xlsx'))
sheet = workbook_locations.sheet_by_index(0)

years = [x for x in list(range(2000, 2001))] 
search_year_type = "None" # "Single" "Group" "None" wordcloud per yr, per 80's, or just all (None)
states = ['California','New York','Georgia','New Jersey','Virginia','Texas',
          'Pennsylvania','Ohio','Michigan','Louisiana','Illinois','Florida']

# Loop though the states and save in dict
dict_state_tracks = {}
min_number_tracks = 1e16
ThresholdNumberTracks = 10

lib_list = [ "glove-twitter-25"]#, "glove-twitter-50", "glove-twitter-100", "glove-twitter-200", 
           # "glove-wiki-gigaword-50", "glove-wiki-gigaword-100", "glove-wiki-gigaword-200", "glove-wiki-gigaword-300", 
           # "fasttext-wiki-news-subwords-300", "conceptnet-numberbatch-17-06-300", "word2vec-google-news-300" ]

if search_year_type == "Single":
    yearsloop = years
else:
    yearsloop = [0]
for year in yearsloop:
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
             
                #   Create and save selected subset data
                # Subset the State Tracks by group of years.. e.g. get all the 90s tracks from New York        
                if search_year_type == "Group":
                    str_yr_name = str(int(min(years))) + str(int(max(years))) 
                    for ir in df_state_tracks.itertuples():
                        if ir[3] != "UnKnown" and int(float(ir[3])) in years:
                            NlpPrep = NlpPreprocessing(ir[4])
                            clean_track = NlpPrep.start_preprocess()
                            list_tracks.append(clean_track) 
                            list_state_name.append(state) 
                            list_y_val.append(i) 
                            
                # Subset the State Tracks by year.. e.g. get all the tracks from New York in the year 1991 
                elif search_year_type == "Single":  
                    str_yr_name = str(int(year)) 
                    for ir in df_state_tracks.itertuples():                 
                        if ir[3] != "UnKnown" and int(float(ir[3])) == year:
                            NlpPrep = NlpPreprocessing(ir[4])
                            clean_track = NlpPrep.start_preprocess()
                            list_tracks.append(clean_track) 
                            list_state_name.append(state) 
                            list_y_val.append(i) 
                                        
                # Subset the State Tracks by State only.. e.g. get all the tracks from New York out of the data we have            
                else:
                    str_yr_name = 'NoYear' 
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
                    if df_artist_total is not None:
                        df_artist_total = pd.concat([df_artist_total, df_state_temp])
                    else:
                        df_artist_total = df_state_temp  
    
    #%%
            folder = embed_lib + '_' + str(num_first_words_2_use) + '_' + str_yr_name
            res_folder = results_path / folder
            Path(res_folder).mkdir(parents=True, exist_ok=True)
            
            text = df_artist_total['text']
            statename = df_artist_total['state']
            y_true = np.array(df_artist_total['label'], dtype=int)

            # For reproducability of the results
            np.random.seed(42)
            # shuffle the data
            rndperm = np.random.permutation(df_artist_total.shape[0])
            
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
            
            #   get tfidf vector for each lyric file
            corpus = list(text)
            tfidf = TfidfVectorizer(max_features = num_first_words_2_use * embed_len) 
            tfidf.fit(corpus)
            tfidf_features = tfidf.transform(corpus)
            
            tfidf_np = tfidf_features.todense()
            tfidf_cols = [ 'tfidf'+str(i) for i in range(tfidf_np.shape[1]) ]
            df1 = pd.DataFrame(tfidf_np)
            df1.reset_index(drop=True, inplace=True)
            df2 = pd.DataFrame(statename)
            df2.reset_index(drop=True, inplace=True)
            df3 = pd.DataFrame(y_true)
            df3.reset_index(drop=True, inplace=True)
            tfidf_df = pd.concat([df1, df2, df3], axis=1)
            tfidf_df.columns = tfidf_cols +[ 'state', 'label']
            tfidf_df = tfidf_df.loc[rndperm,:]
          
            cal_and_plot_PCA(embedding_df, embed_cols, res_folder, 'embedding')
            cal_and_plot_PCA(tfidf_df, tfidf_cols, res_folder, 'tfidf')
            
            cal_and_plot_UMAP(embedding_df, embed_cols, res_folder, 'embedding')
            cal_and_plot_UMAP(tfidf_df, tfidf_cols, res_folder, 'tfidf')
            
            #   TSNE tf_idf
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(tfidf_df[tfidf_cols].values)
            
            cal_and_plot_TSNE(embedding_df, embed_cols, res_folder, 'embedding')
            cal_and_plot_TSNE(tfidf_df, tfidf_cols, res_folder, 'tfidf')
            
            for pc_comp in [25,50,75,100]:
                cal_and_plot_PCA_TSNE(embedding_df, embed_cols, res_folder, 'embedding', pc_comp)
                cal_and_plot_PCA_TSNE(tfidf_df, tfidf_cols, res_folder, 'tfidf', pc_comp)
        
        