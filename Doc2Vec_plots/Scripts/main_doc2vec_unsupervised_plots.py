# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 20:42:21 2020

@author: Dave
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from random import randint
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import umap
import multiprocessing
cores = multiprocessing.cpu_count()
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import xlrd
from pathlib import Path
from NlpPreprocessing import NlpPreprocessing

#%%
'''
dm  --  If dm=0, distributed bag of words (PV-DBOW) is used; if dm=1,‘distributed memory’ (PV-DM) is used.
vector_size -- 300- dimensional feature vectors.
min_count=2 -- ignores all words with total frequency lower than this.
negative=5 -- specifies how many “noise words” should be drawn.
hs=0 -- and negative is non-zero, negative sampling will be used.
sample=0 -- the threshold for configuring which higher-frequency words are randomly down sampled.
workers=cores -- use these many worker threads to train the model (=faster training with multicore machines).
'''

#   Functions
def cal_and_plot_UMAP(df, columns, path, name, dict_artist_2_number):      
      
    classes = list(np.unique(df['state']))
    unique_y = np.unique(df['label'])
    color = []
    n = len(classes)
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    
    #   2D
    embedding = umap.UMAP(n_neighbors=50, min_dist=0.3,
                                  n_components=2, random_state=42).fit_transform(df[columns].values)
    embedding_df = pd.DataFrame(embedding)

    sns_plot = sns.scatterplot(
            x=embedding_df.iloc[:, 0], y=embedding_df.iloc[:, 1],
            hue=df['state'],
            palette=color,
            data=embedding_df,
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
    
def cal_and_plot_PCA(df, columns, path, name, dict_artist_2_number):
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
    print(pc_comp)
    color = []
    n = len(np.unique(df['state']))
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    
    pca = PCA(n_components=pc_comp)
    pca_result = pca.fit_transform(df[columns].values)
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

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
    
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors
                
#%%
home = os.path.expanduser('~')
homebase_path = Path(home)
country_folder = homebase_path / "Desktop" / "USA"
artists_tracks_folder = country_folder / "Got_Tracks"
results_path = homebase_path / "Desktop" / "Doc2Vec_plots" / "Results"

workbook_locations = xlrd.open_workbook(os.path.join(country_folder, 'USA Artists Locations.xlsx'))
sheet = workbook_locations.sheet_by_index(0)

states = ['California','New York','Georgia','New Jersey','Virginia','Texas',
          'Pennsylvania','Ohio','Michigan','Louisiana','Illinois','Florida']

# Loop though the states and save in dict
dict_state_tracks = {}
min_number_tracks = 1e16
ThresholdNumberTracks = 10
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
    
    text = df_artist_total['text']
    statename = df_artist_total['state']
    y_true = np.array(df_artist_total['label'], dtype=int)
    
#%%
#   EDA
cnt_pro = df_artist_total['state'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('state', fontsize=12)
plt.xticks(rotation=90)
Path(results_path).mkdir(parents=True, exist_ok=True)
output = 'Barchart Class numbers.png'
plt.savefig(results_path / output, dpi=600)

#%%
for num_epochs in [30, 50, 100, 150]:
    for vector_size in [10, 50, 100, 200, 300, 400, 500]:   # vector size 
        df_tagged = df_artist_total.apply(
            lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.state]), axis=1)

        #   Distributed Bag of Words (DBOW)
        model_dbow = Doc2Vec(dm=0, vector_size=vector_size, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
        model_dbow.build_vocab([x for x in tqdm(df_tagged.values)])
        for epoch in range(num_epochs):
            model_dbow.train(utils.shuffle([x for x in tqdm(df_tagged.values)]), total_examples=len(df_tagged.values), epochs=1)
            model_dbow.alpha -= 0.002
            model_dbow.min_alpha = model_dbow.alpha
        y, X = vec_for_learning(model_dbow, df_tagged)

        folder = 'VectorSize__' + str(vector_size) + '__Epochs__' + str(num_epochs)
        res_folder = results_path / folder
        Path(res_folder).mkdir(parents=True, exist_ok=True)
        
        colnames = [ 'embed'+str(i) for i in range(np.array(X).shape[1]) ]
        embed_df = pd.concat([pd.DataFrame(np.array(X)),pd.DataFrame(statename).reset_index(drop=True),pd.DataFrame(y_true)], axis=1)
        embed_df.columns = colnames +[ 'state', 'label']

        cal_and_plot_PCA(embed_df, colnames, res_folder, 'dbow', dict_artist_2_number)
        cal_and_plot_TSNE(embed_df, colnames, res_folder, 'dbow')
        cal_and_plot_UMAP(embed_df, colnames, res_folder, 'dbow', dict_artist_2_number)
        for pc_comp in [25,50,75,100]:
            if pc_comp > vector_size:
                continue
            cal_and_plot_PCA_TSNE(embed_df, colnames, res_folder, 'dbow', pc_comp)
            
        #%%
        '''
        Distributed Memory (DM)
        Distributed Memory (DM) acts as a memory that remembers what is missing from 
        the current context — or as the topic of the paragraph. 
        While the word vectors represent the concept of a word, 
        the document vector intends to represent the concept of a document.
        '''
        model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
        model_dmm.build_vocab([x for x in tqdm(df_tagged.values)])
        for epoch in range(num_epochs):
            model_dmm.train(utils.shuffle([x for x in tqdm(df_tagged.values)]), total_examples=len(df_tagged.values), epochs=1)
            model_dmm.alpha -= 0.002
            model_dmm.min_alpha = model_dmm.alpha    
    
        y, X = vec_for_learning(model_dmm, df_tagged)
        
        folder = str(vector_size) + '_' + str(num_epochs) 
        res_folder = results_path / folder
        Path(res_folder).mkdir(parents=True, exist_ok=True)

        colnames = [ 'embed'+str(i) for i in range(np.array(X).shape[1]) ]
        embed_df = pd.concat([pd.DataFrame(np.array(X)),pd.DataFrame(statename).reset_index(drop=True),pd.DataFrame(y_true)], axis=1)
        embed_df.columns = colnames +[ 'state', 'label']

        cal_and_plot_PCA(embed_df, colnames, res_folder, 'dmm', dict_artist_2_number)
        cal_and_plot_TSNE(embed_df, colnames, res_folder, 'dmm')
        cal_and_plot_UMAP(embed_df, colnames, res_folder, 'dmm', dict_artist_2_number)
        for pc_comp in [25,50,75,100]:
            if pc_comp > vector_size:
                continue
            cal_and_plot_PCA_TSNE(embed_df, colnames, res_folder, 'dmm', pc_comp)
            
        #%%
        '''
        Model Pairing
        According to Gensim doc2vec tutorial on the IMDB sentiment data set, 
        combining a paragraph vector from Distributed Bag of Words (DBOW) and 
        Distributed Memory (DM) improves performance. We will follow, 
        pairing the models together for evaluation.
        First, we delete temporary training data to free up RAM.
        '''
        model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        #   join models
        new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
        y, X = vec_for_learning(new_model, df_tagged)

        folder = str(vector_size) + '_' + str(num_epochs) 
        res_folder = results_path / folder
        Path(res_folder).mkdir(parents=True, exist_ok=True)

        colnames = [ 'embed'+str(i) for i in range(np.array(X).shape[1]) ]
        embed_df = pd.concat([pd.DataFrame(np.array(X)),pd.DataFrame(statename).reset_index(drop=True),pd.DataFrame(y_true)], axis=1)
        embed_df.columns = colnames +[ 'state', 'label']

        cal_and_plot_PCA(embed_df, colnames, res_folder, 'joint', dict_artist_2_number)
        cal_and_plot_TSNE(embed_df, colnames, res_folder, 'joint')
        cal_and_plot_UMAP(embed_df, colnames, res_folder, 'joint', dict_artist_2_number)
        for pc_comp in [25,50,75,100]:
            if pc_comp > vector_size:
                continue
            cal_and_plot_PCA_TSNE(embed_df, colnames, res_folder, 'joint', pc_comp)
