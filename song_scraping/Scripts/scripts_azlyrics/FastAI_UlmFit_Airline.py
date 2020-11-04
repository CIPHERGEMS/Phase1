# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:40:27 2020

@author: Dave
"""

#   https://towardsdatascience.com/nlp-classification-with-universal-language-model-fine-tuning-ulmfit-4e1d5077372b

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import itertools
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from fastai.text import *

df = pd.read_csv("Tweets.csv", low_memory = False)
df.head()

print(df.isnull().sum())
pos_sum = df[df['airline_sentiment']=='positive']
neu_sum = df[df['airline_sentiment']=='neutral']
neg_sum = df[df['airline_sentiment']=='negative']
zero_sum = df[df['negativereason_confidence']==0]
print('------------------------------------------------------')
print('total_non_neg = ',len(pos_sum)+len(neu_sum))
print('total zeros in neg_confidence = ',len(zero_sum))
print('------------------------------------------------------')
print('total_rows = ',len(pos_sum)+len(neu_sum)+len(neg_sum))

df_new = df.drop(['airline_sentiment_gold', 'negativereason_gold','tweet_coord'], axis = 1)
df_new.head()

data_lm = (TextList
           .from_csv(path, 'Tweets.csv', cols='text')
           #Where are the text? Column 'text' of tweets.csv
           .split_by_rand_pct(0.2)
           #How to split it? Randomly with the default 20% in valid
           .label_for_lm()
           #Label it for a language model
           .databunch(bs=48))
           #Finally we convert to a DataBunch
data_lm.show_batch()   

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
#find the optimal learning rate & visualize it
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(6,5e-2, moms=(0.85,0.75))

learn.unfreeze()
learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))

TEXT = "The flight got delayed"
N_WORDS = 40
N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
#Save fine-tuned model for future use
learn.save_encoder('fine_tuned_enc')

data_clas = (TextList.from_csv(path, 'Tweets.csv', cols='text')
             #Where are the text? Column 'text' of tweets.csv
             .split_by_rand_pct(0.2)
             #How to split it? Randomly with the default 20% in valid
             .label_from_df(cols='airline_sentiment')
             #specify the label column
             .databunch(bs=48))
             #Create databunch
data_clas.show_batch()

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')


learn.lr_find()
learn.recorder.plot(skip_end=15)

learn.fit_one_cycle(4, 5e-2, moms=(0.8,0.7))

learn.freeze_to(-2)
learn.fit_one_cycle(4, slice(1e-3/(2.6**4), 1e-3), moms=(0.8,0.7))

learn.freeze_to(-3)
learn.fit_one_cycle(4, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

learn.unfreeze()
learn.lr_find()
learn.recorder.plot(skip_end=15)

learn.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.save('fwd_clas')

pred_fwd,lbl_fwd = learn.get_preds(ordered=True)
accuracy(pred_fwd, lbl_fwd)

learn.predict("I love traveling with Vistara Airways!!!!  Awesome service.. Thank you!!")












































