# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 23:10:20 2020

@author: Dave
"""

import nltk
#nltk.download()
import re
from nltk.corpus import stopwords

class NlpPreprocessing():
    
    def __init__(self, song_text):

        puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
                  '+', '\\', '•', '~', '@', '£',
                  '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
                  '½', 'à', '…',
                  '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
                  '▓', '—', '‹', '─',
                  '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
                  'Ã', '⋅', '‘', '∞',
                  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
                  '≤', '‡', '√']
        
        contractions = {"'cause": 'because',
                        ',cause': 'because',
                        ';cause': 'because',
                        "ain't": 'am not',
                        'ain,t': 'am not',
                        'ain;t': 'am not',
                        'ain´t': 'am not',
                        'ain’t': 'am not',
                        "aren't": 'are not',
                        'aren,t': 'are not',
                        'aren;t': 'are not',
                        'aren´t': 'are not',
                        'aren’t': 'are not',
                        "can't": 'cannot',
                        "can't've": 'cannot have',
                        'can,t': 'cannot',
                        'can,t,ve': 'cannot have',
                        'can;t': 'cannot',
                        'can;t;ve': 'cannot have',
                        'can´t': 'cannot',
                        'can´t´ve': 'cannot have',
                        'can’t': 'cannot',
                        'can’t’ve': 'cannot have',
                        "could've": 'could have',
                        'could,ve': 'could have',
                        'could;ve': 'could have',
                        "couldn't": 'could not',
                        "couldn't've": 'could not have',
                        'couldn,t': 'could not',
                        'couldn,t,ve': 'could not have',
                        'couldn;t': 'could not',
                        'couldn;t;ve': 'could not have',
                        'couldn´t': 'could not',
                        'couldn´t´ve': 'could not have',
                        'couldn’t': 'could not',
                        'couldn’t’ve': 'could not have',
                        'could´ve': 'could have',
                        'could’ve': 'could have',
                        'da': 'the',
                        "didn't": 'did not',
                        'didn,t': 'did not',
                        'didn;t': 'did not',
                        'didn´t': 'did not',
                        'didn’t': 'did not',
                        "doesn't": 'does not',
                        'doesn,t': 'does not',
                        'doesn;t': 'does not',
                        'doesn´t': 'does not',
                        'doesn’t': 'does not',
                        "don't": 'do not',
                        'don,t': 'do not',
                        'don;t': 'do not',
                        'don´t': 'do not',
                        'don’t': 'do not',
                        '’em':  'them',
                        'em':  'them',
                        'gonna': 'going to',
                        'gotcha': 'got you',
                        "hadn't": 'had not',
                        "hadn't've": 'had not have',
                        'hadn,t': 'had not',
                        'hadn,t,ve': 'had not have',
                        'hadn;t': 'had not',
                        'hadn;t;ve': 'had not have',
                        'hadn´t': 'had not',
                        'hadn´t´ve': 'had not have',
                        'hadn’t': 'had not',
                        'hadn’t’ve': 'had not have',
                        "hasn't": 'has not',
                        'hasn,t': 'has not',
                        'hasn;t': 'has not',
                        'hasn´t': 'has not',
                        'hasn’t': 'has not',
                        "haven't": 'have not',
                        'haven,t': 'have not',
                        'haven;t': 'have not',
                        'haven´t': 'have not',
                        'haven’t': 'have not',
                        "he'd": 'he would',
                        "he'd've": 'he would have',
                        "he'll": 'he will',
                        "he's": 'he is',
                        'he,d': 'he would',
                        'he,d,ve': 'he would have',
                        'he,ll': 'he will',
                        'he,s': 'he is',
                        'he;d': 'he would',
                        'he;d;ve': 'he would have',
                        'he;ll': 'he will',
                        'he;s': 'he is',
                        'he´d': 'he would',
                        'he´d´ve': 'he would have',
                        'he´ll': 'he will',
                        'he´s': 'he is',
                        'he’d': 'he would',
                        'he’d’ve': 'he would have',
                        'he’ll': 'he will',
                        'he’s': 'he is',
                        "how'd": 'how did',
                        "how'll": 'how will',
                        "how's": 'how is',
                        'how,d': 'how did',
                        'how,ll': 'how will',
                        'how,s': 'how is',
                        'how;d': 'how did',
                        'how;ll': 'how will',
                        'how;s': 'how is',
                        'how´d': 'how did',
                        'how´ll': 'how will',
                        'how´s': 'how is',
                        'how’d': 'how did',
                        'how’ll': 'how will',
                        'how’s': 'how is',
                        "i'd": 'i would',
                        "i'll": 'i will',
                        "i'm": 'i am',
                        "i've": 'i have',
                        'i,d': 'i would',
                        'i,ll': 'i will',
                        'i,m': 'i am',
                        'i,ve': 'i have',
                        'i;d': 'i would',
                        'i;ll': 'i will',
                        'i;m': 'i am',
                        'i;ve': 'i have',
                        "isn't": 'is not',
                        'isn,t': 'is not',
                        'isn;t': 'is not',
                        'isn´t': 'is not',
                        'isn’t': 'is not',
                        "it'd": 'it would',
                        "it'll": 'it will',
                        "it's": 'it is',
                        'it,d': 'it would',
                        'it,ll': 'it will',
                        'it,s': 'it is',
                        'it;d': 'it would',
                        'it;ll': 'it will',
                        'it;s': 'it is',
                        'it´d': 'it would',
                        'it´ll': 'it will',
                        'it´s': 'it is',
                        'it’d': 'it would',
                        'it’ll': 'it will',
                        'it’s': 'it is',
                        'i´d': 'i would',
                        'i´ll': 'i will',
                        'i´m': 'i am',
                        'i´ve': 'i have',
                        'i’d': 'i would',
                        'i’ll': 'i will',
                        'i’m': 'i am',
                        'i’ma': 'i am going to',
                        'i’ve': 'i have',
                        "let's": 'let us',
                        'let,s': 'let us',
                        'let;s': 'let us',
                        'let´s': 'let us',
                        'let’s': 'let us',
                        "ma'am": 'madam',
                        'ma,am': 'madam',
                        'ma;am': 'madam',
                        "mayn't": 'may not',
                        'mayn,t': 'may not',
                        'mayn;t': 'may not',
                        'mayn´t': 'may not',
                        'mayn’t': 'may not',
                        'ma´am': 'madam',
                        'ma’am': 'madam',
                        "might've": 'might have',
                        'might,ve': 'might have',
                        'might;ve': 'might have',
                        "mightn't": 'might not',
                        'mightn,t': 'might not',
                        'mightn;t': 'might not',
                        'mightn´t': 'might not',
                        'mightn’t': 'might not',
                        'might´ve': 'might have',
                        'might’ve': 'might have',
                        "must've": 'must have',
                        'must,ve': 'must have',
                        'must;ve': 'must have',
                        "mustn't": 'must not',
                        'mustn,t': 'must not',
                        'mustn;t': 'must not',
                        'mustn´t': 'must not',
                        'mustn’t': 'must not',
                        'must´ve': 'must have',
                        'must’ve': 'must have',
                        "needn't": 'need not',
                        'needn,t': 'need not',
                        'needn;t': 'need not',
                        'needn´t': 'need not',
                        'needn’t': 'need not',
                        "oughtn't": 'ought not',
                        'oughtn,t': 'ought not',
                        'oughtn;t': 'ought not',
                        'oughtn´t': 'ought not',
                        'oughtn’t': 'ought not',
                        "sha'n't": 'shall not',
                        'sha,n,t': 'shall not',
                        'sha;n;t': 'shall not',
                        "shan't": 'shall not',
                        'shan,t': 'shall not',
                        'shan;t': 'shall not',
                        'shan´t': 'shall not',
                        'shan’t': 'shall not',
                        'sha´n´t': 'shall not',
                        'sha’n’t': 'shall not',
                        "she'd": 'she would',
                        "she'll": 'she will',
                        "she's": 'she is',
                        'she,d': 'she would',
                        'she,ll': 'she will',
                        'she,s': 'she is',
                        'she;d': 'she would',
                        'she;ll': 'she will',
                        'she;s': 'she is',
                        'she´d': 'she would',
                        'she´ll': 'she will',
                        'she´s': 'she is',
                        'she’d': 'she would',
                        'she’ll': 'she will',
                        'she’s': 'she is',
                        "should've": 'should have',
                        'should,ve': 'should have',
                        'should;ve': 'should have',
                        "shouldn't": 'should not',
                        'shouldn,t': 'should not',
                        'shouldn;t': 'should not',
                        'shouldn´t': 'should not',
                        'shouldn’t': 'should not',
                        'should´ve': 'should have',
                        'should’ve': 'should have',
                        "that'd": 'that would',
                        "that's": 'that is',
                        'that,d': 'that would',
                        'that,s': 'that is',
                        'that;d': 'that would',
                        'that;s': 'that is',
                        'that´d': 'that would',
                        'that´s': 'that is',
                        'that’d': 'that would',
                        'that’s': 'that is',
                        "there'd": 'there had',
                        "there's": 'there is',
                        'there,d': 'there had',
                        'there,s': 'there is',
                        'there;d': 'there had',
                        'there;s': 'there is',
                        'there´d': 'there had',
                        'there´s': 'there is',
                        'there’d': 'there had',
                        'there’s': 'there is',
                        "they'd": 'they would',
                        "they'll": 'they will',
                        "they're": 'they are',
                        "they've": 'they have',
                        'they,d': 'they would',
                        'they,ll': 'they will',
                        'they,re': 'they are',
                        'they,ve': 'they have',
                        'they;d': 'they would',
                        'they;ll': 'they will',
                        'they;re': 'they are',
                        'they;ve': 'they have',
                        'they´d': 'they would',
                        'they´ll': 'they will',
                        'they´re': 'they are',
                        'they´ve': 'they have',
                        'they’d': 'they would',
                        'they’ll': 'they will',
                        'they’re': 'they are',
                        'they’ve': 'they have',
                        "wasn't": 'was not',
                        'wasn,t': 'was not',
                        'wasn;t': 'was not',
                        'wasn´t': 'was not',
                        'wasn’t': 'was not',
                        "we'd": 'we would',
                        "we'll": 'we will',
                        "we're": 'we are',
                        "we've": 'we have',
                        'we,d': 'we would',
                        'we,ll': 'we will',
                        'we,re': 'we are',
                        'we,ve': 'we have',
                        'we;d': 'we would',
                        'we;ll': 'we will',
                        'we;re': 'we are',
                        'we;ve': 'we have',
                        "weren't": 'were not',
                        'weren,t': 'were not',
                        'weren;t': 'were not',
                        'weren´t': 'were not',
                        'weren’t': 'were not',
                        'we´d': 'we would',
                        'we´ll': 'we will',
                        'we´re': 'we are',
                        'we´ve': 'we have',
                        'we’d': 'we would',
                        'we’ll': 'we will',
                        'we’re': 'we are',
                        'we’ve': 'we have',
                        "what'll": 'what will',
                        "what're": 'what are',
                        "what's": 'what is',
                        "what've": 'what have',
                        'what,ll': 'what will',
                        'what,re': 'what are',
                        'what,s': 'what is',
                        'what,ve': 'what have',
                        'what;ll': 'what will',
                        'what;re': 'what are',
                        'what;s': 'what is',
                        'what;ve': 'what have',
                        'what´ll': 'what will',
                        'what´re': 'what are',
                        'what´s': 'what is',
                        'what´ve': 'what have',
                        'what’ll': 'what will',
                        'what’re': 'what are',
                        'what’s': 'what is',
                        'what’ve': 'what have',
                        "where'd": 'where did',
                        "where's": 'where is',
                        'where,d': 'where did',
                        'where,s': 'where is',
                        'where;d': 'where did',
                        'where;s': 'where is',
                        'where´d': 'where did',
                        'where´s': 'where is',
                        'where’d': 'where did',
                        'where’s': 'where is',
                        "who'll": 'who will',
                        "who's": 'who is',
                        'who,ll': 'who will',
                        'who,s': 'who is',
                        'who;ll': 'who will',
                        'who;s': 'who is',
                        'who´ll': 'who will',
                        'who´s': 'who is',
                        'who’ll': 'who will',
                        'who’s': 'who is',
                        "won't": 'will not',
                        'won,t': 'will not',
                        'won;t': 'will not',
                        'won´t': 'will not',
                        'won’t': 'will not',
                        "wouldn't": 'would not',
                        'wouldn,t': 'would not',
                        'wouldn;t': 'would not',
                        'wouldn´t': 'would not',
                        'wouldn’t': 'would not',
                        'wontcha': 'will not you',
                        'wancha':   'want you',
                        'woulda': 'would have',
                        'whatcha':  'what are you',
                        'wanna':    'want to',
                        'ya':   'you',
                        "you'd": 'you would',
                        "you'll": 'you will',
                        "you're": 'you are',
                        'you,d': 'you would',
                        'you,ll': 'you will',
                        'you,re': 'you are',
                        'you;d': 'you would',
                        'you;ll': 'you will',
                        'you;re': 'you are',
                        'you´d': 'you would',
                        'you´ll': 'you will',
                        'you´re': 'you are',
                        'you’d': 'you would',
                        'you’ll': 'you will',
                        'you’re': 'you are',
                        '´cause': 'because',
                        '’cause': 'because',
                        "you've": "you have",
                        "could'nt": 'could not',
                        "havn't": 'have not',
                        "here’s": "here is",
                        'i""m': 'i am',
                        "i'am": 'i am',
                        "i'l": "i will",
                        "i'v": 'i have',
                        "wan't": 'want',
                        "was'nt": "was not",
                        "who'd": "who would",
                        "who're": "who are",
                        "who've": "who have",
                        "why'd": "why would",
                        "would've": "would have",
                        "y'all": "you all",
                        "y'know": "you know",
                        "you.i": "you i",
                        "your'e": "you are",
                        "arn't": "are not",
                        "agains't": "against",
                        "c'mon": "common",
                        "doens't": "does not",
                        'don""t': "do not",
                        "dosen't": "does not",
                        "dosn't": "does not",
                        "shoudn't": "should not",
                        "that'll": "that will",
                        "there'll": "there will",
                        "there're": "there are",
                        "this'll": "this all",
                        "u're": "you are",
                        "ya'll": "you all",
                        "you'r": "you are",
                        "you’ve": "you have",
                        "d'int": "did not",
                        "did'nt": "did not",
                        "din't": "did not",
                        "dont't": "do not",
                        "gov't": "government",
                        "i'ma": "i am",
                        "is'nt": "is not"}
        
        self.song_text = song_text
        self.puncts = puncts
        self.contractions = contractions

    def clean_text(self):
        x = str(self.word)
        for punct in self.puncts:
            x = x.replace(punct, '')
        return x
    
    def remove_puncts(self):
        new_list_text = []
        for line in self.list_text:
            words = line.split()
            new_words = []
            for self.word in words:            
                new_words.append(self.clean_text())
            new_list_text.append(" ".join(new_words))
        return new_list_text
    
    def replace_contractions(self):
        new_list_text = []
        for line in self.list_text:
            words = line.split()
            new_words = []
            for word in words:
                if word in self.contractions:
                    new_words.append(self.contractions[word])
                    continue
                if word.lower() in self.contractions:
                    new_words.append(self.contractions[word.lower()])
                    continue
                else:
                    new_words.append(word)
            new_list_text.append(" ".join(new_words))
        return new_list_text
    
    def replace_ing(self):
        new_list_text = []
        for line in self.list_text:
            words = line.split()
            new_words = []
            for word in words:
                if word.endswith("in'"):
                    word = word.replace("in'", "ing")
                if len(word) > 2 and word.endswith("in"):
                    word = word.replace("in'", "ing")
                new_words.append(word)
            new_list_text.append(" ".join(new_words))
        return new_list_text
    
    def remove_between_brackets(self):
        song_str = ''
        for line in self.song_text.split('\n'):
            line = line.rstrip('\n')
            if not line:
                continue 
            song_str += line + ' ... '
            new_song_str = re.sub("[\(\[].*?[\)\]]", " ", song_str)
        new_list_text = []
        line = '' 
        for word in new_song_str.split():
            if word != '...':
                line += word + ' '
            else:
                if not line:
                    continue 
                line = line.strip()
                new_list_text.append(line)
                line = ''
        return new_list_text
    
    def remove_stopwords(self):
        extra_stopwords = {"got","like","im","know","cause","said","get","put","take","going","made","let"}
        stop_words = set(stopwords.words('english')) 
        stop_words = stop_words.union(extra_stopwords)
        new_list_text = []
        for line in self.list_text:
            words = line.split()
            new_words = []
            for word in words:
                if word not in stop_words: 
                    new_words.append(word)
            new_list_text.append(" ".join(new_words))
        return new_list_text

    def stemming(string):
        snowballer = SnowballStemmer()    
        # tokens of words  
        word_tokens = word_tokenize(string) 
        return  " ".join([snowballer.stem(word) for word in word_tokens])

    def lemming(string):
        lemmatizer=WordNetLemmatizer()
        word_tokens = word_tokenize(string) 
        return  " ".join([lemmatizer.lemmatize(word) for word in word_tokens])   
    
    def join_sentences(self):
        return ' '.join(self.list_text)
        
    def start_preprocess(self):
        self.song_text = re.sub(r'[^\x00-\x7f]',r'', self.song_text) 
        self.song_text = self.song_text.lower()
        self.list_text = self.remove_between_brackets()
        self.list_text = self.replace_ing()
        self.list_text = self.replace_contractions()
        self.list_text = self.remove_puncts()
        self.list_text = self.remove_stopwords()
        self.song_text = self.join_sentences()
        return self.song_text