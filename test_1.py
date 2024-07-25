import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pymorphy2 import MorphAnalyzer
from gensim import models
from gensim import corpora 
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt 
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
import PIL as pil

import asyncio
import datetime
from telethon import TelegramClient

from multiapp import MultiApp
from gensim.models.word2vec import Word2Vec
import seaborn as sns
from sklearn.manifold import TSNE
import operator

import feedparser
#**********************************
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#**********************************
import pyLDAvis
import pyLDAvis.gensim 
import xlwt  
import multiprocessing  

cl_mas_data=[]
cl_mas_date=[]

minf=0.1
maxf=1.0
delw=[]
cur_del_words=[]
corpus=[]
all_mes_words=[]

# получен  запросом - await client.start(phone=phone, code_callback=code_callback)
max_posts=1000

#stemmer=nltk.stem.SnowballStemmer(language="russian")
stopwords = stopwords.words('russian') 
morph = MorphAnalyzer() 

flagLocal=False
cl_mas_data=[]

#*****************************************************************

def check_url(url_feed): #функция получает линк на рсс ленту, возвращает        
    # распаршенную ленту с помощью feedpaeser
    return feedparser.parse(url_feed)  

def getDescriptionsDates(url_feed, cntd): #функция для получения описания новости
    
    max_len_data=1000
    #----------------------------------
    date_end=datetime.date.today()
    date_beg=date_end-datetime.timedelta(days=int(cntd)-1) 
    ss="Группа ноостей за "+cntd+" дней в период "+date_beg.strftime('%d %b %Y')+" - "+date_end.strftime('%d %b %Y')
    st.info(ss)
        
    descriptions = []
    lenta = check_url(url_feed)
    for item_of_news in lenta['items']:
        descriptions.append(item_of_news ['description'])
        
    descriptions_filter = []
    dates = []
    lenta = check_url(url_feed)
    i=0
    for item_of_news in lenta['items']:
        #st.info(item_of_news['published'])
        #st.info(date_beg)
        sdate=item_of_news['published']
        sdate=sdate[5:16]
        cur_date=datetime.datetime.strptime(sdate, "%d %b %Y")
        cur_date=cur_date.date()
               
        if cur_date>=date_beg and cur_date<=date_end:
            descriptions_filter.append(descriptions[i])
            dates.append(cur_date.strftime("%d %b %Y"))    
            if len(descriptions_filter)>max_len_data: break
        i+=1                 
    return descriptions_filter, dates

#*****************************************************************
async def rss_parser(httpx_client, posted_q, n_test_chars, filename): 
    st.info("Парсинг новостной ленты "+filename)
    rss_link = 'https://rssexport.rbc.ru/rbcnews/news/20/full.rss'
    max_data=20
    max_request=3
    cur_request=0
    send_message_func=None
        
    while True:
        try:
            response = await httpx_client.get(rss_link)
            cur_request+=1
            st.text('cur_request = '+str(cur_request))
        except:
            await asyncio.sleep(10)
            continue

        feed = feedparser.parse(response.text)

        for entry in feed.entries[::-1]:
            summary = entry['summary']
            title = entry['title']

            news_text = f'{title}\n{summary}'

            head = news_text[:n_test_chars].strip()

            if head in posted_q:
                continue

            if send_message_func is None:
                #st.text(str(cur_request))
                st.info(str(len(cl_mas_data)+1))
                st.text(news_text)
                cl_mas_data.append(news_text)
                cl_mas_date.append(len(cl_mas_data))
                if len(cl_mas_data)>=max_data or cur_request>max_request: return(cl_mas_data, cl_mas_date) 
            else:
                await send_message_func(f'rbc.ru\n{news_text}')

            posted_q.appendleft(head)

        await asyncio.sleep(5)
        
    return(cl_mas_data, cl_mas_date)

def read_excel():
    #*************************************
    try:
        df = pd.read_excel('F:/_Data Sience/Веб_приложения/Streamlit/demo_test_1/postnews1.xlsx')
    except:
        df = pd.read_excel('postnews1.xlsx')
    mas_data = list(df.iloc[0:,0])
    cl_mas_data =[]
    for mes in mas_data:
        strmes=str(mes)
        if len(strmes.strip())>0: cl_mas_data.append(strmes) 
    st.text("принято сообщений канала - "+str(len(cl_mas_data)))
    #*************************************
    cl_mas_date=[]
    return cl_mas_data, cl_mas_date

#*****************************************************************

async def work(filename, cnt_days):
    
    api_id = 16387030
    api_hash = '07bfab67941aa8ebe50f836e3b5c5704'
    ses_name='telemesmonitor'
    phone='+998909790855'
    code='78661' 
    cnt_mes=1500     
    cdays=int(cnt_days)
    date_end=datetime.date.today()
    date_beg=date_end-datetime.timedelta(days=cdays)
      
    loop=asyncio.new_event_loop()
    
    #*************************************
    try:
        client = TelegramClient(ses_name, api_id, api_hash, loop=loop)
        await client.start(phone=phone, code_callback=code_callback)   
    except:
        st.error("Client create/start Error!")
        return cl_mas_data, cl_mas_date
    
    try:
        channel_entity=await client.get_entity(filename)
    except: 
        st.error("Connect Error!")
        return
    try:
        st.text("channel_entity="+str(channel_entity))
        #st.text("44444444444444444444444444444444444444")
        messages = await client.get_messages(channel_entity, limit=cnt_mes)
    except:
        st.error("Channel_entity Error!")
        return
            
    for message in messages:
        mes_date=message.date.date()
        cl_mas_date.append(mes_date)
        if mes_date>=date_beg and mes_date<=date_end:
            #st.text(str(mes_date)) 
            mes=message.message
            if isinstance(mes,str):
                if len(mes.strip())>0: cl_mas_data.append(mes) 
                #st.text(mes)
        
    await client.disconnect()
    
    text_2="Отобрано "+str(len(cl_mas_data))+" сообщений в диапазоне "+str(date_beg)+" - "+str(date_end)
    text_2 = '<p style="font-family:sans-serif; color:Black; font-size: 16px;">'+text_2+'</p>'
    st.markdown(text_2, unsafe_allow_html=True)
 
    return cl_mas_data, cl_mas_date

#*****************************************************************

def code_callback():
   code=st.secrets['code_callback']
   #st.warning(code)
   return code
   
   #while True:
       #ждем код телеграмме, а потом подставляем его в эту функцию 
       #code=st.secrets['code_callback'] 
       #return code
       
#*****************************************************************

class word2vec(object):
    
    def __init__(self, texts, nkw, filename):
        self.texts=texts
        self.nkw=nkw
        self.filename=filename
        self.wrds=[]
        self.cods=[]   
        self.wrdcod=[]   
        
    def tsne_plot(self, model, base_word,list_words,new_gr_words):
        sns.set (font_scale=2.0) 
        labels = []
        tokens = []
        colors=[]
        fontsizes=[]
    
        # Extracting words and their vectors from our trained model 
        close_words = model.wv.most_similar(base_word)
        cl_words=[row[0] for row in close_words]
                
        for word in model.wv.index_to_key:
            wrd_vector = model.wv.word_vec(word)
            tokens.append(wrd_vector)
            labels.append(word)  
            if word in list_words:
                colors.append('blue')
                fontsizes.append(48) 
            elif word==base_word: 
                colors.append('red')
                fontsizes.append(48)
            else:
                colors.append('black')
                fontsizes.append(32)
                
        st.text("Началось сжатие векторов - "+str(datetime.datetime.now()))
        tsne_model = TSNE(perplexity=15, n_components=2, init='pca', random_state=0)
        
        st.text("Началось вычисление близости слов на плоскости - "+str(datetime.datetime.now()))
        new_values = tsne_model.fit_transform(tokens)
        
        st.text("Началась визуализация близости слов на плоскости - "+str(datetime.datetime.now()))
        x = []
        y = []
    
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
       
        plt.figure(figsize=(30, 30)) 
        for i in range(len(x)):
            st.sidebar.text(str(i))
            if labels[i] in cl_words:
                colors[i]='green'
                fontsizes[i]=40
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     fontsize=fontsizes[i],  
                     textcoords='offset points',
                     color=colors[i],  
                     ha='right',
                     va='bottom')
            plt.xlabel("X")
            plt.ylabel("Y")
        canvas = mplt.pyplot.get_current_fig_manager().canvas
        canvas.draw()
        buf = pil.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        st.image(buf,60)
        st.text("Завершена визуализация близости слов на плоскости - "+str(datetime.datetime.now()))
        #for mes in close_words:  st.info(mes)      
           
         
    def view_word2vec(self,model, word, list_names):
        sns.set (font_scale=1.0) 
        
        vectors_words = [model.wv.word_vec(word)]
        word_labels = [word]
        color_list = ['red']
        close_words = model.wv.most_similar(word)
        
        for wrd_score in close_words:
            wrd_vector = model.wv.word_vec(wrd_score[0])
            vectors_words.append(wrd_vector)
            word_labels.append(wrd_score[0])
            color_list.append('blue')
                                
        for wrd in list_names:
            wrd_vector = model.wv.word_vec(wrd)
            vectors_words.append(wrd_vector)
            word_labels.append(wrd)
            color_list.append('green')
        
        # t-SNE reduction
        Y = (TSNE(n_components=2, random_state=0, perplexity=5, init="pca"))
        st.text("Сжатие векторов завершено "+str(datetime.datetime.now()))
        st.text("Подождите. Началось вычисление близости векторов слов на плоскости ... "+str(datetime.datetime.now()))
        Y =Y.fit_transform(vectors_words)
        
        st.text("Началась визуализация близости слов на плоскости - "+str(datetime.datetime.now()))
        # Sets everything up to plot
        df = pd.DataFrame({"x": [x for x in Y[:, 0]],
                    "y": [y for y in Y[:, 1]],
                    "words": word_labels,
                    "color": color_list})
        st.info(df.loc[:,'words'])       
        st.info(df) 
        
        fig, _ = mplt.pyplot.subplots()
        fig.set_size_inches(9, 9)
        # Basic plot
        p1 = sns.regplot(data=df,
                    x="x",
                    y="y",
                    fit_reg=False,
                    marker="o",
                    scatter_kws={"s": 40,
                                "facecolors": df["color"]}
        )
        # Adds annotations one by one with a loop
        for line in range(0, df.shape[0]):
            p1.text(df["x"][line],
            df["y"][line],
            " " + df["words"][line].title(),
            horizontalalignment="left",
            verticalalignment="bottom", size="medium",
            color=df["color"][line],
            weight="normal"
        ).set_size(15)
        mplt.pyplot.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
        mplt.pyplot.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
        mplt.pyplot.title('Визуализация контекстной близости выбранных и других слов к базовому слову <{}>'.format(word.title()))
       
        canvas = mplt.pyplot.get_current_fig_manager().canvas
        canvas.draw()
        buf = pil.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        st.image(buf,60)
        st.text("Завершена визуализация близости слов на плоскости - "+str(datetime.datetime.now()))
        
               
        for mes in close_words: 
            st.info(mes)      
                
        if len(close_words)>3: close_words=close_words[0:3]   
                
        for i in range(len(close_words)):
            self.wrdcod.append(close_words[i])
            self.wrds.append(close_words[i][0])
            self.cods.append(close_words[i][1])
                               
    def model_train(self,texts):
        cores = multiprocessing.cpu_count() 
        w2v_model = Word2Vec(
        min_count=2,
        window=10,
        vector_size=50,
        negative=10,
        workers=cores-1,
        alpha=0.03,
        min_alpha=0.0007,
        sample=6e-5,
        sg=1) 
        st.warning("Начат процесс векторизации слов ...")
        w2v_model.build_vocab(texts)
        st.text("Словарь создан - "+str(datetime.datetime.now()))
        w2v_model.train(texts, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
        st.text("Обучение векторов завершено - "+str(datetime.datetime.now()))        
        return w2v_model
    
    def start_word_2_vec(self,new_gr_words):
        nkw=self.nkw
        l=len(nkw)
        texts=self.texts
        #st.text("***********************************************")
          
        if (l<=0): 
            st.text("Error! Key words?")
            return 
        
        base_word=nkw[0]
        list_words=nkw[1:l]
        
        #st.text("**********************************************************")
        #st.text(base_word)
        #st.text(list_words)
        #st.text("**********************************************************")
     
        w2v_model=self.model_train(texts)   
        self.view_word2vec(w2v_model, base_word,list_words)
        #self.tsne_plot(w2v_model, base_word,list_words,new_gr_words)
        st.text("Векторизация завершена")
            
#*****************************************************************

class LDA(object):
    
    def __init__(self,num_topics,num_words,input_text,nm_chan):
        self.fig_lda=0
        self.buf_lda=0
        self.list_lda=[]
        self.gr_wrd=[]
        self.lda_analysis(num_topics,num_words,input_text,nm_chan)
        self.num_topics
              
    # Предварительная обработка предложений
    def lda_analysis(self,num_topics,num_words,tokens,nm_chan):
        
        # выделение предложений слов с предварительной обработкой
        #st.text(tokens)
    
        # Создание словаря на основе токенизированных предложений
        dict_tokens = corpora.Dictionary(tokens) 
        #print(dict_tokens)
        # Создание терм-документной матрицы (корпуса)
        doc_term_mat = [dict_tokens.doc2bow(token) for token in tokens]
                          
        #*********************************************************************
        # Генерирование LDА-модели
        #ldamodel = models.ldamodel.LdaModel(
        #    doc_term_mat,
        #    num_topics=num_topics,
        #    id2word=dict_tokens,
        #    passes=25)
        #******************************************************************************************
        #   pyLDAVis
        #******************************************************************************************
        best_num_topics=0
        best_coherence_score = 0.00 
        
        for num_topics in range(1,11):
            ldamodel = models.ldamodel.LdaModel(
                corpus=doc_term_mat,
                id2word=dict_tokens,
                num_topics=num_topics,
                random_state=0,
                chunksize=50,
                alpha='auto',
                per_word_topics=True)
        
            coherence_model = models.coherencemodel.CoherenceModel(model=ldamodel, texts=tokens, dictionary=dict_tokens, coherence='c_v')
            coherence_score = coherence_model.get_coherence()   
            #st.warning("Оценка текущей модели LDA для "+str(num_topics)+" тем = "+str(coherence_score))
            if coherence_score>best_coherence_score:
                best_coherence_score=coherence_score
                best_num_topics=num_topics
        
        st.warning("Лучшая модель LDA достигается когда количество тем = "+str(best_num_topics)+" при коэффициенте согласия = "+str(best_coherence_score))
        num_topics=best_num_topics
        self.num_topics=num_topics 
        
        ldamodel = models.ldamodel.LdaModel(
                corpus=doc_term_mat,
                id2word=dict_tokens,
                num_topics=num_topics,
                random_state=0,
                chunksize=50,
                alpha='auto',
                per_word_topics=True)
        
        p = pyLDAvis.gensim.prepare(ldamodel, doc_term_mat, dict_tokens)
        html_string = pyLDAvis.prepared_data_to_html(p)
        #st.warning('***************************************************')
        #st.warning(html_string)
        
        components.html(html_string, width=1300, height=800, scrolling=True)  
        
        #******************************************************************************************
        lst_frm=[]
        new_words=[]
        maxval=0
        list_posts=[]
        
        list_posts.append("Классификация текста канала - "+str(nm_chan) +" по "+str(num_topics)+" категориям")
        list_posts.append(str(num_words) + ' наиболее значимых слов для каждой категории:')
        
        self.gr_wrd=[]
               
        k=0    
        for item in ldamodel.print_topics(num_topics=num_topics, num_words=num_words):
            k+=1
            #list_posts.append('\n ******************************************                Категория - '+str(item[0]))
            # Вывод представительных слов вместе с их
            # относительными вкладами
            list_of_strings = item[1].split(' + ')
            
            cur_wrd=[]
            l=0              
            for text in list_of_strings:
                l+=1
                row_frm=[]             
                weight = text.split('*') [0]
                word = text.split('*') [1]
                #*****************************************************
                #print(word, '==>', round(float(weight) * 100,2) + 1%1)
                #ex.list_posts.addItem(word+'==>'+str(round(float(weight) * 100,2) + 1%1))
                #*****************************************************
                try:
                    ind_word=new_words.index(word)
                except ValueError:    
                    new_words.append(word)
                    ind_word=len(new_words)-1
                    print("new_word="+word)
                    #*************************
                    for i in range(num_topics+1):
                        if i==0: row_frm.append('-')
                        else:    row_frm.append(0)         
                    lst_frm.append(row_frm)
            
                for i in range(num_topics+1):
                    if i==0: lst_frm[ind_word][0]=word 
                    if i==int(item[0]+1):
                        lst_frm[ind_word][int(item[0]+1)]=int(float(weight) * 1000)
                        if round(float(weight) * 1000,0)>maxval: maxval=int(float(weight) * 1000)
                #*****************************************************        
                #list_posts.append(str(k-1)+'/'+str(l)+' --- '+word+' --- '+str(round(float(weight) * 100,2) + 1%1)+'('+str(lst_frm[ind_word][int(item[0]+1)])+')')
                cur_wrd.append(word)
                
            self.gr_wrd.append(cur_wrd)
            
        #for i in range(num_topics):
        #    for j in range(num_words):
        #        st.text(str(i)+"/"+str(j)+"/"+self.gr_wrd[i][j])             
        
        #*****************************************************
        
        frequency={}
        for word in tokens:
            if word in new_words:
                count = frequency.get(word,0)
                frequency[word] = count + 1
        
        frequency_list = frequency.keys()        
        for words in frequency_list:
            list_posts.append(str(words)+' / '+str(frequency[words]))
        
       #*****************************************************
        df=pd.DataFrame(lst_frm)
               
        cols=[]
        for i in range(num_topics+1):
            if i==0: cols.append('word')
            else:    cols.append('gr-'+str(i-1))
        df.columns=cols
        dff=df.copy()
        #***********************************
        
        #***********************************
        mapsize=(60,120)
        fig,ax = mplt.pyplot.subplots(figsize=mapsize)
        mplt.pyplot.title('Тематический профиль канала - '+str(nm_chan),fontsize=80, loc='left')
        dff = df.drop(columns='word')  
        dff.index=new_words
        #st.info(dff)
        sns.heatmap(dff, cmap='Blues_r', linewidths= 5, annot=True, fmt='d')
      
        canvas = mplt.pyplot.get_current_fig_manager().canvas
        canvas.draw()
        buf = pil.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())         
           
        dff=pd.DataFrame(lst_frm)
        dff.columns=cols 
        
        #***********************************
        self.fig_lda=fig
        self.buf_lda=buf
        self.list_lda=list_posts.copy() 
        
        return 

#*****************************************************************

class Prepare(object):    
    
    def __init__(self, mas, del_words, minf, maxf, code_type, min_freq, max_freq):
        #self.stemmer=stemmer 
        self.ru_stopwords = stopwords
        self.morph = morph 
        self.patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
         
        self.mas=mas
        self.del_words=del_words
        self.minf=minf
        self.maxf=maxf
        self.code_type=code_type
        self.min_freq=min_freq
        self.max_freq=max_freq
                        
    def prepareWord(self, old_word):
        new_word=old_word
        if not isinstance(old_word,str): return(" ") 
        #new_word=re.sub(self.patterns, ' ', new_word) 
        #new_word=new_word.translate(new_word,self.patterns)
        new_word=new_word.lower()
        #new_word=stemmer.stem(new_word)
        #new_word=Porter.stem(u(new_word))
        
        if new_word not in self.ru_stopwords and new_word not in self.del_words:  
            if len(new_word)>3:
                if 'NOUN' in morph.tag(new_word)[0]:
                    #print("("+old_word+") = "+new_word)
                    #print("*****************")             
                    return morph.parse(new_word)[0].normal_form            
        return " "     
    
#**********************************************************    

    def histogramm(self, all_mes_words):
    
        st.info("2. Началось формирование гистограммы частот слов в сообщениях") 
        
        word_weight =[]
        new_freqs=[]
        new_words=[]
        new_del_words=[] 
        val=[]
        sort_fwd=[]
        corpus=[]
                 
        #*********************************************************************
        if self.code_type=="абсолютная частота":
            
            my_dictionary = []
            word_freq =[]
                 
            minfreq=1000000
            maxfreq=0
            
            #st.info("**************************************")
            # создание унмкального словаря и подсчет частот слов
            for j in range(len(all_mes_words)):
                for word in all_mes_words[j]: 
                    if word not in my_dictionary:
                        my_dictionary.append(word)
            for i in range(len(my_dictionary)):
                word=my_dictionary[i]
                freq=0
                for j in range(len(all_mes_words)):
                    tmp=all_mes_words[j]
                    freq=freq+tmp.count(word)
                    #if freq>0: st.text(str(freq))
                            
                word_freq.append(freq)
                if minfreq>freq: minfreq=freq
                if maxfreq<freq: maxfreq=freq
                
            st.warning('Исходная информация о корпусе слов до фильтрации')
            st.info('Минимальная абсолютная частота слов до фильтрации = '+str(minfreq))
            st.info('Максимальная абсолютная частота слов до фильтрации = '+str(maxfreq))
            
            #st.info("*********  words/freqs/deciles sorted  by freq  *****************************")
            # создание фрейма слова-частоты с сортировкой по возрастанию частоты
            list_tuples = list(zip( word_freq, my_dictionary))  
            #st.info(list_tuples)
            dfw = pd.DataFrame(list_tuples,columns=['freqs','words'])
            dfw = dfw.sort_values(by='freqs')
            dfw['Decile'] = pd.cut(dfw['freqs'], 10, labels= False)
            len_dfw=len(dfw['Decile'])
            #st.info(str(len_dfw)) 
                                   
            sort_fwd=dfw.values.tolist()
            #st.info(sort_fwd) 
                        
            st.info("***********  Распределение частот по децилям  ***************************")
            # нормализация к диапазону 0.0 - 1.0 
            sum_dec=[]
            for i in range(10):
                sum_dec.append(0)
            for i in range(len_dfw):   
                k=sort_fwd[i][2] 
                sum_dec[k]=sum_dec[k]+sort_fwd[i][0] 
                sort_fwd[i][2]=sort_fwd[i][2]/10
                #st.text(str(sort_fwd[i][1]+" / частота = "+str(sort_fwd[i][0])+" / дециль = "+str(sort_fwd[i][2])))
            for i in range(10):    
                st.text(" дециль = "+str(i/10)+" / суммарная частота = "+str(sum_dec[i])) 
                     
                                    
            #st.info("********** filter decile/words ****************************")   
            # удление редких и частых слов по фильтру 
            minfreq_filter=10000000
            maxfreq_filter=0  
            k1=0
            k2=0
            val=[]
            for i in range(10): val.append(0)
            
            #st.info("minf="+str(self.minf)+" / maxf= "+str(self.maxf))
            for i in range(len_dfw):
                if sort_fwd[i][2]>=self.minf and sort_fwd[i][2]<=self.maxf:
                    if minfreq_filter>sort_fwd[i][0]: minfreq_filter=sort_fwd[i][0]
                    if maxfreq_filter<sort_fwd[i][0]: maxfreq_filter=sort_fwd[i][0]
            
            if minfreq_filter<self.min_freq: minfreq_filter=self.min_freq
            if maxfreq_filter>self.max_freq: maxfreq_filter=self.max_freq
                             
            for i in range(len_dfw):
                if sort_fwd[i][0]>=minfreq_filter and sort_fwd[i][0]<=maxfreq_filter: 
                    new_freqs.append(sort_fwd[i][0])  
                    new_words.append(sort_fwd[i][1])
                    k=int(sort_fwd[i][2]*10) 
                    val[k]=val[k]+sort_fwd[i][0]
                    #st.text('оставлено слово ='+sort_fwd[i][1]+' с частотой='+str(sort_fwd[i][0]))
                    k1+=1
                else:
                    new_del_words.append(sort_fwd[i][1])
                    #st.text('удалено слово ='+sort_fwd[i][1]+' с частотой='+str(sort_fwd[i][0]))   
                    k2+=1    
            
            st.warning('Информация о корпусе слов после фильтрации')
            st.info('Число оставшихся слов = '+str(k1)+', число удаленных слов = '+str(k2))        
            st.info('Минимальная абсолютная частота слов после фильтрации = '+str(minfreq_filter)+" / слова с меньшей частотой удалены")
            st.info('Максимальная абсолютная частота слов после фильтрации = '+str(maxfreq_filter))
            #st.info("**************************************")    
            #**********************************************************
            # вывод гистогаммы частот слов
            #**********************************************************
            sort_freqs=sorted(new_freqs,reverse=False)
            #st.info(sort_freqs)
            new_freqs_words=[]
            cur_freq=sort_freqs[0]
            sum_freq=0
        
            for i in range(len(sort_freqs)):
                if sort_freqs[i]==freq: sum_freq+=sort_freqs[i]
                else:
                    tmp=[]
                    tmp.append(sum_freq)
                    tmp.append(str(cur_freq))
                    new_freqs_words.append(tmp)
                    cur_freq=sort_freqs[i]
                    sum_freq=sort_freqs[i]
                if i==len(sort_freqs)-1:    
                    tmp=[]
                    tmp.append(sum_freq)
                    tmp.append(str(cur_freq))
                    new_freqs_words.append(tmp)
                                  
            df=pd.DataFrame(new_freqs_words, columns=["freq","word"])
            df_freqs=df["freq"]
            df_names=df["word"]
        
            fig, ax = mplt.pyplot.subplots(figsize =(10, 7))
            ax.barh(df_names, df_freqs, color='blue') 
            canvas = mplt.pyplot.get_current_fig_manager().canvas
            canvas.draw()
            buf = pil.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())               
            st.image(buf,60)
            
            #**********************************************************                     
            
            if minfreq_filter>0.1: cur_freq=minfreq_filter 
            else: cur_freq=1 
            cur_words=[]
            unic_words=[]
            #for row in dfw.itertuples():
            #    if row.freqs>1:
            #        st.info(str(row.freqs) +"/"+ str(row.words)+"/"+ str(row.Decile))
            i=0  
            for row in dfw.itertuples():
                i+=1
                #st.warning(str(i))
                #st.warning(str(len_dfw))
                if row.freqs<minfreq_filter or row.freqs>maxfreq_filter:
                    continue 
                #st.info(row.freqs)
                #st.info(cur_freq)
                if row.freqs==cur_freq:
                    cur_words.append(row.words)
                    unic_words.append(row.words)
                    #st.info(cur_words)
                else:
                    if len(cur_words)>0:
                        st.info(str(len(cur_words))+' слов с частотой - '+str(cur_freq))
                        st.text(cur_words)
                    cur_freq=row.freqs
                    cur_words=[]
                    cur_words.append(row.words)
                    unic_words.append(row.words)
                    
            if i>=len_dfw:    
                st.info(str(len(cur_words))+' слов с частотой - '+str(cur_freq))
                st.text(cur_words)
                    
            #st.info(str(row.freqs) +"/"+ str(row.words)+"/"+ str(row.Decile)) 
            
            for i in range(10):     
                st.text(" дециль = "+str(i/10)+" / суммарная частота = "+str(val[i])) 
            
                       
        #*********************************************************************        
        if self.code_type=="относительная частота":
                      
            mydict= corpora.Dictionary(all_mes_words)
            mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in all_mes_words] 
            tfidf = models.TfidfModel(mycorpus, smartirs='ntn')
            
    
            maxf=0
            minf=1000
            for doc in tfidf[mycorpus]:
                for id, freq in doc:
                    freq=np.around(freq,2)
                    if freq>maxf: maxf=freq
                    if freq<minf: minf=freq
            delta=maxf-minf
            minfreq=minf+self.minf*delta             
            maxfreq=minf+self.maxf*delta
            corpus=[]
            for doc in tfidf[mycorpus]:
                curmes=[]
                for id, freq in doc:
                    if freq>=minfreq and freq<=maxfreq: 
                        curmes.append(mydict[id])
                        val.append(freq/maxf)
                corpus.append(curmes)
                
            #st.warning(val)
            #for mes in corpus:
            #    st.warning(mes)
            #st.info("*************************************")
            st.info('Минимальная относительная частота слов после фильтрации = '+str(minfreq))                 
            st.info('Максимальная относительная частота слов после фильтрации = '+str(maxfreq))     
            st.info("**************************************")
        
            fig, ax = mplt.pyplot.subplots(figsize =(10, 7))
            ax.hist(val, bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            canvas = mplt.pyplot.get_current_fig_manager().canvas
            canvas.draw()
            buf = pil.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
            st.image(buf,60)
        
        return new_del_words, fig, buf, val, sort_fwd, corpus, unic_words 

#**********************************************************
        
    def prepare_all(self):
        st.info("1. Началось создание корпуса слов")
        all_mes_words=[]
        all_sent_words=[]
        all_words=[]
         
        for line in self.mas:  
            if len(line)<10: continue
            cur_mes_words=[]
            for sent in nltk.sent_tokenize(line): 
                cur_sent_words=[]
                for word in nltk.word_tokenize(sent):
                    word=self.prepareWord(word)  
                    if word!=" ":
                        cur_sent_words.append(word)
                        all_words.append(word)
                        cur_sent_words.append(word)
                        cur_mes_words.append(word)
                all_sent_words.append(cur_sent_words)        
            all_mes_words.append(cur_mes_words)    

        new_del_words, fig, buf, val,sort_fwd, corpus, unic_words = self.histogramm(all_mes_words)
        return all_mes_words, all_sent_words, all_words, new_del_words, fig, buf, val, sort_fwd, corpus, unic_words
    
    
#**********************************************************

def cluster_doc2vec(sel_mas_data, all_mes_words):

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_mes_words)] 
    print(documents)
    cnt_epochs=30

    model = Doc2Vec(vector_size=300, workers=4, epochs=cnt_epochs)   
    model.build_vocab(documents) 
   
    #print(model)
    #print(len(documents))

    epochs = 0
    while epochs < cnt_epochs:
        model.train(documents, total_examples=len(documents), epochs=cnt_epochs)  
        epochs=epochs+1
    kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=100) 
    X = kmeans_model.fit(model.dv.vectors)
    labels=kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(model.dv.vectors)
    pca = PCA(n_components=2).fit(model.dv.vectors)
    datapoint = pca.transform(model.dv.vectors)

    cnt_cluster=4
    label1 = ["#008000", "#0000FF", "#800080","#FFFF00"]
    color = [label1[i] for i in labels]

    fig=mplt.pyplot.figure()
    fig.set_size_inches(15,10) 
    mplt.pyplot.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)
    mplt.pyplot.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
 
    ax1 = mplt.pyplot.gca() 
    ax1.set_title('Визуальная картина распределения отобранных сообщений по 4 кластерам',fontsize=14)
    mes=[]
    for i in range(cnt_cluster): mes.append("Кластер_"+str(i+1)) 

    for i in range(cnt_cluster):
        mplt.pyplot.text(6.0,1.0+1.0*i, mes[i].upper(), fontsize=10, color=label1[i])
 
    canvas = mplt.pyplot.get_current_fig_manager().canvas
    canvas.draw()
    buf_d2v = pil.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    st.image(buf_d2v,40)
    
    
    cur_row=[]
    for i in range(cnt_cluster):
        cur_row.append(0)
    
    cluster_text = [0] * 500 
    for i in range(4): cluster_text[i] = [0] * 500 
                 
    for i in range(len(labels)-1):
        #wb.get_sheet(labels[i]+1).write(cur_row[labels[i]]+2, 0, mas_all[i], style0)
        cluster_text[labels[i]][cur_row[labels[i]]]=sel_mas_data[i]   
        cur_row[labels[i]]+=1
       
    all_row=0
    for i in range(4):
        all_row=all_row+cur_row[i]
           
    st.info("Распределение текстов отобранных сообщений по 4 кластерам")
    st.info("Всего отобрано "+str(all_row)+" сообщений")
    
    for i in range(4):
        st.warning("Кластер_"+str(i+1) + "/ сообщений = "+str(cur_row[i])) 
        st.info("**********************************************************************")
        for j in range(cur_row[i]):
            st.text(" ***  сообщение -"+str(j+1)+"    "+cluster_text[i][j]) 
    


def start_corpus(mas_data, minf, maxf, code_type, min_freq, max_freq):    
    #start_corpus(file, minf, maxf):   
    #df = pd.read_excel('postnews1.xlsx')
    #df.columns=['A']
    #mas_data = list(df['A'])
            
    prep = Prepare(mas_data, delw, minf, maxf, code_type, min_freq, max_freq)
    all_mes_words, all_sent_words, all_words, curdelw, fig, buf, val, sort_fwd, corpus, unic_words = prep.prepare_all()
    
    list_posts=[]
    list_posts.append(" *****   Информация о корпусе после удаления редких/частых слов    *****")
    list_posts.append("Всего сообщений = "+str(len(mas_data)))
    list_posts.append("Всего преддложений = "+str(len(all_sent_words)))
    list_posts.append("Всего слов = "+str(len(all_words)))
    list_posts.append("Всего удалено слов по фильтру = "+str(len(curdelw)))
    list_posts.append("Всего осталось слов после фильтрации = "+str(len(all_words)-len(curdelw)))
    list_posts.append("Всего осталось уникальных слов после фильтрации = "+str(len(unic_words)))
    
    old_all_mes_words=[]
    if len(corpus)>0:
        old_all_mes_words=all_mes_words.copy()
        all_mes_words=corpus.copy()
                        
    return buf, fig, list_posts, all_mes_words, all_sent_words, curdelw, all_words, sort_fwd, old_all_mes_words

def save_corpus_to_excel(allmes, all_words, del_words, cl_mas_data, all_mes_words, sort_fwd, old_all_mes_words):
    #**************************************************
    #for w in del_words: st.info(w)
    path='F:/_Data Sience/Веб_приложения/Streamlit/demo_test_1' 
    corpus_file=path+'/1_corpus.xls' 
    # Создать новую книгу
    wb = xlwt.Workbook()
    # Добавить новую страниц
    ws = wb.add_sheet('mes_words')
    ws1 = wb.add_sheet('words_freqs')
    ws2 = wb.add_sheet('messages')
    ws3 = wb.add_sheet('new_mes_words')
    ws4 = wb.add_sheet('new_unic_words')	
    ws5 = wb.add_sheet('sort_fwd')	  
    ws6 = wb.add_sheet('tf_idf')
    ws7 = wb.add_sheet('tf_idf_new')
         
    #установить шрифт и стиль вывода
    # 0-black
    font0 = xlwt.Font()
    font0.name = 'Times New Roman'
    font0.colour_index = 0 # 0-0black
    # 1-red
    font1 = xlwt.Font()
    font1.name = 'Times New Roman'
    font1.colour_index = 4 
    # 2-blue
    font2 = xlwt.Font()
    font2.name = 'Times New Roman'
    font2.colour_index = 2
    # 3-green
    font3 = xlwt.Font()
    font3.name = 'Times New Roman'
    font3.colour_index = 3
    
    style0 = xlwt.XFStyle()
    style0.font = font0
    style0.alignment.horizontal='center'
    style0.alignment.vertical='center'
    style1 = xlwt.XFStyle()
    style1.font = font1
    style2 = xlwt.XFStyle()
    style2.font = font2
    style3 = xlwt.XFStyle()
    style3.font = font3

    ws.write(0, 0, "Содержание корпуса: слова-существительные каждого сообщения", style1)
    for i in range(len(allmes)):
        ws.write(i+1, 0, str(i+1), style1)
        for j in range(len(allmes[i])):
                ws.write(i+1, j+1, str(allmes[i][j]), style0) 
        
    ws1.write(0, 0, "Уникальные слова всех сообщений, отсортированные по алфавиту и по убыванию их частоты, а также удаленные слова по убыванию их частоты ", style1)
    lst_word_freq=[]
    sort_allwords=sorted(list(set(all_words)))
    for i in range(len(sort_allwords)):
        ws1.write(i+1, 0, str(i+1), style0)
        wrd=sort_allwords[i]
        ws1.write(i+1, 1, wrd, style1)
        cnt=all_words.count(sort_allwords[i])
        ws1.write(i+1, 2, cnt, style0)    
        tmp=[]
        tmp.append(cnt)
        tmp.append(wrd)
        lst_word_freq.append(tmp)
        
    sum_all_words_freq=0
    cnt_long_words=0
    df_word_freq=pd.DataFrame(lst_word_freq)
    df_word_freq.columns=["freq", "word"]
    sort_df_word_freq=df_word_freq.sort_values(by='freq', ascending=False)
        
    unic_50=[]
    cur_freq_group=sort_df_word_freq.iloc[0,0]
    s=str(cur_freq_group)+" / "
    
    for i in range(len(sort_df_word_freq)):
        cur_freq=sort_df_word_freq.iloc[i,0]
        cnt_long_words=cnt_long_words+1
        ws1.write(i+1, 4, str(cur_freq), style1)
        sum_all_words_freq=sum_all_words_freq+cur_freq
        ws1.write(i+1, 5, sort_df_word_freq.iloc[i,1], style0)
        if cur_freq_group==cur_freq: s=s+sort_df_word_freq.iloc[i,1]+" / "
        else:
            unic_50.append(s)
            cur_freq_group=cur_freq
            s=str(cur_freq_group)+" / "+sort_df_word_freq.iloc[i,1]+" / "
            #st.info(str(i+1)+" - "+sort_df_word_freq.iloc[i,1]+" - "+str(cur_freq))
                           
    ws1.write(len(sort_df_word_freq)+2, 4, str(sum_all_words_freq), style2)
        
    ws2.write(0, 0, "Список сообщений", style2)
    for i in range(len(cl_mas_data)):
        ws2.write(i+1, 0, str(i+1), style0)			
        ws2.write(i+1, 1, cl_mas_data[i], style0)
        
    ws3.write(0, 0, "Список существительных в сообщениях без удаленных слов", style2)
    unic_new_words=[] 
    for i in range(len(all_mes_words)):
        ws3.write(i+1, 0, str(i+1), style0)
        j=2
        for w in all_mes_words[i]:
            if w not in unic_new_words: 
                unic_new_words.append(w) 
            ws3.write(i+1, j, w, style0)
            j+=1
	
    ws4.write(0, 0, "Список уникальных слов после удаления", style2)
    for i in range(len(unic_new_words)): 
        ws4.write(i+1, 0, str(i+1), style0)
        ws4.write(i+1, 1, unic_new_words[i], style0)
    
    ws5.write(0, 0, "Список слов, частот, децилей", style2)
    for i in range(len(sort_fwd)): 
        ws5.write(i+1, 0, str(i+1), style0)
        ws5.write(i+1, 1, sort_fwd[i][0], style0)
        ws5.write(i+1, 2, sort_fwd[i][1], style0)
        ws5.write(i+1, 3, sort_fwd[i][2], style0)
    
    mydict= corpora.Dictionary(old_all_mes_words )
    mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in old_all_mes_words]  
    tfidf = models.TfidfModel(mycorpus, smartirs='ntn') 
    
    i=0
    for doc in tfidf[mycorpus]:
        j=0
        for id, freq in doc:
            ws6.write(i+1, j, mydict[id]+"("+str(np.around(freq, 0, out=None))+")", style0)
            j+=1
            #if freq>=minfreq and freq<=maxfreq: curmes.append(curword)
        i+=1
    
    i=0
    for mes in all_mes_words:
        j=0
        for w in mes: 
            ws7.write(i+1, j, w, style0)
            j+=1
        i+=1    
    #st.text(" ----------------------------------------------------------- ")
    #st.info("Общее количество слов всех сообщений с частотой более 1 - "+str(cnt_long_words))
    #st.info("Общая частота слов всех сообщений с частотой более 1 - "+str(sum_all_words_freq))
        
    if len(del_words)>0:
        
        st.text(" ----------------------------------------------------------- ")
        #st.info("Уникальные удаленные слова всех сообщений, отсортированные по убыванию их частоты - "+str(len(del_words)))
            
        sort_delwords=sorted(del_words)
        lstdel_word_freq=[]
        for i in range(len(sort_delwords)):
            wrd=sort_delwords[i]
            cnt=all_words.count(sort_delwords[i])
            tmp=[]
            tmp.append(cnt)
            tmp.append(wrd)
            lstdel_word_freq.append(tmp)
            
        sum_del_words_freq=0
        df_delword_freq=pd.DataFrame(lstdel_word_freq)
        df_delword_freq.columns=["freq", "word"]
        sort_df_delword_freq=df_delword_freq.sort_values(by='freq', ascending=False)
        for i in range(len(sort_df_delword_freq)):
            ws1.write(i+1, 9, str(i+1), style0) 
            cur_freq=sort_df_delword_freq.iloc[i,0]
            if cur_freq>0:
                sum_del_words_freq=sum_del_words_freq+cur_freq
                ws1.write(i+1, 10, str(cur_freq), style1)
                ws1.write(i+1, 11, sort_df_delword_freq.iloc[i,1], style0)
                #st.info(str(i+1)+" - "+sort_df_delword_freq.iloc[i,1]+" - "+str(cur_freq))
        ws1.write(len(sort_df_delword_freq)+2, 10, str(sum_del_words_freq), style2)
        #st.text(" ----------------------------------------------------------- ")
        #st.info("Общee количество (и частота) удаленных слов всех сообщений - "+str(sum_del_words_freq))             
            	
    wb.save(corpus_file) 
    st.warning("Данные корпуса сохранены")

#**************************************************************

st.set_page_config(layout="wide")

if 'file_name' not in st.session_state:
    st.session_state.file_name = " "
if 'lda_group_words' not in st.session_state:
    st.session_state.lda_group_words = []
if 'all_mes_words' not in st.session_state:
    st.session_state.all_mes_words = []
if 'sent_words' not in st.session_state:
    st.session_state.sent_words = []
if 'cl_mas_data' not in st.session_state:
    st.session_state.cl_mas_data = []
if 'cl_mas_date' not in st.session_state:
    st.session_state.cl_mas_date = []


st.header('Web-сервис: тематичеcкий онлайн анализ контента новостных каналов')
try:
    img=pil.Image.open('photo.jpg')
except:
    img=pil.Image.open('F:/_Data Sience/Веб_приложения/Streamlit/demo_test_1/photo.jpg')    
st.sidebar.image(img, width=250)
    
def corpus():
    
    flagExcel=False 
    flagTelegram=True
    sns.set(font_scale=1)
   
    text_1 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Создание корпуса слов выбранного канала</p>'
    st.markdown(text_1, unsafe_allow_html=True)
    #list_chan=["https://www.kommersant.ru/RSS/news.xml", "https://lenta.ru/rss/","https://www.vesti.ru/vesti.rss"]
    list_chan=["@kunuzru", "@gazetauz","@podrobno"]
    filename = st.sidebar.selectbox("Выберите новостной канал",list_chan)
    
    cnt_days = st.sidebar.selectbox("Выберите количество дней от текущей даты",["1","2","3","4","5","6","7","8","9","10","20","30"],index=11)
    min_freq=st.sidebar.selectbox("Выберите минимальную частоту слов",["1","2","3","4","5","6","7","8","9","10"],index=0)
    min_freq=int(min_freq)
    max_freq=st.sidebar.number_input("Выберите максимальную частоту слов")
    max_freq=int(max_freq)
    
    code_type = st.sidebar.selectbox("Выберите тип кодирования частоты слов",["абсолютная частота","относительная частота"],index=0)
    min_tfidf = st.sidebar.selectbox("Выберите мин. уровень частоты слов",["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"],index=0)
    max_tfidf = st.sidebar.selectbox("Выберите макс. уровень частоты слов",["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],index=9)
    minf=float(min_tfidf)
    maxf=float(max_tfidf)
   
    allmes=[]
    cl_mas_data=[]
    cl_mas_date=[]
    all_mes_words=[]
            
    but_corpus=st.sidebar.button("Создать корпус")
    if but_corpus:
          
        if flagExcel==True:
            cl_mas_data, cl_mas_date = read_excel()
            st.text("len_cl_mas_data="+str(len(cl_mas_data)))
            st.session_state.cl_mas_data=cl_mas_data
            st.session_state.cl_mas_date=cl_mas_date
        else:
            if flagTelegram==True:
                st.info("Парсинг телеграмм-канала")
                cl_mas_data, cl_mas_date = asyncio.run(work(filename, cnt_days)) 
            else:    
                url=filename  
                st.info("Парсинг новостной ленты "+url)
                cl_mas_data, cl_mas_date = getDescriptionsDates(url, cnt_days) 
            
        if len(cl_mas_data)==0: return
                
                    
        #for mes in cl_mas_data:
        #    st.text(mes)
            
        buf, fig, listp, allmes, sent_words, del_words, all_words, sort_fwd, old_all_mes_words = start_corpus(cl_mas_data, minf, maxf, code_type, min_freq, max_freq)
        
        st.session_state.sent_words=sent_words 
        
        #st.text(""+str(len(allmes)))
        if len(allmes)>0:
            st.info("3. Корпус создан")
           
            for curmes in listp:
                st.info(curmes)
            for i in range(0,len(cl_mas_data)):
                #st.info(str(i+1)) 
                st.text(str(i+1)+".   /"+str(cl_mas_date[i])+"/ "+str(cl_mas_data[i]))                
        else:
            st.error("Ошибка! Корпус не создан")
    
        for i in range(len(allmes)):
            curmes=[]    
            for word in allmes[i]:
                if word not in del_words:
                    curmes.append(word)    
            all_mes_words.append(curmes)     
            
        #save_corpus_to_excel(allmes, all_words, del_words, cl_mas_data, all_mes_words, sort_fwd, old_all_mes_words) 
            
    st.session_state.file_name=filename
    st.session_state.all_mes_words = all_mes_words
    st.session_state.cl_mas_data=cl_mas_data
    st.session_state.cl_mas_date=cl_mas_date      
         
def profil():  
    
    sns.set(font_scale=5)
    sel_cntgroup = 10
    sel_cntwords = 10
    
    text_1 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;>Глобальный тематический профиль -  основные слова выбранного канала<, объединенные в группы</p>'
    st.markdown(text_1, unsafe_allow_html=True)
    
    filename=st.session_state.file_name
    allmes=st.session_state.all_mes_words
    cl_mas_data=st.session_state.cl_mas_data
    
    if len(allmes)==0:
        st.error("Корпус не создан!")
        return
    
    #sel_cntgroup = st.sidebar.selectbox("Выберите количество тематических групп",["1","2","3","4","5","6","7","8","9","10"],index=9)
    #sel_cntwords = st.sidebar.selectbox("Выберите количество слов в группе",["1","2","3","4","5","6","7","8","9","10"],index=9)
    #sel_cntgroup=int(sel_cntgroup)
    #sel_cntwords=int(sel_cntwords)
        
    but_lda=st.sidebar.button("Создать глобальный профиль")
    if but_lda:             
        st.info("1. Начался анализ слов методом латентного размещения Дирихле(LDA)")
        st.warning("Подождите ...")
        lda=LDA(sel_cntgroup,sel_cntwords,allmes,filename) 
        st.info("2. Вывод тепловой карты (более светлый цвет - более частое использование слова)")
        st.image(lda.buf_lda,20)
        st.session_state.lda_group_words = lda.gr_wrd
        st.session_state.buf_lda=lda.buf_lda 
        for mes in lda.list_lda:
            st.info(mes)
        #st.write(st.session_state.lda_group_words)
        #*****************************************************************
        # Биграмы/триграмы, связанные с ключевыми словами профиля 
        #****************************************************************
        text_all=''
        for line in cl_mas_data: 
            text_all=text_all+' '+"".join(line)

        df1 = pd.DataFrame({'text': [text_all]})     
        vectorizer = CountVectorizer(ngram_range=(2, 3))  
        doc_vec = vectorizer.fit_transform(df1.iloc[0])
        t_freqs=doc_vec.toarray().transpose().tolist()
        freqs=[]
        for i in range(len(t_freqs)): freqs.append(t_freqs[i][0])
        names=vectorizer.get_feature_names()
                    
        lst_bigram=[]
        for i in range(len(freqs)):
            if len(names[i])==0: continue
            t=[]
            t.append(names[i])
            t.append(freqs[i])
            lst_bigram.append(t)
        df2 = pd.DataFrame(lst_bigram, columns=['2_3_gram', 'freq'])  
        sorted_df = df2.sort_values(by='freq', ascending=False)
        #***********************************
        st.warning('Тематическая группы, их ключевые слова и вязанные биграммы/триграммы')
        #st.info(sorted_df)
        min_freq=5
                                                
        for i in range(lda.num_topics):
            st.warning('Тематическая группа - '+str(i))
            for ii in range(sel_cntwords): 
                st.info(str(ii)+". "+lda.gr_wrd[i][ii])
                k=0
                j=0
                cur_bigrams=[] 
                while j<len(sorted_df) and k<9 and sorted_df.iloc[j,1]>=min_freq:
                    ngr_lst=sorted_df.iloc[j,0].split()
                    ngr_new=[] 
                    for wrd in ngr_lst:
                        if 'NOUN' in morph.tag(wrd)[0]: ngr_new.append(morph.parse(wrd)[0].normal_form)
                            
                    #st.info(lda.gr_wrd[i][ii])
                    wrd=lda.gr_wrd[i][ii]
                    wrd=wrd[1:len(wrd)-1]
                    #st.info(wrd)
                    if wrd in ngr_new:
                        text=sorted_df.iloc[j,0]+"("+str(sorted_df.iloc[j,1])+")"
                        cur_bigrams.append(text)
                        k+=1
                    j+=1
                if len(cur_bigrams)>0: st.info(cur_bigrams)
        
        
        
def search():
    
    sns.set(font_scale=1)
    
    text_2 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Локальный тематический профиль - сообщения, содержащие ключевые слова выбранной группы</p>'
    st.markdown(text_2, unsafe_allow_html=True)    
    
    filename=st.session_state.file_name
    gr_wrd=st.session_state.lda_group_words
    all_mes=st.session_state.all_mes_words
    sent_words=st.session_state.sent_words
    cl_data=st.session_state.cl_mas_data
    cl_date=st.session_state.cl_mas_date
    buf_lda=st.session_state.buf_lda 
    
    #for i in range(len(cl_data)):
    #    st.info(cl_data[i])
    #    st.info(all_mes[i])
    
    if len(gr_wrd)==0: 
        st.error("Ошибка! Глобальный тематический профиль не создан.")
        return
    
    #for curmes in lda.list_lda:
    #    st.text(curmes)
    st.info("При выборе группы и ключевых слов для локального профиля - используйте данные глобального профиля")
    st.image(buf_lda,20)    
    
    sel_findgroup = st.sidebar.selectbox("Выберите группу для поиска",["0","1","2","3","4","5","6","7","8","9"],index=0)
    if sel_findgroup:
        new_gr_words=[]
        old_gr_words=gr_wrd[int(sel_findgroup)]
        for curw in old_gr_words:
            new_gr_words.append(curw)
        sel_findwords = st.sidebar.multiselect("Выберите для анализа локального профиля до трех ключевых слов в порядке их важности",(new_gr_words))
        if sel_findwords:
            but_find=st.sidebar.button("Создать локальный профиль")  
            if but_find:
                progress_bar = st.progress(0)  
                st.warning("Подождите ...")
                srch_mes=[]
                cntmes=len(all_mes)
                if cntmes>=100: delta=(cntmes//10)
                else: delta=100//cntmes
                curdelta=0
                sel_mas_data=[]
                sel_data=[]
                dbeg=""
                dend=""
                k=1
                cod_mes=[]
                for i in range(len(all_mes)):
                    if i>curdelta:
                        curdelta+=delta
                        if curdelta<100: progress_bar.progress(curdelta)
                        else:
                           progress_bar.progress(100)
                           curdelta=1000000
                    
                    tmp_sel_findwords=sel_findwords.copy()
                    sel_findwords=[]
                    for kwd in tmp_sel_findwords:
                        kwd=kwd.replace('"','',2)
                        sel_findwords.append(kwd)
                                            
                    keywrd=list(set(all_mes[i])&set(sel_findwords))
                                           
                    if len(keywrd)>0:
                        text_tmp=str(k)+" ("+str(i)+")  *** "+str(cl_date[i])+" - ("+", ".join(keywrd)+" ) ***** "
                        text_tmp=text_tmp+"                  "+cl_data[i]
                        srch_mes.append(text_tmp)
                        sel_mas_data.append(cl_data[i]) 
                        sel_data.append(all_mes[i])
                        if k==1: dbeg=str(cl_date[i])
                        dend=str(cl_date[i])
                        #st.text("*************")
                        #st.text(keywrd)
                        maxcode=0
                        curcode=0
                        for j in range(len(sel_findwords)-1,-1,-1):
                            #st.text(str(j)+"----------------"+sel_findwords[j])
                            for jj in range(len(keywrd)):
                                #st.text(str(jj)+"-"+keywrd[jj])
                                if sel_findwords[j]==keywrd[jj]:
                                    #st.text("code="+str(len(sel_findwords)-1-j))
                                    #st.text("*************")
                                    curcode=len(sel_findwords)-1-j
                                    if curcode>maxcode: maxcode=curcode
                        cod_mes.append([text_tmp,maxcode])        
                                    
                        k+=1
                                                
                #******************************************************************
                text_2 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Отобранные по ключевым словам сообщения</p>'
                st.markdown(text_2, unsafe_allow_html=True)
                text_2="В диапазоне "+dbeg+" - "+dend+" отобрано "+str(len(cod_mes))+" сообщений"
                text_2 = '<p style="font-family:sans-serif; color:Black; font-size: 20px;">'+text_2+'</p>'
                st.markdown(text_2, unsafe_allow_html=True)
                cod_mes=sorted(cod_mes, key=operator.itemgetter(1), reverse = True)
                for mes in cod_mes:
                    st.info(mes[0])
                    
                if len(srch_mes)==0:
                    st.error("Сообщения не найдены") 

                #******************************************************************
                # векторный анализ близости слов к выбранным
                #******************************************************************
                #''
                srch_mes_new=[] 
                wrd_cods=[]
                if len(sel_findwords)>0:
                    model_texts=sel_data 
                    w2vec=word2vec(sel_data, sel_findwords, filename)
                    w2vec.start_word_2_vec(new_gr_words)
                    for wcod in w2vec.wrdcod:
                        wrd_cods.append(wcod)
                                       
                    if len(w2vec.wrds)>0:
                        dbeg=""
                        dend=""
                        k=0
                        
                        for i in range(len(all_mes)):
                            keywrd=list(set(all_mes[i])&set(w2vec.wrds)) 
                            if len(keywrd)>0:
                                text_tmp=" ("+str(i)+")  *** "+str(cl_date[i])+" - ("+", ".join(keywrd)+" ) ***** "
                                srch_mes_new.append(text_tmp+"                  "+cl_data[i])
                                if k==1: dbeg=str(cl_date[i])
                                dend=str(cl_date[i])
                                k+=1
                    st.warning("Подождите ...")
                    cluster_doc2vec(sel_mas_data, sel_data)  
                
                if len(wrd_cods)==0: return
                    
                text_1 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Список слов (до трех), наиболее контекстно связанных с базовым ключевым словом</p>'
                text_1 =text_1+'<p style="font-family:sans-serif; color:Red; font-size: 24px;">'+sel_findwords[0]+'</p>'
                st.markdown(text_1, unsafe_allow_html=True)
                for wcod in wrd_cods:
                    st.info(wcod)
                
                if len(srch_mes_new)==0: return
                        
                text_2 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Дополнительно отобранные сообщения, контекстно связанные с заданными ключевыми словами </p>'
                st.markdown(text_2, unsafe_allow_html=True)
                k=1
                for mes in srch_mes_new:
                    st.info("("+str(k)+") "+mes)
                    k+=1
                #'''    
                
def myhelp():
    st.text("HELP") 
   
app = MultiApp()
app.add_app("Создание корпуса", corpus)
app.add_app("Анализ глобального профиля", profil)
app.add_app("Анализ локального профиля", search)
app.add_app("Инструкция", myhelp)
app.run()


