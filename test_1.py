import streamlit as st
import pandas as pd
from pymorphy2 import MorphAnalyzer
from gensim import models, corpora
import numpy as np
import matplotlib as mplt
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
import PIL as pil
import openpyxl

import io
import asyncio
import datetime
from telethon import TelegramClient

from multiapp import MultiApp
from gensim.models.word2vec import Word2Vec
import seaborn as sns
from sklearn.manifold import TSNE
import operator

import httpx
from collections import deque
import feedparser


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
    
    date_end=datetime.date.today()
    date_beg=date_end-datetime.timedelta(days=int(cntd)) 
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
    if flagLocal==True:
        df = pd.read_excel('F:/_Data Sience/Веб_приложения/Streamlit/demo_test_1/postnews1.xlsx')
    else:
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

async def work(filename, cnt_days):
    
    api_id = 16387030
    api_hash = '07bfab67941aa8ebe50f836e3b5c5704'
    ses_name='telemesmonitor'
    phone='+998909790855'
    code='26975'    
    cnt_mes=1500     
    cdays=int(cnt_days)
    date_end=datetime.date.today()
    date_beg=date_end-datetime.timedelta(days=cdays)
      
    loop=asyncio.new_event_loop()
    
    #*************************************
    try:
        client = TelegramClient(ses_name, api_id, api_hash,loop=loop)
        # #st.text("22222222222222222222222222222222222222")
        await client.start(phone=phone, code_callback=code_callback) 
    except:
        st.error("Client create/start Error!")
        return cl_mas_data, cl_mas_date
        
    #st.text("33333333333333333333333333333333333333")
    
    try:
        channel_entity=await client.get_entity(filename)
    except: 
        st.error("Connect Error!")
        return
    try:
        #st.text("channel_entity="+str(channel_entity))
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
        
def code_callback():
   while True:
       #ждем код телеграмме, а потом подставляем его в эту функцию 
       code='26975'
       return code
     
#*****************************************************************

class word2vec(object):
    
    def __init__(self, texts, nkw, filename):
        self.texts=texts
        self.nkw=nkw
        self.filename=filename
        self.wrds=[]
        self.cods=[]   
        self.wrdcod=[]        
        
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
        Y = (TSNE(n_components=2, random_state=0, perplexity=15, init="pca")
            .fit_transform(vectors_words))
        # Sets everything up to plot
        df = pd.DataFrame({"x": [x for x in Y[:, 0]],
                    "y": [y for y in Y[:, 1]],
                    "words": word_labels,
                    "color": color_list})
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
        
        add_wrds=model.wv.most_similar(positive=word)
        for mes in add_wrds:
            st.info(mes)      
                
        if len(add_wrds)>3: add_wrds=add_wrds[0:3] 
                
        for i in range(len(add_wrds)):
            self.wrdcod.append(add_wrds[i])
            self.wrds.append(add_wrds[i][0])
            self.cods.append(add_wrds[i][1])
                                    
    def start_word_2_vec(self):
        nkw=self.nkw
        texts=self.texts
        l=len(nkw)
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

        w2v_model = Word2Vec(
        min_count=2,
        window=10,
        vector_size=50,
        negative=10,
        alpha=0.03,
        min_alpha=0.0007,
        sample=6e-5,
        sg=1)

        w2v_model.build_vocab(texts)
        w2v_model.train(texts, total_examples=w2v_model.corpus_count, epochs=1000, report_delay=1)
        p=[]
        p=w2v_model.wv.most_similar(positive=[base_word])
        #for word in p:
        #    st.text(word)
        #    st.text(base_word)
        self.view_word2vec(w2v_model, base_word,list_words)
        
        

#*****************************************************************

class LDA(object):
    
    def __init__(self,num_topics,num_words,input_text,nm_chan):
        self.fig_lda=0
        self.buf_lda=0
        self.list_lda=[]
        self.gr_wrd=[]
        self.lda_analysis(num_topics,num_words,input_text,nm_chan)
              
    # Предварительная обработка предложений
    def lda_analysis(self,num_topics,num_words,tokens,nm_chan):
        
        # выделение предложений слов с предварительной обработкой
        #print(tokens)
    
        # Создание словаря на основе токенизированных предложений
        dict_tokens = corpora.Dictionary(tokens) 
        #print(dict_tokens)
        # Создание терм-документной матрицы
        doc_term_mat = [dict_tokens.doc2bow(token) for token in tokens]
         
        #*********************************************************************
        # Генерирование LDА-модели
        ldamodel = models.ldamodel.LdaModel(doc_term_mat, num_topics=num_topics, id2word=dict_tokens, passes=25)
        
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
            list_posts.append('\n ******************************************                Категория - '+str(item[0]))
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
                list_posts.append(str(k-1)+'/'+str(l)+' --- '+word+' --- '+str(round(float(weight) * 100,2) + 1%1)+'('+str(lst_frm[ind_word][int(item[0]+1)])+')')
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
        #mapsize=(40,60)
        #fig,ax = mplt.pyplot.subplots(figsize = mapsize)
        mplt.pyplot.title('Тематический профиль канала - '+str(nm_chan),fontsize=50, loc='left')
        dff = df.drop(columns='word')  
        dff.index=new_words
        #sns.set(font_scale=5)
        fig,ax =sns.heatmap(dff, cmap='Blues_r', linewidths= 5, annot=True, annot_kws={"size": 20}, cbar=True)
        #sns.set(font_scale=1)
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
    
    def __init__(self, mas, del_words, minf, maxf):
        #self.stemmer=stemmer 
        self.ru_stopwords = stopwords
        self.morph = morph 
        self.patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
         
        self.mas=mas
        self.del_words=del_words
        self.minf=minf
        self.maxf=maxf
                        
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
    
        st.info("2. Началось формирование гистограммы обратных частот слов в сообщениях") 
         
        my_dictionary = corpora.Dictionary(all_mes_words)
        bow_corpus =[my_dictionary.doc2bow(mes, allow_update = True) for mes in all_mes_words]
   
        #print(bow_corpus)
        #print("*************************************")
        word_weight =[]
        for doc in bow_corpus:
            for id, freq in doc:
                word_weight.append([my_dictionary[id], freq])
        #print(word_weight)
        #print("*************************************")
        tfIdf = models.TfidfModel(bow_corpus, smartirs ='ntc')

        weight_tfidf =[]
        for doc in tfIdf[bow_corpus]:
            for id, freq in doc:
                weight_tfidf.append([my_dictionary[id], np.around(freq, decimals=3)]) 

        sort_weight_tfidf=sorted(weight_tfidf,key=lambda freq: freq[1]) 

        wrd=[]
        val=[]
        new_del_words=[]
        for i in range(len(sort_weight_tfidf)):
            curval=float(sort_weight_tfidf[i][1])
            if curval>=self.minf and curval<self.maxf: 
                #print(str(i))
                #print(sort_weight_tfidf[i]) 
                wrd.append(sort_weight_tfidf[i][0])
                val.append(float(sort_weight_tfidf[i][1]))
            else:
                new_del_words.append(sort_weight_tfidf[i][0])
        #print("*************************************")

        fig, ax = mplt.pyplot.subplots(figsize =(10, 7)) 
        ax.hist(val, bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        canvas = mplt.pyplot.get_current_fig_manager().canvas
        canvas.draw()
        buf = pil.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
            
        return new_del_words, fig, buf

#**********************************************************
        
    def prepare_all(self):
        st.info("1. Началось создание корпуса слов")
        all_mes_words=[]
        all_sent_words=[]
        all_words=[]
        print("*************************************") 
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

        new_del_words, fig, buf=self.histogramm(all_mes_words)
        return all_mes_words, all_sent_words, all_words, new_del_words, fig, buf
    
    
#**********************************************************

def start_corpus(mas_data, minf, maxf):    
    #start_corpus(file, minf, maxf):   
    #df = pd.read_excel('postnews1.xlsx')
    #df.columns=['A']
    #mas_data = list(df['A'])
            
    prep = Prepare(mas_data, delw, minf, maxf)
    all_mes_words, all_sent_words, all_words, curdelw, fig, buf = prep.prepare_all()
    cur_del_words=curdelw
    corpus=all_mes_words
    
    list_posts=[]
    list_posts.append(" *****   Информация о корпусе слов     *****")
    list_posts.append("Всего преддложений = "+str(len(all_sent_words)))
    list_posts.append("Всего слов = "+str(len(all_words)))
    list_posts.append("Всего удалено слов = "+str(len(curdelw)))
    list_posts.append("Всего осталось слов = "+str(len(all_words)-len(curdelw)))
                 
    return buf, fig, list_posts, all_mes_words, all_sent_words


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
if flagLocal==True:img=pil.Image.open('F:/_Data Sience/Веб_приложения/Streamlit/demo_test_1/photo.jpg')
else: img=pil.Image.open('photo.jpg')
st.sidebar.image(img, width=250)
    
def corpus():

    text_1 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Создание корпуса слов выбранного канала</p>'
    st.markdown(text_1, unsafe_allow_html=True)
    list_chan=["https://www.kommersant.ru/RSS/news.xml", "https://lenta.ru/rss/","https://www.vesti.ru/vesti.rss"]
    filename = st.sidebar.selectbox("Выберите новостной канал",list_chan)
    
    cnt_days = st.sidebar.selectbox("Выберите количество дней от текущей даты",["1","2","3","4","5","6","7","8","9","10","20","30"],index=11)
    min_tfidf = st.sidebar.selectbox("Выберите мин. уровень обр. частоты слов",["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"],index=0)
    max_tfidf = st.sidebar.selectbox("Выберите макс. уровень обр. частоты слов",["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],index=9)
    minf=float(min_tfidf)
    maxf=float(max_tfidf)
   
    allmes=[]
    mas_date=[]
    cl_mas_data=[]
    cl_mas_date=[]
    but_corpus=st.sidebar.button("Создать корпус")
    if but_corpus:
        flagExcel=False  
        if flagExcel==True:
            cl_mas_data, cl_mas_date = read_excel()
            st.text("len_cl_mas_data="+str(len(cl_mas_data)))
            st.session_state.cl_mas_data=cl_mas_data
            st.session_state.cl_mas_date=cl_mas_date
        else:
            #posted_q = deque(maxlen=20)
            #n_test_chars = 50
            #httpx_client = httpx.AsyncClient()
            #cl_mas_data, cl_mas_date=asyncio.run(rss_parser(httpx_client, posted_q, n_test_chars, filename))
            
            #our_feeds = {'Kommersant': 'https://www.kommersant.ru/RSS/news.xml',
            #'Lenta.ru': 'https://lenta.ru/rss/',
            #'Vesti': 'https://www.vesti.ru/vesti.rss'} #пример словаря RSS-лент 
            
             
            url=filename  
            st.info("Парсинг новостной ленты "+url)
            cl_mas_data, cl_mas_date = getDescriptionsDates(url, cnt_days) 
            for i in range(0,len(cl_mas_data)):
                #st.info(str(i+1)) 
                st.text(str(i+1)+".   /"+cl_mas_date[i]+"/ "+cl_mas_data[i])                
            #st.info("************************************************")
            #st.info(cl_mas_data)
            #st.info("************************************************")
            if len(cl_mas_data)>0:
                st.session_state.cl_mas_data=cl_mas_data
                st.session_state.cl_mas_date=cl_mas_date
            else: 
                return
                
                    
        #for mes in cl_mas_data:
        #    st.text(mes)
            
        buf, fig, listp, allmes, sent_words =start_corpus(cl_mas_data, minf, maxf)
        #fig, listp, allmes =start_corpus(filename, minf, maxf)
        st.session_state.sent_words=sent_words
        
        #st.text(""+str(len(allmes)))
        if len(allmes)>0:
            st.info("3. Корпус создан. Вывод гистограммы")
            st.image(buf,60)
            for curmes in listp:
                st.info(curmes)
        else:
            st.error("Ошибка! Корпус не создан")
        
    st.session_state.file_name=filename
    st.session_state.all_mes_words = allmes      
         
def profil():  
    
    text_1 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Создание тематического профиля группы/слова для выбранного канала</p>'
    st.markdown(text_1, unsafe_allow_html=True)
    
    filename=st.session_state.file_name
    allmes=st.session_state.all_mes_words
    if len(allmes)==0:
        st.error("Корпус не создан!")
        return
    
    sel_cntgroup = st.sidebar.selectbox("Выберите количество тематических групп",["1","2","3","4","5","6","7","8","9","10"],index=9)
    sel_cntwords = st.sidebar.selectbox("Выберите количество слов в группе",["1","2","3","4","5","6","7","8","9","10"],index=9)
    sel_cntgroup=int(sel_cntgroup)
    sel_cntwords=int(sel_cntwords)
        
    but_lda=st.sidebar.button("Создать профиль")
    if but_lda:             
        st.info("1. Начался анализ слов методом латентного размещения Дирихле(LDA)")
        st.warning("Подождите ...")
        lda=LDA(sel_cntgroup,sel_cntwords,allmes,filename) 
        st.info("2. Вывод тепловой карты (более светлый цвет - более частое использование слова)")
        st.image(lda.buf_lda,20)
        st.session_state.lda_group_words = lda.gr_wrd
        for mes in lda.list_lda:
            st.info(mes)
        #st.write(st.session_state.lda_group_words)
    
def search():

    text_2 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Отбор и анализ сообщений по ключевым словам выбранной группы</p>'
    st.markdown(text_2, unsafe_allow_html=True)    
    
    filename=st.session_state.file_name
    gr_wrd=st.session_state.lda_group_words
    all_mes=st.session_state.all_mes_words
    sent_words=st.session_state.sent_words
    cl_data=st.session_state.cl_mas_data
    cl_date=st.session_state.cl_mas_date
    
    #for i in range(len(cl_data)):
    #    st.info(cl_data[i])
    #    st.info(all_mes[i])
    
    if len(gr_wrd)==0: 
        st.error("Ошибка! Тематический профиль не создан.")
        return
    
    #for curmes in lda.list_lda:
    #    st.text(curmes)
        
    sel_findgroup = st.sidebar.selectbox("Выберите группу для поиска",["0","1","2","3","4","5","6","7","8","9"],index=0)
    if sel_findgroup:
        new_gr_words=[]
        old_gr_words=gr_wrd[int(sel_findgroup)]
        for curw in old_gr_words:
            new_gr_words.append(curw)
        sel_findwords = st.sidebar.multiselect("Выберите слова для поиска в порядке их важности (для анализа связанности - выберите не менее трех слов)",(new_gr_words))
        if sel_findwords:
            but_find=st.sidebar.button("Начать поиск сообщений")  
            if but_find:
                progress_bar = st.progress(0)             
                srch_mes=[]
                cntmes=len(all_mes)
                if cntmes>=100: delta=(cntmes//10)
                else: delta=100//cntmes
                curdelta=0
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
                wrd_cods=[]
                if len(sel_findwords)>=3:
                    w2vec=word2vec(sel_data, sel_findwords, filename)
                    w2vec.start_word_2_vec()
                    for wcod in w2vec.wrdcod:
                        wrd_cods.append(wcod)
                                       
                    if len(w2vec.wrds)>0:
                        dbeg=""
                        dend=""
                        k=0
                        srch_mes_new=[]
                        for i in range(len(all_mes)):
                            keywrd=list(set(all_mes[i])&set(w2vec.wrds)) 
                            if len(keywrd)>0:
                                text_tmp=" ("+str(i)+")  *** "+str(cl_date[i])+" - ("+", ".join(keywrd)+" ) ***** "
                                srch_mes_new.append(text_tmp+"                  "+cl_data[i])
                                if k==1: dbeg=str(cl_date[i])
                                dend=str(cl_date[i])
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
                #******************************************************************    
                text_1 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Список слов (до трех), наиболее контекстно связанных с базовым ключевым словом</p>'
                text_1 =text_1+'<p style="font-family:sans-serif; color:Red; font-size: 24px;">'+sel_findwords[0]+'</p>'
                st.markdown(text_1, unsafe_allow_html=True)
                for wcod in wrd_cods:
                    st.info(wcod)
                                        
                text_2 = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">Дополнительно отобранные сообщения, контекстно связанные с заданными ключевыми словами </p>'
                st.markdown(text_2, unsafe_allow_html=True)
                k=1
                for mes in srch_mes_new:
                    st.info("("+str(k)+") "+mes)
                    k+=1
                    
                if len(srch_mes)==0:
                    st.error("Сообщения не найдены") 

def myhelp():
    st.text("HELP")  

app = MultiApp()
app.add_app("Создание корпуса", corpus)
app.add_app("Анализ глобального профиля", profil)
app.add_app("Анализ локального профиля", search)
app.add_app("Инструкция", myhelp)
app.run()


