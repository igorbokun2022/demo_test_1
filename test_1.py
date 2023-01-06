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
import asyncio
import datetime
from telethon import TelegramClient

from multiapp import MultiApp

cl_mas_data=[]
cl_mas_date=[]

filename=""
minf=0.1
maxf=1.0
delw=[]
cur_del_words=[]
corpus=[]
all_mes_words=[]

api_id = 16387030
api_hash = '07bfab67941aa8ebe50f836e3b5c5704'
ses_name='telemesmonitor'
phone='+998909790855'
code='22561'
# получен  запросом - await client.start(phone=phone, code_callback=code_callback)
max_posts=1000
cnt_mes=500

stemmer=nltk.stem.SnowballStemmer(language="russian")
stopwords = stopwords.words('russian') 
morph = MorphAnalyzer() 

#*****************************************************************
async def work():
    
    mas_mes=[] 
    mas_mes_date=[]     
    loop=asyncio.new_event_loop()
   
    #*************************************
    #df = pd.read_excel('F:/_Data Sience/Веб_приложения/Streamlit/demo_test_1/postnews1.xlsx')
    df = pd.read_excel('postnews1.xlsx')
    df.columns=['A']
    cl_mas_data = list(df['A'])
    st.text("принято сообщений канала - "+str(len(cl_mas_data)))
    #*************************************
        
    return cl_mas_data
        
def code_callback():
   while True:
       #ждем код телеграмме, а потом подставляем его в эту функцию 
       #code='18562'
       return code
     

#*****************************************************************

class LDA(object):
    
    def __init__(self,num_topics,num_words,input_text,nm_chan):
        self.fig_lda=0
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
        
        list_posts.append("*****************************************************************")
        list_posts.append("Классификация текста канала - "+str(nm_chan) +" по "+str(num_topics)+" категориям")
        list_posts.append(str(num_words) + ' наиболее значимых слов для каждой категории:')
        
        self.gr_wrd=[]
                
        for item in ldamodel.print_topics(num_topics=num_topics, num_words=num_words):
            #st.text('\n Категория - '+str(item[0]))
            list_posts.append('\n Категория - '+str(item[0]))
            list_posts.append('**********************************')        
            # Вывод представительных слов вместе с их
            # относительными вкладами
            list_of_strings = item[1].split(' + ')
            
            cur_wrd=[]              
            for text in list_of_strings:
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
                list_posts.append(word+' ==> '+str(round(float(weight) * 100,2) + 1%1)+'('+str(lst_frm[ind_word][int(item[0]+1)])+')')
                cur_wrd.append(word)
                
            self.gr_wrd.append(cur_wrd[1:len(cur_wrd)-1])
        
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
        #***********************************
        color_sq =  ['#eeeeeeFF','#bcbddcF0','#9e9ac8F0','#807dbaF0','#6a51a3F0','#54278fF0']
        cnt_color=6
        delta_color=maxval//cnt_color
            
        dw=0.06
        dh=0.08
        mapsize=(60,80) 
        
        fig,ax = mplt.pyplot.subplots(figsize = mapsize)
        mplt.pyplot.title('Семантический профиль канала - '+str(nm_chan)+ '  на основе последних сообщений',fontsize=68, loc='left')
        
        for j in range(num_topics):
            mplt.pyplot.text(0.26+dw*j, 1.1, 'гр-'+str(j), fontsize=48, color='navy')
    
        mplt.pyplot.axhline(y=0.9+dh, xmin=0, xmax=1.0, color='black')
        for i in range(len(new_words)):
            mplt.pyplot.text(0, 0.91-dh*i, new_words[i], fontsize=60, color='black')
            mplt.pyplot.axhline(y=0.9-dh*i, xmin=0, xmax=1.0, color='black')
            for j in range(num_topics+1):
                if j>0:
                    lst_frm[i][j]=lst_frm[i][j]//delta_color
                    if lst_frm[i][j]>cnt_color-1: lst_frm[i][j]=cnt_color-1
                    xy=(0.19+dw*j, 0.9-dh*i)
                    col=color_sq[lst_frm[i][j]]
                
                    mplt.pyplot.gca().add_patch(mplt.patches.Rectangle(xy, dw+0.02, dh,
                        edgecolor='black',
                        facecolor=col,
                        lw=4))
                                
            #plt.text(x0, y0+dh*j, new_words[i], fontsize=14, color='navy')
               
        dff=pd.DataFrame(lst_frm)
        dff.columns=cols
        
        #***********************************
        self.fig_lda=fig
        self.list_lda=list_posts.copy() 
        
        return 


class Prepare(object):    
    
    def __init__(self, mas, del_words, minf, maxf):
        self.stemmer=stemmer 
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
        new_word=stemmer.stem(new_word) 
        
        if new_word not in self.ru_stopwords and new_word not in self.del_words:  
            if len(new_word)>3:
                if 'NOUN' in morph.tag(new_word)[0]:
                    #print("("+old_word+") = "+new_word)
                    #print("*****************")             
                    return new_word            
        return " "     
    
#**********************************************************    

    def histogramm(self, all_mes_words):
    
        st.text("2. Началось сформирование гистограммы обратных частот слов в сообщениях") 
         
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
            
        return new_del_words, fig

#**********************************************************
        
    def prepare_all(self):
        st.text("1. Началось создание корпуса слов")
        all_mes_words=[]
        all_sent_words=[]
        all_words=[]
        print("*************************************") 
        for line in self.mas:  
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

        new_del_words, fig=self.histogramm(all_mes_words)
        return all_mes_words, all_sent_words, all_words, new_del_words, fig
    
    
#**********************************************************

def start_corpus(mas_data, minf, maxf):    
    #start_corpus(file, minf, maxf):   
    #df = pd.read_excel('postnews1.xlsx')
    #df.columns=['A']
    #mas_data = list(df['A'])
            
    prep = Prepare(mas_data, delw, minf, maxf)
    all_mes_words, all_sent_words, all_words, curdelw, fig = prep.prepare_all()
    cur_del_words=curdelw
    corpus=all_mes_words
    
    list_posts=[]
    list_posts.append("*********************************************************")
    list_posts.append("Информация о корпусе слов")
    list_posts.append("Всего сообщений = "+str(len(all_mes_words)))
    list_posts.append("Всего преддложений = "+str(len(all_sent_words)))
    list_posts.append("Всего слов = "+str(len(all_words)))
    list_posts.append("Всего удалено слов = "+str(len(curdelw)))
    list_posts.append("Всего осталось слов = "+str(len(all_words)-len(curdelw)))
    list_posts.append("*********************************************************")
     
         
    return fig, list_posts, all_mes_words


#**************************************************************

if 'lda_group_words' not in st.session_state:
    st.session_state.lda_group_words = []
if 'all_mes_words' not in st.session_state:
    st.session_state.all_mes_words = []
if 'cl_mas_data' not in st.session_state:
    st.session_state.cl_mas_data = []

st.header('web-сервис: тематичеcкий анализ контента телеграм-каналов')
st.text("(перейдите в режим широкого экрана - три черточни в правом углу, Settings, WideMode)")
#img=pil.Image.open('F:/_Data Sience/Веб_приложения/Streamlit/demo_test_1/photo.jpg')
img=pil.Image.open('photo.jpg')
st.sidebar.image(img)

def profil():

    st.text("Создание корпуса слов и тематического профиля выбранного канала")
    filename = st.sidebar.selectbox("Выберите телеграм-канал",["t.me.rian_ru","@kunuzru","@gazetauz"])

    min_tfidf = st.sidebar.selectbox("Выберите мин. уровень обр. частоты слов",["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"])
    max_tfidf = st.sidebar.selectbox("Выберите макс. уровень обр. частоты слов",["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],index=9)
    minf=float(min_tfidf)
    maxf=float(max_tfidf)
    allmes=[]

    sel_cntgroup = st.sidebar.selectbox("Выберите количество тематических групп",["1","2","3","4","5","6","7","8","9","10"],index=9)
    sel_cntwords = st.sidebar.selectbox("Выберите количество слов в группе",["1","2","3","4","5","6","7","8","9","10"],index=9)
    sel_cntgroup=int(sel_cntgroup)
    sel_cntwords=int(sel_cntwords)

    but_lda=st.sidebar.button("Создать тематический профиль")
    if but_lda: 
        mas_date=[]
    
        try:
            cl_mas_data = asyncio.run(work())
            st.session_state.cl_mas_data=cl_mas_data
            st.text("принято сообщений канала) - "+str(len(cl_mas_data)))     
        except: 
            st.text("ошибка чтения канала!")
        
        fig, listp, allmes =start_corpus(cl_mas_data, minf, maxf)
        #fig, listp, allmes =start_corpus(filename, minf, maxf)
        
        if len(allmes)>0:
            st.text("3. Корпус создан. Вывод гистограммы")
            st.pyplot(fig)
            for curmes in listp:
                st.text(curmes)
            st.text("1. Начался анализ слов методом латентного размещения Дирихле(LDA)")
            lda=LDA(sel_cntgroup,sel_cntwords,allmes,filename) 
            st.text("2. Вывод тепловой карты (более темный цвет - более частое использование слова)")
            st.pyplot(lda.fig_lda) 
            
            st.session_state.lda_group_words = lda.gr_wrd
            st.write(st.session_state.lda_group_words)
            st.session_state.all_mes_words = allmes
            st.write(st.session_state.all_mes_words)
        else:
            st.text("Ошибка! Корпус не создан")

def search():

    st.text("Найти сообщения по ключевым словам выбранной группы") 
    st.text("*************************************")
    gr_wrd=st.session_state.lda_group_words
    all_mes=st.session_state.all_mes_words
    cl_data=st.session_state.cl_mas_data
           
    if len(gr_wrd)==0: 
        st.text("Ошибка! Тематический профиль не создан.")
        return
    
    #for curmes in lda.list_lda:
    #    st.text(curmes)
        
    sel_findgroup = st.sidebar.selectbox("Выберите группу для поиска",["0","1","2","3","4","5","6","7","8","9"],index=0)
    if sel_findgroup:
        progress_bar = st.sidebar.progress(0)
        new_gr_words=[]
        old_gr_words=gr_wrd[int(sel_findgroup)]
        for curw in old_gr_words:
            new_gr_words.append(curw[1:len(curw)-1])
        sel_findwords = st.sidebar.multiselect("Выберите слова для поиска",(new_gr_words))
        if sel_findwords:
            but_find=st.sidebar.button("Начать поиск сообщений")  
            if but_find:
                srch_mes=[]
                cntmes=len(all_mes)
                k=(cntmes//100)+1
                for i in range(cntmes):
                    progress_bar.progress(i//k)
                    if len(list(set(all_mes[i])&set(sel_findwords)))>0:
                        #st.text(all_mes[i])
                        #st.text("-----")
                        #st.text(cl_data[i])
                        #st.text("*************************************")
                        srch_mes.append("("+str(i)+")  *** "+cl_data[i])
                             
                for mes in srch_mes:
                    st.text(mes)
                    st.text("*************************************")
                if len(srch_mes)==0:
                    st.text("Сообщения не найдены")                
app = MultiApp()
app.add_app("Профиль", profil)
app.add_app("Поиск", search)
app.run()


