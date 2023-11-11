# streamlit run app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from catboost import CatBoostClassifier, Pool
import spacy

import re

st.set_page_config(
    page_title="Модель для генерации продающих текстов",
    page_icon="📝", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None,'Report a bug': None,'About': None})

hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

############################################################################
# функции
############################################################################
lemmatizer = spacy.load('ru_core_news_md', disable = ['parser', 'ner'])
stopwords_nltk=[]

def full_clean(s):
    #подготовка текста к подаче в модель
    s=re.sub(r"[^a-zA-Zа-яА-ЯйЙ#]", " ", s)
    s = s.lower()
    s = re.sub(" +", " ", s) #оставляем только 1 пробел
    text = " ".join([token.lemma_ for token in lemmatizer(s) if token.lemma_ not in stopwords_nltk])
    
    return text

def tfidf_featuring(tfidf, df):   
    '''Преобразование текста в мешок слов'''
    X_tfidf = tfidf.transform(df)
    feature_names = tfidf.get_feature_names_out()
    X_tfidf = pd.DataFrame(X_tfidf.toarray(), columns = feature_names, index = df.index)
    
    return X_tfidf

def get_params(data, start, emotional):
    # номер диaлога и тональность
    nums=[]
    num = start
    for i, row in data.iterrows():
        if row["№ сообщения"]==1:
            num+=1
        nums.append(num)
    data['№ диалога']=nums
    data['emotional']=emotional
    return data

def create_texts(data):
    # тексты пользователя для датасета
    texts=[]
    string = ''
    for i, row in data.iterrows():
        if row["№ сообщения"]==1:
            if string != '':
                texts.append(string)
                string = str(row["Текст"])+' '
            else:
                string += str(row["Текст"])+' '
        else:
            if row["Направление"]=='in':
                string += str(row["Текст"])+' '
    texts.append(string)
    return texts

@st.cache_data
def load_file(falepath):
    '''Загрузка файла'''
    df = pd.read_excel(falepath, engine='openpyxl') #sheet_name='valid', 
    df = get_params(df, 0, "unknow")
    data = df[['№ диалога', 'emotional']].drop_duplicates()
    data['texts']=create_texts(df)
    data['отказ от продуктов']=0
    data['жалобы']=0
    data['просроченная задолженность']=0
    data['мошенничество']=0
    data['утеря/кража карты']=0
    data['clean'] = data['texts'].apply(lambda x: full_clean(x))
    return df, data

@st.cache_data
def load_tfidf(falepath):
    '''Загрузка векторизатора'''
    tfidf = TfidfVectorizer()
    tfidf = joblib.load(falepath)
    return tfidf

@st.cache_data
def load_model1(falepath):
    '''Загрузка модели тональности'''
    model_1 = CatBoostClassifier()
    model_1.load_model(falepath)
    return model_1 

@st.cache_data
def load_model1(falepath):
    '''Загрузка модели стоп тем'''
    model_2 = CatBoostClassifier()
    model_2.load_model(falepath)
    return model_2

############################################################################
# загрузка
############################################################################

stop_theme = ['отказ от продуктов', 'жалобы', 'просроченная задолженность', 'мошенничество', 'утеря/кража карты']

#векторизатор
tfidf = joblib.load('model/tfidf.pkl')

#тональность
model_1 = load_model1('model/model1.cbm')

#стоп темы
model_2 = load_model1('model/model2.cbm')


#df, data=load_file('data/Газпром_valid.xlsx')

############################################################################
# вывод результатов
############################################################################

# Заголовок страницы
st.title("Результаты анализа")

# Создание кнопки загрузки файла
upload_button = st.file_uploader("Выберите файл для загрузки", type=["xlsx", "csv"])

if upload_button is not None:
    # Загрузка выбранного файла, если кнопка нажата
    df, data = load_file(upload_button)
else:
    # Загрузка файла по умолчанию, если кнопка не нажата
    df, data = load_file('data/Газпром_valid.xlsx')

X_tfidf = tfidf_featuring(tfidf, data['clean'])

valid_predict1 = model_1.predict(X_tfidf)
valid_predict2 = model_2.predict(X_tfidf)

t = pd.DataFrame(valid_predict2, columns=stop_theme, index = list(data['№ диалога']))
t['Стоп темы']=t.astype(int).dot(t.columns+',').str[:-1]
t['№ диалога']=t.index
t['Тональность']=valid_predict1
t['Тональность']=t['Тональность'].map({1:'positive', 0: "neutral", -1:'negative'})

# Создаем ползунок для выбора индекса диалога
selected_dialog_index = st.slider("Выберите номер диалога", 0, len(df['№ диалога'].unique()) - 1)

# Получаем номер диалога по индексу
selected_dialog = df['№ диалога'].unique()[selected_dialog_index]

# Фильтруем данные для выбранного диалога
dialog_data = df[df['№ диалога'] == selected_dialog]

# Улучшаем стиль вывода информации о выбранном диалоге
st.subheader("Диалог")
st.table(df[df['№ диалога'] == selected_dialog][['№ сообщения', 'Текст', 'Направление']].set_index("№ сообщения"))

# Используем выбранный диалог для фильтрации прогнозов
selected_predictions = t[t['№ диалога'] == selected_dialog]

# Улучшаем стиль вывода информации о прогнозе
st.subheader("Прогноз")

# Отдельно выводим тональность
tonality = selected_predictions['Тональность'].iloc[0]
st.write(f"**Тональность:** {tonality}")

# Отдельно выводим стоп темы и их значения
stop_themes = selected_predictions['Стоп темы'].iloc[0].split(',')
stop_themes_str = ', '.join(stop_themes)

st.write(f"**Стоп темы:** {stop_themes_str}")