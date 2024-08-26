import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

df = pd.read_csv("C:/ML/python/data/spacenews.csv",delimiter=',')

df.info()
df.duplicated().sum()
df.isna().sum()
df['content'] = df['content'].fillna("")

df['postexcerpt'] = df['postexcerpt'].fillna("")

import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    
    text = str(text).lower()
    
    text = re.sub('<.*?>+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub("'", '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\d', '', text)
    text = re.sub(' +', ' ', text)


    return text
    
df['postexcerpt'] = df['postexcerpt'].apply(clean_text)



sw = stopwords.words("english")

from nltk.tokenize import word_tokenize


def remove_stopwords(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.lower() not in sw]
    return " ".join(cleaned_tokens)


df['postexcerpt'] = df['postexcerpt'].apply(remove_stopwords)

lemma = WordNetLemmatizer()


def lemmatizer(text):
    tokens = word_tokenize(text)
    lemma_tokens = [lemma.lemmatize(token) for token in tokens]
    return " ".join(lemma_tokens)

df['postexcerpt'] = df['postexcerpt'].apply(lemmatizer)

text = " ".join(i for i in df['postexcerpt'])


from wordcloud import WordCloud
from textblob import TextBlob

import matplotlib.pyplot as plt
import seaborn as sns

wc = WordCloud(colormap="Set3",collocations=False).generate(text)
plt.imshow(wc,interpolation="blackman")
plt.show()


blob = TextBlob(text)


from nltk.probability import freq_dist

most_frequent_words = FreqDist(blob.words)
top_50_most_used = most_frequent_words.most_common(50)
print("top 50 most used words in space articles: ",top_50_most_used)


def polarity(text):
    return TextBlob(text).polarity



df['polarity'] = df['postexcerpt'].apply(polarity)



def sentiment(label):
    if label <0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label >= 0:
        return "Positive"


df['sentiment'] = df['polarity'].apply(sentiment)

df['sentiment'].value_counts().plot(kind='pie',autopct='%1.1f%%')

fig,axs = plt.subplots(figure=(10,6))
df['sentiment'].value_counts().plot(kind='bar',rot=0)
plt.show()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X = df['postexcerpt']
y = df['sentiment']
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.15,random_state=1)


from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print(len(word_index))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


max_length = 0
for sequence in X_train:
    sequence_length = len(sequence)
    if sequence_length > max_length:
        max_length = sequence_length

print("Max Length of Sequences: ",max_length)


### padding

from tensorflow.keras.utils import pad_sequences,to_categorical


X_train = pad_sequences(X_train,max_length,padding="post")
X_test = pad_sequences(X_test,max_length,padding="post")

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,SpatialDropout1D,LSTM,Embedding

RNN = Sequential()
RNN.add(Embedding(len(word_index)+1,output_dim=200,input_length=max_length))
RNN.add(SpatialDropout1D(0.5))
RNN.add(Bidirectional(LSTM(150,dropout=0.1,recurrent_dropout=0.1)))
RNN.add(Dropout(0.3))
RNN.add(Dropout(0.1))
RNN.add(Dense(3,activation='sigmoid'))
RNN.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])
history = RNN.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=(0.1))



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history["val_loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()




