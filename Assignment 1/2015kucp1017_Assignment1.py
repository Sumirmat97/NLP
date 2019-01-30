import nltk, re
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer as tfv

def standardize_text(data):
    data = [text.lower() for text in data]
    data = [re.sub(r'[^\w\s]','',text) for text in data]
    data = [re.sub(r'[\n]',' ',text) for text in data]
    data = [nltk.word_tokenize(text) for text in data]
    stops = set(nltk.corpus.stopwords.words("english"))
    for pos,text in enumerate(data):
        text = [w for w in text if not w in stops]
        data[pos] = " ".join(text)
    return data

print("Enter the name of directory of train files: ")
dir = str(input())

corpus = sorted(glob(dir + "/*.txt"))

data = [open(file, encoding="utf8").read() for file in corpus]
data = standardize_text(data)
tfv = tfv(analyzer='word', token_pattern=r'\w+')
tf = tfv.fit_transform(data)

multipliedMatrix = (tf*tf.T).A
for x in range(0, len(corpus)):
    for y in range(x+1, len(corpus)):
        if multipliedMatrix[x,y] > 0.0:
            print("File " + corpus[x] + " and File " + corpus[y] + " is plagiarised by: " + str(round(multipliedMatrix[x,y]*100,2)) + "%")