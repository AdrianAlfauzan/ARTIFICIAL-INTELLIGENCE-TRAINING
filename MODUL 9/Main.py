import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pandas as pd

# Download tokenizer from NLTK
nltk.download('punkt')

# Load dataset
DF_latih = pd.read_csv('D:/PRAKTIKUM KECERDASAN BUATAN/MODUL 9/dataset/Data_latih.csv')
DF_uji = pd.read_csv('D:/PRAKTIKUM KECERDASAN BUATAN/MODUL 9/dataset/Data_uji.csv')

# Display first few rows of datasets
print("DATA LATIHAN:")
print(DF_latih.head())
print("\nDATA UJI:")
print(DF_uji.head())

# Check the distribution of labels
print("\nJUMLAH LABEL:")
print(DF_latih['label'].value_counts())

# PREPROCESS DATA
x = DF_latih['judul']
y = DF_latih['label']

# Lowercase
x = [entry.lower() for entry in x]

# Tokenization
x = [word_tokenize(entry) for entry in x]

# Convert tokens to strings
x = [' '.join(entry) for entry in x]

# Split data
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(x, y, test_size=0.3)

# Convert label class
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# Vectorize using TF-IDF
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(x)
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Display sorted vocabulary
sorted_vocab = dict(sorted(Tfidf_vect.vocabulary_.items(), key=lambda item: item[1]))
print("\nVOCABULARY:")
print(sorted_vocab)

# MODELING
# Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("\nNaive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)

# SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)

# TESTING SENTENCES
def kalimat_to_tfidf(kalimat):
    kalimat = [word_tokenize(entry) for entry in kalimat]
    kalimat = [' '.join(entry) for entry in kalimat]
    kalimat_tf_idf = Tfidf_vect.transform(kalimat)
    return kalimat_tf_idf

kalimat = ["Narasi Budi Berideologi Komunis"]
kalimat_to_tfidf = kalimat_to_tfidf(kalimat)

kelas_prediksi_nb = Naive.predict(kalimat_to_tfidf)
kelas_prediksi_svm = SVM.predict(kalimat_to_tfidf)

print("\nPrediksi Naive Bayes:", kelas_prediksi_nb)
print("Prediksi SVM:", kelas_prediksi_svm)
