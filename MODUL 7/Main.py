from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Membuat data customer dalam bentuk dataframe
data = pd.DataFrame({
    'Usia' : [30,20,35,25,40,30,45,35,50,40],
    'Pendapatan' : [200,1500,2500,1800,300,2200,3500,2800,400,3200],
    'Jumlah Pembelian':[5,3,7,4,9,6,11,8,13,10],
    'Jenis Kelamin': ['Laki - Laki', 'Perempuan', 'Laki - Laki', 'Perempuan',
                      'Laki - Laki', 'Perempuan', 'Laki - Laki', 'Perempuan',
                      'Laki - Laki', 'Perempuan'],
    'Status Pernikahan' : ['Belum Menikah','Belum Menikah','Menikah',
                           'Belum Menikah','Menikah','Belum Menikah',
                           'Menikah','Menikah','Menikah','Menikah']
})

# Memilih kolom yang akan dinormalisasi
kolom = ['Usia','Pendapatan','Jumlah Pembelian']

# Menggunakan metode MinMaxScaler untuk mernomalisasi data
scaler = MinMaxScaler()
data[kolom] = scaler.fit_transform(data[kolom])

# Mengubah nilai 'Laki - Laki' menjadi 1 dan 'Perempuan' menjadi 0 pada atribut jenis kelamin
data['Jenis Kelamin'] = data['Jenis Kelamin'].replace({'Laki - Laki' : 1, 'Perempuan': 0})

# Mengubah nilai 'Menikah' menjadi 1 dan 'Belum menikah' menjadi 0 pada atribut Status Pernikahan
data['Status Pernikahan'] = data['Status Pernikahan'].replace({'Menikah' : 1, 'Belum Menikah': 0})

# Menampilkan hasil normalisasi data
print(data)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Memilih kolom yang akan dijadikan fitur untuk pengelompokan
fitur = ['Usia','Pendapatan','Jumlah Pembelian']
X = data[fitur]

# Melakukan analisis elbow untuk menentukan nilai k terbaik
wcss = []
for i in range(1,7):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Menampilkan visualisasi elbow
plt.plot(range(1,7), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('wcss')
plt.show()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Melakukan pengelompokkan data customer menggunakan algoritma k - means clustering
jumlah_cluser = 2
kmeans = KMeans(n_clusters=jumlah_cluser, init='k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Melakukan reduksi dimensi menggunakan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Menampilkan visualisasi clustering dalam scatter plot
for i in range(jumlah_cluser):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1],
                s = 100, c = np.random.rand(3,), label = 'Cluster {}'.format(i))
    
plt.title('Customer Segmentation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
