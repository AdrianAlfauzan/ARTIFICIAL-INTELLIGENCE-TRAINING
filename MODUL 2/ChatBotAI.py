import nltk
from nltk.chat.util import Chat, reflections
import os
os.system("clear")
os.system("cls")

# definisi aturan
pairs = [
    [r"(hi|halo)",["Halo Mahasiswa unjani, siapa nama kamu?"],],

    [r"nama saya (.*)",["Halo %1! apakah ada yang ingin di pertanyakan tentang informatika?"],],

    [r"yes|oke|boleh|baiklah|siap|iya",["Baiklah silahkan."]],
    # Apa itu algoritma?
    [r"(apa itu algoritma|algoritma|algoritma adalah)",["Algoritma adalah serangkaian langkah-langkah terurut yang dirancang untuk menyelesaikan masalah atau melakukan tugas tertentu."]],

    # Apa perbedaan antara sistem operasi Windows dan Linux?
    [r"(apa itu windows|windows|windows adalah)",["Windows adalah sistem operasi yang dikembangkan oleh Microsoft, sementara Linux adalah sistem operasi open-source yang memiliki banyak distribusi seperti Ubuntu, Fedora, dan CentOS. Windows lebih umum digunakan di desktop, sementara Linux lebih umum digunakan di server dan sistem embedded."]],
    
    # # Apa itu bahasa pemrograman Python?
    [r"(apa itu python|python|python adalah)",["Python adalah bahasa pemrograman tingkat tinggi yang mudah dipelajari dan digunakan. Ia dikenal dengan sintaks yang sederhana dan ekstensibilitasnya yang kuat, dan sering digunakan dalam pengembangan web, analisis data, kecerdasan buatan, dan banyak aplikasi lainnya."]],

    # # Apa itu database?
    [r"(apa itu database|database|database adalah)",["Database adalah kumpulan data yang terorganisir secara sistematis sehingga dapat diakses, dikelola, dan diperbarui dengan mudah. Ini adalah komponen penting dalam sistem informasi dan digunakan untuk menyimpan informasi yang terstruktur."]],


    # # Apa yang dimaksud dengan kecerdasan buatan (AI)?
    [r"(apa itu AI|apa itu ai|AI|kecerdasan buatan|AI adalah|ai adalah)",["Kecerdasan buatan adalah bidang dalam ilmu komputer yang bertujuan untuk menciptakan mesin yang dapat berperilaku seperti manusia, termasuk kemampuan untuk belajar, memecahkan masalah, dan mengambil keputusan."]],

    # # Apa yang dimaksud dengan jaringan komputer?
    [r"(apa itu jaringan komputer|jaringan komputer|jaringan komputer adalah)",["Jaringan komputer adalah kumpulan komputer dan perangkat lain yang terhubung satu sama lain untuk berbagi sumber daya seperti data, printer, dan koneksi internet. Ini memungkinkan komunikasi dan kolaborasi antara pengguna dan sistem yang berbeda."]],

    [r"quit",["Sampai jumpa! Senang mengobrol denganmu ..."],]
  ]
print("Halo, saya adalah Bot dan saya senang mengobrol. Gunakan bahasa Indonesia yang baku tanpa singkatan. Ketik 'quit' untuk mengakhiri obrolan!")

chat = Chat(pairs, reflections)
chat.converse()