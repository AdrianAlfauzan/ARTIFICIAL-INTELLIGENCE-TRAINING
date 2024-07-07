import nltk
from nltk.chat.util import Chat, reflections
# definisi aturan
pairs = [
    [r"(hi|halo)",["Halo Mahasiswa unjani, apa kabar?"],],
    [r"namaku (.*)",["Salam kenal %1, bagaimana kabarmu hari ini?"],],
    [r"kabarku (.*)",["Senang mendengarnya. Apa yang kamu lakukan hari ini?"]],
    [r"saya (.*)",["Semangat ya!","Tetaplah seperti itu, lanjutkan!"],],
    [r"quit",["Sampai jumpa! Senang mengobrol denganmu ..."],]
  ]
print("Halo, saya adalah Bot dan saya senang mengobrol. Gunakan bahasa Indonesia yang baku tanpa singkatan. Ketik 'quit' untuk mengakhiri obrolan!")

chat = Chat(pairs, reflections)
chat.converse()