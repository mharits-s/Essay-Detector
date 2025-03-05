# Flow of Code

## Data Collection

Pengumpulan data terdiri dari 3 data yaitu:
1. Esai Siswa (1 topik)
2. Esai ChatGPT (1 topik)
3. Pengetahuan ChatGPT (Multi topik)

Panjang data terbagi menjadi 3 yaitu:
1. Data pendek, 200 - 300 karakter
2. Data sedang, 301 - 600 karakter
3. Data panjang, 601 - 1200 karakter

### Esai Siswa

Esai siswa diperoleh melalui tugas yang diberikan oleh guru secara daring menggunakan Google Form. Setiap siswa diminta untuk menjawab 3 pertanyaan esai sesuai dengan batas maksimal karakter yang telah ditentukan sebelumnya. Topik esai disesuaikan dengan arahan dari guru, dan siswa diharapkan mengerjakannya secara mandiri. Waktu pengerjaan tugas adalah selama satu minggu, dengan minggu pertama untuk pemberian tugas dan minggu kedua sebagai batas akhir pengumpulan.

### Esai ChatGPT

Esai yang dihasilkan oleh ChatGPT diperoleh melalui pemrograman Python dengan menggunakan API OpenAI. Jumlah sesi disesuaikan dengan jumlah siswa yang telah mengumpulkan tugas. Setiap sesi terdiri dari 3 prompt yang dijawab secara berurutan. Prompt yang digunakan sama dengan soal yang diberikan melalui Google Form, dengan tambahan batasan maksimal karakter yang ditentukan untuk setiap jawaban.


### Pengetahuan ChatGPT

Pengetahuan yang dihasilkan oleh ChatGPT diperoleh dengan cara yang sama, yaitu menggunakan Python dan API OpenAI. Setiap subbab dalam suatu bab akan memiliki tiga jenis respons dengan panjang dan gaya penyampaian yang berbeda: teks pendek berbentuk reflektif, teks sedang berbentuk argumentatif, dan teks panjang berbentuk ekspositori. Jumlah subbab dan bab akan disesuaikan dengan Rencana Pembelajaran Semester (RPS) selama dua semester pembelajaran.

### Dataset

**1. Dataset Esai Siswa & Esai ChatGPT (Student_ChatGPT):**

- **Format File:** Excel (.xlsx) atau CSV (.csv).
- **Kolom Data:** 
  - Human: Berisi esai yang ditulis oleh siswa.
  - AI: Berisi esai yang dihasilkan oleh ChatGPT.
- **Struktur Data:** 
  - Baris pertama adalah header yang mencantumkan label kolom: Human dan AI.
  - Baris kedua dan seterusnya berisi data esai:
  - Contoh:
      - **Row 2, Col 1 (Human):** Esai pendek yang ditulis oleh Siswa A.
      - **Row 6, Col 2 (AI):** Esai sedang yang dihasilkan oleh ChatGPT.

**2. Dataset Pengetahuan ChatGPT (Only_ChatGPT):**

- **Format File:** Excel (.xlsx) atau CSV (.csv).
- **Kolom Data:** 
  - AI: Berisi esai yang dihasilkan oleh ChatGPT berdasarkan prompting.
- **Struktur Data:** 
  - Baris pertama adalah header dengan label kolom: AI.
  - Baris kedua dan seterusnya hanya berisi teks esai yang diperoleh dari hasil prompting, tanpa kolom tambahan.
  - Contoh:
      - **Row 2 (AI):** Teks esai pendek berbentuk reflektif.
      - **Row 3 (AI):** Teks esai sedang berbentuk argumentatif.
      - **Row 4 (AI):** Teks esai panjang berbentuk ekspositori.


---

## Data Preparation & Feature Extraction

Terdapat beberapa tahapan untuk menyiapkan data sebelum pelatihan. Alur prosesnya sebagai berikut:
1. Pembagian data, Training set 70%, Validation set 20%, Test set 10%
2. Segmentati teks, mengubah 1 data yang masih berupa paragraf menjadi kalimat yang terpisah (menyesuaikan jumlah kalimat yang ada pada 1 data/paragraf tersebut.)
3. BERT Tokenizer menggunakan IndoBERT-base untuk keperluan similarity
4. Ekstraksi fitur stylometric seperti rata-rata panjang kata, rasio kata unik terhadap total jumlah kata, rasio tanda baca, panjang kalimat. Untuk keperluan klasifikasi.

---

## Semantic Similarity Model

Model Semantic Similarity ini akan digunakan pada dua dataset berbeda dan training terpisah, yaitu dataset `Student_ChatGPT` dan `Only_ChatGPT`. Model yang sudah dievaluasi akan disimpan dalam format `.h5`. Machine learning framework yang digunakan pada modelling ini yaitu TensorFlow.

Berikut Alur Training Model Semantic Similarity
1. Data training digunakan untuk pelatihan model.
2. Base model yang digunakan yaitu IndoBERT-base
3. Fine-tuning untuk kebutuhan semantic similarity yaitu menggunakan Dense Layer, Dropout, Contrastive Loss, dan Adam Optimizer.

Pada akhirnya nanti **terdapat 2 model Semantic Similarity** yang dibuat. **Model #1** untuk memahami perbadaan/kemiripan antara esai siswa dan esai ChatGPT. **Model #2** untuk memahami gaya penulisan ChatGPT.


### Training #1 dataset `Student_ChatGPT`

Training pertama untuk model semantic similarity yaitu menggunakan dataset `Student_ChatGPT` yang bertujuan mengetahui perbedaan/kemiripan antara esai siswa dan esai ChatGPT. 

**Pasangan Positif**
- Esai Siswa - Esai Siswa (within-data)
- Esai Siswa - Esai Siswa (cross-data)
- Esai ChatGPT - Esai ChatGPT (within-data)
- Esai ChatGPT - Esai ChatGPT (cross-data)

**Pasangan Negatif**
- Esai Siswa - Esai ChatGPT (cross-data)
  
### Training #2 dataset `Only_ChatGPT`

Training kedua untuk model semantic similarity yaitu menggunakan dataset `Only_ChatGPT` yang bertujuan memahami gaya penulisan ChatGPT pada pengetahuannya terhadap RPS mata pelejaran. 

**Pasangan Positif**
- Pengetahuan ChatGPT - Pengetahuan ChatGPT(within-data)
- Pengetahuan ChatGPT - Pengetahuan ChatGPT(cross-data)

---

## Classification

Model klasifikasi ini bertujuan untuk mengategorikan teks input ini diklasifikasikan sebagai buatan manusia atau AI. Fitur yang digunakan dalam proses pelatihan ini yaitu sebagai berikut:
1. Hasil Embedding dari model #1 (Dataset `Student_ChatGPT`)
2. Hasil Embedding dari model #2 (Dataset `Only_ChatGPT`)
3. Hasil Similarity Score dari model #1 (Dataset `Student_ChatGPT`, terhadap label Human)
4. Hasil Similarity Score dari model #1 (Dataset `Student_ChatGPT`, terhadap label AI)
5. Hasil Similarity Score dari model #2 (Dataset `Only_ChatGPT`, terhadap label AI)
6. Fitur Stylometric

Dataset yang digunakan yaitu gabungan Dataset training dari `Student_ChatGPT` dan `Only_ChatGPT`. 