# Dokumentasi Kode — Deteksi Teks AI pada Esai Mahasiswa

> Dokumentasi ini mencakup empat varian eksperimen berdasarkan **granularitas segmentasi teks** (paragraf vs. kalimat) dan **penggunaan pengetahuan ChatGPT** (tanpa pengetahuan vs. dengan pengetahuan), menghasilkan kombinasi:
>
> | Kode | Granularitas | Pengetahuan ChatGPT |
> |------|-------------|----------------------|
> | `code_TA_paragraph_noknowledge` | Paragraf | ✗ Tidak |
> | `code_TA_paragraph_withknowledge` | Paragraf | ✓ Ya |
> | `code_TA_sentence_noknowledge` | Kalimat | ✗ Tidak |
> | `code_TA_sentence_withknowledge` | Kalimat | ✓ Ya |

---

## Flow of Code

```
Dataset
  ├── Esai Siswa (Student_ChatGPT)
  ├── Esai ChatGPT (Student_ChatGPT)
  └── Pengetahuan ChatGPT (Only_ChatGPT) ──── [Hanya pada varian "with knowledge"]
         │
         ▼
[1] Data Loading & Segmentasi Teks
      ├── Granularitas Paragraf → preprocess_paragraph()
      └── Granularitas Kalimat  → preprocess_sentence()
         │
         ▼
[2] Data Splitting
      ├── Training Set : 75%
      ├── Validation Set: 5%
      └── Test Set     : 20%
         │
         ▼
[3] Tokenisasi — IndoBERT Tokenizer
      └── BertTokenizer (indobenchmark/indobert-base-p2)
          max_length = 256
         │
         ▼
[4] Semantic Similarity Model
      ├── Base Model  : TFBertModel (IndoBERT-base-p2) — Frozen
      ├── Arsitektur  : Bi-Encoder
      ├── Training    : Triplet Loss + Adam Optimizer (lr=2e-5)
      ├── Pasangan Data:
      │     ├── Tanpa Pengetahuan: Student ↔ ChatGPT Essay
      │     └── Dengan Pengetahuan: Student ↔ ChatGPT Essay + ChatGPT Essay ↔ Pengetahuan
      └── Output Model: semantic_model.h5
         │
         ▼
[5] Embedding & Similarity Score
      ├── Generate Embeddings (All Text, Training, Validation, Test Set)
      └── Cosine Similarity Score antar pasangan teks
         │
         ▼
[6] Ekstraksi Fitur Linguistik (Stylometric)
      ├── Lexical Diversity
      ├── Total Words
      ├── Total Unique Words
      ├── Modals
      ├── Stopwords Ratio
      ├── Average Sentence Length
      ├── Sentence Length Variation
      └── Punctuation Ratio
         │
         ▼
[7] Klasifikasi
      ├── Input:
      │     ├── Embeddings (dim 256)
      │     ├── Similarity Score (dim 2 atau 3*)
      │     └── Linguistic Features (dim 8)
      ├── Arsitektur: Multi-Input Dense Neural Network
      ├── Training  : 30 Epochs, EarlyStopping (patience=2)
      └── Output Model: classification_model.h5
         │
         ▼
[8] Evaluasi & Penyimpanan Model
      ├── Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
      ├── Confusion Matrix & Classification Report
      ├── Misclassified Essay Report
      └── Simpan: semantic_model.h5, classification_model.h5,
                  reference_embeddings.pkl, scaler_linguistic.pkl, tokenizer/
```

> *Dimensi similarity score pada varian **with knowledge** bertambah karena menghitung kemiripan terhadap dua referensi: esai ChatGPT dan pengetahuan ChatGPT.

---

## 1. Import Libraries

Semua varian menggunakan library yang sama:

```python
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import pandas as pd
import pickle, os, re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 2. Load Dataset

### Varian Tanpa Pengetahuan (`noknowledge`)

```python
stdset  = pd.read_csv("datasets_ta/esai_siswa.csv")      # Kolom: Essay
gptset  = pd.read_csv("datasets_ta/esai_gpt_new.csv")    # Kolom: Response
```

Dataset yang digunakan hanya dua: esai siswa dan esai ChatGPT pada topik yang sama.

### Varian Dengan Pengetahuan (`withknowledge`)

```python
stdset   = pd.read_csv("datasets_ta/esai_siswa.csv")
gpt1set  = pd.read_csv("datasets_ta/esai_gpt_new.csv")
gpt2set  = pd.read_csv("datasets_ta/pengetahuan_gpt_new.csv")  # Kolom: Response
```

Dataset ketiga (`gpt2set`) berisi pengetahuan ChatGPT yang di-generate berdasarkan RPS mata kuliah selama dua semester, mencakup banyak topik/subbab.

---

## 3. Segmentasi Teks (Perbedaan Paragraf vs. Kalimat)

### Granularitas Paragraf — `preprocess_paragraph()`

Setiap esai dipecah berdasarkan baris kosong atau pola paragraf. Satu unit teks = satu paragraf. Cocok untuk menangkap gaya penulisan dalam cakupan yang lebih luas.

```python
def preprocess_paragraph(text):
    # - Lowercase
    # - Bersihkan whitespace berlebih
    # - Pisahkan menjadi paragraf
    # Returns: list of paragraph strings
```

### Granularitas Kalimat — `preprocess_sentence()`

Setiap esai dipecah pada level kalimat menggunakan regex. Satu unit teks = satu kalimat. Cocok untuk analisis yang lebih granular dan sensitif terhadap pola sintaksis.

```python
def preprocess_sentence(text):
    # - Lowercase
    # - Pisahkan kalimat via regex (titik, tanda tanya, tanda seru)
    # - Pertahankan kalimat meskipun tidak diakhiri tanda baca standar
    # Returns: list of sentence strings
```

---

## 4. Load Dataset for Training

### Varian Tanpa Pengetahuan

```python
# Label: Student = 0, ChatGPT = 1
std_df  = pd.DataFrame({'text': std_txt,  'label': 0})
gpt1_df = pd.DataFrame({'text': gpt_txt,  'label': 1})
data    = pd.concat([std_df, gpt1_df], ignore_index=True)
```

### Varian Dengan Pengetahuan

```python
# gpt2 digabung ke dalam label ChatGPT (label 1) saat klasifikasi
std_df  = pd.DataFrame({'text': std_txt,   'label': 0})
gpt1_df = pd.DataFrame({'text': gpt1_txt,  'label': 1})
gpt2_df = pd.DataFrame({'text': gpt2_txt,  'label': 1})
data    = pd.concat([std_df, gpt1_df, gpt2_df], ignore_index=True)
```

---

## 5. Data Splitting

Pembagian data dilakukan secara stratified untuk menjaga distribusi label:

```
Training Set  : 75%
Validation Set:  5%
Test Set      : 20%
```

```python
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data['label']
)
train_set, val_set = train_test_split(
    train_data, test_size=0.0625, random_state=42, stratify=train_data['label']
)
# 0.0625 dari 80% = 5% dari keseluruhan data
```

---

## 6. Tokenisasi — IndoBERT Tokenizer

```python
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")

def tokenize_text(texts, max_length=256):
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
```

Token yang dihasilkan disimpan sebagai numpy array (`input_ids`, `attention_mask`) untuk efisiensi memori.

### Varian Dengan Pengetahuan — Tokenisasi 3 Kelompok

```python
std_tokens  = tokenize_text(std_txt)
gpt1_tokens = tokenize_text(gpt1_txt)   # Esai ChatGPT
gpt2_tokens = tokenize_text(gpt2_txt)   # Pengetahuan ChatGPT
```

---

## 7. Semantic Similarity Model

### Arsitektur — Bi-Encoder dengan IndoBERT

```python
bert_model = TFBertModel.from_pretrained("indobenchmark/indobert-base-p2")

# Freeze semua layer BERT agar pelatihan efisien
for layer in bert_model.layers:
    layer.trainable = False

def model(bert_model):
    input_ids      = tf.keras.layers.Input(shape=(256,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(256,), dtype=tf.int32)
    
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    cls_output  = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
    
    output = tf.keras.layers.Dense(256, activation='relu')(cls_output)
    output = tf.keras.layers.Dropout(0.1)(output)
    output = tf.keras.layers.Dense(256)(output)          # embedding output
    
    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
```

### Contrastive Pairs

**Varian Tanpa Pengetahuan**

| Pasangan | Label |
|----------|-------|
| Esai Siswa ↔ Esai Siswa (within/cross) | Positif |
| Esai ChatGPT ↔ Esai ChatGPT (within/cross) | Positif |
| Esai Siswa ↔ Esai ChatGPT | Negatif |

**Varian Dengan Pengetahuan**

| Pasangan | Label |
|----------|-------|
| Esai Siswa ↔ Esai Siswa (within/cross) | Positif |
| Esai ChatGPT ↔ Esai ChatGPT (within/cross) | Positif |
| Esai ChatGPT ↔ Pengetahuan ChatGPT | Positif |
| Esai Siswa ↔ Pengetahuan ChatGPT | Negatif |
| Esai Siswa ↔ Esai ChatGPT | Negatif |

### Triplet Loss & Training

```python
def triplet_loss(y_true, y_pred):
    # y_pred: [positive_similarity, negative_similarity]
    margin = 0.5
    loss = tf.maximum(0.0, margin - y_pred[:, 0] + y_pred[:, 1])
    return tf.reduce_mean(loss)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)  # mixed_float16

build_triplet.compile(optimizer=optimizer, loss=triplet_loss)
history = build_triplet.fit(
    x=[anchor, positive, negative],
    validation_data=([va_anchor, va_positive, va_negative], va_labels),
    epochs=10,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
)
```

---

## 8. Generate Embeddings & Cosine Similarity

```python
def gen_emb(tokens, model, batch_size=32):
    # Generate embeddings secara batch untuk efisiensi memori
    embeddings = []
    for i in range(0, len(tokens['input_ids']), batch_size):
        batch = {k: v[i:i+batch_size] for k, v in tokens.items()}
        emb   = model(batch)
        embeddings.append(emb.numpy())
    return np.vstack(embeddings)

def cos_sim(embedding1, embedding2):
    # Rata-rata cosine similarity antara satu embedding dan sekumpulan referensi
    sim = tf.keras.losses.cosine_similarity(embedding1, embedding2)
    return tf.reduce_mean(-sim).numpy()
```

Similarity score dihitung untuk semua kombinasi pasangan (Student-Student, Student-ChatGPT, ChatGPT-ChatGPT) pada keseluruhan data, test set, training set, dan validation set.

---

## 9. Ekstraksi Fitur Linguistik (Stylometric)

Delapan fitur stylometric diekstrak dari setiap unit teks:

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | `lexical_diversity` | Rasio kata unik terhadap total kata |
| 2 | `total_words` | Jumlah total kata |
| 3 | `total_unique_words` | Jumlah kata unik |
| 4 | `modals` | Jumlah kata modal (harus, dapat, akan, dll.) |
| 5 | `stopwords_ratio` | Rasio stopword terhadap total kata |
| 6 | `avg_sentence_length` | Rata-rata panjang kalimat (dalam kata) |
| 7 | `sentence_length_variation` | Standar deviasi panjang kalimat |
| 8 | `punctuation_ratio` | Rasio tanda baca terhadap total karakter |

Fitur dinormalisasi menggunakan `StandardScaler` yang di-fit pada training set lalu diterapkan ke validation dan test set.

---

## 10. Klasifikasi

### Input Model

Model klasifikasi menerima tiga jenis input secara paralel:

```
Input 1 — Embeddings         : shape (N, 256)  → Dense(128, relu)
Input 2 — Similarity Scores  : shape (N, 2/3*) → Dense(16, relu)
Input 3 — Linguistic Features: shape (N, 8)    → Dense(16, relu)
                                                        │
                                               Concatenate → Dense(64, relu)
                                                           → Dropout(0.3)
                                                           → Dense(1, sigmoid)
```

> *Dimensi similarity score = 2 pada varian tanpa pengetahuan (Student-Student, Student-ChatGPT), dan 3 pada varian dengan pengetahuan (ditambah Student-Pengetahuan ChatGPT).

### Training Klasifikasi

```python
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_classifier = classifier.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    epochs=30,
    batch_size=16,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
)
```

---

## 11. Evaluasi Model

Evaluasi dilakukan pada test set dengan metrik:

- **Accuracy** — Proporsi prediksi yang benar
- **Precision** — Ketepatan prediksi positif
- **Recall** — Kelengkapan deteksi kelas positif
- **F1-Score** — Harmonic mean dari Precision dan Recall
- **ROC-AUC** — Area under the ROC Curve
- **Confusion Matrix**
- **Classification Report** per kelas
- **Misclassified Essay Report** — Laporan detail teks yang salah diklasifikasikan

---

## 12. Simpan Model

Semua artefak disimpan per varian eksperimen:

| Artefak | Format | Deskripsi |
|---------|--------|-----------|
| `semantic_model.h5` | HDF5 | Model Bi-Encoder semantic similarity |
| `classification_model.h5` | HDF5 | Model klasifikasi multi-input |
| `reference_embeddings.pkl` | Pickle | Embeddings referensi seluruh data |
| `scaler_linguistic.pkl` | Pickle | Fitted StandardScaler untuk fitur linguistik |
| `tokenizer/` | Direktori | IndoBERT tokenizer |

Struktur direktori output:

```
ta_paragraph_1/    ← paragraph, tanpa pengetahuan
ta_paragraph_2/    ← paragraph, dengan pengetahuan
ta_sentence_1/     ← kalimat, tanpa pengetahuan
ta_sentence_2/     ← kalimat, dengan pengetahuan
```

---

## Ringkasan Perbedaan Antar Varian

| Aspek | Tanpa Pengetahuan | Dengan Pengetahuan |
|-------|------------------|--------------------|
| **Dataset** | Esai Siswa + Esai ChatGPT | Esai Siswa + Esai ChatGPT + Pengetahuan ChatGPT |
| **Segmentasi (Paragraf)** | `preprocess_paragraph()` | `preprocess_paragraph()` |
| **Segmentasi (Kalimat)** | `preprocess_sentence()` | `preprocess_sentence()` |
| **Tokenized groups** | 2 (std, gpt) | 3 (std, gpt1, gpt2) |
| **Pasangan Negatif** | Siswa ↔ ChatGPT Essay | Siswa ↔ ChatGPT Essay + Siswa ↔ Pengetahuan |
| **Pasangan Positif Tambahan** | — | ChatGPT Essay ↔ Pengetahuan ChatGPT |
| **Dim Similarity Score (Klasifikasi)** | 2 | 3 |
| **Tujuan Utama** | Membedakan gaya esai siswa vs. AI | Memperkuat sinyal AI via pengetahuan domainnya |
