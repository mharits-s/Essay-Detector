from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import time
import os

load_dotenv()
api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)
responses = []

for i in range(10):
    try:
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "text": "Bayangkan kamu adalah seorang siswa SMA yang sedang mengikuti lomba menulis esai tingkat nasional. Tulis esai secara langsung, dimulai dari judul dan dilanjutkan dengan paragraf-paragraf isi esai.\nTema umum: “Hidup Adalah untuk Mengolah Hidup” (Tidak perlu menulis tema ini secara tekstual. Tuangkan makna dan gagasan tema secara kreatif dan kontekstual. Gunakan pendekatan unik agar tidak terkesan kaku.)\nKetentuan teknis penulisan:\n* Format tulisan mengikuti: A4, font Times New Roman ukuran 12 pt, spasi 1.5, margin atas 3 cm, bawah 3 cm, kiri 4 cm, kanan 3 cm, rata kanan-kiri (justify)\n* Panjang esai: 3–5 halaman setara Word atau sekitar 10–20 paragraf (idealnya di atas 14 paragraf)\n* Gaya bahasa: Bahasa Indonesia yang baik dan benar (sesuai PUEBI), tapi tetap luwes dan komunikatif seperti siswa SMA\nStruktur penulisan:\n* Judul (dibuat menarik dan mencerminkan gagasan)\n* Isi/Naskah Esai (dengan alur berpikir yang jelas, argumentatif, dan mengalir)\nAspek yang dinilai:\n* Gagasan orisinil, kreatif, aktual.\n* Kesesuaian dengan tema.\n* Kebahasaan (tata bahasa).\n* Kejelasan dan ketepatan argumen.\n* Organisasi dan koherensi teks.\n* Dampak dan keterlibatan pembaca\n* Kemampuan Pengetahuan Bahasa Indonesia.\nLangsung tulis esai lengkap, mulai dari judul yang menarik dan sesuai, diikuti oleh paragraf-paragraf isi esai.\n\nCatatan:\nTema diambil dari larik puisi WS Rendra berjudul Sajak Seorang Tua untuk Istrinya. Tidak perlu menuliskan asal tema atau penyairnya dalam esai.",
                "type": "text"
                }
            ]
            }
        ],
        response_format={
            "type": "text"
        },
        temperature=1,
        max_completion_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        store=False
        )
        text_response = response.choices[0].message.content
        
        # Menyimpan jawaban
        responses.append(text_response)
        print(f"Essay {i+1} completed")
        time.sleep(1)

    except Exception as e:
        print(f"Terjadi kesalahan pada sesi {i+1}: {e}")
        break

os.makedirs("datasets", exist_ok=True)

# Menyimpan jawaban ke file CSV
df = pd.DataFrame(responses, columns=["Response"])
df.to_csv("datasets/esai_gpt_25.csv", index=False)

print("Jawaban berhasil disimpan dalam file esai_gpt_25.csv")
