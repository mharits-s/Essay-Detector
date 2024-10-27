import openai
import pandas as pd
import time
import os


# Ambil API key dari variabel lingkungan
openai.api_key = os.getenv("API_KEY")

# Prompt yang ingin digunakan
prompt = "Buatkan esai tentang Bijak Menggunakan Sosmed bagi Remaja"

# Inisialisasi daftar untuk menyimpan jawaban
responses = []

for i in range(5):  # Sesuaikan dengan range(100) jika ingin 100 jawaban
    try:
        # Mengirim permintaan ke OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=1.2,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=1,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Buatkan esai tentang Bijak Menggunakan Sosmed bagi Remaja maksimal 600 karakter berupa kalimat biasa tanpa bold"
                }
            ]
        )
        
        # Ambil teks respons dan hilangkan spasi berlebih
        text_response = response.choices[0].message.content
        
        # Menyimpan jawaban
        responses.append(text_response)
        
        # Memberikan jeda untuk menghindari rate limit
        time.sleep(1)

    except Exception as e:
        print(f"Terjadi kesalahan pada sesi {i+1}: {e}")
        break

# Menyimpan jawaban ke file CSV
df = pd.DataFrame(responses, columns=["Response"])
df.to_csv("example_datasets/responses5.csv", index=False)

print("Jawaban berhasil disimpan dalam file responses5.csv")