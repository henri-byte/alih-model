# alih-model

Penerjemah **Jepang → Indonesia** yang paham konteks, untuk aplikasi Alih Manga (mode offline).

- Model dasar: [Helsinki-NLP/opus-mt-ja-ms](https://huggingface.co/Helsinki-NLP/opus-mt-ja-ms) (Apache-2.0), dilatih ulang langsung ke bahasa Indonesia.
- Data: korpus terbuka [OPUS](https://opus.nlpl.eu) — OpenSubtitles, TED2020, QED, Tatoeba.
- Konteks: masukan berisi maks. 2 kalimat sebelumnya, `>>ind<< kalimat1 ◇ kalimat2 ◆ kalimat yang diterjemahkan`.

Urutan: **1. Siapkan data** → **2. Latih model** (berantai otomatis) → **3. Ekspor & uji model**. Hasil ada di Releases → `model`.
