# preproeeg
Prapemrosesan Sinyal EEG (Preprocessing) 🧠

Tahap ini adalah yang paling penting untuk memastikan sinyal yang dianalisis bersih dari gangguan (noise) dan siap untuk diekstraksi fiturnya. Berikut langkah-langkah yang dilakukan:

Penghilangan Noise: Sinyal EEG mentah sangat rentan terhadap noise dari berbagai sumber (misalnya, gerakan otot, kedipan mata, interferensi listrik). Untuk membersihkannya, dua jenis filter digital digunakan:

Filter Butterworth Low-Pass Orde Kelima: Filter ini digunakan untuk menghilangkan frekuensi tinggi yang tidak relevan dengan aktivitas otak, dengan frekuensi cut-off diatur pada 100 Hz.

Notch Filter 50 Hz: Filter ini secara spesifik menargetkan dan menghilangkan noise dari interferensi jaringan listrik, yang umumnya berada pada frekuensi 50 Hz.

Dekomposisi Sinyal menjadi Sub-band: Setelah bersih, sinyal EEG dipecah menjadi enam pita frekuensi (sub-band) yang berbeda. Setiap pita frekuensi ini diketahui berkaitan dengan kondisi kognitif dan motorik yang berbeda. Proses ini menggunakan filter Butterworth bandpass:

Delta (δ): 0.5–4 Hz

Theta (θ): 4–8 Hz

Alpha (α): 8–13 Hz

Beta Rendah (β₁): 13–20 Hz

Beta Tinggi (β₂): 20–30 Hz

Gamma (γ): 30–100 Hz
