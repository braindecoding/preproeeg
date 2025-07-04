# AC-GAN

## 🎯 **Interpretasi Terbaik yang Diimplementasikan:**

### **1. Resolusi Inkonsistensi Input Shape**
- **Problem**: Paper menunjukkan input `(None, 9, 32, 1)` tapi menggunakan 4 channels
- **Solusi**: Setiap channel (T7, P7, T8, P8) diproses terpisah dengan 9 windows
- **Result**: Input ke CNN adalah `(9, 32, 1)` per channel

### **2. Sliding Window Logic yang Benar**
- **From**: 256 timepoints × 4 channels 
- **To**: 9 windows × 32 timepoints × 1 channel
- **Method**: Sliding window dengan overlap 4 samples
- **Output**: `(n_samples × 4_channels, 9, 32, 1)`

### **3. CNN Architecture yang Masuk Akal**
- **Fixed**: Kernel sizes yang tidak konsisten dalam paper
- **Solution**: Menggunakan kernel yang logis untuk input shape
- **Maintained**: Struktur layer dan hyperparameter yang disebutkan

### **4. Preprocessing Pipeline yang Konsisten**
```
Raw Data (n_samples, 14_channels, 256_timepoints)
    ↓ Notch Filter (50Hz)
    ↓ Bandpass Filter (0.4-60Hz)
    ↓ MNE Artifact Removal (100µV)
    ↓ Common Average Reference
    ↓ Correlation Selection (ρ > 0.9)
    ↓ Channel Selection (T7, P7, T8, P8)
    ↓ Sliding Windows (9 × 32)
Final Data (n_samples×4, 9, 32, 1)
```

### **5. Hyperparameter Sesuai Paper**
- ✅ Adam optimizer (lr=0.001, β₁=0.8)
- ✅ Batch size 128
- ✅ Max epochs 150
- ✅ Dropout 0.1
- ✅ L2 regularization
- ✅ BatchNormalization placement

## 🔧 **Keunggulan Implementasi Ini:**

1. **Consistent Logic**: Menyelesaikan inkonsistensi dalam paper
2. **Realistic Data Flow**: Pipeline yang masuk akal untuk EEG processing
3. **Proper Dimensionality**: Input/output shapes yang konsisten
4. **Complete Pipeline**: Dari raw data hingga latent vectors
5. **Comprehensive Logging**: Tracking setiap tahap preprocessing

## 📊 **Expected Results:**
- **Classification Accuracy**: ~92% (target dari paper)
- **Latent Vectors**: 128-dimensional untuk GAN input
- **Data Reduction**: Sesuai dengan Table 1 dalam paper
- **SNR Improvement**: Before/after CAR selection

## 🚀 **Ready for GAN Training:**
Latent vectors yang dihasilkan (128-dimensional) siap untuk:
- AC-GAN training
- Image reconstruction
- MNIST digit generation

Implementation ini memberikan **interpretasi terbaik** yang menyelesaikan inkonsistensi dalam paper sambil mempertahankan esensi metodologi yang dijelaskan oleh penulis.

https://www.scitepress.org/Papers/2025/131493/131493.pdf#page=9.53
