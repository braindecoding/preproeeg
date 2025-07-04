import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, 
    BatchNormalization, Flatten, Embedding, Reshape,
    UpSampling2D, LeakyReLU, Multiply, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EEGDataProcessor:
    """
    Kelas untuk memproses data EEG sesuai dengan metodologi dalam paper
    Interpretasi terbaik berdasarkan analisis menyeluruh
    """
    
    def __init__(self, sampling_rate=128):
        self.sampling_rate = sampling_rate
        self.selected_channels = ['T7', 'P7', 'T8', 'P8']  # 4 channels discriminative
        self.scaler = StandardScaler()
        
    def load_mindbigdata_format(self, file_path, target_channels=None):
        """
        Load MindBigData format untuk Emotiv EPOC
        
        Format file MindBigData:
        [id][event][device][channel][code][size][data]
        
        Untuk EPOC (EP):
        - device: "EP" 
        - channels: "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
        - code: 0-9 untuk digits, -1 untuk random
        - size: ~256 values (2 seconds x 128Hz)
        - data: comma-separated real numbers
        """
        if target_channels is None:
            # Emotiv EPOC 14 channels sesuai MindBigData
            target_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", 
                             "P8", "T8", "FC6", "F4", "F8", "AF4"]
        
        print(f"Loading MindBigData format from: {file_path}")
        print(f"Target channels: {target_channels}")
        
        # Dictionary untuk menyimpan data per event
        events_data = {}
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num % 10000 == 0:
                    print(f"Processing line {line_num}...")
                
                # Parse line (tab-separated)
                parts = line.strip().split('\t')
                
                if len(parts) != 6:
                    continue
                
                try:
                    id_val = int(parts[0])
                    event_id = int(parts[1])
                    device = parts[2]
                    channel = parts[3]
                    code = int(parts[4])
                    size = int(parts[5])
                    data_str = parts[6] if len(parts) > 6 else ""
                    
                    # Filter untuk Emotiv EPOC dan digit yang valid
                    if device != "EP" or code < 0 or code > 9:
                        continue
                    
                    # Filter untuk channel yang diinginkan
                    if channel not in target_channels:
                        continue
                    
                    # Parse data values
                    if data_str:
                        data_values = [float(x) for x in data_str.split(',')]
                    else:
                        continue
                    
                    # Pastikan data memiliki ukuran yang konsisten (~256 untuk 2 detik)
                    if len(data_values) < 200 or len(data_values) > 300:
                        continue
                    
                    # Resize ke 256 timepoints (interpolasi atau padding)
                    if len(data_values) != 256:
                        data_values = self._resize_signal(data_values, 256)
                    
                    # Simpan data per event
                    if event_id not in events_data:
                        events_data[event_id] = {
                            'code': code,
                            'channels': {}
                        }
                    
                    events_data[event_id]['channels'][channel] = np.array(data_values)
                    
                except (ValueError, IndexError) as e:
                    continue
        
        print(f"Loaded {len(events_data)} events")
        
        # Convert ke format yang dibutuhkan
        return self._convert_events_to_arrays(events_data, target_channels)
    
    def _resize_signal(self, signal, target_length):
        """Resize signal ke target length menggunakan interpolasi"""
        if len(signal) == target_length:
            return signal
        
        # Interpolasi linear
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_length)
        signal_resized = np.interp(x_new, x_old, signal)
        
        return signal_resized
    
    def _convert_events_to_arrays(self, events_data, target_channels):
        """Convert events dictionary ke numpy arrays"""
        raw_data = []
        labels = []
        
        for event_id, event_info in events_data.items():
            # Periksa apakah semua channel tersedia
            if len(event_info['channels']) != len(target_channels):
                continue
            
            # Susun data per channel sesuai urutan target_channels
            sample_data = []
            for channel in target_channels:
                if channel in event_info['channels']:
                    sample_data.append(event_info['channels'][channel])
                else:
                    # Skip event jika ada channel yang hilang
                    break
            
            if len(sample_data) == len(target_channels):
                raw_data.append(sample_data)
                labels.append(event_info['code'])
        
        raw_data = np.array(raw_data)  # (n_samples, n_channels, n_timepoints)
        labels = np.array(labels)
        
        print(f"Final data shape: {raw_data.shape}")
        print(f"Final labels shape: {labels.shape}")
        
        # Display distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("Data distribution:")
        for digit, count in zip(unique, counts):
            print(f"  Digit {digit}: {count} samples")
        
        return raw_data, labels
    
    def load_data(self, file_path, file_type='mindbigdata'):
        """
        Load data EEG dari berbagai format
        
        Parameters:
        file_path: path ke file data
        file_type: 'mindbigdata', 'csv', 'npy'
        """
        if file_type == 'mindbigdata':
            return self.load_mindbigdata_format(file_path)
            
        elif file_type == 'csv':
            # Format CSV tradisional
            data = pd.read_csv(file_path)
            labels = data.iloc[:, -1].values
            eeg_data = data.iloc[:, :-1].values
            
            n_samples = eeg_data.shape[0]
            n_channels = 14  # Emotiv EPOC
            n_timepoints = eeg_data.shape[1] // n_channels
            
            raw_data = eeg_data.reshape(n_samples, n_channels, n_timepoints)
            
        elif file_type == 'npy':
            data = np.load(file_path, allow_pickle=True)
            raw_data = data['eeg_data']
            labels = data['labels']
            
        return raw_data, labels
    
    def notch_filter(self, data, notch_freq=50, quality_factor=30):
        """Zero-phase AC notch filter at 50Hz with 1Hz band"""
        nyquist = self.sampling_rate / 2
        notch_freq_norm = notch_freq / nyquist
        
        b, a = signal.iirnotch(notch_freq_norm, quality_factor)
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        
        return filtered_data
    
    def bandpass_filter(self, data, low_freq=0.4, high_freq=60, order=5):
        """
        5th order Butterworth non-causal bandpass filter (0.4-60Hz)
        -6dB cutoff frequency dengan Hamming window
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        
        return filtered_data
    
    def remove_artifacts_mne(self, data, threshold=200):
        """
        Remove epochs menggunakan MNE package functionality
        Maximum 200µV peak-to-peak threshold (adjusted for synthetic data)
        """
        # Hitung peak-to-peak amplitude
        peak_to_peak = np.max(data, axis=-1) - np.min(data, axis=-1)

        # Identifikasi epochs yang exceed threshold
        artifact_mask = np.any(peak_to_peak > threshold, axis=1)

        clean_data = data[~artifact_mask]
        clean_indices = np.where(~artifact_mask)[0]

        print(f"   Removed {np.sum(artifact_mask)} epochs due to artifacts (>{threshold}µV)")

        return clean_data, clean_indices
    
    def common_average_reference(self, data):
        """
        Apply Common Average Reference (CAR) method
        CAR = Xi - (1/N) * Σ(Xi) where N=14 channels
        """
        # Hitung rata-rata dari semua 14 channels
        car_reference = np.mean(data, axis=1, keepdims=True)
        car_data = data - car_reference
        
        return car_data
    
    def correlation_selection_per_digit(self, data, labels, correlation_threshold=0.3):
        """
        CAR correlation selection method berdasarkan paper:
        1. Untuk setiap digit, hitung average signal dari semua 14 channels
        2. Hitung correlation coefficient pearson antara setiap channel dan mean signal
        3. Pilih samples dengan pearson correlation > 0.3 (adjusted for synthetic data)
        """
        selected_data = []
        selected_labels = []
        total_samples = 0
        selected_samples = 0

        for digit in range(10):
            digit_indices = np.where(labels == digit)[0]
            if len(digit_indices) == 0:
                continue

            digit_data = data[digit_indices]

            for sample in digit_data:
                total_samples += 1
                # Hitung mean signal dari semua 14 channels
                mean_signal = np.mean(sample, axis=0)  # (256,)

                # Hitung korelasi setiap channel dengan mean signal
                correlations = []
                for ch_idx in range(sample.shape[0]):
                    channel_signal = sample[ch_idx]

                    if np.std(channel_signal) > 0 and np.std(mean_signal) > 0:
                        corr, _ = pearsonr(channel_signal, mean_signal)
                        correlations.append(corr)
                    else:
                        correlations.append(0.0)

                # Periksa apakah ada correlation > threshold
                max_corr = np.max(correlations) if correlations else 0.0
                if max_corr > correlation_threshold:
                    selected_data.append(sample)
                    selected_labels.append(digit)
                    selected_samples += 1

        print(f"   Correlation selection: {selected_samples}/{total_samples} samples passed")
        print(f"   Max correlation found: {np.max([np.max([pearsonr(sample[ch], np.mean(sample, axis=0))[0] for ch in range(sample.shape[0])]) for sample in data[:10]]) if len(data) > 0 else 0:.3f}")

        return np.array(selected_data), np.array(selected_labels)
    
    def select_channels_t7_p7_t8_p8(self, data, channel_names=None):
        """
        Select 4 discriminative channels: T7, P7, T8, P8
        Berdasarkan Mishra et al. (2021) yang dikutip dalam paper
        
        MindBigData EPOC channel order:
        ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        """
        if channel_names is None:
            # MindBigData EPOC channel order (berbeda dari default saya sebelumnya)
            channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", 
                           "P8", "T8", "FC6", "F4", "F8", "AF4"]
        
        target_channels = ['T7', 'P7', 'T8', 'P8']
        channel_indices = []
        
        for ch in target_channels:
            if ch in channel_names:
                channel_indices.append(channel_names.index(ch))
            else:
                raise ValueError(f"Channel {ch} not found in channel list")
        
        selected_data = data[:, channel_indices, :]
        print(f"   Selected channels: {[channel_names[i] for i in channel_indices]}")
        print(f"   Channel indices: {channel_indices}")
        
        return selected_data
    
    def create_sliding_windows_paper_method(self, data, window_size=32, overlap=4):
        """
        Create sliding windows sesuai dengan paper method:
        - Input: (n_samples, 4_channels, 256_timepoints)
        - Create 9 windows per channel dengan 32 timepoints dan 4 overlap
        - Output: (n_samples, 4_channels, 9_windows, 32_timepoints)
        
        Kemudian reshape untuk CNN input yang akan menggunakan format:
        (n_samples × 4_channels, 9_windows, 32_timepoints, 1)
        """
        windowed_data = []
        
        for sample in data:
            n_channels, n_timepoints = sample.shape  # (4, 256)
            sample_windows = []
            
            for ch in range(n_channels):
                channel_data = sample[ch]  # (256,)
                
                # Create sliding windows for this channel
                channel_windows = []
                step = window_size - overlap  # 28
                
                for start in range(0, n_timepoints - window_size + 1, step):
                    end = start + window_size
                    window = channel_data[start:end]
                    channel_windows.append(window)
                
                # Pastikan ada tepat 9 windows
                if len(channel_windows) >= 9:
                    channel_windows = channel_windows[:9]
                else:
                    # Pad dengan zeros jika kurang
                    while len(channel_windows) < 9:
                        channel_windows.append(np.zeros(window_size))
                
                sample_windows.append(channel_windows)
            
            windowed_data.append(sample_windows)
        
        # Convert to numpy array: (n_samples, 4_channels, 9_windows, 32_timepoints)
        windowed_data = np.array(windowed_data)
        
        # Reshape untuk CNN: (n_samples × 4_channels, 9_windows, 32_timepoints, 1)
        n_samples, n_channels, n_windows, n_timepoints = windowed_data.shape
        final_data = windowed_data.reshape(n_samples * n_channels, n_windows, n_timepoints, 1)
        
        return final_data
    
    def calculate_snr(self, data):
        """Calculate Signal-to-Noise Ratio"""
        signal_power = np.mean(data**2, axis=-1)
        noise_power = np.var(data, axis=-1)
        snr = signal_power / (noise_power + 1e-10)
        return np.mean(snr)
    
    def preprocess_pipeline(self, raw_data, labels, channel_names=None):
        """
        Complete preprocessing pipeline berdasarkan interpretasi terbaik dari paper
        """
        print("="*60)
        print("STARTING EEG PREPROCESSING PIPELINE")
        print("="*60)
        print(f"Initial data shape: {raw_data.shape}")
        print(f"Initial labels shape: {labels.shape}")
        
        # Display initial distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nInitial distribution:")
        for digit, count in zip(unique, counts):
            print(f"  Digit {digit}: {count} samples")
        
        # 1. Notch filter (50Hz)
        print("\n1. Applying zero-phase AC notch filter (50Hz, 1Hz band)...")
        filtered_data = self.notch_filter(raw_data)
        
        # 2. Bandpass filter (0.4-60Hz)
        print("2. Applying 5th order Butterworth bandpass filter (0.4-60Hz)...")
        filtered_data = self.bandpass_filter(filtered_data)
        
        # 3. MNE artifact removal
        print("3. Removing artifacts using MNE functionality (200µV threshold)...")
        clean_data, clean_indices = self.remove_artifacts_mne(filtered_data, threshold=200)
        clean_labels = labels[clean_indices]
        print(f"   Data after MNE thresholding: {clean_data.shape}")
        
        # 4. Common Average Reference
        print("4. Applying Common Average Reference (CAR)...")
        car_data = self.common_average_reference(clean_data)
        snr_before = self.calculate_snr(car_data)
        print(f"   SNR before CAR selection: {snr_before:.3f}")
        
        # 5. CAR correlation selection (ρ > 0.3)
        print("5. CAR correlation selection (ρ > 0.3)...")
        selected_data, selected_labels = self.correlation_selection_per_digit(
            car_data, clean_labels, correlation_threshold=0.3
        )
        print(f"   Data after CAR selection: {selected_data.shape}")
        
        if len(selected_data) > 0:
            snr_after = self.calculate_snr(selected_data)
            print(f"   SNR after CAR selection: {snr_after:.3f}")
            
            # Display distribution after CAR
            unique, counts = np.unique(selected_labels, return_counts=True)
            print("   Distribution after CAR selection:")
            for digit, count in zip(unique, counts):
                print(f"     Digit {digit}: {count} samples")
        
        # 6. Channel selection (T7, P7, T8, P8)
        print("6. Selecting discriminative channels (T7, P7, T8, P8)...")
        if len(selected_data) > 0:
            selected_data = self.select_channels_t7_p7_t8_p8(selected_data, channel_names)
            print(f"   Data after channel selection: {selected_data.shape}")
        
        # 7. Sliding window creation
        print("7. Creating sliding windows (32 samples, 4 overlap → 9 windows)...")
        if len(selected_data) > 0:
            windowed_data = self.create_sliding_windows_paper_method(selected_data)
            print(f"   Data after windowing: {windowed_data.shape}")
            
            # Expand labels (4 channels × n_samples)
            expanded_labels = np.repeat(selected_labels, 4)
            print(f"   Expanded labels shape: {expanded_labels.shape}")
            
            # Final distribution
            unique, counts = np.unique(expanded_labels, return_counts=True)
            print("   Final distribution:")
            for digit, count in zip(unique, counts):
                print(f"     Digit {digit}: {count} samples")
        else:
            print("   ERROR: No data survived preprocessing!")
            windowed_data = np.array([])
            expanded_labels = np.array([])
        
        print("="*60)
        print("PREPROCESSING COMPLETED")
        print("="*60)
        
        return windowed_data, expanded_labels

class EEGClassifier:
    """
    CNN Classifier berdasarkan interpretasi terbaik dari paper
    Input: (9_windows, 32_timepoints, 1_channel)
    """
    
    def __init__(self, input_shape=(9, 32, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """
        Build CNN model berdasarkan interpretasi terbaik dari paper
        Mengatasi inkonsistensi dengan pendekatan yang logis
        """
        inputs = Input(shape=self.input_shape)  # (9, 32, 1)
        
        # Layer 1: BatchNormalization
        x = BatchNormalization()(inputs)
        
        # Layer 2: Conv2D - kernel disesuaikan untuk input (9, 32, 1)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)  # (9, 32, 128)
        
        # Layer 3: Conv2D 
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)  # (9, 32, 64)
        
        # Layer 4: MaxPooling2D
        x = MaxPooling2D((1, 2))(x)  # (9, 16, 64)
        
        # Layer 5: Conv2D - kernel disesuaikan
        x = Conv2D(64, (3, 4), padding='valid', activation='relu')(x)  # (7, 13, 64)
        
        # Layer 6: MaxPooling2D
        x = MaxPooling2D((1, 2))(x)  # (7, 6, 64)
        
        # Layer 7: Conv2D - kernel disesuaikan
        x = Conv2D(128, (3, 3), padding='valid', activation='relu')(x)  # (5, 4, 128)
        
        # Layer 8: Flatten
        x = Flatten()(x)
        
        # Layer 9: BatchNormalization
        x = BatchNormalization()(x)
        
        # Layer 10: Dense (512 nodes)
        x = Dense(512, activation='relu')(x)
        
        # Layer 11: Dropout (0.1)
        x = Dropout(0.1)(x)
        
        # Layer 12: Dense (256 nodes)
        x = Dense(256, activation='relu')(x)
        
        # Layer 13: Dropout (0.1)
        x = Dropout(0.1)(x)
        
        # Layer 14: Dense (128 nodes) - Latent space
        latent_vector = Dense(128, activation='relu', name='latent_vector')(x)
        
        # Layer 15: Dropout (0.1)
        latent_vector = Dropout(0.1)(latent_vector)
        
        # Layer 16: BatchNormalization
        latent_vector = BatchNormalization()(latent_vector)
        
        # Layer 17: Dense (10 nodes) - Classification
        predictions = Dense(self.num_classes, activation='softmax', 
                          kernel_regularizer=tf.keras.regularizers.l2(0.01), 
                          name='predictions')(latent_vector)
        
        self.model = Model(inputs=inputs, outputs=[predictions, latent_vector])
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, momentum=0.8):
        """
        Compile dengan hyperparameter sesuai paper
        """
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, beta_1=momentum),
            loss={'predictions': 'categorical_crossentropy'},
            metrics={'predictions': 'accuracy'}
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=128):
        """
        Train model dengan hyperparameter dari paper
        """
        if self.model is None:
            self.compile_model()
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model dan return results"""
        predictions, latent_vectors = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Classification report
        report = classification_report(y_test, y_pred, 
                                     target_names=[str(i) for i in range(10)])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return report, cm, latent_vectors
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['predictions_accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_predictions_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

# Contoh penggunaan dengan MindBigData format
def main():
    """
    Contoh penggunaan dengan MindBigData format yang sebenarnya
    """
    print("EEG-to-Image Reconstruction Pipeline")
    print("MindBigData Emotiv EPOC Format")
    print("="*80)
    
    # Inisialisasi
    processor = EEGDataProcessor(sampling_rate=128)
    
    # MindBigData EPOC channel order (sesuai dokumentasi)
    channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", 
                    "P8", "T8", "FC6", "F4", "F8", "AF4"]
    
    # Load data dari MindBigData format
    file_path = "path/to/your/mindbigdata_file.txt"  # Ganti dengan path yang sebenarnya
    
    # Untuk demo, kita akan generate data dummy yang sesuai dengan format MindBigData
    print("Creating dummy data in MindBigData format...")
    
    # Generate dummy data dengan format yang realistis
    n_samples = 2000
    n_channels = 14
    n_timepoints = 256
    
    np.random.seed(42)
    
    # Generate data dengan pola yang lebih realistis untuk EEG
    raw_data = []
    labels = []
    
    for i in range(n_samples):
        digit = i % 10
        
        # Generate sample untuk semua 14 channels
        sample = np.zeros((n_channels, n_timepoints))
        
        for ch in range(n_channels):
            # Base signal (background EEG)
            base_signal = np.random.randn(n_timepoints) * 20  # µV
            
            # Add digit-specific patterns (lebih kuat pada channels T7, P7, T8, P8)
            if channel_names[ch] in ['T7', 'P7', 'T8', 'P8']:
                # Stronger signal for discriminative channels
                pattern = np.sin(2 * np.pi * (digit + 1) * 
                               np.linspace(0, 2, n_timepoints)) * 25
                base_signal += pattern
            else:
                # Weaker pattern for other channels
                pattern = np.sin(2 * np.pi * (digit + 1) * 
                               np.linspace(0, 2, n_timepoints)) * 5
                base_signal += pattern
            
            # Add some noise
            base_signal += np.random.randn(n_timepoints) * 10
            
            sample[ch] = base_signal
        
        raw_data.append(sample)
        labels.append(digit)
    
    raw_data = np.array(raw_data)
    labels = np.array(labels)
    
    print(f"Generated data shape: {raw_data.shape}")
    print(f"Generated labels shape: {labels.shape}")
    
    # Jika Anda memiliki file MindBigData yang sebenarnya, uncomment baris ini:
    # raw_data, labels = processor.load_data(file_path, file_type='mindbigdata')
    
    # Preprocessing
    processed_data, processed_labels = processor.preprocess_pipeline(
        raw_data, labels, channel_names
    )
    
    if len(processed_data) == 0:
        print("ERROR: No data survived preprocessing!")
        return
    
    # Data splitting sesuai dengan paper (80% train, 20% test)
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data, processed_labels, 
        test_size=0.2, random_state=42, stratify=processed_labels
    )
    
    # Training data split (75% train, 25% val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.25, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize dan compile CNN
    classifier = EEGClassifier(input_shape=X_train.shape[1:])
    model = classifier.compile_model(learning_rate=0.001, momentum=0.8)
    
    print("\nCNN Model Architecture:")
    print(f"Input shape: {X_train.shape[1:]}")
    model.summary()
    
    # Training dengan hyperparameter dari paper
    print("\nTraining CNN classifier...")
    print("Hyperparameters:")
    print("- Learning rate: 0.001")
    print("- Momentum (beta_1): 0.8") 
    print("- Batch size: 128")
    print("- Max epochs: 20 (demo, paper uses 150)")
    
    history = classifier.train(
        X_train, y_train, X_val, y_val, 
        epochs=20, batch_size=128  # Reduced for demo
    )
    
    # Evaluation
    print("\nEvaluating model...")
    report, cm, latent_vectors = classifier.evaluate(X_test, y_test)
    
    print("\nClassification Report:")
    print(report)
    
    # Calculate accuracy
    predictions, _ = classifier.model.predict(X_test)
    final_accuracy = np.mean(np.argmax(predictions, axis=1) == y_test)
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # Compare with paper results
    print(f"\nComparison with paper:")
    print(f"Paper accuracy (post-CAR): ~92%")
    print(f"Our accuracy: {final_accuracy*100:.2f}%")
    
    if final_accuracy > 0.85:
        print("✅ Good accuracy achieved!")
    else:
        print("⚠️  Consider more training epochs or parameter tuning")
    
    # Plotting
    classifier.plot_training_history(history)
    classifier.plot_confusion_matrix(cm)
    
    # Latent vectors untuk GAN
    print(f"\nLatent vectors for GAN:")
    print(f"Shape: {latent_vectors.shape}")
    print(f"Mean: {np.mean(latent_vectors):.4f}")
    print(f"Std: {np.std(latent_vectors):.4f}")
    print("Ready for AC-GAN training!")
    
    return processed_data, processed_labels, latent_vectors

def load_real_mindbigdata_file(file_path):
    """
    Contoh fungsi untuk load file MindBigData yang sebenarnya
    """
    processor = EEGDataProcessor(sampling_rate=128)
    
    try:
        raw_data, labels = processor.load_data(file_path, file_type='mindbigdata')
        print(f"Successfully loaded {len(raw_data)} samples")
        return raw_data, labels
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

# Contoh format file MindBigData
def create_sample_mindbigdata_file(output_path, n_samples=100):
    """
    Membuat sample file dalam format MindBigData untuk testing
    """
    channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", 
               "P8", "T8", "FC6", "F4", "F8", "AF4"]
    
    print(f"Creating sample MindBigData file: {output_path}")
    
    with open(output_path, 'w') as f:
        for i in range(n_samples):
            for ch_idx, channel in enumerate(channels):
                id_val = i * 14 + ch_idx
                event_id = i
                device = "EP"
                code = i % 10  # Digit 0-9
                size = 256
                
                # Generate dummy data
                data_values = np.random.randn(size) * 30 + np.sin(
                    2 * np.pi * (code + 1) * np.linspace(0, 2, size)
                ) * 20
                
                data_str = ','.join([f"{x:.6f}" for x in data_values])
                
                line = f"{id_val}\t{event_id}\tEP\t{channel}\t{code}\t{size}\t{data_str}\n"
                f.write(line)
    
    print(f"Sample file created with {n_samples} samples × 14 channels")

if __name__ == "__main__":
    # Uncomment untuk membuat sample file
    # create_sample_mindbigdata_file("sample_mindbigdata.txt", n_samples=500)
    
    main()
