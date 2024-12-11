import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from collections import Counter

class RoulettePrediction:
    def __init__(self):
        self.data_file = "roulette_data.csv"
        self.model_file = "roulette_models.joblib"
        self.sequence_length = 10  # Mengurangi sequence length untuk menghindari overfitting
        
        # Inisialisasi model dengan parameter yang dioptimalkan
        self.model_angka = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            class_weight='balanced_subsample',
            bootstrap=True,
            n_jobs=-1
        )
        
        self.model_warna = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            subsample=0.8,
            min_samples_split=2
        )
        
        self.scaler = StandardScaler()
        self.load_data()
        self.load_models()
        
        # Menambahkan historical patterns
        self.pattern_length = 5
        self.patterns = {}

    def load_data(self):
        if os.path.exists(self.data_file):
            self.df = pd.read_csv(self.data_file)
        else:
            self.df = pd.DataFrame(columns=[
                "Timestamp", "Angka", "Warna", 
                "Hot_Numbers", "Cold_Numbers",
                "Consecutive_Colors", "Last_Numbers"
            ])

    def load_models(self):
        if os.path.exists(self.model_file):
            try:
                models = joblib.load(self.model_file)
                self.model_angka = models['model_angka']
                self.model_warna = models['model_warna']
                self.scaler = models['scaler']
                print("Model sebelumnya berhasil dimuat.")
            except:
                print("Membuat model baru...")

    def save_models(self):
        joblib.dump({
            'model_angka': self.model_angka,
            'model_warna': self.model_warna,
            'scaler': self.scaler
        }, self.model_file)

    def tentukan_warna(self, angka):
        if angka == 0:
            return "Hijau"
        merah = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
        return "Merah" if angka in merah else "Hitam"

    def generate_features(self, data):
        if len(data) < self.sequence_length:
            return None
        
        # Fitur dasar
        last_numbers = data['Angka'].tail(self.sequence_length).values
        last_colors = data['Warna'].tail(self.sequence_length).values
        
        # Fitur baru: Pola berulang
        number_patterns = self.find_patterns(last_numbers)
        color_patterns = self.find_patterns(last_colors)
        
        # Fitur baru: Statistik sektoral
        sector_stats = self.calculate_sector_stats(last_numbers)
        
        # Fitur baru: Rasio dan proporsi
        odd_even_ratio = sum(1 for x in last_numbers if x % 2 == 1) / self.sequence_length
        high_low_ratio = sum(1 for x in last_numbers if x > 18) / self.sequence_length
        
        # Konversi warna ke numerik dengan one-hot encoding
        color_map = {'Merah': [1,0,0], 'Hitam': [0,1,0], 'Hijau': [0,0,1]}
        color_encoded = np.array([color_map[c] for c in last_colors]).flatten()
        
        # Menambahkan fitur cyclic untuk angka
        sin_numbers = np.sin(2 * np.pi * last_numbers / 37)
        cos_numbers = np.cos(2 * np.pi * last_numbers / 37)
        
        # Statistik tambahan
        number_freq = pd.Series(last_numbers).value_counts()
        hot_numbers = len(number_freq[number_freq > 1])
        cold_numbers = len(set(range(37)) - set(number_freq.index))
        
        # Analisis sektor
        sector_1 = sum(1 for x in last_numbers if 1 <= x <= 12) / self.sequence_length
        sector_2 = sum(1 for x in last_numbers if 13 <= x <= 24) / self.sequence_length
        sector_3 = sum(1 for x in last_numbers if 25 <= x <= 36) / self.sequence_length
        
        # Streak warna
        color_streak = 1
        last_color = last_colors[-1]
        for i in range(len(last_colors)-2, -1, -1):
            if last_colors[i] == last_color:
                color_streak += 1
            else:
                break
        
        # Gabungkan semua fitur
        features = np.concatenate([
            last_numbers,
            color_encoded,
            number_patterns,
            color_patterns,
            sector_stats,
            [odd_even_ratio, high_low_ratio],
            sin_numbers,
            cos_numbers,
            [hot_numbers, cold_numbers, color_streak],
            [sector_1, sector_2, sector_3]
        ])
        
        return features.reshape(1, -1)

    def tambah_histori(self, angka):
        warna = self.tentukan_warna(angka)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if len(self.df) >= self.sequence_length:
            last_sequence = self.df.tail(self.sequence_length)
            hot_numbers = ','.join(map(str, 
                last_sequence['Angka'].value_counts()[
                    last_sequence['Angka'].value_counts() > 1
                ].index.tolist()
            ))
            cold_numbers = ','.join(map(str, 
                list(set(range(37)) - set(last_sequence['Angka'].unique()))
            ))
            consecutive_colors = 1
            last_numbers = ','.join(map(str, last_sequence['Angka'].tolist()))
        else:
            hot_numbers = ""
            cold_numbers = ','.join(map(str, range(37)))
            consecutive_colors = 1
            last_numbers = str(angka)
        
        new_row = pd.DataFrame([[
            timestamp, angka, warna,
            hot_numbers, cold_numbers,
            consecutive_colors, last_numbers
        ]], columns=self.df.columns)
        
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.data_file, index=False)
        
        # Tambahkan perintah clear screen di sini
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"\nData ditambahkan: {angka} ({warna})")
        
        self.analyze_patterns()
        self.latih_dan_prediksi()

    def analyze_patterns(self):
        try:
            num_analyze = min(100, len(self.df))
            if num_analyze < 10:
                print("\nBelum cukup data untuk analisis pola (minimal 10 data).")
                return
            
            recent_data = self.df.tail(num_analyze)
            recent_numbers = recent_data['Angka'].tolist()
            recent_colors = recent_data['Warna'].tolist()
            
            print(f"\nAnalisis {num_analyze} Putaran Terakhir:")
            
            # Analisis angka
            number_freq = pd.Series(recent_numbers).value_counts()
            print("\nAngka yang paling sering muncul:")
            for num, freq in number_freq.head(5).items():
                print(f"Angka {num}: {freq} kali ({freq/num_analyze*100:.1f}%)")
            
            # Angka yang belum muncul
            missing_numbers = set(range(37)) - set(recent_numbers)
            if missing_numbers:
                print("\nAngka yang belum muncul:")
                print(', '.join(map(str, sorted(missing_numbers))))
            
            # Analisis warna
            color_freq = pd.Series(recent_colors).value_counts()
            print("\nDistribusi warna:")
            for color, freq in color_freq.items():
                print(f"{color}: {freq} kali ({freq/num_analyze*100:.1f}%)")
            
            # Analisis sektor
            sectors = {
                "1-12": sum(1 for x in recent_numbers if 1 <= x <= 12),
                "13-24": sum(1 for x in recent_numbers if 13 <= x <= 24),
                "25-36": sum(1 for x in recent_numbers if 25 <= x <= 36),
                "0": sum(1 for x in recent_numbers if x == 0)
            }
            print("\nDistribusi sektor:")
            for sector, count in sectors.items():
                print(f"Sektor {sector}: {count} kali ({count/num_analyze*100:.1f}%)")
            
        except Exception as e:
            print(f"Error dalam analisis pola: {str(e)}")

    def find_patterns(self, sequence):
        patterns = []
        for i in range(len(sequence) - self.pattern_length + 1):
            pattern = tuple(sequence[i:i+self.pattern_length])
            if pattern in self.patterns:
                self.patterns[pattern] += 1
            else:
                self.patterns[pattern] = 1
            patterns.append(self.patterns[pattern])
        
        while len(patterns) < self.sequence_length:
            patterns.append(0)
        
        return np.array(patterns)

    def calculate_sector_stats(self, numbers):
        sectors = {
            '1-12': sum(1 for x in numbers if 1 <= x <= 12),
            '13-24': sum(1 for x in numbers if 13 <= x <= 24),
            '25-36': sum(1 for x in numbers if 25 <= x <= 36)
        }
        
        total = sum(sectors.values())
        if total == 0:
            return np.zeros(3)
        
        return np.array([v/total for v in sectors.values()])

    def latih_dan_prediksi(self):
        try:
            if len(self.df) < self.sequence_length:
                print(f"\nBelum cukup data untuk prediksi (minimal {self.sequence_length} data).")
                return

            features_list = []
            labels_angka = []
            labels_warna = []
            
            # Membuat dataset untuk training
            for i in range(self.sequence_length, len(self.df)):
                features = self.generate_features(self.df.iloc[:i])
                if features is not None:
                    features_list.append(features[0])
                    labels_angka.append(self.df.iloc[i]['Angka'])
                    labels_warna.append(self.df.iloc[i]['Warna'])
            
            if not features_list:
                return
                
            X = np.array(features_list)
            y_angka = np.array(labels_angka)
            y_warna = np.array(labels_warna)
            
            # Split data untuk validasi
            X_train, X_val, y_train_angka, y_val_angka = train_test_split(
                X, y_angka, test_size=0.2, random_state=42
            )
            
            # Normalisasi fitur
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Training model
            self.model_angka.fit(X_train_scaled, y_train_angka)
            self.model_warna.fit(X_train_scaled, y_train_angka > 18)  # Simplified color prediction
            
            # Prediksi
            X_pred = self.generate_features(self.df)
            if X_pred is None:
                return
                
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Prediksi probabilitas untuk semua angka
            prob_angka = self.model_angka.predict_proba(X_pred_scaled)[0]
            
            # Mengambil top 5 prediksi dengan distribusi yang lebih merata
            top_indices = np.argsort(prob_angka)[-10:]  # Ambil top 10 untuk variasi
            top_probs = prob_angka[top_indices]
            
            # Normalisasi probabilitas
            top_probs = top_probs / top_probs.sum()
            
            print("\nPrediksi untuk putaran berikutnya:")
            print("\nTop 5 Angka yang mungkin keluar:")
            for idx, prob in zip(top_indices[-5:], top_probs[-5:]):
                print(f"Angka {idx}: {prob:.2%} probabilitas")
            
            # Prediksi warna berdasarkan pola historis
            last_colors = self.df['Warna'].tail(5).tolist()
            color_freq = Counter(last_colors)
            most_common_color = color_freq.most_common(1)[0][0]
            
            # Menambahkan variasi ke prediksi warna
            color_probs = {
                'Merah': 0.33 + (0.1 if most_common_color == 'Merah' else 0),
                'Hitam': 0.33 + (0.1 if most_common_color == 'Hitam' else 0),
                'Hijau': 0.34 - (0.1 if most_common_color != 'Hijau' else 0)
            }
            
            pred_warna = max(color_probs.items(), key=lambda x: x[1])
            print(f"\nPrediksi Warna: {pred_warna[0]} ({pred_warna[1]:.2%} probabilitas)")
            
            # Validasi akurasi
            if len(X_val) > 0:
                val_pred_angka = self.model_angka.predict(X_val_scaled)
                accuracy_angka = sum(val_pred_angka == y_val_angka) / len(y_val_angka)
                print(f"\nAkurasi validasi (angka): {accuracy_angka:.2%}")
            
        except Exception as e:
            print(f"Error dalam prediksi: {str(e)}")

    def hitung_akurasi_terakhir(self):
        try:
            last_10_actual = self.df['Angka'].tail(10).values
            last_10_actual_warna = self.df['Warna'].tail(10).values
            
            correct_angka = 0
            correct_warna = 0
            
            for i in range(max(0, len(self.df)-10), len(self.df)):
                if i >= self.sequence_length:
                    X = self.generate_features(self.df.iloc[:i])
                    if X is not None:
                        X_scaled = self.scaler.transform(X)
                        pred_angka = self.model_angka.predict(X_scaled)[0]
                        pred_warna = self.model_warna.predict(X_scaled)[0]
                        
                        if pred_angka == self.df.iloc[i]['Angka']:
                            correct_angka += 1
                        if pred_warna == self.df.iloc[i]['Warna']:
                            correct_warna += 1
            
            return {
                'angka': (correct_angka / 10) * 100,
                'warna': (correct_warna / 10) * 100
            }
            
        except Exception as e:
            print(f"Error dalam penghitungan akurasi: {str(e)}")
            return {'angka': 0, 'warna': 0}

def main():
    predictor = RoulettePrediction()
    print("Selamat datang di Sistem Prediksi Roulette")
    print("==========================================")
    
    while True:
        try:
            print("\nMasukkan hasil putaran (0-36) atau 'q' untuk keluar:")
            input_user = input().strip().lower()
            
            if input_user == 'q':
                print("Terima kasih telah menggunakan sistem prediksi.")
                break
            
            angka = int(input_user)
            if 0 <= angka <= 36:
                predictor.tambah_histori(angka)
            else:
                print("Angka tidak valid. Masukkan angka antara 0 dan 36.")
                
        except ValueError:
            print("Input tidak valid. Masukkan angka antara 0-36 atau 'q' untuk keluar.")
        except Exception as e:
            print(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    main()