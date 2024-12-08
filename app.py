from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load model dari file pickle
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Inisialisasi Flask
app = Flask(__name__)

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form dan konversi ke float
        data = [float(x) for x in request.form.values()]
        
        # Pastikan jumlah input sesuai fitur model
        if len(data) != 13:  # Misalnya model memiliki 13 fitur
            raise ValueError("Jumlah input tidak sesuai dengan fitur model.")
        
        # Konversi ke array NumPy
        features = np.array([data])
        
        # Prediksi menggunakan model
        prediction = model.predict(features)
        output = "Ya" if prediction[0] == 1 else "Tidak"
        
        # Kirim hasil prediksi ke halaman HTML
        return render_template('index.html', prediction_text=f'Risiko Serangan Jantung: {output}')
    except ValueError as ve:
        return render_template('index.html', prediction_text=f'Error: {str(ve)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Terjadi kesalahan: {str(e)}')

# Jalankan Flask
if __name__ == '__main__':
    app.run(debug=True)
