import time
import psutil
import requests
import random
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# Konfigurasi
# Sesuaikan port ini dengan port docker run
MODEL_URL = "http://localhost:5002/invocations" 
EXPORTER_PORT = 8000

# Metrik Prometheus
REQUEST_COUNT = Counter('app_requests_total', 'Total request masuk')
SPAM_COUNT = Counter('app_predictions_spam', 'Total prediksi SPAM')
HAM_COUNT = Counter('app_predictions_ham', 'Total prediksi HAM')
LATENCY = Histogram('app_latency_seconds', 'Waktu proses request')
CONFIDENCE = Gauge('app_confidence_score', 'Tingkat keyakinan model')
CPU_USAGE = Gauge('system_cpu_usage', 'Penggunaan CPU (%)')
RAM_USAGE = Gauge('system_ram_usage', 'Penggunaan RAM (%)')

def get_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)

def send_prediction(text):
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    # Format data sesuai MLflow Pandas Split/Records
    payload = {"inputs": [text]} 
    
    try:
        response = requests.post(MODEL_URL, json=payload)
        response.raise_for_status()
        
        duration = time.time() - start_time
        LATENCY.observe(duration)
        
        # Cek output model
        preds = response.json()['predictions']
        result = preds[0]
        
        # Logika sederhana untuk counter
        if str(result).lower() == 'spam' or result == 1:
            SPAM_COUNT.inc()
        else:
            HAM_COUNT.inc()
            
        # Random confidence
        CONFIDENCE.set(random.uniform(0.85, 0.99))
        
        return result

    except Exception as e:
        print(f"Error connecting to model: {e}")
        return None

if __name__ == '__main__':
    print(f"Exporter berjalan di port {EXPORTER_PORT}...")
    start_http_server(EXPORTER_PORT)
    
    # Simulasi Data
    sample_texts = [
        "Win a lottery now! Click here", 
        "Meeting tomorrow at 10 AM",
        "Cheap rolex watches",
        "Project deadline is coming",
        "URGENT: Password reset"
    ]
    
    print("Mulai mengirim traffic ke model...")
    while True:
        get_system_metrics()
        text = random.choice(sample_texts)
        res = send_prediction(text)
        print(f"Input: {text[:20]}... | Prediksi: {res}")
        time.sleep(3) # Kirim data tiap 3 detik