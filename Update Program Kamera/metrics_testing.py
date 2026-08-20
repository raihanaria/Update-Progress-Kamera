import time
import csv
import os
import psutil
import subprocess
import threading
import urllib.parse

class SystemTester:
    def __init__(self, log_file="log/durability_network_test.csv", interval_sec=5):
        self.log_file = log_file
        self.interval = interval_sec
        self.is_running = False
        self.thread = None
        self.camera_ip = None

        # Format header CSV
        self.headers = [
            "timestamp", 
            "cpu_temp_c", 
            "cpu_usage_pct", 
            "ram_usage_pct", 
            "ram_used_mb", 
            "fps", 
            "network_status", 
            "ping_latency_ms"
        ]
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def extract_ip_from_rtsp(self, rtsp_url):
        """Mengekstrak IP address dari URL RTSP untuk keperluan ping test"""
        try:
            parsed = urllib.parse.urlparse(rtsp_url)
            return parsed.hostname
        except Exception:
            return None

    def get_cpu_temp(self):
        """Membaca suhu Raspberry Pi (vcgencmd)"""
        try:
            res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2)
            temp_str = res.stdout.strip().replace("temp=", "").replace("'C", "")
            return float(temp_str)
        except Exception:
            return 0.0

    def check_network_ping(self, target_ip):
        """Menguji latency & status koneksi jaringan ke Kamera RTSP (Uji Resiliensi C)"""
        if not target_ip:
            return "UNKNOWN", 0.0

        try:
            # Ping 1 packet dengan timeout 1 detik
            cmd = ["ping", "-c", "1", "-W", "1", target_ip]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            
            if res.returncode == 0:
                # Ambil time=XX ms dari output ping
                for line in res.stdout.splitlines():
                    if "time=" in line:
                        ms_str = line.split("time=")[1].split(" ")[0]
                        return "CONNECTED", float(ms_str)
                return "CONNECTED", 0.0
            else:
                return "DISCONNECTED / TIMEOUT", -1.0
        except Exception:
            return "ERROR", -1.0

    def start(self, engine_instance, rtsp_url=None):
        """Menjalankan background thread testing"""
        if self.is_running:
            return

        if rtsp_url:
            self.camera_ip = self.extract_ip_from_rtsp(rtsp_url)

        self.is_running = True
        self.thread = threading.Thread(
            target=self._test_loop, 
            args=(engine_instance,), 
            daemon=True
        )
        self.thread.start()
        print(f"[TESTER] Modul Testing Durabilitas & Jaringan AKTIF (Interval: {self.interval}s)")

    def _test_loop(self, engine):
        while self.is_running:
            try:
                # 1. Ambil telemetry dari AI Engine
                data = engine.get_data() if hasattr(engine, 'get_data') else {}
                current_fps = data.get("fps", 0.0)

                # 2. Ambil metrik Durabilitas System (Uji A)
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                cpu_temp = self.get_cpu_temp()
                cpu_usage = psutil.cpu_percent()
                ram = psutil.virtual_memory()
                ram_pct = ram.percent
                ram_mb = round(ram.used / (1024 * 1024), 2)

                # 3. Ambil metrik Resiliensi Jaringan (Uji C)
                net_status, ping_ms = self.check_network_ping(self.camera_ip)

                # 4. Tulis ke CSV
                with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        now, cpu_temp, cpu_usage, ram_pct, ram_mb,
                        current_fps, net_status, ping_ms
                    ])

            except Exception as e:
                print(f"[TESTER] Error pada logger test: {e}")

            time.sleep(self.interval)

    def stop(self):
        self.is_running = False
        print("[TESTER] Modul Testing Diberhentikan.")