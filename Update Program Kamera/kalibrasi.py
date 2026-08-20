"""
Kalibrasi empiris untuk estimate_distance_ground().

Cara pakai:
1. Taruh 1 orang berdiri tegak di beberapa jarak yang DIUKUR MANUAL dari kamera
   (misal: 1, 2, 3, 4, 5, 6, 7, 7.2 m — makin banyak titik, makin bagus,
   terutama padatkan titik-titik di jarak jauh karena di situ resolusi
   piksel paling kasar).
2. Untuk tiap jarak, catat nilai ymax (piksel, di resolusi 640x480 yang
   dipakai pipeline) dari bounding box yang terdeteksi. Ambil rata-rata
   dari beberapa frame biar stabil.
3. Isi DATA_KALIBRASI di bawah, lalu jalankan skrip ini.
4. Skrip akan mencari CAM_HEIGHT, CAM_PITCH_DEG, FOCAL_LENGTH_Y yang
   paling cocok dengan data lapangan Anda (least squares fit),
   sekaligus menunjukkan seberapa besar error sisa (residual).
"""

import math
import numpy as np
from scipy.optimize import curve_fit

FRAME_HEIGHT = 480

# ==== ISI DATA HASIL PENGUKURAN LAPANGAN DI SINI ====
# format: (ymax_pixel_terdeteksi, jarak_real_meter)
DATA_KALIBRASI = [
    #(570, 2.0),
    #(487, 2.5),
    (438, 3.0),
    #(385, 3.5),
    #(345, 4.0),
    #(305, 4.5),
    (281, 5.0),
    #(256, 5.5),
    #(236, 6.0),
    #(225, 6.5),
    #(206, 7.0),
    (203, 7.2),
    #(180, 8.0)
    # tambahkan titik lain, makin banyak makin baik
]
# =====================================================


def model_ymax(distance, cam_height, pitch_deg, focal_y):
    """Prediksi ymax dari jarak, dengan parameter kamera tertentu."""
    y_center = FRAME_HEIGHT / 2.0
    pitch_rad = math.radians(pitch_deg)
    theta = np.arctan(cam_height / distance)          # sudut ke titik tanah
    angle_offset = theta - pitch_rad
    delta_y = focal_y * np.tan(angle_offset)
    return y_center + delta_y


def model_ymax_wrapper(distance_arr, cam_height, pitch_deg, focal_y):
    return np.array([
        model_ymax(d, cam_height, pitch_deg, focal_y) for d in distance_arr
    ])


def fit_parameters(data):
    ymax_arr = np.array([d[0] for d in data], dtype=float)
    dist_arr = np.array([d[1] for d in data], dtype=float)

    # tebakan awal = nilai yang sekarang dipakai di detection6.py
    p0 = [1.95, 27.0, 826.33]

    popt, _ = curve_fit(
        model_ymax_wrapper, dist_arr, ymax_arr, p0=p0, maxfev=20000
    )
    cam_height, pitch_deg, focal_y = popt

    pred = model_ymax_wrapper(dist_arr, *popt)
    residual = ymax_arr - pred

    print("=== Hasil Kalibrasi ===")
    print(f"CAM_HEIGHT     = {cam_height:.4f}")
    print(f"CAM_PITCH_DEG  = {pitch_deg:.4f}")
    print(f"FOCAL_LENGTH_Y = {focal_y:.2f}")
    print()
    print("Perbandingan ymax aktual vs prediksi model baru:")
    for (y_real, d_real), y_pred, res in zip(data, pred, residual):
        print(f"  jarak={d_real:>5.1f}m  ymax_real={y_real:>6.1f}  "
              f"ymax_model={y_pred:>7.1f}  selisih={res:>6.1f}px")

    return cam_height, pitch_deg, focal_y


if __name__ == "__main__":
    if len(DATA_KALIBRASI) < 3:
        print("Isi minimal 3 titik data (idealnya 6-8, sebar dari dekat ke jauh) "
              "di DATA_KALIBRASI sebelum menjalankan skrip ini.")
    else:
        fit_parameters(DATA_KALIBRASI)