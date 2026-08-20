import os
import threading
import time

# Sembunyikan pesan welcome dari Pygame saat disimpor
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame


class AudioAlertManager:

    def __init__(
        self, sound_path: str = "awas.wav", cooldown_seconds: float = 2.5
    ):
        """Manager Audio Peringatan (Non-blocking & Cooldown).

        :param sound_path: Path file audio (.wav/.mp3)
        :param cooldown_seconds: Jeda minimal antarsuara diputar (detik)
        """
        self.sound_path = sound_path
        self.cooldown = cooldown_seconds
        self.last_played = 0.0
        self.is_initialized = False

        self._init_audio()

    def _init_audio(self):
        try:
            pygame.mixer.init()
            if os.path.exists(self.sound_path):
                self.sound = pygame.mixer.Sound(self.sound_path)
                self.is_initialized = True
                print(
                    f"[AUDIO] Audio Alert initialized using file: {self.sound_path}"
                )
            else:
                print(
                    f"[AUDIO WARNING] File audio '{self.sound_path}' tidak ditemukan!"
                )
        except Exception as e:
            print(f"[AUDIO ERROR] Gagal menginisialisasi audio mixer: {e}")

    def play_alert(self):
        """Memutar suara peringatan pada background thread jika cooldown terpenuhi."""
        if not self.is_initialized:
            return

        current_time = time.time()
        # Cek cooldown waktu agar audio tidak beruntun berlebihan
        if current_time - self.last_played >= self.cooldown:
            self.last_played = current_time
            threading.Thread(target=self._play_thread, daemon=True).start()

    def _play_thread(self):
        try:
            self.sound.play()
        except Exception as e:
            print(f"[AUDIO ERROR] Error saat memutar audio: {e}")