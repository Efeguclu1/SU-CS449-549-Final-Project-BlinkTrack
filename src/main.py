"""
BlinkTrack - Demo Launcher
Tüm demo'ları tek yerden başlat
"""

import sys
import os

def print_banner():
    print("\n" + "="*60)
    print("""
    ____  _ _       _   _____               _    
   | __ )| (_)_ __ | | |_   _| __ __ _  ___| | __
   |  _ \\| | | '_ \\| |/ /| || '__/ _` |/ __| |/ /
   | |_) | | | | | |   < | || | | (_| | (__|   < 
   |____/|_|_|_| |_|_|\\_\\|_||_|  \\__,_|\\___|_|\\_\\
    
    Hands-Free Interaction System
    CS449 - Human Computer Interaction
    """)
    print("="*60)


def print_menu():
    print("\n  Demo Seçin:\n")
    print("  [1] Entegre Sistem Testi")
    print("      └─ Bakış + Blink + Cursor kontrolü")
    print()
    print("  [2] Fitts' Law Hedef Demo")
    print("      └─ Performans ölçümü (süre, doğruluk)")
    print()
    print("  [3] Menü Navigasyon Demo")
    print("      └─ Bakış ile menü seçimi")
    print()
    print("  [4] Sadece Blink Testi")
    print("      └─ Tek/Çift/Uzun blink algılama")
    print()
    print("  [5] Sadece Gaze Testi")
    print("      └─ Bakış yönü tespiti")
    print()
    print("  [Q] Çıkış")
    print()


def main():
    print_banner()
    
    while True:
        print_menu()
        choice = input("  Seçiminiz (1-5, Q): ").strip().lower()
        
        if choice == '1':
            print("\n  → Entegre Sistem başlatılıyor...\n")
            from blinktrack_integrated import main as run_integrated
            run_integrated()
            
        elif choice == '2':
            print("\n  → Fitts' Law Demo başlatılıyor...\n")
            from fitts_demo import main as run_fitts
            run_fitts()
            
        elif choice == '3':
            print("\n  → Menü Demo başlatılıyor...\n")
            from menu_demo import main as run_menu
            run_menu()
            
        elif choice == '4':
            print("\n  → Blink Testi başlatılıyor...\n")
            from blink_detection.blink_test import main as run_blink
            run_blink()
            
        elif choice == '5':
            print("\n  → Gaze Testi başlatılıyor...\n")
            from gaze_tracking.gaze_test import main as run_gaze
            run_gaze()
            
        elif choice == 'q':
            print("\n  Görüşmek üzere!\n")
            sys.exit(0)
            
        else:
            print("\n  ⚠ Geçersiz seçim! 1-5 veya Q girin.\n")


if __name__ == "__main__":
    main()