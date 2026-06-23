"""
DEEPTRUST — Model Auto-Downloader
Runs at app startup on Streamlit Cloud to download model files from Google Drive.
On your local laptop, this is skipped because the models/ folder already exists.
"""

import os
import gdown

MODEL_DIR = "models"

# Google Drive file IDs 
GOOGLE_DRIVE_IDS = {
    "deeptrust_spotter_v2_final.keras": "1fTyQbF0QoDBeaPFByvjVRZwGjHXEkzgx",
    "deeptrust_seq_ae.keras":           "1_fva2iyObUNglA3b5JCbtUs4DI_JQ3Ft",
    "deeptrust_meta_seq_cpu.joblib":    "17YZ8UwFtNGT16CA7PN0KtNgsxgCYNygv",
    "deeptrust_scaler_seq_cpu.joblib":  "1NsM4WODawIZfh9Rh1Ihvk6iGT5UgGDQH",
    "feat_min.npy":                     "1byUsak_0MO4xRlUXOjxKnHqi42ZdFeDy",
    "feat_max.npy":                     "1vsffmkA4ALjjCgprxBVRlFpqvc_Zw7Vy",
}


def download_models():
    """
    Downloads all missing model files from Google Drive.
    Called once at app startup before load_models() runs.
    
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    all_present = True
    for filename, file_id in GOOGLE_DRIVE_IDS.items():
        filepath = os.path.join(MODEL_DIR, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"[OK] {filename} already present")
            continue

        # Download from Google Drive
        print(f"[DOWNLOAD] {filename} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, filepath, quiet=False)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"[DONE] {filename} ({size_mb:.1f} MB)")
            else:
                print(f"[FAIL] {filename} download failed")
                all_present = False
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            all_present = False

    return all_present


if __name__ == "__main__":
    print("DEEPTRUST Model Downloader")
    print("=" * 40)
    result = download_models()
    print("=" * 40)
    print("All models ready." if result else "Some models missing.")
