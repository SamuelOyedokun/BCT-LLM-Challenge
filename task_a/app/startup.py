import os
import gdown

FILES = {
    "businesses.csv":         "1lhSSza1NHyPymjEFlhpt0eyThiSMr3QC",
    "tips_with_business.csv": "1H0b2O20gFcpDUNAS9P-9k08Xn_MsSMpO",
    "user_profiles.csv":      "1cw57UALFzT3pTZl6zul1yH1FO57k9uLN",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_all():
    print("Checking data files...")
    for filename, file_id in FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024*1024:
            size = os.path.getsize(dest) / (1024*1024)
            print(f"   ✅ {filename} already exists — {size:.1f} MB")
        else:
            print(f"   Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest, quiet=False, fuzzy=True)
            size = os.path.getsize(dest) / (1024*1024)
            print(f"   ✅ {filename} — {size:.1f} MB")
    print("All data files ready.")

if __name__ == "__main__":
    download_all()
