import os
import requests

FILES = {
    "businesses.csv":         "1lhSSza1NHyPymjEFlhpt0eyThiSMr3QC",
    "tips_with_business.csv": "1H0b2O20gFcpDUNAS9P-9k08Xn_MsSMpO",
    "user_profiles.csv":      "1cw57UALFzT3pTZl6zul1yH1FO57k9uLN",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(file_id: str, destination: str):
    print(f"   Downloading {os.path.basename(destination)}...")
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    # First request to get confirmation token
    r = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for key, value in r.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    # If large file, need confirmation token
    if token:
        r = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    # Write file in chunks
    with open(destination, "wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    size = os.path.getsize(destination) / (1024*1024)
    print(f"   ✅ {os.path.basename(destination)} — {size:.1f} MB")

def download_all():
    print("Checking data files...")
    for filename, file_id in FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            size = os.path.getsize(dest) / (1024*1024)
            print(f"   ✅ {filename} already exists — {size:.1f} MB")
        else:
            download_file(file_id, dest)
    print("All data files ready.")

if __name__ == "__main__":
    download_all()
