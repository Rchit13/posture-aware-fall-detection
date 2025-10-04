import os
import requests
from pyunpack import Archive
import shutil
import time

# === Required Setup ===
RAR_URLS = [
    "https://dataverse.harvard.edu/api/access/datafile/10857218",
    "https://dataverse.harvard.edu/api/access/datafile/10857222",
    "https://dataverse.harvard.edu/api/access/datafile/10857223",
    "https://dataverse.harvard.edu/api/access/datafile/10857224",
    "https://dataverse.harvard.edu/api/access/datafile/10857225",
    "https://dataverse.harvard.edu/api/access/datafile/10857226",
    "https://dataverse.harvard.edu/api/access/datafile/10857227",
    "https://dataverse.harvard.edu/api/access/datafile/10857228",
    "https://dataverse.harvard.edu/api/access/datafile/10857229",
    "https://dataverse.harvard.edu/api/access/datafile/10857230",
    "https://dataverse.harvard.edu/api/access/datafile/10857231",
    "https://dataverse.harvard.edu/api/access/datafile/10857232",
    "https://dataverse.harvard.edu/api/access/datafile/10857233",
    "https://dataverse.harvard.edu/api/access/datafile/10857234",
    "https://dataverse.harvard.edu/api/access/datafile/10857235",
    "https://dataverse.harvard.edu/api/access/datafile/10857236",
    "https://dataverse.harvard.edu/api/access/datafile/10857237",
    "https://dataverse.harvard.edu/api/access/datafile/10857238",
    "https://dataverse.harvard.edu/api/access/datafile/10857239",
    "https://dataverse.harvard.edu/api/access/datafile/10857240"
]

BASE_DIR = "./data"
RAR_DIR = os.path.join(BASE_DIR, "harvard_rars")
EXTRACTED_DIR = os.path.join(BASE_DIR, "harvard_raw")
FINAL_CSV_DIR = os.path.join(BASE_DIR, "all_keypoints_csv")

os.makedirs(RAR_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(FINAL_CSV_DIR, exist_ok=True)

# === Download and Extract One File ===
def download_and_extract(url):
    filename = url.split("/")[-1] + ".rar"
    rar_path = os.path.join(RAR_DIR, filename)

    headers = {"User-Agent": "Mozilla/5.0"}
    print(f"📥 Downloading {filename}...")

    response = requests.get(url, headers=headers, stream=True)
    if response.status_code != 200:
        print(f"❌ Failed to download {url} (Status {response.status_code})")
        return None

    with open(rar_path, 'wb') as f:
        f.write(response.content)

    extract_to = os.path.join(EXTRACTED_DIR, filename.replace(".rar", ""))
    os.makedirs(extract_to, exist_ok=True)

    print(f"📦 Extracting to {extract_to}...")
    try:
        Archive(rar_path).extractall(extract_to)
    except Exception as e:
        print(f"❌ Extraction failed for {filename}: {e}")
        return None

    return extract_to

# === Run on All URLs ===
all_csv_folders = []
for url in RAR_URLS:
    folder = download_and_extract(url)
    if folder:
        all_csv_folders.append(folder)
    time.sleep(1)  # polite delay

# === Consolidate CSVs ===
print("\n📁 Moving CSVs to:", FINAL_CSV_DIR)
count = 0
for folder in all_csv_folders:
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            unique_name = f"{os.path.basename(folder)}__{file}"
            shutil.move(os.path.join(folder, file), os.path.join(FINAL_CSV_DIR, unique_name))
            count += 1

print(f"\n✅ {count} CSV files consolidated in: {FINAL_CSV_DIR}")
