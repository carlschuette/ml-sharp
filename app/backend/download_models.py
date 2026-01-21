import logging
import requests
from pathlib import Path
from tqdm import tqdm
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger("ModelDownloader")

# Constants (matching main.py)
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "sharp_model.pt"

def download_file(url, filename):
    """
    Download a file with a progress bar.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        # tqm context manager
        with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {filename.name}", ncols=80) as progress_bar:
            with open(filename, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        
        return True
    except Exception as e:
        LOGGER.error(f"Failed to download {url}: {e}")
        # Clean up partial download
        if filename.exists():
            filename.unlink()
        return False

def main():
    # Ensure checkpoint directory exists
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    if CHECKPOINT_PATH.exists():
        LOGGER.info(f"Model found at {CHECKPOINT_PATH}")
        return

    LOGGER.info(f"Model not found at {CHECKPOINT_PATH}. Downloading from {DEFAULT_MODEL_URL}...")
    success = download_file(DEFAULT_MODEL_URL, CHECKPOINT_PATH)
    
    if not success:
        LOGGER.error("Failed to download model.")
        sys.exit(1)
        
    LOGGER.info("Model download complete.")

if __name__ == "__main__":
    main()
