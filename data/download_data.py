from huggingface_hub import hf_hub_download
from pathlib import Path
import tarfile
import os

REPO = "HarryMaguire1993/ebh"
REPO_TYPE = "dataset"

SUBFOLDER = "ebh/bondi-100"
TARFILE = "3d-h5-sphere.tar.gz"
INTERNAL_ROOT = "3d-h5-sphere"

PSCRATCH = Path(os.environ["PSCRATCH"])
DEST_DIR = PSCRATCH / "mhd" / "bondi-100"

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    tar_path = hf_hub_download(
        repo_id=REPO,
        repo_type=REPO_TYPE,
        subfolder=SUBFOLDER,
        filename=TARFILE,
        cache_dir=PSCRATCH / "hf_cache",
        resume_download=True,
    )

    print("downloaded:", tar_path)

    if not (DEST_DIR / INTERNAL_ROOT).exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=DEST_DIR)
        print("extracted to:", DEST_DIR)

if __name__ == "__main__":
    main()
