"""
Download NASA C-MAPSS dataset.

Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
Alternative: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

The dataset is also available on Kaggle:
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
"""

import os
import sys
import zipfile
from pathlib import Path


def download_cmapss(output_dir: str = "data/raw") -> None:
    """Download and extract C-MAPSS dataset.

    If automatic download fails, prints manual instructions.
    """
    out_path = Path(output_dir)
    cmapss_dir = out_path / "CMAPSSData"

    # Check if already downloaded
    if cmapss_dir.exists() and any(cmapss_dir.glob("*.txt")):
        n_files = len(list(cmapss_dir.glob("*.txt")))
        print(f"C-MAPSS data already exists at {cmapss_dir} ({n_files} files)")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    # Try downloading from NASA
    url = "https://data.nasa.gov/api/views/xaut-bemq/files/08adac93-7e41-4c4a-94ed-dc1e42f6a43b?download=true&filename=CMAPSSData.zip"

    print(f"Downloading C-MAPSS from NASA Open Data Portal...")
    print(f"URL: {url}")

    try:
        import urllib.request
        zip_path = out_path / "CMAPSSData.zip"
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_path)
        print(f"Extracted to {out_path}")

        # Cleanup zip
        zip_path.unlink()

    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print("\n--- MANUAL DOWNLOAD INSTRUCTIONS ---")
        print("1. Go to: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data")
        print("   Or Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print(f"2. Download and extract to: {cmapss_dir}/")
        print("3. Verify these files exist:")
        for subset in ["FD001", "FD002", "FD003", "FD004"]:
            print(f"   - train_{subset}.txt")
            print(f"   - test_{subset}.txt")
            print(f"   - RUL_{subset}.txt")
        sys.exit(1)

    # Verify
    expected_files = []
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        expected_files.extend([
            f"train_{subset}.txt",
            f"test_{subset}.txt",
            f"RUL_{subset}.txt",
        ])

    missing = [f for f in expected_files if not (cmapss_dir / f).exists()]
    if missing:
        print(f"WARNING: Missing files: {missing}")
    else:
        print(f"All 12 data files verified in {cmapss_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    download_cmapss(args.output_dir)
