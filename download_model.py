"""
FallGuard AI — Model Downloader

Downloads the MediaPipe Pose Landmarker FULL model
to the project folder.
"""

import os
import sys
import urllib.request

# Project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
SAVE_PATH = os.path.join(PROJECT_DIR, "pose_landmarker.task")

# Full model URL
MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

# Minimum size to consider the download valid
# Full model is around 9 MB, so use a conservative threshold
MIN_SIZE_MB = 5


def is_valid(path):
    """Check if the model file exists and is large enough."""

    if not os.path.exists(path):
        return False

    size = os.path.getsize(path) / 1_000_000

    if size < MIN_SIZE_MB:
        print(
            f"  ⚠ Model file too small "
            f"({size:.1f} MB)"
        )
        return False

    return True


def download_model():

    print("=" * 58)
    print("  FallGuard AI — Model Downloader")
    print("=" * 58)

    print(f"\n  Model path:")
    print(f"  {SAVE_PATH}")

    # Check existing model
    if is_valid(SAVE_PATH):

        size = os.path.getsize(SAVE_PATH) / 1_000_000

        print(
            f"\n  ✓ Full Pose Landmarker model already exists "
            f"({size:.1f} MB)"
        )

        print("\n  No download required.")
        print("  Run: python run.py")

        return True

    # Remove invalid model
    if os.path.exists(SAVE_PATH):

        print("\n  Removing invalid model...")
        os.remove(SAVE_PATH)

    # Download
    print("\n  Downloading FULL Pose Landmarker model...")
    print(f"  URL: {MODEL_URL}")

    try:

        req = urllib.request.Request(
            MODEL_URL,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            req,
            timeout=120
        ) as response:

            total = int(
                response.headers.get(
                    "Content-Length",
                    0
                )
            )

            downloaded = 0

            with open(
                SAVE_PATH,
                "wb"
            ) as file:

                while True:

                    chunk = response.read(
                        64 * 1024
                    )

                    if not chunk:
                        break

                    file.write(chunk)

                    downloaded += len(chunk)

                    if total:

                        percentage = (
                            downloaded
                            / total
                            * 100
                        )

                        print(
                            f"\r  Downloading: "
                            f"{percentage:.1f}%",
                            end="",
                            flush=True
                        )

        print()

    except Exception as e:

        print(
            f"\n  ✗ Download failed: {e}"
        )

        return False

    # Verify download
    if is_valid(SAVE_PATH):

        size = (
            os.path.getsize(SAVE_PATH)
            / 1_000_000
        )

        print(
            f"\n  ✓ Full model downloaded "
            f"successfully ({size:.1f} MB)"
        )

        print("\n  Run: python run.py")

        return True

    print(
        "\n  ✗ Downloaded model failed validation."
    )

    return False


if __name__ == "__main__":

    success = download_model()

    input(
        "\n  Press Enter to exit..."
    )

    sys.exit(
        0 if success else 1
    )