"""SGF downloader for dohungo.

Downloads professional Go game records from u-go.net/gamerecords/ and caches them locally.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin

import requests
from tqdm import tqdm


def download_kgs_index(
    data_dir: Path = Path("data/raw"),
    board_size: int = 19,
    max_games: int = 1000,
) -> list[Path]:
    """Download SGF files from u-go.net gamerecords.
    
    Args:
        data_dir: Directory to save downloaded files.
        board_size: Board size (9 or 19).
        max_games: Maximum number of games to download.
        
    Returns:
        List of paths to downloaded SGF files.
        
    Raises:
        ValueError: If board_size is not 9 or 19.
        requests.RequestException: If download fails.
    """
    if board_size not in (9, 19):
        raise ValueError(f"board_size must be 9 or 19, got {board_size}")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    sgf_dir = data_dir / f"sgf_{board_size}x{board_size}"
    sgf_dir.mkdir(exist_ok=True)
    
    # u-go.net has different URLs for different board sizes
    base_url = "https://u-go.net/gamerecords/"
    if board_size == 19:
        index_url = urljoin(base_url, "")
    else:
        index_url = urljoin(base_url, "9x9/")
    
    print(f"Downloading SGF games from {index_url}")
    
    # Get the index page to find available SGF files
    try:
        response = requests.get(index_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch index page: {e}")
    
    # Extract SGF file links from the HTML
    sgf_links = re.findall(r'href="([^"]*\.sgf)"', response.text)
    if not sgf_links:
        # Try ZIP files containing SGFs
        zip_links = re.findall(r'href="([^"]*\.zip)"', response.text)
        sgf_links = zip_links[:max_games]
    
    downloaded_files = []
    
    for i, sgf_link in enumerate(tqdm(sgf_links[:max_games], desc="Downloading")):
        sgf_url = urljoin(index_url, sgf_link)
        filename = Path(sgf_link).name
        local_path = sgf_dir / filename
        
        # Skip if already exists
        if local_path.exists():
            downloaded_files.append(local_path)
            continue
            
        try:
            sgf_response = requests.get(sgf_url, timeout=30)
            sgf_response.raise_for_status()
            
            with open(local_path, "wb") as f:
                f.write(sgf_response.content)
            
            # If it's a ZIP file, extract SGFs from it
            if local_path.suffix.lower() == ".zip":
                extracted_files = _extract_sgf_from_zip(local_path, sgf_dir)
                downloaded_files.extend(extracted_files)
                local_path.unlink()  # Remove the ZIP file
            else:
                downloaded_files.append(local_path)
                
        except requests.RequestException as e:
            print(f"Failed to download {sgf_url}: {e}")
            continue
    
    print(f"Downloaded {len(downloaded_files)} SGF files to {sgf_dir}")
    return downloaded_files


def _extract_sgf_from_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    """Extract SGF files from a ZIP archive.
    
    Args:
        zip_path: Path to the ZIP file.
        extract_dir: Directory to extract files to.
        
    Returns:
        List of extracted SGF file paths.
    """
    extracted_files = []
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if member.lower().endswith(".sgf"):
                    # Extract to the target directory
                    extracted_path = extract_dir / Path(member).name
                    with open(extracted_path, "wb") as f:
                        f.write(zip_ref.read(member))
                    extracted_files.append(extracted_path)
    except zipfile.BadZipFile:
        print(f"Warning: {zip_path} is not a valid ZIP file")
    
    return extracted_files


def iter_sgf_files(data_dir: Path = Path("data/raw"), board_size: int = 19) -> Generator[Path, None, None]:
    """Iterate over downloaded SGF files.
    
    Args:
        data_dir: Directory containing SGF files.
        board_size: Board size to filter by.
        
    Yields:
        Path objects for each SGF file.
    """
    sgf_dir = data_dir / f"sgf_{board_size}x{board_size}"
    if not sgf_dir.exists():
        return
    
    for sgf_file in sgf_dir.glob("*.sgf"):
        yield sgf_file


if __name__ == "__main__":
    # Download a small sample for testing
    files = download_kgs_index(max_games=10)
    print(f"Downloaded {len(files)} files for testing") 