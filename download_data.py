"""Download the research dataset once. Run with: python download_data.py"""

from colabeling.data import download_data

if __name__ == "__main__":
    download_data()
    print("BBCSport is ready. The source data is ignored by Git.")
