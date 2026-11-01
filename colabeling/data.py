"""Original BBCSport features, safe downloads, and strictly separated data splits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from scipy import sparse
from scipy.io import mmread
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["athletics", "cricket", "football", "rugby", "tennis"]
DATASET_PAGE = "https://derekgreene.com/bbc/"
ARCHIVES = {
    "bbcsport.zip": "f41c3d3c2b39c91af60205efa08933d3f19e6986074d4686393dfced4d50a0a9",
    "bbcsport-fulltext.zip": "3c9a8cec47cc153ae1b05de1a995e224e4ccfb0aa644d9f1334ecc155b623561",
}


def download_data(data_dir: Path = ROOT / "data") -> None:
    """Download/check the creator's archives; source text stays outside Git."""
    downloads = data_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        destination = downloads / name
        if not destination.exists():
            print(f"Downloading {name} from the dataset creator...", flush=True)
            request = Request(
                f"https://derekgreene.com/datasets/{name}",
                headers={"User-Agent": "BBCSport-course-project/1.0"},
            )
            with urlopen(request, timeout=90) as response:
                payload = response.read(20_000_001)
            if len(payload) > 20_000_000:
                raise ValueError("Dataset archive exceeds the expected size limit.")
            if sha256(payload).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch for {name}; no file was installed.")
            temporary = destination.with_suffix(".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
        if sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {destination}; download it again.")
        target = data_dir / ("features" if name == "bbcsport.zip" else "raw")
        target.mkdir(parents=True, exist_ok=True)
        with ZipFile(destination) as archive:
            if sum(info.file_size for info in archive.infolist()) > 50_000_000:
                raise ValueError("Unexpected uncompressed dataset size.")
            for info in archive.infolist():
                output = (target / info.filename).resolve()
                if not output.is_relative_to(target.resolve()):
                    raise ValueError("Unsafe path in dataset archive.")
                if info.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if not output.exists():
                        output.write_bytes(archive.read(info))


@dataclass
class BBCSport:
    counts: sparse.csr_matrix
    labels: np.ndarray
    terms: list[str]
    document_ids: list[str]
    titles: list[str]


def load_data(data_dir: Path = ROOT / "data") -> BBCSport:
    features = data_dir / "features"
    if not (features / "bbcsport.mtx").exists():
        raise FileNotFoundError("Run: python download_data.py")
    counts = sparse.csr_matrix(mmread(features / "bbcsport.mtx", spmatrix=True).T, dtype=np.float64)
    labels = np.loadtxt(features / "bbcsport.classes", comments="%", dtype=int)[:, 1]
    document_ids = (features / "bbcsport.docs").read_text().splitlines()
    terms = (features / "bbcsport.terms").read_text().splitlines()
    if counts.shape != (737, 4613) or len(labels) != 737:
        raise ValueError("The BBCSport feature dimensions do not match the original dataset.")
    if not np.array_equal(np.unique(labels), np.arange(5)) or np.any(counts.data < 0):
        raise ValueError("Invalid classes or term frequencies.")
    titles = []
    for document_id in document_ids:
        category, number = document_id.split(".")
        path = data_dir / "raw" / "bbcsport" / category / f"{number}.txt"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
        titles.append(lines[0].strip() if lines else f"Article {len(titles) + 1}")
    return BBCSport(counts, labels, terms, document_ids, titles)


@dataclass
class DataSplit:
    train: np.ndarray
    test: np.ndarray
    known_mask: np.ndarray
    known_classes: np.ndarray
    word_indices: tuple[np.ndarray, np.ndarray]
    train_views: tuple[np.ndarray, np.ndarray]
    test_views: tuple[np.ndarray, np.ndarray]


def make_split(dataset: BBCSport, seed: int, labels_per_class: int = 2) -> DataSplit:
    """Ceil-half per class gives 370 training / 367 test articles, as in Table 2.

    Full labels are used here only to construct the stratified benchmark split
    and select the requested labeled examples. Learners receive known_classes,
    with -1 for every remaining training answer.
    """
    if labels_per_class < 1:
        raise ValueError("labels_per_class must be positive.")
    split_rng, words_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    training, testing, known = [], [], []
    for category in range(5):
        shuffled = split_rng.permutation(np.flatnonzero(dataset.labels == category))
        midpoint = (len(shuffled) + 1) // 2
        if labels_per_class > midpoint:
            raise ValueError("Too many requested labels for the training split.")
        training.extend(shuffled[:midpoint])
        testing.extend(shuffled[midpoint:])
        known.extend(shuffled[:labels_per_class])
    train, test = np.sort(training), np.sort(testing)
    known_mask = np.isin(train, known)
    known_classes = np.full(len(train), -1, dtype=int)
    known_classes[known_mask] = dataset.labels[train[known_mask]]
    vocabulary = words_rng.permutation(dataset.counts.shape[1])
    cut = len(vocabulary) // 2
    indices = (np.sort(vocabulary[:cut]), np.sort(vocabulary[cut:]))
    views = tuple(normalize(dataset.counts[:, words], norm="l1").toarray() for words in indices)
    return DataSplit(
        train, test, known_mask, known_classes, indices,
        tuple(view[train] for view in views), tuple(view[test] for view in views),
    )
