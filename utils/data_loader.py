"""
Data loading and preprocessing module for CUAD dataset.
"""

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import pandas as pd

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


@dataclass
class ContractData:
    """Single contract with clauses."""
    contract_id: str
    text: str
    clauses: Dict[str, bool]  # clause_type -> is_present
    file_path: Optional[str] = None


def load_cuad_dataset(
    dataset_path: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None
) -> List[ContractData]:
    """Load CUAD v1 from data/CUAD_v1.json or Hugging Face, or a local fallback.

    CUAD ships as one SQuAD 2.0 file with no splits, so contracts are split
    80/10/10 into train, validation, and test by a stable hash of the title.
    A clause type counts as present when its category has an answer span.

    Args:
        dataset_path: Optional local directory for the fallback loader
        split: Dataset split ('train', 'validation', 'test')
        max_samples: Maximum number of contracts to load

    Returns:
        List of ContractData objects
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    try:
        path = _cuad_json_path()
    except Exception as e:
        logger.error(f"Could not fetch CUAD from Hugging Face: {e}")
        logger.info("Attempting to load from local path...")
        return load_local_dataset(dataset_path, split, max_samples)

    contracts = parse_cuad_json(path, split, max_samples)
    logger.info(f"Loaded {len(contracts)} contracts from {split} split")
    return contracts


def _cuad_json_path() -> str:
    """Return a local copy of CUAD_v1.json, downloading it if needed."""
    local = os.path.join(config.paths.data_dir, "CUAD_v1.json")
    if os.path.exists(local):
        return local

    from huggingface_hub import hf_hub_download
    logger.info(f"Downloading {config.data.dataset_file} from {config.data.dataset_repo}...")
    return hf_hub_download(
        repo_id=config.data.dataset_repo,
        filename=config.data.dataset_file,
        repo_type="dataset",
    )


def split_of(title: str) -> str:
    """Assign a contract to a split by a stable hash of its title."""
    bucket = int(hashlib.sha256(title.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def parse_cuad_json(
    path: str,
    split: str,
    max_samples: Optional[int] = None,
    clause_types: Optional[List[str]] = None,
) -> List[ContractData]:
    """Parse a CUAD SQuAD 2.0 file into ContractData for one split.

    Each question id ends in "__<category>".  Clause types are matched to
    categories ignoring case, and an unknown clause type raises ValueError
    rather than silently labelling every contract as absent.
    """
    if clause_types is None:
        clause_types = config.data.clause_types

    with open(path, encoding="utf-8") as fh:
        documents = json.load(fh)["data"]

    docs = sorted(
        (d for d in documents if split_of(d["title"]) == split),
        key=lambda d: d["title"],
    )
    if max_samples:
        docs = docs[:max_samples]

    parsed = []
    categories = set()
    for doc in docs:
        texts = []
        present: Dict[str, bool] = {}
        for paragraph in doc["paragraphs"]:
            texts.append(paragraph["context"])
            for qa in paragraph["qas"]:
                category = qa["id"].rsplit("__", 1)[-1]
                answered = bool(qa.get("answers")) and not qa.get("is_impossible", False)
                present[category] = present.get(category, False) or answered
        categories.update(present)
        parsed.append((doc["title"], "\n".join(texts), present))

    if not parsed:
        return []

    by_key = {c.casefold(): c for c in categories}
    unknown = [ct for ct in clause_types if ct.casefold() not in by_key]
    if unknown:
        raise ValueError(
            f"Not CUAD categories: {unknown}.  Valid categories: {sorted(categories)}"
        )

    return [
        ContractData(
            contract_id=title,
            text=text,
            clauses={ct: present.get(by_key[ct.casefold()], False) for ct in clause_types},
        )
        for title, text, present in parsed
    ]


def load_local_dataset(
    data_dir: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None
) -> List[ContractData]:
    """Load CUAD dataset from local directory.
    
    Args:
        data_dir: Path to local dataset directory
        split: Dataset split
        max_samples: Maximum number of samples
        
    Returns:
        List of ContractData objects
    """
    if data_dir is None:
        data_dir = config.paths.data_dir
    
    # Check for various CUAD file formats
    csv_path = os.path.join(data_dir, f"{split}.csv")
    json_path = os.path.join(data_dir, f"{split}.json")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return parse_dataframe(df, split, max_samples)
    elif os.path.exists(json_path):
        df = pd.read_json(json_path)
        return parse_dataframe(df, split, max_samples)
    else:
        # Look for any data files
        data_files = [f for f in os.listdir(data_dir) 
                      if f.endswith(('.csv', '.json', '.parquet'))]
        if data_files:
            logger.info(f"Found data files: {data_files}")
            # Try first file
            first_file = os.path.join(data_dir, data_files[0])
            if first_file.endswith('.csv'):
                df = pd.read_csv(first_file)
            elif first_file.endswith('.json'):
                df = pd.read_json(first_file)
            elif first_file.endswith('.parquet'):
                df = pd.read_parquet(first_file)
            return parse_dataframe(df, split, max_samples)
    
    logger.error("No valid data files found")
    return []


def parse_dataframe(df: pd.DataFrame, split: str, max_samples: Optional[int]) -> List[ContractData]:
    """Parse DataFrame into ContractData objects."""
    if max_samples:
        df = df.head(max_samples)
    
    contracts = []
    for i, row in df.iterrows():
        # Extract text - try various column names
        text = None
        for col in ['text', 'contract_text', 'content', 'document', 'full_text']:
            if col in df.columns:
                text = row[col]
                break
        
        if text is None:
            # Combine text columns if available
            text_parts = []
            for col in df.columns:
                if 'text' in col.lower() or 'content' in col.lower():
                    text_parts.append(str(row[col]))
            text = " ".join(text_parts)
        
        # Extract clause information
        clauses = extract_clauses_from_row(row)
        
        contract_id = row.get('contract_id', f"local_{split}_{i}")
        if isinstance(contract_id, float) and pd.isna(contract_id):
            contract_id = f"local_{split}_{i}"
        
        contract = ContractData(
            contract_id=str(contract_id),
            text=str(text)[:10000],  # Limit text length
            clauses=clauses
        )
        contracts.append(contract)
    
    logger.info(f"Parsed {len(contracts)} contracts from DataFrame")
    return contracts


def extract_clauses_from_row(row) -> Dict[str, bool]:
    """Alternative clause extraction from row."""
    clauses = {}
    
    for clause_type in config.data.clause_types:
        # Try different naming conventions
        possible_names = [
            clause_type.lower().replace(" ", "_"),
            clause_type.lower().replace(" ", ""),
            clause_type.replace(" ", "_").lower(),
            clause_type.lower(),
        ]
        
        for name in possible_names:
            # Look for exact match or partial match
            for col in row.index:
                if name in col.lower():
                    value = row[col]
                    clauses[clause_type] = bool(value) if pd.notna(value) else False
                    break
            else:
                continue
            break
        else:
            clauses[clause_type] = False
    
    return clauses


def preprocess_data(
    contracts: List[ContractData],
    clause_types: Optional[List[str]] = None,
    max_length: int = 512
) -> Tuple[List[str], List[int]]:
    """Preprocess contracts for classification.
    
    Args:
        contracts: List of ContractData objects
        clause_types: List of clause types to classify (None for all)
        max_length: Maximum text length
        
    Returns:
        Tuple of (texts, labels) where labels are binary (0/1)
    """
    if clause_types is None:
        clause_types = config.data.clause_types
    
    texts = []
    labels = []
    
    for contract in contracts:
        # Truncate text if needed
        text = contract.text[:max_length]
        
        for clause_type in clause_types:
            is_present = contract.clauses.get(clause_type, False)
            texts.append(text)
            labels.append(1 if is_present else 0)
    
    logger.info(f"Preprocessed {len(texts)} samples for {len(clause_types)} clause types")
    return texts, labels


def get_clause_distribution(contracts: List[ContractData]) -> pd.DataFrame:
    """Get distribution of clause types across contracts.
    
    Args:
        contracts: List of ContractData objects
        
    Returns:
        DataFrame with clause type distribution
    """
    distribution = {}
    
    for contract in contracts:
        for clause_type, is_present in contract.clauses.items():
            if clause_type not in distribution:
                distribution[clause_type] = {"present": 0, "absent": 0}
            if is_present:
                distribution[clause_type]["present"] += 1
            else:
                distribution[clause_type]["absent"] += 1
    
    df = pd.DataFrame(distribution).T
    df["total"] = df["present"] + df["absent"]
    df["presence_rate"] = df["present"] / df["total"]
    
    return df.sort_values("presence_rate", ascending=False)
