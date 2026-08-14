"""
Data loading and preprocessing module for CUAD dataset.
"""

import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import pandas as pd

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Load CUAD dataset from HuggingFace or local path.
    
    Args:
        dataset_path: Optional local path to dataset
        split: Dataset split ('train', 'validation', 'test')
        max_samples: Maximum number of samples to load
        
    Returns:
        List of ContractData objects
    """
    try:
        from datasets import load_dataset
        logger.info("Loading CUAD dataset from HuggingFace...")
        
        # CUAD v2 is the main version
        dataset = load_dataset("lexnecn/contract-understanding-annotated-dataset", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        contracts = []
        for i, row in enumerate(dataset):
            contract = ContractData(
                contract_id=f"cuad_{split}_{i}",
                text=row.get("text", ""),
                clauses=extract_clauses(row),
                file_path=row.get("file_name", None)
            )
            contracts.append(contract)
        
        logger.info(f"Loaded {len(contracts)} contracts from {split} split")
        return contracts
        
    except Exception as e:
        logger.error(f"Error loading from HuggingFace: {e}")
        logger.info("Attempting to load from local path...")
        return load_local_dataset(dataset_path, split, max_samples)


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


def extract_clauses(row) -> Dict[str, bool]:
    """Extract clause annotations from a dataset row."""
    clauses = {}
    
    # CUAD dataset typically has binary labels for each clause type
    clause_columns = [col for col in row.index 
                      if any(ct in col for ct in config.data.clause_types)]
    
    for clause_type in config.data.clause_types:
        # Look for column with clause type in name
        for col in row.index:
            if clause_type.lower() in col.lower():
                value = row[col]
                clauses[clause_type] = bool(value) if pd.notna(value) else False
                break
        else:
            # Default to False if not found
            clauses[clause_type] = False
    
    return clauses


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
