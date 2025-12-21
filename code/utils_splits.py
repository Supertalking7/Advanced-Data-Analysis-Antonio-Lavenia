"""
Time-based splitting utilities for Airbnb dynamic pricing project.

This module provides functions for creating time-based train/test splits
to prevent data leakage in time series prediction tasks.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datetime import datetime, date


def create_time_split(
    df: pd.DataFrame,
    test_start_date: str,
    date_col: str = 'date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create time-based train/test split.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date column
    test_start_date : str
        Cutoff date for test set (format: 'YYYY-MM-DD')
    date_col : str
        Name of the date column
        
    Returns:
    --------
    train_df, test_df : Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames
    """
    # Convert date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Convert test_start_date to datetime
    test_start = pd.to_datetime(test_start_date)
    
    # Create masks
    train_mask = df[date_col] < test_start
    test_mask = df[date_col] >= test_start
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Validation checks
    assert len(train_df) > 0, "Train set is empty"
    assert len(test_df) > 0, "Test set is empty"
    assert train_df[date_col].max() < test_df[date_col].min(), \
        "Train dates must be strictly before test dates"
    
    print(f"Train set: {len(train_df)} rows, dates {train_df[date_col].min()} to {train_df[date_col].max()}")
    print(f"Test set: {len(test_df)} rows, dates {test_df[date_col].min()} to {test_df[date_col].max()}")
    
    return train_df, test_df


def create_city_split(
    df: pd.DataFrame,
    train_cities: List[str],
    test_cities: List[str],
    city_col: str = 'city'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create cross-city validation split.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with city information
    train_cities : List[str]
        Cities to use for training
    test_cities : List[str]
        Cities to use for testing
    city_col : str
        Name of the city column
        
    Returns:
    --------
    train_df, test_df : Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames
    """
    train_mask = df[city_col].isin(train_cities)
    test_mask = df[city_col].isin(test_cities)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Validation checks
    assert len(train_df) > 0, "Train set is empty"
    assert len(test_df) > 0, "Test set is empty"
    assert set(train_cities).isdisjoint(set(test_cities)), \
        "Train and test cities must be disjoint"
    
    print(f"Train cities: {train_cities}, {len(train_df)} rows")
    print(f"Test cities: {test_cities}, {len(test_df)} rows")
    
    return train_df, test_df


def create_time_city_split(
    df: pd.DataFrame,
    test_start_date: str,
    train_cities: List[str],
    test_cities: List[str],
    date_col: str = 'date',
    city_col: str = 'city'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create combined time-based and city-based split.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date and city columns
    test_start_date : str
        Cutoff date for test set
    train_cities : List[str]
        Cities to use for training
    test_cities : List[str]
        Cities to use for testing
    date_col : str
        Name of the date column
    city_col : str
        Name of the city column
        
    Returns:
    --------
    train_df, test_df : Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames
    """
    # Convert date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    test_start = pd.to_datetime(test_start_date)
    
    # Create combined masks
    train_mask = (df[date_col] < test_start) & (df[city_col].isin(train_cities))
    test_mask = (df[date_col] >= test_start) & (df[city_col].isin(test_cities))
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Validation checks
    assert len(train_df) > 0, "Train set is empty"
    assert len(test_df) > 0, "Test set is empty"
    
    print(f"Train set: {len(train_df)} rows")
    print(f"  - Cities: {train_df[city_col].unique()}")
    print(f"  - Date range: {train_df[date_col].min()} to {train_df[date_col].max()}")
    print(f"Test set: {len(test_df)} rows")
    print(f"  - Cities: {test_df[city_col].unique()}")
    print(f"  - Date range: {test_df[date_col].min()} to {test_df[date_col].max()}")
    
    return train_df, test_df


def validate_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = 'date'
) -> bool:
    """
    Validate that there is no temporal leakage between train and test sets.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame
    test_df : pd.DataFrame
        Test DataFrame
    date_col : str
        Name of the date column
        
    Returns:
    --------
    bool
        True if no leakage detected
    """
    train_max = train_df[date_col].max()
    test_min = test_df[date_col].min()
    
    if train_max >= test_min:
        raise ValueError(f"Temporal leakage detected: train max ({train_max}) >= test min ({test_min})")
    
    return True


def get_split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = 'date',
    target_col: str = 'price'
) -> dict:
    """
    Get summary statistics for train/test split.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame
    test_df : pd.DataFrame
        Test DataFrame
    date_col : str
        Name of the date column
    target_col : str
        Name of the target column
        
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'train_date_range': (train_df[date_col].min(), train_df[date_col].max()),
        'test_date_range': (test_df[date_col].min(), test_df[date_col].max()),
        'train_target_mean': train_df[target_col].mean() if target_col in train_df.columns else None,
        'test_target_mean': test_df[target_col].mean() if target_col in test_df.columns else None,
        'train_target_std': train_df[target_col].std() if target_col in train_df.columns else None,
        'test_target_std': test_df[target_col].std() if target_col in test_df.columns else None,
    }
    
    return summary














