"""
Random DataFrame Generator

A module for generating random pandas DataFrames with various column types and distributions.
Useful for creating demonstration data, testing, and prototyping data analysis code.
"""

from .random_dataframe import create_dataframe
from .utils import (
    create_boolean_column,
    create_category_column,
    create_date_column,
    create_float_column,
    create_id_column,
    create_integer_column,
    create_text_column,
)

__all__ = [
    "create_dataframe",
    "create_id_column",
    "create_boolean_column",
    "create_integer_column",
    "create_float_column",
    "create_date_column",
    "create_category_column",
    "create_text_column",
]
