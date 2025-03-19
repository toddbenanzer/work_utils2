"""Tests for the main module."""
import pytest

from main import hello


def test_hello_default() -> None:
    """Test hello function with default parameters."""
    result = hello()
    assert result == "Hello, world!"
    assert isinstance(result, str)


def test_hello_with_name() -> None:
    """Test hello function with a name."""
    name = "Python"
    result = hello(name)
    assert result == f"Hello, {name}!"
    assert isinstance(result, str)