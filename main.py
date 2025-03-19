"""
Main application entry point.

This module serves as the entry point for the application.
"""
from typing import List, Optional


def hello(name: Optional[str] = None) -> str:
    """
    Generate a greeting message.

    Args:
        name: The name to greet. If None, a generic greeting is returned.

    Returns:
        A greeting message.
    """
    if name:
        return f"Hello, {name}!"
    return "Hello, world!"


def main() -> None:
    """Execute the main program logic."""
    message = hello()
    print(message)


if __name__ == "__main__":
    main()