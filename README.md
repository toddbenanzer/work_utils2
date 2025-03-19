# Modern VS Code Python Development Environment

This repository provides a modern, containerized Python development environment using VS Code, Docker, Poetry, and various code quality tools.

## Features

- **Python 3.12** - Latest stable Python version
- **Docker** - Containerized development environment
- **Dev Containers** - VS Code integration with Docker
- **Code Quality Tools**:
  - Black - Code formatting
  - Flake8 - Linting
  - MyPy - Type checking
  - Ruff - Fast linter and fixer
  - Pytest - Testing framework
  - Pre-commit - Git hooks for code quality checks

## Getting Started

### Prerequisites

- [VS Code](https://code.visualstudio.com/)
- [Docker](https://www.docker.com/products/docker-desktop)
- [VS Code Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Usage

The development container provides:

- A full Python environment with all dependencies
- Code formatting on save
- Linting and type checking
- Testing support
- Git integration

#### Running the application

```python
python main.py
```

#### Running tests

```bash
pytest
```

#### Adding dependencies

With pip:

```bash
pip install <package-name>
```

## Project Structure

```text
.
├── .devcontainer/             # VS Code devcontainer configuration
│   ├── Dockerfile             # Container definition
│   └── devcontainer.json      # VS Code container settings
├── .vscode/                   # VS Code specific settings
│   ├── launch.json            # Debug configurations
│   └── settings.json          # Editor settings
├── requirements.txt           # Package dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
└── main.py                    # Entry point
```

## Customizing

- Adjust Python dependencies in `requirements.txt`
- Modify VS Code settings in `.vscode/settings.json`
- Change container configuration in `.devcontainer/`
- Update code quality settings in `requirements.txt` under respective tool sections
