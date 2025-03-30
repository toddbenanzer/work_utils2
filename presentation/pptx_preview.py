"""
Utility for previewing PowerPoint presentations in Jupyter notebooks.
Uses a simple approach to convert PowerPoint slides to images through PDF intermediary.
"""

import glob
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
from pptx import Presentation


def preview_presentation(
    presentation: Union[str, Presentation],
    slide_numbers: Optional[Union[int, List[int], Tuple[int, int]]] = None,
    figsize: Tuple[int, int] = (10, 7.5),
    dpi: int = 150,
) -> None:
    """
    Display preview images of slides from a PowerPoint presentation in Jupyter Notebook.

    Args:
        presentation: Either a pptx.Presentation object or a string filepath to a .pptx file
        slide_numbers: Optional. Can be:
            - None: Display all slides
            - int: Display a specific slide (1-based index)
            - list: Display specific slides (e.g., [1, 3, 5])
            - tuple: Display a range of slides (e.g., (2, 5) for slides 2 through 5)
        figsize: Figure size for the displayed images (width, height) in inches
        dpi: Resolution for the displayed images

    Returns:
        None, displays slides inline in the notebook
    """
    # Handle the input presentation, which could be a path or a Presentation object
    temp_dirs = []
    temp_files = []
    pptx_path = None

    try:
        if isinstance(presentation, str):
            pptx_path = presentation
        else:
            # If it's a Presentation object, save it to a temporary file
            temp_fd, pptx_path = tempfile.mkstemp(suffix=".pptx")
            os.close(temp_fd)
            temp_files.append(pptx_path)
            presentation.save(pptx_path)

        # Create a temporary directory for images
        temp_img_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_img_dir)

        # Check if LibreOffice is available
        try:
            subprocess.run(
                ["libreoffice", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            libreoffice_cmd = "libreoffice"
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                subprocess.run(
                    ["soffice", "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                libreoffice_cmd = "soffice"
            except (subprocess.SubprocessError, FileNotFoundError):
                raise RuntimeError(
                    "LibreOffice not found. Please ensure 'libreoffice' or 'soffice' is installed and in your PATH."
                )

        # Use LibreOffice to export each slide directly to PNG
        # This is more reliable than the PDF conversion approach
        cmd = [
            libreoffice_cmd,
            "--headless",
            "--convert-to",
            "png",
            "--outdir",
            temp_img_dir,
            pptx_path,
        ]

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if process.returncode != 0:
            raise RuntimeError(
                f"Failed to convert PPTX to PNG: {process.stderr.decode()}"
            )

        # Get a list of all generated image files
        # LibreOffice will name them with the base filename followed by page number
        base_filename = os.path.basename(pptx_path).replace(".pptx", "")
        image_files = sorted(glob.glob(f"{temp_img_dir}/{base_filename}*.png"))

        if not image_files:
            raise RuntimeError(
                "No image files were generated. Check that the PowerPoint file has slides."
            )

        # Determine which slides to display based on slide_numbers parameter
        slide_indices = []

        if slide_numbers is None:
            # Display all slides
            slide_indices = list(range(len(image_files)))
        elif isinstance(slide_numbers, int):
            # Display a specific slide
            if 1 <= slide_numbers <= len(image_files):
                slide_indices = [slide_numbers - 1]  # Convert to 0-based index
            else:
                raise ValueError(
                    f"Slide number {slide_numbers} is out of range (1-{len(image_files)})"
                )
        elif isinstance(slide_numbers, list):
            # Display specific slides
            for num in slide_numbers:
                if 1 <= num <= len(image_files):
                    slide_indices.append(num - 1)  # Convert to 0-based index
                else:
                    print(
                        f"Warning: Slide number {num} is out of range (1-{len(image_files)}) and will be skipped"
                    )
        elif isinstance(slide_numbers, tuple) and len(slide_numbers) == 2:
            # Display a range of slides
            start, end = slide_numbers
            if 1 <= start <= len(image_files) and 1 <= end <= len(image_files):
                slide_indices = list(
                    range(start - 1, end)
                )  # Convert to 0-based indices
            else:
                raise ValueError(
                    f"Slide range {slide_numbers} is out of bounds (1-{len(image_files)})"
                )
        else:
            raise ValueError(
                "slide_numbers must be None, an integer, a list of integers, or a tuple of (start, end)"
            )

        # Display the selected slides
        for idx in slide_indices:
            img_path = image_files[idx]
            img = Image.open(img_path)

            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Slide {idx + 1}")
            plt.tight_layout()
            display(plt.gcf())
            plt.close()

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_file}: {e}")

        # Clean up temporary directories
        for temp_dir in temp_dirs:
            try:
                # Remove all files in the directory
                for file_path in glob.glob(f"{temp_dir}/*"):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        print(f"Warning: Failed to delete file {file_path}: {e}")
                # Remove the directory
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to delete temporary directory {temp_dir}: {e}")


def check_dependencies():
    """
    Check if LibreOffice is installed, which is the only dependency required for this simplified approach.

    Returns:
        bool: True if LibreOffice is available, False otherwise
    """
    missing = []

    # Check for LibreOffice
    try:
        subprocess.run(
            ["libreoffice", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            subprocess.run(
                ["soffice", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            missing.append("LibreOffice (libreoffice or soffice command)")

    if missing:
        print("The following dependencies are missing:")
        for dep in missing:
            print(f"- {dep}")
        print("\nInstallation instructions:")
        print("Ubuntu/Debian: sudo apt-get install libreoffice")
        print("CentOS/RHEL: sudo yum install libreoffice")
        print("Alpine: apk add libreoffice")
        return False

    return True
