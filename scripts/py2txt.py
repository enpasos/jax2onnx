#!/usr/bin/env python3

import os
import shutil


def main():
    # Define the output directory
    output_dir = os.path.join("..", "output", "py2txt")
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the absolute path of this script (to optionally skip it)
    current_script = os.path.abspath(__file__)

    # Walk through the current directory and subdirectories
    for root, dirs, files in os.walk(".."):
        for filename in files:
            if filename.endswith(".py"):
                source_path = os.path.join(root, filename)

                # Optional: skip copying this script itself
                if os.path.abspath(source_path) == current_script:
                    continue

                # Convert filename from something.py to something.txt
                base_name, _ = os.path.splitext(filename)
                new_filename = base_name + ".txt"

                # Construct the target file path in the output directory
                # Here, we just copy files into the same folder, ignoring subdirectory structure
                target_path = os.path.join(output_dir, new_filename)

                # Copy the file (preserves metadata)
                shutil.copy2(source_path, target_path)
                print(f"Copied: {source_path} -> {target_path}")


if __name__ == "__main__":
    main()
