# docs/generate_full_docs.py
import json
import os
import shutil
import subprocess
import sys

# --- Configuration ---
DOCS_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)  # This script's directory (docs/)
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_ROOT, ".."))  # One level up
JSON_CONFIG_PATH = os.path.join(DOCS_ROOT, "docs_config.json")
INDEX_RST_PATH = os.path.join(DOCS_ROOT, "index.rst")
# Ensure this matches the name of your generator script (e.g., gen_recipes.py or gen_menu.py)
GENERATOR_SCRIPT_NAME = (
    "gen_menu.py"  # <-- IMPORTANT: CHANGE THIS IF YOUR SCRIPT IS NAMED gen_menu.py
)
GENERATOR_SCRIPT_PATH = os.path.join(DOCS_ROOT, GENERATOR_SCRIPT_NAME)

GENERATED_MENUS_RST_PATH = os.path.join(DOCS_ROOT, "_generated_menus.rst")

# --- Helper Functions ---


def clean_generated_directories(config_data):
    """Removes all directories that will be created by the generator script."""
    print("Cleaning up previous generated documentation directories...")
    # Clean the 'generated' folder (for autosummary stubs)
    autosummary_generated_path = os.path.join(DOCS_ROOT, "generated")
    if os.path.exists(autosummary_generated_path):
        shutil.rmtree(autosummary_generated_path)
        print(f"  Removed: {autosummary_generated_path}")

    # Clean the output_dir_name directories specified in the config
    # Use a set to avoid deleting the same directory multiple times
    output_dirs_to_clean = set()
    for category_items in config_data.values():
        for item_params in category_items:
            output_dir = item_params.get(
                "output_dir_name", "automenu"
            )  # Default must match gen_recipes.py
            output_dirs_to_clean.add(os.path.join(DOCS_ROOT, output_dir))

    for dir_path in output_dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"  Removed: {dir_path}")
    print("Cleanup complete.")


def run_generator_script(params):
    """Constructs and runs the gen_recipes.py script with given parameters."""
    print(f"\nRunning {GENERATOR_SCRIPT_NAME} for: {params['path']}")
    command = [
        sys.executable,
        GENERATOR_SCRIPT_PATH,
    ]  # Use sys.executable for current python interpreter
    for key, value in params.items():
        # Adjust keys to match gen_recipes.py's argument names
        arg_key = f"--{key}"
        command.extend(
            [arg_key, str(value)]
        )  # Convert value to string for command line

    try:
        # Run the generator script, capturing its output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"  {GENERATOR_SCRIPT_NAME} stdout:\n{result.stdout}")
        if result.stderr:
            print(f"  {GENERATOR_SCRIPT_NAME} stderr:\n{result.stderr}")
        print(f"  Successfully ran {GENERATOR_SCRIPT_NAME} for {params['path']}")
    except subprocess.CalledProcessError as e:
        print(
            f"Error running {GENERATOR_SCRIPT_NAME} for {params['path']}:",
            file=sys.stderr,
        )
        print(f"  Command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"  Return Code: {e.returncode}", file=sys.stderr)
        print(f"  Stdout:\n{e.stdout}", file=sys.stderr)
        print(f"  Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)  # Exit if a generator script fails
    except FileNotFoundError:
        print(
            f"Error: {GENERATOR_SCRIPT_NAME} not found at {GENERATOR_SCRIPT_PATH}. "
            "Please ensure the script exists and GENERATOR_SCRIPT_NAME is correct.",
            file=sys.stderr,
        )
        sys.exit(1)


def generate_toctree_content(config_data):
    """Generates the RST content for the toctrees to be inserted into index.rst."""
    toctree_blocks = []
    for category_name, items in config_data.items():
        toctree_blocks.append(f"\n.. toctree::")
        toctree_blocks.append(f"   :maxdepth: 1")  # You can change this maxdepth
        toctree_blocks.append(f"   :caption: {category_name}:")
        toctree_blocks.append(f"")  # Blank line after caption

        for item_params in items:
            output_dir = item_params.get(
                "output_dir_name", "automenu"
            )  # Default must match gen_recipes.py
            output_file = item_params.get(
                "output_file_name", "recipes_index"
            )  # Default must match gen_recipes.py
            # Path relative to docs/
            toctree_blocks.append(
                f"   {output_dir}/{output_file}"
            )  # No .rst extension in toctree
        toctree_blocks.append(f"\n")  # Blank line after each toctree block

    return "\n".join(toctree_blocks)


def write_generated_menus_rst(generated_content):
    """Writes the generated toctree content to a dedicated _generated_menus.rst file."""
    print(f"\nWriting generated menus to {GENERATED_MENUS_RST_PATH}...")
    with open(GENERATED_MENUS_RST_PATH, "w", encoding="utf-8") as f:
        f.write(generated_content)  # Write the string directly
    print("_generated_menus.rst updated successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting full documentation generation...")

    # 1. Load configuration
    try:
        with open(JSON_CONFIG_PATH, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        print(f"Loaded configuration from {JSON_CONFIG_PATH}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {JSON_CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {JSON_CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    # 2. Clean previously generated directories
    clean_generated_directories(config_data)

    # 3. Run the generator script for each item in the config
    for category_name, items in config_data.items():
        for item_params in items:
            run_generator_script(item_params)

    # 4. Generate the RST content for index.rst
    generated_toctree_content = generate_toctree_content(config_data)

    # 5. Write the generated content to the new file
    write_generated_menus_rst(generated_toctree_content)

    print(
        "\nFull documentation generation process finished. Now run 'make html' from your docs/ directory."
    )
    print("Example: cd docs/ && make html")
