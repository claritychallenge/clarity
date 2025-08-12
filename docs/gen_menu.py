import argparse
import os
import sys

# --- Configuration ---
script_dir = os.path.dirname(__file__)  # Path to the 'docs/' directory
project_root = os.path.abspath(
    os.path.join(script_dir, "..")
)  # Path to the project root


# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Generate Sphinx RST files for a specified content "
    "root within the project."
)
parser.add_argument(
    "--path",
    type=str,
    required=True,
    help="Relative path from project root to the content directory to scan "
    "(e.g., 'recipes', 'recipes/cad1', 'clarity/evaluator').",
)
parser.add_argument(
    "--levels",
    type=int,
    default=1,
    help="Number of menu levels (N) to display in the left doctree. "
    "The last level (N-1) will show all its children with relative paths.",
)
parser.add_argument(
    "--output_dir_name",
    type=str,
    default="automenu",  # Can be customized, e.g., 'clarity_docs'
    help="Name of the output directory within 'docs/' where generated "
    "RST files will be placed.",
)
parser.add_argument(
    "--output_file_name",
    type=str,
    default="index.rst",  # Can be customized, e.g., 'clarity_docs'
    help="Name of the output directory within 'docs/'"
    " where generated RST files will be placed.",
)
parser.add_argument(
    "--top_level_title",
    type=str,
    default=None,  # Default to None, meaning it will be derived if not provided
    help="Optional: Custom title for the very top-level"
    " generated RST file (e.g., 'My API Overview'). "
    "If not provided, the title is derived from the --path argument.",
)

args = parser.parse_args()

# Validate levels
if args.levels < 1:
    print("ERROR: --levels must be at least 1.")
    sys.exit(1)


# Input parameters from arguments
input_root_path_from_project_root = args.path
max_menu_levels = args.levels  # N
output_base_dir_name = args.output_dir_name
main_index_file_name = f"{args.output_file_name}.rst"
custom_top_level_title = args.top_level_title

# Full path to the directory to scan
base_scan_path = os.path.join(project_root, input_root_path_from_project_root)
# Full path to the output directory within docs/
output_base_dir = os.path.join(script_dir, output_base_dir_name)

# Data structure to hold content as a hierarchical tree
# This will replace 'categorized_content' from the old script logic
content_tree: dict = {}  # Root of the generated documentation tree

# --- Discover and Categorize Content (Modules and YAMLs) ---
if not os.path.isdir(base_scan_path):
    print(f"ERROR: The specified content directory was not found at: {base_scan_path}")
    sys.exit(1)

print(f"INFO: Scanning content from: {base_scan_path}")
print(f"INFO: Generating RST files to: {output_base_dir}")
print(f"INFO: Desired menu levels (N): {max_menu_levels}")

for root, dirs, files in os.walk(base_scan_path):
    # Prune search to skip common non-source directories/files
    dirs[:] = [d for d in dirs if not d.startswith((".", "_", "temp"))]

    # Calculate current path relative to the base_scan_path
    # E.g., 'task1' or 'task1/baseline' if base_scan_path was 'recipes/cad1'
    current_relative_path_from_scan_root = os.path.relpath(root, base_scan_path)

    # Build the path segments for navigation in the content_tree
    path_segments = (
        current_relative_path_from_scan_root.split(os.sep)
        if current_relative_path_from_scan_root != "."
        else []
    )

    # Navigate or create nodes in the content_tree
    current_node = content_tree
    for segment in path_segments:
        current_node = current_node.setdefault(segment, {})

    # The actual content (modules/yamls) is stored under a special
    # '.' key at the leaf node
    current_node.setdefault(".", {"modules": [], "yaml_files": []})

    for file in files:
        full_file_path = os.path.join(root, file)

        # Calculate module path relative to the project root
        # (for Sphinx autodoc import)
        if file.endswith(".py") and file != "__init__.py":
            module_path = (
                os.path.relpath(full_file_path, project_root)
                .replace(os.sep, ".")
                .replace(".py", "")
            )
            current_node["."]["modules"].append(module_path)

        # Calculate YAML file path relative to the project root (for literalinclude)
        elif file.endswith((".yaml", ".yml")):
            rel_yaml_path_from_project_root = os.path.relpath(
                full_file_path, project_root
            )
            current_node["."]["yaml_files"].append(rel_yaml_path_from_project_root)


# --- Function to recursively generate nested directories
#     and RST files for sub-nodes ---
def generate_nested_docs(
    node_content_tree,
    current_output_physical_path,
    logical_path_segments,  # e.g., ['cad1', 'task1']
    current_depth,  # current nesting level (0-indexed from base_scan_path)
    max_menu_levels,  # User defined N
    custom_top_level_title_arg,
):
    # Determine the title for this index.rst file
    # If it's the root of the scanned path, title is based on the
    # input_root_path_from_project_root
    # This is the very first (root) index.rst for the scanned path
    if not logical_path_segments:
        if custom_top_level_title_arg:  # Use custom title if provided
            rst_title = custom_top_level_title_arg
        else:  # Fallback to deriving from path
            rst_title_parts = (
                input_root_path_from_project_root.replace(os.sep, " ").title().split()
            )
            if not rst_title_parts:
                rst_title = "Project Root"
            else:
                rst_title = " ".join(rst_title_parts)
    else:
        # For sub-directories, title is based on the last segment of the logical path
        rst_title = logical_path_segments[-1].replace("_", " ").title()

    # Determine the physical path for the output RST file
    output_rst_file_path = os.path.join(
        current_output_physical_path, main_index_file_name
    )

    rst_lines = [
        f"{rst_title}\n",
        f"{'=' * len(rst_title)}\n\n",  # Primary heading for this file
    ]

    # Content at the current node's own level (stored under '.' key)
    content_at_current_node = node_content_tree.get(
        ".", {"modules": [], "yaml_files": []}
    )
    modules_here = sorted(content_at_current_node["modules"])
    yaml_files_here = sorted(content_at_current_node["yaml_files"])

    # Add Python modules if present at this exact level
    if modules_here:
        rst_lines.append(".. rubric:: Python Modules\n\n")
        rst_lines.append(".. autosummary::\n")

        # Path to 'generated/' directory relative to the current RST file
        # being written
        rel_path_to_generated = os.path.relpath(
            os.path.join(script_dir, "generated"),  # Path to docs/generated/
            current_output_physical_path,  # Path to where curr RST being written
        ).replace(
            os.sep, "/"
        )  # Use forward slashes for Sphinx paths

        rst_lines.append(f"   :toctree: {rel_path_to_generated}/\n")
        rst_lines.append("   :nosignatures:\n\n")

        for module in modules_here:
            rst_lines.append(f"   {module}\n")
        rst_lines.append("\n")

    # Add YAML content if present at this exact level
    if yaml_files_here:
        rst_lines.append(".. rubric:: Configuration Files\n\n")
        for y_file_rel_from_project_root in yaml_files_here:
            full_yaml_path = os.path.join(project_root, y_file_rel_from_project_root)
            yaml_title = os.path.basename(full_yaml_path)

            # Path to YAML file relative to the current RST file
            rel_yaml_path_for_literalinclude = os.path.relpath(
                full_yaml_path,
                current_output_physical_path,
            ).replace(os.sep, "/")

            rst_lines.append(f"**{yaml_title}**\n\n")
            rst_lines.append(
                f".. literalinclude:: {rel_yaml_path_for_literalinclude}\n"
            )
            rst_lines.append("   :language: yaml\n")
            rst_lines.append("   :linenos:\n\n")
        rst_lines.append("\n")

    # Add a toctree for sub-directories if they exist in the content_tree
    sub_dir_keys = sorted([k for k in node_content_tree.keys() if k != "."])
    if sub_dir_keys:
        rst_lines.append(".. toctree::\n")

        # Determine toctree maxdepth based on current_depth and max_menu_levels
        if current_depth < max_menu_levels - 1:
            # Not yet at the last menu level. Show only the next level in the menu.
            rst_lines.append("   :maxdepth: 1\n")
            caption_text = (
                "Sub-sections:" if current_depth > 0 else "Sections:"
            )  # Or customize based on logical_path_segments
            rst_lines.append(f"   :caption: {caption_text}\n\n")
        elif current_depth == max_menu_levels - 1:
            # This is the last menu level. Show all descendants
            # (modules, yamls, deeper dirs).
            rst_lines.append("   :maxdepth: -1\n")  # -1 means show all descendants
            caption_text = (
                f"{logical_path_segments[-1].replace('_', ' ').title()} Details:"
                if logical_path_segments
                else "Details:"
            )
            rst_lines.append(f"   :caption: {caption_text}\n\n")
        else:
            # We should technically not hit this if maxdepth: -1 is used above,
            # as the toctree will already list all descendants.
            # If we wanted to hide further menu items but still generate pages,
            # this would require a different strategy (e.g., no toctree at all).
            pass

        for sub_key in sub_dir_keys:
            # Add entry to the current toctree
            rst_lines.append(
                f"   {sub_key}/{main_index_file_name.replace('.rst', '')}\n"
            )

            # Recurse for deeper levels, creating subdirectories and their index.rst
            sub_output_dir = os.path.join(current_output_physical_path, sub_key)
            os.makedirs(sub_output_dir, exist_ok=True)  # Ensure directory exists

            generate_nested_docs(
                node_content_tree[sub_key],
                sub_output_dir,
                logical_path_segments + [sub_key],
                current_depth + 1,
                max_menu_levels,
                None,
            )
        rst_lines.append("\n")  # Blank line after toctree

    # Only write the file if there's actual content or sub-sections
    if modules_here or yaml_files_here or sub_dir_keys:
        with open(output_rst_file_path, "w", encoding="utf-8") as f:
            f.writelines(rst_lines)
        print(f"INFO: Generated RST file: {output_rst_file_path}")
    else:
        print(f"INFO: Skipping empty RST file generation for: {output_rst_file_path}")


# --- Main execution ---

# Clear previous generated output
if os.path.exists(output_base_dir):
    import shutil

    print(f"INFO: Cleaning existing output directory: {output_base_dir}")
    shutil.rmtree(output_base_dir)
os.makedirs(output_base_dir, exist_ok=True)


# Generate the main index file for the specified input root
# (e.g., docs/recipes_api/recipes_index.rst)
# This serves as the top-level entry point for the generated documentation tree.

# Determine title for the top-level index file
# This is the block where top_level_title gets its value
# (either from custom_top_level_title
# or by deriving it from the path)
if custom_top_level_title:
    top_level_title = custom_top_level_title
else:
    top_level_title_parts = (
        input_root_path_from_project_root.replace(os.sep, " ").title().split()
    )
    if not top_level_title_parts or top_level_title_parts == [""]:
        top_level_title = "Project Content Overview"
    else:
        top_level_title = " ".join(top_level_title_parts)

# Construct the full path for the top-level index file
# This assumes output_file_name_base is what you get from
# --output_file_name (e.g., 'icassp24')
top_level_index_path = os.path.join(output_base_dir, f"{main_index_file_name}.rst")

top_level_index_path = os.path.join(output_base_dir, main_index_file_name)
top_level_rst_lines = []
# Line 1: Add the title itself
top_level_rst_lines.append(f"{top_level_title}\n")
# Line 2: Add the underline for the title
top_level_rst_lines.append(f"{'=' * len(top_level_title)}\n\n")
# Add the toctree directive for children
top_level_rst_lines.append(".. toctree::\n")
top_level_rst_lines.append("   :maxdepth: 2\n")  # Or your desired maxdepth
top_level_rst_lines.append("   :caption: Content Overview:\n\n")


# Get keys for direct sub-directories of the scanned root (e.g., 'cad1', 'evaluator')
# These are the keys in the 'content_tree' that are not '.'
direct_sub_dirs_of_scan_root = sorted([k for k in content_tree.keys() if k != "."])

# Also check for content directly at the scan root
scan_root_content = content_tree.get(".", {"modules": [], "yaml_files": []})
modules_at_scan_root = sorted(scan_root_content["modules"])
yaml_files_at_scan_root = sorted(scan_root_content["yaml_files"])

# ADD THIS DEBUG PRINT HERE
print(f"DEBUG: modules_at_scan_root before autosummary loop: {modules_at_scan_root}")
print(f"DEBUG: direct_sub_dirs_of_scan_root: {direct_sub_dirs_of_scan_root}")

# Add content directly at the scanned root if any
if modules_at_scan_root:
    top_level_rst_lines.append(".. rubric:: Python Modules (Root Level)\n\n")
    top_level_rst_lines.append(".. autosummary::\n")
    rel_path_to_generated = os.path.relpath(
        os.path.join(script_dir, "generated"),
        output_base_dir,
    ).replace(os.sep, "/")
    top_level_rst_lines.append(f"   :toctree: {rel_path_to_generated}/\n")
    top_level_rst_lines.append("   :nosignatures:\n\n")
    for module in modules_at_scan_root:
        top_level_rst_lines.append(f"   {module}\n")
    top_level_rst_lines.append("\n")

if yaml_files_at_scan_root:
    top_level_rst_lines.append(".. rubric:: Configuration Files (Root Level)\n\n")
    for y_file_rel_from_project_root in yaml_files_at_scan_root:
        full_yaml_path = os.path.join(project_root, y_file_rel_from_project_root)
        yaml_title = os.path.basename(full_yaml_path)
        rel_yaml_path_for_literalinclude = os.path.relpath(
            full_yaml_path, output_base_dir
        ).replace(os.sep, "/")
        top_level_rst_lines.append(f"**{yaml_title}**\n\n")
        top_level_rst_lines.append(
            f".. literalinclude:: {rel_yaml_path_for_literalinclude}\n"
        )
        top_level_rst_lines.append("   :language: yaml\n")
        top_level_rst_lines.append("   :linenos:\n\n")
    top_level_rst_lines.append("\n")


# Add sub-directories of the scanned root to its main index.rst
if direct_sub_dirs_of_scan_root:
    top_level_rst_lines.append(".. toctree::\n")  # Start of toctree
    top_level_rst_lines.append("   :maxdepth: 2\n")
    top_level_rst_lines.append(
        "   :caption: Sub-sections:\n\n"
    )  # Or appropriate caption

    for key in direct_sub_dirs_of_scan_root:
        top_level_rst_lines.append(
            f"   {key}/{main_index_file_name.replace('.rst', '')}\n"
        )

        # Recurse for the first level of sub-directories
        child_output_dir = os.path.join(output_base_dir, key)
        os.makedirs(child_output_dir, exist_ok=True)
        generate_nested_docs(
            content_tree[key],
            child_output_dir,
            [key],  # Start logical path segments for the child
            1,  # current_depth is 1 for direct children of the scan root
            max_menu_levels,
            None,
        )
    top_level_rst_lines.append("\n")  # Blank line after toctree

with open(top_level_index_path, "w", encoding="utf-8") as f:
    f.writelines(top_level_rst_lines)
print(f"INFO: Generated main content index: {top_level_index_path}")

print("Documentation generation complete.")
