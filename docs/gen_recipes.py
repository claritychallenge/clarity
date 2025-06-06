import os
import sys
from collections import defaultdict

# --- Configuration ---
script_dir = os.path.dirname(__file__)  # Path to the 'docs/' directory
project_root = os.path.abspath(
    os.path.join(script_dir, "../..")
)  # Path to the project root
recipes_package_path = os.path.join(
    project_root, "recipes"
)  # Path to the actual 'recipes' source directory

output_base_dir = os.path.join(
    script_dir, "recipes_api"
)  # Output directory within docs/
main_index_file_name = "recipes_index.rst"  # Default index file name for each directory

# Data structure to hold content grouped by category and then by full relative path within that category
# Example:
# categorized_content['cec1']['e009_sheffield']['modules'] = [...]
# categorized_content['cec1']['e009_sheffield']['yaml_files'] = [...]
# categorized_content['cad1']['task1/baseline']['modules'] = [...]
categorized_content = defaultdict(
    lambda: defaultdict(lambda: {"modules": [], "yaml_files": []})
)

# print(f"DEBUG: gen_recipes.py running from: {os.getcwd()}")
# print(f"DEBUG: Scanning recipes package at: {recipes_package_path}")
# print(f"DEBUG: Outputting generated RST files to: {output_base_dir}")
# print(f"DEBUG: Does recipes_package_path exist? {os.path.exists(recipes_package_path)}")
# print(
#     f"DEBUG: Is recipes_package_path a directory? {os.path.isdir(recipes_package_path)}"
# )

# --- Discover and Categorize Content (Modules and YAMLs) ---
if not os.path.isdir(recipes_package_path):
    print(
        f"ERROR: The 'recipes' package directory was not found at: {recipes_package_path}"
    )
    sys.exit(1)

for root, dirs, files in os.walk(recipes_package_path):
    # Prune search to skip common non-source directories/files
    dirs[:] = [
        d for d in dirs if not d.startswith((".", "_", "temp"))
    ]  # Added 'temp' if you have temp dirs

    # Calculate current path relative to the top-level 'recipes' package
    # E.g., 'cec1/e009_sheffield' or 'cad1/task2/baseline'
    current_relative_path_from_recipes_root = os.path.relpath(
        root, recipes_package_path
    )

    # Determine the top-level category (e.g., 'cec1', 'cad1')
    # If at the 'recipes/' root, category is 'recipes_root_level'
    path_parts = current_relative_path_from_recipes_root.split(os.sep)

    if current_relative_path_from_recipes_root == ".":
        category_name = "recipes_root_level"  # Special category for content directly under 'recipes/'
        path_within_category = "."  # Key for content directly at this level
    else:
        category_name = path_parts[0]
        # Use the full relative path from the category root for unique grouping
        # E.g., 'e009_sheffield', 'task1/baseline', or '.' for category root itself
        path_within_category = (
            os.path.join(*path_parts[1:]) if len(path_parts) > 1 else "."
        )

    # Ensure path_within_category is not empty for root of a category
    if (
        not path_within_category
    ):  # Should already be handled by the above `if len(path_parts) > 1`
        path_within_category = "."

    for file in files:
        full_file_path = os.path.join(root, file)

        # Calculate module path relative to the project root (for Sphinx import)
        if file.endswith(".py") and file != "__init__.py":
            module_path = (
                os.path.relpath(full_file_path, project_root)
                .replace(os.sep, ".")
                .replace(".py", "")
            )
            categorized_content[category_name][path_within_category]["modules"].append(
                module_path
            )
            # print(
            #     f"DEBUG: Found module '{module_path}' in category '{category_name}', path '{path_within_category}'"
            # )

        # Calculate YAML file path relative to the 'docs/' directory (for literalinclude)
        elif file.endswith((".yaml", ".yml")):

            # Storing relative to project_root is safest for later processing
            rel_yaml_path_from_project_root = os.path.relpath(
                full_file_path, project_root
            )
            categorized_content[category_name][path_within_category][
                "yaml_files"
            ].append(rel_yaml_path_from_project_root)
            # print(
            #     f"DEBUG: Found YAML '{rel_yaml_path_from_project_root}' in category '{category_name}', path '{path_within_category}'"
        # )


# --- Function to recursively generate nested directories and RST files for sub-nodes ---
def generate_nested_docs(
    node_content_dict, current_output_base_path, logical_path_segments
):
    # node_content_dict: A dictionary representing the current level of the tree.
    #                    Keys are directory names, values are either empty dicts (further subdirs)
    #                    or have a '.' key if this node contains content.
    # current_output_base_path: The physical path where to create current level's content.
    # logical_path_segments: List of segments representing the logical path from the category root (e.g., ['task1', 'baseline']).

    # Determine the title for this index.rst file based on its logical path
    rst_title_parts = [p.replace("_", " ").title() for p in logical_path_segments]
    rst_title = " ".join(rst_title_parts)  # + " API"

    # Determine the physical path for the output RST file
    output_rst_file_path = os.path.join(current_output_base_path, main_index_file_name)

    rst_lines = [
        f"{rst_title}\n",
        f"{'=' * len(rst_title)}\n\n",  # Primary heading for this file
    ]

    # Process content at the current node's own level (if it exists)
    content_at_current_node = node_content_dict.get(
        ".", {"modules": [], "yaml_files": []}
    )
    modules_here = sorted(content_at_current_node["modules"])
    yaml_files_here = sorted(content_at_current_node["yaml_files"])

    # Add Python modules if present at this exact level (i.e., under the '.' key)
    if modules_here:
        rst_lines.append(".. rubric:: Python Modules\n\n")
        rst_lines.append(".. autosummary::\n")

        # Calculate path to 'generated/' directory relative to the current RST file
        rel_path_to_generated = os.path.relpath(
            os.path.join(script_dir, "generated"),  # Path to docs/generated/
            current_output_base_path,
            # Path to where current RST file is being written
        ).replace(
            os.sep, "/"
        )  # Use forward slashes for Sphinx paths

        rst_lines.append(f"   :toctree: {rel_path_to_generated}/\n")
        rst_lines.append("   :nosignatures:\n\n")

        for module in modules_here:
            rst_lines.append(f"   {module}\n")
        rst_lines.append("\n")  # Add a blank line for separation

    # Add YAML content if present at this exact level (i.e., under the '.' key)
    if yaml_files_here:
        # rst_lines.append(".. rubric:: Configuration Files\n\n")
        for y_file_rel_from_project_root in yaml_files_here:
            full_yaml_path = os.path.join(project_root, y_file_rel_from_project_root)
            yaml_title = os.path.basename(full_yaml_path)

            # Calculate path to YAML file relative to the current RST file
            rel_yaml_path_for_literalinclude = os.path.relpath(
                full_yaml_path,
                current_output_base_path,
                # Relative to where current RST file is being written
            ).replace(os.sep, "/")

            rst_lines.append(f"**{yaml_title}**\n\n")
            rst_lines.append(
                f".. literalinclude:: {rel_yaml_path_for_literalinclude}\n"
            )
            rst_lines.append("   :language: yaml\n")
            rst_lines.append("   :linenos:\n\n")
        rst_lines.append("\n")

    # Add a toctree for sub-directories if they exist
    sub_dir_keys = sorted([k for k in node_content_dict.keys() if k != "."])
    if sub_dir_keys:
        rst_lines.append(".. toctree::\n")
        rst_lines.append("   :maxdepth: 8\n")
        rst_lines.append("   :caption: Sub-sections:\n\n")

        for sub_key in sub_dir_keys:
            # Add entry to the current toctree
            rst_lines.append(
                f"   {sub_key}/{main_index_file_name.replace('.rst', '')}\n"
            )

            # Recurse for deeper levels, creating subdirectories and their index.rst
            sub_output_dir = os.path.join(current_output_base_path, sub_key)
            os.makedirs(sub_output_dir, exist_ok=True)  # Ensure directory exists

            generate_nested_docs(
                node_content_dict[sub_key],
                sub_output_dir,
                logical_path_segments + [sub_key],
            )
        rst_lines.append("\n")  # Blank line after toctree

    # Only write the file if there's actual content or sub-sections
    if modules_here or yaml_files_here or sub_dir_keys:
        with open(output_rst_file_path, "w", encoding="utf-8") as f:
            f.writelines(rst_lines)
        # print(f"DEBUG: Generated nested RST file: {output_rst_file_path}")
    else:
        print(f"DEBUG: Skipping empty RST file generation for: {output_rst_file_path}")


# --- Generate Main Recipes API Index File and start recursive process for categories ---
# Initialize main_index_toctree_entries BEFORE it's used by the main loop
main_index_toctree_entries = []  # This is for the top-level clarity_recipes/index.rst

# Create the base output directory for generated recipes_api docs
os.makedirs(output_base_dir, exist_ok=True)

for category_name, content_by_path in sorted(categorized_content.items()):
    if not content_by_path:  # Skip empty categories
        continue

    # Create the base output directory for this category (e.g., docs/clarity_recipes/cad1/)
    category_base_output_dir = os.path.join(output_base_dir, category_name)
    os.makedirs(category_base_output_dir, exist_ok=True)

    # Add this category's main index file to the top-level main_index_toctree_entries
    main_index_toctree_entries.append(
        f"{category_name}/{main_index_file_name.replace('.rst', '')}"
    )

    # Build a conceptual tree for this category's content based on its paths
    # Example: {'task1': {'baseline': {'.': {'modules':[], 'yamls':[]}} }, 'task2': {'.': {'modules':[]}} }
    category_tree_root = {}

    for path_in_category, content_data in sorted(content_by_path.items()):
        # Split path like 'task1/baseline' into ['task1', 'baseline']
        parts = path_in_category.split(os.sep) if path_in_category != "." else []

        current_node_in_tree = category_tree_root
        for part in parts:
            # Create sub-dictionaries for each path segment
            current_node_in_tree = current_node_in_tree.setdefault(part, {})

        # Assign the actual content (modules/yamls) to the leaf node, under a special '.' key
        current_node_in_tree["."] = content_data

    # --- Generate the category's top-level index.rst (e.g., docs/clarity_recipes/cad1/index.rst) ---
    category_index_path = os.path.join(category_base_output_dir, main_index_file_name)

    # Determine the title for this category's main index file
    if category_name == "recipes_root_level":
        category_title_for_index = "RECIPES OVERVIEW"
    else:
        category_title_for_index = category_name.replace("_", " ").upper()

    category_index_lines = [
        f"{category_title_for_index}\n",
        f"{'=' * len(category_title_for_index)}\n\n",
        (
            f"Detailed API documentation for the {category_name.replace('_', ' ')} recipes."
            if category_name != "recipes_root_level"
            else "Detailed API documentation for core recipes."
        ),
        "\n\n",
    ]

    # Add content directly at the category root (if any) to its index.rst
    root_category_direct_content = category_tree_root.get(
        ".", {"modules": [], "yaml_files": []}
    )
    modules_at_category_root = sorted(root_category_direct_content["modules"])
    yaml_files_at_category_root = sorted(root_category_direct_content["yaml_files"])

    if modules_at_category_root:
        category_index_lines.append(".. rubric:: Python Modules\n\n")
        category_index_lines.append(".. autosummary::\n")
        rel_path_to_generated = os.path.relpath(
            os.path.join(script_dir, "generated"),
            category_base_output_dir,
        ).replace(os.sep, "/")
        category_index_lines.append(f"   :toctree: {rel_path_to_generated}/\n")
        category_index_lines.append("   :nosignatures:\n\n")
        for module in modules_at_category_root:
            category_index_lines.append(f"   {module}\n")
        category_index_lines.append("\n")

    if yaml_files_at_category_root:
        category_index_lines.append(".. rubric:: Configuration Files\n\n")
        for y_file_rel_from_project_root in yaml_files_at_category_root:
            full_yaml_path = os.path.join(project_root, y_file_rel_from_project_root)
            yaml_title = os.path.basename(full_yaml_path)
            rel_yaml_path_for_literalinclude = os.path.relpath(
                full_yaml_path, category_base_output_dir
            ).replace(os.sep, "/")
            category_index_lines.append(f"**{yaml_title}**\n\n")
            category_index_lines.append(
                f".. literalinclude:: {rel_yaml_path_for_literalinclude}\n"
            )
            category_index_lines.append("   :language: yaml\n")
            category_index_lines.append("   :linenos:\n\n")
        category_index_lines.append("\n")

    # Add toctree for direct sub-directories of this category (e.g., task1, task2)
    direct_sub_dirs_for_category_toctree = sorted(
        [k for k in category_tree_root.keys() if k != "."]
    )
    if direct_sub_dirs_for_category_toctree:
        category_index_lines.append(".. toctree::\n")
        category_index_lines.append("   :maxdepth: 8\n")
        category_index_lines.append(
            "   :caption: Sections:\n\n"
        )  # Changed from Sub-sections to Sections

        for key in direct_sub_dirs_for_category_toctree:
            # Add entry to the category's toctree
            category_index_lines.append(
                f"   {key}/{main_index_file_name.replace('.rst', '')}\n"
            )

            # Now, call the recursive function to generate docs for this sub-directory (e.g., task1/index.rst)
            child_output_dir = os.path.join(category_base_output_dir, key)
            os.makedirs(child_output_dir, exist_ok=True)
            generate_nested_docs(
                category_tree_root[key],
                child_output_dir,
                [key],  # Start logical path segments for the child
            )
        category_index_lines.append("\n")  # Blank line after toctree

    with open(category_index_path, "w", encoding="utf-8") as f:
        f.writelines(category_index_lines)
    # print(f"DEBUG: Generated category index: {category_index_path}")

# --- Final Main Recipes API Index File (docs/clarity_recipes/index.rst) ---
main_index_file_path = os.path.join(output_base_dir, main_index_file_name)

main_rst_lines = [
    "Recipes\n",
    "===========\n\n",
    "This section provides an overview of the PyClarity recipes API, grouped by challenge or category.\n\n",
    ".. toctree::\n",
    "   :maxdepth: 8\n",
    "   :caption: Recipe Categories:\n\n",
]

if main_index_toctree_entries:
    main_rst_lines.extend(
        [f"   {entry}\n" for entry in sorted(main_index_toctree_entries)]
    )
else:
    main_rst_lines.append("   No recipe categories found yet.\n")

with open(main_index_file_path, "w", encoding="utf-8") as f:
    f.writelines(main_rst_lines)
print(f"Generated main recipes index: {main_index_file_path}")

print("Recipe API generation complete.")
