# file: scripts/generate_readme.py

import os
import time
import importlib
import pkgutil
import logging
import subprocess
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Paths
PLUGIN_DIR = os.path.join(os.path.dirname(__file__), "../jax2onnx/plugins")
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../jax2onnx/examples")
README_PATH = os.path.join(os.path.dirname(__file__), "../README.md")

# Markers for the auto-generated sections
START_MARKER = "<!-- AUTOGENERATED TABLE START -->"
END_MARKER = "<!-- AUTOGENERATED TABLE END -->"

EXAMPLES_START_MARKER = "<!-- AUTOGENERATED EXAMPLES TABLE START -->"
EXAMPLES_END_MARKER = "<!-- AUTOGENERATED EXAMPLES TABLE END -->"

NETRON_BASE_URL = "https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/"


def run_pytest():
    """Runs pytest and captures actual pass/fail results using JSON output."""
    logging.info("🛠 Running full tests...")

    subprocess.run(
        ["pytest", "--json-report", "--json-report-file=output/pytest_report.json"],
        capture_output=True,
        text=True,
    )

    test_results = {}

    # Read the JSON report
    report_path = "output/pytest_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for test in data.get("tests", []):
            if "test_onnx_export[" in test["nodeid"]:
                testcase_raw = test["nodeid"].split("[")[-1].rstrip("]")

                # Normalize test case names by removing (Plugin) and (Example) suffixes
                testcase = testcase_raw.replace(" (Plugin)", "").replace(
                    " (Example)", ""
                )
                parts = testcase.split("_")
                variant = parts[-1]  # e.g. "10"
                base_name = "_".join(parts[:-1])  # e.g. "conv_3x3_1"

                status = "✅" if test["outcome"] == "passed" else "❌"
                test_results[f"{base_name}_{variant}"] = status

    logging.info(f"✅ {len(test_results)} tests completed.")
    return test_results


def extract_metadata(base_path, source_type):
    """Dynamically loads metadata from plugins/examples."""
    logging.info(f"📡 Extracting metadata from {source_type.lower()}s...")

    metadata_list = []
    for _, name, _ in pkgutil.walk_packages(
        [base_path], prefix=f"jax2onnx.{source_type.lower()}."
    ):
        module = importlib.import_module(name)
        if hasattr(module, "get_test_params"):
            for entry in module.get_test_params():
                entry["testcases"] = {
                    tc["testcase"]: "➖" for tc in entry.get("testcases", [])
                }
                metadata_list.append(entry)

    logging.info(f"✅ {len(metadata_list)} {source_type.lower()} components found.")
    return metadata_list


def update_readme(metadata_plugins, metadata_examples, test_results):
    """Updates README.md with two auto-generated tables."""
    logging.info("📄 Updating README...")

    # Table headers
    table_header_plugins = """
| JAX Component | ONNX Components | Testcases | Since |
|:-------------|:---------------|:---------|:------|
""".strip()

    table_header_examples = """
| Component | Description | Children | Testcases | Since |
|:----------|:------------|:---------|:---------|:------|
""".strip()

    # Sort plugins & examples
    metadata_plugins = sorted(metadata_plugins, key=lambda x: x["jax_component"])
    metadata_examples = sorted(metadata_examples, key=lambda x: x["component"])

    # Define the mapping of variants to tooltips
    tooltips = {
        "00": "static batch dim",
        "10": "static batch dim + more shape info",
        "01": "dynamic batch dim",
        "11": "dynamic batch dim + more shape info",
    }

    # Generate plugins table
    plugin_rows = []
    for entry in metadata_plugins:
        onnx_links = "<br>".join(
            [f"[{op['component']}]({op['doc']})" for op in entry["onnx"]]
        )

        # Prepare testcases with multiple variants
        testcases_column = []
        base_names = set(tc for tc in entry.get("testcases", {}).keys())

        for base_name in sorted(base_names):
            urls = {
                variant: f"https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/{base_name}_{variant}.onnx"
                for variant in tooltips
                if f"{base_name}_{variant}" in test_results
            }

            # Use pass/fail status for the icons
            testcase_row = f"`{base_name}` " + " ".join(
                f"[{test_results.get(f'{base_name}_{variant}', '❌')}]({urls[variant]} \"{tooltips[variant]}\")"
                for variant in tooltips
                if f"{base_name}_{variant}" in test_results
            )
            testcases_column.append(testcase_row)

        testcases_column = "<br>".join(testcases_column) if testcases_column else "➖"

        plugin_rows.append(
            f"| [{entry['jax_component']}]({entry['jax_doc']}) "
            f"| {onnx_links} "
            f"| {testcases_column} "
            f"| {entry['since']} |"
        )

    # Generate examples table
    example_rows = []
    for entry in metadata_examples:
        children_list = "<br>".join(entry["children"])

        base_names = set(tc for tc in entry.get("testcases", {}).keys())
        testcases_column = []

        for base_name in sorted(base_names):
            urls = {
                variant: f"https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/{base_name}_{variant}.onnx"
                for variant in tooltips
                if f"{base_name}_{variant}" in test_results
            }

            testcase_row = f"`{base_name}` " + " ".join(
                f"[{test_results.get(f'{base_name}_{variant}', '❌')}]({urls[variant]} \"{tooltips[variant]}\")"
                for variant in tooltips
                if f"{base_name}_{variant}" in test_results
            )
            testcases_column.append(testcase_row)

        testcases_column = "<br>".join(testcases_column) if testcases_column else "➖"

        example_rows.append(
            f"| {entry['component']} "
            f"| {entry['description']} "
            f"| {children_list} "
            f"| {testcases_column} "
            f"| {entry['since']} |"
        )

    table_plugins = "\n".join(plugin_rows)
    table_examples = "\n".join(example_rows)

    # Read and update README content
    with open(README_PATH, "r", encoding="utf-8") as file:
        readme_content = file.read()

    # Insert plugins table
    start_idx = readme_content.find(START_MARKER)
    end_idx = readme_content.find(END_MARKER)
    if (start_idx == -1) or (end_idx == -1):
        raise ValueError("Start or End marker for plugins not found in README.md")

    readme_content = (
        readme_content[: start_idx + len(START_MARKER)]
        + f"\n\n{table_header_plugins}\n{table_plugins}\n\n"
        + readme_content[end_idx:]
    )

    # Insert examples table
    start_idx = readme_content.find(EXAMPLES_START_MARKER)
    end_idx = readme_content.find(EXAMPLES_END_MARKER)
    if (start_idx == -1) or (end_idx == -1):
        raise ValueError("Start or End marker for examples not found in README.md")

    readme_content = (
        readme_content[: start_idx + len(EXAMPLES_START_MARKER)]
        + f"\n\n{table_header_examples}\n{table_examples}\n\n"
        + readme_content[end_idx:]
    )

    # Write back to README.md
    with open(README_PATH, "w", encoding="utf-8") as file:
        file.write(readme_content)

    logging.info("✅ README.md updated successfully!")


if __name__ == "__main__":
    start_time = time.time()

    test_results = run_pytest()  # Run full tests to capture pass/fail results
    metadata_plugins = extract_metadata(PLUGIN_DIR, "plugins")
    metadata_examples = extract_metadata(EXAMPLES_DIR, "examples")
    update_readme(metadata_plugins, metadata_examples, test_results)

    logging.info(f"⏳ Total execution time: {time.time() - start_time:.2f}s")
