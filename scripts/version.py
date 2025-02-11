import datetime
from pathlib import Path

BASE_VERSION = "0.1.0"


def generate_version():
    date = datetime.datetime.now().strftime("%Y%m%d")
    # try:
    #     git_hash = subprocess.check_output(
    #         ["git", "rev-parse", "--short", "HEAD"],
    #         stderr=subprocess.DEVNULL
    #     ).decode("utf-8").strip()
    # except Exception:
    #     git_hash = "nogit"
    return f"{BASE_VERSION}.dev{date}"  # +{git_hash}"


def update_pyproject_toml(version):
    pyproject_path = Path("../pyproject.toml")
    content = pyproject_path.read_text()
    # Replace the version field
    new_content = []
    for line in content.splitlines():
        if line.startswith("version = "):
            new_content.append(f'version = "{version}"')
        else:
            new_content.append(line)
    pyproject_path.write_text("\n".join(new_content))


if __name__ == "__main__":
    version = generate_version()
    update_pyproject_toml(version)
    print(f"Updated version to: {version}")
