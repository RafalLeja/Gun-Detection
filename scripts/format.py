import os
import subprocess


def format():
    files = subprocess.check_output(["git", "ls-files"]).decode().splitlines()
    python_files = [f for f in files if f.endswith(".py") and os.path.exists(f)]

    if python_files:
        subprocess.run(
            ["autoflake", "--in-place", "--remove-all-unused-imports", "--remove-unused-variables"] + python_files
        )
        subprocess.run(["isort", "--profile", "black"] + python_files)
        subprocess.run(
            ["autopep8", "--in-place", "--aggressive", "--aggressive", "--max-line-length=120"] + python_files
        )
        subprocess.run(["black", "--line-length=120"] + python_files)
        subprocess.run(["flake8", "--max-line-length=120", "--extend-ignore=E203,W503"] + python_files)
        return 1


if __name__ == "__main__":
    format()
