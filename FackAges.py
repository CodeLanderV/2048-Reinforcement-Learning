import subprocess
import sys

def install_requirements(requirements_file="requirements.txt"):
    """
    Installs packages listed in a requirements file.
    """
    try:
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("All packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: The file '{requirements_file}' was not found.")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()