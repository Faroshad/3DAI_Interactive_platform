import subprocess
import sys

def fix_environment():
    """
    Fix the environment by downgrading protobuf to a compatible version.
    """
    print("Fixing environment...")
    
    # Uninstall current protobuf
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "protobuf"])
    
    # Install compatible version
    subprocess.run([sys.executable, "-m", "pip", "install", "protobuf==3.20.0"])
    
    print("Environment fixed. Please try running your application again.")

if __name__ == "__main__":
    fix_environment() 