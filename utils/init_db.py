import sys
from pathlib import Path

# Add the project root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from data_handler import DataHandler


def main():
    print("Initializing database...\n")
    try:
        # Reset and initialize database
        DataHandler.reset_database()
        print("\n✅ Database initialized successfully!")
    except Exception as e:
        print(f"\n❌ Error initializing database: {str(e)}")


if __name__ == "__main__":
    main()