#!/usr/bin/env python3
"""Verify RAG system setup and configuration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    required = {
        "qdrant_client": "Qdrant client",
        "openai": "OpenAI",
        "tiktoken": "Tiktoken",
        "spacy": "spaCy",
        "streamlit": "Streamlit",
        "yaml": "PyYAML",
        "dotenv": "python-dotenv",
    }

    missing = []

    for package, name in required.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)

    return len(missing) == 0, missing


def check_spacy_model():
    """Check if spaCy model is installed."""
    print("\nChecking spaCy model...")

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("  ✓ en_core_web_sm installed")
        return True
    except OSError:
        print("  ✗ en_core_web_sm - MISSING")
        print("    Run: python -m spacy download en_core_web_sm")
        return False


def check_env_file():
    """Check if .env file exists and has required keys."""
    print("\nChecking environment configuration...")

    env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        print("  ✗ .env file not found")
        print("    Run: cp .env.example .env")
        return False

    print("  ✓ .env file exists")

    # Check for API key
    with open(env_path) as f:
        content = f.read()

    if "OPENAI_API_KEY=" in content and "your_openai_api_key_here" not in content:
        # Check if key is not just empty
        for line in content.split("\n"):
            if line.startswith("OPENAI_API_KEY=") and len(line.split("=")[1].strip()) > 0:
                print("  ✓ OpenAI API key configured")
                return True

        print("  ⚠ OpenAI API key appears empty")
        return False
    else:
        print("  ✗ OpenAI API key not configured")
        print("    Edit .env and add your API key")
        return False


def check_qdrant():
    """Check if Qdrant is running."""
    print("\nChecking Qdrant connection...")

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()

        print("  ✓ Qdrant is running")
        print(f"  ℹ Collections: {len(collections.collections)}")

        return True

    except Exception as e:
        print(f"  ✗ Cannot connect to Qdrant: {e}")
        print("    Run: make setup")
        return False


def check_configuration():
    """Check if configuration files exist."""
    print("\nChecking configuration files...")

    project_root = Path(__file__).parent.parent

    config_files = [
        "config/settings.yaml",
        "config/domain_vocabulary.json",
        "config/evaluation_config.yaml",
    ]

    all_exist = True

    for config_file in config_files:
        path = project_root / config_file
        if path.exists():
            print(f"  ✓ {config_file}")
        else:
            print(f"  ✗ {config_file} - MISSING")
            all_exist = False

    return all_exist


def check_directories():
    """Check if required directories exist."""
    print("\nChecking directory structure...")

    project_root = Path(__file__).parent.parent

    required_dirs = [
        "data/raw",
        "data/processed",
        "logs",
        "reports",
        "qdrant_storage",
    ]

    all_exist = True

    for dir_path in required_dirs:
        path = project_root / dir_path
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - MISSING")
            all_exist = False

    return all_exist


def main():
    """Run all checks."""
    print("=" * 60)
    print("RAG System Setup Verification")
    print("=" * 60)
    print()

    checks = [
        ("Dependencies", check_dependencies),
        ("spaCy Model", check_spacy_model),
        ("Environment File", check_env_file),
        ("Qdrant Connection", check_qdrant),
        ("Configuration", check_configuration),
        ("Directories", check_directories),
    ]

    results = {}

    for name, check_func in checks:
        result = check_func()

        # Handle tuple return from check_dependencies
        if isinstance(result, tuple):
            result = result[0]

        results[name] = result
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {name}")

    print()
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print()
        print("✅ All checks passed! Your RAG system is ready to use.")
        print()
        print("Next steps:")
        print("  1. Add documents to data/raw/")
        print("  2. Run: make ingest")
        print("  3. Run: make run-ui")
        return 0
    else:
        print()
        print("⚠️  Some checks failed. Please resolve the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
