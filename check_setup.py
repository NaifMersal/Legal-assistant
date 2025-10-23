"""Quick test script to verify the backend setup"""
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    print("🔍 Checking required files...\n")
    
    base_dir = Path(__file__).parent
    required_files = [
        "data/m3_legal_faiss.index",
        "data/saudi_laws_scraped.json",
        "app/utils/retriever.py",
        "app/utils/rag.py",
        ".env"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = base_dir / file_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_env():
    """Check if GOOGLE_API_KEY is set"""
    print("\n🔑 Checking environment variables...\n")
    
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if api_key and api_key != "your-google-api-key-here":
            print("✅ GOOGLE_API_KEY is set")
            return True
        else:
            print("❌ GOOGLE_API_KEY is not set or is still the example value")
            print("   Please edit .env file and add your actual API key")
            return False
    except ImportError:
        print("⚠️  python-dotenv not installed. Run: pip install python-dotenv")
        return False

def check_dependencies():
    """Check if key dependencies are installed"""
    print("\n📦 Checking dependencies...\n")
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("langchain_google_genai", "LangChain Google GenAI"),
        ("faiss", "FAISS"),
        ("FlagEmbedding", "BGE-M3"),
    ]
    
    all_installed = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_installed = False
    
    return all_installed

def main():
    print("=" * 60)
    print("🏛️  Legal Assistant Backend - Setup Verification")
    print("=" * 60)
    print()
    
    files_ok = check_files()
    env_ok = check_env()
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    
    if files_ok and env_ok and deps_ok:
        print("✅ All checks passed! You're ready to start the server.")
        print("\n🚀 To start the server, run:")
        print("   ./start_server.sh")
        print("\n   Or manually:")
        print("   python -m uvicorn app.main:app --reload")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        if not deps_ok:
            print("\n📥 To install dependencies, run:")
            print("   pip install -r requirements.txt")
        if not env_ok:
            print("\n🔑 To set up your API key:")
            print("   1. Get your key from: https://makersuite.google.com/app/apikey")
            print("   2. Edit .env file and replace 'your-google-api-key-here'")
    
    print("=" * 60)
    
    return 0 if (files_ok and env_ok and deps_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
