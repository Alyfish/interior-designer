#!/usr/bin/env python3
"""
Simple script to set up the .env file with proper formatting
"""

import os

def create_env_file():
    """Create a properly formatted .env file with placeholder values"""
    
    # Get the path to the .env file (in parent directory)
    env_path = os.path.join("..", ".env")
    
    # Define the content for the .env file - USE ONLY PLACEHOLDERS
    env_content = """OPENAI_API_KEY=your_openai_api_key_here
SERP_API_KEY=your_serpapi_key_here
IMGBB_API_KEY=your_imgbb_api_key_here
REVERSE_IMAGE_SEARCH=on
"""
    
    try:
        # Write the content to the .env file
        with open(env_path, 'w') as f:
            f.write(env_content.strip())
        
        print("✅ Successfully created .env file template!")
        print("📁 Location:", os.path.abspath(env_path))
        print("\n📋 Template created with placeholder values:")
        print(env_content.strip())
        
        print("\n⚠️ IMPORTANT: Replace ALL placeholder values with your actual API keys!")
        print("🔑 Get your API keys from:")
        print("   • OpenAI: https://platform.openai.com/api-keys")
        print("   • SerpAPI: https://serpapi.com/manage-api-key")
        print("   • ImgBB: https://api.imgbb.com/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

if __name__ == "__main__":
    create_env_file() 