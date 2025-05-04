import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Interior Designer: Object Detection and Product Matching")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for results")
    parser.add_argument("--api", type=str, choices=["laozhang", "perplexity"], default="laozhang", 
                        help="API to use for furniture analysis")
    parser.add_argument("--serve", action="store_true", help="Start Streamlit web interface")
    args = parser.parse_args()

    # If --serve is provided, start the Streamlit app
    if args.serve:
        import subprocess
        subprocess.run([
            "streamlit", "run", 
            os.path.join(os.path.dirname(__file__), "ui", "app.py"),
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
        return

    # Process a single image
    if args.image:
        from interior_designer.connector import InteriorDesignConnector
        
        # Get API keys from environment variables
        laozhang_api_key = os.environ.get("LAOZHANG_API_KEY", "sk-JDMYnZIoNuIHnw560b1a618f3a1c4d2eA7778577012dAeF8")
        perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY", "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")
        
        # Create connector
        connector = InteriorDesignConnector(
            laozhang_api_key=laozhang_api_key,
            perplexity_api_key=perplexity_api_key
        )
        
        # Process the image
        use_perplexity = args.api == "perplexity"
        results = connector.process_image(args.image, use_perplexity=use_perplexity)
        
        print("\nProcessing completed successfully!")
        print(f"Segmented image: {results['segmented_image_path']}")
        print(f"Furniture analysis: {results['analysis_text_path']}")
        print(f"JSON results: {results['analysis_json_path']}")
        
        # Print summary of findings
        print("\nDetected objects:")
        object_counts = {}
        for obj in results["objects"]:
            class_name = obj.get("class", "unknown")
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {class_name}: {count}")
        
        print("\nFurniture items with product matches:")
        for item in results["furniture_items"]:
            valid_urls = [url for url in item.product_urls if url.is_valid]
            print(f"  - {item.category}: {len(valid_urls)} product matches")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 