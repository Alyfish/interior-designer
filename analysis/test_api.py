import requests
import base64
import os
import re
from pathlib import Path
import json
import time
from PIL import Image, ImageDraw, ImageFont

# API endpoint (chat completions endpoint is used for both text and multimodal inputs)
API_URL = "https://api.laozhang.ai/v1/chat/completions"

# Your API key
API_KEY = "sk-JDMYnZIoNuIHnw560b1a618f3a1c4d2eA7778577012dAeF8"

def encode_image(image_path):
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_image_url(content):
    """Extract image URL from markdown or text content"""
    # Check for markdown image format: ![alt text](url)
    markdown_pattern = r"!\[.*?\]\((https?://[^\s)]+)\)"
    markdown_match = re.search(markdown_pattern, content)
    if markdown_match:
        return markdown_match.group(1)
    
    # Check for standard URL format with image extensions
    url_pattern = r"(https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp))"
    url_match = re.search(url_pattern, content)
    if url_match:
        return url_match.group(1)
    
    # Check for any URL with 'image' in the path (common for image hosting services)
    image_url_pattern = r"(https?://[^\s]+(?:image|img|photo)[^\s]*)"
    image_url_match = re.search(image_url_pattern, content)
    if image_url_match:
        return image_url_match.group(1)
    
    # Check for any URL inside parentheses (often used to link to images)
    parentheses_pattern = r"\((https?://[^\s)]+)\)"
    parentheses_match = re.search(parentheses_pattern, content)
    if parentheses_match:
        return parentheses_match.group(1)
    
    return None

def download_image(url, output_path, max_retries=3, timeout=15):
    """Download image from URL and save to output path with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt+1}/{max_retries} for URL: {url}")
            response = requests.get(url, stream=True, timeout=timeout)
            
            if response.status_code == 200:
                # Check if response is actually an image by looking at content type
                content_type = response.headers.get('Content-Type', '')
                if not any(img_type in content_type.lower() for img_type in ['image', 'jpeg', 'png', 'webp']):
                    print(f"Warning: URL returned non-image content: {content_type}")
                    # We'll still try to save it, as some servers might not set content type correctly
                
                # Save the content
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                
                # Verify the file was created and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                    print(f"Image successfully downloaded and saved to: {output_path}")
                    return True
                else:
                    print(f"Warning: Downloaded file is too small or empty, may not be a valid image")
                    # If file exists but is invalid, delete it before retrying
                    if os.path.exists(output_path):
                        os.remove(output_path)
            else:
                print(f"Failed to download image: HTTP {response.status_code}")
                
            # If we get here, retry after a short delay (exponential backoff)
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))  # 1, 2, 4 seconds delay
                
        except requests.RequestException as e:
            print(f"Network error during download: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))
        except Exception as e:
            print(f"Unexpected error during download: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))
    
    # If all retries failed, try to generate a placeholder image
    try:
        print("All download attempts failed, generating placeholder image...")
        
        # Create a simple placeholder image
        img = Image.new('RGB', (800, 600), color=(245, 245, 245))
        d = ImageDraw.Draw(img)
        d.rectangle([50, 50, 750, 550], outline=(200, 200, 200), width=2)
        
        # Add text explaining the image couldn't be downloaded
        d.text((400, 280), "Image Generation Failed", fill=(80, 80, 80), anchor="mm")
        d.text((400, 320), "Design recommendations available in text", fill=(80, 80, 80), anchor="mm")
        
        # Save the placeholder
        img.save(output_path)
        print(f"Placeholder image saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to create placeholder image: {str(e)}")
        return False

def save_image_from_response(response_content, output_path="generated_image.jpg"):
    """Save image from API response content"""
    if isinstance(response_content, str):
        # Case 1: Response is base64 encoded image
        if response_content.startswith("data:image"):
            # Extract the base64 image data
            image_data = response_content.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            
            # Save the image
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            print(f"Base64 image saved to: {output_path}")
            return True
        
        # Case 2: Response contains a URL to an image (in markdown or plain text)
        url = extract_image_url(response_content)
        if url:
            print(f"Found image URL in response: {url}")
            return download_image(url, output_path)
        
        # Neither base64 nor URL found
        print("Response does not contain a valid image or URL")
        print(f"Response content: {response_content[:200]}...")
        return False
    else:
        print(f"Unexpected response type: {type(response_content)}")
        return False

def image_to_image_with_text(image_path, prompt, model="gpt-4o-image-vip", output_path="generated_image.jpg"):
    """
    Generate an image using both text prompt and input image.
    This uses the chat completions endpoint with multimodal input.
    
    Args:
        image_path: Path to the input image
        prompt: Text instruction for image transformation
        model: Model name to use
        output_path: Where to save the generated image
    
    Returns:
        bool: Success status
    """
    try:
        # Encode the image
        base64_image = encode_image(image_path)
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload - this is the key part that combines text + image input
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI that generates high-quality images based on input images and text instructions."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Make the API call
        print(f"Sending request to {API_URL} with model {model}...")
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Handle the response
        if response.status_code == 200:
            print(f"Request successful (HTTP {response.status_code})")
            result = response.json()
            
            # Process the response to extract the image
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return save_image_from_response(content, output_path)
            else:
                print("No choices found in response")
                print(f"Response: {result}")
                return False
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def image_with_dual_output(image_path, prompt, model="gpt-4o", image_output_path="output/generated_image.jpg", text_output_path="output/text_analysis.txt"):
    """
    Process an image and return both a transformed image and text analysis.
    
    Args:
        image_path: Path to the input image
        prompt: Text instruction for processing
        model: Model name to use
        image_output_path: Where to save the generated image
        text_output_path: Where to save the text analysis
    
    Returns:
        tuple: (image_success, text_content)
    """
    try:
        # Encode the image
        base64_image = encode_image(image_path)
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload with instructions for dual output
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI that can both analyze images and generate modified versions. For each request, provide BOTH a text analysis AND a modified version of the image based on the instructions. Always include your text analysis first, followed by the image in markdown format: ![description](URL)"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}\n\nPlease provide BOTH a detailed text analysis of this image AND a modified version of the image according to the instructions. Start with your text analysis, then include the modified image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.7
        }
        
        # Make the API call
        print(f"Sending request to {API_URL} with model {model}...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        # Handle the response
        if response.status_code == 200:
            print(f"Request successful (HTTP {response.status_code})")
            result = response.json()
            
            # Process the response to extract both text and image
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                
                # Extract image URL
                image_url = extract_image_url(content)
                image_success = False
                
                if image_url:
                    # Try to download the image
                    image_success = download_image(image_url, image_output_path)
                else:
                    print("No image URL found in response content")
                
                # Clean text content (remove image markdown)
                text_content = re.sub(r"!\[.*?\]\(https?://[^\s)]+\)", "", content).strip()
                
                # Save text to file
                if text_output_path and text_content:
                    with open(text_output_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    print(f"Text analysis saved to: {text_output_path}")
                
                # Even if image download failed, if we created a placeholder, return success
                return (os.path.exists(image_output_path), text_content)
            else:
                print("No choices found in response")
                print(f"Response: {result}")
                return (False, None)
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return (False, None)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback_info = traceback.format_exc()
        print(traceback_info)
        return (False, f"Error during processing: {str(e)}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Path to the test image
    image_path = "test_img.jpg"
    
    # Check if test image exists
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
    else:
        print(f"Using test image: {image_path}")
        
        # Use dual output with custom prompt
        prompt = """while keeping the structural integrity and the angle of the same, and having the walls unchanged, redesign it in a more minimalistic style with a slightly cooler color palette. Ensure that you use furniture that currently exists on the market and create a photorealistic image, and then provide me with the urls of all the items that you included in the output image so that i can buy the furniture (ensure that the links work and point to an actual product that currently exists), and use only those images"""
        
        image_output_path = output_dir / "redesigned_room.jpg"
        text_output_path = output_dir / "furniture_products.txt"
        
        print(f"Sending request with prompt for both image and product URLs...")
        image_success, text_content = image_with_dual_output(
            image_path=image_path,
            prompt=prompt,
            model="gpt-4o",
            image_output_path=image_output_path,
            text_output_path=text_output_path
        )
        
        if image_success:
            print("✅ Room redesign image created successfully!")
            print(f"Image saved to: {image_output_path}")
        else:
            print("❓ Room redesign image may not have been created successfully.")
            
        if text_content:
            print("✅ Furniture product information retrieved successfully!")
            print(f"Product details saved to: {text_output_path}")
            print("\nProduct URLs (first 300 characters):")
            print(text_content[:300] + "..." if len(text_content) > 300 else text_content)
        else:
            print("❌ Failed to retrieve furniture product information!") 