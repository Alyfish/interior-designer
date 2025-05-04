import os
import sys
import json
import time
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import re
import streamlit.components.v1 as components
import requests

# Add the root directory to the Python path to allow importing from interior_designer package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now import from interior_designer package
from interior_designer.connector import InteriorDesignConnector
from interior_designer.ui.interactive_ui import create_interactive_html
from interior_designer.product_finder import find_products_sync

# Default API keys - hardcoded instead of user input
LAOZHANG_API_KEY = "sk-JDMYnZIoNuIHnw560b1a618f3a1c4d2eA7778577012dAeF8"
PERPLEXITY_API_KEY = "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"
SERP_API_KEY = "e5fd89a1eb892fcd714959c6d6824e5d87c2cd2e8904f7b8cd9124c13fa6d86d"

# Set page config
st.set_page_config(
    page_title="Interior Designer: Detection & Shopping Tool",
    page_icon="🏠",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        margin-bottom: 0.5rem;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .furniture-item {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .product-link {
        border-left: 3px solid #4CAF50;
        padding-left: 10px;
        margin-bottom: 10px;
        background-color: white;
    }
    .google-shopping-product {
        border-left: 3px solid #4285F4;
        padding-left: 10px;
        margin-bottom: 10px;
        background-color: #f8f9fa;
    }
    .serp-api-product {
        border-left: 3px solid #FF5722;
        padding-left: 10px;
        margin-bottom: 10px;
        background-color: #fff8f5;
    }
    .invalid-product {
        border-left: 3px solid #f44336;
    }
    .product-retailer {
        font-weight: bold;
        color: #333;
    }
    .product-price {
        color: #4CAF50;
        font-weight: bold;
    }
    footer {
        visibility: hidden;
    }
    .object-list {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
    }
    .detection-stats {
        background-color: #f0f8ff;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .canvas-container {
        position: relative;
        margin: 0 auto;
        border: 1px solid #eee;
        overflow: hidden;
    }
    .canvas-layer {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        cursor: pointer;
    }
    .api-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 5px;
    }
    .laozhang-badge {
        background-color: #e6f7ff;
        color: #0066cc;
        border: 1px solid #0066cc;
    }
    .perplexity-badge {
        background-color: #e6f4ea;
        color: #137333;
        border: 1px solid #137333;
    }
    .serp-badge {
        background-color: #fff0e0;
        color: #994500;
        border: 1px solid #994500;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to extract product ID from Google Shopping URL
def extract_google_product_id(url):
    try:
        # Case 1: URLs with pid parameter
        pid_match = re.search(r'pid:(\d+)', url)
        if pid_match:
            return pid_match.group(1)
        
        # Case 2: URLs with product ID in path
        product_match = re.search(r'/product/(\d+)', url)
        if product_match:
            return product_match.group(1)
            
        # Case 3: URLs with product ID after /product/1?
        product_one_match = re.search(r'/product/1\?.*?pid:(\d+)', url)
        if product_one_match:
            return product_one_match.group(1)
        
        return "Unknown"
    except Exception:
        return "Unknown"

# Function to call SerpAPI for image search
def search_with_serpapi(image_path, search_query=None):
    try:
        # Read the image file
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
        
        # Prepare the request
        url = "https://serpapi.com/v1/reverse_image_search"
        
        # Create a multipart form-data payload
        files = {'image': ('image.jpg', img_data, 'image/jpeg')}
        
        # Add parameters to the payload
        data = {'api_key': SERP_API_KEY}
        if search_query:
            data['query'] = search_query
        
        # Make the request
        print(f"Calling SerpAPI for image search with query: {search_query}")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"SerpAPI error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error calling SerpAPI: {str(e)}")
        return None

# Function to extract products from SerpAPI response
def extract_serp_products(serp_results, obj_class):
    if not serp_results or "shopping_results" not in serp_results:
        return []
    
    products = []
    for item in serp_results.get("shopping_results", [])[:20]:  # Limit to top 20 results
        product_url = item.get("link", "")
        price = item.get("price", "Unknown price")
        title = item.get("title", "Product")
        source = item.get("source", "")
        
        # Create product details
        products.append({
            "url": product_url,
            "retailer": source,
            "price": price,
            "description": title,
            "is_valid": True,
            "error_message": "",
            "source": "SerpAPI"
        })
    
    return products

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_objects' not in st.session_state:
    st.session_state.selected_objects = []
if 'show_products' not in st.session_state:
    st.session_state.show_products = False
if 'all_products' not in st.session_state:
    st.session_state.all_products = {}

def main():
    # Main content
    st.title("Interior Designer: Detection & Shopping Tool")
    st.markdown("Upload an interior image to detect furniture and find matching products")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Process the image if uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Process image button
        process_button = st.button("Detect Objects")
        
        if process_button:
            with st.spinner("Processing image..."):
                # Save uploaded file temporarily
                temp_img_path = "temp_uploaded_image.jpg"
                with open(temp_img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize connector with hardcoded API keys
                connector = InteriorDesignConnector(
                    laozhang_api_key=LAOZHANG_API_KEY,
                    perplexity_api_key=PERPLEXITY_API_KEY
                )
                
                # Process the image - detect objects only, don't analyze furniture yet
                try:
                    # Only detect objects
                    image, objects, segmented_image_path = connector.detect_objects(temp_img_path)
                    
                    # Store results in session state
                    st.session_state.image = image
                    st.session_state.objects = objects
                    st.session_state.segmented_image_path = segmented_image_path
                    st.session_state.temp_img_path = temp_img_path
                    st.session_state.processed = True
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        
        # Display interactive selection UI if processed
        if st.session_state.processed:
            # Display segmented image in the second column
            with col2:
                st.subheader("Detected Objects")
                seg_image = Image.open(st.session_state.segmented_image_path)
                st.image(seg_image, use_container_width=True)
            
            st.subheader("Select Objects")
            st.markdown("Click on the furniture items you want to find products for:")
            
            # Create interactive HTML for object selection
            interactive_html = create_interactive_html(st.session_state.image, st.session_state.objects)
            
            # Display interactive component
            component_height = min(st.session_state.image.shape[0] * 1.2, 800)
            components.html(
                interactive_html,
                height=component_height,
                scrolling=True
            )
            
            # Display object detection stats
            st.subheader("Detection Results")
            object_counts = {}
            for obj in st.session_state.objects:
                class_name = obj.get("class", "unknown")
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            # Display object counts in a horizontal layout
            st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
            cols = st.columns(4)
            for i, (class_name, count) in enumerate(sorted(object_counts.items(), key=lambda x: x[1], reverse=True)):
                col_idx = i % 4
                cols[col_idx].metric(class_name.title(), count)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Auto-select furniture items
            furniture_items = ["chair", "sofa", "couch", "dining table", "tv", "bed", "table", "bookshelf", "cabinet", "desk"]
            st.session_state.selected_objects = [obj for obj in st.session_state.objects 
                                              if obj.get("class", "").lower() in furniture_items]
            st.write(f"Automatically selected {len(st.session_state.selected_objects)} furniture items for product search")
            
            # Display selected objects info
            selection_container = st.container()
            with selection_container:
                if st.session_state.selected_objects:
                    selected_count = len(st.session_state.selected_objects)
                    st.success(f"Selected {selected_count} objects")
                    
                    # Show selected object classes
                    selected_classes = {}
                    for obj in st.session_state.selected_objects:
                        class_name = obj.get('class', 'unknown')
                        selected_classes[class_name] = selected_classes.get(class_name, 0) + 1
                    
                    st.write("Selected items:")
                    for class_name, count in selected_classes.items():
                        st.write(f"- {class_name.capitalize()}: {count}")
            
            # Save selected items button
            if st.button("Save Selected Items"):
                st.write("DEBUG: Save Selected Items clicked")
                st.write(f"DEBUG: current selected_objects: {st.session_state.selected_objects}")
                if not st.session_state.selected_objects:
                    st.warning("Please select at least one furniture item by clicking on it in the image above.")
                else:
                    st.write(f"DEBUG: about to save {len(st.session_state.selected_objects)} items")
                    # Initialize connector to get output_dir
                    connector = InteriorDesignConnector(
                        laozhang_api_key=LAOZHANG_API_KEY,
                        perplexity_api_key=PERPLEXITY_API_KEY
                    )
                    saved_dir = os.path.join(connector.output_dir, "saved_selections")
                    os.makedirs(saved_dir, exist_ok=True)
                    metadata = []
                    # Crop and save each selected object
                    for idx, obj in enumerate(st.session_state.selected_objects):
                        st.write(f"DEBUG: cropping object {idx} with id {obj.get('id')}")
                        class_name = obj.get('class', 'unknown')
                        contours = obj.get('contours', [])
                        # Compute bounding box
                        xs = [p[0] for contour in contours for p in contour]
                        ys = [p[1] for contour in contours for p in contour]
                        if not xs or not ys:
                            continue
                        x1, x2 = int(min(xs)), int(max(xs))
                        y1, y2 = int(min(ys)), int(max(ys))
                        crop = st.session_state.image[y1:y2, x1:x2]
                        file_name = f"selected_{idx}_{class_name}.jpg"
                        crop_path = os.path.join(saved_dir, file_name)
                        st.write(f"DEBUG: writing crop to {crop_path}")
                        cv2.imwrite(crop_path, crop)
                        metadata.append({
                            "id": obj.get('id'),
                            "class": class_name,
                            "confidence": obj.get('confidence', None),
                            "crop_path": crop_path,
                            "bbox": [x1, y1, x2, y2]
                        })
                    # Save metadata to JSON
                    meta_path = os.path.join(saved_dir, "metadata.json")
                    st.write(f"DEBUG: writing metadata to {meta_path}")
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(metadata, mf, indent=2)
                    st.success(f"Saved {len(metadata)} items to {saved_dir}")
                    st.write(metadata)

            # Find products button
            if st.button("Find Products"):
                furniture_to_search = st.session_state.selected_objects
                if not furniture_to_search:
                    st.warning("Please select at least one furniture item by clicking on it in the image above.")
                else:
                    st.write(f"Starting product search for {len(furniture_to_search)} furniture items using all available APIs...")
                    # Build results per selected object
                    results_by_object = {}
                    for idx, obj in enumerate(furniture_to_search):
                        st.write(f"Processing search for {obj.get('class')} #{idx+1}...")
                        class_name = obj.get('class', 'unknown')
                        # Compute bounding box
                        contours = obj.get('contours', [])
                        xs = [p[0] for contour in contours for p in contour]
                        ys = [p[1] for contour in contours for p in contour]
                        if not xs or not ys:
                            continue
                        x1, x2 = int(min(xs)), int(max(xs))
                        y1, y2 = int(min(ys)), int(max(ys))
                        # Crop object
                        crop = st.session_state.image[y1:y2, x1:x2]
                        # Save temp crop
                        crop_path = f"temp_crop_{idx}.jpg"
                        cv2.imwrite(crop_path, crop)
                        
                        # Add object metadata for detailed search
                        obj_meta = {
                            "class": class_name,
                            "confidence": obj.get('confidence', 0.0),
                            "bbox": [x1, y1, x2, y2],
                            "dimensions": f"{x2-x1}x{y2-y1} pixels",
                            "segmented_image": st.session_state.segmented_image_path
                        }
                        
                        # Call product finder API for this object
                        products = find_products_sync(crop_path, class_name, max_results=5, 
                                                     object_metadata=obj_meta)
                        results_by_object[f"{class_name}_{idx}"] = {
                            "metadata": {"id": obj.get('id'), "class": class_name, "confidence": obj.get('confidence')},
                            "products": products
                        }
                    st.session_state.all_products = results_by_object
                    st.session_state.show_products = True

            # Display products if available
            if st.session_state.show_products and st.session_state.all_products:
                st.header("Found matching products:")
                
                # Add source filtering
                available_sources = set()
                for obj_key, data in st.session_state.all_products.items():
                    for product in data.get('products', []):
                        if 'source' in product:
                            available_sources.add(product['source'])
                
                # Default to all sources selected
                if 'selected_sources' not in st.session_state:
                    st.session_state.selected_sources = list(available_sources)
                
                # Source filter UI
                source_options = sorted(list(available_sources))
                if source_options:
                    st.session_state.selected_sources = st.multiselect(
                        "Filter by source:", 
                        source_options,
                        default=st.session_state.selected_sources
                    )
                
                # Display results by object
                for obj_key, data in st.session_state.all_products.items():
                    metadata = data.get('metadata', {})
                    products = data.get('products', [])
                    
                    if products:
                        # Filter products by selected sources
                        filtered_products = [p for p in products if p.get('source') in st.session_state.selected_sources]
                        
                        if filtered_products:
                            st.subheader(f"{metadata.get('class', 'Object').title()} {metadata.get('id', '')}")
                            
                            # Create columns for product cards
                            cols = st.columns(3)
                            
                            # Display products in columns
                            for i, product in enumerate(filtered_products):
                                with cols[i % 3]:
                                    # Create an empty container for progressive loading
                                    card_container = st.empty()
                                    
                                    # Build card content
                                    title = product.get('title', 'Product')
                                    price = product.get('price', 'Price unavailable')
                                    image_url = product.get('image', '')
                                    retailer = product.get('retailer', product.get('source', 'Retailer'))
                                    product_url = product.get('url', '#')
                                    
                                    card_html = f"""
                                    <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px; height:100%;">
                                        <h3 style="font-size:16px; height:50px; overflow:hidden;">{title}</h3>
                                        {'<img src="' + image_url + '" style="max-width:100%; max-height:150px; display:block; margin:auto;">' if image_url else ''}
                                        <p style="font-weight:bold; margin-top:10px;">{price}</p>
                                        <p style="color:#666; font-size:12px;">From: {retailer}</p>
                                        <a href="{product_url}" target="_blank" style="display:block; text-align:center; background-color:#4CAF50; color:white; padding:8px; border-radius:4px; text-decoration:none; margin-top:10px;">View Product</a>
                                    </div>
                                    """
                                    
                                    # Update the container with the card content
                                    card_container.markdown(card_html, unsafe_allow_html=True)
                        else:
                            st.info(f"No products found for {metadata.get('class', 'object')} with selected filters.")
                    else:
                        st.warning(f"No products found for {metadata.get('class', 'object')}.")
    
    else:
        # Display demo information
        st.info("Upload an image to begin analyzing your interior space")
        
        # Example usage
        st.markdown("### How It Works")
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown("#### 1. Upload Image")
            st.markdown("Start by uploading your interior room photo")
        
        with cols[1]:
            st.markdown("#### 2. Detect Objects")
            st.markdown("AI identifies furniture, walls, floors and more")
        
        with cols[2]:
            st.markdown("#### 3. Find Products")
            st.markdown("Get links to similar furniture you can purchase")

if __name__ == "__main__":
    main() 