"""
Object-Product Integration Module

This module handles the integration between detected objects and product matching,
including CLIP embedding extraction and side-by-side display of results.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)


def parse_product_line(product_line: str) -> Dict[str, str]:
    """
    Parse a product line to extract title, price, store, and link.
    
    Args:
        product_line: Product line string
        
    Returns:
        Dictionary with parsed product information
    """
    # Default values
    result = {
        'title': 'Product',
        'price': 'Price not available',
        'store': 'Store',
        'link': '#'
    }
    
    try:
        # Common patterns in product results
        # Example: "1. Modern Chair - Contemporary Design - Price: $449.99 - Store: IKEA - Link: https://..."
        
        # Extract title (usually after number and before first hyphen or price)
        import re
        
        # Remove leading number if present
        line = re.sub(r'^\d+\.\s*', '', product_line)
        
        # Try to extract price
        price_match = re.search(r'(?:Price:\s*)?(\$[\d,]+\.?\d*)', line)
        if price_match:
            result['price'] = price_match.group(1)
        
        # Try to extract store
        store_match = re.search(r'Store:\s*([^-]+)', line)
        if store_match:
            result['store'] = store_match.group(1).strip()
        
        # Try to extract link
        link_match = re.search(r'https?://[^\s]+', line)
        if link_match:
            result['link'] = link_match.group(0)
        
        # Extract title (everything before price or store)
        title_end = line.find('Price:') if 'Price:' in line else line.find('Store:')
        if title_end > 0:
            title = line[:title_end].strip(' -')
        else:
            # If no price/store markers, take first part
            parts = line.split(' - ')
            title = parts[0] if parts else line
        
        result['title'] = title.strip()
        
    except Exception as e:
        logger.error(f"Error parsing product line: {e}")
    
    return result


def process_detected_objects_with_features(
    selected_objects: List[Dict[str, Any]], 
    original_image: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Process detected objects to extract visual features and prepare for product search.
    
    Args:
        selected_objects: List of detected objects with bounding boxes
        original_image: Original image as numpy array
        
    Returns:
        List of objects enhanced with CLIP embeddings and captions
    """
    try:
        from vision_features import extract_clip_embedding, get_caption_json
        
        processed_objects = []
        
        for idx, obj in enumerate(selected_objects):
            try:
                # Extract object crop
                x1, y1, x2, y2 = obj['bbox']
                cropped_img = original_image[y1:y2, x1:x2]
                
                # Save crop temporarily
                crop_path = Path(f"temp_crop_{idx}.jpg")
                import cv2
                cv2.imwrite(str(crop_path), cropped_img)
                
                # Extract CLIP embedding
                embedding = extract_clip_embedding(str(crop_path))
                
                # Generate caption
                caption_data = get_caption_json(cropped_img, designer=True)
                
                # Enhanced object data
                enhanced_obj = {
                    **obj,
                    'crop_path': str(crop_path),
                    'clip_embedding': embedding,
                    'caption_data': caption_data,
                    'caption': caption_data.get('caption', ''),
                    'style': caption_data.get('style', 'unknown'),
                    'material': caption_data.get('material', 'unknown'),
                    'color': caption_data.get('colour', 'unknown')
                }
                
                processed_objects.append(enhanced_obj)
                
            except Exception as e:
                logger.error(f"Error processing object {idx}: {e}")
                processed_objects.append(obj)  # Add original if processing fails
                
        return processed_objects
        
    except Exception as e:
        logger.error(f"Error in process_detected_objects_with_features: {e}")
        return selected_objects


def search_products_for_object(
    obj: Dict[str, Any],
    search_method: str = 'hybrid',
    max_results: int = 5,
    use_enhanced_search: bool = True
) -> Dict[str, Any]:
    """
    Search for products matching a single detected object.
    
    Args:
        obj: Object dictionary with features
        search_method: 'text_only', 'visual_only', or 'hybrid'
        max_results: Maximum number of products to return
        use_enhanced_search: Whether to use enhanced search with better query construction
        
    Returns:
        Dictionary with search results and metadata
    """
    try:
        if use_enhanced_search:
            # Use enhanced search pipeline
            from enhanced_product_search import EnhancedProductSearcher
            from new_product_matcher import search_products_enhanced
            
            searcher = EnhancedProductSearcher()
            
            # Construct optimized query
            optimized_query = searcher.construct_optimized_query(
                obj.get('caption_data', {}),
                obj.get('class', 'furniture'),
                use_gpt_caption=True
            )
            
            # Search with enhanced method
            results = search_products_enhanced(
                query=optimized_query,
                style_info=obj.get('caption_data', {}),
                query_embedding=obj.get('clip_embedding'),
                search_method=search_method,
                image_path=obj.get('crop_path')
            )
            
            # Rank results if they're in list format
            if isinstance(results, list):
                results = searcher.rank_products_by_similarity(
                    results,
                    query_embedding=obj.get('clip_embedding'),
                    caption_data=obj.get('caption_data', {})
                )
            
            return {
                'object_id': obj.get('id', 0),
                'object_class': obj.get('class', 'unknown'),
                'caption': obj.get('caption', ''),
                'optimized_query': optimized_query,
                'products': results,
                'search_method': search_method
            }
        else:
            # Use original search method
            from new_product_matcher import search_products_enhanced
            
            # Prepare search query
            query = obj.get('caption', obj.get('class', 'furniture'))
            
            # Search with appropriate method
            if search_method == 'hybrid' and obj.get('clip_embedding') is not None:
                results = search_products_enhanced(
                    query=query,
                    style_info=obj.get('caption_data', {}),
                    query_embedding=obj.get('clip_embedding'),
                    search_method='hybrid',
                    image_path=obj.get('crop_path')
                )
            elif search_method == 'visual_only' and obj.get('crop_path'):
                # Use reverse image search if available
                results = search_products_enhanced(
                    query=query,
                    style_info=obj.get('caption_data', {}),
                    search_method='text_only',  # Will upgrade to reverse image if available
                    image_path=obj.get('crop_path')
                )
            else:
                # Text-only search
                results = search_products_enhanced(
                    query=query,
                    style_info=obj.get('caption_data', {}),
                    search_method='text_only'
                )
                
            return {
                'object_id': obj.get('id', 0),
                'object_class': obj.get('class', 'unknown'),
                'caption': obj.get('caption', ''),
                'products': results,
                'search_method': search_method
            }
        
    except Exception as e:
        logger.error(f"Error searching products for object: {e}")
        return {
            'object_id': obj.get('id', 0),
            'object_class': obj.get('class', 'unknown'),
            'error': str(e),
            'products': []
        }


def search_products_parallel(
    objects: List[Dict[str, Any]],
    search_method: str = 'hybrid',
    max_workers: int = 3,
    use_enhanced_search: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for products for multiple objects in parallel.
    
    Args:
        objects: List of processed objects
        search_method: Search method to use
        max_workers: Maximum parallel workers
        use_enhanced_search: Whether to use enhanced search
        
    Returns:
        List of search results for each object
    """
    if use_enhanced_search:
        # Use enhanced search pipeline for better performance
        try:
            from enhanced_product_search import create_enhanced_search_pipeline
            return create_enhanced_search_pipeline(
                objects,
                search_method=search_method,
                use_caching=True
            )
        except ImportError:
            logger.warning("Enhanced search not available, falling back to standard search")
    
    # Standard parallel search
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_obj = {
            executor.submit(search_products_for_object, obj, search_method, use_enhanced_search=use_enhanced_search): obj
            for obj in objects
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_obj):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                obj = future_to_obj[future]
                logger.error(f"Error searching for object {obj.get('id')}: {e}")
                results.append({
                    'object_id': obj.get('id', 0),
                    'error': str(e),
                    'products': []
                })
                
    # Sort by object ID to maintain order
    results.sort(key=lambda x: x.get('object_id', 0))
    return results


def display_object_with_products(
    obj: Dict[str, Any],
    search_result: Dict[str, Any],
    container=None
):
    """
    Display a detected object alongside its matching products in a side-by-side layout.
    
    Args:
        obj: Object dictionary with crop image
        search_result: Product search results
        container: Streamlit container to render in
    """
    if container is None:
        container = st
        
    # Create columns: [Object Image | Product Results]
    col1, col2 = container.columns([1, 3])
    
    with col1:
        # Display object in a styled container
        st.markdown(
            """
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6;">
                <h4 style="margin: 0 0 10px 0; color: #495057;">Detected Object</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if obj.get('crop_path') and Path(obj['crop_path']).exists():
            st.image(obj['crop_path'], use_container_width=True)
        
        # Display object details in a clean format
        details_html = f"""
        <div style="background: white; padding: 10px; border-radius: 8px; margin-top: 10px;">
            <p style="margin: 5px 0;"><strong>Type:</strong> {obj.get('class', 'Unknown')}</p>
            <p style="margin: 5px 0;"><strong>Confidence:</strong> {obj.get('confidence', 0):.1%}</p>
        """
        
        if obj.get('caption_data'):
            caption_data = obj['caption_data']
            details_html += f"""
            <hr style="margin: 10px 0; border: none; border-top: 1px solid #eee;">
            <p style="margin: 5px 0;"><strong>Style:</strong> {caption_data.get('style', 'Unknown')}</p>
            <p style="margin: 5px 0;"><strong>Material:</strong> {caption_data.get('material', 'Unknown')}</p>
            <p style="margin: 5px 0;"><strong>Color:</strong> {caption_data.get('colour', 'Unknown')}</p>
            """
        
        # Show optimized query if available
        if search_result.get('optimized_query'):
            details_html += f"""
            <hr style="margin: 10px 0; border: none; border-top: 1px solid #eee;">
            <p style="margin: 5px 0;"><strong>Search Query:</strong><br><em>{search_result['optimized_query']}</em></p>
            """
        
        details_html += "</div>"
        st.markdown(details_html, unsafe_allow_html=True)
    
    with col2:
        # Display product results
        st.markdown("### Matching Products")
        
        if search_result.get('error'):
            st.error(f"Search error: {search_result['error']}")
        elif search_result.get('products'):
            # Parse products if it's a string
            products = search_result['products']
            if isinstance(products, str):
                # Parse product lines
                product_lines = [line.strip() for line in products.strip().split('\n') if line.strip()]
                
                # Filter out section headers
                product_lines = [line for line in product_lines if not line.startswith('===')]
                
                # Create 3-column grid for products
                prod_cols = st.columns(3)
                
                for idx, product_line in enumerate(product_lines[:6]):  # Show max 6 products
                    col_idx = idx % 3
                    with prod_cols[col_idx]:
                        # Parse product details from line
                        product_info = parse_product_line(product_line)
                        
                        # Create product card
                        with st.container():
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 8px; height: 200px;">
                                    <h4 style="margin: 0 0 10px 0; font-size: 16px;">{product_info['title']}</h4>
                                    <p style="color: #666; margin: 5px 0;">{product_info['price']}</p>
                                    <p style="color: #888; margin: 5px 0; font-size: 14px;">{product_info['store']}</p>
                                    <a href="{product_info['link']}" target="_blank" style="color: #0066cc; text-decoration: none;">
                                        View Product →
                                    </a>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            else:
                st.info("No products found for this object")
        else:
            st.info("No products found for this object")


def display_all_objects_with_products(
    objects: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    container=None
):
    """
    Display all detected objects with their product matches.
    
    Args:
        objects: List of processed objects
        search_results: List of search results
        container: Streamlit container
    """
    if container is None:
        container = st
        
    # Create a mapping of object_id to search results
    results_map = {r['object_id']: r for r in search_results}
    
    for obj in objects:
        obj_id = obj.get('id', 0)
        search_result = results_map.get(obj_id, {'products': []})
        
        # Add some spacing between objects
        container.markdown("---")
        
        # Display this object with its products
        display_object_with_products(obj, search_result, container)
        
        # Add extra spacing
        container.markdown("<br>", unsafe_allow_html=True)


def create_product_search_ui(container=None):
    """
    Create the UI controls for product search configuration.
    
    Args:
        container: Streamlit container
        
    Returns:
        Dictionary with search configuration
    """
    if container is None:
        container = st
        
    with container.expander("🔍 Product Search Settings", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            search_method = st.selectbox(
                "Search Method",
                ["hybrid", "text_only", "visual_only"],
                help="Hybrid combines text and visual similarity"
            )
            
        with col2:
            max_results = st.slider(
                "Max Products per Object",
                min_value=3,
                max_value=10,
                value=5
            )
            
        with col3:
            price_range = st.select_slider(
                "Price Range",
                options=["Any", "$", "$$", "$$$", "$$$$"],
                value="Any"
            )
            
        with col4:
            use_enhanced = st.checkbox(
                "Enhanced Search",
                value=True,
                help="Use improved query construction and result ranking"
            )
            
    return {
        'search_method': search_method,
        'max_results': max_results,
        'price_range': price_range,
        'use_enhanced_search': use_enhanced
    }