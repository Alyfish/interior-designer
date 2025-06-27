#!/usr/bin/env python3
"""
🏡 AI Interior Designer with Furniture Object Detection and Product Matching

This app enables users to:
1. Upload interior images  
2. Auto-detect furniture using YOLOv8-Seg
3. Extract visual features using CLIP and BLIP
4. Search for matching products using LangChain agents and SerpAPI

Architecture:
- YOLOv8-Seg for object detection and segmentation
- CLIP for visual feature embeddings  
- BLIP for image captioning
- GPT-4 Vision for detailed descriptions
- LangChain agents with SerpAPI for product searches

Main components:
- new_object_detector.py: YOLOv8 detection logic
- vision_features.py: CLIP embeddings and BLIP captions  
- new_product_matcher.py: LangChain product search agents
"""

# Suppress warnings before imports
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*bottleneck.*")
warnings.filterwarnings("ignore", message=".*RuntimeError.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# --- Utility Function (import-safe) -----------------------------------------
# Placed before heavy Streamlit imports so other modules/tests can import this
# file just to access `get_config_status` without triggering Streamlit logic.

def get_config_status(
    serp_api_key: str,
    imgbb_api_key: str,
    reverse_image_search_enabled: bool,
):
    """Return a dictionary describing UI configuration readiness.

    Keys returned:
        search_mode: str  # 'hybrid_available', 'hybrid_degraded_no_upload',
                          # 'hybrid_degraded_no_visual_api', 'text_only_feature_off',
                          # 'text_only_serp_missing'
        serp_api_ready: bool
        visual_search_api_ready: bool
        image_upload_ready: bool
        reverse_image_search_enabled: bool
    """
    serp_ready = bool(serp_api_key)
    visual_ready = bool(serp_api_key)  # Using same key for visual search now
    upload_ready = bool(imgbb_api_key)

    if not serp_ready:
        mode = "text_only_serp_missing"
    elif not reverse_image_search_enabled:
        mode = "text_only_feature_off"
    else:
        # Feature flag ON, text key present
        if visual_ready and upload_ready:
            mode = "hybrid_available"
        elif not visual_ready:
            mode = "hybrid_degraded_no_visual_api"
        else:  # upload missing
            mode = "hybrid_degraded_no_upload"

    return {
        "search_mode": mode,
        "serp_api_ready": serp_ready,
        "visual_search_api_ready": visual_ready,
        "image_upload_ready": upload_ready,
        "reverse_image_search_enabled": reverse_image_search_enabled,
    }

# ---------------------------------------------------------------------------

# Fix PyTorch/Streamlit compatibility issues
import os
os.environ['TORCH_MULTIPROCESSING_SHARING_STRATEGY'] = 'file_system'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Additional torch settings to prevent event loop issues
try:
    import torch
    if hasattr(torch, 'set_sharing_strategy'):
        torch.set_sharing_strategy('file_system')
    torch.set_num_threads(1)
except ImportError:
    pass

# Standard library imports
import sys
import logging
import tempfile
import json
import time
import base64
import io
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional

# Third-party imports
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Local imports - Fixed import path
from vision_features import generate_blip_caption, extract_clip_embedding, get_caption_json, get_caption_json_gpt4v, merge_captions, polish_caption_with_openai
from new_object_detector import ObjectDetector
from new_product_matcher import (
    create_new_product_search_agent, 
    parse_agent_response_to_products, 
    search_products_hybrid, 
    search_products_enhanced,
    search_products_with_visual_similarity,
    ENABLE_REVERSE_IMAGE_SEARCH
)
from config import *
from config import REPLICATE_API_TOKEN

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Run API key check from config when Streamlit app starts
check_api_keys()

# Environment variables are now available from the imports above
# OPENAI_API_KEY and SERP_API_KEY are imported from config
SERP_KEY = SERP_API_KEY  # Rename for consistency with the rest of the code

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Interior Designer",
    page_icon="🏠",
    layout="wide"
)

# --- Custom CSS (Matching Original Style) ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
    }
    .detection-stats {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background-color: #ffffff;
    }
    .product-card img {
        width: 100%;
        height: 180px;
        object-fit: cover;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .product-card h3 {
        font-size: 14px;
        margin: 8px 0 4px 0;
        color: #333;
    }
    .product-card p {
        font-size: 12px;
        margin: 2px 0;
        color: #666;
    }
    .product-card a {
        display: inline-block;
        background-color: #ff4b4b;
        color: white;
        padding: 8px;
        border-radius: 4px;
        text-decoration: none;
        margin-top: auto; /* Pushes button to bottom */
    }
    .object-item {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .object-item:hover {
        border-color: #ff4b4b;
        background-color: #fff5f5;
    }
    .object-item.selected {
        border-color: #ff4b4b;
        background-color: #ffe5e5;
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
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
    .selected-item {
        background-color: #e8f5e8;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .product-price {
        font-size: 1.2em;
        font-weight: bold;
        color: #e74c3c;
    }
    .product-title {
        font-size: 1.1em;
        margin-bottom: 10px;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "image_processed": False,
        "original_image_cv": None,
        "detected_objects": [],
        "segmented_image_path": None,
        "selected_object_indices": [], # Store indices of st.session_state.detected_objects
        "product_search_results": {},
        "ai_product_agent": None,
        "temp_upload_path": None,
        "selected_objects": [], # For compatibility with original app
        'processed': False,
        'image': None,
        'objects': [],
        'temp_img_path': None,
        'object_detector': None,
        'product_agent': None,
        'detection_results': None,
        'interactive_html': None,
        'search_method': "Hybrid Search (Text + Visual)"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Helper Functions ---
@st.cache_resource
def get_object_detector():
    """Simple singleton for the detector"""
    # Correctly instantiate and return ObjectDetector
    try:
        detector = ObjectDetector() 
        logger.info("✅ Object detector initialized successfully")
        return detector
    except Exception as e:
        st.error(f"Failed to initialize Object Detector: {e}")
        logger.error(f"❌ Failed to initialize object detector: {e}")
        return None

@st.cache_resource
def get_product_agent():
    """Get product matching agent"""
    try:
        agent = create_new_product_search_agent()
        logger.info("✅ Product search agent initialized successfully")
        return agent
    except Exception as e:
        st.error(f"Failed to initialize Product Agent: {e}")
        logger.error(f"❌ Failed to initialize product agent: {e}")
        return None

def create_interactive_html(image, objects, products_data=None, debug_mode=False):
    """Create interactive HTML with proper coordinate handling"""
    import json
    import base64
    import io
    from PIL import Image
    
    # Convert image to base64 PNG
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Get actual image dimensions
    img_height, img_width = image.shape[:2]
    
    # Calculate display dimensions with constraints
    max_width = 1200
    max_height = 800
    scale = min(max_width / img_width, max_height / img_height, 1)
    display_width = int(img_width * scale)
    display_height = int(img_height * scale)
    
    logger.info(f"Image scaling - Original: {img_width}x{img_height}, Display: {display_width}x{display_height}, Scale: {scale}")
    
    # Color mapping for different object types
    object_colors = {
        'wall': '#8B4513',      # Brown
        'floor': '#DEB887',     # BurlyWood
        'chair': '#DAA520',     # GoldenRod
        'couch': '#BC8F8F',     # RosyBrown
        'sofa': '#D2691E',      # Chocolate
        'bed': '#FA8072',       # Salmon
        'dining table': '#CD853F', # Peru
        'tv': '#1E90FF',        # DodgerBlue
        'potted plant': '#228B22', # ForestGreen
        'custom': '#FF1493',    # DeepPink
    }
    
    # Prepare objects with validated scaling
    js_objects = []
    for i, obj in enumerate(objects):
        try:
            # Scale contours for display
            scaled_contours = []
            if 'contours' in obj and obj['contours']:
                for contour in obj['contours']:
                    if isinstance(contour, list):
                        scaled_contour = []
                        for point in contour:
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                # Ensure coordinates are valid
                                x = max(0, min(point[0], img_width))
                                y = max(0, min(point[1], img_height))
                                scaled_point = [int(x * scale), int(y * scale)]
                                scaled_contour.append(scaled_point)
                        if scaled_contour:
                            scaled_contours.append(scaled_contour)
            
            # Scale bbox for display
            scaled_bbox = None
            if 'bbox' in obj and len(obj['bbox']) == 4:
                x1, y1, x2, y2 = obj['bbox']
                # Validate bbox coordinates
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                scaled_bbox = [
                    int(x1 * scale),
                    int(y1 * scale),
                    int(x2 * scale),
                    int(y2 * scale)
                ]
            
            obj_data = {
                'id': obj.get('id', i),
                'class': obj.get('class', 'unknown'),
                'confidence': obj.get('confidence', 0.0),
                'contours': scaled_contours,
                'bbox': scaled_bbox,
                'source': obj.get('source', 'detection'),
            }
            js_objects.append(obj_data)
            
        except Exception as e:
            logger.error(f"Error processing object {i}: {e}")
            continue

    # Prepare products data for JavaScript
    js_products_data = {}
    if products_data:
        for obj_id, products in products_data.items():
            js_products_data[str(obj_id)] = products

    # Create the HTML with modal carousel
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive Image</title>
        <style>
            body {{ font-family: sans-serif; margin: 0; padding: 0; }}
            .container {{ position: relative; max-width: 100%; }}
            canvas {{ position: absolute; top: 0; left: 0; cursor: pointer; }}
            #tooltip {{
                position: absolute;
                display: none;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none; /* So it doesn't interfere with mouse events */
            }}
            #custom-box {{
                position: absolute;
                border: 2px dashed #ff4b4b;
                pointer-events: none;
                display: none;
            }}
        </style>
    </head>
    <body>
        <div class="container" id="container">
            <img id="base-image" src="data:image/jpeg;base64,{img_str}" alt="Uploaded image" style="max-width: 100%; display: block;">
            <canvas id="outline-canvas"></canvas>
            <canvas id="highlight-canvas"></canvas>
            <canvas id="hover-canvas"></canvas>
            <div id="tooltip"></div>
            <div id="custom-box"></div>
        </div>

        <script>
            const container = document.getElementById('container');
            const image = document.getElementById('base-image');
            const outlineCanvas = document.getElementById('outline-canvas');
            const highlightCanvas = document.getElementById('highlight-canvas');
            const hoverCanvas = document.getElementById('hover-canvas');
            const tooltip = document.getElementById('tooltip');
            const customBoxDiv = document.getElementById('custom-box');

            // Filter out non-interactive objects like walls, floor, ceiling
            const allObjects = {json.dumps(js_objects)};
            const objects = allObjects.filter(obj => {{
                const nonInteractive = ['wall', 'floor', 'ceiling', 'door', 'signboard'];
                return !nonInteractive.includes(obj.class.toLowerCase());
            }});
            
            if ({str(debug_mode).lower()}) console.log(`Filtered ${{allObjects.length}} objects down to ${{objects.length}} interactive objects`);
            
            let customBoxes = [];

            let scaleX, scaleY;
            let selectedObjectIndex = null;
            let hoveredObjectIndex = null;
            
            // --- Coordinate and Geometry Functions ---
            
            function updateCanvasSize() {{
                const rect = image.getBoundingClientRect();
                [outlineCanvas, highlightCanvas, hoverCanvas].forEach(c => {{
                    c.width = rect.width;
                    c.height = rect.height;
                    c.style.width = rect.width + 'px';
                    c.style.height = rect.height + 'px';
                }});
                
                // Scale factors for converting image coordinates to display coordinates
                scaleX = rect.width / image.naturalWidth;
                scaleY = rect.height / image.naturalHeight;
                
                drawAllOutlines();
                if (selectedObjectIndex !== null) {{
                    drawHighlight(selectedObjectIndex, true);
                }}
            }}


            
            // --- Main Application Logic ---
            
            function validateAndScaleObjects(objects) {{
                if (debugMode) console.log('🔧 Validating and scaling ' + objects.length + ' objects');
                
                return objects.map((obj, idx) => {{
                    const scaled = {{
                        ...obj,
                        contours: [],
                        bbox: null
                    }};
                    
                    // Validate and scale contours (coordinates already in original image space)
                    if (obj.contours && obj.contours.length > 0) {{
                        scaled.contours = obj.contours.map(contour => {{
                            if (!Array.isArray(contour)) return [];
                            return contour.map(point => {{
                                if (!Array.isArray(point) || point.length < 2) return [0, 0];
                                // Scale to display coordinates
                                return [
                                    Math.round(point[0] * scaleX),
                                    Math.round(point[1] * scaleY)
                                ];
                            }}).filter(p => p[0] >= 0 && p[1] >= 0);
                        }}).filter(c => c.length > 2);
                    }}
                    
                    // Scale bbox to display coordinates
                    if (obj.bbox && obj.bbox.length === 4) {{
                        scaled.bbox = [
                            Math.round(obj.bbox[0] * scaleX),
                            Math.round(obj.bbox[1] * scaleY),
                            Math.round(obj.bbox[2] * scaleX),
                            Math.round(obj.bbox[3] * scaleY)
                        ];
                    }}
                    
                    if (debugMode) console.log('Object ' + idx + ' (' + obj.class + '): bbox=' + JSON.stringify(scaled.bbox) + ', contours=' + scaled.contours.length);
                    return scaled;
                }});
            }}

            function getObjectAtPoint(x, y) {{
                if (debugMode) console.log('🎯 Checking point (' + x.toFixed(1) + ', ' + y.toFixed(1) + ') against ' + objects.length + ' objects');
                
                let candidates = [];
                
                // Check all objects and score them
                for (let i = 0; i < objects.length; i++) {{
                    const obj = objects[i];
                    let isInside = false;
                    let distance = Infinity;
                    let area = 0;
                    
                    // First check bounding box for quick elimination
                    if (obj.bbox) {{
                        const [x1, y1, x2, y2] = obj.bbox;
                        area = (x2 - x1) * (y2 - y1);
                        
                        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {{
                            // Inside bounding box - check precise contours if available
                            if (obj.contours && obj.contours.length > 0) {{
                                for (const contour of obj.contours) {{
                                    if (pointInPolygon(x, y, contour)) {{
                                        isInside = true;
                                        distance = 0;
                                        break;
                                    }}
                                }}
                                
                                // If not inside contour, calculate distance to nearest edge
                                if (!isInside) {{
                                    distance = getDistanceToContour(x, y, obj.contours);
                                }}
                            }} else {{
                                // No contours, bbox hit is enough
                                isInside = true;
                                distance = 0;
                            }}
                            
                            if (isInside || distance < 10) {{ // 10px tolerance
                                candidates.push({{
                                    index: i,
                                    distance: distance,
                                    area: area,
                                    isInside: isInside,
                                    obj: obj
                                }});
                                if (debugMode) console.log('   ✅ Hit object ' + i + ' (' + obj.class + '), distance=' + distance.toFixed(1) + ', area=' + area.toFixed(0));
                            }}
                        }}
                    }}
                }}
                
                // Also check custom boxes
                for (let i = customBoxes.length - 1; i >= 0; i--) {{
                    const box = customBoxes[i];
                    if (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) {{
                        candidates.push({{
                            index: 'custom_' + i,
                            distance: 0,
                            area: box.width * box.height,
                            isInside: true,
                            obj: box
                        }});
                    }}
                }}
                
                // Sort candidates: prefer inside > closest > smallest area
                if (candidates.length === 0) {{
                    if (debugMode) console.log('   ❌ No objects hit');
                    return null;
                }}
                
                // Sort by: inside first, then by distance, then by area (smallest first = on top)
                candidates.sort((a, b) => {{
                    if (a.isInside && !b.isInside) return -1;
                    if (!a.isInside && b.isInside) return 1;
                    if (Math.abs(a.distance - b.distance) > 0.1) return a.distance - b.distance;
                    return a.area - b.area; // Smaller objects on top
                }});
                
                const winner = candidates[0];
                if (debugMode) console.log('   🏆 Selected object ' + winner.index + ' (' + winner.obj.class + ') with distance=' + winner.distance.toFixed(1) + ', area=' + winner.area.toFixed(0));
                return winner.index;
            }}

            // Add distance calculation for better edge detection
            function getDistanceToContour(x, y, contours) {{
                let minDistance = Infinity;
                
                for (const contour of contours) {{
                    for (let i = 0; i < contour.length; i++) {{
                        const j = (i + 1) % contour.length;
                        const dist = distanceToLineSegment(
                            x, y,
                            contour[i][0], contour[i][1],
                            contour[j][0], contour[j][1]
                        );
                        minDistance = Math.min(minDistance, dist);
                    }}
                }}
                
                return minDistance;
            }}

            function distanceToLineSegment(px, py, x1, y1, x2, y2) {{
                const dx = x2 - x1;
                const dy = y2 - y1;
                const lenSq = dx * dx + dy * dy;
                
                if (lenSq === 0) {{
                    // Point-to-point distance if the line segment is actually a point
                    return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
                }}
                
                const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
                const nearestX = x1 + t * dx;
                const nearestY = y1 + t * dy;
                return Math.sqrt((px - nearestX) ** 2 + (py - nearestY) ** 2);
            }}

            // --- Drawing Functions ---

            function drawAllOutlines() {{
                const ctx = outlineCanvas.getContext('2d');
                ctx.clearRect(0, 0, outlineCanvas.width, outlineCanvas.height);
                
                // Only draw debug bounding boxes if debug mode is enabled
                if ({str(debug_mode).lower()}) {{
                    debugDrawBoundingBoxes();
                }}
                
                // Don't draw white outlines by default - only on hover
            }}
            
            // Add debug function
            function debugDrawBoundingBoxes() {{
                const ctx = outlineCanvas.getContext('2d');
                ctx.save();
                ctx.strokeStyle = 'yellow';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                
                objects.forEach((obj, index) => {{
                    if (obj.bbox) {{
                        const [x1, y1, x2, y2] = obj.bbox;
                        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                        ctx.fillStyle = 'yellow';
                        ctx.font = '12px Arial';
                        ctx.fillText(`${{obj.class}} (${{obj.source}})`, x1, y1 - 5);
                    }}
                }});
                
                ctx.restore();
            }}

            function drawHighlight(index, isSelection = false) {{
                const canvas = isSelection ? highlightCanvas : hoverCanvas;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                let obj;
                if (typeof index === 'string' && index.startsWith('custom_')) {{
                    const customIndex = parseInt(index.split('_')[1], 10);
                    const box = customBoxes[customIndex];
                    obj = {{ 
                        bbox: [box.x, box.y, box.x + box.width, box.y + box.height], 
                        contours: [],
                        class: 'Custom Selection' 
                    }};
                }} else {{
                    obj = objects[index];
                }}
                
                if (!obj) return;

                ctx.save();
                
                if (isSelection) {{
                    // Use a distinct color for selected objects
                    const color = getColorForIndex(index);
                    ctx.fillStyle = 'rgba(' + color[0] + ', ' + color[1] + ', ' + color[2] + ', 0.3)';
                    ctx.strokeStyle = 'rgb(' + color[0] + ', ' + color[1] + ', ' + color[2] + ')';
                    ctx.lineWidth = 3;
                }} else {{
                    // Hover style - bright red outline only
                    ctx.strokeStyle = '#FF0000';
                    ctx.lineWidth = 4;
                    ctx.shadowColor = '#FF0000';
                    ctx.shadowBlur = 15;
                    ctx.shadowOffsetX = 0;
                    ctx.shadowOffsetY = 0;
                }}

                // Draw contours if available
                if (obj.contours && obj.contours.length > 0) {{
                    obj.contours.forEach(contour => {{
                        if (contour.length > 2) {{
                            ctx.beginPath();
                            ctx.moveTo(contour[0][0], contour[0][1]);
                            for (let i = 1; i < contour.length; i++) {{
                                ctx.lineTo(contour[i][0], contour[i][1]);
                            }}
                            ctx.closePath();
                            
                            // Only fill for selections, not hover
                            if (isSelection) {{
                                ctx.fill();
                            }}
                            ctx.stroke();
                        }}
                    }});
                }} else if (obj.bbox) {{
                    // Use bbox as fallback
                    const [x1, y1, x2, y2] = obj.bbox;
                    const w = x2 - x1;
                    const h = y2 - y1;
                    
                    if (isSelection) {{
                        ctx.fillRect(x1, y1, w, h);
                    }}
                    ctx.strokeRect(x1, y1, w, h);
                }}
                
                ctx.restore();
            }}
            
            function updateTooltip(event, objectIndex) {{
                if (objectIndex !== null) {{
                    let objClass;
                    if (typeof objectIndex === 'string' && objectIndex.startsWith('custom_')) {{
                        objClass = 'Custom Selection';
                    }} else {{
                        objClass = objects[objectIndex].class || 'Unknown';
                    }}
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 10) + 'px';
                    tooltip.style.top = (event.clientY + 10) + 'px';
                    tooltip.textContent = objClass;
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }}

            // --- Missing Geometry Functions ---
            
            function pointInPolygon(x, y, polygon) {{
                let inside = false;
                for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {{
                    const xi = polygon[i][0], yi = polygon[i][1];
                    const xj = polygon[j][0], yj = polygon[j][1];
                    
                    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {{
                        inside = !inside;
                    }}
                }}
                return inside;
            }}
            
            function isNearPolygon(x, y, polygon, threshold) {{
                for (let i = 0; i < polygon.length; i++) {{
                    const j = (i + 1) % polygon.length;
                    const dist = distanceToLineSegment(x, y, polygon[i][0], polygon[i][1], polygon[j][0], polygon[j][1]);
                    if (dist <= threshold) return true;
                }}
                return false;
            }}
            
            function drawPolygon(ctx, polygon, fill = false) {{
                if (polygon.length < 2) return;
                
                ctx.beginPath();
                ctx.moveTo(polygon[0][0], polygon[0][1]); // Coordinates already scaled
                for (let i = 1; i < polygon.length; i++) {{
                    ctx.lineTo(polygon[i][0], polygon[i][1]);
                }}
                ctx.closePath();
                
                if (fill) ctx.fill();
                ctx.stroke();
            }}
            
            function calculatePolygonArea(polygon) {{
                let area = 0;
                for (let i = 0; i < polygon.length; i++) {{
                    const j = (i + 1) % polygon.length;
                    area += polygon[i][0] * polygon[j][1];
                    area -= polygon[j][0] * polygon[i][1];
                }}
                return Math.abs(area) / 2;
            }}
            
            function getColorForIndex(index) {{
                const colors = [
                    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                    [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 255, 0],
                    [255, 0, 128], [128, 0, 255], [0, 128, 255], [0, 255, 128]
                ];
                return colors[index % colors.length];
            }}

            // --- Event Handlers ---
            
            window.addEventListener('resize', function() {{
                updateCanvasSize();
                // Re-scale objects when window resizes
                objects = validateAndScaleObjects(objects);
                drawAllOutlines();
            }});
            
            // Enhanced mouse handling with proper coordinate mapping
            let lastMouseX = 0;
            let lastMouseY = 0;
            let debugMode = {str(debug_mode).lower()};
            
            function handleMouseMove(event) {{
                const rect = hoverCanvas.getBoundingClientRect();
                // Get mouse position relative to canvas in display coordinates
                lastMouseX = event.clientX - rect.left;
                lastMouseY = event.clientY - rect.top;
                
                // Debug logging
                if (debugMode) {{
                    console.log(`Mouse: display(${{lastMouseX.toFixed(0)}}, ${{lastMouseY.toFixed(0)}}), client(${{event.clientX}}, ${{event.clientY}})`);
                }}
                
                const hit = getObjectAtPoint(lastMouseX, lastMouseY);
                
                if (hit !== hoveredObjectIndex) {{
                    hoveredObjectIndex = hit;
                    
                    // Clear hover canvas
                    const ctx = hoverCanvas.getContext('2d');
                    ctx.clearRect(0, 0, hoverCanvas.width, hoverCanvas.height);
                    
                    // Draw new highlight if we have a hit
                    if (hit !== null) {{
                        drawHighlight(hit, false);
                    }}
                    
                    updateTooltip(event, hit);
                }}
            }}
            
            hoverCanvas.addEventListener('mousemove', handleMouseMove);
            
            hoverCanvas.addEventListener('mouseleave', function() {{
                hoveredObjectIndex = null;
                const ctx = hoverCanvas.getContext('2d');
                ctx.clearRect(0, 0, hoverCanvas.width, hoverCanvas.height);
                updateTooltip(null, null);
            }});
            
            hoverCanvas.addEventListener('click', function(event) {{
                const rect = hoverCanvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                
                const hit = getObjectAtPoint(x, y);
                
                if (hit !== null) {{
                    selectedObjectIndex = hit;
                    drawHighlight(selectedObjectIndex, true);
                    
                    // Send data back to Streamlit
                    let selectedObject;
                    if (typeof hit === 'string' && hit.startsWith('custom_')) {{
                        const customIndex = parseInt(hit.split('_')[1], 10);
                        selectedObject = customBoxes[customIndex];
                    }} else {{
                        selectedObject = objects[hit];
                    }}
                    
                    const selectedData = {{
                        selectedObjects: [selectedObject],
                        selectedIndices: [hit]
                    }};
                    
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: selectedData
                    }}, '*');
                }}
            }});
            
            // Enhanced initialization with proper object scaling
            function initializeCanvas() {{
                updateCanvasSize();
                
                // Scale all objects to display coordinates
                objects = validateAndScaleObjects(objects);
                
                // Draw initial outlines
                drawAllOutlines();
                
                if (debugMode) {{
                    console.log('🎨 Canvas initialized with ' + objects.length + ' objects');
                    console.log('📏 Canvas dimensions: ' + outlineCanvas.width + 'x' + outlineCanvas.height);
                    console.log('📏 Image dimensions: ' + image.naturalWidth + 'x' + image.naturalHeight);
                    console.log('📐 Scale factors: ' + scaleX.toFixed(3) + 'x, ' + scaleY.toFixed(3) + 'y');
                }}
            }}
            
            // Initialize
            if (image.complete) {{
                initializeCanvas();
            }} else {{
                image.onload = initializeCanvas;
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content

def auto_select_furniture_objects():
    """Auto-select furniture items like the original app"""
    furniture_items = ["chair", "sofa", "couch", "dining table", "tv", "bed", "table", "bookshelf", "cabinet", "desk"]
    st.session_state.selected_objects = [obj for obj in st.session_state.detected_objects 
                                        if obj.get("class", "").lower() in furniture_items]
    # Update indices as well
    st.session_state.selected_object_indices = [
        i for i, obj in enumerate(st.session_state.detected_objects)
        if obj.get("class", "").lower() in furniture_items
    ]

def crop_and_save_selected_objects():
    """Crops selected objects from the original image and saves them with vision features."""
    if st.session_state.image is None or not st.session_state.selected_objects:
        # Don't assign the result of st.warning to anything - this prevents DeltaGenerator issues
        st.warning("Please upload an image and select objects first.")
        st.session_state.processed_crop_details = []
        return

    cropped_object_details = []
    output_dir = os.path.join("output", "crops")
    os.makedirs(output_dir, exist_ok=True)

    original_image_cv = st.session_state.image # This is already a CV2 image (numpy array)
    # Get room style caption once for the entire image
    # Ensure original_image_cv is not None before proceeding
    room_style_data = {"caption": "an unknown style"} # Default if image is None
    if original_image_cv is not None:
        logger.info("Generating room style caption for the main image.")
        room_style_data = get_caption_json(original_image_cv, designer=False)
        logger.info(f"Room style caption generated: {room_style_data.get('caption')}")
    else:
        logger.warning("Original image is None, cannot generate room style caption.")

    for obj_index, obj_data in enumerate(st.session_state.selected_objects):
        bbox = obj_data.get("bbox")
        
        if bbox:
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_img_cv = original_image_cv[y_min:y_max, x_min:x_max]
            
            if cropped_img_cv.size == 0:
                logger.warning(f"Skipping empty crop for object {obj_data.get('id', obj_index)} at bbox {bbox}")
                continue

            # Sanitize class name for filename
            class_name = obj_data.get("class", "unknown_object")
            filename_safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
            
            timestamp = int(time.time() * 1000)
            crop_filename = f"{filename_safe_class_name}_{obj_data.get('id', obj_index)}_{timestamp}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            
            try:
                cv2.imwrite(crop_path, cropped_img_cv)
                logger.info(f"✅ Saved crop: {crop_path} for object ID {obj_data.get('id', obj_index)}")
                
                # Update the object in st.session_state.selected_objects directly
                st.session_state.selected_objects[obj_index]["crop_path"] = crop_path

                # Generate and store new designer caption JSON
                final_caption_json = {"style": "N/A", "material": "N/A", "colour": "N/A", "era": "N/A", "caption": "Caption generation failed"}
                try:
                    logger.info(f"Generating designer caption for crop: {crop_path}")
                    # Pass the cropped image (NumPy array) directly to get_caption_json
                    designer_caption_json = get_caption_json(cropped_img_cv, designer=True)
                    logger.info(f"Raw crop JSON for {crop_path}: {designer_caption_json}")
                    
                    # The 'blip_caption' field will now store the most important text part of the JSON for display
                    st.session_state.selected_objects[obj_index]["blip_caption"] = designer_caption_json.get('caption', 'No caption generated.')

                    # No longer merging with room style
                    # merged_json = merge_captions(designer_caption_json, st.session_state.get("room_style_caption", ""))
                    # st.session_state.selected_objects[obj_index]["merged_caption_json"] = merged_json

                    # Optional: Polish the final caption if needed (can be enabled later)
                    # if merged_json.get('caption'):

                    # CLIP embedding (existing logic)
                    try:
                        embedding = extract_clip_embedding(crop_path)
                        # Convert embedding to list for JSON serializability if needed by Streamlit's state
                        st.session_state.selected_objects[obj_index]["clip_embedding"] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                        logger.info(f"✅ Extracted CLIP embedding for {crop_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to extract CLIP embedding for {crop_path}: {e}")
                        st.session_state.selected_objects[obj_index]["clip_embedding"] = None
                    
                    cropped_object_details.append({
                        "id": obj_data.get("id", obj_index),
                        "class": class_name,
                        "confidence": obj_data.get("confidence", 0.8),
                        "crop_path": crop_path,
                        "blip_caption": st.session_state.selected_objects[obj_index].get("blip_caption"),
                        "designer_caption_json": st.session_state.selected_objects[obj_index].get("designer_caption_json"),
                        "clip_embedding": st.session_state.selected_objects[obj_index].get("clip_embedding") is not None
                    })

                except Exception as e:
                    logger.error(f"❌ Failed to generate designer caption for {crop_path}: {e}", exc_info=True)
                    st.session_state.selected_objects[obj_index]["designer_caption_json"] = final_caption_json # Store default error json
                    st.session_state.selected_objects[obj_index]["blip_caption"] = "Error: Caption generation failed"

            except Exception as e:
                logger.error(f"❌ Failed to save or process crop {crop_path}: {e}")
        else:
            logger.warning(f"Skipping object {obj_data.get('id', obj_index)} due to missing bounding box.")

    # Display UI elements but don't assign their return values
    if cropped_object_details:
        # These UI elements are for display only - don't assign their results
        st.success(f"✅ Successfully cropped {len(cropped_object_details)} selected objects and generated vision features.")
        st.subheader("🖼️ Cropped Objects and Features")
        cols = st.columns(3)
        for i, detail in enumerate(cropped_object_details):
            with cols[i % 3]:
                st.image(detail["crop_path"], caption=f"{detail['class']} (ID: {detail['id']})", use_container_width=True)
                st.markdown(f"**Caption:** `{detail.get('blip_caption', 'N/A')}`") # Changed from BLIP Caption to just Caption
                designer_info = detail.get("designer_caption_json")
                if designer_info and isinstance(designer_info, dict):
                    st.markdown(f"**Style:** `{designer_info.get('style', 'N/A')}`")
                    st.markdown(f"**Material:** `{designer_info.get('material', 'N/A')}`")
                    st.markdown(f"**Colour:** `{designer_info.get('colour', 'N/A')}`")
                
                # Safe CLIP embedding check - handle boolean or None values
                clip_embedding = detail.get('clip_embedding')
                clip_status = "Generated" if clip_embedding else "Failed"
                st.markdown(f"**CLIP Embedding:** `{clip_status}`")
    else:
        # Don't assign the result of st.warning
        st.warning("No objects were selected or could be cropped.")

    logger.info(f"crop_and_save_selected_objects explicitly storing {len(cropped_object_details)} items in session_state.")
    # Store results in session state to avoid DeltaGenerator issues
    st.session_state.processed_crop_details = cropped_object_details
    # Return nothing to avoid any DeltaGenerator issues
    return

def display_products(products_by_object, selected_object_details):
    """Displays product search results in cards, organized by selected object."""
    # Implementation of display_products function
    pass



# --- Main Application Logic ---
def main_app():
    """Main function to run the Streamlit application."""
    
    init_session_state()

    st.sidebar.title("Settings")
    
    # Detection Settings
    st.sidebar.subheader("🔍 Detection Settings")
    
    # SAM Detection Controls
    sam_enabled = os.getenv("ENABLE_SAM_DETECTION", "on").lower() in ["on", "true", "1"]
    has_replicate_token = bool(REPLICATE_API_TOKEN)
    
    if has_replicate_token and sam_enabled:
        use_sam = st.sidebar.checkbox(
            "Enable SAM Detection", 
            value=True,
            help="Semantic Segment Anything provides detailed object detection but may take 3-5 minutes on first run (cold start)"
        )
        
        if use_sam:
            sam_timeout = st.sidebar.slider(
                "SAM Timeout (seconds)",
                min_value=180,
                max_value=600,
                value=300,
                step=30,
                help="How long to wait for SAM detection. First run may need 5+ minutes due to cold start."
            )
            
            # Store timeout in session state for use by detector
            st.session_state.sam_timeout = sam_timeout
            
            # Show SAM status info
            st.sidebar.info(
                "ℹ️ **SAM Cold Start Info:**\n"
                "- First run: 3-5 minutes\n"
                "- Subsequent runs: 2-3 minutes\n"
                "- Provides semantic labels like 'Armchair', 'Coffee Table'"
            )
    else:
        st.sidebar.warning("⚠️ SAM Detection disabled\n(check REPLICATE_API_TOKEN)")
    
    # Detection Status Indicator
    if st.sidebar.checkbox("Show Detection Status"):
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            yolo_status = "✅ Ready" if st.session_state.get('object_detector') else "❌ Not initialized"
            st.sidebar.markdown(f"**YOLO:** {yolo_status}")
        
        with col2:
            sam_status = "✅ Enabled" if (has_replicate_token and sam_enabled) else "⚠️ Disabled"
            st.sidebar.markdown(f"**SAM:** {sam_status}")
            
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload a room image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of your interior space"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Process image button with protection against multiple clicks and cancel option
        if 'detection_in_progress' not in st.session_state:
            st.session_state.detection_in_progress = False
        
        # Create button layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Disable button if detection is in progress
            button_disabled = st.session_state.detection_in_progress
            button_text = "⏳ Processing..." if button_disabled else "🔍 Detect Objects"
            
            detect_clicked = st.button(button_text, key="detect_btn", disabled=button_disabled, use_container_width=True)
        
        with col2:
            # Show cancel button only if detection is in progress
            if st.session_state.detection_in_progress:
                if st.button("❌ Cancel", key="cancel_btn", use_container_width=True, type="secondary"):
                    try:
                        # Try to cancel SAM detection if running
                        detector_instance = get_object_detector()
                        if detector_instance and hasattr(detector_instance, 'sam_detector'):
                            detector_instance.sam_detector._cancel_global_prediction()
                        st.session_state.detection_in_progress = False
                        st.success("Detection cancelled!")
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Cancel attempt: {e}")
                        st.session_state.detection_in_progress = False
                        st.rerun()
            else:
                st.empty()  # Maintain layout
        
        if detect_clicked:
            st.session_state.detection_in_progress = True
            
            with st.spinner("🤖 Processing image for object detection..."):
                try:
                    # Save uploaded file temporarily
                    temp_img_path = "temp_uploaded_image.jpg"
                    with open(temp_img_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    detector_instance = get_object_detector()
                    if detector_instance:
                        original_image_cv, detected_objects, segmented_image_path = detector_instance.detect_objects(temp_img_path)
                        
                        if detected_objects:
                            # Store results
                            st.session_state.image = original_image_cv
                            st.session_state.objects = detected_objects
                            st.session_state.segmented_image_path = segmented_image_path
                            st.session_state.temp_img_path = temp_img_path
                            st.session_state.processed = True
                            
                            # Force clear any cached components to ensure refresh
                            st.session_state.interactive_html = None
                            
                            logger.info(f"✅ Detection complete. Stored image with shape {original_image_cv.shape} in session.")
                            st.rerun() # Force a rerun to update the UI immediately
                            
                        else:
                            st.error("❌ Failed to process image - no objects detected")
                            
                except Exception as e:
                    logger.error(f"❌ Error during object detection: {e}", exc_info=True)
                    st.error(f"❌ Error processing image: {str(e)}")
                finally:
                    st.session_state.detection_in_progress = False
        
        # Display interactive selection UI if processed
        if st.session_state.get("processed") and st.session_state.get("objects"):
            display_interactive_ui(st)

def display_interactive_ui(st_instance):
    """Refactored UI display logic for clarity."""
    
    with st_instance.columns(2)[1]: # Display in the second column
        st_instance.subheader("🎯 Detected Objects")
        if st.session_state.segmented_image_path and os.path.exists(st.session_state.segmented_image_path):
            seg_image = Image.open(st.session_state.segmented_image_path)
            st_instance.image(seg_image, use_container_width=True)
        else:
            st_instance.warning("⚠️ Segmented image not available.")
    
    st_instance.subheader("🛋️ Interactive Object Selection")
    st_instance.markdown("**Hover** over objects to see details, **click** to select/deselect items.")
    
    # Add debug mode control
    debug_mode = st_instance.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show object bounding boxes and mouse coordinates")
    
    # Create and display the interactive component
    if 'interactive_html' not in st.session_state or st.session_state.interactive_html is None:
        st.session_state.interactive_html = create_interactive_html(
            st.session_state.image, st.session_state.objects, debug_mode=debug_mode
        )
    
    component_height = min(st.session_state.image.shape[0] * 1.2, 800)
    selected_data = components.html(
        st.session_state.interactive_html,
        height=component_height,
        scrolling=True
    )
    
    # Handle selection data
    if selected_data and isinstance(selected_data, dict):
        st.session_state.selected_objects = selected_data.get('selectedObjects', [])
        st.session_state.include_walls = selected_data.get('includeWalls', False)

    # ... The rest of the UI logic ...

if __name__ == "__main__":
    # Fix asyncio event loop issues on Windows
    try:
        import asyncio
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
            pass

if __name__ == "__main__":
    main_app() 