import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import time

def create_semantic_map():
    """Create a mapping of semantic classes to colors"""
    return {
        'wall': '#8B4513',      # Brown
        'floor': '#DEB887',     # BurlyWood
        'ceiling': '#F5F5F5',   # WhiteSmoke
        'window': '#87CEEB',    # SkyBlue
        'door': '#A0522D',      # Sienna
        'chair': '#DAA520',     # GoldenRod
        'couch': '#BC8F8F',     # RosyBrown
        'sofa': '#D2691E',      # Chocolate
        'bed': '#FA8072',       # Salmon
        'dining table': '#CD853F', # Peru
        'toilet': '#B0E0E6',    # PowderBlue
        'potted plant': '#228B22', # ForestGreen
        'tv': '#1E90FF',        # DodgerBlue
        'lamp': '#FFD700',      # Gold
        'book': '#F0FFF0',      # Honeydew
        'pillow': '#FFC0CB',    # Pink
        'rug': '#6B8E23',       # OliveDrab
        'carpet': '#8A2BE2',    # BlueViolet
        'artwork': '#BC8F8F',   # RosyBrown
        'custom': '#FF1493',    # DeepPink
    }

def process_with_yolo(image, model):
    """Process image through YOLO model"""
    results = model.predict(image, conf=0.25)[0]
    return results

def detect_walls_and_floors(image):
    """Detect walls and floors using edge detection and color analysis"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours in the dilated edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = image.shape[:2]
    wall_masks = []
    floor_masks = []
    
    # Process large contours for walls and floors
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.01 * height * width:  # Skip small contours
            continue
            
        # Create mask for this contour
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Analyze contour position and shape
        if y < height * 0.4 and h > height * 0.3:  # Upper part of image, tall
            wall_masks.append((mask, contour))
        elif y > height * 0.6:  # Lower part of image
            floor_masks.append((mask, contour))
        elif aspect_ratio > 3 or aspect_ratio < 0.3:  # Very wide or very tall
            wall_masks.append((mask, contour))
    
    return wall_masks, floor_masks

def create_yolo_objects(results, image):
    """Create object maps from YOLO results"""
    objects = []
    semantic_colors = create_semantic_map()
    
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()
        img_height, img_width = results.orig_shape
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            class_id = int(box[5])
            class_name = results.names[class_id]
            confidence = float(box[4])
            
            # Resize mask to original image size
            resized_mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0.5).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Simplify contours
            simplified_contours = []
            for contour in contours:
                if len(contour) >= 3:
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    points = approx.reshape(-1, 2).tolist()
                    if len(points) >= 3:
                        simplified_contours.append(points)
            
            if simplified_contours:
                object_id = f"yolo_{i}_{class_name}_{uuid.uuid4().hex[:8]}"
                color = semantic_colors.get(class_name, '#808080')
                
                objects.append({
                    'id': object_id,
                    'class': class_name,
                    'confidence': confidence,
                    'contours': simplified_contours,
                    'color': color,
                    'mask': binary_mask.tolist(),
                    'source': 'yolo',
                    'timestamp': time.time()
                })
    
    # Detect walls and floors if not detected by YOLO
    has_wall = any(obj['class'] == 'wall' for obj in objects)
    has_floor = any(obj['class'] == 'floor' for obj in objects)
    
    if not has_wall or not has_floor:
        wall_masks, floor_masks = detect_walls_and_floors(image)
        
        # Add detected walls
        if not has_wall and wall_masks:
            for i, (mask, contour) in enumerate(wall_masks):
                # Simplify contour
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2).tolist()
                if len(points) >= 3:
                    object_id = f"wall_{i}_{uuid.uuid4().hex[:8]}"
                    objects.append({
                        'id': object_id,
                        'class': 'wall',
                        'confidence': 0.85,  # Reasonable confidence value
                        'contours': [points],
                        'color': semantic_colors.get('wall'),
                        'mask': mask.tolist(),
                        'source': 'opencv',
                        'timestamp': time.time()
                    })
        
        # Add detected floors
        if not has_floor and floor_masks:
            for i, (mask, contour) in enumerate(floor_masks):
                # Simplify contour
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2).tolist()
                if len(points) >= 3:
                    object_id = f"floor_{i}_{uuid.uuid4().hex[:8]}"
                    objects.append({
                        'id': object_id,
                        'class': 'floor',
                        'confidence': 0.85,  # Reasonable confidence value
                        'contours': [points],
                        'color': semantic_colors.get('floor'),
                        'mask': mask.tolist(),
                        'source': 'opencv',
                        'timestamp': time.time()
                    })
    
    return objects

def process_with_sam(image, sam_model, click_x, click_y):
    """Process a user click with SAM model"""
    if sam_model is None:
        return None, 0.0
        
    try:
        from segment_anything import SamPredictor
        
        # Ensure image is RGB (SAM requires RGB input)
        if image.shape[2] == 4:  # With alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set up SAM predictor
        predictor = SamPredictor(sam_model)
        predictor.set_image(image)
        
        # Convert click coordinates to numpy array
        input_point = np.array([[click_x, click_y]])
        input_label = np.array([1])  # 1 indicates foreground
        
        # Generate mask
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,  # Return multiple masks
        )
        
        # Return the highest scoring mask
        if scores.size > 0:
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            score = float(scores[best_mask_idx])
            return best_mask.astype(np.uint8), score
    except Exception as e:
        print(f"Error processing with SAM: {e}")
    
    return None, 0.0

def create_sam_object(mask, score, class_name, color=None):
    """Create an object entry from a SAM-generated mask"""
    semantic_colors = create_semantic_map()
    
    # Use provided color or get from semantic map or use custom color
    if color is None:
        color = semantic_colors.get(class_name, semantic_colors.get('custom', '#FF1493'))
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Simplify contours
    simplified_contours = []
    for contour in contours:
        if len(contour) >= 3:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2).tolist()
            if len(points) >= 3:
                simplified_contours.append(points)
    
    if simplified_contours:
        # Generate a unique ID
        object_id = f"sam_{class_name}_{uuid.uuid4().hex[:8]}"
        
        return {
            'id': object_id,
            'class': class_name,
            'confidence': score,
            'contours': simplified_contours,
            'color': color,
            'mask': mask.tolist(),  # For GPT-4o
            'source': 'sam',        # Indicates detection source
            'timestamp': time.time()  # When it was detected
        }
    
    return None 