import streamlit as st
import json
import base64
import io
from PIL import Image
import cv2
import numpy as np

def create_interactive_html(image, objects):
    """
    Interactive HTML component that provides visualization of detected objects.
    """
    import json, base64, io
    from PIL import Image
    
    # Convert image to base64 PNG
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    img_height, img_width = image.shape[:2]
    max_width = 1200
    max_height = 800
    scale = min(max_width / img_width, max_height / img_height, 1)
    display_width = int(img_width * scale)
    display_height = int(img_height * scale)
    
    html_content = f"""
    <style>
      .canvas-container {{
          position: relative;
          width: {display_width}px;
          height: {display_height}px;
          margin: auto;
          border: 1px solid #eee;
          overflow: hidden;
      }}
      .canvas-layer {{
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          cursor: pointer;
      }}
      .overlay-text {{
          position: absolute;
          background-color: rgba(0, 0, 0, 0.75);
          color: white;
          padding: 3px 6px;
          border-radius: 3px;
          font-size: 12px;
          pointer-events: none;
          z-index: 10;
          white-space: nowrap;
      }}
      .controls {{
          margin-bottom: 10px;
          text-align: center;
      }}
      .control-btn {{
          padding: 8px 16px;
          margin: 0 4px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-weight: bold;
          transition: background-color 0.2s;
      }}
      .control-btn:hover {{
          background-color: #ddd;
      }}
      .selection-summary {{
          position: fixed;
          bottom: 10px;
          left: 10px;
          background-color: rgba(255,255,255,0.95);
          border: 1px solid #ccc;
          padding: 8px;
          border-radius: 4px;
          font-size: 14px;
          max-height: 150px;
          overflow-y: auto;
          z-index: 20;
      }}
      canvas {{
          transition: all 0.2s ease-in-out;
      }}
    </style>
    
    <div class="controls">
        <button id="brushModeBtn" class="control-btn">Draw to Detect</button>
        <button id="selectAllBtn" class="control-btn">Select All</button>
        <button id="resetBtn" class="control-btn" style="background-color:#f44336; color:white;">Reset</button>
    </div>
    
    <div class="canvas-container">
        <canvas id="imageCanvas" class="canvas-layer" width="{display_width}" height="{display_height}"></canvas>
        <canvas id="overlayCanvas" class="canvas-layer" width="{display_width}" height="{display_height}"></canvas>
        <div id="tooltip" class="overlay-text" style="display: none;"></div>
    </div>
    
    <div id="selectionSummary" class="selection-summary" style="display:none;"></div>
    
    <script>
        // Debug logging function
        function debugLog(message) {{
            console.log("[DEBUG] " + message);
        }}

        // Initialization of objects, selected IDs, and mode flags.
        var objects = {json.dumps(objects)};
        debugLog("Initialized with " + objects.length + " objects");
        
        var selectedObjectIds = new Set();
        var isBrushMode = false;
        var brushPoints = [];
        var isDrawing = false;
        var stackedObjects = [];
        var currentStackIndex = 0;
        var pendingMask = null;
        var hoveredObject = null;
        
        // Dimensions and scale.
        var scale = {scale};
        var canvasWidth = {display_width};
        var canvasHeight = {display_height};
        var imgWidth = {img_width};
        var imgHeight = {img_height};
        
        // Canvas elements and contexts.
        var imageCanvas = document.getElementById('imageCanvas');
        var overlayCanvas = document.getElementById('overlayCanvas');
        var imageCtx = imageCanvas.getContext('2d');
        var overlayCtx = overlayCanvas.getContext('2d');
        var tooltip = document.getElementById('tooltip');
        var selectionSummary = document.getElementById('selectionSummary');
        var brushModeBtn = document.getElementById('brushModeBtn');
        var resetBtn = document.getElementById('resetBtn');
        
        // Load and draw base image.
        var img = new Image();
        img.onload = function() {{
            imageCtx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
            drawSegmentation();
            debugLog("Base image loaded and drawn");
            
            // Add event listeners after image is loaded
            setupEventListeners();
        }};
        img.src = 'data:image/png;base64,{img_str}';
        
        function setupEventListeners() {{
            debugLog("Setting up event listeners");
            
            // Brush Mode toggle
            brushModeBtn.addEventListener("click", function() {{
                isBrushMode = !isBrushMode;
                debugLog("Brush mode toggled: " + (isBrushMode ? "ON" : "OFF"));
                brushModeBtn.textContent = isBrushMode ? "Cancel" : "Draw to Detect";
                if (!isBrushMode) {{
                    brushPoints = [];
                    drawSegmentation();
                }}
            }});
            
            // Reset button
            resetBtn.addEventListener("click", function() {{
                debugLog("Reset button clicked");
                selectedObjectIds.clear();
                brushPoints = [];
                isBrushMode = false;
                brushModeBtn.textContent = "Draw to Detect";
                updateSelectionSummary();
                drawSegmentation();
            }});
            
            // Select All button
            var selectAllBtn = document.getElementById('selectAllBtn');
            selectAllBtn.addEventListener("click", function() {{
                debugLog("Select All button clicked");
                objects.forEach(function(obj) {{
                    selectedObjectIds.add(obj.id);
                }});
                updateSelectionSummary();
                drawSegmentation();
            }});
            
            // Mouse down event for brush mode
            overlayCanvas.addEventListener("mousedown", function(event) {{
                if (isBrushMode) {{
                    isDrawing = true;
                    brushPoints = [];
                    var rect = overlayCanvas.getBoundingClientRect();
                    var x = event.clientX - rect.left;
                    var y = event.clientY - rect.top;
                    brushPoints.push([Math.floor(x / scale), Math.floor(y / scale)]);
                    drawSegmentation();
                }}
            }});
            
            // Mouse move event for brush mode
            overlayCanvas.addEventListener("mousemove", function(event) {{
                if (isBrushMode && isDrawing) {{
                    var rect = overlayCanvas.getBoundingClientRect();
                    var x = event.clientX - rect.left;
                    var y = event.clientY - rect.top;
                    brushPoints.push([Math.floor(x / scale), Math.floor(y / scale)]);
                    drawSegmentation();
                }}
            }});
            
            // Mouse up event for brush mode
            overlayCanvas.addEventListener("mouseup", function(event) {{
                if (isBrushMode && isDrawing) {{
                    isDrawing = false;
                    if (brushPoints.length >= 3) {{
                        processBrushStroke(brushPoints);
                    }} else {{
                        brushPoints = [];
                        drawSegmentation();
                    }}
                }}
            }});
            
            // Click event for selection
            overlayCanvas.addEventListener("click", function(event) {{
                if (!isBrushMode) {{
                    var rect = overlayCanvas.getBoundingClientRect();
                    var x = event.clientX - rect.left;
                    var y = event.clientY - rect.top;
                    var found = getObjectsAtPoint(x, y);
                    
                    if (found.length > 0) {{
                        var obj = found[0];
                        debugLog("Toggling selection for: " + obj.class);
                        if (selectedObjectIds.has(obj.id)) {{
                            selectedObjectIds.delete(obj.id);
                            debugLog("Deselected: " + obj.class);
                        }} else {{
                            selectedObjectIds.add(obj.id);
                            debugLog("Selected: " + obj.class);
                        }}
                        updateSelectionSummary();
                        drawSegmentation();
                    }} else {{
                        debugLog("No object found at click point, starting manual segmentation");
                        manualSegmentation(x, y);
                    }}
                }}
            }});
            
            // Right-click event
            overlayCanvas.addEventListener("contextmenu", function(event) {{
                debugLog("Right-click event detected");
                event.preventDefault();
                var rect = overlayCanvas.getBoundingClientRect();
                var x = event.clientX - rect.left;
                var y = event.clientY - rect.top;
                stackedObjects = getObjectsAtPoint(x, y);
                
                if (stackedObjects.length > 1) {{
                    currentStackIndex = (currentStackIndex + 1) % stackedObjects.length;
                    hoveredObject = stackedObjects[currentStackIndex];
                    debugLog("Cycling to object: " + hoveredObject.class);
                    updateTooltip(x, y, "Cycling: " + hoveredObject.class + " (" + (hoveredObject.confidence ? hoveredObject.confidence.toFixed(2) : "n/a") + ")");
                    drawSegmentation();
                }}
            }});
            
            // Mouse move event
            overlayCanvas.addEventListener("mousemove", function(event) {{
                var rect = overlayCanvas.getBoundingClientRect();
                var x = event.clientX - rect.left;
                var y = event.clientY - rect.top;
                var found = getObjectsAtPoint(x, y);
                
                if (found.length > 0) {{
                    var newHoveredObject = found[0];
                    if (hoveredObject !== newHoveredObject) {{
                        hoveredObject = newHoveredObject;
                        debugLog("Hovering over: " + hoveredObject.class);
                        updateTooltip(x, y, hoveredObject.class + " (" + (hoveredObject.confidence ? hoveredObject.confidence.toFixed(2) : "n/a") + ")");
                        drawSegmentation();
                    }}
                }} else if (hoveredObject !== null) {{
                    hoveredObject = null;
                    hideTooltip();
                    drawSegmentation();
                }}
            }});
            
            // Mouse leave event
            overlayCanvas.addEventListener("mouseleave", function() {{
                if (hoveredObject !== null) {{
                    hoveredObject = null;
                    hideTooltip();
                    drawSegmentation();
                }}
            }});
        }}
        
        // Draw segmentation overlays with hover and selection outlines.
        function drawSegmentation() {{
            debugLog("Drawing segmentation overlays");
            overlayCtx.clearRect(0, 0, canvasWidth, canvasHeight);
            overlayCtx.save();
            overlayCtx.scale(scale, scale);
            
            // Draw existing objects
            objects.forEach(function(obj) {{
                var isSelected = selectedObjectIds.has(obj.id);
                debugLog("Drawing object: " + obj.class + " (selected: " + isSelected + ")");
                
                // Only draw outlines for selected objects or when hovered
                if (isSelected || (hoveredObject && hoveredObject.id === obj.id)) {{
                    // If this object is hovered (but not selected), draw dashed outline
                    if (hoveredObject && hoveredObject.id === obj.id && !isSelected) {{
                        debugLog("Drawing hover outline for: " + obj.class);
                        overlayCtx.setLineDash([4, 2]);
                        overlayCtx.strokeStyle = "#FF0000";
                        overlayCtx.lineWidth = 3 / scale;
                        overlayCtx.globalAlpha = 1.0;
                    }} else if (isSelected) {{
                        debugLog("Drawing selection outline for: " + obj.class);
                        overlayCtx.setLineDash([]);
                        overlayCtx.strokeStyle = "#00FF00";
                        overlayCtx.lineWidth = 4 / scale;
                        overlayCtx.globalAlpha = 1.0;
                    }}
                    
                    obj.contours.forEach(function(contour) {{
                        overlayCtx.beginPath();
                        overlayCtx.moveTo(contour[0][0], contour[0][1]);
                        contour.slice(1).forEach(function(point) {{
                            overlayCtx.lineTo(point[0], point[1]);
                        }});
                        overlayCtx.closePath();
                        overlayCtx.stroke();
                    }});
                }}
            }});
            
            // Draw brush stroke if in brush mode
            if (isBrushMode && brushPoints.length > 1) {{
                debugLog("Drawing brush stroke with " + brushPoints.length + " points");
                overlayCtx.setLineDash([]);
                overlayCtx.strokeStyle = "#00BFFF";
                overlayCtx.lineWidth = 3 / scale;
                overlayCtx.beginPath();
                overlayCtx.moveTo(brushPoints[0][0], brushPoints[0][1]);
                for (var i = 1; i < brushPoints.length; i++) {{
                    overlayCtx.lineTo(brushPoints[i][0], brushPoints[i][1]);
                }}
                overlayCtx.stroke();
            }}
            
            overlayCtx.restore();
        }}
        
        // Tooltip functions.
        function updateTooltip(x, y, text) {{
            debugLog("Updating tooltip: " + text);
            tooltip.style.display = "block";
            tooltip.style.left = (x + 10) + "px";
            tooltip.style.top = (y + 10) + "px";
            tooltip.textContent = text;
        }}
        function hideTooltip() {{
            debugLog("Hiding tooltip");
            tooltip.style.display = "none";
        }}
        
        // Update selection summary panel.
        function updateSelectionSummary() {{
            debugLog("Updating selection summary");
            var summaryHTML = "<b>Selected Objects (" + selectedObjectIds.size + "):</b><br>";
            objects.forEach(function(obj) {{
                if (selectedObjectIds.has(obj.id)) {{
                    summaryHTML += obj.class + " (" + (obj.confidence ? obj.confidence.toFixed(2) : "n/a") + ")<br>";
                }}
            }});
            selectionSummary.innerHTML = summaryHTML;
            selectionSummary.style.display = selectedObjectIds.size > 0 ? "block" : "none";
            
            // Send selection information to Streamlit
            if (window.parent && window.parent.postMessage) {{
                const selectedObjects = objects.filter(obj => selectedObjectIds.has(obj.id));
                window.parent.postMessage({{
                    type: "streamlit:setComponentValue",
                    value: selectedObjects
                }}, "*");
            }}
        }}
        
        // Check if a point is in a polygon (ray-casting algorithm).
        function isPointInPolygon(point, vs) {{
            var x = point[0], y = point[1];
            var inside = false;
            for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {{
                var xi = vs[i][0], yi = vs[i][1];
                var xj = vs[j][0], yj = vs[j][1];
                var intersect = ((yi > y) != (yj > y)) && 
                                (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
                if (intersect) inside = !inside;
            }}
            return inside;
        }}
        
        // Get objects at a given canvas point.
        function getObjectsAtPoint(x, y) {{
            var found = [];
            var imgX = x / scale;
            var imgY = y / scale;
            debugLog("Checking point: " + imgX + ", " + imgY);
            
            objects.forEach(function(obj) {{
                obj.contours.forEach(function(contour) {{
                    if (isPointInPolygon([imgX, imgY], contour)) {{
                        debugLog("Found object at point: " + obj.class);
                        found.push(obj);
                    }}
                }});
            }});
            return found;
        }}
        
        // Manual segmentation fallback if no object is detected.
        function manualSegmentation(canvasX, canvasY) {{
            debugLog("Starting manual segmentation at: " + canvasX + ", " + canvasY);
            var imgX = Math.floor(canvasX / scale);
            var imgY = Math.floor(canvasY / scale);
            var defaultWidth = Math.floor(imgWidth * 0.1);
            var defaultHeight = Math.floor(imgHeight * 0.1);
            var x1 = Math.max(0, imgX - Math.floor(defaultWidth / 2));
            var y1 = Math.max(0, imgY - Math.floor(defaultHeight / 2));
            var x2 = Math.min(imgWidth, imgX + Math.floor(defaultWidth / 2));
            var y2 = Math.min(imgHeight, imgY + Math.floor(defaultHeight / 2));
            var contour = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
            
            debugLog("Created manual selection box: " + JSON.stringify(contour));
            
            // Create a dummy mask (2D array) for now.
            var mask = [];
            for (var i = 0; i < imgHeight; i++) {{
                var row = [];
                for (var j = 0; j < imgWidth; j++) {{
                    row.push((j >= x1 && j <= x2 && i >= y1 && i <= y2) ? 1 : 0);
                }}
                mask.push(row);
            }}
            pendingMask = {{
                success: true,
                score: 0.8,
                contours: [contour],
                mask: mask
            }};
            // Prompt the user to label the new region.
            showLabelPanel();
        }}
        
        // Label panel functions: prompt user to assign a label to a manual selection.
        function showLabelPanel() {{
            debugLog("Showing label panel");
            // For simplicity, use a prompt
            var label = prompt("Enter label for selected region:", "custom");
            if (label && pendingMask) {{
                debugLog("Processing new label: " + label);
                processPendingMask(label);
            }}
        }}
        
        function processPendingMask(label) {{
            debugLog("Processing pending mask with label: " + label);
            var color = "#FF1493"; // Use your custom color map logic here if needed.
            var newObject = {{
                id: "manual_" + label + "_" + Date.now(),
                class: label,
                confidence: pendingMask.score || 0.8,
                contours: pendingMask.contours,
                color: color,
                mask: pendingMask.mask,
                source: "manual",
                timestamp: Date.now() / 1000
            }};
            objects.push(newObject);
            selectedObjectIds.add(newObject.id);
            debugLog("Added new object: " + label);
            updateSelectionSummary();
            drawSegmentation();
        }}
        
        // Process brush stroke and create new object
        function processBrushStroke(points) {{
            if (points.length < 3) return;
            
            // Create a temporary canvas to generate a mask
            var tempCanvas = document.createElement('canvas');
            tempCanvas.width = imgWidth;
            tempCanvas.height = imgHeight;
            var tempCtx = tempCanvas.getContext('2d');
            
            // Draw the brush stroke
            tempCtx.beginPath();
            tempCtx.moveTo(points[0][0], points[0][1]);
            for (var i = 1; i < points.length; i++) {{
                tempCtx.lineTo(points[i][0], points[i][1]);
            }}
            tempCtx.closePath();
            tempCtx.fill();
            
            // Get the mask data
            var mask = [];
            var imageData = tempCtx.getImageData(0, 0, imgWidth, imgHeight);
            for (var i = 0; i < imageData.data.length; i += 4) {{
                mask.push(imageData.data[i] > 0 ? 1 : 0);
            }}
            
            var label = prompt("Label this object:", "custom");
            if (label) {{
                var newObject = {{
                    id: "brush_" + label + "_" + Date.now(),
                    class: label,
                    confidence: 0.8,
                    contours: [points],
                    color: "#FF1493",
                    mask: mask,
                    source: "brush",
                    timestamp: Date.now() / 1000
                }};
                objects.push(newObject);
                selectedObjectIds.add(newObject.id);
                debugLog("Added new brush object: " + label);
                updateSelectionSummary();
            }}
            
            // Reset brush mode
            brushPoints = [];
            isBrushMode = false;
            brushModeBtn.textContent = "Draw to Detect";
            drawSegmentation();
        }}
    </script>
    """
    return html_content 