# Development Guidelines - DO NOT BREAK EXISTING CODE

## 🚨 CRITICAL: Preservation of Existing Functionality

This document outlines the features that **MUST NOT BE MODIFIED** when adding new functionality to the AI Interior Designer application. All new features must be **ADDITIVE ONLY** and should not alter the existing user experience.

## Protected Features (DO NOT MODIFY)

### 1. Object Detection Pipeline
**Files:** `new_object_detector.py`, `mask2former_detector.py`, `yolov8x-seg.pt`

**Protected Functionality:**
- YOLOv8 segmentation model initialization and inference
- Mask2Former integration (when available)
- Combined detection backend logic
- Object segmentation and mask generation
- Bounding box extraction
- Confidence score thresholds
- Object class filtering

**What NOT to change:**
```python
# DO NOT modify detection logic
def detect_objects(self, image_path):
    # This function's behavior must remain unchanged
    pass
```

### 2. Interactive UI Components
**File:** `new_streamlit_app.py` (lines 1108-1640)

**Protected Functionality:**
- Interactive HTML canvas generation
- Hover detection and highlighting
- Click selection/deselection mechanics
- Tooltip display on hover
- Object outline rendering
- Custom box drawing mode
- Selection state management
- Canvas scaling and coordinate transformation

**Critical Functions to Preserve:**
- `create_interactive_html()` - Core interactive UI generation
- `getObjectAtPoint()` - Hover detection logic
- `drawSelection()` - Object highlighting
- `handleMouseMove/Click/Up/Down()` - Event handlers

### 3. Product Search Pipeline
**Files:** `new_product_matcher.py`, `utils/enhanced_product_search.py`

**Protected Functionality:**
- SerpAPI integration for text search
- Reverse image search via Google Lens
- Basic product parsing
- Cache mechanisms
- Current search result format

**What NOT to change:**
```python
# These functions must maintain their current behavior
search_products_serpapi_tool()
search_products_reverse_image_serpapi()
parse_agent_response_to_products()
```

### 4. Vision Features
**File:** `vision_features.py`

**Protected Functionality:**
- CLIP embedding extraction
- BLIP caption generation
- GPT-4V integration
- Image preprocessing pipelines

## Safe Development Practices

### 1. Use Feature Flags
All new features MUST be behind feature flags:
```python
# In config.py
USE_NEW_FEATURE = os.getenv("USE_NEW_FEATURE", "false").lower() == "true"

# In your code
if USE_NEW_FEATURE:
    # New functionality
else:
    # Existing behavior unchanged
```

### 2. Create New Files
- Place new features in separate modules
- Use clear naming: `smart_search/`, `enhanced_features/`
- Import existing functions, don't modify them

### 3. Additive Parameters Only
When extending existing functions:
```python
# GOOD - Optional parameter with default
def existing_function(param1, param2, new_param=None):
    if new_param is not None:
        # New behavior
    # Original behavior preserved

# BAD - Changing required parameters
def existing_function(param1, new_param, param2):  # DON'T DO THIS
```

### 4. Testing Requirements
Before ANY deployment:
- [ ] Test with all feature flags OFF
- [ ] Verify hover functionality unchanged
- [ ] Verify selection mechanics unchanged  
- [ ] Verify object detection unchanged
- [ ] Verify existing search works
- [ ] Check existing UI components
- [ ] Run full user workflow test
- [ ] Only then test new features with flags ON

### 5. Error Handling
New features must fail gracefully:
```python
try:
    # New feature code
except Exception as e:
    logger.error(f"New feature failed: {e}")
    # Fall back to existing behavior
    return existing_function_result()
```

## Integration Points

### Safe Places to Add Features:

1. **After object selection** in `new_streamlit_app.py`:
   ```python
   # Line ~1534 - After "Find Matching Products" button
   if USE_SMART_SEARCH:
       # Add new UI elements
   ```

2. **In search pipeline** via `enhanced_product_search.py`:
   ```python
   # In create_enhanced_search_pipeline()
   if USE_SMART_SEARCH and feature_enabled:
       # Use new search logic
   else:
       # Use existing pipeline
   ```

3. **New menu items** in sidebar:
   ```python
   # In main_app() sidebar section
   if st.sidebar.checkbox("Advanced Options"):
       # New options that don't affect core features
   ```

## Version Control Guidelines

1. **Branch naming:** `feature/additive-<feature-name>`
2. **Commit messages:** Clearly indicate "ADDITIVE: <description>"
3. **PR requirements:**
   - Must include feature flag
   - Must pass existing tests
   - Must include new tests for new features
   - Must update this document if adding protected features

## Monitoring

After deployment:
1. Monitor error rates on existing features
2. Check performance metrics
3. Verify no regression in core functionality
4. Track feature flag usage

## Emergency Rollback

If existing features break:
1. Immediately set feature flag to FALSE
2. Revert deployment if necessary
3. Investigate without affecting users
4. Fix and re-test thoroughly

---

**Remember:** The existing application works well. Our goal is to enhance it without disrupting current users. When in doubt, create a new file or function rather than modifying existing code.