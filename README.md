# Interior Designer

An advanced interior design application that combines object detection with furniture shopping recommendations.

## Features

- **Object Detection**: Uses YOLOv8-segmentation to identify furniture and architectural elements
- **Interactive Interface**: Select objects, hover for information, and right-click to cycle through overlapping items
- **Custom Object Creation**: Use "Draw to Detect" tool to create custom objects
- **Furniture Analysis**: Uses AI to find real, available products matching your furniture
- **Product Links**: Get direct links to retailers like IKEA, Wayfair, Target, Amazon, and Walmart

## New Features & Optimizations

The application includes the following optimizations and enhancements that can be enabled via feature flags:

### Available Feature Flags

Set these in your `.env` file to enable optional features:

- `FINETUNE_MODEL=on` - Use a fine-tuned YOLOv8 segmentation model (requires the model file in `/models`)
- `VECTOR_SEARCH=on` - Enable vector database similarity search
- `LLM_RERANK=on` - Use LLM to rerank product search results
- `EMBEDDINGS=on` - Generate CLIP/DINO embeddings for furniture crops
- `EXT_CACHE=on` - Enable Redis and S3 caching for better performance
- `METRICS=on` - Enable Prometheus metrics collection
- `HEADLESS_VERIFY=on` - Verify product links using headless browser
- `PROGRESSIVE_CARDS=on` - Show progressive loading of results (enabled by default)

### Performance Optimizations

The application includes several performance optimizations:

1. **Tiered Caching System**
   - In-memory LRU cache for fast access
   - Optional Redis cache for persistence across restarts
   - Optional S3 storage for binary data

2. **Concurrency Control**
   - Throttling to limit API calls per service
   - Rate limiting to avoid overloading external services
   - Circuit breaker pattern for resilience
   - Exponential backoff and retry for transient errors

3. **Monitoring**
   - Optional Prometheus metrics for monitoring
   - Performance tracking for API calls and search operations
   - Latency measurements for system optimization

### UI Enhancements

1. **Progressive Card Loading**
   - Products appear as they're loaded, rather than waiting for all results
   - Better user experience for slower searches

2. **Source Filtering**
   - Filter results by source (Google, Amazon, Bing, etc.)
   - Focus on preferred retailers

## Configuration

The application uses JSON configuration files for various settings:

- `config/retailers.json` - Supported retailers with verification patterns and selectors
- `config/semantic_colors.json` - Color coding for different object types

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download YOLOv8 model:
   ```
   # The model will be downloaded automatically on first run
   # or you can download it manually from Ultralytics
   ```

## Usage

### Web Interface

```bash
python -m interior_designer.main --serve
```

This will start a Streamlit web server that you can access at http://localhost:8501

### Command Line

```bash
# Process an image with LaoZhang API
python -m interior_designer.main --image path/to/image.jpg --api laozhang

# Process an image with Perplexity API
python -m interior_designer.main --image path/to/image.jpg --api perplexity
```

## API Keys

- Create a `.env` file in the project root with the following variables:
  ```ini
  SERP_KEY=your_serpapi_key_here
  AMAZON_RAPID_KEY=your_amazon_rapidapi_key
  LAOZHANG_API_KEY=your_laozhang_api_key
  PERPLEXITY_API_KEY=your_perplexity_api_key
  # Optional:
  BING_KEY=your_bing_visual_search_key
  OPENAI_API_KEY=your_openai_api_key
  ```

- **IMPORTANT: Security Best Practices**
  - Never commit your actual API keys to Git
  - The `.env` file is included in `.gitignore` to prevent accidental commits
  - You can copy the structure from `env.example` in the project root
  - Share API keys with collaborators via secure channels (not Git)

- The app will raise an error if `SERP_KEY` is missing. Amazon search will be skipped if `AMAZON_RAPID_KEY` is not set.

## Smoke Test

A simple smoke test script verifies that the product finder can return valid product links:

```bash
# Using pytest
pytest test_product_finder.py

# Or run directly
python test_product_finder.py
```

Exit code `0` indicates success. A non-zero exit code means no products were found or an error occurred.

## How It Works

1. **Detection**: The application uses YOLOv8 with segmentation to detect furniture, walls, floors, etc.
2. **User Interaction**: You can select objects using the interactive interface
3. **Furniture Analysis**: Selected objects are analyzed by AI to find matching products
4. **Shopping Recommendations**: View matching furniture items with links to purchase

## Directory Structure

```
interior_designer/
├── analysis/              # Furniture analysis modules
│   ├── furniture_analyzer.py      # LaoZhang API analysis
│   └── perplexity_analyzer.py     # Perplexity API analysis
├── detection/             # Object detection modules
│   └── yolo_detector.py           # YOLOv8 detection
├── ui/                    # User interface
│   ├── app.py                     # Streamlit app
│   └── interactive_ui.py          # Interactive object selection
├── utils/                 # Utility functions
├── output/                # Output directory for results
├── connector.py           # Main connector between detection & analysis
└── main.py                # Entry point
```

## Credits

This project uses:
- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- [Streamlit](https://streamlit.io/) for the web interface
- [LaoZhang AI API](https://laozhang.ai) & [Perplexity API](https://perplexity.ai) for furniture analysis 