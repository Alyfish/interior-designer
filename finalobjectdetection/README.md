# 🏡 AI Interior Designer - Final Object Detection Version

An advanced AI-powered interior design application that detects furniture in room images and finds matching products through a beautiful interactive interface.

## ✨ Features

- **🔍 Intelligent Object Detection**: Uses YOLOv8-Seg for precise furniture detection
- **🎨 Interactive Modal Carousel**: Click detected objects to view product recommendations
- **🔄 Hybrid Product Search**: Combines text and visual similarity search
- **💡 Advanced Vision Analysis**: CLIP embeddings and GPT-4V captions
- **🛍️ Real Product Results**: Live shopping results from Google Shopping
- **📱 Modern UI**: Responsive design with smooth animations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- Internet connection for API services

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Copy the template file and fill in your API keys:

```bash
cp .env.template .env
```

Then edit the `.env` file with your actual API keys:

```
OPENAI_API_KEY=your_actual_openai_api_key_here
SERP_API_KEY=your_actual_serpapi_key_here
IMGBB_API_KEY=your_actual_imgbb_api_key_here
REPLICATE_API_TOKEN=your_actual_replicate_api_token_here
REVERSE_IMAGE_SEARCH=on
```

#### Get Your API Keys:

- **OpenAI API**: https://platform.openai.com/api-keys
- **SerpAPI**: https://serpapi.com/manage-api-key
- **ImgBB**: https://api.imgbb.com/
- **Replicate**: https://replicate.com/account/api-tokens

### 3. Run the Application

```bash
python3 -m streamlit run new_streamlit_app.py
```

The application will be available at: http://localhost:8501

**Note**: The YOLOv8 model (yolov8x-seg.pt) will be downloaded automatically on first run (~137MB).

## 🎯 How to Use

### Basic Workflow

1. **Upload Image**: Use the sidebar to upload a room image
2. **Detect Objects**: Click "🔍 Detect Objects" to analyze the image
3. **Find Products**: Click "🛍️ Find Products" to search for matching items
4. **Browse Results**: Click on detected objects in the image to view product carousels

### Advanced Features

- **Search Methods**: Choose between Hybrid, Text-only, or Visual-only search
- **Interactive Navigation**: Use carousel arrows to browse products
- **Product Details**: View prices, ratings, and store information
- **Direct Shopping**: Click product links to visit retailer websites

## 📁 Project Structure

```
finalobjectdetection/
├── 📄 new_streamlit_app.py      # Main Streamlit application
├── 🔍 new_object_detector.py    # YOLOv8 object detection
├── 🛍️ new_product_matcher.py    # Product search & matching
├── 👁️ vision_features.py        # CLIP & BLIP vision analysis
├── ⚙️ config.py                # Configuration management
├── 📋 requirements.txt         # Python dependencies
├── 📁 utils/                   # Utility modules
│   ├── filters.py              # Product filtering
│   ├── cache.py               # Caching system
│   ├── hybrid_ranking.py      # Search ranking
│   ├── object_detection.py    # Detection utilities
│   ├── product_search.py      # Search utilities
│   └── image_analysis.py      # Image analysis utilities
└── 🎯 yolov8x-seg.pt          # YOLOv8 model (downloaded automatically)
```

## 🧪 Testing

The application has been tested and verified to work with:
- Streamlit 1.46.1
- Python 3.8+
- All required dependencies from requirements.txt

## 🔧 Troubleshooting

### Common Issues

**"ModuleNotFoundError" errors**:
```bash
pip install -r requirements.txt
```

**API key errors**:
- Verify your `.env` file exists and contains valid keys
- Check API key permissions and quotas

**YOLOv8 model not found**:
- The model downloads automatically on first run
- Ensure you have internet connection
- Check disk space (model is ~137MB)

**Streamlit won't start**:
```bash
python3 -m streamlit run new_streamlit_app.py
```

## 🎨 Customization

### Adding New Furniture Categories

Edit `new_object_detector.py` to add custom object classes:

```python
FURNITURE_CLASSES = [
    'chair', 'table', 'sofa', 'bed', 
    'your_custom_furniture_type'  # Add here
]
```

### Modifying Search Parameters

Update `new_product_matcher.py`:

```python
# Adjust search parameters
SEARCH_PARAMS = {
    'num_results': 10,        # Number of products to return
    'price_range': (0, 1000), # Price filtering
    'enable_filters': True    # Enable advanced filtering
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **OpenAI**: GPT-4V for image analysis
- **SerpAPI**: Google Shopping integration
- **Streamlit**: Web application framework
- **Open-CLIP**: Visual similarity search

---

Made with ❤️ for interior design enthusiasts

## Version Information

This is the **Final Object Detection Version** - a stable, tested, and production-ready implementation of the AI Interior Designer application. 