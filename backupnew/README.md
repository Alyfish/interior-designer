# 🏡 AI Interior Designer

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

### 1. Clone the Repository

```bash
git clone https://github.com/Alyfish/interior-designer.git
cd interior-designer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the project root directory:

```bash
# Copy the template
python setup_env.py
```

Then edit the `.env` file with your actual API keys:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here
SERP_API_KEY=your_actual_serpapi_key_here
IMGBB_API_KEY=your_actual_imgbb_api_key_here
REVERSE_IMAGE_SEARCH=on
REPLICATE_API_TOKEN=r8_...
ENABLE_SAM_DETECTION=on
```

#### Get Your API Keys:

- **OpenAI API**: https://platform.openai.com/api-keys
- **SerpAPI**: https://serpapi.com/manage-api-key  
- **ImgBB**: https://api.imgbb.com/

### 4. Download the YOLOv8 Model

The YOLOv8 model will be downloaded automatically on first run, or you can download it manually:

```bash
# The app will download yolov8x-seg.pt automatically to the project directory
```

### 5. Run the Application

```bash
streamlit run new_streamlit_app.py
```

The application will be available at: http://localhost:8501

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

## 🛠️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | For GPT-4V image analysis |
| `SERP_API_KEY` | Yes | For Google Shopping search |
| `IMGBB_API_KEY` | Optional | For reverse image search |
| `REVERSE_IMAGE_SEARCH` | Optional | Enable visual search (`on`/`off`) |
| `REPLICATE_API_TOKEN` | Optional | Enables advanced object detection with Semantic Segment Anything. |
| `ENABLE_SAM_DETECTION` | Optional | Toggles the SAM integration (`on`/`off`). |

### Search Configuration

Edit `config.py` to customize:

- API endpoints
- Search parameters  
- Model configurations
- Feature flags

## 📁 Project Structure

```
interior-designer/
├── 📄 new_streamlit_app.py      # Main Streamlit application
├── 🔍 new_object_detector.py    # YOLOv8 object detection
├── 🛍️ new_product_matcher.py    # Product search & matching
├── 👁️ vision_features.py        # CLIP & BLIP vision analysis
├── ⚙️ config.py                # Configuration management
├── 📋 requirements.txt         # Python dependencies
├── 🌐 setup_env.py             # Environment setup helper
├── 📁 utils/                   # Utility modules
│   ├── filters.py              # Product filtering
│   ├── cache.py               # Caching system
│   ├── hybrid_ranking.py      # Search ranking
│   ├── object_detection.py    # Detection utilities
│   ├── product_search.py      # Search utilities
│   └── image_analysis.py      # Image analysis utilities
├── 📁 input/                  # Sample input images
├── 📁 output/                 # Generated output files
└── 📁 tests/                  # Test files
```

## 🧪 Testing

### Run Tests

```bash
# Full integration test
python test_full_integration.py

# Individual component tests
python test_enhanced_features.py
python test_streamlit_integration.py
python test_imgbb_integration.py
```

### Test Your Setup

```bash
# Test API connections
python direct_serpapi_test.py

# Test product matching
python test_product_matching.py
```

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
# Try specifying the full path
streamlit run /full/path/to/new_streamlit_app.py
```

### Performance Tips

- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster inference
- **Memory**: Close other applications if experiencing memory issues
- **Network**: Stable internet connection required for API calls

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

### UI Customization

Modify the CSS in `new_streamlit_app.py`:

```python
st.markdown("""
<style>
    /* Add your custom CSS here */
    .main-header {
        background: your-custom-gradient;
    }
</style>
""", unsafe_allow_html=True)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## ✨ Semantic Segment Anything Integration

This project integrates with [Replicate's Semantic Segment Anything (SAM)](https://replicate.com/cjwbw/semantic-segment-anything) model to provide a more advanced and detailed object detection capability. When enabled, it works alongside the primary YOLOv8 detector to identify a wider range of objects that YOLO might miss.

### Benefits
- **Enhanced Accuracy**: Detects a greater variety of objects within the image.
- **Improved Detail**: Provides more granular segmentation for complex scenes.
- **Smarter Search**: Leads to more relevant product recommendations by identifying more items.

### Configuration
1.  **Get a Replicate API Token**:
    *   Sign up or log in at [Replicate](https://replicate.com/signin).
    *   Navigate to your **Account Settings** and copy your API token.
2.  **Enable in `.env`**:
    *   Add your token to the `.env` file: `REPLICATE_API_TOKEN=r8_...`
    *   Ensure the feature is turned on: `ENABLE_SAM_DETECTION=on`

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **OpenAI**: GPT-4V for image analysis  
- **SerpAPI**: Google Shopping integration
- **Streamlit**: Web application framework
- **Open-CLIP**: Visual similarity search

## 📞 Support

- 🐛 **Issues**: https://github.com/Alyfish/interior-designer/issues
- 💬 **Discussions**: https://github.com/Alyfish/interior-designer/discussions
- 📧 **Email**: [Your contact email]

---

Made with ❤️ for interior design enthusiasts 