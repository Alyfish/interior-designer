# Product Matching Enhancement Sprint

This README documents the enhancements made to the product matching functionality of the Interior Designer tool.

## Overview of Improvements

1. **True Reverse-Image Search**: Implemented direct file upload to SerpAPI Google Lens API for better visual matching.
2. **Rich Caption Generation**: Added GPT-4o-based caption generation with style, color, and material extraction.
3. **Color Extraction**: Implemented k-means clustering to extract dominant colors from furniture images.
4. **Parallel Search Strategy**: Enabled searching with multiple captions concurrently for better recall.
5. **SSIM Product Validation**: Added structural similarity comparison between product images and original crops.
6. **Enhanced Ranking**: Improved product ordering based on confidence, color/material matches, and more.
7. **Feature Flag System**: Added controls to enable/disable features and graceful fallbacks.

## Feature Flags

All new features are controlled by feature flags for easy toggling. You can enable or disable features by setting environment variables:

```
SERPAPI_REVIMG=on        # SerpAPI reverse image search
GPT4O_CAPTIONS=on        # GPT-4o caption generation
COLOR_EXTRACTOR=on       # Color extraction using k-means
MATERIAL_DETECTOR=on     # Material detection using GPT-4o
SSIM_PRODUCT_VALIDATION=on  # Product validation using image similarity
NEW_PRODUCT_RANKER=on    # Enhanced product ranking
PARALLEL_CAPTION_SEARCH=on  # Search with multiple captions in parallel
```

Set these in your `.env` file or as environment variables. Values can be `on`, `true`, `1`, or `yes` to enable a feature.

## How to Roll Back

If issues arise with any new feature, you can disable it by setting the corresponding feature flag to `off`:

```
# Example: Turn off SSIM validation if it's causing issues
SSIM_PRODUCT_VALIDATION=off
```

The system will automatically fall back to the previous behavior when a feature is disabled.

## API Keys Required

Some features require additional API keys:

- `OPENAI_API_KEY`: Required for GPT-4o caption generation and material detection.
- `SERP_KEY`: Required for SerpAPI Google Shopping and reverse image search.
- `AMAZON_RAPID_KEY`: Required for Amazon product search.
- `BING_KEY`: Optional for Bing visual search.

## Testing the Improvements

1. Upload an interior image
2. Run object detection to identify furniture
3. Select furniture items and find matching products
4. Observe results with improved matching quality and ranking

## Technical Implementation

### Added Files
- `utils/image_color.py`: Color extraction using k-means clustering
- `README_sprint.md`: This documentation file

### Modified Files
- `product_finder.py`: Core enhancements to search and ranking
- `utils/feature_flags/__init__.py`: Added new feature flags
- `requirements.txt`: Added dependencies for new features

## Future Work

Additional potential improvements:
- Image embeddings for better similarity search
- E-commerce-specific object detection fine-tuning
- Learning-to-rank model for product ordering
- User click feedback for result quality improvement 