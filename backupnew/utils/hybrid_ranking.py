"""
Simple hybrid ranking system for product results.
Combines text ranking with visual similarity for better results.
"""

import logging
import requests
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from io import BytesIO

logger = logging.getLogger(__name__)

def download_image(url: str, timeout: int = 10) -> Optional[str]:
    """
    Download image from URL and save to temporary file.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
    
    Returns:
        Path to temporary image file or None if download fails
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return tmp_file.name
            
    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {e}")
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Cosine similarity score (0-1)
    """
    if a is None or b is None:
        return 0.0
    
    if a.shape != b.shape or a.ndim != 1:
        return 0.0
    
    # Normalize vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    return float(np.dot(a_norm, b_norm))

def extract_thumbnail_embedding(image_url: str) -> Optional[np.ndarray]:
    """
    Extract CLIP embedding from product thumbnail.
    
    Args:
        image_url: URL of product thumbnail
    
    Returns:
        CLIP embedding vector or None if extraction fails
    """
    try:
        # Download image
        temp_path = download_image(image_url)
        if not temp_path:
            return None
        
        # Extract embedding
        from vision_features import extract_clip_embedding
        embedding = extract_clip_embedding(temp_path)
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return embedding
        
    except Exception as e:
        logger.warning(f"Failed to extract embedding from {image_url}: {e}")
        return None

def hybrid_rank_results(serp_results: List[Dict[str, Any]], 
                       query_embedding: Optional[np.ndarray] = None,
                       text_weight: float = 0.4,
                       visual_weight: float = 0.6) -> List[Dict[str, Any]]:
    """
    Rank SerpAPI results using hybrid text + visual similarity.
    
    Args:
        serp_results: List of SerpAPI product results
        query_embedding: CLIP embedding of the query image
        text_weight: Weight for text ranking (0-1)
        visual_weight: Weight for visual similarity (0-1)
    
    Returns:
        Ranked list of results with hybrid scores
    """
    if not serp_results:
        return serp_results
    
    # If no query embedding, return original ranking
    if query_embedding is None:
        logger.info("No query embedding available, using original ranking")
        for i, result in enumerate(serp_results):
            result['hybrid_score'] = 1.0 / (i + 1)  # Original rank score
        return serp_results
    
    logger.info(f"Computing hybrid scores for {len(serp_results)} results")
    
    # Calculate visual similarities
    visual_scores = []
    for i, result in enumerate(serp_results):
        # Try to get thumbnail URL
        thumbnail_url = result.get('thumbnail') or result.get('image')
        
        if thumbnail_url:
            # Extract embedding from thumbnail
            thumb_embedding = extract_thumbnail_embedding(thumbnail_url)
            if thumb_embedding is not None:
                visual_sim = cosine_similarity(query_embedding, thumb_embedding)
                visual_scores.append(visual_sim)
                logger.debug(f"Result {i}: visual similarity = {visual_sim:.3f}")
            else:
                visual_scores.append(0.0)
                logger.debug(f"Result {i}: no thumbnail embedding")
        else:
            visual_scores.append(0.0)
            logger.debug(f"Result {i}: no thumbnail URL")
    
    # Calculate hybrid scores
    for i, result in enumerate(serp_results):
        # Text rank score (inverse of position)
        text_score = 1.0 / (i + 1)
        
        # Visual similarity score
        visual_score = visual_scores[i] if i < len(visual_scores) else 0.0
        
        # Combined score
        hybrid_score = text_weight * text_score + visual_weight * visual_score
        
        result['hybrid_score'] = hybrid_score
        result['text_score'] = text_score
        result['visual_score'] = visual_score
        
        logger.debug(f"Result {i}: text={text_score:.3f}, visual={visual_score:.3f}, hybrid={hybrid_score:.3f}")
    
    # Sort by hybrid score
    ranked_results = sorted(serp_results, key=lambda x: x['hybrid_score'], reverse=True)
    
    logger.info(f"Hybrid ranking completed for {len(ranked_results)} results")
    return ranked_results

def simple_visual_filter(results: List[Dict[str, Any]], 
                        query_embedding: Optional[np.ndarray] = None,
                        min_similarity: float = 0.3) -> List[Dict[str, Any]]:
    """
    Simple filter to remove results with very low visual similarity.
    
    Args:
        results: List of product results
        query_embedding: CLIP embedding of query image
        min_similarity: Minimum visual similarity threshold
    
    Returns:
        Filtered results
    """
    if query_embedding is None or not results:
        return results
    
    filtered_results = []
    
    for result in results:
        thumbnail_url = result.get('thumbnail') or result.get('image')
        
        if not thumbnail_url:
            # Keep results without thumbnails
            filtered_results.append(result)
            continue
        
        # Check visual similarity
        thumb_embedding = extract_thumbnail_embedding(thumbnail_url)
        if thumb_embedding is not None:
            similarity = cosine_similarity(query_embedding, thumb_embedding)
            if similarity >= min_similarity:
                filtered_results.append(result)
            else:
                logger.debug(f"Filtered out result with low similarity ({similarity:.3f}): {result.get('title', 'Unknown')}")
        else:
            # Keep results where we can't extract embedding
            filtered_results.append(result)
    
    logger.info(f"Visual filtering: {len(results)} -> {len(filtered_results)} results")
    return filtered_results 