"""
Tiered caching system for product search results.

Implements a multi-level caching strategy:
1. Local memory caching using LRU (always on)
2. Redis caching for persistence across restarts (optional)
3. S3 storage for binary data like images (optional)
"""
import os
import json
import hashlib
import functools
from typing import Any, Callable, Optional
import time

from interior_designer.utils.feature_flags import is_enabled

# Constants
LOCAL_CACHE_SIZE = 100         # Number of items in LRU cache
LOCAL_CACHE_TTL = 60 * 10      # 10 minutes TTL for in-memory cache
REDIS_CACHE_TTL = 60 * 60 * 24 # 24 hours TTL for Redis cache

def generate_cache_key(image_path: str, obj_class: str = None, **kwargs) -> str:
    """
    Generate a cache key based on image content hash and other parameters
    
    Args:
        image_path: Path to the image file
        obj_class: Object class name
        kwargs: Additional parameters that affect the search
        
    Returns:
        str: Cache key for the query
    """
    # Hash the image content
    try:
        with open(image_path, 'rb') as f:
            img_hash = hashlib.sha1(f.read()).hexdigest()
    except Exception:
        # Fallback to path if file can't be read
        img_hash = hashlib.sha1(image_path.encode()).hexdigest()
    
    # Combine with other parameters
    params = {
        'class': obj_class or 'unknown',
        **kwargs
    }
    
    # Create a deterministic string representation
    param_str = json.dumps(params, sort_keys=True)
    
    # Combine image hash with parameters
    key = f"{img_hash}_{hashlib.sha1(param_str.encode()).hexdigest()}"
    return key

class TieredCache:
    """
    Manages caching across multiple layers: in-memory, Redis, and S3.
    """
    def __init__(self):
        self.memory_cache = {}
        self.memory_timestamps = {}
        
        # Initialize Redis connection if enabled
        self.redis_client = None
        if is_enabled('EXT_CACHE'):
            try:
                import redis
                redis_url = os.getenv('REDIS_URL')
                if redis_url:
                    self.redis_client = redis.from_url(redis_url)
                    print(f"Redis cache enabled: {redis_url}")
            except ImportError:
                print("Redis package not installed. External caching limited.")
        
        # Initialize S3 connection if enabled
        self.s3_client = None
        if is_enabled('EXT_CACHE'):
            try:
                import boto3
                aws_access_key = os.getenv('AWS_ACCESS_KEY')
                aws_secret_key = os.getenv('AWS_SECRET_KEY')
                s3_bucket = os.getenv('S3_BUCKET')
                
                if aws_access_key and aws_secret_key and s3_bucket:
                    self.s3_client = boto3.client(
                        's3',
                        aws_access_key_id=aws_access_key,
                        aws_secret_access_key=aws_secret_key
                    )
                    self.s3_bucket = s3_bucket
                    print(f"S3 storage enabled: {s3_bucket}")
            except ImportError:
                print("Boto3 package not installed. S3 storage disabled.")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache (any tier)
        """
        # Try memory cache first (fastest)
        if key in self.memory_cache:
            timestamp = self.memory_timestamps.get(key, 0)
            if time.time() - timestamp <= LOCAL_CACHE_TTL:
                return self.memory_cache[key]
            else:
                # Expired, remove from memory
                del self.memory_cache[key]
                del self.memory_timestamps[key]
        
        # Try Redis next if enabled
        if self.redis_client:
            redis_value = self.redis_client.get(key)
            if redis_value:
                # Found in Redis, deserialize and update memory cache
                try:
                    value = json.loads(redis_value)
                    self.memory_cache[key] = value
                    self.memory_timestamps[key] = time.time()
                    return value
                except json.JSONDecodeError:
                    # Invalid JSON, ignore
                    pass
        
        # Not found in any cache
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store item in all cache tiers
        """
        # Update memory cache
        self.memory_cache[key] = value
        self.memory_timestamps[key] = time.time()
        
        # Update Redis cache if enabled
        if self.redis_client:
            try:
                serialized = json.dumps(value)
                self.redis_client.setex(key, REDIS_CACHE_TTL, serialized)
            except Exception as e:
                print(f"Error caching to Redis: {e}")
        
        # Manage memory cache size (LRU)
        if len(self.memory_cache) > LOCAL_CACHE_SIZE:
            # Remove oldest entry
            oldest_key = min(self.memory_timestamps, key=self.memory_timestamps.get)
            del self.memory_cache[oldest_key]
            del self.memory_timestamps[oldest_key]
    
    def store_binary(self, key: str, data: bytes) -> Optional[str]:
        """
        Store binary data in S3 if enabled
        
        Returns:
            str: URL to stored data or None if storage failed
        """
        if not self.s3_client:
            return None
        
        try:
            # Store in S3
            object_key = f"crops/{key}.jpg"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=object_key,
                Body=data,
                ContentType='image/jpeg'
            )
            
            # Return the URL
            return f"https://{self.s3_bucket}.s3.amazonaws.com/{object_key}"
        except Exception as e:
            print(f"Error storing binary data in S3: {e}")
            return None
    
    def clear(self) -> None:
        """Clear local memory cache"""
        self.memory_cache.clear()
        self.memory_timestamps.clear()

# Singleton instance
_cache = TieredCache()

def cached(func):
    """
    Decorator for caching function results using the tiered cache system
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Generate key based on function name, args and kwargs
        if len(args) > 0 and isinstance(args[0], str) and os.path.isfile(args[0]):
            # First arg is a file path, use it for content-based hashing
            image_path = args[0]
            remaining_args = args[1:]
            
            # Generate key based on image content and other args
            key = f"{func.__name__}_{generate_cache_key(image_path, **kwargs)}"
            
            # Check cache
            cached_result = _cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Call the function if not cached
            result = await func(image_path, *remaining_args, **kwargs)
            
            # Store result in cache
            _cache.set(key, result)
            return result
        else:
            # Regular function without image path, use standard caching
            return await func(*args, **kwargs)
    
    return wrapper 