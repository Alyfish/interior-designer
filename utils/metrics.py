"""
Metrics collection and reporting for the Interior Designer application.
Uses Prometheus for metrics collection if available, with fallback to logging.
"""
import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Callable
import os

from interior_designer.utils.feature_flags import is_enabled

# Set up logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('interior_designer.metrics')

# Global state
_metrics_enabled = False
_prometheus_client = None
_counters = {}
_histograms = {}

# Initialize metrics system
def init_metrics():
    """Initialize the metrics collection system."""
    global _metrics_enabled, _prometheus_client
    
    if is_enabled('METRICS'):
        try:
            import prometheus_client
            _prometheus_client = prometheus_client
            
            # Start Prometheus HTTP server on port 8000 (default)
            metrics_port = int(os.getenv('METRICS_PORT', 8000))
            prometheus_client.start_http_server(metrics_port)
            logger.info(f"Prometheus metrics server started on port {metrics_port}")
            
            _metrics_enabled = True
        except ImportError:
            logger.warning("Prometheus client not installed, metrics will be logged only")
            _metrics_enabled = False
    else:
        logger.info("Metrics collection disabled (METRICS=off)")
        _metrics_enabled = False

def counter(name: str, description: str, labels: Optional[List[str]] = None) -> Callable:
    """
    Get a counter metric or create it if it doesn't exist
    
    Args:
        name: Name of the counter
        description: Description of the counter
        labels: Optional list of label names
        
    Returns:
        Function to increment the counter
    """
    if not labels:
        labels = []
    
    # Create a key for this counter
    key = f"{name}_{','.join(labels)}"
    
    # Create the counter if it doesn't exist
    if key not in _counters:
        if _metrics_enabled and _prometheus_client:
            _counters[key] = _prometheus_client.Counter(
                name,
                description,
                labels
            )
        else:
            # Use a simple dictionary for counters
            _counters[key] = {
                'name': name,
                'description': description,
                'labels': labels,
                'values': {}
            }
    
    # Return a function to increment the counter
    def increment(amount: float = 1, **label_values):
        """Increment the counter by the given amount"""
        if _metrics_enabled and _prometheus_client:
            if labels:
                # Convert label values to a tuple for label lookup
                label_tuple = tuple(label_values.get(label, "") for label in labels)
                _counters[key].labels(*label_tuple).inc(amount)
            else:
                _counters[key].inc(amount)
        else:
            # Simple counter implementation
            if labels:
                # Create a key from the label values
                label_key = "_".join(f"{k}={v}" for k, v in label_values.items())
                if label_key not in _counters[key]['values']:
                    _counters[key]['values'][label_key] = 0
                _counters[key]['values'][label_key] += amount
            else:
                if 'value' not in _counters[key]:
                    _counters[key]['value'] = 0
                _counters[key]['value'] += amount
            
            # Log the increment
            label_str = ", ".join(f"{k}={v}" for k, v in label_values.items()) if label_values else ""
            logger.debug(f"Counter {name} {label_str} incremented by {amount}")
    
    return increment

def histogram(name: str, description: str, labels: Optional[List[str]] = None, 
              buckets: Optional[List[float]] = None) -> Callable:
    """
    Get a histogram metric or create it if it doesn't exist
    
    Args:
        name: Name of the histogram
        description: Description of the histogram
        labels: Optional list of label names
        buckets: Optional list of histogram buckets
        
    Returns:
        Function to observe values for the histogram
    """
    if not labels:
        labels = []
    
    if not buckets:
        # Default buckets for response times in milliseconds
        buckets = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    
    # Create a key for this histogram
    key = f"{name}_{','.join(labels)}"
    
    # Create the histogram if it doesn't exist
    if key not in _histograms:
        if _metrics_enabled and _prometheus_client:
            _histograms[key] = _prometheus_client.Histogram(
                name,
                description,
                labels,
                buckets=buckets
            )
        else:
            # Use a simple dictionary for histograms
            _histograms[key] = {
                'name': name,
                'description': description,
                'labels': labels,
                'buckets': buckets,
                'values': {}
            }
    
    # Return a function to observe values for the histogram
    def observe(value: float, **label_values):
        """Observe a value for the histogram"""
        if _metrics_enabled and _prometheus_client:
            if labels:
                # Convert label values to a tuple for label lookup
                label_tuple = tuple(label_values.get(label, "") for label in labels)
                _histograms[key].labels(*label_tuple).observe(value)
            else:
                _histograms[key].observe(value)
        else:
            # Simple histogram implementation (just log the value)
            if labels:
                # Create a key from the label values
                label_key = "_".join(f"{k}={v}" for k, v in label_values.items())
                if label_key not in _histograms[key]['values']:
                    _histograms[key]['values'][label_key] = []
                _histograms[key]['values'][label_key].append(value)
            else:
                if 'values' not in _histograms[key]:
                    _histograms[key]['values'] = []
                _histograms[key]['values'].append(value)
            
            # Log the observation
            label_str = ", ".join(f"{k}={v}" for k, v in label_values.items()) if label_values else ""
            logger.debug(f"Histogram {name} {label_str} observed value {value}")
    
    return observe

@contextmanager
def timed_execution(name: str, labels: Optional[Dict[str, str]] = None):
    """
    Context manager to time execution of a code block and record to histogram
    
    Args:
        name: Name of the operation being timed
        labels: Optional dictionary of label key-value pairs
    """
    if not labels:
        labels = {}
    
    # Create a histogram for this operation if needed
    observe = histogram(
        f"{name}_duration_ms",
        f"Execution time of {name} in milliseconds",
        list(labels.keys())
    )
    
    # Record start time
    start_time = time.time()
    
    try:
        # Yield control to the wrapped code
        yield
    finally:
        # Record execution time in milliseconds
        execution_time = (time.time() - start_time) * 1000.0
        observe(execution_time, **labels)
        logger.debug(f"{name} completed in {execution_time:.2f}ms")

# Initialize metrics on module import if feature flag is enabled
if is_enabled('METRICS'):
    init_metrics()

# Create some common metrics
api_requests = counter(
    "api_requests_total",
    "Total number of API requests",
    ["provider", "endpoint"]
)

api_errors = counter(
    "api_errors_total",
    "Total number of API request errors",
    ["provider", "endpoint", "error_type"]
)

api_latency = histogram(
    "api_request_duration_ms",
    "API request latency in milliseconds",
    ["provider", "endpoint"]
)

product_found = counter(
    "products_found_total",
    "Total number of products found",
    ["source", "has_price", "has_image"]
)

search_requests = counter(
    "search_requests_total",
    "Total number of search requests",
    ["object_class", "has_metadata"]
)

cache_stats = counter(
    "cache_operations_total",
    "Cache operation statistics",
    ["operation", "tier", "hit"]
) 