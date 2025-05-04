"""
Async utilities for throttling and concurrency control.
"""
import asyncio
import functools
import random
import time
from typing import TypeVar, Callable, Awaitable, Any, Dict, Optional, List, Tuple

T = TypeVar('T')

# Global semaphores for different services to avoid overwhelming them
_semaphores: Dict[str, asyncio.Semaphore] = {}
_rate_limiters: Dict[str, Dict[str, float]] = {}  # service -> {endpoint -> last_call_time}

# Default concurrency limits per service
DEFAULT_CONCURRENCY = 8
SERVICE_LIMITS = {
    "serpapi": 5,       # SerpAPI
    "rapidapi": 5,      # RapidAPI (Amazon)
    "bing": 10,         # Bing Visual Search
    "openai": 5,        # OpenAI API
    "playwright": 3,    # Browser automation concurrency
    "default": DEFAULT_CONCURRENCY  # Default for other services
}

def get_semaphore(service: str) -> asyncio.Semaphore:
    """
    Get or create a semaphore for a specific service.
    
    Args:
        service: Service name (serpapi, rapidapi, bing, etc.)
        
    Returns:
        asyncio.Semaphore: Semaphore for the service
    """
    if service not in _semaphores:
        limit = SERVICE_LIMITS.get(service, SERVICE_LIMITS["default"])
        _semaphores[service] = asyncio.Semaphore(limit)
    return _semaphores[service]

def rate_limited(min_interval: float = 0.2):
    """
    Rate limiting decorator to enforce minimum time between subsequent calls
    to the same service endpoint.
    
    Args:
        min_interval: Minimum time in seconds between calls
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create unique endpoint identifier
            service = kwargs.get('service', 'default')
            endpoint = kwargs.get('endpoint', func.__name__)
            key = f"{service}:{endpoint}"
            
            # Initialize rate limiter for this service if needed
            if service not in _rate_limiters:
                _rate_limiters[service] = {}
                
            # Get last call time
            last_call = _rate_limiters[service].get(endpoint, 0)
            now = time.time()
            
            # If we need to wait, sleep for the remaining time
            elapsed = now - last_call
            if elapsed < min_interval:
                delay = min_interval - elapsed
                await asyncio.sleep(delay)
            
            # Update last call time
            _rate_limiters[service][endpoint] = time.time()
            
            # Call the function
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def throttled(service: str = "default"):
    """
    Throttle API calls to limit concurrency using semaphores.
    
    Args:
        service: Service name to throttle (serpapi, rapidapi, etc.)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the semaphore for this service
            sem = get_semaphore(service)
            
            # Use the semaphore as a context manager
            async with sem:
                return await func(*args, **kwargs)
        return wrapper
    return decorator

async def gather_with_concurrency(n: int, *tasks):
    """
    Run up to n tasks concurrently.
    Similar to asyncio.gather but with a concurrency limit.
    
    Args:
        n: Maximum number of concurrent tasks
        tasks: Tasks to run
        
    Returns:
        List of task results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*(sem_task(task) for task in tasks))

def retry_async(
    max_retries: int = 3,
    exceptions: Tuple = (Exception,),
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0
):
    """
    Retry decorator for async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        exceptions: Exceptions to catch for retry
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for exponential backoff
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        # Max retries exceeded, re-raise the exception
                        raise
                    
                    # Calculate delay with jitter
                    delay = min(base_delay * (backoff_factor ** (retry_count - 1)), max_delay)
                    jitter = random.uniform(0.8, 1.2)  # +/- 20% jitter
                    delay *= jitter
                    
                    print(f"Retry {retry_count}/{max_retries} for {func.__name__} after {delay:.2f}s - {e}")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

async def process_concurrently(items: List[Any], processor_func, max_concurrency: int = DEFAULT_CONCURRENCY):
    """
    Process a list of items concurrently with limited concurrency.
    
    Args:
        items: List of items to process
        processor_func: Async function to process each item
        max_concurrency: Maximum number of concurrent tasks
        
    Returns:
        List of results from processing each item
    """
    tasks = [processor_func(item) for item in items]
    return await gather_with_concurrency(max_concurrency, *tasks)

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for async functions.
    Tracks failures and stops calling the protected function when too many failures occur.
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        window_size: int = 10
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.window_size = window_size
        self.failures: Dict[str, List[float]] = {}  # service -> list of failure timestamps
        self.open_circuits: Dict[str, float] = {}   # service -> time when circuit opened
    
    def is_open(self, service: str) -> bool:
        """Check if circuit is open (too many failures)"""
        if service in self.open_circuits:
            open_time = self.open_circuits[service]
            now = time.time()
            
            # Check if recovery timeout has passed
            if now - open_time > self.recovery_timeout:
                # Reset circuit to half-open state
                del self.open_circuits[service]
                return False
            
            # Circuit is still open
            return True
        
        # Circuit is closed or half-open
        return False
    
    def record_failure(self, service: str) -> None:
        """Record a failure for the service"""
        now = time.time()
        
        # Initialize failure list if needed
        if service not in self.failures:
            self.failures[service] = []
        
        # Add failure timestamp
        self.failures[service].append(now)
        
        # Remove old failures outside the window
        window_start = now - self.recovery_timeout
        self.failures[service] = [t for t in self.failures[service] if t >= window_start]
        
        # Check if threshold is exceeded
        if len(self.failures[service]) >= self.failure_threshold:
            # Open the circuit
            self.open_circuits[service] = now
    
    def record_success(self, service: str) -> None:
        """Record a successful call to the service"""
        # If service has an open circuit, close it (move to half-open)
        if service in self.open_circuits:
            del self.open_circuits[service]
        
        # Reset failures for this service
        if service in self.failures:
            self.failures[service] = []

# Global circuit breaker instance
circuit_breaker = CircuitBreaker()

def with_circuit_breaker(service: str):
    """
    Circuit breaker decorator for async functions.
    
    Args:
        service: Service name to track (serpapi, rapidapi, etc.)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if circuit is open
            if circuit_breaker.is_open(service):
                raise Exception(f"Circuit open for {service} - too many failures")
            
            try:
                result = await func(*args, **kwargs)
                # Record success
                circuit_breaker.record_success(service)
                return result
            except Exception as e:
                # Record failure
                circuit_breaker.record_failure(service)
                raise e
        return wrapper
    return decorator 