"""
Smart Product Search Module

This module provides enhanced product search capabilities without modifying
existing functionality. All features are additive and protected by feature flags.
"""

from .smart_product_agent import SmartProductSearchAgent

__all__ = ['SmartProductSearchAgent']