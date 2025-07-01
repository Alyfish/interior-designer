"""
Location Service - Handle location-based product filtering and availability
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st

logger = logging.getLogger(__name__)


class LocationService:
    """Service for handling location-based features"""
    
    # Country codes mapping for SerpAPI
    COUNTRY_CODES = {
        "United States": {"code": "us", "currency": "$", "language": "en"},
        "United Kingdom": {"code": "uk", "currency": "£", "language": "en"},
        "Canada": {"code": "ca", "currency": "$", "language": "en"},
        "Australia": {"code": "au", "currency": "$", "language": "en"},
        "Germany": {"code": "de", "currency": "€", "language": "de"},
        "France": {"code": "fr", "currency": "€", "language": "fr"},
        "Italy": {"code": "it", "currency": "€", "language": "it"},
        "Spain": {"code": "es", "currency": "€", "language": "es"},
        "Japan": {"code": "jp", "currency": "¥", "language": "ja"},
        "India": {"code": "in", "currency": "₹", "language": "en"},
        "Brazil": {"code": "br", "currency": "R$", "language": "pt"},
        "Mexico": {"code": "mx", "currency": "$", "language": "es"}
    }
    
    # Major cities for location-specific searches
    MAJOR_CITIES = {
        "us": ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ"],
        "uk": ["London", "Manchester", "Birmingham", "Leeds", "Glasgow"],
        "ca": ["Toronto, ON", "Vancouver, BC", "Montreal, QC", "Calgary, AB", "Ottawa, ON"],
        "au": ["Sydney, NSW", "Melbourne, VIC", "Brisbane, QLD", "Perth, WA", "Adelaide, SA"],
        "de": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne"],
        "fr": ["Paris", "Marseille", "Lyon", "Toulouse", "Nice"],
        "jp": ["Tokyo", "Osaka", "Nagoya", "Yokohama", "Kobe"]
    }
    
    @classmethod
    def get_location_params(cls, country_code: str, city: Optional[str] = None) -> Dict[str, Any]:
        """
        Get location parameters for API searches
        
        Args:
            country_code: Two-letter country code
            city: Optional city name
            
        Returns:
            Dictionary with location parameters for API calls
        """
        params = {
            "gl": country_code,  # Google country parameter
            "hl": "en"  # Default to English
        }
        
        # Get country-specific settings
        for country_name, country_data in cls.COUNTRY_CODES.items():
            if country_data["code"] == country_code:
                params["hl"] = country_data.get("language", "en")
                params["currency"] = country_data.get("currency", "$")
                break
        
        # Add city-specific location if provided
        if city:
            params["location"] = city
            logger.info(f"Location params set for: {city}, {country_code}")
        else:
            logger.info(f"Location params set for country: {country_code}")
        
        return params
    
    @classmethod
    def filter_products_by_location(cls, products: List[Dict], country_code: str) -> List[Dict]:
        """
        Filter products based on shipping/availability to location
        
        Args:
            products: List of product dictionaries
            country_code: Target country code
            
        Returns:
            Filtered list of products available in the location
        """
        filtered_products = []
        
        for product in products:
            # Check if product has shipping information
            shipping_info = product.get('shipping', {})
            
            # If no shipping info, include by default (assume available)
            if not shipping_info:
                product['local_availability'] = True
                filtered_products.append(product)
                continue
            
            # Check if ships to country
            ships_to = shipping_info.get('ships_to', [])
            if isinstance(ships_to, str):
                ships_to = [ships_to]
            
            # Check various shipping indicators
            if (country_code in ships_to or 
                'worldwide' in str(ships_to).lower() or
                'international' in str(ships_to).lower() or
                not ships_to):  # Empty means no restrictions
                
                product['local_availability'] = True
                filtered_products.append(product)
            else:
                # Mark as not available locally but still include with warning
                product['local_availability'] = False
                product['availability_warning'] = f"May not ship to {country_code.upper()}"
                filtered_products.append(product)
        
        logger.info(f"Filtered {len(products)} products for {country_code}: {len(filtered_products)} available")
        return filtered_products
    
    @classmethod
    def enhance_search_query_with_location(cls, query: str, country_code: str, city: Optional[str] = None) -> str:
        """
        Enhance search query with location-specific terms
        
        Args:
            query: Original search query
            country_code: Country code
            city: Optional city name
            
        Returns:
            Enhanced query with location context
        """
        # Add location-specific terms based on country
        location_terms = {
            "us": ["USA", "American"],
            "uk": ["UK", "British"],
            "ca": ["Canada", "Canadian"],
            "au": ["Australia", "Australian"],
            "de": ["Germany", "German", "Deutschland"],
            "fr": ["France", "French"],
            "jp": ["Japan", "Japanese"]
        }
        
        # Don't add location to query if it might limit results too much
        # Instead, use location parameters in API call
        return query
    
    @classmethod
    def get_local_stores(cls, country_code: str, product_category: str) -> List[str]:
        """
        Get list of popular stores for the country and product category
        
        Args:
            country_code: Country code
            product_category: Type of product (e.g., "furniture", "sofa")
            
        Returns:
            List of store names popular in that country
        """
        stores_by_country = {
            "us": ["Wayfair", "IKEA", "West Elm", "CB2", "Crate & Barrel", "Amazon", "Home Depot", "Target", "Pottery Barn"],
            "uk": ["IKEA", "John Lewis", "Argos", "Dunelm", "Made.com", "Habitat", "Next Home", "Marks & Spencer"],
            "ca": ["IKEA", "Wayfair.ca", "HomeSense", "The Bay", "Canadian Tire", "Structube", "Article"],
            "au": ["IKEA", "Freedom", "Fantastic Furniture", "Harvey Norman", "Temple & Webster", "Bunnings"],
            "de": ["IKEA", "Otto", "Höffner", "XXXLutz", "Home24", "Wayfair.de"],
            "fr": ["IKEA", "Conforama", "But", "Maisons du Monde", "La Redoute", "Leroy Merlin"],
            "jp": ["IKEA", "Nitori", "MUJI", "Francfranc", "Amazon.jp"]
        }
        
        return stores_by_country.get(country_code, ["IKEA", "Amazon", "Local Furniture Stores"])
    
    @classmethod
    def format_price_for_location(cls, price: float, country_code: str) -> str:
        """
        Format price according to country's currency
        
        Args:
            price: Price value
            country_code: Country code
            
        Returns:
            Formatted price string with appropriate currency symbol
        """
        # Get currency symbol for country
        currency_symbol = "$"  # Default
        for country_name, country_data in cls.COUNTRY_CODES.items():
            if country_data["code"] == country_code:
                currency_symbol = country_data.get("currency", "$")
                break
        
        # Format based on currency
        if currency_symbol in ["€", "£"]:
            return f"{currency_symbol}{price:,.2f}"
        elif currency_symbol == "¥":
            return f"{currency_symbol}{int(price):,}"
        elif currency_symbol == "₹":
            return f"{currency_symbol}{price:,.0f}"
        else:
            return f"{currency_symbol}{price:,.2f}"
    
    @staticmethod
    def save_location_preference():
        """Save user's location preference to session state"""
        if 'location_preference' not in st.session_state:
            st.session_state.location_preference = {
                'country_code': 'us',
                'country_name': 'United States',
                'city': None
            }
    
    @staticmethod
    def get_location_preference() -> Dict[str, Any]:
        """Get user's saved location preference"""
        return st.session_state.get('location_preference', {
            'country_code': 'us',
            'country_name': 'United States',
            'city': None
        })