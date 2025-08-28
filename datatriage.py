#!/usr/bin/env python3
"""
Enhanced Blackbird Data Triage Script - With Place ID Verification
Key improvements:
1. PLACE ID VERIFICATION PHASE - Validates all existing Place IDs before matching
2. Intelligent pre-filtering to reduce API calls by 90%+
3. Advanced address normalization and parsing
4. Geographic bucketing for efficient matching
5. Name similarity pre-screening before Claude calls
"""

import pandas as pd
import googlemaps
import anthropic
import json
import time
import logging
import re
import os
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from collections import defaultdict
import hashlib
import pickle
from pathlib import Path

# For fuzzy string matching
try:
    from rapidfuzz import fuzz, process
except ImportError:
    print("Installing rapidfuzz for better string matching...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'rapidfuzz'])
    from rapidfuzz import fuzz, process

print("=" * 80)
print("🚀 ENHANCED BLACKBIRD DATA TRIAGE SCRIPT - WITH PLACE ID VERIFICATION")
print("=" * 80)
print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'enhanced_triage_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_place_id(place_id) -> Optional[str]:
    """
    Safely convert Place ID to string, handling NaN, None, and other data types
    
    Returns:
        String Place ID if valid, None otherwise
    """
    if place_id is None or pd.isna(place_id):
        return None
    
    try:
        place_id_str = str(place_id).strip()
        if place_id_str.lower() in ['nan', 'none', '']:
            return None
        return place_id_str
    except Exception:
        return None

def safe_string(value, default: str = '') -> str:
    """
    Safely convert any value to string, handling NaN and None
    """
    if value is None or pd.isna(value):
        return default
    try:
        return str(value).strip()
    except Exception:
        return default


# ============================================================================
# ADDRESS NORMALIZATION MODULE
# ============================================================================

class AddressNormalizer:
    """Advanced address normalization and parsing"""
    
    # Common abbreviations and their full forms
    STREET_ABBR = {
        'st': 'street', 'ave': 'avenue', 'rd': 'road', 'blvd': 'boulevard',
        'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
        'sq': 'square', 'ter': 'terrace', 'pkwy': 'parkway', 'hwy': 'highway',
        'cir': 'circle', 'trl': 'trail', 'way': 'way', 'plz': 'plaza'
    }
    
    # Directional abbreviations
    DIRECTIONS = {
        'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
        'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest'
    }
    
    # State abbreviations mapping
    STATES = {
        'ny': 'new york', 'ca': 'california', 'sc': 'south carolina',
        'co': 'colorado', 'tn': 'tennessee', 'la': 'louisiana',
        'dc': 'district of columbia', 'fl': 'florida', 'il': 'illinois',
        'ma': 'massachusetts', 'nj': 'new jersey', 'pa': 'pennsylvania'
    }
    
    # City to Macro Geo mapping based on your data
    CITY_TO_MACRO = {
        'new york': 'NYC', 'brooklyn': 'NYC', 'queens': 'NYC', 
        'bronx': 'NYC', 'staten island': 'NYC', 'manhattan': 'NYC',
        'san francisco': 'SF', 'oakland': 'SF', 'berkeley': 'SF',
        'charleston': 'CHS', 'mt pleasant': 'CHS', 'mount pleasant': 'CHS',
        'north charleston': 'CHS', 'denver': 'DEN', 'boulder': 'DEN',
        'montauk': 'LI', 'east hampton': 'LI', 'amagansett': 'LI',
        'sag harbor': 'LI', 'bridgehampton': 'LI', 'southampton': 'LI'
    }
    
    @staticmethod
    def normalize_address(address: str) -> Dict[str, str]:
        """
        Normalize and parse an address into components
        
        Returns:
            Dict with keys: normalized, street, city, state, zip, tokens
        """
        if not address or pd.isna(address):
            return {'normalized': '', 'street': '', 'city': '', 'state': '', 'zip': '', 'tokens': []}
        
        # Convert to lowercase and clean
        addr = str(address).lower().strip()
        
        # Remove extra spaces and special characters
        addr = re.sub(r'\s+', ' ', addr)
        addr = re.sub(r'[^\w\s,.-]', '', addr)
        
        # Extract ZIP code if present
        zip_match = re.search(r'\b(\d{5})(-\d{4})?\b', addr)
        zip_code = zip_match.group(1) if zip_match else ''
        if zip_match:
            addr = addr[:zip_match.start()] + addr[zip_match.end():]
        
        # Parse components
        components = addr.split(',')
        
        result = {
            'normalized': addr,
            'street': '',
            'city': '',
            'state': '',
            'zip': zip_code,
            'tokens': []
        }
        
        if len(components) >= 1:
            result['street'] = AddressNormalizer._normalize_street(components[0].strip())
        
        if len(components) >= 2:
            # Parse city and state from second component
            city_state = components[1].strip()
            state_match = re.search(r'\b([a-z]{2})\b$', city_state)
            
            if state_match:
                result['state'] = state_match.group(1)
                result['city'] = city_state[:state_match.start()].strip()
            else:
                result['city'] = city_state
        
        if len(components) >= 3:
            # Third component is likely state if not already found
            if not result['state']:
                state_part = components[2].strip()
                state_match = re.search(r'\b([a-z]{2})\b', state_part)
                if state_match:
                    result['state'] = state_match.group(1)
        
        # Generate tokens for matching
        result['tokens'] = AddressNormalizer._generate_tokens(result)
        
        # Create final normalized version
        parts = [result['street'], result['city'], result['state'], result['zip']]
        result['normalized'] = ', '.join([p for p in parts if p])
        
        return result
    
    @staticmethod
    def _normalize_street(street: str) -> str:
        """Normalize street address component"""
        if not street:
            return ''
        
        street = street.lower().strip()
        
        # Remove suite/apt/floor indicators
        street = re.sub(r'\b(suite|ste|apt|apartment|unit|floor|fl)\b.*', '', street, flags=re.IGNORECASE)
        
        # Expand abbreviations
        tokens = street.split()
        normalized_tokens = []
        
        for token in tokens:
            # Check if it's a direction
            if token in AddressNormalizer.DIRECTIONS:
                normalized_tokens.append(AddressNormalizer.DIRECTIONS[token])
            # Check if it's a street type
            elif token in AddressNormalizer.STREET_ABBR:
                normalized_tokens.append(AddressNormalizer.STREET_ABBR[token])
            else:
                normalized_tokens.append(token)
        
        return ' '.join(normalized_tokens).strip()
    
    @staticmethod
    def _generate_tokens(addr_dict: Dict[str, str]) -> List[str]:
        """Generate matching tokens from address components"""
        tokens = []
        
        # Add street number and name tokens
        if addr_dict['street']:
            street_tokens = addr_dict['street'].split()
            tokens.extend(street_tokens)
            
            # Add just the street number if present
            if street_tokens and street_tokens[0].isdigit():
                tokens.append(street_tokens[0])
        
        # Add city tokens
        if addr_dict['city']:
            tokens.append(addr_dict['city'])
            # Add individual words from city name
            tokens.extend(addr_dict['city'].split())
        
        # Add state
        if addr_dict['state']:
            tokens.append(addr_dict['state'])
        
        # Add zip
        if addr_dict['zip']:
            tokens.append(addr_dict['zip'])
        
        return [t for t in tokens if t]  # Remove empty tokens
    
    @staticmethod
    def get_macro_geo(city: str, state: str = None) -> str:
        """Map city to macro geographic region"""
        if not city:
            return 'Unknown'
        
        city_lower = city.lower().strip()
        
        # Direct city match
        if city_lower in AddressNormalizer.CITY_TO_MACRO:
            return AddressNormalizer.CITY_TO_MACRO[city_lower]
        
        # Check if city contains any known city name
        for known_city, macro in AddressNormalizer.CITY_TO_MACRO.items():
            if known_city in city_lower:
                return macro
        
        # Fall back to state-based mapping
        if state:
            state_lower = state.lower().strip()
            if state_lower in ['ny', 'new york']:
                return 'NYC'
            elif state_lower in ['ca', 'california']:
                return 'SF'
            elif state_lower in ['sc', 'south carolina']:
                return 'CHS'
            elif state_lower in ['co', 'colorado']:
                return 'DEN'
        
        return 'Unknown'


# ============================================================================
# NAME NORMALIZATION MODULE  
# ============================================================================

class RestaurantNameNormalizer:
    """Normalize restaurant names for better matching"""
    
    # Common suffixes to remove or standardize
    SUFFIXES_TO_REMOVE = [
        'restaurant', 'cafe', 'bistro', 'grill', 'grille', 'bar', 
        'kitchen', 'eatery', 'diner', 'tavern', 'pub', 'lounge',
        'llc', 'inc', 'corp', 'corporation', 'company', 'co'
    ]
    
    # Location indicators that might be part of name
    LOCATION_INDICATORS = [
        'nyc', 'sf', 'dc', 'la', 'brooklyn', 'manhattan', 'queens',
        'downtown', 'uptown', 'midtown', 'soma', 'fidi', 'tribeca',
        'chelsea', 'williamsburg', 'bushwick', 'harlem', 'financial district',
        'alphabet city', 'east village', 'west village', 'upper east side',
        'upper west side', 'gramercy', 'soho', 'carroll gardens'
    ]
    
    @staticmethod
    def normalize(name: str) -> Dict[str, Any]:
        """
        Normalize restaurant name and extract components
        
        Returns:
            Dict with normalized name, tokens, and location hints
        """
        if not name or pd.isna(name):
            return {'normalized': '', 'tokens': [], 'location_hint': ''}
        
        original = str(name)
        name = original.lower().strip()
        
        # Remove special characters but keep spaces and basic punctuation
        name = re.sub(r'[^\w\s&\'-]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Extract location hints if present
        location_hint = ''
        for loc in RestaurantNameNormalizer.LOCATION_INDICATORS:
            if loc in name:
                location_hint = loc
                # Keep location in name for multi-location restaurants
                # Don't remove it as it might be important for distinction
        
        # Remove common suffixes
        tokens = name.split()
        filtered_tokens = []
        for token in tokens:
            if token not in RestaurantNameNormalizer.SUFFIXES_TO_REMOVE:
                filtered_tokens.append(token)
        
        # Generate normalized version
        normalized = ' '.join(filtered_tokens)
        
        # Generate character n-grams for fuzzy matching
        ngrams = RestaurantNameNormalizer._generate_ngrams(normalized, 3)
        
        return {
            'original': original,
            'normalized': normalized,
            'tokens': filtered_tokens,
            'location_hint': location_hint,
            'ngrams': ngrams
        }
    
    @staticmethod
    def _generate_ngrams(text: str, n: int = 3) -> Set[str]:
        """Generate character n-grams for fuzzy matching"""
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}


# ============================================================================
# ENHANCED RESTAURANT RECORD
# ============================================================================

@dataclass
class EnhancedRestaurantRecord:
    """Enhanced data class with normalized fields"""
    source: str
    row_index: int
    
    # Original fields
    deal_name: Optional[str] = None
    company_name: Optional[str] = None
    restaurant_name: Optional[str] = None
    location_name: Optional[str] = None
    address: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    coordinate: Optional[str] = None
    google_places_id: Optional[str] = None
    macro_geo: Optional[str] = None
    
    # Normalized fields (computed after initialization)
    normalized_name: Dict[str, Any] = field(default_factory=dict)
    normalized_address: Dict[str, str] = field(default_factory=dict)
    computed_macro_geo: Optional[str] = None
    signature: Optional[str] = None  # For deduplication
    
    def __post_init__(self):
        """Compute normalized fields after initialization"""
        # Normalize name (prefer Deal Name for HubSpot as per instruction)
        if self.source == 'hubspot' and self.deal_name:
            self.normalized_name = RestaurantNameNormalizer.normalize(self.deal_name)
        elif self.restaurant_name:
            self.normalized_name = RestaurantNameNormalizer.normalize(self.restaurant_name)
        elif self.company_name:
            self.normalized_name = RestaurantNameNormalizer.normalize(self.company_name)
        elif self.deal_name:
            self.normalized_name = RestaurantNameNormalizer.normalize(self.deal_name)
        else:
            self.normalized_name = RestaurantNameNormalizer.normalize('')
        
        # Normalize address
        if self.address and not pd.isna(self.address):
            self.normalized_address = AddressNormalizer.normalize_address(str(self.address))
        elif self.street and not pd.isna(self.street):
            # Construct address from components
            addr_parts = [self.street, self.city, self.state, str(self.zipcode) if self.zipcode else '']
            full_addr = ', '.join([p for p in addr_parts if p and str(p).strip() and str(p).strip().lower() != 'nan'])
            self.normalized_address = AddressNormalizer.normalize_address(full_addr)
        else:
            self.normalized_address = AddressNormalizer.normalize_address('')
        
        # Compute macro geo - FIX: Check if macro_geo is a string before calling upper()
        if self.macro_geo and not pd.isna(self.macro_geo):
            # Convert to string and clean up macro geo (handle variations like 'nyc', 'NYC', 'New York')
            self.computed_macro_geo = str(self.macro_geo).upper().strip()
            if self.computed_macro_geo in ['NEW YORK', 'UWS/UES', 'UES/UWS']:
                self.computed_macro_geo = 'NYC'
        else:
            # Try to infer from address
            city = self.normalized_address.get('city', '') or self.city or ''
            state = self.normalized_address.get('state', '') or self.state or ''
            self.computed_macro_geo = AddressNormalizer.get_macro_geo(city, state)
        
        # Generate signature for deduplication
        self._generate_signature()
    
    def _generate_signature(self):
        """Generate a signature for identifying potential duplicates"""
        # Use normalized name and key address components
        sig_parts = [
            self.normalized_name.get('normalized', ''),
            self.normalized_address.get('street', ''),
            self.normalized_address.get('city', ''),
            self.computed_macro_geo or ''
        ]
        sig_string = '|'.join([str(p).lower() for p in sig_parts if p])
        self.signature = hashlib.md5(sig_string.encode()).hexdigest()
    
    def get_search_query(self) -> str:
        """
        Generate best search query for Google Places
        For Database: Use address (known to be correct)
        For HubSpot: Use name for multi-location, address if seems unique
        """
        if self.source == 'database':
            # Database addresses are trusted - use full address
            if self.address:
                return self.address
            elif self.street and self.city:
                return f"{self.street}, {self.city}, {self.state} {self.zipcode}".strip()
            else:
                # Fallback to name + location
                return f"{self.normalized_name.get('original', '')} {self.location_name or ''} {self.city or ''}".strip()
        
        else:  # HubSpot
            # For HubSpot, check if name contains location indicators (suggests multi-location)
            name = self.normalized_name.get('original', '')
            has_location_in_name = any(loc in name.lower() for loc in RestaurantNameNormalizer.LOCATION_INDICATORS)
            
            if has_location_in_name:
                # Multi-location restaurant - use name as primary search
                if self.city or self.normalized_address.get('city'):
                    city = self.city or self.normalized_address.get('city')
                    return f"{name}, {city}"
                else:
                    return name
            else:
                # Single location - can trust address more
                if self.address and not pd.isna(self.address) and isinstance(self.address, str) and len(self.address) > 10:  # Has substantial address
                    return f"{name}, {self.address}"
                else:
                    # Just use name
                    return name


# ============================================================================
# PLACE ID VERIFICATION MODULE
# ============================================================================

class PlaceIDVerifier:
    """Verify and correct Google Place IDs before matching"""
    
    def __init__(self, gmaps_client):
        self.gmaps = gmaps_client
        self.verification_cache = {}
        self.api_calls = 0
    
    def verify_place_id(self, record: EnhancedRestaurantRecord) -> Dict[str, Any]:
        """
        Verify if a Place ID is correct for the given restaurant
        
        Returns:
            Dict with keys: is_valid, correct_place_id, place_name, confidence, reason
        """
        place_id = safe_place_id(record.google_places_id)
        if not place_id:
            return {
                'is_valid': False,
                'correct_place_id': None,
                'place_name': None,
                'confidence': 0,
                'reason': 'No Place ID provided'
            }
        
        # Check cache
        cache_key = f"{place_id}_{record.signature}"
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
        
        try:
            # Get place details
            place_details = self.gmaps.place(
                place_id=place_id,
                fields=['name', 'formatted_address', 'place_id', 'type', 'business_status']
            )
            self.api_calls += 1
            
            if 'result' not in place_details:
                result = {
                    'is_valid': False,
                    'correct_place_id': None,
                    'place_name': None,
                    'confidence': 0,
                    'reason': 'Place ID not found'
                }
            else:
                place = place_details['result']
                
                # Compare place details with record
                place_name = place.get('name', '').lower()
                place_address = place.get('formatted_address', '').lower()
                
                record_name = record.normalized_name.get('normalized', '').lower()
                record_tokens = set(record.normalized_name.get('tokens', []))
                
                # Calculate match confidence
                confidence = 0
                reasons = []
                
                # Name similarity check
                if record_name and place_name:
                    name_similarity = fuzz.ratio(record_name, place_name) / 100.0
                    if name_similarity > 0.8:
                        confidence += 0.5
                        reasons.append(f"Name match ({name_similarity:.2f})")
                    elif name_similarity > 0.6:
                        confidence += 0.3
                        reasons.append(f"Partial name match ({name_similarity:.2f})")
                    else:
                        # Check token overlap
                        place_tokens = set(place_name.split())
                        if record_tokens and place_tokens:
                            overlap = len(record_tokens & place_tokens) / len(record_tokens | place_tokens)
                            if overlap > 0.5:
                                confidence += 0.2
                                reasons.append(f"Token overlap ({overlap:.2f})")
                
                # Address similarity check
                if record.normalized_address.get('street'):
                    addr_similarity = fuzz.partial_ratio(
                        record.normalized_address.get('normalized', ''),
                        place_address
                    ) / 100.0
                    if addr_similarity > 0.8:
                        confidence += 0.5
                        reasons.append(f"Address match ({addr_similarity:.2f})")
                    elif addr_similarity > 0.6:
                        confidence += 0.3
                        reasons.append(f"Partial address match ({addr_similarity:.2f})")
                
                # Check if it's a restaurant/food establishment
                place_types = place.get('type', [])
                if any(t in place_types for t in ['restaurant', 'food', 'bar', 'cafe', 'bakery', 'meal_takeaway']):
                    confidence += 0.1
                    reasons.append("Is food establishment")
                
                result = {
                    'is_valid': confidence >= 0.6,
                    'correct_place_id': place_id if confidence >= 0.6 else None,
                    'place_name': place.get('name'),
                    'place_address': place.get('formatted_address'),
                    'confidence': confidence,
                    'reason': '; '.join(reasons) if reasons else 'Low confidence match'
                }
            
            self.verification_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error verifying Place ID {place_id}: {e}")
            return {
                'is_valid': False,
                'correct_place_id': None,
                'place_name': None,
                'confidence': 0,
                'reason': f'Error: {str(e)}'
            }
    
    def find_correct_place_id(self, record: EnhancedRestaurantRecord, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Search for the correct Place ID for a restaurant
        
        Returns:
            Dict with keys: found, place_id, place_name, confidence, search_query
        """
        search_query = record.get_search_query()
        
        if not search_query:
            return {
                'found': False,
                'place_id': None,
                'place_name': None,
                'confidence': 0,
                'search_query': None
            }
        
        try:
            # Search for the place
            search_results = self.gmaps.places(query=search_query)
            self.api_calls += 1
            
            if not search_results.get('results'):
                # Try alternative search with just name
                if record.normalized_name.get('original'):
                    alt_query = record.normalized_name.get('original')
                    if record.city:
                        alt_query += f", {record.city}"
                    
                    search_results = self.gmaps.places(query=alt_query)
                    self.api_calls += 1
                    
                    if not search_results.get('results'):
                        return {
                            'found': False,
                            'place_id': None,
                            'place_name': None,
                            'confidence': 0,
                            'search_query': search_query
                        }
            
            # Analyze top results
            best_match = None
            best_confidence = 0
            
            for place in search_results['results'][:min(3, len(search_results['results']))]:
                place_name = place.get('name', '').lower()
                place_address = place.get('formatted_address', '').lower()
                
                # Calculate match confidence
                confidence = 0
                
                # Name similarity
                if record.normalized_name.get('normalized'):
                    name_sim = fuzz.ratio(
                        record.normalized_name.get('normalized'),
                        place_name
                    ) / 100.0
                    confidence += name_sim * 0.5
                
                # Address similarity
                if record.normalized_address.get('normalized'):
                    addr_sim = fuzz.partial_ratio(
                        record.normalized_address.get('normalized'),
                        place_address
                    ) / 100.0
                    confidence += addr_sim * 0.5
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = place
            
            if best_match and best_confidence >= 0.6:
                return {
                    'found': True,
                    'place_id': best_match['place_id'],
                    'place_name': best_match.get('name'),
                    'place_address': best_match.get('formatted_address'),
                    'confidence': best_confidence,
                    'search_query': search_query
                }
            else:
                return {
                    'found': False,
                    'place_id': None,
                    'place_name': None,
                    'confidence': best_confidence,
                    'search_query': search_query
                }
                
        except Exception as e:
            logger.error(f"Error searching for place: {e}")
            return {
                'found': False,
                'place_id': None,
                'place_name': None,
                'confidence': 0,
                'search_query': search_query,
                'error': str(e)
            }


# ============================================================================
# INTELLIGENT PRE-FILTERING ENGINE
# ============================================================================

class IntelligentPreFilter:
    """Pre-filter candidates to drastically reduce API calls"""
    
    def __init__(self):
        self.name_normalizer = RestaurantNameNormalizer()
        self.address_normalizer = AddressNormalizer()
        
        # Caches for optimization
        self.similarity_cache = {}
        self.candidate_cache = {}
    
    def find_candidates(self, 
                       query_record: EnhancedRestaurantRecord,
                       target_records: List[EnhancedRestaurantRecord],
                       max_candidates: int = 10,
                       min_similarity: float = 0.4) -> List[Tuple[EnhancedRestaurantRecord, float]]:
        """
        Find best matching candidates using multiple strategies
        
        Returns:
            List of (record, score) tuples, sorted by score descending
        """
        candidates = []
        
        # Strategy 1: Exact Google Place ID match (if available and verified)
        query_place_id = safe_place_id(query_record.google_places_id)
        if query_place_id:
            for target in target_records:
                target_place_id = safe_place_id(target.google_places_id)
                if target_place_id and target_place_id == query_place_id:
                    candidates.append((target, 1.0))  # Perfect score for Place ID match
        
        # Strategy 2: Geographic bucketing - only compare within same region
        geo_filtered = self._filter_by_geography(query_record, target_records)
        
        # Strategy 3: Name-based similarity
        name_candidates = self._find_by_name_similarity(query_record, geo_filtered, max_candidates)
        
        # Strategy 4: Address-based similarity (if we have address data)
        if query_record.normalized_address.get('street'):
            addr_candidates = self._find_by_address_similarity(query_record, geo_filtered, max_candidates)
            # Merge address candidates with lower weight
            for target, score in addr_candidates:
                # Check if already in candidates
                existing = next((c for c in name_candidates if c[0].row_index == target.row_index), None)
                if existing:
                    # Combine scores (weighted average)
                    old_score = existing[1]
                    new_score = old_score * 0.7 + score * 0.3
                    name_candidates = [(t, s) if t.row_index != target.row_index else (t, new_score) 
                                     for t, s in name_candidates]
                else:
                    name_candidates.append((target, score * 0.5))  # Lower weight for address-only match
        
        # Combine all candidates
        candidates.extend(name_candidates)
        
        # Remove duplicates and filter by minimum similarity
        seen = set()
        final_candidates = []
        for target, score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if target.row_index not in seen and score >= min_similarity:
                seen.add(target.row_index)
                final_candidates.append((target, score))
                if len(final_candidates) >= max_candidates:
                    break
        
        return final_candidates
    
    def _filter_by_geography(self, 
                           query: EnhancedRestaurantRecord,
                           targets: List[EnhancedRestaurantRecord]) -> List[EnhancedRestaurantRecord]:
        """Filter targets to same geographic region"""
        if not query.computed_macro_geo or query.computed_macro_geo == 'Unknown':
            return targets  # Can't filter if we don't know the region
        
        filtered = []
        for target in targets:
            # Include if same macro geo OR if target geo is unknown (might still match)
            if (target.computed_macro_geo == query.computed_macro_geo or 
                target.computed_macro_geo == 'Unknown' or
                not target.computed_macro_geo):
                filtered.append(target)
        
        # If filtering removed too many candidates, include some without geo
        if len(filtered) < 20:
            for target in targets:
                if target not in filtered:
                    filtered.append(target)
                    if len(filtered) >= 50:  # Reasonable upper limit
                        break
        
        return filtered
    
    def _find_by_name_similarity(self,
                                query: EnhancedRestaurantRecord,
                                targets: List[EnhancedRestaurantRecord],
                                max_candidates: int) -> List[Tuple[EnhancedRestaurantRecord, float]]:
        """Find candidates by restaurant name similarity"""
        if not query.normalized_name.get('normalized'):
            return []
        
        query_name = query.normalized_name['normalized']
        query_tokens = set(query.normalized_name['tokens'])
        
        scored_targets = []
        for target in targets:
            if not target.normalized_name.get('normalized'):
                continue
            
            target_name = target.normalized_name['normalized']
            target_tokens = set(target.normalized_name['tokens'])
            
            # Calculate multiple similarity metrics
            
            # 1. Fuzzy string similarity
            fuzzy_score = fuzz.ratio(query_name, target_name) / 100.0
            
            # 2. Token overlap (Jaccard similarity)
            if query_tokens and target_tokens:
                token_overlap = len(query_tokens & target_tokens) / len(query_tokens | target_tokens)
            else:
                token_overlap = 0
            
            # 3. Partial ratio (handles substrings)
            partial_score = fuzz.partial_ratio(query_name, target_name) / 100.0
            
            # 4. Token sort ratio (handles word order differences)
            token_sort_score = fuzz.token_sort_ratio(query_name, target_name) / 100.0
            
            # Combine scores with weights
            combined_score = (
                fuzzy_score * 0.3 +
                token_overlap * 0.2 +
                partial_score * 0.25 +
                token_sort_score * 0.25
            )
            
            # Boost score if location hints match
            if (query.normalized_name.get('location_hint') and 
                target.normalized_name.get('location_hint') and
                query.normalized_name['location_hint'] == target.normalized_name['location_hint']):
                combined_score = min(1.0, combined_score * 1.2)
            
            scored_targets.append((target, combined_score))
        
        # Sort by score and return top candidates
        scored_targets.sort(key=lambda x: x[1], reverse=True)
        return scored_targets[:max_candidates]
    
    def _find_by_address_similarity(self,
                                   query: EnhancedRestaurantRecord,
                                   targets: List[EnhancedRestaurantRecord],
                                   max_candidates: int) -> List[Tuple[EnhancedRestaurantRecord, float]]:
        """Find candidates by address similarity"""
        if not query.normalized_address.get('street'):
            return []
        
        query_tokens = set(query.normalized_address.get('tokens', []))
        if not query_tokens:
            return []
        
        scored_targets = []
        for target in targets:
            target_tokens = set(target.normalized_address.get('tokens', []))
            if not target_tokens:
                continue
            
            # Token overlap score
            overlap_score = len(query_tokens & target_tokens) / len(query_tokens | target_tokens)
            
            # Street number match bonus
            query_street = query.normalized_address.get('street', '')
            target_street = target.normalized_address.get('street', '')
            
            if query_street and target_street:
                # Extract street numbers
                query_num = re.match(r'^(\d+)', query_street)
                target_num = re.match(r'^(\d+)', target_street)
                
                if query_num and target_num and query_num.group(1) == target_num.group(1):
                    overlap_score = min(1.0, overlap_score * 1.5)  # Boost for matching street number
            
            # ZIP code match bonus
            if (query.normalized_address.get('zip') and 
                target.normalized_address.get('zip') and
                query.normalized_address['zip'] == target.normalized_address['zip']):
                overlap_score = min(1.0, overlap_score * 1.3)
            
            if overlap_score > 0.2:  # Minimum threshold
                scored_targets.append((target, overlap_score))
        
        scored_targets.sort(key=lambda x: x[1], reverse=True)
        return scored_targets[:max_candidates]


# ============================================================================
# ENHANCED DATA TRIAGE AGENT
# ============================================================================

class EnhancedDataTriageAgent:
    """Enhanced agent with Place ID verification and intelligent pre-filtering"""
    
    def __init__(self, google_api_key: str, anthropic_api_key: str):
        """Initialize the enhanced triage agent"""
        print("🔧 Initializing Enhanced DataTriageAgent with Place ID Verification...")
        
        self.gmaps = googlemaps.Client(key=google_api_key)
        self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        self.pre_filter = IntelligentPreFilter()
        self.place_verifier = PlaceIDVerifier(self.gmaps)
        
        # Caches to avoid redundant API calls
        self.google_cache = {}
        self.claude_cache = {}
        
        # API call counters
        self.google_api_calls = 0
        self.claude_api_calls = 0
        self.pre_filter_eliminations = 0
        self.place_id_corrections = 0
        
        print("✅ Enhanced DataTriageAgent initialization complete")
    
    def verify_and_correct_place_ids(self, 
                                    records: List[EnhancedRestaurantRecord],
                                    df: pd.DataFrame,
                                    source_name: str) -> int:
        """
        Verify all Place IDs and correct them if needed
        
        Returns:
            Number of corrections made
        """
        print(f"\n🔍 Verifying {source_name} Place IDs...")
        corrections = 0
        verified = 0
        invalid = 0
        
        verification_errors = []
        
        for i, record in enumerate(records):
            if i % 50 == 0 and i > 0:
                print(f"  Progress: {i}/{len(records)} records verified")
            
            try:
                place_id = safe_place_id(record.google_places_id)
                if not place_id:
                    # Try to find Place ID
                    result = self.place_verifier.find_correct_place_id(record)
                    if result['found']:
                        df.at[record.row_index, 'google_places_id' if source_name == 'HubSpot' else 'google_place_id'] = result['place_id']
                        df.at[record.row_index, 'place_id_verification'] = f"Found: {result['place_name']} (conf: {result['confidence']:.2f})"
                        corrections += 1
                        print(f"  ✅ Found Place ID for {record.normalized_name.get('original')}: {result['place_name']}")
                    else:
                        df.at[record.row_index, 'place_id_verification'] = "Not found"
                else:
                    # Verify existing Place ID
                    verification = self.place_verifier.verify_place_id(record)
                    
                    if verification['is_valid']:
                        df.at[record.row_index, 'place_id_verification'] = f"Valid: {verification['place_name']} (conf: {verification['confidence']:.2f})"
                        verified += 1
                    else:
                        # Try to find correct Place ID
                        result = self.place_verifier.find_correct_place_id(record)
                        if result['found'] and result['place_id'] != record.google_places_id:
                            old_id = record.google_places_id
                            df.at[record.row_index, 'google_places_id' if source_name == 'HubSpot' else 'google_place_id'] = result['place_id']
                            df.at[record.row_index, 'place_id_verification'] = f"Corrected: {result['place_name']} (conf: {result['confidence']:.2f})"
                            corrections += 1
                            print(f"  🔄 Corrected Place ID for {record.normalized_name.get('original')}")
                            print(f"     Old: {old_id}")
                            print(f"     New: {result['place_id']} ({result['place_name']})")
                        else:
                            df.at[record.row_index, 'place_id_verification'] = f"Invalid: {verification['reason']}"
                            invalid += 1
                            
            except Exception as e:
                error_msg = f"Error verifying Place ID for record {i} ({record.normalized_name.get('original', 'Unknown')}): {str(e)}"
                logger.error(error_msg, exc_info=True)
                verification_errors.append({
                    'record_index': i,
                    'record_name': record.normalized_name.get('original', 'Unknown'),
                    'error': str(e)
                })
                
                # Mark as verification error
                df.at[record.row_index, 'place_id_verification'] = f"Error: {str(e)[:50]}"
                invalid += 1
                print(f"  ❌ Error verifying {record.normalized_name.get('original', 'Unknown')}: {str(e)}")
        
        print(f"\n📊 {source_name} Place ID Verification Results:")
        if verification_errors:
            print(f"  ⚠️ Verification errors: {len(verification_errors)}")
        print(f"  ✅ Valid: {verified}")
        print(f"  🔄 Corrected: {corrections}")
        print(f"  ❌ Invalid/Not Found: {invalid}")
        print(f"  Total API calls: {self.place_verifier.api_calls}")
        
        return corrections
    
    def process_datasets(self, 
                        hubspot_df: pd.DataFrame, 
                        database_df: pd.DataFrame,
                        output_timestamp: str,
                        test_mode: bool = False,
                        skip_verification: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process datasets with Place ID verification and intelligent pre-filtering"""
        
        print("\n" + "="*80)
        print("📊 ENHANCED PROCESSING WITH PLACE ID VERIFICATION")
        print("="*80)
        print(f"📋 HubSpot records: {len(hubspot_df)}")
        print(f"🗄️ Database records: {len(database_df)}")
        
        if test_mode:
            print("\n⚠️ TEST MODE - Sorting alphabetically and taking top 30 for better overlap")
            
            # Sort HubSpot by Deal Name
            hubspot_df = hubspot_df.sort_values('Deal Name', na_position='last').copy()
            print(f"  📋 HubSpot sorted by Deal Name")
            
            # Sort Database by restaurant_name
            database_df = database_df.sort_values('restaurant_name', na_position='last').copy()
            print(f"  🗄️ Database sorted by restaurant_name")
            
            # Take first 30 from each sorted dataset and reset index
            hubspot_df = hubspot_df.head(30).copy().reset_index(drop=True)
            database_df = database_df.head(30).copy().reset_index(drop=True)
            
            print(f"  📋 Limited to first 30 HubSpot records (alphabetically)")
            print(f"  🗄️ Limited to first 30 Database records (alphabetically)")
            
            # Show what we're working with
            print("\n  Sample of HubSpot records:")
            for i in range(min(5, len(hubspot_df))):
                print(f"    {i+1}. {hubspot_df.iloc[i]['Deal Name']}")
            
            print("\n  Sample of Database records:")
            for i in range(min(5, len(database_df))):
                print(f"    {i+1}. {database_df.iloc[i]['restaurant_name']}")
        
        # Add tracking columns
        for df in [hubspot_df, database_df]:
            df['place_id_verification'] = ''
            df['match_status'] = ''
            df['match_confidence'] = 0.0
            df['match_method'] = ''
            df['verified_place_id'] = ''
            df['match_notes'] = ''
        
        # Convert to Enhanced Restaurant Records
        print("\n🔄 Converting to Enhanced Restaurant Records with normalization...")
        
        hubspot_records = []
        record_creation_errors = []
        
        for idx, row in hubspot_df.iterrows():
            try:
                record = EnhancedRestaurantRecord(
                    source='hubspot',
                    row_index=idx,
                    deal_name=row.get('Deal Name'),
                    company_name=row.get('Company name'),
                    address=row.get('Address'),
                    google_places_id=row.get('google_places_id'),
                    macro_geo=row.get('Macro Geo (NYC, SF, CHS, DC, LA, NASH, DEN)')
                )
                hubspot_records.append(record)
            except Exception as e:
                error_msg = f"Error creating HubSpot record {idx}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                record_creation_errors.append({
                    'source': 'hubspot',
                    'index': idx,
                    'error': str(e)
                })
                # Mark the row with error
                hubspot_df.at[idx, 'match_status'] = 'CREATION_ERROR'
                hubspot_df.at[idx, 'match_notes'] = f'Record creation error: {str(e)[:100]}'
        
        database_records = []
        for idx, row in database_df.iterrows():
            try:
                record = EnhancedRestaurantRecord(
                    source='database',
                    row_index=idx,
                    restaurant_name=row.get('restaurant_name'),
                    location_name=row.get('location_name'),
                    street=row.get('street'),
                    city=row.get('city'),
                    state=row.get('state'),
                    zipcode=str(row.get('zipcode')) if pd.notna(row.get('zipcode')) else None,
                    coordinate=row.get('coordinate'),
                    google_places_id=row.get('google_place_id')
                )
                database_records.append(record)
            except Exception as e:
                error_msg = f"Error creating Database record {idx}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                record_creation_errors.append({
                    'source': 'database',
                    'index': idx,
                    'error': str(e)
                })
                # Mark the row with error
                database_df.at[idx, 'match_status'] = 'CREATION_ERROR'
                database_df.at[idx, 'match_notes'] = f'Record creation error: {str(e)[:100]}'
        
        if record_creation_errors:
            print(f"\n⚠️ Record creation errors: {len(record_creation_errors)}")
            for error in record_creation_errors[:3]:
                print(f"  - {error['source']} record {error['index']}: {error['error'][:60]}")
        
        # PHASE 1: PLACE ID VERIFICATION
        if not skip_verification:
            print("\n" + "="*60)
            print("PHASE 1: VERIFYING AND CORRECTING PLACE IDs")
            print("="*60)
            
            # Verify Database Place IDs (trusted addresses)
            db_corrections = self.verify_and_correct_place_ids(
                database_records, database_df, "Database"
            )
            
            # Verify HubSpot Place IDs (may need more corrections)
            hs_corrections = self.verify_and_correct_place_ids(
                hubspot_records, hubspot_df, "HubSpot"
            )
            
            self.place_id_corrections = db_corrections + hs_corrections
            
            # Update records with corrected Place IDs
            for record, idx in zip(hubspot_records, range(len(hubspot_df))):
                record.google_places_id = hubspot_df.at[idx, 'google_places_id']
            
            for record, idx in zip(database_records, range(len(database_df))):
                record.google_places_id = database_df.at[idx, 'google_place_id']
            
            # Update Google API counter
            self.google_api_calls += self.place_verifier.api_calls
            
            print(f"\n✅ Place ID verification complete. Total corrections: {self.place_id_corrections}")
        else:
            print("\n⚠️ Skipping Place ID verification (skip_verification=True)")
        
        # PHASE 2: INTELLIGENT MATCHING
        print("\n" + "="*60)
        print("PHASE 2: INTELLIGENT MATCHING WITH PRE-FILTERING")
        print("="*60)
        
        # Create geographic buckets for efficiency
        print("\n🌍 Creating geographic buckets...")
        geo_buckets_db = defaultdict(list)
        for rec in database_records:
            geo_buckets_db[rec.computed_macro_geo].append(rec)
        
        print(f"Geographic distribution in database:")
        for geo, records in sorted(geo_buckets_db.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {geo}: {len(records)} records")
        
        # Track matched records
        matched_hs = set()
        matched_db = set()
        
        total_comparisons_without_filter = len(hubspot_records) * len(database_records)
        actual_comparisons = 0
        place_id_matches = 0
        claude_matches = 0
        
        # Track processing errors
        processing_errors = []
        
        for hs_idx, hs_record in enumerate(hubspot_records):
            if hs_record.row_index in matched_hs:
                continue
            
            try:
                print(f"\n[{hs_idx+1}/{len(hubspot_records)}] Processing: {hs_record.normalized_name.get('original', 'Unknown')}")
                
                # First, check for exact Place ID match (after verification)
                hs_place_id = safe_place_id(hs_record.google_places_id)
                if hs_place_id:
                    place_id_match = None
                    for db_record in database_records:
                        db_place_id = safe_place_id(db_record.google_places_id)
                        if (db_record.row_index not in matched_db and 
                            db_place_id and db_place_id == hs_place_id):
                            place_id_match = db_record
                            break
                
                if place_id_match:
                    print(f"  ✅ Exact Place ID match found (verified)")
                    
                    # Record the match
                    hubspot_df.at[hs_record.row_index, 'match_status'] = 'MATCHED'
                    hubspot_df.at[hs_record.row_index, 'match_confidence'] = 1.0
                    hubspot_df.at[hs_record.row_index, 'match_method'] = 'verified_place_id'
                    hubspot_df.at[hs_record.row_index, 'match_notes'] = f'Matched to DB row {place_id_match.row_index}'
                    
                    database_df.at[place_id_match.row_index, 'match_status'] = 'MATCHED'
                    database_df.at[place_id_match.row_index, 'match_confidence'] = 1.0
                    database_df.at[place_id_match.row_index, 'match_method'] = 'verified_place_id'
                    database_df.at[place_id_match.row_index, 'match_notes'] = f'Matched to HS row {hs_record.row_index}'
                    
                    matched_hs.add(hs_record.row_index)
                    matched_db.add(place_id_match.row_index)
                    place_id_matches += 1
                    continue
                
                # Use pre-filter to find candidates
                candidates = self.pre_filter.find_candidates(
                    hs_record, 
                    [r for r in database_records if r.row_index not in matched_db],
                    max_candidates=5,
                    min_similarity=0.5
                )
                
                self.pre_filter_eliminations += len(database_records) - len(candidates)
                
                if not candidates:
                    print(f"  ❌ No candidates found by pre-filter")
                    hubspot_df.at[hs_record.row_index, 'match_status'] = 'NO_MATCH'
                    hubspot_df.at[hs_record.row_index, 'match_notes'] = 'No candidates passed pre-filtering'
                    continue
                
                print(f"  📊 Pre-filter found {len(candidates)} candidates")
            
                # Check top candidates with Claude
                best_match = None
                best_score = 0
                best_method = ''
                
                for db_record, pre_score in candidates[:3]:  # Only check top 3
                    actual_comparisons += 1
                    
                    print(f"    🤖 Checking: {db_record.normalized_name.get('original')} (pre-score: {pre_score:.2f})")
                    
                    # High confidence without Claude if names and addresses are very similar
                    if pre_score > 0.85:
                        print(f"      ✅ Auto-matched (high similarity)")
                        best_match = db_record
                        best_score = pre_score
                        best_method = 'high_similarity'
                        break
                    
                    # Use Claude for verification
                    verification = self._claude_verify_match(hs_record, db_record, pre_score)
                    self.claude_api_calls += 1
                    
                    if verification['is_match'] and verification['confidence'] > best_score:
                        best_match = db_record
                        best_score = verification['confidence']
                        best_method = 'claude_verified'
                        print(f"      ✅ Claude confirmed (confidence: {verification['confidence']:.2f})")
                    else:
                        print(f"      ❌ Claude rejected (confidence: {verification.get('confidence', 0):.2f})")
                
                # Record the match if found
                if best_match and best_score >= 0.6:
                    hubspot_df.at[hs_record.row_index, 'match_status'] = 'MATCHED'
                    hubspot_df.at[hs_record.row_index, 'match_confidence'] = best_score
                    hubspot_df.at[hs_record.row_index, 'match_method'] = best_method
                    hubspot_df.at[hs_record.row_index, 'match_notes'] = f'Matched to DB row {best_match.row_index}'
                    
                    database_df.at[best_match.row_index, 'match_status'] = 'MATCHED'
                    database_df.at[best_match.row_index, 'match_confidence'] = best_score
                    database_df.at[best_match.row_index, 'match_method'] = best_method
                    database_df.at[best_match.row_index, 'match_notes'] = f'Matched to HS row {hs_record.row_index}'
                    
                    matched_hs.add(hs_record.row_index)
                    matched_db.add(best_match.row_index)
                    
                    if best_method == 'claude_verified':
                        claude_matches += 1
                    
                    print(f"  ✅ MATCHED with confidence {best_score:.2f}")
                else:
                    hubspot_df.at[hs_record.row_index, 'match_status'] = 'NO_MATCH'
                    hubspot_df.at[hs_record.row_index, 'match_notes'] = 'No confident match found'
                    print(f"  ❌ No confident match found")
                
            except Exception as e:
                error_msg = f"Error processing record {hs_idx+1} ({hs_record.normalized_name.get('original', 'Unknown')}): {str(e)}"
                logger.error(error_msg, exc_info=True)
                processing_errors.append({
                    'record_index': hs_idx,
                    'record_name': hs_record.normalized_name.get('original', 'Unknown'),
                    'error': str(e)
                })
                
                # Mark record as error
                hubspot_df.at[hs_record.row_index, 'match_status'] = 'ERROR'
                hubspot_df.at[hs_record.row_index, 'match_notes'] = f'Processing error: {str(e)[:100]}'
                print(f"  ❌ ERROR: {str(e)}")
                print(f"  ⏭️  Continuing with next record...")
                continue
        
        # Mark remaining unmatched records
        for idx in range(len(database_df)):
            if idx not in matched_db:
                database_df.at[idx, 'match_status'] = 'UNMATCHED'
                database_df.at[idx, 'match_notes'] = 'Not matched to any HubSpot record'
        
        # Print comprehensive statistics
        print("\n" + "="*80)
        print("📊 PROCESSING COMPLETE - COMPREHENSIVE REPORT")
        print("="*80)
        
        print("\n🔍 PLACE ID VERIFICATION:")
        print(f"  Total corrections made: {self.place_id_corrections}")
        
        print("\n🎯 MATCHING RESULTS:")
        print(f"  Total HubSpot records: {len(hubspot_df)}")
        print(f"  Total Database records: {len(database_df)}")
        print(f"  Matched records: {len(matched_hs)}")
        print(f"    - Via verified Place ID: {place_id_matches}")
        print(f"    - Via Claude verification: {claude_matches}")
        print(f"    - Via high similarity: {len(matched_hs) - place_id_matches - claude_matches}")
        
        if processing_errors:
            print(f"\n⚠️ PROCESSING ERRORS:")
            print(f"  Records with errors: {len(processing_errors)}")
            print("  Error details:")
            for error in processing_errors[:5]:  # Show first 5 errors
                print(f"    - {error['record_name']}: {error['error'][:80]}")
            if len(processing_errors) > 5:
                print(f"    ... and {len(processing_errors) - 5} more errors (check log file)")
        
        print("\n⚡ EFFICIENCY METRICS:")
        print(f"  Possible comparisons without filtering: {total_comparisons_without_filter:,}")
        print(f"  Actual comparisons made: {actual_comparisons:,}")
        print(f"  Comparisons eliminated: {self.pre_filter_eliminations:,}")
        print(f"  Efficiency gain: {(1 - actual_comparisons/max(1, total_comparisons_without_filter))*100:.1f}%")
        
        print("\n💰 API USAGE & COSTS:")
        print(f"  Google Places API calls: {self.google_api_calls}")
        print(f"    - Place ID verifications: {self.place_verifier.api_calls}")
        print(f"  Claude API calls: {self.claude_api_calls}")
        print(f"  Estimated Google cost: ${self.google_api_calls * 0.005:.2f}")
        print(f"  Estimated Claude cost: ${self.claude_api_calls * 0.003:.2f}")
        print(f"  Total estimated cost: ${self.google_api_calls * 0.005 + self.claude_api_calls * 0.003:.2f}")
        
        return hubspot_df, database_df
    
    def _claude_verify_match(self, 
                           record1: EnhancedRestaurantRecord, 
                           record2: EnhancedRestaurantRecord,
                           pre_score: float) -> Dict[str, Any]:
        """Use Claude to verify if two records match"""
        
        prompt = f"""
        Determine if these two restaurant records represent the same restaurant.
        
        RECORD 1 (from {record1.source}):
        - Name: {record1.normalized_name.get('original', 'N/A')}
        - Normalized Name: {record1.normalized_name.get('normalized', 'N/A')}
        - Address: {record1.address or f"{record1.street}, {record1.city}, {record1.state}"}
        - Normalized Address: {record1.normalized_address.get('normalized', 'N/A')}
        - Location/Neighborhood: {record1.location_name}
        - Macro Region: {record1.computed_macro_geo}
        - Google Place ID: {record1.google_places_id}
        
        RECORD 2 (from {record2.source}):
        - Name: {record2.normalized_name.get('original', 'N/A')}
        - Normalized Name: {record2.normalized_name.get('normalized', 'N/A')}
        - Address: {record2.address or f"{record2.street}, {record2.city}, {record2.state}"}
        - Normalized Address: {record2.normalized_address.get('normalized', 'N/A')}
        - Location/Neighborhood: {record2.location_name}
        - Macro Region: {record2.computed_macro_geo}
        - Google Place ID: {record2.google_places_id}
        
        Pre-filtering similarity score: {pre_score:.2f}
        
        IMPORTANT CONTEXT:
        - The Deal Name (Record 1) from HubSpot is usually most accurate
        - Database addresses are always correct
        - HubSpot addresses may be wrong for multi-location restaurants
        - Names may have slight variations (e.g., "Joe's Pizza" vs "Joe's Pizza NYC")
        - Location suffixes in names often indicate different branches
        
        Consider:
        - Is this the same restaurant location, not just same brand?
        - Do the neighborhoods/locations match?
        - For chains, are these the same specific location?
        
        Respond with JSON only:
        {{
            "is_match": boolean,
            "confidence": 0.0 to 1.0,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.claude.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "is_match": False,
                    "confidence": 0,
                    "reasoning": "Could not parse Claude response"
                }
                
        except Exception as e:
            logger.error(f"Error in Claude verification: {e}")
            return {
                "is_match": False,
                "confidence": 0,
                "reasoning": f"Error: {str(e)}"
            }


def main(test_mode=True, skip_verification=False):
    """
    Main execution function
    
    Args:
        test_mode: If True, process limited records for testing
        skip_verification: If True, skip Place ID verification phase
    """
    print("\n" + "="*80)
    print("🏁 ENHANCED MAIN FUNCTION WITH PLACE ID VERIFICATION")
    print("="*80)
    
    # Configuration
    GOOGLE_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    
    if not GOOGLE_API_KEY or not ANTHROPIC_API_KEY:
        print("❌ ERROR: Missing required API keys")
        print("Please set GOOGLE_PLACES_API_KEY and ANTHROPIC_API_KEY environment variables")
        return
    
    # File paths
    HUBSPOT_FILE = 'hubspot_data.csv'
    DATABASE_FILE = 'database_data.csv'
    
    try:
        # Load data
        print("\n📂 Loading data files...")
        hubspot_df = pd.read_csv(HUBSPOT_FILE)
        database_df = pd.read_csv(DATABASE_FILE)
        print(f"✅ Data loaded: {len(hubspot_df)} HubSpot, {len(database_df)} Database records")
        
        # Initialize enhanced agent
        agent = EnhancedDataTriageAgent(GOOGLE_API_KEY, ANTHROPIC_API_KEY)
        
        # Process with enhanced algorithm
        output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_hubspot, processed_database = agent.process_datasets(
            hubspot_df, database_df, output_timestamp, 
            test_mode=test_mode,
            skip_verification=skip_verification
        )
        
        # Save results
        hubspot_output = f'enhanced_hubspot_{output_timestamp}.csv'
        database_output = f'enhanced_database_{output_timestamp}.csv'
        
        processed_hubspot.to_csv(hubspot_output, index=False)
        processed_database.to_csv(database_output, index=False)
        
        print(f"\n✅ Results saved to:")
        print(f"  📄 {hubspot_output}")
        print(f"  📄 {database_output}")
        print("\n🎉 ENHANCED SCRIPT COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        logger.error(f"Error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Configuration flags
    TEST_MODE = False  # KEEPING TEST MODE ON AS REQUESTED
    SKIP_VERIFICATION = False  # Set to True to skip Place ID verification
    
    if TEST_MODE:
        print("⚠️ TEST MODE IS ON - Processing alphabetically sorted top 30 records")
        print("Set TEST_MODE = False for full processing")
    
    if SKIP_VERIFICATION:
        print("⚠️ Place ID verification will be skipped")
    
    main(test_mode=TEST_MODE, skip_verification=SKIP_VERIFICATION)
