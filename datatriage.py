#!/usr/bin/env python3
"""
Blackbird Data Triage Script - Claude-Driven Matching
Reconciles restaurant data between backend database and HubSpot using:
1. Claude AI for ALL matching decisions (using Sonnet for cost efficiency)
2. Google Places API for verification and enrichment
3. Multi-stage matching with Claude verification at each step
"""

import pandas as pd
import googlemaps
import anthropic
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import os
from dataclasses import dataclass
import re
from difflib import SequenceMatcher

print("=" * 80)
print("🚀 BLACKBIRD DATA TRIAGE SCRIPT - CLAUDE-DRIVEN VERSION")
print("=" * 80)
print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_triage_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("✅ Logging configuration complete")

@dataclass
class RestaurantRecord:
    """Data class for restaurant information"""
    source: str  # 'hubspot' or 'database'
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
    row_index: Optional[int] = None

print("✅ RestaurantRecord dataclass defined")

class DataTriageAgent:
    """Main agent for triaging restaurant data between HubSpot and Database"""
    
    # Locations to skip
    SKIP_KEYWORDS = [
        'burger league', 'breakfast club', 'bar blackbird',
        'test', 'demo', 'example'
    ]
    
    def __init__(self, google_api_key: str, anthropic_api_key: str):
        """
        Initialize the triage agent
        
        Args:
            google_api_key: Google Places API key
            anthropic_api_key: Anthropic API key for Claude
        """
        print("🔧 Initializing DataTriageAgent...")
        print(f"   📍 Google API Key: {'✅ Present' if google_api_key else '❌ Missing'}")
        print(f"   🤖 Anthropic API Key: {'✅ Present' if anthropic_api_key else '❌ Missing'}")
        
        try:
            print("   🌐 Initializing Google Maps client...")
            self.gmaps = googlemaps.Client(key=google_api_key)
            print("   ✅ Google Maps client initialized")
        except Exception as e:
            print(f"   ❌ Error initializing Google Maps: {e}")
            raise
            
        try:
            print("   🤖 Initializing Anthropic client...")
            self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
            print("   ✅ Anthropic client initialized")
        except Exception as e:
            print(f"   ❌ Error initializing Anthropic: {e}")
            raise
            
        self.google_api_calls = 0
        self.claude_api_calls = 0
        print("✅ DataTriageAgent initialization complete")
    
    def should_skip_record(self, record: RestaurantRecord) -> bool:
        """Check if a record should be skipped based on keywords"""
        check_fields = [
            record.deal_name, record.company_name, 
            record.restaurant_name, record.location_name
        ]
        
        for field in check_fields:
            if field and isinstance(field, str):
                field_lower = field.lower()
                for keyword in self.SKIP_KEYWORDS:
                    if keyword in field_lower:
                        logger.info(f"Skipping record with keyword '{keyword}': {field}")
                        return True
        return False
    
    def build_search_variations(self, record: RestaurantRecord) -> List[str]:
        """
        Build multiple search query variations for better matching
        
        Returns list of search queries to try, from most to least specific
        """
        queries = []
        
        def make_query(parts):
            valid_parts = [str(p).strip() for p in parts if p and str(p).strip().lower() != 'nan']
            return ', '.join(valid_parts) if valid_parts else None
        
        # Get name variations
        names = [record.deal_name, record.restaurant_name, record.company_name]
        names = [n for n in names if n]
        
        # Full address query
        if record.address:
            for name in names:
                q = make_query([name, record.address])
                if q: queries.append(q)
        
        # Name + City + State
        if record.city:
            for name in names:
                q = make_query([name, record.city, record.state])
                if q: queries.append(q)
        
        # Just name + city
        if record.city:
            for name in names:
                q = make_query([name, record.city])
                if q: queries.append(q)
        
        # Try partial names (first few words) + city
        if record.city:
            for name in names:
                if name:
                    words = name.split()[:2]  # First 2 words
                    if words:
                        q = make_query([' '.join(words), record.city])
                        if q: queries.append(q)
        
        # Just the name
        for name in names:
            if name: queries.append(name)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:10]  # Limit to 10 variations
    
    def search_google_places_multi(self, record: RestaurantRecord, 
                                  existing_place_id: Optional[str] = None,
                                  max_attempts: int = 5) -> Optional[Dict]:
        """
        Search for a place using multiple query variations
        
        Returns dictionary with place information or None
        """
        # First, try to verify existing place ID if provided
        if existing_place_id and isinstance(existing_place_id, str) and existing_place_id.strip():
            try:
                place_result = self.gmaps.place(
                    place_id=existing_place_id,
                    fields=['name', 'formatted_address', 'place_id', 'type', 'business_status']
                )
                self.google_api_calls += 1
                if place_result.get('result'):
                    logger.info(f"Verified existing place ID: {existing_place_id}")
                    return place_result['result']
            except Exception as e:
                logger.warning(f"Could not verify place ID {existing_place_id}: {e}")
        
        # Get query variations
        queries = self.build_search_variations(record)
        
        if not queries:
            logger.warning("No valid search queries could be built")
            return None
        
        # Try each query variation
        for i, query in enumerate(queries[:max_attempts]):
            try:
                print(f"      🔍 Google search attempt {i+1}/{min(len(queries), max_attempts)}: '{query}'")
                search_results = self.gmaps.places(query=query)
                self.google_api_calls += 1
                
                if search_results.get('results'):
                    # Get the first result
                    place = search_results['results'][0]
                    
                    # Get detailed information
                    place_details = self.gmaps.place(
                        place_id=place['place_id'],
                        fields=['name', 'formatted_address', 'place_id', 'type', 
                               'business_status', 'geometry', 'website', 'formatted_phone_number']
                    )
                    self.google_api_calls += 1
                    
                    result = place_details.get('result', place)
                    print(f"      ✅ Found: {result.get('name')} at {result.get('formatted_address', 'N/A')}")
                    return result
                    
            except Exception as e:
                logger.error(f"Error with query '{query}': {e}")
                continue
        
        print(f"      ❌ No Google results found after {min(len(queries), max_attempts)} attempts")
        return None
    
    def claude_verify_place_id_match(self, record1: RestaurantRecord, record2: RestaurantRecord) -> Dict:
        """
        Use Claude to verify if two records with the same Google Place ID are actually duplicates
        or if one might be incorrect
        """
        prompt = f"""
        Analyze if these two restaurant records with the same Google Place ID are actually the same restaurant:
        
        RECORD 1 (from {record1.source}):
        - Name: {record1.deal_name or record1.restaurant_name or record1.company_name}
        - Location: {record1.location_name}
        - Address: {record1.address or f"{record1.street}, {record1.city}, {record1.state} {record1.zipcode}"}
        - Google Place ID: {record1.google_places_id}
        
        RECORD 2 (from {record2.source}):
        - Name: {record2.deal_name or record2.restaurant_name or record2.company_name}
        - Location: {record2.location_name}
        - Address: {record2.address or f"{record2.street}, {record2.city}, {record2.state} {record2.zipcode}"}
        - Google Place ID: {record2.google_places_id}
        
        Consider:
        - Could these be the same restaurant with slightly different names/info?
        - Could one have an incorrect Place ID (copy/paste error)?
        - Are the locations consistent?
        
        Respond with JSON:
        {{
            "is_valid_match": boolean,
            "confidence_score": 0.0 to 1.0,
            "reasoning": "explanation",
            "likely_same_restaurant": boolean
        }}
        """
        
        try:
            response = self.claude.messages.create(
                model="claude-3-5-sonnet-20241022",  # Using Sonnet for cost efficiency
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            self.claude_api_calls += 1
            
            response_text = response.content[0].text
            json_match = re.search(r'\{{.*\}}', response_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "is_valid_match": False,
                    "confidence_score": 0,
                    "reasoning": "Could not parse response",
                    "likely_same_restaurant": False
                }
                
        except Exception as e:
            logger.error(f"Error in Claude verification: {e}")
            return {
                "is_valid_match": False,
                "confidence_score": 0,
                "reasoning": f"Error: {str(e)}",
                "likely_same_restaurant": False
            }
    
    def claude_fuzzy_match(self, record1: RestaurantRecord, record2: RestaurantRecord, 
                          google_data1: Optional[Dict] = None, 
                          google_data2: Optional[Dict] = None) -> Dict:
        """
        Use Claude to determine if two records are the same restaurant using all available data
        """
        prompt = f"""
        Determine if these two restaurant records represent the same restaurant:
        
        RECORD 1 (from {record1.source}):
        - Name: {record1.deal_name or record1.restaurant_name or record1.company_name}
        - Location: {record1.location_name}
        - Address: {record1.address or f"{record1.street}, {record1.city}, {record1.state} {record1.zipcode}"}
        - Google Place ID: {record1.google_places_id}
        
        RECORD 2 (from {record2.source}):
        - Name: {record2.deal_name or record2.restaurant_name or record2.company_name}  
        - Location: {record2.location_name}
        - Address: {record2.address or f"{record2.street}, {record2.city}, {record2.state} {record2.zipcode}"}
        - Google Place ID: {record2.google_places_id}
        
        {f'''GOOGLE DATA FOR RECORD 1:
        - Name: {google_data1.get('name')}
        - Address: {google_data1.get('formatted_address')}
        - Place ID: {google_data1.get('place_id')}
        - Status: {google_data1.get('business_status')}''' if google_data1 else 'No Google data for Record 1'}
        
        {f'''GOOGLE DATA FOR RECORD 2:
        - Name: {google_data2.get('name')}
        - Address: {google_data2.get('formatted_address')}
        - Place ID: {google_data2.get('place_id')}
        - Status: {google_data2.get('business_status')}''' if google_data2 else 'No Google data for Record 2'}
        
        Consider:
        - Name variations (e.g., "Joe's Pizza" vs "Joe's Pizza NYC")
        - Address proximity and variations
        - Common abbreviations or naming differences
        - Chain locations vs unique restaurants
        
        Be reasonably flexible with matches - restaurants often have slight name variations across systems.
        
        Respond with JSON:
        {{
            "is_match": boolean,
            "confidence_score": 0.0 to 1.0,
            "reasoning": "explanation",
            "suggested_place_id": "the correct Google Place ID if known, else null"
        }}
        """
        
        try:
            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            self.claude_api_calls += 1
            
            response_text = response.content[0].text
            json_match = re.search(r'\{{.*\}}', response_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "is_match": False,
                    "confidence_score": 0,
                    "reasoning": "Could not parse response",
                    "suggested_place_id": None
                }
                
        except Exception as e:
            logger.error(f"Error in Claude fuzzy matching: {e}")
            return {
                "is_match": False,
                "confidence_score": 0,
                "reasoning": f"Error: {str(e)}",
                "suggested_place_id": None
            }
    
    def process_datasets(self, hubspot_df: pd.DataFrame, database_df: pd.DataFrame, 
                        output_timestamp: str, test_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process both datasets with Claude-driven matching at every step
        """
        print("\n" + "="*80)
        print("📊 PROCESSING DATASETS - CLAUDE-DRIVEN MATCHING")
        print("="*80)
        print(f"📋 HubSpot records: {len(hubspot_df)}")
        print(f"🗄️ Database records: {len(database_df)}")
        
        # TEST MODE: Limit to first 10 records if enabled
        if test_mode:
            print("\n⚠️ TEST MODE ENABLED - Processing only first 10 records from each dataset")
            hubspot_df = hubspot_df.head(10).copy()
            database_df = database_df.head(10).copy()
            print(f"📋 Limited HubSpot to: {len(hubspot_df)} records")
            print(f"🗄️ Limited Database to: {len(database_df)} records")
        
        # Set up incremental save file names
        hubspot_temp_file = f'hubspot_progress_{output_timestamp}.csv'
        database_temp_file = f'database_progress_{output_timestamp}.csv'
        
        # Add processing columns
        for df in [hubspot_df, database_df]:
            df['verified_place_id'] = ''
            df['confidence_score'] = 0.0
            df['match_status'] = ''
            df['match_method'] = ''
            df['notes'] = ''
            df['processed'] = False
        
        # Convert to RestaurantRecord objects
        print("\n🔄 Converting DataFrames to RestaurantRecord objects...")
        hubspot_records = []
        for idx, row in hubspot_df.iterrows():
            record = RestaurantRecord(
                source='hubspot',
                deal_name=row.get('Deal Name'),
                company_name=row.get('Company name'),
                address=row.get('Address'),
                google_places_id=row.get('google_places_id'),
                macro_geo=row.get('Macro Geo'),
                row_index=idx
            )
            hubspot_records.append(record)
        
        database_records = []
        for idx, row in database_df.iterrows():
            record = RestaurantRecord(
                source='database',
                restaurant_name=row.get('restaurant_name'),
                location_name=row.get('location_name'),
                street=row.get('street'),
                city=row.get('city'),
                state=row.get('state'),
                zipcode=str(row.get('zipcode')) if pd.notna(row.get('zipcode')) else None,
                coordinate=row.get('coordinate'),
                google_places_id=row.get('google_place_id'),
                row_index=idx
            )
            database_records.append(record)
        
        print(f"✅ Created {len(hubspot_records)} HubSpot and {len(database_records)} Database records")
        
        # Track matched records to avoid duplicate matching
        matched_hs = set()
        matched_db = set()
        
        # PHASE 1: Match by Google Place IDs with Claude verification
        print("\n" + "="*60)
        print("PHASE 1: MATCHING BY GOOGLE PLACE IDs (WITH CLAUDE VERIFICATION)")
        print("="*60)
        
        # Build Place ID maps
        db_place_id_map = {}
        for rec in database_records:
            if rec.google_places_id and isinstance(rec.google_places_id, str) and rec.google_places_id.strip():
                if rec.google_places_id not in db_place_id_map:
                    db_place_id_map[rec.google_places_id] = []
                db_place_id_map[rec.google_places_id].append(rec)
        
        # Check each HubSpot record with a Place ID
        place_id_matches = 0
        for hs_record in hubspot_records:
            if hs_record.row_index in matched_hs:
                continue
                
            if hs_record.google_places_id and isinstance(hs_record.google_places_id, str) and hs_record.google_places_id.strip():
                if hs_record.google_places_id in db_place_id_map:
                    db_candidates = db_place_id_map[hs_record.google_places_id]
                    
                    # Have Claude verify each potential match
                    for db_record in db_candidates:
                        if db_record.row_index in matched_db:
                            continue
                            
                        print(f"\n🤖 Claude verifying Place ID match: HS row {hs_record.row_index} ↔ DB row {db_record.row_index}")
                        verification = self.claude_verify_place_id_match(hs_record, db_record)
                        
                        if verification['is_valid_match'] and verification['confidence_score'] >= 0.7:
                            print(f"   ✅ Match confirmed (confidence: {verification['confidence_score']:.2f})")
                            
                            # Update HubSpot record
                            hubspot_df.at[hs_record.row_index, 'match_status'] = f'MATCHED (DB row {db_record.row_index})'
                            hubspot_df.at[hs_record.row_index, 'match_method'] = 'place_id_verified'
                            hubspot_df.at[hs_record.row_index, 'confidence_score'] = verification['confidence_score']
                            hubspot_df.at[hs_record.row_index, 'verified_place_id'] = hs_record.google_places_id
                            hubspot_df.at[hs_record.row_index, 'notes'] = verification['reasoning']
                            
                            # Update Database record
                            database_df.at[db_record.row_index, 'match_status'] = f'MATCHED (HS row {hs_record.row_index})'
                            database_df.at[db_record.row_index, 'match_method'] = 'place_id_verified'
                            database_df.at[db_record.row_index, 'confidence_score'] = verification['confidence_score']
                            database_df.at[db_record.row_index, 'verified_place_id'] = db_record.google_places_id
                            database_df.at[db_record.row_index, 'notes'] = verification['reasoning']
                            
                            matched_hs.add(hs_record.row_index)
                            matched_db.add(db_record.row_index)
                            place_id_matches += 1
                            break
                        else:
                            print(f"   ❌ Match rejected (confidence: {verification['confidence_score']:.2f})")
                            print(f"      Reason: {verification['reasoning']}")
        
        print(f"\n📊 Phase 1 complete: {place_id_matches} verified matches via Place ID")
        
        # Save progress
        hubspot_df.to_csv(hubspot_temp_file, index=False)
        database_df.to_csv(database_temp_file, index=False)
        
        # PHASE 2: Claude-driven fuzzy matching with Google enrichment
        print("\n" + "="*60)
        print("PHASE 2: CLAUDE-DRIVEN FUZZY MATCHING")
        print("="*60)
        
        unmatched_hs = [r for r in hubspot_records if r.row_index not in matched_hs]
        unmatched_db = [r for r in database_records if r.row_index not in matched_db]
        
        print(f"Remaining unmatched: {len(unmatched_hs)} HubSpot, {len(unmatched_db)} Database records")
        
        fuzzy_matches = 0
        for hs_record in unmatched_hs:
            if self.should_skip_record(hs_record):
                hubspot_df.at[hs_record.row_index, 'match_status'] = 'SKIPPED'
                hubspot_df.at[hs_record.row_index, 'notes'] = 'Contains skip keyword'
                continue
            
            print(f"\n🔍 Processing: {hs_record.deal_name or hs_record.company_name}")
            
            # Get Google data for the HubSpot record (if not too many API calls)
            google_hs = None
            if self.google_api_calls < 100:  # Limit API calls in test mode
                google_hs = self.search_google_places_multi(hs_record, hs_record.google_places_id, max_attempts=3)
                if google_hs:
                    hubspot_df.at[hs_record.row_index, 'verified_place_id'] = google_hs.get('place_id', '')
            
            # Check against each unmatched database record
            best_match = None
            best_score = 0
            best_verification = None
            
            for db_record in unmatched_db:
                if db_record.row_index in matched_db:
                    continue
                
                # Get Google data for the database record (if needed and not too many calls)
                google_db = None
                if self.google_api_calls < 100 and db_record.google_places_id and isinstance(db_record.google_places_id, str):
                    google_db = self.search_google_places_multi(db_record, db_record.google_places_id, max_attempts=2)
                
                # Have Claude evaluate the match
                print(f"   🤖 Claude comparing with DB: {db_record.restaurant_name}")
                verification = self.claude_fuzzy_match(hs_record, db_record, google_hs, google_db)
                
                if verification['is_match'] and verification['confidence_score'] > best_score:
                    best_match = db_record
                    best_score = verification['confidence_score']
                    best_verification = verification
            
            # If we found a good match, record it
            if best_match and best_score >= 0.6:
                print(f"   ✅ Best match found: DB row {best_match.row_index} (confidence: {best_score:.2f})")
                
                # Update HubSpot record
                hubspot_df.at[hs_record.row_index, 'match_status'] = f'MATCHED (DB row {best_match.row_index})'
                hubspot_df.at[hs_record.row_index, 'match_method'] = 'claude_fuzzy'
                hubspot_df.at[hs_record.row_index, 'confidence_score'] = best_score
                if best_verification['suggested_place_id']:
                    hubspot_df.at[hs_record.row_index, 'verified_place_id'] = best_verification['suggested_place_id']
                hubspot_df.at[hs_record.row_index, 'notes'] = best_verification['reasoning']
                
                # Update Database record
                database_df.at[best_match.row_index, 'match_status'] = f'MATCHED (HS row {hs_record.row_index})'
                database_df.at[best_match.row_index, 'match_method'] = 'claude_fuzzy'
                database_df.at[best_match.row_index, 'confidence_score'] = best_score
                if best_verification['suggested_place_id']:
                    database_df.at[best_match.row_index, 'verified_place_id'] = best_verification['suggested_place_id']
                database_df.at[best_match.row_index, 'notes'] = best_verification['reasoning']
                
                matched_hs.add(hs_record.row_index)
                matched_db.add(best_match.row_index)
                fuzzy_matches += 1
            else:
                print(f"   ❌ No confident match found")
                hubspot_df.at[hs_record.row_index, 'match_status'] = 'NO_MATCH_IN_DATABASE'
                if google_hs:
                    hubspot_df.at[hs_record.row_index, 'notes'] = f"Google: {google_hs.get('name', 'N/A')}"
            
            # Save progress periodically
            if (len(matched_hs) % 5) == 0:
                hubspot_df.to_csv(hubspot_temp_file, index=False)
                database_df.to_csv(database_temp_file, index=False)
            
            # Rate limiting
            time.sleep(0.2)  # Small delay to avoid overwhelming APIs
        
        print(f"\n📊 Phase 2 complete: {fuzzy_matches} matches via Claude fuzzy matching")
        
        # Mark remaining unmatched records
        for idx in range(len(hubspot_df)):
            if not hubspot_df.at[idx, 'match_status']:
                hubspot_df.at[idx, 'match_status'] = 'NO_MATCH_IN_DATABASE'
        
        for idx in range(len(database_df)):
            if not database_df.at[idx, 'match_status']:
                database_df.at[idx, 'match_status'] = 'NO_MATCH_IN_HUBSPOT'
        
        # Final save
        hubspot_df.to_csv(hubspot_temp_file, index=False)
        database_df.to_csv(database_temp_file, index=False)
        
        print(f"\n📊 PROCESSING COMPLETE:")
        print(f"   Total Google API calls: {self.google_api_calls}")
        print(f"   Total Claude API calls: {self.claude_api_calls}")
        print(f"   Estimated Google cost: ${self.google_api_calls * 0.005:.2f}")
        print(f"   Estimated Claude cost: ${self.claude_api_calls * 0.003:.2f} (Sonnet pricing)")
        print(f"   Total estimated cost: ${(self.google_api_calls * 0.005) + (self.claude_api_calls * 0.003):.2f}")
        
        return hubspot_df, database_df
    
    def generate_summary_report(self, hubspot_df: pd.DataFrame, database_df: pd.DataFrame) -> str:
        """Generate a summary report of the matching process"""
        report = []
        report.append("=" * 80)
        report.append("DATA TRIAGE SUMMARY REPORT - CLAUDE-DRIVEN VERSION")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Google API calls: {self.google_api_calls}")
        report.append(f"Total Claude API calls: {self.claude_api_calls}")
        report.append(f"Estimated total cost: ${(self.google_api_calls * 0.005) + (self.claude_api_calls * 0.003):.2f}")
        report.append("")
        
        # HubSpot summary
        report.append("HUBSPOT DATA SUMMARY:")
        report.append(f"Total records: {len(hubspot_df)}")
        
        # Match statistics
        matched_mask = hubspot_df['match_status'].astype(str).str.contains('MATCHED', na=False)
        report.append(f"Matched records: {matched_mask.sum()}")
        
        # Match methods breakdown
        if 'match_method' in hubspot_df.columns:
            place_id_matches = (hubspot_df['match_method'] == 'place_id_verified').sum()
            fuzzy_matches = (hubspot_df['match_method'] == 'claude_fuzzy').sum()
            report.append(f"  - Via Place ID (Claude verified): {place_id_matches}")
            report.append(f"  - Via Claude fuzzy matching: {fuzzy_matches}")
        
        report.append(f"No match in Database: {(hubspot_df['match_status'] == 'NO_MATCH_IN_DATABASE').sum()}")
        report.append(f"Skipped: {(hubspot_df['match_status'] == 'SKIPPED').sum()}")
        
        # Confidence score distribution
        conf_scores = hubspot_df[hubspot_df['confidence_score'] > 0]['confidence_score']
        if len(conf_scores) > 0:
            report.append(f"Confidence scores:")
            report.append(f"  - Average: {conf_scores.mean():.2f}")
            report.append(f"  - High confidence (>0.9): {(conf_scores > 0.9).sum()}")
            report.append(f"  - Medium confidence (0.7-0.9): {((conf_scores >= 0.7) & (conf_scores <= 0.9)).sum()}")
            report.append(f"  - Low confidence (<0.7): {(conf_scores < 0.7).sum()}")
        report.append("")
        
        # Database summary
        report.append("DATABASE SUMMARY:")
        report.append(f"Total records: {len(database_df)}")
        
        matched_mask_db = database_df['match_status'].astype(str).str.contains('MATCHED', na=False)
        report.append(f"Matched records: {matched_mask_db.sum()}")
        
        if 'match_method' in database_df.columns:
            place_id_matches = (database_df['match_method'] == 'place_id_verified').sum()
            fuzzy_matches = (database_df['match_method'] == 'claude_fuzzy').sum()
            report.append(f"  - Via Place ID (Claude verified): {place_id_matches}")
            report.append(f"  - Via Claude fuzzy matching: {fuzzy_matches}")
        
        report.append(f"No match in HubSpot: {(database_df['match_status'] == 'NO_MATCH_IN_HUBSPOT').sum()}")
        report.append(f"Skipped: {(database_df['match_status'] == 'SKIPPED').sum()}")
        report.append("")
        
        # Issues requiring attention
        report.append("RECORDS REQUIRING MANUAL REVIEW:")
        
        # Low confidence matches
        low_conf_threshold = 0.7
        low_confidence_hs = hubspot_df[(hubspot_df['confidence_score'] > 0) & (hubspot_df['confidence_score'] < low_conf_threshold)]
        if not low_confidence_hs.empty:
            report.append(f"\nLow confidence HubSpot matches (<{low_conf_threshold}):")
            for idx, row in low_confidence_hs.head(10).iterrows():
                report.append(f"  - Row {idx}: {row.get('Deal Name', 'N/A')}")
                report.append(f"    Confidence: {row['confidence_score']:.2f}")
                report.append(f"    Reason: {row.get('notes', 'N/A')[:100]}")
        
        report.append("")
        report.append("=" * 80)
        
        return '\n'.join(report)

def main(test_mode=False):
    """
    Main execution function
    
    Args:
        test_mode: If True, only process first 10 records from each dataset
    """
    print("\n" + "="*80)
    print("🏁 MAIN FUNCTION STARTING - CLAUDE-DRIVEN VERSION")
    if test_mode:
        print("⚠️ TEST MODE ENABLED - Will only process 10 records per dataset")
    print("="*80)
    
    # Configuration
    print("🔧 Loading configuration...")
    GOOGLE_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    
    print(f"📍 Google API Key: {'✅ Found' if GOOGLE_API_KEY else '❌ Missing'}")
    print(f"🤖 Anthropic API Key: {'✅ Found' if ANTHROPIC_API_KEY else '❌ Missing'}")
    
    if not GOOGLE_API_KEY or not ANTHROPIC_API_KEY:
        print("❌ ERROR: Missing required API keys")
        logger.error("Please set GOOGLE_PLACES_API_KEY and ANTHROPIC_API_KEY environment variables")
        return
    
    # File paths
    HUBSPOT_FILE = 'hubspot_data.csv'
    DATABASE_FILE = 'database_data.csv'
    
    print(f"📁 HubSpot file: {HUBSPOT_FILE}")
    print(f"📁 Database file: {DATABASE_FILE}")
    
    try:
        # Load data
        print("\n📂 Loading data files...")
        
        print(f"📋 Loading HubSpot data from {HUBSPOT_FILE}...")
        hubspot_df = pd.read_csv(HUBSPOT_FILE)
        print(f"✅ HubSpot data loaded: {len(hubspot_df)} records")
        
        print(f"\n🗄️ Loading Database data from {DATABASE_FILE}...")
        database_df = pd.read_csv(DATABASE_FILE)
        print(f"✅ Database data loaded: {len(database_df)} records")
        
        # Initialize agent
        print("\n🤖 Initializing DataTriageAgent...")
        agent = DataTriageAgent(GOOGLE_API_KEY, ANTHROPIC_API_KEY)
        
        # Generate timestamp for this run
        output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"⏰ Output timestamp: {output_timestamp}")
        
        # Process datasets
        print("\n🚀 Starting data triage process...")
        processed_hubspot, processed_database = agent.process_datasets(
            hubspot_df, database_df, output_timestamp, test_mode=test_mode
        )
        
        # Save final results
        print("\n💾 Saving final results...")
        
        hubspot_output = f'hubspot_processed_{output_timestamp}.csv'
        database_output = f'database_processed_{output_timestamp}.csv'
        
        # Remove tracking columns from final output
        final_hubspot = processed_hubspot.drop('processed', axis=1) if 'processed' in processed_hubspot.columns else processed_hubspot
        final_database = processed_database.drop('processed', axis=1) if 'processed' in processed_database.columns else processed_database
        
        final_hubspot.to_csv(hubspot_output, index=False)
        final_database.to_csv(database_output, index=False)
        
        print(f"✅ Final results saved to {hubspot_output} and {database_output}")
        
        # Generate and save summary report
        print(f"\n📋 Generating summary report...")
        summary = agent.generate_summary_report(processed_hubspot, processed_database)
        
        report_file = f'triage_report_{output_timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(summary)
        
        print(f"✅ Report saved to {report_file}")
        
        print("\n" + "="*80)
        print("🎉 SCRIPT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(summary)
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find input file: {e}")
        logger.error(f"Could not find input file: {e}")
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred: {e}")
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    print("🎬 Script started from command line")
    
    # SET TEST MODE HERE
    # Change to False when ready to process all records
    TEST_MODE = True
    
    if TEST_MODE:
        print("⚠️ TEST MODE IS ON - Processing only 10 records per dataset")
        print("To process all records, set TEST_MODE = False in the script")
    
    main(test_mode=TEST_MODE)
    print("🏁 Script execution finished")
