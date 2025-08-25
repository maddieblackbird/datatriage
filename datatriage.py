#!/usr/bin/env python3
"""
Blackbird Data Triage Script
Reconciles restaurant data between backend database and HubSpot using Google Places API
and Claude AI for intelligent matching and verification.
"""

import pandas as pd
import googlemaps
import anthropic
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
from dataclasses import dataclass
import re

print("=" * 80)
print("🚀 BLACKBIRD DATA TRIAGE SCRIPT STARTING")
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
            
        self.processed_records = []
        self.match_results = []
        print("✅ DataTriageAgent initialization complete")
        
    def should_skip_record(self, record: RestaurantRecord) -> bool:
        """Check if a record should be skipped based on keywords"""
        print(f"   🔍 Checking if record should be skipped...")
        check_fields = [
            record.deal_name, record.company_name, 
            record.restaurant_name, record.location_name
        ]
        
        for field in check_fields:
            if field:
                field_lower = field.lower()
                print(f"      📝 Checking field: '{field}'")
                for keyword in self.SKIP_KEYWORDS:
                    if keyword in field_lower:
                        print(f"      ⚠️ SKIP: Found keyword '{keyword}' in '{field}'")
                        logger.info(f"Skipping record with keyword '{keyword}': {field}")
                        return True
        print(f"   ✅ Record passed skip check")
        return False
    
    def build_search_query(self, record: RestaurantRecord) -> str:
        """Build a search query for Google Places API"""
        print(f"   🔨 Building search query for {record.source} record...")
        query_parts = []
        
        def safe_add_field(field_value, field_name):
            """Safely add a field to query_parts, converting to string if needed"""
            if field_value is not None:
                if isinstance(field_value, float) and pd.isna(field_value):
                    return  # Skip NaN values
                try:
                    str_value = str(field_value).strip()
                    if str_value and str_value.lower() != 'nan':
                        print(f"      ✅ Adding {field_name}: '{str_value}'")
                        query_parts.append(str_value)
                    else:
                        print(f"      ⚠️ Skipping empty {field_name}")
                except Exception as e:
                    print(f"      ❌ Error converting {field_name} to string: {e}")
        
        # For HubSpot, prioritize deal name
        if record.source == 'hubspot' and record.deal_name:
            safe_add_field(record.deal_name, "deal name")
        elif record.restaurant_name:
            safe_add_field(record.restaurant_name, "restaurant name")
        elif record.company_name:
            safe_add_field(record.company_name, "company name")
        
        # Add location information
        if record.address:
            safe_add_field(record.address, "full address")
        else:
            print(f"      📍 Building address from components...")
            safe_add_field(record.street, "street")
            safe_add_field(record.city, "city")
            safe_add_field(record.state, "state")
            safe_add_field(record.zipcode, "zipcode")
        
        if not query_parts:
            print(f"   ⚠️ No valid query parts found")
            return "restaurant"  # Fallback query
            
        query = ', '.join(query_parts)
        print(f"   ✅ Built query: '{query}'")
        return query
    
    def search_google_places(self, query: str, existing_place_id: Optional[str] = None) -> Dict:
        """
        Search for a place using Google Places API
        
        Args:
            query: Search query string
            existing_place_id: Existing place ID to verify
            
        Returns:
            Dictionary with place information
        """
        print(f"   🌐 Searching Google Places...")
        print(f"      🔍 Query: '{query}'")
        print(f"      🆔 Existing Place ID: {existing_place_id or 'None'}")
        
        try:
            # First, try to verify existing place ID if provided
            if existing_place_id:
                print(f"      ✅ Verifying existing place ID: {existing_place_id}")
                try:
                    place_result = self.gmaps.place(
                        place_id=existing_place_id,
                        fields=['name', 'formatted_address', 'place_id', 'types', 'business_status']
                    )
                    if place_result.get('result'):
                        print(f"      ✅ Existing place ID verified successfully")
                        logger.info(f"Verified existing place ID: {existing_place_id}")
                        return place_result['result']
                except Exception as e:
                    print(f"      ⚠️ Could not verify existing place ID: {e}")
                    logger.warning(f"Could not verify place ID {existing_place_id}: {e}")
            
            # Search for the place
            print(f"      🔍 Performing new search...")
            search_results = self.gmaps.places(query=query)
            
            if search_results['results']:
                print(f"      ✅ Found {len(search_results['results'])} results")
                # Get the first result (most relevant)
                place = search_results['results'][0]
                print(f"      📍 Top result: '{place.get('name', 'N/A')}'")
                
                # Get detailed information
                print(f"      📋 Getting place details...")
                place_details = self.gmaps.place(
                    place_id=place['place_id'],
                    fields=['name', 'formatted_address', 'place_id', 'types', 
                           'business_status', 'geometry', 'website', 'phone_number']
                )
                
                result = place_details.get('result', place)
                print(f"      ✅ Place details retrieved")
                return result
            else:
                print(f"      ❌ No results found")
                return None
            
        except Exception as e:
            print(f"      ❌ Error searching Google Places: {e}")
            logger.error(f"Error searching Google Places for '{query}': {e}")
            return None
    
    def verify_with_claude(self, record: RestaurantRecord, google_result: Optional[Dict]) -> Dict:
        """
        Use Claude to verify and analyze the match
        
        Args:
            record: Restaurant record
            google_result: Result from Google Places API
            
        Returns:
            Dictionary with Claude's analysis
        """
        print(f"   🤖 Verifying with Claude AI...")
        
        prompt = f"""
        You are helping to verify restaurant data matching. Please analyze the following:
        
        ORIGINAL RECORD ({record.source}):
        - Name: {record.deal_name or record.restaurant_name or record.company_name}
        - Location Name: {record.location_name}
        - Address: {record.address or f"{record.street}, {record.city}, {record.state} {record.zipcode}"}
        - Existing Google Places ID: {record.google_places_id}
        - Macro Geo: {record.macro_geo}
        
        GOOGLE PLACES RESULT:
        {json.dumps(google_result, indent=2) if google_result else "No result found"}
        
        Please provide your analysis in JSON format with the following fields:
        1. "is_valid_match": boolean - Is this a valid restaurant match?
        2. "confidence_score": float (0-1) - How confident are you in this match?
        3. "correct_place_id": string - The correct Google Places ID (or null if not found)
        4. "reasoning": string - Brief explanation of your decision
        5. "is_real_restaurant": boolean - Is this an actual restaurant (not a test/demo)?
        6. "notes": string - Any additional notes or concerns
        
        Consider:
        - Is the address/location consistent?
        - Is this actually a restaurant/food establishment?
        - Are there any red flags or mismatches?
        - For HubSpot records, prioritize the Deal Name for matching
        """
        
        try:
            print(f"      📤 Sending request to Claude...")
            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",  # Claude Sonnet 4.0 - superior intelligence and reasoning
                max_tokens=1000,
                temperature=0,  # Use 0 for most deterministic responses
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            print(f"      📥 Received response from Claude")
            
            # Extract JSON from Claude's response
            response_text = response.content[0].text
            print(f"      🔍 Parsing Claude's response...")
            
            # Try to parse JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"      ✅ Successfully parsed JSON response")
                print(f"      📊 Confidence: {result.get('confidence_score', 0):.2f}")
                print(f"      🎯 Valid Match: {result.get('is_valid_match', False)}")
                return result
            else:
                print(f"      ❌ Could not parse JSON from Claude's response")
                logger.warning("Could not parse JSON from Claude's response")
                return {
                    "is_valid_match": False,
                    "confidence_score": 0,
                    "correct_place_id": None,
                    "reasoning": "Could not parse response",
                    "is_real_restaurant": False,
                    "notes": response_text
                }
                
        except Exception as e:
            print(f"      ❌ Error getting Claude verification: {e}")
            logger.error(f"Error getting Claude verification: {e}")
            return {
                "is_valid_match": False,
                "confidence_score": 0,
                "correct_place_id": None,
                "reasoning": f"Error: {str(e)}",
                "is_real_restaurant": False,
                "notes": ""
            }
    
    def find_matching_record(self, record: RestaurantRecord, other_records: List[RestaurantRecord]) -> Optional[RestaurantRecord]:
        """
        Find a matching record in the other dataset based on Google Places ID
        
        Args:
            record: Record to match
            other_records: List of records from the other source
            
        Returns:
            Matching record or None
        """
        print(f"   🔍 Looking for matching record...")
        print(f"      🆔 Searching for Place ID: {record.google_places_id}")
        
        if not record.google_places_id:
            print(f"      ❌ No place ID to match with")
            return None
        
        matches_found = 0
        for other in other_records:
            if other.google_places_id == record.google_places_id:
                matches_found += 1
                print(f"      ✅ Found match at row {other.row_index}")
                return other
        
        print(f"      ❌ No matching record found (checked {len(other_records)} records)")
        return None
    
    def process_datasets(self, hubspot_df: pd.DataFrame, database_df: pd.DataFrame, 
                        output_timestamp: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process both datasets and perform matching
        
        Args:
            hubspot_df: HubSpot DataFrame
            database_df: Database DataFrame
            output_timestamp: Timestamp for output files
            
        Returns:
            Tuple of processed DataFrames with new columns
        """
        print("\n" + "="*80)
        print("📊 PROCESSING DATASETS")
        print("="*80)
        print(f"📋 HubSpot records: {len(hubspot_df)}")
        print(f"🗄️ Database records: {len(database_df)}")
        
        # Set up incremental save file names
        hubspot_temp_file = f'hubspot_progress_{output_timestamp}.csv'
        database_temp_file = f'database_progress_{output_timestamp}.csv'
        
        print(f"💾 Incremental save files:")
        print(f"   📋 HubSpot: {hubspot_temp_file}")
        print(f"   🗄️ Database: {database_temp_file}")
        
        # Add processing columns
        print("📝 Adding processing columns...")
        hubspot_df['verified_place_id'] = ''
        hubspot_df['confidence_score'] = 0.0
        hubspot_df['match_status'] = ''
        hubspot_df['notes'] = ''
        hubspot_df['processed'] = False  # Track which records are done
        
        database_df['verified_place_id'] = ''
        database_df['confidence_score'] = 0.0
        database_df['match_status'] = ''
        database_df['notes'] = ''
        database_df['processed'] = False  # Track which records are done
        print("✅ Processing columns added")
        
        # Convert DataFrames to RestaurantRecord objects
        print("\n🔄 Converting DataFrames to RestaurantRecord objects...")
        hubspot_records = []
        print("📋 Processing HubSpot records...")
        for idx, row in hubspot_df.iterrows():
            print(f"   📝 Creating record {idx + 1}/{len(hubspot_df)}")
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
        print(f"✅ Created {len(hubspot_records)} HubSpot records")
        
        database_records = []
        print("🗄️ Processing Database records...")
        for idx, row in database_df.iterrows():
            print(f"   📝 Creating record {idx + 1}/{len(database_df)}")
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
        print(f"✅ Created {len(database_records)} Database records")
        
    def process_single_record(self, record: RestaurantRecord, other_records: List[RestaurantRecord], 
                             df: pd.DataFrame, record_type: str) -> bool:
        """
        Process a single record with comprehensive error handling
        
        Args:
            record: Record to process
            other_records: Records from other dataset for matching
            df: DataFrame to update
            record_type: "HubSpot" or "Database" for logging
            
        Returns:
            True if processed successfully, False if error occurred
        """
        try:
            print(f"🔧 Processing {record_type} record...")
            
            # Check if should skip
            if self.should_skip_record(record):
                print(f"   ⏭️ SKIPPING this record")
                df.at[record.row_index, 'match_status'] = 'SKIPPED'
                df.at[record.row_index, 'notes'] = 'Contains skip keyword'
                df.at[record.row_index, 'processed'] = True
                return True
            
            # Build search query with error handling
            try:
                query = self.build_search_query(record)
            except Exception as e:
                print(f"   ❌ Error building search query: {e}")
                df.at[record.row_index, 'match_status'] = 'ERROR'
                df.at[record.row_index, 'notes'] = f'Error building query: {str(e)}'
                df.at[record.row_index, 'confidence_score'] = 0
                df.at[record.row_index, 'processed'] = True
                logger.error(f"Error building query for row {record.row_index}: {e}")
                return False
            
            # Search Google Places with error handling
            try:
                print(f"🌐 Searching Google Places...")
                google_result = self.search_google_places(query, record.google_places_id)
            except Exception as e:
                print(f"   ❌ Error searching Google Places: {e}")
                df.at[record.row_index, 'match_status'] = 'ERROR'
                df.at[record.row_index, 'notes'] = f'Error searching Google Places: {str(e)}'
                df.at[record.row_index, 'confidence_score'] = 0
                df.at[record.row_index, 'processed'] = True
                logger.error(f"Error searching Google Places for row {record.row_index}: {e}")
                return False
            
            # Verify with Claude with error handling
            try:
                print(f"🤖 Verifying with Claude AI...")
                verification = self.verify_with_claude(record, google_result)
            except Exception as e:
                print(f"   ❌ Error verifying with Claude: {e}")
                df.at[record.row_index, 'match_status'] = 'ERROR'
                df.at[record.row_index, 'notes'] = f'Error verifying with Claude: {str(e)}'
                df.at[record.row_index, 'confidence_score'] = 0
                df.at[record.row_index, 'processed'] = True
                logger.error(f"Error verifying with Claude for row {record.row_index}: {e}")
                return False
            
            # Update DataFrame with results
            try:
                print(f"💾 Updating DataFrame...")
                df.at[record.row_index, 'verified_place_id'] = verification.get('correct_place_id', '')
                df.at[record.row_index, 'confidence_score'] = verification.get('confidence_score', 0)
                
                # Check for match in other dataset
                print(f"🔍 Checking for match in other dataset...")
                if verification.get('correct_place_id'):
                    record.google_places_id = verification['correct_place_id']
                    match = self.find_matching_record(record, other_records)
                    if match:
                        other_dataset = "DB" if record_type == "HubSpot" else "HS"
                        print(f"   ✅ MATCHED with {other_dataset} row {match.row_index}")
                        df.at[record.row_index, 'match_status'] = f'MATCHED ({other_dataset} row {match.row_index})'
                    else:
                        other_dataset_name = "database" if record_type == "HubSpot" else "HubSpot"
                        print(f"   ❌ No match found in {other_dataset_name}")
                        df.at[record.row_index, 'match_status'] = f'NO_MATCH_IN_{other_dataset_name.upper()}'
                else:
                    print(f"   ❌ Place ID not found")
                    df.at[record.row_index, 'match_status'] = 'PLACE_ID_NOT_FOUND'
                
                df.at[record.row_index, 'notes'] = f"{verification.get('reasoning', '')} | {verification.get('notes', '')}"
                df.at[record.row_index, 'processed'] = True
                
                return True
                
            except Exception as e:
                print(f"   ❌ Error updating DataFrame: {e}")
                df.at[record.row_index, 'match_status'] = 'ERROR'
                df.at[record.row_index, 'notes'] = f'Error updating results: {str(e)}'
                df.at[record.row_index, 'confidence_score'] = 0
                df.at[record.row_index, 'processed'] = True
                logger.error(f"Error updating DataFrame for row {record.row_index}: {e}")
                return False
                
        except Exception as e:
            print(f"   ❌ Unexpected error processing record: {e}")
            try:
                df.at[record.row_index, 'match_status'] = 'ERROR'
                df.at[record.row_index, 'notes'] = f'Unexpected error: {str(e)}'
                df.at[record.row_index, 'confidence_score'] = 0
                df.at[record.row_index, 'processed'] = True
            except:
                pass  # If we can't even update the DataFrame, just log and continue
            logger.error(f"Unexpected error processing row {record.row_index}: {e}")
            return False
        print("\n" + "="*60)
        print("📋 PROCESSING HUBSPOT RECORDS")
        print("="*60)
        logger.info("Processing HubSpot records...")
        
        hubspot_processed = 0
        hubspot_skipped = 0
        hubspot_matched = 0
        
        # Save initial state with all records
        print(f"💾 Saving initial HubSpot state...")
        hubspot_df.to_csv(hubspot_temp_file, index=False)
        print(f"✅ Initial state saved to {hubspot_temp_file}")
        
        for i, record in enumerate(hubspot_records):
            print(f"\n📋 Processing HubSpot record {i + 1}/{len(hubspot_records)}")
            print(f"   📝 Deal Name: {record.deal_name}")
            print(f"   🏢 Company: {record.company_name}")
            print(f"   📍 Address: {record.address}")
            
            if self.should_skip_record(record):
                print(f"   ⏭️ SKIPPING this record")
                hubspot_df.at[record.row_index, 'match_status'] = 'SKIPPED'
                hubspot_df.at[record.row_index, 'notes'] = 'Contains skip keyword'
                hubspot_df.at[record.row_index, 'processed'] = True
                hubspot_skipped += 1
                
                # Save progress after each record
                print(f"   💾 Saving progress...")
                hubspot_df.to_csv(hubspot_temp_file, index=False)
                continue
            
            # Search Google Places
            query = self.build_search_query(record)
            print(f"🌐 Searching Google Places...")
            logger.info(f"Searching for: {query}")
            google_result = self.search_google_places(query, record.google_places_id)
            
            # Verify with Claude
            print(f"🤖 Verifying with Claude AI...")
            verification = self.verify_with_claude(record, google_result)
            
            # Update DataFrame
            print(f"💾 Updating DataFrame...")
            hubspot_df.at[record.row_index, 'verified_place_id'] = verification.get('correct_place_id', '')
            hubspot_df.at[record.row_index, 'confidence_score'] = verification.get('confidence_score', 0)
            
            # Check for match in database
            print(f"🔍 Checking for database match...")
            if verification.get('correct_place_id'):
                record.google_places_id = verification['correct_place_id']
                match = self.find_matching_record(record, database_records)
                if match:
                    print(f"   ✅ MATCHED with database row {match.row_index}")
                    hubspot_df.at[record.row_index, 'match_status'] = f'MATCHED (DB row {match.row_index})'
                    hubspot_matched += 1
                else:
                    print(f"   ❌ No match found in database")
                    hubspot_df.at[record.row_index, 'match_status'] = 'NO_MATCH_IN_DB'
            else:
                print(f"   ❌ Place ID not found")
                hubspot_df.at[record.row_index, 'match_status'] = 'PLACE_ID_NOT_FOUND'
            
            hubspot_df.at[record.row_index, 'notes'] = f"{verification.get('reasoning', '')} | {verification.get('notes', '')}"
            hubspot_df.at[record.row_index, 'processed'] = True
            hubspot_processed += 1
            
            # Save progress after each record
            print(f"💾 Saving progress to {hubspot_temp_file}...")
            hubspot_df.to_csv(hubspot_temp_file, index=False)
            
            # Rate limiting
            print(f"⏳ Rate limiting pause (0.5 seconds)...")
            time.sleep(0.5)
            
            print(f"✅ HubSpot record {i + 1} complete and saved")
        
        print(f"\n📊 HubSpot Processing Summary:")
        print(f"   ✅ Processed: {hubspot_processed}")
        print(f"   ⏭️ Skipped: {hubspot_skipped}")
        print(f"   🎯 Matched: {hubspot_matched}")
        print(f"   💾 All progress saved to: {hubspot_temp_file}")
        
        # Process Database records
        print("\n" + "="*60)
        print("🗄️ PROCESSING DATABASE RECORDS")
        print("="*60)
        logger.info("Processing Database records...")
        
        database_processed = 0
        database_skipped = 0
        database_matched = 0
        
        # Save initial state with all records
        print(f"💾 Saving initial Database state...")
        database_df.to_csv(database_temp_file, index=False)
        print(f"✅ Initial state saved to {database_temp_file}")
        
        for i, record in enumerate(database_records):
            print(f"\n🗄️ Processing Database record {i + 1}/{len(database_records)}")
            print(f"   🍽️ Restaurant Name: {record.restaurant_name}")
            print(f"   📍 Location: {record.location_name}")
            print(f"   🏙️ City: {record.city}, {record.state}")
            
            if self.should_skip_record(record):
                print(f"   ⏭️ SKIPPING this record")
                database_df.at[record.row_index, 'match_status'] = 'SKIPPED'
                database_df.at[record.row_index, 'notes'] = 'Contains skip keyword'
                database_df.at[record.row_index, 'processed'] = True
                database_skipped += 1
                
                # Save progress after each record
                print(f"   💾 Saving progress...")
                database_df.to_csv(database_temp_file, index=False)
                continue
            
            # Search Google Places
            query = self.build_search_query(record)
            print(f"🌐 Searching Google Places...")
            logger.info(f"Searching for: {query}")
            google_result = self.search_google_places(query, record.google_places_id)
            
            # Verify with Claude
            print(f"🤖 Verifying with Claude AI...")
            verification = self.verify_with_claude(record, google_result)
            
            # Update DataFrame
            print(f"💾 Updating DataFrame...")
            database_df.at[record.row_index, 'verified_place_id'] = verification.get('correct_place_id', '')
            database_df.at[record.row_index, 'confidence_score'] = verification.get('confidence_score', 0)
            
            # Check for match in HubSpot
            print(f"🔍 Checking for HubSpot match...")
            if verification.get('correct_place_id'):
                record.google_places_id = verification['correct_place_id']
                match = self.find_matching_record(record, hubspot_records)
                if match:
                    print(f"   ✅ MATCHED with HubSpot row {match.row_index}")
                    database_df.at[record.row_index, 'match_status'] = f'MATCHED (HS row {match.row_index})'
                    database_matched += 1
                else:
                    print(f"   ❌ No match found in HubSpot")
                    database_df.at[record.row_index, 'match_status'] = 'NO_MATCH_IN_HUBSPOT'
            else:
                print(f"   ❌ Place ID not found")
                database_df.at[record.row_index, 'match_status'] = 'PLACE_ID_NOT_FOUND'
            
            database_df.at[record.row_index, 'notes'] = f"{verification.get('reasoning', '')} | {verification.get('notes', '')}"
            database_df.at[record.row_index, 'processed'] = True
            database_processed += 1
            
            # Save progress after each record
            print(f"💾 Saving progress to {database_temp_file}...")
            database_df.to_csv(database_temp_file, index=False)
            
            # Rate limiting
            print(f"⏳ Rate limiting pause (0.5 seconds)...")
            time.sleep(0.5)
            
            print(f"✅ Database record {i + 1} complete and saved")
        
        print(f"\n📊 Database Processing Summary:")
        print(f"   ✅ Processed: {database_processed}")
        print(f"   ⏭️ Skipped: {database_skipped}")
        print(f"   🎯 Matched: {database_matched}")
        print(f"   💾 All progress saved to: {database_temp_file}")
        
        print(f"\n🎉 ALL PROCESSING COMPLETE!")
        print(f"💾 Progress files available:")
        print(f"   📋 HubSpot: {hubspot_temp_file}")
        print(f"   🗄️ Database: {database_temp_file}")
        
        return hubspot_df, database_df
    
    def generate_summary_report(self, hubspot_df: pd.DataFrame, database_df: pd.DataFrame) -> str:
        """Generate a summary report of the matching process"""
        print("\n📋 Generating summary report...")
        report = []
        report.append("=" * 80)
        report.append("DATA TRIAGE SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # HubSpot summary
        report.append("HUBSPOT DATA SUMMARY:")
        report.append(f"Total records: {len(hubspot_df)}")
        report.append(f"Matched records: {hubspot_df['match_status'].str.contains('MATCHED').sum()}")
        report.append(f"No match in DB: {(hubspot_df['match_status'] == 'NO_MATCH_IN_DB').sum()}")
        report.append(f"Place ID not found: {(hubspot_df['match_status'] == 'PLACE_ID_NOT_FOUND').sum()}")
        report.append(f"Skipped: {(hubspot_df['match_status'] == 'SKIPPED').sum()}")
        report.append(f"Average confidence: {hubspot_df['confidence_score'].mean():.2f}")
        report.append("")
        
        # Database summary
        report.append("DATABASE SUMMARY:")
        report.append(f"Total records: {len(database_df)}")
        report.append(f"Matched records: {database_df['match_status'].str.contains('MATCHED').sum()}")
        report.append(f"No match in HubSpot: {(database_df['match_status'] == 'NO_MATCH_IN_HUBSPOT').sum()}")
        report.append(f"Place ID not found: {(database_df['match_status'] == 'PLACE_ID_NOT_FOUND').sum()}")
        report.append(f"Skipped: {(database_df['match_status'] == 'SKIPPED').sum()}")
        report.append(f"Average confidence: {database_df['confidence_score'].mean():.2f}")
        report.append("")
        
        # Issues requiring attention
        report.append("ISSUES REQUIRING ATTENTION:")
        
        # Low confidence matches
        low_confidence_hs = hubspot_df[hubspot_df['confidence_score'] < 0.7]
        if not low_confidence_hs.empty:
            report.append(f"\nLow confidence HubSpot matches (<0.7): {len(low_confidence_hs)}")
            for idx, row in low_confidence_hs.iterrows():
                report.append(f"  - Row {idx}: {row.get('Deal Name', 'N/A')} (confidence: {row['confidence_score']:.2f})")
        
        low_confidence_db = database_df[database_df['confidence_score'] < 0.7]
        if not low_confidence_db.empty:
            report.append(f"\nLow confidence Database matches (<0.7): {len(low_confidence_db)}")
            for idx, row in low_confidence_db.iterrows():
                report.append(f"  - Row {idx}: {row.get('restaurant_name', 'N/A')} (confidence: {row['confidence_score']:.2f})")
        
        report.append("")
        report.append("=" * 80)
        
        print("✅ Summary report generated")
        return '\n'.join(report)

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("🏁 MAIN FUNCTION STARTING")
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
    HUBSPOT_FILE = 'hubspot_data.csv'  # Update with your file path
    DATABASE_FILE = 'database_data.csv'  # Update with your file path
    
    print(f"📁 HubSpot file: {HUBSPOT_FILE}")
    print(f"📁 Database file: {DATABASE_FILE}")
    
    try:
        # Load data
        print("\n📂 Loading data files...")
        
        print(f"📋 Loading HubSpot data from {HUBSPOT_FILE}...")
        logger.info(f"Loading HubSpot data from {HUBSPOT_FILE}")
        hubspot_df = pd.read_csv(HUBSPOT_FILE)
        print(f"✅ HubSpot data loaded: {len(hubspot_df)} records")
        print(f"   📊 Columns: {list(hubspot_df.columns)}")
        
        print(f"\n🗄️ Loading Database data from {DATABASE_FILE}...")
        logger.info(f"Loading Database data from {DATABASE_FILE}")
        database_df = pd.read_csv(DATABASE_FILE)
        print(f"✅ Database data loaded: {len(database_df)} records")
        print(f"   📊 Columns: {list(database_df.columns)}")
        
        # Initialize agent
        print("\n🤖 Initializing DataTriageAgent...")
        agent = DataTriageAgent(GOOGLE_API_KEY, ANTHROPIC_API_KEY)
        
        # Generate timestamp for this run
        output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"⏰ Output timestamp: {output_timestamp}")
        
        # Process datasets
        print("\n🚀 Starting data triage process...")
        logger.info("Starting data triage process...")
        processed_hubspot, processed_database = agent.process_datasets(hubspot_df, database_df, output_timestamp)
        
        # Save final results
        print("\n💾 Saving final results...")
        output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"⏰ Final output timestamp: {output_timestamp}")
        
        hubspot_output = f'hubspot_processed_{output_timestamp}.csv'
        database_output = f'database_processed_{output_timestamp}.csv'
        
        print(f"💾 Saving final HubSpot results to: {hubspot_output}")
        # Remove the 'processed' tracking column from final output
        final_hubspot = processed_hubspot.drop('processed', axis=1) if 'processed' in processed_hubspot.columns else processed_hubspot
        final_hubspot.to_csv(hubspot_output, index=False)
        print(f"✅ Final HubSpot results saved")
        
        print(f"💾 Saving final Database results to: {database_output}")
        # Remove the 'processed' tracking column from final output
        final_database = processed_database.drop('processed', axis=1) if 'processed' in processed_database.columns else processed_database
        final_database.to_csv(database_output, index=False)
        print(f"✅ Final Database results saved")
        
        logger.info(f"Saved processed HubSpot data to {hubspot_output}")
        logger.info(f"Saved processed Database data to {database_output}")
        
        # Generate and save summary report
        print(f"\n📋 Generating summary report...")
        summary = agent.generate_summary_report(processed_hubspot, processed_database)
        
        report_file = f'triage_report_{output_timestamp}.txt'
        print(f"💾 Saving report to: {report_file}")
        with open(report_file, 'w') as f:
            f.write(summary)
        
        print(f"✅ Report saved")
        logger.info(f"Saved summary report to {report_file}")
        
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
    main()
    print("🏁 Script execution finished")
