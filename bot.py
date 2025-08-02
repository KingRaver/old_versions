#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# CORE IMPORTS - REQUIRED
# ============================================================================
from typing import Dict, List, Optional, Any, Union, Tuple
import sys
import os
import time
import requests
import re
import numpy as np
from datetime import datetime, timedelta, timezone
from datetime_utils import strip_timezone, ensure_naive_datetimes, safe_datetime_diff
import anthropic
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
import random
import statistics
import threading
import queue
import json
import traceback

# ============================================================================
# M4 OPTIMIZATION IMPORTS - WITH GRACEFUL FALLBACKS
# ============================================================================

# Global flags for optimization availability
POLARS_AVAILABLE = False
NUMBA_AVAILABLE = False
AIOHTTP_AVAILABLE = False
ASYNCIO_AVAILABLE = False

# Try Polars for ultra-fast data processing
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None

# Try Numba for JIT compilation
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    # Dummy decorators for compatibility
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    
    def prange(*args, **kwargs):
        return range(*args)

# Try aiohttp for async HTTP
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None

# Try asyncio for async operations
try:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    ASYNCIO_AVAILABLE = True
except ImportError:
    asyncio = None
    ThreadPoolExecutor = None

# ============================================================================
# BOT-SPECIFIC IMPORTS
# ============================================================================
from llm_provider import LLMProvider
from database import CryptoDatabase

from utils.logger import logger
from utils.browser import browser
from config import config
from coingecko_handler import CoinGeckoHandler
from mood_config import MoodIndicators, determine_advanced_mood, Mood, MemePhraseGenerator
from meme_phrases import MEME_PHRASES
from prediction_engine import EnhancedPredictionEngine, MachineLearningModels, StatisticalModels
from api_manager import create_api_manager

# Import modules for reply functionality
from timeline_scraper import TimelineScraper
from reply_handler import ReplyHandler
from content_analyzer import ContentAnalyzer

class CryptoAnalysisBot:
    """
    Enhanced crypto analysis bot with market and tech content capabilities.
    Handles market analysis, predictions, and social media engagement.
    """

    def __init__(self, database, llm_provider, config=None):
        self.browser = browser
        self.db = database
        if config is None:
            from config import config as imported_config
            self.config = imported_config
        else:
            self.config = config
        """
        Initialize the crypto analysis bot with improved configuration and tracking.
        Sets up connections to browser, database, and API services.
        """
        self.browser = browser
        self.llm_provider = LLMProvider(self.config)  
        self.past_predictions = []
        self.meme_phrases = MEME_PHRASES
        self.market_conditions = {}
        self.last_check_time = strip_timezone(datetime.now())
        self.last_market_data = {}
        self.last_reply_time = strip_timezone(datetime.now())
        
        # Multi-timeframe prediction tracking
        self.timeframes = ["1h", "24h", "7d"]
        self.timeframe_predictions = {tf: {} for tf in self.timeframes}
        self.timeframe_last_post = {tf: strip_timezone(datetime.now() - timedelta(hours=3)) for tf in self.timeframes}
       
        # Timeframe posting frequency controls (in hours)
        self.timeframe_posting_frequency = {
            "1h": 1,    # Every hour
            "24h": 6,   # Every 6 hours
            "7d": 24    # Once per day
        }
       
        # Prediction accuracy tracking by timeframe
        self.prediction_accuracy = {tf: {'correct': 0, 'total': 0} for tf in self.timeframes}
       
        # Initialize prediction engine with database and LLM Provider
        self.prediction_engine = EnhancedPredictionEngine(
            database=self.db,
            llm_provider=self.llm_provider,
            bot=self
        )
       
        # Create a queue for predictions to process
        self.prediction_queue = queue.Queue()
       
        # Initialize thread for async prediction generation
        self.prediction_thread = None
        self.prediction_thread_running = False
        
        # Target chains to analyze - now dynamically sourced from database
        try:
            database_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=25)
            if not database_tokens:
                logger.logger.error("❌ CRITICAL: Database returned no tokens - shutting down bot")
                raise SystemExit("Database connection lost - no target chains available")
            
            # Build target_chains dynamically from database results using TokenMappingManager
            self.target_chains = {}
            for db_token in database_tokens:
                # Convert database name to short symbol (BITCOIN -> BTC)
                short_symbol = self.config.token_mapper.database_name_to_symbol(db_token)
                
                # Convert short symbol to CoinGecko ID (BTC -> bitcoin)
                coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(short_symbol)
                
                # Add to target_chains
                self.target_chains[short_symbol] = coingecko_id

            # Also update reference_tokens to use the short symbols
            self.reference_tokens = list(self.target_chains.keys())

            logger.logger.info(f"✅ Built target_chains for {len(self.target_chains)} tokens using TokenMappingManager")
            logger.logger.debug(f"🎯 Target chains: {list(self.target_chains.keys())[:10]}...")  # Log first 10
            
        except Exception as e:
            logger.logger.error(f"❌ CRITICAL: Database access failed - shutting down bot: {str(e)}")
            raise SystemExit(f"Database connection lost: {str(e)}")

        # All tokens for reference and comparison - now database-driven
        self.reference_tokens = list(self.target_chains.keys())
       
        # Chain name mapping for display
        self.chain_name_mapping = self.target_chains.copy()
       
        self.CORRELATION_THRESHOLD = 0.75  
        self.VOLUME_THRESHOLD = 0.60  
        self.TIME_WINDOW = 24
       
        # Smart money thresholds
        self.SMART_MONEY_VOLUME_THRESHOLD = 1.5  # 50% above average
        self.SMART_MONEY_ZSCORE_THRESHOLD = 2.0  # 2 standard deviations
       
        # Timeframe-specific triggers and thresholds
        self.timeframe_thresholds = {
            "1h": {
                "price_change": 3.0,    # 3% price change for 1h predictions
                "volume_change": 8.0,   # 8% volume change
                "confidence": 70,       # Minimum confidence percentage
                "fomo_factor": 1.0      # FOMO enhancement factor
            },
            "24h": {
                "price_change": 5.0,    # 5% price change for 24h predictions
                "volume_change": 12.0,  # 12% volume change
                "confidence": 65,       # Slightly lower confidence for longer timeframe
                "fomo_factor": 1.2      # Higher FOMO factor
            },
            "7d": {
                "price_change": 8.0,    # 8% price change for 7d predictions
                "volume_change": 15.0,  # 15% volume change
                "confidence": 60,       # Even lower confidence for weekly predictions
                "fomo_factor": 1.5      # Highest FOMO factor
            }
        }
       
        # Initialize scheduled timeframe posts
        self.next_scheduled_posts = {
            "1h": strip_timezone(datetime.now() + timedelta(minutes=random.randint(10, 30))),
            "24h": strip_timezone(datetime.now() + timedelta(hours=random.randint(1, 3))),
            "7d": strip_timezone(datetime.now() + timedelta(hours=random.randint(4, 8)))
        }
       
        # Initialize reply functionality components with CORRECT database references
        self.timeline_scraper = TimelineScraper(self.browser, self.config, self.db)
        self.reply_handler = ReplyHandler(self.browser, self.config, self.llm_provider, self.db)  # ← FIXED: self.db instead of self.config.db
        self.content_analyzer = ContentAnalyzer(self.config, self.db)

        # Verify database is properly initialized
        if not hasattr(self, 'db') or self.db is None:
            logger.logger.error("❌ CRITICAL: Database not properly initialized before reply components")
            raise Exception("Database must be initialized before reply functionality")

        logger.logger.info("✅ Reply functionality components initialized with proper database reference")
        self.content_analyzer = ContentAnalyzer(self.config, self.db)
       
        api_system = self.initialize_api_system_robust()
        
        # Set instance variables
        self.api_manager = api_system['api_manager']
        self.available_providers = api_system['available_providers']
        
        logger.info(f"✅ Bot initialized with {len(self.available_providers)} API providers")

        # Reply tracking and control
        self.last_reply_check = strip_timezone(datetime.now() - timedelta(minutes=30))  # Start checking soon
        self.reply_check_interval = 20  # Check for posts to reply to every 20 minutes
        self.max_replies_per_cycle = 10  # Maximum 10 replies per cycle
        self.reply_cooldown = 15  # Minutes between reply cycles
        self.last_reply_time = strip_timezone(datetime.now() - timedelta(minutes=self.reply_cooldown))  # Allow immediate first run

        logger.logger.info("Enhanced Prediction Engine initialized with adaptive architecture") 
        logger.log_startup()

    def initialize_api_system_robust(self) -> Dict[str, Any]:
        """
        ROBUST API SYSTEM INITIALIZATION - NEVER FAILS, ALWAYS WORKS
        
        This replaces the old workaround pattern that set api_manager = None
        and fell back to direct CoinGecko handler.
        
        New approach:
        - API Manager ALWAYS succeeds with at least one provider
        - No more fallback workarounds that bypass dual-API functionality
        - Graceful degradation when only one provider is available
        - Clear logging of what's working and what's not
        """
        
        # ================================================================
        # STEP 1: INITIALIZE API MANAGER - GUARANTEED SUCCESS
        # ================================================================
        try:
            from api_manager import create_api_manager, get_api_manager_diagnostics
            
            # Create API manager (this will NEVER fail now)
            api_manager = create_api_manager()
            
            # Get available providers
            available_providers = [name for name, status in api_manager.provider_status.items() 
                                if status['available']]
            
            logger.info(f"🚀 API Manager initialized with {len(available_providers)} providers: {available_providers}")
            
            # ================================================================
            # STEP 2: SET UP BACKWARD COMPATIBILITY REFERENCES  
            # ================================================================
            # Keep these for backward compatibility with existing code
            coingecko_handler = api_manager.providers.get('coingecko')
            coinmarketcap_handler = api_manager.providers.get('coinmarketcap')
            
            # ================================================================
            # STEP 3: PROVIDE SYSTEM STATUS AND RECOMMENDATIONS
            # ================================================================
            if len(available_providers) == 1:
                logger.warning(f"⚠️ Running with single provider: {available_providers[0]}")
                if 'coinmarketcap' not in available_providers:
                    logger.info("💡 Consider adding CoinMarketCap API key for dual-provider redundancy")
                    # Show diagnostics to help with setup
                    diagnostics = get_api_manager_diagnostics()
                    if not diagnostics['provider_requirements']['coinmarketcap']['api_key_found']:
                        logger.info("🔍 Set environment variable: COINMARKETCAP_API_KEY=your_api_key")
            else:
                logger.info("✅ Dual-provider setup active - optimal redundancy achieved!")
            
            return {
                'api_manager': api_manager,
                'coingecko': coingecko_handler,  # For backward compatibility
                'coinmarketcap': coinmarketcap_handler,  # For direct access if needed
                'available_providers': available_providers,
                'initialization_successful': True
            }
            
        except Exception as critical_error:
            # This should NEVER happen with the new robust API manager, but just in case
            logger.error(f"❌ CRITICAL: API Manager initialization failed completely: {str(critical_error)}")
            raise RuntimeError(f"Cannot initialize API system: {str(critical_error)}")    
    
    def _ensure_dict_data(self, data):
        """
        Ensure data is dictionary-like and not a list or string
    
        Args:
            data: Data to check
        
        Returns:
            Dictionary version of data or empty dict if conversion not possible
        """
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            # Try to convert list to dict using 'symbol' as key if available
            result = {}
            for item in data:
                if isinstance(item, dict) and 'symbol' in item:
                    symbol = item['symbol'].upper()
                    result[symbol] = item
            return result
        elif isinstance(data, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(data)
                return self._ensure_dict_data(parsed)  # Recursive call to handle parsed result
            except:
                return {}
        else:
            return {}

    def _standardize_market_data(self, market_data):
        """
        Enhanced market data standardization using TokenMappingManager
        Supports 150+ tokens across database, CoinGecko, and CoinMarketCap formats
        No hardcoded mappings - uses centralized TokenMappingManager system
        """
        # Early returns for edge cases
        if market_data is None:
            return {}

        # GOOD: Only returns if already properly standardized
        if isinstance(market_data, dict) and any(isinstance(key, str) for key in market_data.keys()):
            # Check if keys look like standardized symbols (short, uppercase)
            sample_keys = list(market_data.keys())[:3]
            if all(len(str(key)) <= 6 and str(key).isupper() for key in sample_keys):
                return market_data  # Already standardized
            # Otherwise, proceed with standardization

        # Convert list to dictionary
        if isinstance(market_data, list):
            logger.logger.debug(f"Converting {len(market_data)} market data items to standardized format")
            start_time = time.time()

            result = {}

            # Process each item in the list
            for item in market_data:
                if not isinstance(item, dict):
                    continue

                try:
                    # Detect data source format
                    source = self._detect_item_source(item)
                    
                    # Get standardized symbol using TokenMappingManager
                    if source == 'database':
                        # Database format: {'chain': 'BITCOIN', 'price': 69420, ...}
                        chain_value = item.get('chain', '')
                        if not chain_value:
                            continue
                            
                        # Convert database name to symbol (BITCOIN -> BTC)
                        symbol = self.config.token_mapper.database_name_to_symbol(chain_value)
                        
                        # Standardize database record to expected format
                        standardized_item = {
                            'symbol': symbol,
                            'current_price': self._safe_float(item.get('price', 0)),
                            'price_change_percentage_24h': self._safe_float(item.get('price_change_24h', 0)),
                            'price_change_percentage_1h': self._safe_float(item.get('price_change_1h', 0)),
                            'price_change_percentage_7d': self._safe_float(item.get('price_change_7d', 0)),
                            'total_volume': self._safe_float(item.get('volume', 0)),
                            'volume': self._safe_float(item.get('volume', 0)),  # Alias
                            'market_cap': self._safe_float(item.get('market_cap', 0)),
                            'market_cap_rank': item.get('market_cap_rank', 999),
                            'circulating_supply': self._safe_float(item.get('circulating_supply', 0)),
                            'max_supply': self._safe_float(item.get('max_supply', 0)),
                            'ath_change_percentage': self._safe_float(item.get('ath_change_percentage', 0)),
                            'last_updated': item.get('timestamp') or item.get('last_updated'),
                            
                            # Preserve original fields
                            'id': item.get('id'),
                            'chain': chain_value,
                            'timestamp': item.get('timestamp'),
                            'source': 'database'
                        }
                        
                        result[symbol] = standardized_item
                        logger.logger.debug(f"Database: {chain_value} -> {symbol}")

                    elif source == 'coingecko':
                        # CoinGecko format: {'id': 'bitcoin', 'current_price': 69420, ...}
                        coingecko_id = item.get('id', '')
                        if not coingecko_id:
                            continue
                            
                        # Convert CoinGecko ID to symbol (bitcoin -> BTC)
                        symbol = self.config.token_mapper.coingecko_id_to_symbol(coingecko_id)
                        
                        # CoinGecko format is already mostly standardized
                        standardized_item = item.copy()
                        standardized_item['symbol'] = symbol
                        standardized_item['source'] = 'coingecko'
                        
                        # Ensure required fields exist
                        if 'volume' not in standardized_item and 'total_volume' in standardized_item:
                            standardized_item['volume'] = standardized_item['total_volume']
                        
                        result[symbol] = standardized_item
                        logger.logger.debug(f"CoinGecko: {coingecko_id} -> {symbol}")

                    elif source == 'coinmarketcap':
                        # CoinMarketCap format: {'slug': 'bitcoin', 'quote': {'USD': {...}}, ...}
                        cmc_slug = item.get('slug', '')
                        if not cmc_slug:
                            continue
                            
                        # Convert CoinMarketCap slug to symbol (bitcoin -> BTC)
                        symbol = self.config.token_mapper.cmc_slug_to_symbol(cmc_slug)
                        
                        # Extract data from nested quote structure
                        quote_data = item.get('quote', {}).get('USD', {})
                        
                        standardized_item = {
                            'symbol': symbol,
                            'current_price': self._safe_float(quote_data.get('price', 0)),
                            'price_change_percentage_24h': self._safe_float(quote_data.get('percent_change_24h', 0)),
                            'price_change_percentage_1h': self._safe_float(quote_data.get('percent_change_1h', 0)),
                            'price_change_percentage_7d': self._safe_float(quote_data.get('percent_change_7d', 0)),
                            'total_volume': self._safe_float(quote_data.get('volume_24h', 0)),
                            'volume': self._safe_float(quote_data.get('volume_24h', 0)),  # Alias
                            'market_cap': self._safe_float(quote_data.get('market_cap', 0)),
                            'market_cap_rank': item.get('cmc_rank', 999),
                            'circulating_supply': self._safe_float(item.get('circulating_supply', 0)),
                            'max_supply': self._safe_float(item.get('max_supply', 0)),
                            'last_updated': quote_data.get('last_updated') or item.get('last_updated'),
                            
                            # Preserve original fields
                            'slug': cmc_slug,
                            'source': 'coinmarketcap'
                        }
                        
                        result[symbol] = standardized_item
                        logger.logger.debug(f"CoinMarketCap: {cmc_slug} -> {symbol}")

                    else:
                        # Unknown format - try to extract symbol directly
                        symbol = item.get('symbol', 'UNKNOWN').upper()
                        if symbol != 'UNKNOWN':
                            standardized_item = item.copy()
                            standardized_item['symbol'] = symbol
                            standardized_item['source'] = 'unknown'
                            result[symbol] = standardized_item
                            logger.logger.debug(f"Unknown format: direct symbol {symbol}")

                except Exception as e:
                    logger.logger.warning(f"Error standardizing market data item: {e}")
                    continue

            processing_time = time.time() - start_time
            logger.logger.info(f"✅ Standardized {len(result)} tokens in {processing_time:.3f}s using TokenMappingManager")
            
            # Log sample of results
            if result:
                sample_symbols = list(result.keys())[:5]
                logger.logger.debug(f"📊 Sample standardized symbols: {sample_symbols}")

            return result

        # Return empty dict for other types
        return {}
    
    def fix_database_connection(self):
        """
        Reconnect to the database if the connection has been closed
        """
        # Check if database connection exists and is closed
        if not hasattr(self, 'db') or self.db is None:
            logger.logger.error("❌ Database not initialized, creating new connection")
            from database import CryptoDatabase
            self.db = CryptoDatabase()
            logger.logger.info("✅ Created new database connection")
            return True
            
        # Check if connection is valid by attempting a simple query
        try:
            # Use get_connection method with proper error handling
            if hasattr(self.db, '_get_connection'):
                conn, cursor = self.db._get_connection()
                cursor.execute("SELECT 1")  # Simple test query
                cursor.fetchone()
                logger.logger.debug("✅ Database connection verified")
                return True
        except Exception as e:
            if "Cannot operate on a closed database" in str(e) or "database is closed" in str(e):
                logger.logger.warning(f"⚠️ Database connection closed: {str(e)}")
                # Reinitialize the database connection
                try:
                    del self.db  # Remove the old connection
                    from database import CryptoDatabase
                    self.db = CryptoDatabase()
                    logger.logger.info("✅ Reconnected to database successfully")
                    return True
                except Exception as reconnect_error:
                    logger.logger.error(f"❌ Failed to reconnect to database: {str(reconnect_error)}")
                    return False
            else:
                logger.logger.error(f"❌ Database error: {str(e)}")
                return False
                
        return True  # Connection is valid

    def _detect_item_source(self, item: dict) -> str:
        """
        Detect the source format of a market data item
        
        Args:
            item: Market data item dictionary
            
        Returns:
            Source type: 'database', 'coingecko', 'coinmarketcap', or 'unknown'
        """
        # Priority order is important - database records can have both 'id' and 'chain'
        if 'chain' in item:
            return 'database'
        elif 'id' in item and 'current_price' in item:
            return 'coingecko'
        elif 'slug' in item and 'quote' in item:
            return 'coinmarketcap'
        else:
            return 'unknown'

    def _safe_float(self, value) -> float:
        """
        Safely convert value to float
        
        Args:
            value: Value to convert
            
        Returns:
            Float value or 0.0 if conversion fails
        """
        try:
            if value is None or value == '':
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    @ensure_naive_datetimes
    def _check_for_reply_opportunities(self, market_data: Dict[str, Any]) -> bool:
        """
        Enhanced check for posts to reply to with multiple fallback mechanisms
        and detailed logging for better debugging
    
        Args:
            market_data: Current market data dictionary
        
        Returns:
            True if any replies were posted
        """
        now = strip_timezone(datetime.now())

        # Check if it's time to look for posts to reply to
        time_since_last_check = safe_datetime_diff(now, self.last_reply_check) / 20
        if time_since_last_check < self.reply_check_interval:
            logger.logger.debug(f"Skipping reply check, {time_since_last_check:.1f} minutes since last check (interval: {self.reply_check_interval})")
            return False
    
        # Also check cooldown period
        time_since_last_reply = safe_datetime_diff(now, self.last_reply_time) / 20
        if time_since_last_reply < self.reply_cooldown:
            logger.logger.debug(f"In reply cooldown period, {time_since_last_reply:.1f} minutes since last reply (cooldown: {self.reply_cooldown})")
            return False
    
        logger.logger.info("Starting check for posts to reply to")
        self.last_reply_check = now
    
        try:
            # Try multiple post gathering strategies with fallbacks
            success = self._try_normal_reply_strategy(market_data)
            if success:
                return True
            
            # First fallback: Try with lower threshold for reply-worthy posts
            success = self._try_lower_threshold_reply_strategy(market_data)
            if success:
                return True
            
            # Second fallback: Try replying to trending posts even if not directly crypto-related
            success = self._try_trending_posts_reply_strategy(market_data)
            if success:
                return True
        
            # Final fallback: Try replying to any post from major crypto accounts
            success = self._try_crypto_accounts_reply_strategy(market_data)
            if success:
                return True
        
            logger.logger.warning("All reply strategies failed, no suitable posts found")
            return False
        
        except Exception as e:
            logger.log_error("Check For Reply Opportunities", str(e))
            return False

    @ensure_naive_datetimes
    def _try_normal_reply_strategy(self, market_data: Dict[str, Any]) -> bool:
        """
        Standard reply strategy with normal thresholds
    
        Args:
            market_data: Market data dictionary
        
        Returns:
            True if any replies were posted
        """
        try:
            # Get more posts to increase chances of finding suitable ones
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 3)
            if not posts:
                logger.logger.warning("No posts found during timeline scraping")
                return False
            
            logger.logger.info(f"Timeline scraping completed - found {len(posts)} posts")
        
            # Log sample posts for debugging
            for i, post in enumerate(posts[:3]):
                logger.logger.info(f"Sample post {i}: {post.get('text', '')[:100]}...")
        
            # Find market-related posts
            logger.logger.info(f"Finding market-related posts among {len(posts)} scraped posts")
            market_posts = self.content_analyzer.find_market_related_posts(posts)
            logger.logger.info(f"Found {len(market_posts)} market-related posts")
        
            if not market_posts:
                logger.logger.warning("No market-related posts found")
                return False
            
            # Filter out posts we've already replied to
            unreplied_posts = self.content_analyzer.filter_already_replied_posts(market_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied market-related posts")
        
            if not unreplied_posts:
                logger.logger.warning("All market-related posts have already been replied to")
                return False
            
            # Analyze content of each post for engagement metrics
            analyzed_posts = []
            for post in unreplied_posts:
                analysis = self.content_analyzer.analyze_post(post)
                post['content_analysis'] = analysis
                analyzed_posts.append(post)
        
            # Only reply to posts worth replying to based on analysis
            reply_worthy_posts = [post for post in analyzed_posts if post['content_analysis'].get('reply_worthy', False)]
            logger.logger.info(f"Found {len(reply_worthy_posts)} reply-worthy posts")
        
            if not reply_worthy_posts:
                logger.logger.warning("No reply-worthy posts found among market-related posts")
                return False
        
            # Balance between high value and regular posts
            high_value_posts = [post for post in reply_worthy_posts if post['content_analysis'].get('high_value', False)]
            posts_to_reply = high_value_posts[:int(self.max_replies_per_cycle * 0.7)]
            remaining_slots = self.max_replies_per_cycle - len(posts_to_reply)
        
            if remaining_slots > 0:
                medium_value_posts = [p for p in reply_worthy_posts if p not in high_value_posts]
                medium_value_posts.sort(key=lambda x: x['content_analysis'].get('reply_score', 0), reverse=True)
                posts_to_reply.extend(medium_value_posts[:remaining_slots])
        
            if not posts_to_reply:
                logger.logger.warning("No posts selected for reply after prioritization")
                return False
        
            # Generate and post replies
            logger.logger.info(f"Starting to reply to {len(posts_to_reply)} prioritized posts")
            successful_replies = self.reply_handler.reply_to_posts(posts_to_reply, market_data, max_replies=self.max_replies_per_cycle)
        
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies using normal strategy")
                self.last_reply_time = strip_timezone(datetime.now())
                return True
            else:
                logger.logger.warning("No replies were successfully posted using normal strategy")
                return False
            
        except Exception as e:
            logger.log_error("Normal Reply Strategy", str(e))
            return False

    @ensure_naive_datetimes
    def _try_lower_threshold_reply_strategy(self, market_data: Dict[str, Any]) -> bool:
        """
        Reply strategy with lower thresholds for reply-worthiness
    
        Args:
            market_data: Market data dictionary
        
        Returns:
            True if any replies were posted
        """
        try:
            # Get fresh posts
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 3)
            if not posts:
                logger.logger.warning("No posts found during lower threshold timeline scraping")
                return False
            
            logger.logger.info(f"Lower threshold timeline scraping completed - found {len(posts)} posts")
        
            # Find posts with ANY crypto-related content, not just market-focused
            crypto_posts = []
            for post in posts:
                text = post.get('text', '').lower()
                # Check for ANY crypto-related terms
                if any(term in text for term in ['crypto', 'bitcoin', 'btc', 'eth', 'blockchain', 'token', 'coin', 'defi']):
                    crypto_posts.append(post)
        
            logger.logger.info(f"Found {len(crypto_posts)} crypto-related posts with lower threshold")
        
            if not crypto_posts:
                logger.logger.warning("No crypto-related posts found with lower threshold")
                return False
            
            # Filter out posts we've already replied to
            unreplied_posts = self.content_analyzer.filter_already_replied_posts(crypto_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied crypto-related posts with lower threshold")
        
            if not unreplied_posts:
                return False
            
            # Add basic content analysis but don't filter by reply_worthy
            analyzed_posts = []
            for post in unreplied_posts:
                analysis = self.content_analyzer.analyze_post(post)
                # Override reply_worthy to True for all posts in this fallback
                analysis['reply_worthy'] = True
                post['content_analysis'] = analysis
                analyzed_posts.append(post)
        
            # Just take the top N posts by engagement
            analyzed_posts.sort(key=lambda x: x.get('engagement_score', 0), reverse=True)
            posts_to_reply = analyzed_posts[:self.max_replies_per_cycle]
        
            if not posts_to_reply:
                return False
        
            # Generate and post replies with lower standards
            logger.logger.info(f"Starting to reply to {len(posts_to_reply)} posts with lower threshold")
            successful_replies = self.reply_handler.reply_to_posts(posts_to_reply, market_data, max_replies=self.max_replies_per_cycle)
        
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies using lower threshold strategy")
                self.last_reply_time = strip_timezone(datetime.now())
                return True
            else:
                logger.logger.warning("No replies were successfully posted using lower threshold strategy")
                return False
            
        except Exception as e:
            logger.log_error("Lower Threshold Reply Strategy", str(e))
            return False

    @ensure_naive_datetimes
    def _try_trending_posts_reply_strategy(self, market_data: Dict[str, Any]) -> bool:
        """
        Reply strategy focusing on trending posts regardless of crypto relevance
    
        Args:
            market_data: Market data dictionary
        
        Returns:
            True if any replies were posted
        """
        try:
            # Get trending posts - use a different endpoint if possible
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 2)
            if not posts:
                return False
            
            logger.logger.info(f"Trending posts scraping completed - found {len(posts)} posts")
        
            # Sort by engagement (likes, retweets, etc.) to find trending posts
            posts.sort(key=lambda x: (
                x.get('like_count', 0) + 
                x.get('retweet_count', 0) * 2 + 
                x.get('reply_count', 0) * 0.5
            ), reverse=True)
        
            # Get the top trending posts
            trending_posts = posts[:int(self.max_replies_per_cycle * 1.5)]
        
            # Filter out posts we've already replied to
            unreplied_posts = self.content_analyzer.filter_already_replied_posts(trending_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied trending posts")
        
            if not unreplied_posts:
                return False
            
            # Add minimal content analysis
            for post in unreplied_posts:
                post['content_analysis'] = {'reply_worthy': True, 'reply_score': 75}
        
            # Generate and post replies to trending content
            logger.logger.info(f"Starting to reply to {len(unreplied_posts[:self.max_replies_per_cycle])} trending posts")
            successful_replies = self.reply_handler.reply_to_posts(
                unreplied_posts[:self.max_replies_per_cycle], 
                market_data, 
                max_replies=self.max_replies_per_cycle
            )
        
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies to trending posts")
                self.last_reply_time = strip_timezone(datetime.now())
                return True
            else:
                logger.logger.warning("No replies were successfully posted to trending posts")
                return False
            
        except Exception as e:
            logger.log_error("Trending Posts Reply Strategy", str(e))
            return False

    @ensure_naive_datetimes
    def _try_crypto_accounts_reply_strategy(self, market_data: Dict[str, Any]) -> bool:
        """
        Reply strategy focusing on major crypto accounts regardless of post content
    
        Args:
            market_data: Market data dictionary
        
        Returns:
            True if any replies were posted
        """
        try:
            # Major crypto accounts to target
            crypto_accounts = [
                'cz_binance', 'vitalikbuterin', 'SBF_FTX', 'aantonop', 'cryptohayes', 'coinbase',
                'kraken', 'whale_alert', 'CoinDesk', 'Cointelegraph', 'binance', 'BitcoinMagazine'
            ]
        
            all_posts = []
        
            # Try to get posts from specific accounts
            for account in crypto_accounts[:3]:  # Limit to 3 accounts to avoid too many requests
                try:
                    # This would need an account-specific scraper method
                    # For now, use regular timeline as placeholder
                    posts = self.timeline_scraper.scrape_timeline(count=5)
                    if posts:
                        all_posts.extend(posts)
                except Exception as e:
                    logger.logger.debug(f"Error getting posts for account {account}: {str(e)}")
                    continue
        
            # If no account-specific posts, get timeline posts and filter
            if not all_posts:
                posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 3)
            
                # Filter for posts from crypto accounts (based on handle or name)
                for post in posts:
                    handle = post.get('author_handle', '').lower()
                    name = post.get('author_name', '').lower()
                
                    if any(account.lower() in handle or account.lower() in name for account in crypto_accounts):
                        all_posts.append(post)
                    
                    # Also include posts with many crypto terms
                    text = post.get('text', '').lower()
                    crypto_terms = ['crypto', 'bitcoin', 'btc', 'eth', 'blockchain', 'token', 'coin', 'defi', 
                                   'altcoin', 'nft', 'mining', 'wallet', 'address', 'exchange']
                    if sum(1 for term in crypto_terms if term in text) >= 3:
                        all_posts.append(post)
        
            # Remove duplicates
            unique_posts = []
            post_ids = set()
            for post in all_posts:
                post_id = post.get('post_id')
                if post_id and post_id not in post_ids:
                    post_ids.add(post_id)
                    unique_posts.append(post)
        
            logger.logger.info(f"Found {len(unique_posts)} posts from crypto accounts")
        
            # Filter out posts we've already replied to
            unreplied_posts = self.content_analyzer.filter_already_replied_posts(unique_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied posts from crypto accounts")
        
            if not unreplied_posts:
                return False
            
            # Add minimal content analysis
            for post in unreplied_posts:
                post['content_analysis'] = {'reply_worthy': True, 'reply_score': 80}
        
            # Generate and post replies to crypto accounts
            logger.logger.info(f"Starting to reply to {len(unreplied_posts[:self.max_replies_per_cycle])} crypto account posts")
            successful_replies = self.reply_handler.reply_to_posts(
                unreplied_posts[:self.max_replies_per_cycle], 
                market_data, 
                max_replies=self.max_replies_per_cycle
            )
        
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies to crypto accounts")
                self.last_reply_time = strip_timezone(datetime.now())
                return True
            else:
                logger.logger.warning("No replies were successfully posted to crypto accounts")
                return False
            
        except Exception as e:
            logger.log_error("Crypto Accounts Reply Strategy", str(e))
            return False   

    def _get_historical_volume_data(self, token: str, timeframe: str = "1h") -> List[float]:
        """
        Get historical volume data for volume trend analysis
        
        Args:
            token: Token symbol
            timeframe: Timeframe for analysis
        
        Returns:
            List of volume values
        """
        try:
            # Map timeframe to hours
            timeframe_hours = {
                "1h": 48,    # 48 hours for hourly analysis
                "24h": 168,  # 7 days for daily analysis  
                "7d": 720    # 30 days for weekly analysis
            }
            
            hours = timeframe_hours.get(timeframe, 48)
            
            # Get historical data
            historical_data = self._get_historical_price_data(token, hours, timeframe)
            
            # Handle "Never" response
            if historical_data == "Never" or not historical_data:
                logger.logger.debug(f"No volume data for {token}")
                return []
            
            # Extract volumes
            volumes = []
            for entry in historical_data:
                if isinstance(entry, dict):
                    volume = entry.get('volume', 0.0)
                    if volume > 0:
                        volumes.append(volume)
            
            logger.logger.debug(f"Retrieved {len(volumes)} volume data points for {token}")
            return volumes
            
        except Exception as e:
            logger.log_error(f"Get Historical Volume Data - {token}", str(e))
            return []
       
    def _is_duplicate_analysis(self, new_tweet: str, last_posts: List[str], timeframe: str = "1h") -> bool:
        """
        Enhanced duplicate detection with time-based thresholds and timeframe awareness.
        Applies different checks based on how recently similar content was posted:
        - Very recent posts (< 15 min): Check for exact matches
        - Recent posts (15-30 min): Check for high similarity
        - Older posts (> 30 min): Allow similar content
        
        Args:
            new_tweet: The new tweet text to check for duplication
            last_posts: List of recently posted tweets
            timeframe: Timeframe for the post (1h, 24h, 7d)
            
        Returns:
            Boolean indicating if the tweet is a duplicate
        """
        try:
            # Log that we're using enhanced duplicate detection
            logger.logger.info(f"Using enhanced time-based duplicate detection for {timeframe} timeframe")
           
            # Define time windows for different levels of duplicate checking
            # Adjust windows based on timeframe
            if timeframe == "1h":
                VERY_RECENT_WINDOW_MINUTES = 15
                RECENT_WINDOW_MINUTES = 30
                HIGH_SIMILARITY_THRESHOLD = 0.85  # 85% similar for recent posts
            elif timeframe == "24h":
                VERY_RECENT_WINDOW_MINUTES = 120  # 2 hours
                RECENT_WINDOW_MINUTES = 240       # 4 hours
                HIGH_SIMILARITY_THRESHOLD = 0.80  # Slightly lower threshold for daily predictions
            else:  # 7d
                VERY_RECENT_WINDOW_MINUTES = 720  # 12 hours
                RECENT_WINDOW_MINUTES = 1440      # 24 hours
                HIGH_SIMILARITY_THRESHOLD = 0.75  # Even lower threshold for weekly predictions
           
            # 1. Check for exact matches in very recent database entries
            conn = self.config.db.conn
            cursor = conn.cursor()
           
            # Very recent exact duplicates check
            cursor.execute("""
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
                AND timeframe = ?
            """, (VERY_RECENT_WINDOW_MINUTES, timeframe))
           
            very_recent_posts = [row[0] for row in cursor.fetchall()]
           
            # Check for exact matches in very recent posts
            for post in very_recent_posts:
                if post.strip() == new_tweet.strip():
                    logger.logger.info(f"Exact duplicate detected within last {VERY_RECENT_WINDOW_MINUTES} minutes for {timeframe}")
                    return True
           
            # 2. Check for high similarity in recent posts
            cursor.execute("""
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
                AND timestamp < datetime('now', '-' || ? || ' minutes')
                AND timeframe = ?
            """, (RECENT_WINDOW_MINUTES, VERY_RECENT_WINDOW_MINUTES, timeframe))
           
            recent_posts = [row[0] for row in cursor.fetchall()]
           
            # Calculate similarity for recent posts
            new_content = new_tweet.lower()
           
            for post in recent_posts:
                post_content = post.lower()
               
                # Calculate a simple similarity score based on word overlap
                new_words = set(new_content.split())
                post_words = set(post_content.split())
               
                if new_words and post_words:
                    overlap = len(new_words.intersection(post_words))
                    similarity = overlap / max(len(new_words), len(post_words))
                   
                    # Apply high similarity threshold for recent posts
                    if similarity > HIGH_SIMILARITY_THRESHOLD:
                        logger.logger.info(f"High similarity ({similarity:.2f}) detected within last {RECENT_WINDOW_MINUTES} minutes for {timeframe}")
                        return True
           
            # 3. Also check exact duplicates in last posts from Twitter
            # This prevents double-posting in case of database issues
            for post in last_posts:
                if post.strip() == new_tweet.strip():
                    logger.logger.info(f"Exact duplicate detected in recent Twitter posts for {timeframe}")
                    return True
           
            # If we get here, it's not a duplicate according to our criteria
            logger.logger.info(f"No duplicates detected with enhanced time-based criteria for {timeframe}")
            return False
           
        except Exception as e:
            logger.log_error(f"Duplicate Check - {timeframe}", str(e))
            # If the duplicate check fails, allow the post to be safe
            logger.logger.warning("Duplicate check failed, allowing post to proceed")
            return False

    def _start_prediction_thread(self) -> None:
        """
        Start background thread for asynchronous prediction generation
        """
        if self.prediction_thread is None or not self.prediction_thread.is_alive():
            self.prediction_thread_running = True
            self.prediction_thread = threading.Thread(target=self._process_prediction_queue)
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
            logger.logger.info("Started prediction processing thread")
           
    def _process_prediction_queue(self) -> None:
        """
        Process predictions from the queue in the background
        """
        while self.prediction_thread_running:
            try:
                # Get a prediction task from the queue with timeout
                try:
                    task = self.prediction_queue.get(timeout=10)
                except queue.Empty:
                    # No tasks, just continue the loop
                    continue
                   
                # Process the prediction task
                token, timeframe, market_data = task
               
                logger.logger.debug(f"Processing queued prediction for {token} ({timeframe})")
               
                # Generate the prediction
                prediction = self.prediction_engine._generate_predictions(
                    token=token, 
                    market_data=market_data,
                    timeframe=timeframe
                )
               
                # Store in memory for quick access
                self.timeframe_predictions[timeframe][token] = prediction
               
                # Mark task as done
                self.prediction_queue.task_done()
               
                # Short sleep to prevent CPU overuse
                time.sleep(0.5)
               
            except Exception as e:
                logger.log_error("Prediction Thread Error", str(e))
                time.sleep(5)  # Sleep longer on error
               
        logger.logger.info("Prediction processing thread stopped")

    def _login_to_twitter(self) -> bool:
        """
        Log into Twitter with enhanced verification and detection of existing sessions
    
        Returns:
            Boolean indicating login success
        """
        try:
            logger.logger.info("Starting Twitter login")
        
            # Check if browser and driver are properly initialized
            if not self.browser or not self.browser.driver:
                logger.logger.error("Browser or driver not initialized")
                return False
            
            self.browser.driver.set_page_load_timeout(45)
    
            # First navigate to Twitter home page instead of login page directly
            self.browser.driver.get('https://twitter.com')
            time.sleep(5)
        
            # Check if we're already logged in
            already_logged_in = False
            login_indicators = [
                '[data-testid="SideNav_NewTweet_Button"]',
                '[data-testid="AppTabBar_Profile_Link"]',
                '[aria-label="Tweet"]',
                '.DraftEditor-root'  # Tweet composer element
            ]
        
            for indicator in login_indicators:
                try:
                    if self.browser.check_element_exists(indicator):
                        already_logged_in = True
                        logger.logger.info("Already logged into Twitter, using existing session")
                        return True
                except Exception:
                    continue
        
            if not already_logged_in:
                logger.logger.info("Not logged in, proceeding with login process")
                self.browser.driver.get('https://twitter.com/login')
                time.sleep(5)

                username_field = WebDriverWait(self.browser.driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[autocomplete='username']"))
                )
                username_field.click()
                time.sleep(1)
                username_field.send_keys(config.TWITTER_USERNAME)
                time.sleep(2)

                next_button = WebDriverWait(self.browser.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
                )
                next_button.click()
                time.sleep(3)

                password_field = WebDriverWait(self.browser.driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))
                )
                password_field.click()
                time.sleep(1)
                password_field.send_keys(self.config.TWITTER_PASSWORD)
                time.sleep(2)

                login_button = WebDriverWait(self.browser.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[text()='Log in']"))
                )
                login_button.click()
                time.sleep(10)

            return self._verify_login()

        except Exception as e:
            logger.log_error("Twitter Login", str(e))
            return False

    def _verify_login(self) -> bool:
        """
        Verify Twitter login success with improved error handling and type safety.
    
        Returns:
            Boolean indicating if login verification succeeded
        """
        try:
            # First check if browser and driver are properly initialized
            if not self.browser:
                logger.logger.error("Browser not initialized")
                return False
            
            if not hasattr(self.browser, 'driver') or self.browser.driver is None:
                logger.logger.error("Browser driver not initialized")
                return False
            
            # Store a local reference to driver with proper type annotation
            driver: Optional[WebDriver] = self.browser.driver
        
            # Define verification methods that use the driver variable directly
            def check_new_tweet_button() -> bool:
                try:
                    element = WebDriverWait(driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]'))
                    )
                    return element is not None
                except Exception:
                    return False
                
            def check_profile_link() -> bool:
                try:
                    element = WebDriverWait(driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="AppTabBar_Profile_Link"]'))
                    )
                    return element is not None
                except Exception:
                    return False
                
            def check_url_contains_home() -> bool:
                try:
                    if driver and driver.current_url:
                        return any(path in driver.current_url for path in ['home', 'twitter.com/home'])
                    return False
                except Exception:
                    return False
        
            # Use proper function references instead of lambdas to improve type safety
            verification_methods = [
                check_new_tweet_button,
                check_profile_link,
                check_url_contains_home
            ]
        
            # Try each verification method
            for method in verification_methods:
                try:
                    if method():
                        logger.logger.info("Login verification successful")
                        return True
                except Exception as method_error:
                    logger.logger.debug(f"Verification method failed: {str(method_error)}")
                    continue
        
            logger.logger.warning("All verification methods failed - user not logged in")
            return False
        
        except Exception as e:
            logger.log_error("Login Verification", str(e))
            return False

    def _queue_predictions_for_all_timeframes(self, token: str, market_data: Dict[str, Any]) -> None:
        """
        Queue predictions for all timeframes for a specific token
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
        """
        for timeframe in self.timeframes:
            # Skip if we already have a recent prediction
            if (token in self.timeframe_predictions.get(timeframe, {}) and 
               safe_datetime_diff(datetime.now(), self.timeframe_predictions[timeframe].get(token, {}).get('timestamp', 
                                                                                        datetime.now() - timedelta(hours=3))) 
               < 3600):  # Less than 1 hour old
                logger.logger.debug(f"Skipping {timeframe} prediction for {token} - already have recent prediction")
                continue
               
            # Add prediction task to queue
            self.prediction_queue.put((token, timeframe, market_data))
            logger.logger.debug(f"Queued {timeframe} prediction for {token}")

    @ensure_naive_datetimes
    def _post_analysis(self, tweet_text: str, timeframe: str = "1h") -> bool:
        """
        Modernized post analysis method with proper Twitter compose box interaction
        
        Fixed to properly handle Twitter's contenteditable compose interface by:
        1. Ensuring we're on the right page
        2. Properly activating the compose area
        3. Using correct interaction methods for contenteditable elements
        4. Comprehensive retry logic with detailed debugging
        
        Args:
            tweet_text (str): Text content to post to Twitter
            timeframe (str): Timeframe identifier for scheduling ("1h", "24h", "7d")
        
        Returns:
            bool: True if post was successfully created and posted, False if all attempts failed
        """
        # ================================================================
        # 🔍 PRE-POSTING DATA FLOW VALIDATION
        # ================================================================
        
        logger.logger.debug(f"🚀 FIXED POST ANALYSIS START: {timeframe}")
        logger.logger.debug(f"📝 Content length: {len(tweet_text)} chars")
        logger.logger.debug(f"🕐 Current time: {datetime.now()}")
        logger.logger.debug(f"🔄 Timeframe: {timeframe}")
        
        # Validate input parameters
        if not tweet_text:
            logger.logger.error("❌ POST FAILED - Empty tweet_text provided")
            return False
        
        if not timeframe:
            logger.logger.warning("⚠️ No timeframe provided, defaulting to '1h'")
            timeframe = "1h"
        
        # ================================================================
        # 🛠️ CONTENT PREPROCESSING WITH DEBUG LOGGING
        # ================================================================
        
        original_content = tweet_text
        if not tweet_text or tweet_text.strip().lower() == "neutral":
            logger.logger.warning(f"🔄 CONTENT PREPROCESSING: Detected neutral/empty content")
            
            crypto_buzz_phrases = [
                "To The Moon!!!! 🚀🚀🚀", "HODL Strong! 💎🙌", "Big things coming! 📈",
                "Bullish AF! 🐂", "Buy the dip! 📉➡️📈", "This is the way! ✨",
                "Diamond hands only! 💎", "Breaking out! 📊", "Getting ready to pump! 💪",
                "Solid fundamentals! 👨‍💻", "Undervalued gem! 💎", "Next 100x potential! 🚀",
                "The future is now! 🔮", "Early adopters win! 🏆", "Massive potential! 💰",
                "Smart money flowing in! 🧠💵", "Not financial advice but... 👀",
                "Lightning in a bottle! ⚡", "Paper hands NGMI! 📝❌", "LFG!!!!"
            ]
            
            tweet_text = random.choice(crypto_buzz_phrases)
            logger.logger.info(f"✅ CONTENT REPLACEMENT: Used fallback buzz phrase")
        
        logger.logger.debug(f"📊 FINAL CONTENT: '{tweet_text[:50]}{'...' if len(tweet_text) > 50 else ''}' ({len(tweet_text)} chars)")
        
        # ================================================================
        # 🔗 BROWSER & DRIVER STATE VALIDATION
        # ================================================================
        
        logger.logger.debug(f"🔍 BROWSER STATE CHECK:")
        if not self.browser or not hasattr(self.browser, 'driver') or self.browser.driver is None:
            logger.logger.error("❌ POST FAILED - Browser/driver not initialized")
            return False
        
        logger.logger.debug(f"✅ Browser and driver validated")
        
        # ================================================================
        # 🏠 ENSURE WE'RE ON TWITTER HOME PAGE
        # ================================================================
        
        try:
            current_url = self.browser.driver.current_url
            logger.logger.debug(f"🔍 CURRENT URL: {current_url}")
            
            # Check if we're on the right page
            if "twitter.com" not in current_url and "x.com" not in current_url:
                logger.logger.warning(f"⚠️ Not on Twitter/X - navigating to home")
                self.browser.driver.get('https://x.com/home')
                time.sleep(5)
            elif "login" in current_url.lower():
                logger.logger.error("❌ POST FAILED - Still on login page")
                return False
            elif "home" not in current_url.lower():
                logger.logger.info(f"🔄 Navigating to home page from: {current_url}")
                self.browser.driver.get('https://x.com/home')
                time.sleep(3)
            
            logger.logger.debug(f"✅ Page navigation validated")
            
        except Exception as e:
            logger.logger.error(f"❌ PAGE NAVIGATION FAILED: {str(e)}")
            return False
        
        # ================================================================
        # 🔄 RETRY LOOP WITH COMPREHENSIVE ERROR TRACKING
        # ================================================================
        
        max_retries = 3
        retry_count = 0
        retry_reasons = []
        
        logger.logger.info(f"🎯 STARTING POST ATTEMPTS: Max {max_retries} retries for {timeframe}")
        
        while retry_count < max_retries:
            attempt_number = retry_count + 1
            logger.logger.info(f"🔄 POST ATTEMPT {attempt_number}/{max_retries} - {timeframe}")
            
            try:
                # ================================================================
                # 📝 ENHANCED TWITTER COMPOSE AREA INTERACTION
                # ================================================================
                
                logger.logger.debug(f"🔍 STEP 1: Ensuring page is at top and locating compose area")
                
                # CRITICAL: Force scroll to top to prevent Twitter's scroll-blocking behavior
                try:
                    logger.logger.debug(f"🔝 FORCING SCROLL TO TOP to prevent element interception")
                    self.browser.driver.execute_script("window.scrollTo(0, 0);")
                    time.sleep(2)  # Give time for scroll to complete
                    
                    # Also try to scroll the document body to top
                    self.browser.driver.execute_script("document.body.scrollTop = 0; document.documentElement.scrollTop = 0;")
                    time.sleep(1)
                    
                    logger.logger.debug(f"✅ Page scrolled to top successfully")
                    
                except Exception as scroll_error:
                    logger.logger.warning(f"⚠️ Scroll to top failed: {str(scroll_error)}")
                    # Continue anyway, but this might cause issues
                
                # Try multiple strategies to find and activate the compose area
                compose_activated = False
                text_area = None
                
                # STRATEGY 1: Find the contenteditable div directly
                try:
                    logger.logger.debug(f"📝 STRATEGY 1: Looking for contenteditable compose area")
                    text_area = WebDriverWait(self.browser.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
                    )
                    
                    # Check if element is interactable
                    if text_area.is_displayed() and text_area.is_enabled():
                        logger.logger.debug(f"✅ Found compose area with Strategy 1")
                        compose_activated = True
                    else:
                        logger.logger.debug(f"⚠️ Compose area found but not interactable")
                        
                except TimeoutException:
                    logger.logger.debug(f"❌ Strategy 1 failed - compose area not found")
                
                # STRATEGY 2: Look for "What's happening?" placeholder and click it
                if not compose_activated:
                    try:
                        logger.logger.debug(f"📝 STRATEGY 2: Looking for 'What's happening?' area")
                        
                        # Ensure we're still at the top before looking for placeholder
                        self.browser.driver.execute_script("window.scrollTo(0, 0);")
                        time.sleep(1)
                        
                        placeholder_area = self.browser.driver.find_element(
                            By.XPATH, "//div[contains(text(), \"What's happening?\")]"
                        )
                        
                        if placeholder_area:
                            logger.logger.debug(f"🔍 Found placeholder area, ensuring it's visible")
                            
                            # Make sure the element is in viewport and not covered
                            self.browser.driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center'});", placeholder_area)
                            time.sleep(1)
                            
                            # Use JavaScript click to avoid interception issues
                            logger.logger.debug(f"🔍 JavaScript clicking placeholder to activate")
                            self.browser.driver.execute_script("arguments[0].click();", placeholder_area)
                            time.sleep(2)
                            
                            # Now try to find the active text area
                            text_area = WebDriverWait(self.browser.driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
                            )
                            logger.logger.debug(f"✅ Compose area activated with Strategy 2")
                            compose_activated = True
                            
                    except Exception as e:
                        logger.logger.debug(f"❌ Strategy 2 failed: {str(e)}")
                
                # STRATEGY 3: Look for any contenteditable div in compose area
                if not compose_activated:
                    try:
                        logger.logger.debug(f"📝 STRATEGY 3: Looking for any contenteditable div")
                        
                        # Ensure we're still at the top
                        self.browser.driver.execute_script("window.scrollTo(0, 0);")
                        time.sleep(1)
                        
                        text_area = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, 'div[contenteditable="true"][role="textbox"]'))
                        )
                        logger.logger.debug(f"✅ Found contenteditable area with Strategy 3")
                        compose_activated = True
                        
                    except Exception as e:
                        logger.logger.debug(f"❌ Strategy 3 failed: {str(e)}")
                
                # STRATEGY 4: Force compose box to open using keyboard shortcut
                if not compose_activated:
                    try:
                        logger.logger.debug(f"📝 STRATEGY 4: Using keyboard shortcut to open compose")
                        
                        # Ensure we're at top and focused on page
                        self.browser.driver.execute_script("window.scrollTo(0, 0);")
                        time.sleep(1)
                        
                        # Focus on the body element first
                        body = self.browser.driver.find_element(By.TAG_NAME, "body")
                        body.click()
                        time.sleep(1)
                        
                        # Try Twitter's compose keyboard shortcut (N key)
                        from selenium.webdriver.common.keys import Keys
                        body.send_keys("n")
                        time.sleep(2)
                        
                        # Now try to find the text area
                        text_area = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
                        )
                        logger.logger.debug(f"✅ Compose area activated with Strategy 4 (keyboard)")
                        compose_activated = True
                        
                    except Exception as e:
                        logger.logger.debug(f"❌ Strategy 4 failed: {str(e)}")
                
                # If no strategy worked, fail this attempt
                if not compose_activated or not text_area:
                    error_msg = f"All compose area detection strategies failed"
                    retry_reasons.append(f"Attempt {attempt_number}: {error_msg}")
                    logger.logger.warning(f"⚠️ RETRY REASON: {error_msg}")
                    raise Exception(error_msg)
                
                # ================================================================
                # 🎯 ACTIVATE AND INTERACT WITH COMPOSE AREA
                # ================================================================
                
                logger.logger.debug(f"🔍 STEP 2: Activating compose area for input")
                
                # Ensure the text area is focused and ready
                try:
                    # Force scroll to top one more time before interaction
                    self.browser.driver.execute_script("window.scrollTo(0, 0);")
                    time.sleep(1)
                    
                    # Scroll element into view carefully to avoid covering
                    self.browser.driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center'});", text_area)
                    time.sleep(1)
                    
                    # Check if element is still not covered after scroll
                    element_location = text_area.location
                    logger.logger.debug(f"🔍 Text area location: x={element_location['x']}, y={element_location['y']}")
                    
                    # Focus using JavaScript to avoid click interception
                    self.browser.driver.execute_script("arguments[0].focus();", text_area)
                    time.sleep(1)
                    
                    # Try a gentle JavaScript click as backup
                    self.browser.driver.execute_script("arguments[0].click();", text_area)
                    time.sleep(1)
                    
                    logger.logger.debug(f"✅ Compose area focused and activated")
                    
                except Exception as e:
                    error_msg = f"Failed to activate compose area: {str(e)}"
                    retry_reasons.append(f"Attempt {attempt_number}: {error_msg}")
                    logger.logger.warning(f"⚠️ RETRY REASON: {error_msg}")
                    raise Exception(error_msg)
                
                # ================================================================
                # 🛡️ CONTENT SAFETY AND INPUT
                # ================================================================
                
                logger.logger.debug(f"🔍 STEP 3: Preparing and inputting content")
                
                # Ensure tweet text only contains BMP characters
                safe_tweet_text = ''.join(char for char in tweet_text if ord(char) < 0x10000)
                
                if len(safe_tweet_text) != len(tweet_text):
                    filtered_chars = len(tweet_text) - len(safe_tweet_text)
                    logger.logger.debug(f"🔄 Filtered {filtered_chars} non-BMP characters")
                
                # Clear any existing content first
                try:
                    self.browser.driver.execute_script("arguments[0].innerHTML = '';", text_area)
                    time.sleep(0.5)
                except:
                    pass  # If clearing fails, continue anyway
                
                # Input the text using multiple methods for reliability
                input_success = False
                
                # METHOD 1: Use send_keys
                try:
                    text_area.send_keys(safe_tweet_text)
                    time.sleep(2)
                    
                    # Verify content was input
                    current_content = text_area.get_attribute('textContent') or text_area.text
                    if len(current_content.strip()) > 0:
                        logger.logger.debug(f"✅ Content input successful via send_keys")
                        input_success = True
                        
                except Exception as e:
                    logger.logger.debug(f"⚠️ send_keys method failed: {str(e)}")
                
                # METHOD 2: Use JavaScript if send_keys failed
                if not input_success:
                    try:
                        js_script = f"""
                        arguments[0].focus();
                        arguments[0].textContent = arguments[1];
                        arguments[0].dispatchEvent(new Event('input', {{bubbles: true}}));
                        arguments[0].dispatchEvent(new Event('change', {{bubbles: true}}));
                        """
                        self.browser.driver.execute_script(js_script, text_area, safe_tweet_text)
                        time.sleep(2)
                        
                        # Verify content
                        current_content = text_area.get_attribute('textContent') or text_area.text
                        if len(current_content.strip()) > 0:
                            logger.logger.debug(f"✅ Content input successful via JavaScript")
                            input_success = True
                            
                    except Exception as e:
                        logger.logger.debug(f"⚠️ JavaScript method failed: {str(e)}")
                
                if not input_success:
                    error_msg = f"Failed to input content with all methods"
                    retry_reasons.append(f"Attempt {attempt_number}: {error_msg}")
                    logger.logger.warning(f"⚠️ RETRY REASON: {error_msg}")
                    raise Exception(error_msg)
                
                # ================================================================
                # 🔘 POST BUTTON DETECTION AND INTERACTION
                # ================================================================
                
                logger.logger.debug(f"🔍 STEP 4: Locating and clicking Post button")
                
                post_button = None
                button_found = False
                
                # Enhanced button locator strategies
                button_strategies = [
                    # Strategy 1: Standard testid
                    (By.CSS_SELECTOR, '[data-testid="tweetButton"]'),
                    # Strategy 2: Button with Post text
                    (By.XPATH, "//button[.//span[text()='Post']]"),
                    # Strategy 3: Any button with "Post" text
                    (By.XPATH, "//button[contains(text(), 'Post')]"),
                    # Strategy 4: Role-based button near compose
                    (By.CSS_SELECTOR, 'button[role="button"]'),
                    # Strategy 5: Generic button that might be Post
                    (By.XPATH, "//div[@role='button'][contains(., 'Post')]")
                ]
                
                for i, (by, selector) in enumerate(button_strategies, 1):
                    try:
                        logger.logger.debug(f"🔍 POST BUTTON STRATEGY {i}: {selector}")
                        
                        # Look for the button
                        potential_buttons = self.browser.driver.find_elements(by, selector)
                        
                        for button in potential_buttons:
                            # Check if button is visible and clickable
                            if button.is_displayed() and button.is_enabled():
                                # Additional check for disabled state via attributes
                                disabled = button.get_attribute('disabled')
                                aria_disabled = button.get_attribute('aria-disabled')
                                
                                if not disabled and aria_disabled != 'true':
                                    post_button = button
                                    button_found = True
                                    logger.logger.debug(f"✅ Found active Post button with strategy {i}")
                                    break
                        
                        if button_found:
                            break
                            
                    except Exception as e:
                        logger.logger.debug(f"❌ Strategy {i} error: {str(e)}")
                        continue
                
                if not button_found or not post_button:
                    error_msg = f"Post button not found or not clickable"
                    retry_reasons.append(f"Attempt {attempt_number}: {error_msg}")
                    logger.logger.warning(f"⚠️ RETRY REASON: {error_msg}")
                    raise Exception(error_msg)
                
                # ================================================================
                # 🎯 CLICK POST BUTTON
                # ================================================================
                
                logger.logger.debug(f"🔍 STEP 5: Clicking Post button")
                
                try:
                    # Scroll button into view
                    self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", post_button)
                    time.sleep(1)
                    
                    # Click using JavaScript for reliability
                    self.browser.driver.execute_script("arguments[0].click();", post_button)
                    time.sleep(5)  # Wait for post to complete
                    
                    logger.logger.debug(f"✅ Post button clicked successfully")
                    
                except Exception as e:
                    error_msg = f"Failed to click Post button: {str(e)}"
                    retry_reasons.append(f"Attempt {attempt_number}: {error_msg}")
                    logger.logger.warning(f"⚠️ RETRY REASON: {error_msg}")
                    raise Exception(error_msg)
                
                # ================================================================
                # ⏰ TIMEFRAME SCHEDULING UPDATE
                # ================================================================
                
                logger.logger.debug(f"🔍 STEP 6: Updating timeframe scheduling")
                
                # Update last post time for this timeframe
                current_time = strip_timezone(datetime.now())
                previous_time = self.timeframe_last_post.get(timeframe, "Never")
                self.timeframe_last_post[timeframe] = current_time
                
                # Update next scheduled post time with jitter
                hours_to_add = self.timeframe_posting_frequency.get(timeframe, 1)
                jitter = random.uniform(0.8, 1.2)  # 0.8-1.2 multiplier for randomness
                actual_hours = hours_to_add * jitter
                next_post_time = strip_timezone(current_time + timedelta(hours=actual_hours))
                self.next_scheduled_posts[timeframe] = next_post_time
                
                logger.logger.debug(f"📅 SCHEDULING UPDATE:")
                logger.logger.debug(f"   • Previous: {previous_time}")
                logger.logger.debug(f"   • Current: {current_time}")
                logger.logger.debug(f"   • Next: {next_post_time}")
                logger.logger.debug(f"   • Hours delay: {actual_hours:.2f} (jitter: {jitter:.3f})")
                
                # ================================================================
                # ✅ SUCCESS LOGGING AND RETURN
                # ================================================================
                
                logger.logger.info(f"✅ POST SUCCESS: {timeframe} tweet posted successfully")
                logger.logger.debug(f"📊 SUCCESS STATS:")
                logger.logger.debug(f"   • Attempt: {attempt_number}/{max_retries}")
                logger.logger.debug(f"   • Content length: {len(safe_tweet_text)} chars")
                logger.logger.debug(f"   • Total time: ~{10 + (attempt_number-1)*12} seconds")
                
                return True
                
            except Exception as e:
                # ================================================================
                # ❌ ERROR HANDLING AND RETRY LOGIC
                # ================================================================
                
                error_message = str(e)
                retry_reasons.append(f"Attempt {attempt_number}: {error_message}")
                
                logger.logger.error(f"❌ POST ATTEMPT {attempt_number} FAILED: {error_message}")
                logger.logger.debug(f"🔍 ERROR CONTEXT:")
                logger.logger.debug(f"   • Current URL: {self.browser.driver.current_url}")
                logger.logger.debug(f"   • Page title: {self.browser.driver.title}")
                logger.logger.debug(f"   • Content: '{tweet_text[:30]}...'")
                
                retry_count += 1
                
                if retry_count < max_retries:
                    wait_time = retry_count * 10  # 10s, 20s, 30s
                    logger.logger.warning(f"🔄 PREPARING RETRY {retry_count + 1}/{max_retries}")
                    logger.logger.warning(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    break
        
        # ================================================================
        # 💥 FINAL FAILURE LOGGING
        # ================================================================
        
        logger.logger.error(f"💥 POST COMPLETELY FAILED: {timeframe}")
        logger.logger.error(f"📋 FAILURE SUMMARY:")
        logger.logger.error(f"   • Total attempts: {max_retries}")
        logger.logger.error(f"   • Content length: {len(tweet_text)} chars")
        logger.logger.error(f"   • Final URL: {self.browser.driver.current_url}")
        
        logger.logger.error(f"🔍 COMPLETE RETRY HISTORY:")
        for i, reason in enumerate(retry_reasons, 1):
            logger.logger.error(f"   {i}. {reason}")
        
        logger.log_error(f"Tweet Creation - {timeframe}", f"All strategies failed. Reasons: {'; '.join(retry_reasons)}")
        
        return False
   
    @ensure_naive_datetimes
    def _get_last_posts(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get last N posts from timeline with timeframe detection
    
        Args:
            count: Number of posts to retrieve
            
        Returns:
            List of post information including detected timeframe
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Check if browser and driver are properly initialized
                if not self.browser or not hasattr(self.browser, 'driver') or self.browser.driver is None:
                    logger.logger.error("Browser or driver not initialized for timeline scraping")
                    return []
            
                self.browser.driver.get(f'https://twitter.com/{self.config.TWITTER_USERNAME}')
                time.sleep(3)
        
                # Use explicit waits to ensure elements are loaded
                WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetText"]'))
                )
        
                posts = self.browser.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweetText"]')
        
                # Use an explicit wait for timestamps too
                WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'time'))
                )
        
                timestamps = self.browser.driver.find_elements(By.CSS_SELECTOR, 'time')
        
                # Get only the first count posts
                posts = posts[:count]
                timestamps = timestamps[:count]
        
                result = []
                for i in range(min(len(posts), len(timestamps))):
                    try:
                        post_text = posts[i].text
                        timestamp_str = timestamps[i].get_attribute('datetime') if timestamps[i].get_attribute('datetime') else None
                
                        # Detect timeframe from post content
                        detected_timeframe = "1h"  # Default
                
                        # Look for timeframe indicators in the post
                        if "7D PREDICTION" in post_text.upper() or "7-DAY" in post_text.upper() or "WEEKLY" in post_text.upper():
                            detected_timeframe = "7d"
                        elif "24H PREDICTION" in post_text.upper() or "24-HOUR" in post_text.upper() or "DAILY" in post_text.upper():
                            detected_timeframe = "24h"
                        elif "1H PREDICTION" in post_text.upper() or "1-HOUR" in post_text.upper() or "HOURLY" in post_text.upper():
                            detected_timeframe = "1h"
                
                        post_info = {
                            'text': post_text,
                            'timestamp': strip_timezone(datetime.fromisoformat(timestamp_str)) if timestamp_str else None,
                            'timeframe': detected_timeframe
                        }
                
                        result.append(post_info)
                    except Exception as element_error:
                        # Skip this element if it's stale or otherwise problematic
                        logger.logger.debug(f"Element error while extracting post {i}: {str(element_error)}")
                        continue
        
                return result
            
            except Exception as e:
                retry_count += 1
                logger.logger.warning(f"Error getting last posts (attempt {retry_count}/{max_retries}): {str(e)}")
                time.sleep(2)  # Add a small delay before retry
        
        # If all retries failed, log the error and return an empty list
        logger.log_error("Get Last Posts", f"Maximum retries ({max_retries}) reached")
        return []

    def _get_last_posts_by_timeframe(self, timeframe: str = "1h", count: int = 5) -> List[str]:
        """
        Get last N posts for a specific timeframe
        
        Args:
            timeframe: Timeframe to filter for
            count: Number of posts to retrieve
            
        Returns:
            List of post text content
        """
        all_posts = self._get_last_posts(count=20)  # Get more posts to filter from
        
        # Filter posts by the requested timeframe
        filtered_posts = [post['text'] for post in all_posts if post['timeframe'] == timeframe]
        
        # Return the requested number of posts
        return filtered_posts[:count]

    @ensure_naive_datetimes
    def _schedule_timeframe_post(self, timeframe: str, delay_hours: Optional[float] = None) -> None:
        """
        Schedule the next post for a specific timeframe
    
        Args:
            timeframe: Timeframe to schedule for
            delay_hours: Optional override for delay hours (otherwise uses default frequency)
        """
        if delay_hours is None:
            # Use default frequency with some randomness
            base_hours = self.timeframe_posting_frequency.get(timeframe, 1)
            delay_hours = base_hours * random.uniform(0.9, 1.1)
    
        self.next_scheduled_posts[timeframe] = strip_timezone(datetime.now() + timedelta(hours=delay_hours))
        logger.logger.debug(f"Scheduled next {timeframe} post for {self.next_scheduled_posts[timeframe]}")
   
    @ensure_naive_datetimes
    def _should_post_timeframe_now(self, timeframe: str) -> bool:
        """
        Check if it's time to post for a specific timeframe
        
        Args:
            timeframe: Timeframe to check
            
        Returns:
            Boolean indicating if it's time to post
        """
        try:
            # Debug
            logger.logger.debug(f"Checking if should post for {timeframe}")
            logger.logger.debug(f"  Last post: {self.timeframe_last_post.get(timeframe)} ({type(self.timeframe_last_post.get(timeframe))})")
            logger.logger.debug(f"  Next scheduled: {self.next_scheduled_posts.get(timeframe)} ({type(self.next_scheduled_posts.get(timeframe))})")
        
            # Check if enough time has passed since last post
            min_interval = timedelta(hours=self.timeframe_posting_frequency.get(timeframe, 1) * 0.8)
            last_post_time = self._ensure_datetime(self.timeframe_last_post.get(timeframe, datetime.min))
            logger.logger.debug(f"  Last post time (after ensure): {last_post_time} ({type(last_post_time)})")
        
            now = datetime.now()
            time_since_last = safe_datetime_diff(now, last_post_time) / 3600  # Hours
            
            if time_since_last < min_interval.total_seconds() / 3600:
                hours_remaining = min_interval.total_seconds() / 3600 - time_since_last
                logger.logger.info(f"⏱️ PREDICTION SKIPPED - {timeframe}: Too soon since last post. Need to wait {hours_remaining:.2f} more hours")
                return False
            
            # Check if scheduled time has been reached
            next_scheduled = self._ensure_datetime(self.next_scheduled_posts.get(timeframe, now))
            logger.logger.debug(f"  Next scheduled (after ensure): {next_scheduled} ({type(next_scheduled)})")
        
            if now < next_scheduled:
                wait_time = safe_datetime_diff(next_scheduled, now) / 3600  # Hours
                logger.logger.info(f"⏱️ PREDICTION SKIPPED - {timeframe}: Not yet scheduled. Waiting {wait_time:.2f} more hours until {next_scheduled}")
                return False
                
            logger.logger.info(f"✅ PREDICTION ALLOWED - {timeframe}: Time since last post: {time_since_last:.2f}h, Min interval: {min_interval.total_seconds()/3600:.2f}h")
            return True
        except Exception as e:
            logger.logger.error(f"Error in _should_post_timeframe_now for {timeframe}: {str(e)}")
            # Provide a safe default
            return False
   
    @ensure_naive_datetimes
    def _post_prediction_for_timeframe(self, token: str, market_data: Dict[str, Any], timeframe: str) -> bool:
        """
        Post a prediction for a specific timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for the prediction
            
        Returns:
            Boolean indicating if posting succeeded
        """
        try:
            logger.logger.info(f"🎯 PREDICTION ATTEMPT - Starting prediction process for {token} ({timeframe})")
            
            # Check if we have a prediction
            prediction = self.timeframe_predictions.get(timeframe, {}).get(token)
            
            # Log whether we're using cached prediction or generating new one
            if prediction:
                logger.logger.info(f"📊 Using cached prediction for {token} ({timeframe})")
            else:
                logger.logger.info(f"🔄 Generating new prediction for {token} ({timeframe})")
                
                # Check if market_data contains the token
                if token not in market_data:
                    logger.logger.error(f"❌ PREDICTION FAILED - {token} not found in market data")
                    return False
                    
                # Check if prediction engine is initialized
                if not hasattr(self, 'prediction_engine') or self.prediction_engine is None:
                    logger.logger.error(f"❌ PREDICTION FAILED - Prediction engine not initialized")
                    return False
                
                # Generate prediction
                prediction = self.prediction_engine._generate_predictions(
                    token=token,
                    market_data=market_data,
                    timeframe=timeframe
                )
                
                # Log prediction result
                if not prediction:
                    logger.logger.error(f"❌ PREDICTION FAILED - Prediction engine returned empty result for {token}")
                    return False
                    
                logger.logger.info(f"✅ Successfully generated prediction for {token}: {prediction.get('direction', 'UNKNOWN')} {prediction.get('confidence', 0):.2f}")
            
                # Store for future use
                if timeframe not in self.timeframe_predictions:
                    self.timeframe_predictions[timeframe] = {}
                self.timeframe_predictions[timeframe][token] = prediction
            
            # Format the prediction for posting
            tweet_text = self._format_prediction_tweet(token, prediction, market_data, timeframe)
            logger.logger.debug(f"📝 Formatted prediction tweet: {tweet_text[:50]}... ({len(tweet_text)} chars)")
            
            # Check for duplicates - make sure we're handling datetime properly
            last_posts = self._get_last_posts_by_timeframe(timeframe=timeframe)
            logger.logger.debug(f"📜 Found {len(last_posts)} previous posts for {timeframe} timeframe")
            
            # Ensure datetime compatibility in duplicate check
            if self._is_duplicate_analysis(tweet_text, last_posts, timeframe):
                logger.logger.warning(f"🔄 PREDICTION SKIPPED - Duplicate {timeframe} prediction content for {token}")
                return False
            
            # Post the prediction
            logger.logger.info(f"📣 Attempting to post {timeframe} prediction for {token}")
            if self._post_analysis(tweet_text, timeframe):
                # Store in database
                sentiment = prediction.get("sentiment", "NEUTRAL")
                if token in market_data:
                    price_data = {token: {'price': market_data[token]['current_price'], 
                                        'volume': self._safe_get_volume(market_data[token])}}
                    logger.logger.debug(f"💰 Current price: {market_data[token]['current_price']}, Volume: {self._safe_get_volume(market_data[token])}")
                else:
                    # Handle token no longer in market data
                    logger.logger.error(f"❌ PREDICTION FAILED - Token {token} no longer in market data during posting")
                    return False
            
                # Create storage data
                storage_data = {
                    'content': tweet_text,
                    'sentiment': {token: sentiment},
                    'trigger_type': f"scheduled_{timeframe}_post",
                    'price_data': price_data,
                    'meme_phrases': {token: ""},  # No meme phrases for predictions
                    'is_prediction': True,
                    'prediction_data': prediction,
                    'timeframe': timeframe
                }
            
                # Store in database
                try:
                    self.db.store_posted_content(**storage_data)
                    logger.logger.info(f"💾 Successfully stored prediction in database")
                except Exception as db_err:
                    logger.logger.error(f"⚠️ Failed to store prediction in database: {str(db_err)}")
                    # Continue even if database storage fails
            
                # Update last post time for this timeframe with current datetime
                now = strip_timezone(datetime.now())
                previous_time = self.timeframe_last_post.get(timeframe)
                self.timeframe_last_post[timeframe] = now
                
                # Update next scheduled time
                hours_to_add = self.timeframe_posting_frequency.get(timeframe, 1)
                next_scheduled = now + timedelta(hours=hours_to_add)
                self.next_scheduled_posts[timeframe] = next_scheduled
                
                logger.logger.info(f"⏰ Updated timeframe scheduling - Previous: {previous_time}, New: {now}, Next: {next_scheduled}")
                logger.logger.info(f"✅ PREDICTION SUCCESS - Posted {timeframe} prediction for {token}")
                return True
            else:
                logger.logger.error(f"❌ PREDICTION FAILED - Could not post {timeframe} prediction for {token}")
                return False
            
        except Exception as e:
            logger.log_error(f"❌ PREDICTION FAILED - Error in post_prediction_for_timeframe for {token} ({timeframe})", str(e))
            # Log traceback for debugging
            import traceback
            logger.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
   
    @ensure_naive_datetimes
    def _post_timeframe_rotation(self, market_data: Dict[str, Any]) -> bool:
        """
        Post predictions in a rotation across timeframes with enhanced token selection
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating if a post was made
        """
        # Debug timeframe scheduling data
        logger.logger.debug("TIMEFRAME ROTATION DEBUG:")
        for tf in self.timeframes:
            try:
                now = strip_timezone(datetime.now())
                last_post_time = strip_timezone(self._ensure_datetime(self.timeframe_last_post.get(tf)))
                next_scheduled_time = strip_timezone(self._ensure_datetime(self.next_scheduled_posts.get(tf)))
            
                time_since_last = safe_datetime_diff(now, last_post_time) / 3600
                time_until_next = safe_datetime_diff(next_scheduled_time, now) / 3600
                logger.logger.debug(f"{tf}: {time_since_last:.1f}h since last post, {time_until_next:.1f}h until next")
            except Exception as e:
                logger.logger.error(f"Error calculating timeframe timing for {tf}: {str(e)}")
        
        # First check if any timeframe is due for posting
        due_timeframes = [tf for tf in self.timeframes if self._should_post_timeframe_now(tf)]

        if not due_timeframes:
            logger.logger.debug("No timeframes due for posting")
            return False
    
        try:
            # Pick the most overdue timeframe
            now = strip_timezone(datetime.now())
        
            chosen_timeframe = None
            max_overdue_time = timedelta(0)
        
            for tf in due_timeframes:
                next_scheduled = strip_timezone(self._ensure_datetime(self.next_scheduled_posts.get(tf, datetime.min)))
                overdue_time = safe_datetime_diff(now, next_scheduled)
            
                if overdue_time > max_overdue_time.total_seconds():
                    max_overdue_time = timedelta(seconds=overdue_time)
                    chosen_timeframe = tf
                
            if not chosen_timeframe:
                logger.logger.warning("Could not find most overdue timeframe, using first available")
                chosen_timeframe = due_timeframes[0]
            
        except ValueError as ve:
            if "arg is an empty sequence" in str(ve):
                logger.logger.warning("No timeframes available for rotation, rescheduling all timeframes")
                # Reschedule all timeframes with random delays
                now = strip_timezone(datetime.now())
                for tf in self.timeframes:
                    delay_hours = self.timeframe_posting_frequency.get(tf, 1) * random.uniform(0.1, 0.3)
                    self.next_scheduled_posts[tf] = now + timedelta(hours=delay_hours)
                return False
            else:
                raise  # Re-raise if it's a different ValueError
        
        logger.logger.info(f"Selected {chosen_timeframe} for timeframe rotation posting")

        # Enhanced token selection using content analysis and reply data
        token_to_post = self._select_best_token_for_timeframe(market_data, chosen_timeframe)
    
        if not token_to_post:
            logger.logger.warning(f"No suitable token found for {chosen_timeframe} timeframe")
            # Reschedule this timeframe for later
            now = strip_timezone(datetime.now())
            self._schedule_timeframe_post(chosen_timeframe, delay_hours=1)
            return False
    
        # Before posting, check if there's active community discussion about this token
        # This helps align our posts with current community interests
        try:
            # Get recent timeline posts to analyze community trends
            recent_posts = self.timeline_scraper.scrape_timeline(count=25)
            if recent_posts:
                # Filter for posts related to our selected token
                token_related_posts = [p for p in recent_posts if token_to_post.upper() in p.get('text', '').upper()]
        
                # If we found significant community discussion, give this token higher priority
                if len(token_related_posts) >= 3:
                    logger.logger.info(f"Found active community discussion about {token_to_post} ({len(token_related_posts)} recent posts)")
                    # Analyze sentiment to make our post more contextually relevant
                    sentiment_stats = {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    }
            
                    # Simple sentiment analysis of community posts
                    for post in token_related_posts:
                        analysis = self.content_analyzer.analyze_post(post)
                        sentiment = analysis.get('features', {}).get('sentiment', {}).get('label', 'neutral')
                        if sentiment in ['bullish', 'enthusiastic', 'positive']:
                            sentiment_stats['positive'] += 1
                        elif sentiment in ['bearish', 'negative', 'skeptical']:
                            sentiment_stats['negative'] += 1
                        else:
                            sentiment_stats['neutral'] += 1
            
                    # Log community sentiment
                    dominant_sentiment = max(sentiment_stats.items(), key=lambda x: x[1])[0]
                    logger.logger.info(f"Community sentiment for {token_to_post}: {dominant_sentiment} ({sentiment_stats})")
                else:
                    logger.logger.debug(f"Limited community discussion about {token_to_post} ({len(token_related_posts)} posts)")
        except Exception as e:
            logger.logger.warning(f"Error analyzing community trends: {str(e)}")
    
        # Post the prediction
        success = self._post_prediction_for_timeframe(token_to_post, market_data, chosen_timeframe)
    
        # If post failed, reschedule for later
        if not success:
            now = strip_timezone(datetime.now())
            self._schedule_timeframe_post(chosen_timeframe, delay_hours=1)
    
        return success

    def _analyze_tech_topics(self, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze tech topics for educational content generation
    
        Args:
            market_data: Optional market data for context
    
        Returns:
            Dictionary with tech topic analysis
        """
        try:
            # Get configured tech topics
            tech_topics = config.get_tech_topics()
    
            if not tech_topics:
                logger.logger.warning("No tech topics configured or enabled")
                return {'enabled': False}
        
            # Get recent tech posts from database
            tech_posts = {}
            last_tech_post = strip_timezone(datetime.now() - timedelta(days=1))  # Default fallback
    
            if self.db:
                try:
                    # Query last 24 hours of content
                    recent_posts = self.db.get_recent_posts(hours=24)
            
                    # Filter to tech-related posts
                    for post in recent_posts:
                        if 'tech_category' in post:
                            category = post['tech_category']
                            if category not in tech_posts:
                                tech_posts[category] = []
                            tech_posts[category].append(post)
                    
                            # Update last tech post time
                            post_time = strip_timezone(datetime.fromisoformat(post['timestamp']))
                            if post_time > last_tech_post:
                                last_tech_post = post_time
                except Exception as db_err:
                    logger.logger.warning(f"Error retrieving tech posts: {str(db_err)}")
            
            # Analyze topics for candidacy
            candidate_topics = []
    
            for topic in tech_topics:
                category = topic['category']
                posts_today = len(tech_posts.get(category, []))
        
                # Calculate last post for this category
                category_last_post = last_tech_post
                if category in tech_posts and tech_posts[category]:
                    category_timestamps = [
                        strip_timezone(datetime.fromisoformat(p['timestamp'])) 
                        for p in tech_posts[category]
                    ]
                    if category_timestamps:
                        category_last_post = max(category_timestamps)
        
                # Check if allowed to post about this category
                allowed = config.is_tech_post_allowed(category, category_last_post)
        
                if allowed:
                    # Prepare topic metadata
                    topic_metadata = {
                        'category': category,
                        'priority': topic['priority'],
                        'keywords': topic['keywords'][:5],  # Just first 5 for logging
                        'posts_today': posts_today,
                        'hours_since_last_post': safe_datetime_diff(datetime.now(), category_last_post) / 3600,
                        'selected_token': self._select_token_for_tech_topic(category, market_data)
                    }
            
                    # Add to candidates
                    candidate_topics.append(topic_metadata)
    
            # Order by priority and recency
            if candidate_topics:
                candidate_topics.sort(key=lambda x: (x['priority'], x['hours_since_last_post']), reverse=True)
                logger.logger.info(f"Found {len(candidate_topics)} tech topics eligible for posting")
        
                # Return analysis results
                return {
                    'enabled': True,
                    'candidate_topics': candidate_topics,
                    'tech_posts_today': sum(len(posts) for posts in tech_posts.values()),
                    'max_daily_posts': config.TECH_CONTENT_CONFIG.get('max_daily_tech_posts', 6),
                    'last_tech_post': last_tech_post
                }
            else:
                logger.logger.info("No tech topics are currently eligible for posting")
                return {
                    'enabled': True,
                    'candidate_topics': [],
                    'tech_posts_today': sum(len(posts) for posts in tech_posts.values()),
                    'max_daily_posts': config.TECH_CONTENT_CONFIG.get('max_daily_tech_posts', 6),
                    'last_tech_post': last_tech_post
                }
    
        except Exception as e:
            logger.log_error("Tech Topic Analysis", str(e))
            return {'enabled': False, 'error': str(e)}

    def _select_token_for_tech_topic(self, tech_category: str, market_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Select an appropriate token to pair with a tech topic
        Now uses database-driven token selection instead of hardcoded reference tokens
        
        Args:
            tech_category: Tech category for pairing
            market_data: Market data for context, can be None
        
        Returns:
            Selected token symbol
        """
        try:
            logger.logger.info(f"🎯 Selecting token for '{tech_category}' tech topic")
            
            # Check if database is initialized
            if not hasattr(self, 'db') or self.db is None:
                logger.logger.error("❌ Database not initialized for _select_token_for_tech_topic")
                from database import CryptoDatabase
                self.db = CryptoDatabase()
                logger.logger.info("✅ Created new database connection")
                
            if not market_data:
                # Default to popular tokens if no market data
                logger.logger.warning("No market data provided, using fallback tokens")
                return random.choice(['BTC', 'ETH', 'SOL'])
            
            # Define affinity between tech categories and tokens - will be used for weighting
            tech_token_affinity = {
                'ai': ['ETH', 'SOL', 'DOT'],          # Smart contract platforms
                'quantum': ['BTC', 'XRP', 'AVAX'],    # Security-focused or scaling
                'blockchain_tech': ['ETH', 'SOL', 'BNB', 'NEAR'],  # Advanced platforms
                'advanced_computing': ['SOL', 'AVAX', 'DOT']  # High performance chains
            }
            
            # Get database-driven tokens instead of hardcoded reference tokens
            try:
                # Get top tokens by market cap
                database_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=50)
                logger.logger.info(f"📊 Database returned {len(database_tokens)} tokens: {database_tokens}")
                
                # Get affinity tokens for this category if defined
                affinity_tokens = tech_token_affinity.get(tech_category, [])
                
                # Combine database tokens and affinity tokens for selection pool
                selection_pool = list(set(database_tokens + affinity_tokens))
                
                # Filter to tokens with available market data
                available_tokens = [t for t in selection_pool if t in market_data]
                logger.logger.info(f"🔍 {len(available_tokens)} tokens available in market data: {available_tokens}")
                
                if not available_tokens:
                    logger.logger.warning("⚠️ No database tokens found in market_data, using fallback")
                    return random.choice(['BTC', 'ETH', 'SOL'])
            
            except Exception as db_error:
                logger.logger.error(f"❌ Database query failed: {str(db_error)}")
                # Fallback to simple reference tokens if database query fails
                available_tokens = [t for t in ['BTC', 'ETH', 'SOL', 'XRP', 'BNB'] if t in market_data]
                if not available_tokens:
                    return random.choice(['BTC', 'ETH', 'SOL'])
        
            # Select token with interesting market movement if possible
            interesting_tokens = []
            for token in available_tokens:
                price_change = abs(market_data[token].get('price_change_percentage_24h', 0))
                if price_change > 5.0:  # >5% change is interesting
                    interesting_tokens.append(token)
            
            # Use interesting tokens if available, otherwise use all available tokens
            final_pool = interesting_tokens if interesting_tokens else available_tokens
            
            # Select a token, weighting by market cap if possible
            if len(final_pool) > 1:
                # Extract market caps safely
                market_caps = {}
                for t in final_pool:
                    market_cap = market_data[t].get('market_cap', None)
                    if market_cap is not None:
                        try:
                            market_caps[t] = float(market_cap)
                        except (ValueError, TypeError):
                            market_caps[t] = 1.0
                    else:
                        market_caps[t] = 1.0
                        
                # Add affinity bonus for tokens that match the tech category
                for t in final_pool:
                    if t in tech_token_affinity.get(tech_category, []):
                        market_caps[t] *= 1.5  # 50% bonus for affinity tokens
                    
                # Create weighted probability
                total_cap = sum(market_caps.values())
                weights = [market_caps[t]/total_cap for t in final_pool]
                
                # Select with weights
                selected_token = random.choices(final_pool, weights=weights, k=1)[0]
                logger.logger.info(f"✅ Selected {selected_token} for {tech_category} (weighted selection)")
                return selected_token
            else:
                # Just one token available
                logger.logger.info(f"✅ Selected {final_pool[0]} for {tech_category} (only available token)")
                return final_pool[0]
        
        except Exception as e:
            logger.log_error("Token Selection for Tech Topic", str(e))
            # Safe fallback
            logger.logger.error(f"❌ Error selecting token: {str(e)}")
            return random.choice(['BTC', 'ETH', 'SOL'])

    def _generate_tech_content(self, tech_category: str, token: str, market_data: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate educational tech content for posting with enhanced diversity and natural language
    
        Args:
            tech_category: Tech category to focus on
            token: Token to relate to tech content
            market_data: Market data for context
    
        Returns:
            Tuple of (content_text, metadata)
        """
        try:
            logger.logger.info(f"Generating tech content for {tech_category} related to {token}")

            # Get token data if available
            token_data = {}
            if market_data and token in market_data:
                token_data = market_data[token]
        
            # ====== ENHANCEMENT 1: INCREASED CONTENT DIVERSITY ======
        
            # Determine content type with more variety - 5 types instead of just 2
            content_types = [
                "educational", 
                "integration", 
                "opinion", 
                "news_analysis", 
                "future_prediction"
            ]
        
            # Weights to make some content types more common than others
            type_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            content_type = random.choices(content_types, weights=type_weights, k=1)[0]
        
            # Track last content type for this category to avoid repetition
            diversity_key = f"{tech_category}_{token}"
            last_content_types = getattr(self, '_last_content_types', {})
        
            # Avoid repeating the same content type for the same category-token pair
            retry_count = 0
            while diversity_key in last_content_types and last_content_types[diversity_key] == content_type and retry_count < 3:
                content_type = random.choices(content_types, weights=type_weights, k=1)[0]
                retry_count += 1
        
            # Store this content type for future diversity checking
            if not hasattr(self, '_last_content_types'):
                self._last_content_types = {}
            self._last_content_types[diversity_key] = content_type
        
            # ====== ENHANCEMENT 2: AUDIENCE VARIETY ======
        
            # Select appropriate audience level with more nuance
            audience_levels = ['beginner', 'intermediate', 'advanced', 'expert', 'mixed']
            audience_weights = [0.2, 0.4, 0.2, 0.1, 0.1]  # Intermediate most common but good variety
            audience_level = random.choices(audience_levels, weights=audience_weights, k=1)[0]
        
            # ====== ENHANCEMENT 3: TONE VARIATION ======
        
            # Add tone variation for more natural language
            tones = [
                'conversational', 'analytical', 'enthusiastic', 'thoughtful', 
                'curious', 'speculative', 'cautious', 'balanced', 'candid'
            ]
            tone = random.choice(tones)
        
            # ====== ENHANCEMENT 4: PERSONALIZATION ELEMENTS ======
        
            # Add personalization elements for human-like content
            personalization_elements = [
                "personal_experience", "question", "comparison", "anecdote", 
                "disagreement", "surprise", "recent_insight", "none"
            ]
            personalization = random.choice(personalization_elements)
        
            # ====== ENHANCEMENT 5: DYNAMIC CONTENT STRUCTURE ======
        
            # Vary content structure based on type
            if content_type == "educational":
                structure = random.choice(["explanation", "walkthrough", "concept_breakdown", "analogy_based"])
            elif content_type == "integration":
                structure = random.choice(["use_case", "implementation", "benefits_analysis", "roadmap", "technology_stack"])
            elif content_type == "opinion":
                structure = random.choice(["argument", "critique", "endorsement", "balanced_view", "hot_take"])
            elif content_type == "news_analysis":
                structure = random.choice(["recap", "implications", "stakeholder_impact", "market_effects"])
            else:  # future_prediction
                structure = random.choice(["timeline", "scenario_analysis", "trends", "disruption_potential"])
            
            # ====== ENHANCEMENT 6: RELEVANT CONTEXT INTEGRATION ======
        
            # Market context makes the content more timely and relevant
            market_context = ""
            if token_data:
                price = token_data.get('current_price', 0)
                change = token_data.get('price_change_percentage_24h', 0)
            
                if abs(change) > 5:
                    # Significant price movement to reference
                    direction = "upward" if change > 0 else "downward"
                    market_context = f"With {token}'s recent {direction} price movement of {abs(change):.1f}%, "
                elif 'market_cap' in token_data and token_data['market_cap'] > 0:
                    market_cap = token_data['market_cap']
                    market_cap_str = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.1f}M"
                    market_context = f"As a {market_cap_str} market cap cryptocurrency, "
        
            # ====== ENHANCEMENT 7: TECH STATUS FOR ACCURACY ======
        
            # Get tech status for accuracy and timeliness
            tech_status = self._get_tech_status_summary(tech_category)
        
            # ====== ENHANCEMENT 8: TOPIC-SPECIFIC VARIATION ======
        
            # Prepare specific variations based on topic combination
            topic_variations = {
                "ai": {
                    "BTC": ["mining optimization", "market prediction", "security protocols"],
                    "ETH": ["smart contract optimization", "network efficiency", "governance models"],
                    "SOL": ["transaction throughput enhancement", "parallel processing", "validation algorithms"],
                    "default": ["trading algorithms", "pattern recognition", "fraud detection"]
                },
                "quantum": {
                    "BTC": ["cryptography vulnerability", "hash function resilience", "post-quantum security"],
                    "ETH": ["quantum-resistant algorithms", "zero-knowledge proofs", "computational advantage"],
                    "default": ["security implications", "encryption standards", "computational speedups"]
                },
                "blockchain_tech": {
                    "ETH": ["scalability solutions", "proof-of-stake evolution", "sharding implementation"],
                    "SOL": ["consensus mechanism", "validator networks", "storage optimization"],
                    "default": ["cross-chain compatibility", "protocol development", "decentralized applications"]
                },
                "advanced_computing": {
                    "default": ["processing optimization", "node infrastructure", "network architecture"]
                }
            }
        
            # Get variation options for this category/token combination
            variation_options = topic_variations.get(tech_category, {}).get(token, 
                               topic_variations.get(tech_category, {}).get("default", ["innovation", "development", "integration"]))
        
            # Select a specific variation
            specific_variation = random.choice(variation_options)
        
            # ====== CONTENT GENERATION PROMPT ASSEMBLY ======
        
            # Build a more varied prompt based on all the dynamic elements
            prompt_elements = {
                "tech_topic": tech_category.replace('_', ' ').title(),
                "token": token,
                "audience_level": audience_level,
                "content_type": content_type,
                "tone": tone,
                "personalization": personalization,
                "structure": structure,
                "market_context": market_context,
                "tech_status": tech_status,
                "specific_variation": specific_variation,
                "min_length": config.TWEET_CONSTRAINTS['MIN_LENGTH'],
                "max_length": config.TWEET_CONSTRAINTS['MAX_LENGTH']
            }
        
            # ====== BUILD DYNAMIC PROMPT ======
        
            # Base prompt varies by content type
            if content_type == "educational":
                base_prompt = """
                Create educational content about {tech_topic} that's both informative and engaging.
            
                Generate content that:
                1. Sounds like a real person writing about {tech_topic} in relation to {token}
                2. Uses a {tone} tone that feels authentic and not formulaic
                3. Focuses specifically on {specific_variation} within this technology space
                4. Is accessible to a {audience_level} audience
                5. {market_context}acknowledges the current state of this technology: {tech_status}
                6. Uses a {structure} approach to explaining the concepts
                """
            
                # Add personalization element if applicable
                if personalization == "personal_experience":
                    base_prompt += "7. Includes a believable personal perspective or experience with this technology"
                elif personalization == "question":
                    base_prompt += "7. Incorporates thoughtful questions that engage the reader"
                elif personalization == "comparison":
                    base_prompt += "7. Makes an interesting comparison to help explain the concept"
                elif personalization == "anecdote":
                    base_prompt += "7. Includes a brief, relevant anecdote about real-world application"
                elif personalization == "disagreement":
                    base_prompt += "7. Includes a subtle counterpoint or area where experts might disagree"
                elif personalization == "surprise":
                    base_prompt += "7. Mentions a surprising or counterintuitive aspect of this technology"
                elif personalization == "recent_insight":
                    base_prompt += "7. References a recent development or insight in this field"
            
            elif content_type == "integration":
                base_prompt = """
                Explore the relationship between {token} and {tech_topic} technology.
            
                Generate content that:
                1. Sounds like a knowledgeable crypto enthusiast discussing technology integration
                2. Uses a {tone} tone throughout the discussion
                3. Specifically addresses {specific_variation} as it relates to {token}
                4. Is framed for a {audience_level} audience
                5. {market_context}incorporates awareness of: {tech_status}
                6. Uses a {structure} approach to exploring the integration
                """
            
                # Add personalization element if applicable
                if personalization == "personal_experience":
                    base_prompt += "7. Includes a personal take on this integration's potential"
                elif personalization == "question":
                    base_prompt += "7. Poses thought-provoking questions about implementation challenges"
                elif personalization == "comparison":
                    base_prompt += "7. Compares this integration with similar approaches in other cryptocurrencies"
                elif personalization == "anecdote":
                    base_prompt += "7. References a specific project or initiative in this area"
                elif personalization == "disagreement":
                    base_prompt += "7. Acknowledges competing perspectives on this integration"
                elif personalization == "surprise":
                    base_prompt += "7. Highlights a non-obvious benefit or challenge of this integration"
                elif personalization == "recent_insight":
                    base_prompt += "7. Mentions recent progress in this integration area"
            
            elif content_type == "opinion":
                base_prompt = """
                Share insights about {tech_topic} and its relationship to {token}.
            
                Generate content that:
                1. Reads like a personal opinion from someone knowledgeable about both technologies
                2. Maintains a {tone} tone while expressing clear viewpoints
                3. Specifically addresses perspectives on {specific_variation}
                4. Is appropriate for readers with {audience_level} knowledge
                5. {market_context}acknowledges the current technology status: {tech_status}
                6. Uses a {structure} approach to presenting the opinion
                """
            
                # Add personalization element if applicable
                if personalization == "personal_experience":
                    base_prompt += "7. Includes a first-person perspective on this technology's value"
                elif personalization == "question":
                    base_prompt += "7. Asks readers to consider key questions about the future direction"
                elif personalization == "comparison":
                    base_prompt += "7. Contrasts this view with common alternative perspectives"
                elif personalization == "anecdote":
                    base_prompt += "7. Shares a brief story illustrating a key point in the opinion"
                elif personalization == "disagreement":
                    base_prompt += "7. Respectfully disagrees with a common narrative in this space"
                elif personalization == "surprise":
                    base_prompt += "7. Takes a surprising or contrarian position on some aspect"
                elif personalization == "recent_insight":
                    base_prompt += "7. Bases opinion on a recent development or announcement"
            
            elif content_type == "news_analysis":
                base_prompt = """
                Analyze recent developments in {tech_topic} that could impact {token}.
            
                Generate content that:
                1. Sounds like timely analysis from someone following both technologies closely
                2. Uses a {tone} tone in discussing developments and implications
                3. Specifically examines {specific_variation} from a news analysis perspective
                4. Is tailored for readers with {audience_level} understanding
                5. {market_context}relates to the current state of this technology: {tech_status}
                6. Uses a {structure} approach to the analysis
                """
            
                # Add personalization element if applicable
                if personalization == "personal_experience":
                    base_prompt += "7. Includes personal reaction to these developments"
                elif personalization == "question":
                    base_prompt += "7. Raises questions about what these developments mean for investors"
                elif personalization == "comparison":
                    base_prompt += "7. Compares these developments to historical precedents"
                elif personalization == "anecdote":
                    base_prompt += "7. Includes a specific example that illustrates the impact"
                elif personalization == "disagreement":
                    base_prompt += "7. Notes where this analysis differs from mainstream coverage"
                elif personalization == "surprise":
                    base_prompt += "7. Highlights an overlooked aspect of these developments"
                elif personalization == "recent_insight":
                    base_prompt += "7. Connects multiple recent developments into a coherent narrative"
            
            else:  # future_prediction
                base_prompt = """
                Predict how {tech_topic} might influence the future of {token}.
            
                Generate content that:
                1. Reads like thoughtful speculation from someone deeply familiar with both fields
                2. Maintains a {tone} tone while discussing future possibilities
                3. Specifically explores the future of {specific_variation} 
                4. Is appropriate for {audience_level} understanding of these technologies
                5. {market_context}builds on the current state: {tech_status}
                6. Uses a {structure} approach to considering the future
                """
            
                # Add personalization element if applicable
                if personalization == "personal_experience":
                    base_prompt += "7. Includes personal excitement or concerns about these possibilities"
                elif personalization == "question":
                    base_prompt += "7. Poses thought-provoking questions about what might change"
                elif personalization == "comparison":
                    base_prompt += "7. Compares potential futures with current limitations"
                elif personalization == "anecdote":
                    base_prompt += "7. References an early indicator or project pointing to this future"
                elif personalization == "disagreement":
                    base_prompt += "7. Acknowledges competing predictions about this technology's impact"
                elif personalization == "surprise":
                    base_prompt += "7. Includes a bold or unexpected prediction"
                elif personalization == "recent_insight":
                    base_prompt += "7. Bases predictions on very recent developments or research"
        
            # Add standard length constraints and humanization instructions
            base_prompt += f"""
        
            8. Length should be between {prompt_elements['min_length']}-{prompt_elements['max_length']} characters
        
            Most importantly:
            - Use varied sentence structure (mix of simple, compound, and complex sentences)
            - Include natural transitions between ideas
            - Avoid formulaic or repetitive phrasings
            - Sound like a genuine human perspective, not corporate or academic content
            - Use a few conversational elements (asides, emphasis, personal views)
            - Include subtle indicators of authentic writing (slight tangents, specific examples)
            """
        
            # Process the prompt template with our variables
            prompt = base_prompt.format(**prompt_elements)
        
            # ====== GENERATE CONTENT ======
        
            # Generate content with LLM provider
            logger.logger.debug(f"Generating {tech_category} content with type={content_type}, tone={tone}, structure={structure}")
            content = self.llm_provider.generate_text(prompt, max_tokens=1000)
        
            if not content:
                raise ValueError("Failed to generate tech content")
        
            # ====== CONTENT PROCESSING ======
        
            # Ensure content meets length requirements
            content = self._format_tech_content(content)
        
            # ====== METADATA STORAGE ======
        
            # Prepare metadata with enhanced tracking for diversity
            metadata = {
                'tech_category': tech_category,
                'token': token,
                'content_type': content_type,
                'audience_level': audience_level,
                'tone': tone,
                'personalization': personalization,
                'structure': structure,
                'specific_variation': specific_variation,
                'token_data': token_data,
                'timestamp': strip_timezone(datetime.now())
            }
        
            # Measure sentence structure diversity for analytics
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            metadata['sentence_count'] = len(sentences)
            metadata['avg_sentence_length'] = sum(len(s) for s in sentences) / max(1, len(sentences))
        
            return content, metadata

        except Exception as e:
            logger.log_error("Tech Content Generation", str(e))
            # Return fallback content
            fallback_content = f"Did you know that advances in {tech_category.replace('_', ' ')} technology could significantly impact the future of {token} and the broader crypto ecosystem? The intersection of these fields is creating fascinating new possibilities."
            return fallback_content, {'tech_category': tech_category, 'token': token, 'error': str(e)}

    def _post_tech_content(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Post tech content to Twitter with proper formatting
    
        Args:
            content: Content to post
            metadata: Content metadata for database storage
        
        Returns:
            Boolean indicating if posting succeeded
        """
        try:
            # Check if content is already properly formatted
            if len(content) > config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
                content = self._format_tech_content(content)
            
            # Format as a tweet
            tweet_text = content
        
            # Add a subtle educational hashtag if there's room
            if len(tweet_text) < config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 20:
                tech_category = metadata.get('tech_category', 'technology')
                token = metadata.get('token', '')
            
                # Determine if we should add hashtags
                if random.random() < 0.7:  # 70% chance to add hashtags
                    # Potential hashtags
                    tech_tags = {
                        'ai': ['#AI', '#ArtificialIntelligence', '#MachineLearning'],
                        'quantum': ['#QuantumComputing', '#Quantum', '#QuantumTech'],
                        'blockchain_tech': ['#Blockchain', '#Web3', '#DLT'],
                        'advanced_computing': ['#Computing', '#TechInnovation', '#FutureTech']
                    }
                
                    # Get tech hashtags
                    tech_hashtags = tech_tags.get(tech_category, ['#Technology', '#Innovation'])
                
                    # Add tech hashtag and token
                    hashtag = random.choice(tech_hashtags)
                    if len(tweet_text) + len(hashtag) + len(token) + 2 <= config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
                        tweet_text = f"{tweet_text} {hashtag}"
                    
                        # Maybe add token hashtag too
                        if len(tweet_text) + len(token) + 2 <= config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
                            tweet_text = f"{tweet_text} #{token}"
        
            # Post to Twitter
            logger.logger.info(f"Posting tech content about {metadata.get('tech_category', 'tech')} and {metadata.get('token', 'crypto')}")
            if self._post_analysis(tweet_text):
                logger.logger.info("Successfully posted tech content")
            
                # Record in database
                if self.db:
                    try:
                        # Extract token price data if available
                        token = metadata.get('token', '')
                        token_data = metadata.get('token_data', {})
                        price_data = {
                            token: {
                                'price': token_data.get('current_price', 0),
                                'volume': token_data.get('volume', 0)
                            }
                        }
                    
                        # Store as content with tech category
                        self.config.db.store_posted_content(
                            content=tweet_text,
                            sentiment={},  # No sentiment for educational content
                            trigger_type=f"tech_{metadata.get('tech_category', 'general')}",
                            price_data=price_data,
                            meme_phrases={},  # No meme phrases for educational content
                            tech_category=metadata.get('tech_category', 'technology'),
                            tech_metadata=metadata,
                            is_educational=True
                        )
                    except Exception as db_err:
                        logger.logger.warning(f"Failed to store tech content: {str(db_err)}")
            
                return True
            else:
                logger.logger.warning("Failed to post tech content")
                return False
            
        except Exception as e:
            logger.log_error("Tech Content Posting", str(e))
            return False

    def _validate_and_adjust_length(self, text: str, max_length: int = 275) -> str:
        """
        Final validation and emergency adjustment of text length
        
        Args:
            text: Text to validate
            max_length: Maximum allowed length
            
        Returns:
            Validated and potentially adjusted text
        """
        if len(text) <= max_length:
            return text
        
        logger.logger.warning(f"⚠️ EMERGENCY TRUNCATION: Text {len(text)} chars > {max_length} limit")
        
        # Emergency truncation strategies
        # Strategy 1: Remove excessive whitespace and newlines
        cleaned_text = ' '.join(text.split())
        if len(cleaned_text) <= max_length:
            return cleaned_text
        
        # Strategy 2: Truncate at sentence boundary
        sentences = cleaned_text.replace('!', '.').replace('?', '.').split('.')
        result = ""
        for sentence in sentences:
            if len(result + sentence + ".") <= max_length - 3:
                result += sentence + "."
            else:
                break
        
        if len(result) > 10:  # Ensure we have meaningful content
            return result.strip()
        
        # Strategy 3: Hard truncate with ellipsis
        return cleaned_text[:max_length-3] + "..."

    def _detect_post_button_disabled(self) -> bool:
        """
        Detect if the Twitter post button is disabled due to character limit exceeded
        
        Returns:
            True if post button is disabled, False otherwise
        """
        try:
            # Check if browser and driver are properly initialized
            if not self.browser or not hasattr(self.browser, 'driver') or self.browser.driver is None:
                logger.logger.error("Browser or driver not initialized for post button detection")
                return False
            
            # Multiple detection strategies for disabled post button
            button_selectors = [
                '[data-testid="tweetButton"]',
                '//div[@role="button"][contains(., "Post")]',
                '//span[text()="Post"]'
            ]
            
            for selector in button_selectors:
                try:
                    if selector.startswith('//'):
                        # XPath selector
                        elements = self.browser.driver.find_elements(By.XPATH, selector)
                    else:
                        # CSS selector
                        elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in elements:
                        # Check if button is disabled
                        if element.get_attribute('disabled') == 'true':
                            logger.logger.warning("🚫 POST BUTTON DISABLED - Character limit exceeded")
                            return True
                        
                        # Check for disabled class or aria-disabled
                        classes = element.get_attribute('class') or ''
                        aria_disabled = element.get_attribute('aria-disabled')
                        
                        if 'disabled' in classes.lower() or aria_disabled == 'true':
                            logger.logger.warning("🚫 POST BUTTON DISABLED - Character limit exceeded")
                            return True
                        
                        # Check if button is not clickable (opacity, pointer-events)
                        style = element.get_attribute('style') or ''
                        if 'opacity: 0' in style or 'pointer-events: none' in style:
                            logger.logger.warning("🚫 POST BUTTON DISABLED - Character limit exceeded")
                            return True
                            
                except Exception as e:
                    logger.logger.debug(f"Error checking button selector {selector}: {str(e)}")
                    continue
            
            # Additional check: Look for character count indicator showing negative values
            try:
                char_indicators = self.browser.driver.find_elements(By.CSS_SELECTOR, 
                    'div[role="progressbar"], [data-testid="tweetTextarea_0_character_count"]')
                
                for indicator in char_indicators:
                    text = indicator.text.strip()
                    # Look for negative character count (like "-255")
                    if text.startswith('-') and text[1:].isdigit():
                        over_limit = int(text)
                        logger.logger.warning(f"🚫 CHARACTER LIMIT EXCEEDED by {abs(over_limit)} characters")
                        return True
                        
            except Exception as e:
                logger.logger.debug(f"Error checking character indicator: {str(e)}")
            
            return False
            
        except Exception as e:
            logger.logger.error(f"Error detecting disabled post button: {str(e)}")
            return False

    def _handle_character_limit_failure(self) -> bool:
        """
        Handle the case where character limit is exceeded and we need to discard the post
        Implements the navigation sequence: Back arrow (top left) → Discard button
        Falls back to X button → Discard button if back arrow fails
        
        Returns:
            True if successfully discarded, False otherwise
        """
        try:
            # Check if browser and driver are properly initialized
            if not self.browser or not hasattr(self.browser, 'driver') or self.browser.driver is None:
                logger.logger.error("Browser or driver not initialized for character limit failure handling")
                return False
                
            logger.logger.warning("🔄 HANDLING CHARACTER LIMIT FAILURE - Attempting to discard post")
            
            # PRIMARY STRATEGY: Try clicking the back arrow (top left) first
            logger.logger.info("🔙 ATTEMPTING PRIMARY STRATEGY: Back arrow (top left)")
            back_selectors = [
                '[data-testid="app-bar-back"]',
                '[aria-label="Back"]',
                '//button[@aria-label="Back"]',
                'button[aria-label="Back"]',
                '.css-1dbjc4n[role="button"][tabindex="0"]'  # Common Twitter back button class
            ]
            
            navigation_clicked = False
            for selector in back_selectors:
                try:
                    if selector.startswith('//'):
                        element = WebDriverWait(self.browser.driver, 3).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        element = WebDriverWait(self.browser.driver, 3).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    
                    if element:
                        self.browser.driver.execute_script("arguments[0].click();", element)
                        logger.logger.info("✅ BACK ARROW CLICKED (PRIMARY STRATEGY)")
                        time.sleep(1)
                        navigation_clicked = True
                        break
                        
                except TimeoutException:
                    continue
                except Exception as e:
                    logger.logger.debug(f"Back button selector {selector} failed: {str(e)}")
                    continue
            
            # FALLBACK STRATEGY: If back arrow didn't work, try the X button
            if not navigation_clicked:
                logger.logger.warning("⚠️ PRIMARY STRATEGY FAILED - Attempting FALLBACK: X button")
                x_selectors = [
                    '[data-testid="app-bar-close"]',
                    '[aria-label="Close"]',
                    '//button[@aria-label="Close"]',
                    'button[aria-label="Close"]',
                    '.css-18t94o4.css-1dbjc4n.r-1777fci.r-11cpok1.r-1ny4l3l'  # Twitter X button classes
                ]
                
                for selector in x_selectors:
                    try:
                        if selector.startswith('//'):
                            element = WebDriverWait(self.browser.driver, 3).until(
                                EC.element_to_be_clickable((By.XPATH, selector))
                            )
                        else:
                            element = WebDriverWait(self.browser.driver, 3).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                            )
                        
                        if element:
                            self.browser.driver.execute_script("arguments[0].click();", element)
                            logger.logger.info("✅ X BUTTON CLICKED (FALLBACK STRATEGY)")
                            time.sleep(1)
                            navigation_clicked = True
                            break
                            
                    except TimeoutException:
                        continue
                    except Exception as e:
                        logger.logger.debug(f"X button selector {selector} failed: {str(e)}")
                        continue
            
            if not navigation_clicked:
                logger.logger.error("❌ BOTH STRATEGIES FAILED - Could not click back arrow or X button")
                return False
            
            # Step 3: Wait for and click the "Discard" button
            time.sleep(2)  # Give UI time to respond
            
            discard_selectors = [
                '//span[text()="Discard"]',
                '//button[contains(., "Discard")]',
                '[data-testid="confirmationSheetConfirm"]',
                'button[data-testid="confirmationSheetConfirm"]',
                '//div[@role="button"][contains(., "Discard")]'
            ]
            
            discard_clicked = False
            for selector in discard_selectors:
                try:
                    if selector.startswith('//'):
                        element = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        element = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    
                    if element:
                        self.browser.driver.execute_script("arguments[0].click();", element)
                        logger.logger.info("✅ DISCARD BUTTON CLICKED")
                        time.sleep(2)
                        discard_clicked = True
                        break
                        
                except TimeoutException:
                    continue
                except Exception as e:
                    logger.logger.debug(f"Discard button selector {selector} failed: {str(e)}")
                    continue
            
            if discard_clicked:
                logger.logger.info("✅ SUCCESSFULLY DISCARDED POST DUE TO CHARACTER LIMIT")
                return True
            else:
                logger.logger.error("❌ FAILED to click discard button")
                return False
                
        except Exception as e:
            logger.logger.error(f"Error handling character limit failure: {str(e)}")
            return False

    def _post_analysis_with_char_limit_detection(self, tweet_text: str, timeframe: str = "1h") -> bool:
        """
        Enhanced post analysis method with character limit detection and recovery
        
        Args:
            tweet_text: Text to post
            timeframe: Timeframe for the analysis
        
        Returns:
            Boolean indicating if posting succeeded
        """
        # Check if browser and driver are properly initialized
        if not self.browser or not hasattr(self.browser, 'driver') or self.browser.driver is None:
            logger.logger.error("Browser or driver not initialized for posting analysis")
            return False
        
        # Pre-posting validation
        if len(tweet_text) > 275:
            logger.logger.warning(f"⚠️ PRE-POST VALIDATION FAILED: Tweet {len(tweet_text)} chars exceeds 275 limit")
            # Attempt emergency truncation
            tweet_text = self._validate_and_adjust_length(tweet_text, max_length=275)
            logger.logger.info(f"📏 Emergency truncated to {len(tweet_text)} characters")
        
        # Check for empty or just "neutral" content and replace it
        if not tweet_text or tweet_text.strip().lower() == "neutral":
            # Define a list of exciting crypto buzz phrases
            crypto_buzz_phrases = [
                "To The Moon!!!! 🚀🚀🚀",
                "HODL Strong! 💎🙌",
                "Big things coming! 📈",
                "Bullish AF! 🐂",
                "Buy the dip! 📉➡️📈",
                "This is the way! ✨",
                "Diamond hands only! 💎",
                "Breaking out! 📊",
                "Getting ready to pump! 💪",
                "Solid fundamentals! 👨‍💻"
            ]
            tweet_text = random.choice(crypto_buzz_phrases)
            logger.logger.info(f"🎲 Replaced neutral content with: {tweet_text}")
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Navigate to compose tweet
                self.browser.driver.get('https://twitter.com/compose/tweet')
                time.sleep(3)
                
                # Find and click text area
                text_area = WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
                )
                text_area.click()
                time.sleep(1)
            
                # Ensure tweet text only contains BMP characters
                safe_tweet_text = ''.join(char for char in tweet_text if ord(char) < 0x10000)
            
                # Send the tweet text
                text_area.send_keys(safe_tweet_text)
                time.sleep(2)
                
                # POST-INPUT VALIDATION: Check if post button is disabled
                if self._detect_post_button_disabled():
                    logger.logger.warning("🚫 POST-INPUT VALIDATION FAILED: Post button disabled")
                    
                    # Handle the character limit failure
                    if self._handle_character_limit_failure():
                        # Try with more aggressive truncation
                        if retry_count < max_retries - 1:
                            # More aggressive truncation for retry
                            original_length = len(safe_tweet_text)
                            # Reduce by 20% each retry
                            new_length = int(original_length * (0.8 ** (retry_count + 1)))
                            new_length = max(new_length, 150)  # Don't go below 150 chars
                            
                            tweet_text = self._validate_and_adjust_length(safe_tweet_text, max_length=new_length)
                            logger.logger.info(f"🔄 RETRY {retry_count + 1}: Truncated to {len(tweet_text)} chars")
                            retry_count += 1
                            time.sleep(3)
                            continue
                        else:
                            logger.logger.error("❌ MAXIMUM RETRIES REACHED - Character limit cannot be resolved")
                            return False
                    else:
                        logger.logger.error("❌ FAILED to handle character limit failure")
                        return False

                # Find post button
                post_button = None
                button_locators = [
                    (By.CSS_SELECTOR, '[data-testid="tweetButton"]'),
                    (By.XPATH, "//div[@role='button'][contains(., 'Post')]"),
                    (By.XPATH, "//span[text()='Post']")
                ]

                for locator in button_locators:
                    try:
                        post_button = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable(locator)
                        )
                        if post_button:
                            break
                    except:
                        continue

                if post_button:
                    # Final check: Ensure button is still clickable
                    if self._detect_post_button_disabled():
                        logger.logger.warning("🚫 FINAL CHECK FAILED: Post button disabled at posting time")
                        retry_count += 1
                        continue
                    
                    self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", post_button)
                    time.sleep(1)
                    self.browser.driver.execute_script("arguments[0].click();", post_button)
                    time.sleep(5)
                
                    # Update last post time for this timeframe
                    self.timeframe_last_post[timeframe] = strip_timezone(datetime.now())
                
                    # Update next scheduled post time
                    hours_to_add = self.timeframe_posting_frequency.get(timeframe, 1)
                    # Add some randomness to prevent predictable patterns
                    jitter = random.uniform(0.8, 1.2)
                    self.next_scheduled_posts[timeframe] = strip_timezone(datetime.now() + timedelta(hours=hours_to_add * jitter))
                
                    logger.logger.info(f"✅ {timeframe} tweet posted successfully")
                    logger.logger.debug(f"Next {timeframe} post scheduled for {self.next_scheduled_posts[timeframe]}")
                    return True
                else:
                    logger.logger.error(f"❌ Could not find post button for {timeframe} tweet")
                    retry_count += 1
                    time.sleep(2)
                
            except Exception as e:
                logger.logger.error(f"❌ {timeframe} tweet posting error, attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                wait_time = retry_count * 10
                logger.logger.warning(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

        logger.log_error(f"Tweet Creation - {timeframe}", "Maximum retries reached")
        return False

    def _format_tech_content(self, content: str) -> str:
        """
        Format tech content to meet tweet constraints
    
        Args:
            content: Raw content to format
        
        Returns:
            Formatted content
        """
        # Ensure length is within constraints
        if len(content) > config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
            # Find a good sentence break to truncate
            last_period = content[:config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind('.')
            last_question = content[:config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind('?')
            last_exclamation = content[:config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind('!')
        
            # Find best break point
            break_point = max(last_period, last_question, last_exclamation)
        
            if break_point > config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] * 0.7:
                # Good sentence break found
                content = content[:break_point + 1]
            else:
                # Find word boundary
                last_space = content[:config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind(' ')
                if last_space > 0:
                    content = content[:last_space] + "..."
                else:
                    # Hard truncate with ellipsis
                    content = content[:config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3] + "..."
    
        # Ensure minimum length is met
        if len(content) < config.TWEET_CONSTRAINTS['MIN_LENGTH']:
            logger.logger.warning(f"Tech content too short ({len(content)} chars). Minimum: {self.config.TWEET_CONSTRAINTS['MIN_LENGTH']}")
            # We won't try to expand too-short content
    
        return content

    def _get_tech_status_summary(self, tech_category: str) -> str:
        """
        Get a current status summary for a tech category
    
        Args:
            tech_category: Tech category
        
        Returns:
            Status summary string
        """
        # Status summaries by category
        summaries = {
            'ai': [
                "Rapid advancement in multimodal capabilities",
                "Increasing deployment in enterprise settings",
                "Rising concerns about governance and safety",
                "Growing focus on specialized models",
                "Shift toward open models and distributed research",
                "Mainstream adoption accelerating"
            ],
            'quantum': [
                "Steady progress in error correction",
                "Growing number of qubits in leading systems",
                "Early commercial applications emerging",
                "Increasing focus on quantum-resistant cryptography",
                "Major investment from government and private sectors",
                "Hardware diversity expanding beyond superconducting qubits"
            ],
            'blockchain_tech': [
                "Layer 2 solutions gaining momentum",
                "ZK-rollup technology maturing rapidly",
                "Cross-chain interoperability improving",
                "RWA tokenization expanding use cases",
                "Institutional adoption of infrastructure growing",
                "Privacy-preserving technologies advancing"
            ],
            'advanced_computing': [
                "Specialized AI hardware proliferating",
                "Edge computing deployments accelerating",
                "Neuromorphic computing showing early promise",
                "Post-Moore's Law approaches diversifying",
                "High-performance computing becoming more accessible",
                "Increasing focus on energy efficiency"
            ]
        }
    
        # Get summaries for this category
        category_summaries = summaries.get(tech_category, [
            "Steady technological progress",
            "Growing market adoption",
            "Increasing integration with existing systems",
            "Emerging commercial applications",
            "Active research and development"
        ])
    
        # Return random summary
        return random.choice(category_summaries)

    def _generate_tech_key_points(self, tech_category: str) -> List[str]:
        """
        Generate varied and natural-sounding key educational points for a tech category
        with improved diversity and less repetition
    
        Args:
            tech_category: Tech category
    
        Returns:
            List of key points for educational content
        """
        # Track previously used key points to avoid repetition
        if not hasattr(self, '_recent_tech_key_points'):
            self._recent_tech_key_points = {cat: [] for cat in ['ai', 'quantum', 'blockchain_tech', 'advanced_computing']}
    
        # Ensure we have the category in our tracking dict
        if tech_category not in self._recent_tech_key_points:
            self._recent_tech_key_points[tech_category] = []
    
        # Define expanded key educational points with more natural language and variety
        # Each point written in a more conversational, natural style with varied structure
        key_points = {
            'ai': [
                # Basic concepts with varied sentence structures and conversational style (30 points)
                "How large language models actually work behind the scenes - they're not really 'thinking' as much as making statistical predictions based on patterns",
                "The key difference between narrow AI (which is what we have today) and AGI (artificial general intelligence) - one is specialized, the other would be adaptable to any task",
                "Why multimodal AI that can understand text, images, and audio together is creating such a fundamental shift in what's possible",
                "The surprisingly creative results you can get with the right prompt engineering techniques, even from the same underlying model",
                "Why fine-tuning AI models is a bit like teaching someone a specialized skill - it builds on general knowledge with specific expertise",
                "How neural networks mimic the basic structure of brain neurons, but operate in ways that are fundamentally different from biological thinking",
                "The concept of attention mechanisms in AI, which help models focus on the most relevant parts of input data, similar to how humans prioritize information",
                "Why transformer architecture revolutionized AI by allowing models to process sequences of data more efficiently than previous approaches",
                "How reinforcement learning works by having AI agents learn through trial and error, receiving rewards for good decisions and penalties for bad ones",
                "The difference between supervised learning (learning from labeled examples) and unsupervised learning (finding patterns in unlabeled data)",
                "Why deep learning requires so many layers - each layer learns increasingly complex features, from simple edges to complete objects",
                "How AI models learn to generalize from training data to make predictions about completely new, unseen information",
                "The concept of overfitting, where AI models become too specialized to their training data and perform poorly on new examples",
                "Why transfer learning allows us to take a model trained on one task and adapt it for related tasks with much less training data",
                "How computer vision AI breaks down images into pixels and learns to recognize patterns, objects, and scenes through millions of examples",
                "The way natural language processing models understand context and meaning in text, despite words having multiple meanings",
                "Why AI bias occurs when training data reflects historical inequalities or incomplete representations of the real world",
                "How generative AI creates new content by learning the statistical patterns and structures present in its training data",
                "The difference between classification tasks (putting things into categories) and regression tasks (predicting numerical values)",
                "Why ensemble methods combine multiple AI models to make more accurate predictions than any single model could achieve",
                "How AI models use embeddings to represent complex data like words or images as mathematical vectors in high-dimensional space",
                "The concept of gradient descent, which is how AI models iteratively improve by adjusting their parameters to minimize prediction errors",
                "Why data preprocessing and cleaning is often more important than the actual AI algorithm for achieving good results",
                "How AI models handle uncertainty and express confidence levels in their predictions rather than just giving binary answers",
                "The difference between online learning (continuously updating from new data) and batch learning (training on fixed datasets)",
                "Why feature engineering - selecting and transforming input variables - can dramatically impact AI model performance",
                "How AI models use activation functions to introduce non-linearity, allowing them to learn complex patterns and relationships",
                "The concept of regularization techniques that prevent AI models from becoming too complex and overfitting to training data",
                "Why cross-validation helps ensure AI models will perform well on new data by testing them on multiple subsets of training data",
                "How AI models balance the bias-variance tradeoff to achieve optimal performance on both training and test data",

                # More detailed technical aspects with natural variations (30 points)
                "The absolutely mind-boggling amount of computational power needed to train modern AI models, which is why only big companies and well-funded labs can build them from scratch",
                "How context windows work in language models, and why they're basically the model's short-term memory - explaining a lot about their limitations",
                "The ethical considerations that are becoming increasingly urgent as AI gets deployed in more sensitive domains",
                "How AI models are starting to be deployed at the edge, on smaller devices, despite their previous requirements for massive data centers",
                "Why the rise of specialized AI for specific industries is creating a wave of more practical, immediately useful applications",
                "The technical challenge of model compression - making AI models smaller and faster while maintaining their performance capabilities",
                "How distributed training across multiple GPUs and data centers enables the creation of models with trillions of parameters",
                "Why the attention mechanism in transformers requires quadratic computational complexity, making longer contexts exponentially expensive",
                "The emerging field of neuromorphic computing that mimics brain architecture more closely for potentially more efficient AI processing",
                "How knowledge distillation allows large, complex models to teach smaller, more efficient models their capabilities",
                "The technical details of how backpropagation calculates gradients to update neural network weights during training",
                "Why GPU architecture with thousands of parallel cores is particularly well-suited for the matrix operations that power AI training",
                "How mixed-precision training uses different numerical precisions to speed up model training while maintaining accuracy",
                "The challenge of catastrophic forgetting, where AI models lose previously learned information when trained on new tasks",
                "Why batch normalization and layer normalization help stabilize training of very deep neural networks",
                "How adversarial examples expose vulnerabilities in AI models by making tiny, imperceptible changes that cause misclassification",
                "The technical architecture of diffusion models that generate images by gradually removing noise from random patterns",
                "Why model parallelism and data parallelism represent different approaches to scaling AI training across multiple processors",
                "How retrieval-augmented generation combines pre-trained language models with external knowledge databases for more accurate responses",
                "The technical challenge of alignment in AI systems - ensuring they optimize for human values rather than just their training objectives",
                "Why quantization techniques reduce model size by using fewer bits to represent weights and activations",
                "How federated learning enables AI training across distributed devices while keeping sensitive data local",
                "The emerging field of automated machine learning (AutoML) that uses AI to design and optimize other AI systems",
                "Why sparse models with millions of parameters but only small subsets active at once offer efficiency advantages",
                "How continuous learning systems adapt to new data streams without forgetting previous knowledge",
                "The technical challenge of explainable AI - making complex model decisions interpretable to humans",
                "Why model ensemble techniques like bagging and boosting improve performance by combining multiple weak learners",
                "How neural architecture search uses AI to automatically design optimal network structures for specific tasks",
                "The emerging field of few-shot learning that enables AI models to learn new tasks from just a handful of examples",
                "Why memory-augmented neural networks incorporate external memory systems to enhance model capabilities",

                # Real-world applications with authentic perspective (30 points)
                "How different industries are using AI in completely different ways, from healthcare's focus on diagnosis to finance's use of pattern detection",
                "The growing tension between open-source and closed AI models, and what this means for innovation in the space",
                "Why synthetic data generation is becoming such a big deal for training AI in domains where real data is scarce or sensitive",
                "How AI is being used to analyze and optimize other technologies, creating a sort of technological feedback loop",
                "The real limitations of current AI that most marketing hype doesn't acknowledge, from hallucinations to bias to massive resource requirements",
                "How AI-powered drug discovery is accelerating pharmaceutical research by predicting molecular interactions and identifying promising compounds",
                "The way autonomous vehicles combine computer vision, sensor fusion, and real-time decision making in complex, safety-critical environments",
                "Why AI content moderation faces the impossible challenge of understanding context, sarcasm, and cultural nuances at global scale",
                "How recommendation systems shape what billions of people see, buy, and think about, creating unprecedented influence over human behavior",
                "The application of AI in climate modeling and environmental monitoring to better understand and respond to global challenges",
                "Why AI-assisted coding tools are changing software development, making programmers more productive while raising questions about skill requirements",
                "How AI is revolutionizing medical imaging by detecting patterns in X-rays, MRIs, and CT scans that human doctors might miss",
                "The use of AI in financial fraud detection, where models analyze millions of transactions in real-time to identify suspicious patterns",
                "Why AI-powered personalized education adapts to individual learning styles and paces, potentially transforming how we teach and learn",
                "How AI is being deployed in agriculture for crop monitoring, yield prediction, and precision farming techniques",
                "The application of AI in cybersecurity, where it helps detect novel attack patterns and respond to threats faster than human analysts",
                "Why AI translation services are breaking down language barriers but still struggle with idioms, cultural context, and emotional nuance",
                "How AI is transforming manufacturing through predictive maintenance, quality control, and supply chain optimization",
                "The use of AI in creative industries, from generating music and art to assisting with video editing and game development",
                "Why AI-powered virtual assistants are becoming more sophisticated but still face challenges in understanding complex, multi-turn conversations",
                "How AI is being applied to urban planning and smart city initiatives to optimize traffic flow, energy usage, and resource allocation",
                "The role of AI in scientific research, where it's helping analyze massive datasets and generate new hypotheses across multiple disciplines",
                "Why AI customer service chatbots work well for simple queries but often frustrate users when dealing with complex, emotional, or unusual situations",
                "How AI is being used in entertainment to create more realistic video game NPCs, generate movie special effects, and personalize content recommendations",
                "The application of AI in logistics and transportation to optimize delivery routes, manage inventory, and coordinate complex supply chains",
                "Why AI-powered hiring tools promise to reduce bias but often perpetuate existing inequalities present in historical hiring data",
                "How AI is transforming journalism through automated fact-checking, data analysis, and even generating routine news reports",
                "The use of AI in mental health applications, from chatbot therapists to mood tracking and crisis intervention systems",
                "Why AI in law enforcement raises significant privacy and civil liberties concerns while offering capabilities for pattern recognition and evidence analysis",
                "How AI is being applied to space exploration, from autonomous rovers on Mars to analyzing astronomical data for new discoveries",

                # Connections to crypto with varied approaches (30 points)
                "How trading bots are increasingly using AI to analyze market sentiment from social media and news sources",
                "The way some projects are using AI to automatically audit smart contracts for vulnerabilities, potentially preventing million-dollar hacks",
                "Why decentralized AI training is one of the most intriguing crossover points between AI and blockchain technology",
                "How NFT projects are beginning to incorporate generative AI to create dynamic, evolving digital assets",
                "The potential for AI governance systems to help manage complex DAOs and multi-token ecosystems",
                "How machine learning algorithms analyze blockchain transaction patterns to detect money laundering and other illicit activities",
                "Why AI-powered portfolio optimization is becoming crucial for managing exposure across hundreds of different cryptocurrency assets",
                "The emergence of prediction markets where AI models compete to forecast crypto prices, token launches, and market events",
                "How natural language processing helps analyze cryptocurrency whitepapers and documentation for investment due diligence",
                "Why AI-driven yield farming strategies automatically reallocate capital across DeFi protocols to maximize returns",
                "The use of AI in detecting and preventing crypto exchange hacks by analyzing unusual trading patterns and security breaches",
                "How computer vision AI verifies the authenticity and rarity of NFT artwork and collectibles",
                "Why AI models are being trained on blockchain data to understand network health, adoption patterns, and ecosystem growth",
                "The application of AI in optimizing consensus mechanisms and improving blockchain scalability and efficiency",
                "How sentiment analysis AI tracks social media buzz and influencer opinions to gauge market momentum for specific tokens",
                "Why AI-powered market making algorithms provide liquidity in decentralized exchanges while managing impermanent loss risks",
                "The use of AI to analyze on-chain metrics like wallet addresses, transaction volumes, and token distribution patterns",
                "How machine learning helps identify crypto market manipulation schemes like pump-and-dump operations and coordinated attacks",
                "Why AI-driven research tools scan thousands of crypto projects to identify promising investments based on technical and fundamental analysis",
                "The emergence of AI agents that can autonomously interact with DeFi protocols, executing complex strategies without human intervention",
                "How AI models predict gas fees and optimal transaction timing on congested blockchain networks",
                "Why neural networks are being used to analyze cryptocurrency mining operations and optimize energy efficiency",
                "The application of AI in cross-chain bridge security, detecting anomalous transfers and potential exploit attempts",
                "How AI-powered chatbots help users navigate complex DeFi interfaces and understand yield farming opportunities",
                "Why machine learning algorithms analyze tokenomics models to predict long-term sustainability and price dynamics",
                "The use of AI in crypto tax calculation, automatically categorizing transactions and calculating gains/losses across multiple chains",
                "How AI models analyze the correlation between traditional financial markets and cryptocurrency price movements",
                "Why AI-driven compliance tools help crypto businesses meet regulatory requirements across different jurisdictions",
                "The application of AI in crypto lending platforms to assess borrower risk and optimize interest rates",
                "How AI algorithms analyze staking rewards and validator performance across proof-of-stake blockchain networks"            
            ],
        
            'quantum': [
                # Basic concepts with conversational style (30 points)
                "How quantum bits or 'qubits' fundamentally differ from classical bits - they don't just store 0s and 1s, but can exist in superpositions of both states simultaneously",
                "The mind-bending concept of quantum entanglement, where particles become linked in ways that Einstein famously called 'spooky action at a distance'",
                "Why certain types of mathematical problems become dramatically easier to solve on quantum computers, while others see no benefit at all",
                "The massive challenge of quantum error correction - qubits are incredibly fragile and prone to errors, making stable quantum computing extremely difficult",
                "How different tech companies are pursuing completely different physical approaches to building quantum computers, from superconducting circuits to trapped ions",
                "The concept of quantum interference, where probability waves can cancel each other out or amplify, leading to counterintuitive computational results",
                "Why quantum tunneling allows particles to pass through barriers that should be impossible to cross according to classical physics",
                "How quantum decoherence causes qubits to lose their quantum properties when they interact with the environment, limiting computation time",
                "The principle of quantum uncertainty, which states that you can't precisely know both the position and momentum of a particle simultaneously",
                "Why measuring a quantum system fundamentally changes it - the act of observation collapses quantum superpositions into definite states",
                "How quantum gates manipulate qubits through rotations in quantum space, similar to how logic gates work in classical computers but far more complex",
                "The concept of quantum parallelism, where a quantum computer can explore multiple solution paths simultaneously rather than sequentially",
                "Why quantum computers need to operate at temperatures colder than outer space to maintain quantum coherence in their qubits",
                "How the no-cloning theorem proves that quantum information cannot be perfectly copied, unlike classical information",
                "The difference between quantum annealing and universal quantum computing - one solves optimization problems, the other aims for general computation",
                "Why quantum speedup only applies to specific algorithmic problems and doesn't make quantum computers universally faster than classical ones",
                "How Bell's theorem demonstrates that quantum mechanics violates local realism, proving that quantum correlations are genuinely non-classical",
                "The concept of quantum fidelity, which measures how close a quantum state is to its intended target state after operations and errors",
                "Why quantum volume is a more meaningful metric for quantum computer capability than just counting the number of qubits",
                "How different qubit technologies like photonic, topological, and neutral atom approaches each have unique advantages and challenges",
                "The principle of quantum complementarity, where quantum objects can display either wave-like or particle-like properties, but never both simultaneously",
                "Why quantum computers require specialized programming languages and algorithms that think in terms of probability amplitudes rather than binary logic",
                "How quantum phase kickback allows quantum algorithms to extract information from quantum oracles without directly measuring them",
                "The concept of quantum advantage versus quantum supremacy - the difference between being useful and just being faster on artificial problems",
                "Why adiabatic quantum computing follows a different paradigm, slowly evolving the system to find optimal solutions to complex problems",
                "How quantum error rates currently limit quantum computers to hundreds of operations before errors overwhelm the computation",
                "The fascinating property of quantum contextuality, where measurement outcomes depend on which other measurements are performed simultaneously",
                "Why hybrid quantum-classical algorithms represent the most promising near-term approach to practical quantum computing applications",
                "How quantum state tomography reconstructs unknown quantum states through repeated measurements on identical quantum systems",
                "The principle that quantum computers are reversible by nature, unlike classical computers which lose information through irreversible operations",

                # More technical aspects with natural language (30 points)
                "The real-world timeline for when quantum computing might actually deliver practical advantages - spoiler: it's probably further away than the hype suggests",
                "Why Shor's algorithm has cryptocurrency fans worried - it could theoretically break popular encryption methods that secure many blockchains",
                "How quantum-resistant cryptography is already being developed to prepare for the post-quantum era before it even arrives",
                "The surprising fact that quantum computing will likely complement classical computing rather than replace it - they're good at different things",
                "Why quantum supremacy demonstrations (where a quantum computer outperforms classical computers) are both significant milestones and somewhat overhyped",
                "The technical challenge of building logical qubits from thousands of physical qubits to achieve fault-tolerant quantum computation",
                "How surface codes and other quantum error correction schemes require massive overhead ratios of physical to logical qubits",
                "Why current noisy intermediate-scale quantum (NISQ) computers are limited to shallow circuits before errors accumulate beyond usefulness",
                "The fundamental trade-off between qubit coherence time and gate operation speed that constrains current quantum computer architectures",
                "How quantum channel capacity and quantum communication complexity set theoretical limits on quantum information processing",
                "Why variational quantum eigensolvers represent a promising near-term approach for quantum chemistry and optimization problems",
                "The technical details of how quantum approximate optimization algorithms (QAOA) tackle combinatorial optimization challenges",
                "How quantum machine learning algorithms might provide exponential speedups for certain pattern recognition and data analysis tasks",
                "Why building a universal fault-tolerant quantum computer requires solving engineering challenges that dwarf those of classical computers",
                "The role of quantum control theory in precisely manipulating quantum systems while minimizing unwanted interactions with the environment",
                "How different quantum programming paradigms like gate-based, adiabatic, and measurement-based computation offer distinct computational models",
                "Why quantum metrology and sensing applications can achieve precision improvements even with noisy quantum devices",
                "The technical challenge of quantum state preparation and initialization, which must be performed with extremely high fidelity",
                "How quantum algorithms for linear systems (like HHL) could revolutionize certain numerical computations if scaled up properly",
                "Why the quantum approximate counting algorithm demonstrates how quantum computers can estimate solutions to #P-complete problems",
                "The technical requirements for quantum networks that would enable distributed quantum computing across multiple quantum processors",
                "How quantum walks provide a framework for designing quantum algorithms that can search unstructured databases quadratically faster",
                "Why barren plateaus in quantum machine learning represent a fundamental challenge where optimization landscapes become exponentially flat",
                "The technical approach of quantum circuit synthesis, which compiles high-level quantum algorithms into sequences of elementary quantum gates",
                "How stabilizer codes provide a mathematical framework for understanding and designing quantum error correction schemes",
                "Why the threshold theorem in quantum computing proves that arbitrarily long quantum computations are theoretically possible with sufficient error correction",
                "The challenge of quantum software verification - proving that quantum programs behave correctly is significantly harder than for classical software",
                "How quantum simulation could provide insights into many-body physics problems that are intractable for classical computers",
                "Why the quantum Fourier transform serves as a crucial subroutine in many quantum algorithms but has no classical equivalent",
                "The technical complexity of quantum compiling, which must optimize quantum circuits for specific hardware constraints and connectivity graphs",

                # Practical implications with authentic perspective (30 points)
                "How quantum sensing applications are already delivering real benefits in fields like medicine and geology, even while full quantum computers remain in development",
                "The geopolitical race for quantum advantage that's quietly happening between major world powers, with billions in funding at stake",
                "Why certain industries like pharmaceuticals and materials science are particularly excited about quantum computing's potential",
                "How quantum random number generation provides truly random numbers (unlike classical computers) which has interesting applications for security",
                "The way quantum key distribution could create theoretically unhackable communication channels, even against future quantum computers",
                "How quantum-enhanced atomic clocks are already improving GPS accuracy and enabling more precise timekeeping for financial markets",
                "Why quantum magnetometry allows detection of magnetic fields with unprecedented sensitivity, revolutionizing medical imaging and geological surveys",
                "The practical applications of quantum gravimeters that can detect underground structures and resources with remarkable precision",
                "How quantum radar systems could potentially detect stealth aircraft by exploiting quantum entanglement between transmitted and received photons",
                "Why the pharmaceutical industry sees quantum computing as a potential game-changer for molecular simulation and drug discovery",
                "The current limitations that prevent quantum computers from solving real optimization problems better than classical supercomputers",
                "How quantum communication satellites are beginning to demonstrate global quantum key distribution capabilities",
                "Why the materials science applications of quantum computing could lead to breakthroughs in superconductors, catalysts, and battery technologies",
                "The realistic assessment that most quantum computing applications will require fault-tolerant machines with millions of qubits",
                "How quantum machine learning research is exploring whether quantum computers can provide advantages for artificial intelligence tasks",
                "Why the financial services industry is investigating quantum computing for portfolio optimization and risk analysis applications",
                "The practical challenges of building quantum computers that must operate in extreme isolation from electromagnetic interference",
                "How quantum algorithms for solving systems of linear equations could impact everything from weather prediction to fluid dynamics simulations",
                "Why the logistics and transportation industries are exploring quantum optimization for routing and scheduling problems",
                "The potential for quantum computing to revolutionize cryptanalysis while simultaneously requiring new cryptographic defenses",
                "How quantum sensors in healthcare could enable non-invasive detection of neural activity and early disease diagnosis",
                "Why the energy sector sees potential in quantum computing for optimizing power grid operations and renewable energy integration",
                "The realistic timeline showing that practical quantum advantage will likely emerge gradually in specialized applications rather than suddenly",
                "How quantum computing could transform weather forecasting by enabling more detailed atmospheric modeling and longer-range predictions",
                "Why the automotive industry is interested in quantum computing for optimizing traffic flow and autonomous vehicle coordination",
                "The current state of quantum startups and how venture capital is flowing into quantum technology development across various sectors",
                "How quantum computing research is driving advances in classical algorithms as researchers seek to maintain classical computational advantages",
                "Why the space industry sees quantum technologies as essential for future deep space communication and navigation systems",
                "The practical implications of quantum computing for artificial intelligence, from quantum neural networks to quantum-enhanced optimization",
                "How government agencies worldwide are developing quantum technology strategies to maintain technological competitiveness and national security",

                # Connections to crypto with varied approaches (30 points)
                "How blockchain projects are already preparing for the quantum threat by implementing quantum-resistant algorithms",
                "The potential implications of quantum computing for mining operations and proof-of-work consensus mechanisms",
                "Why quantum computing could potentially break certain crypto wallets while leaving others untouched, depending on their encryption methods",
                "How post-quantum cryptography standards are being developed collaboratively across the crypto ecosystem",
                "The intriguing possibility of 'quantum tokens' that could leverage quantum properties for unique cryptographic applications",
                "Why elliptic curve cryptography used in Bitcoin and Ethereum is particularly vulnerable to Shor's algorithm running on large quantum computers",
                "How lattice-based cryptography represents one of the leading approaches for quantum-resistant blockchain security",
                "The timeline for when quantum computers might pose a real threat to current cryptocurrency security - likely 10-20 years away",
                "Why hash-based signatures could provide quantum-resistant alternatives for blockchain transaction authentication",
                "How quantum key distribution could enable ultra-secure cryptocurrency exchanges that are theoretically immune to any computational attack",
                "The potential for quantum random number generation to improve the security of private key generation in crypto wallets",
                "Why some blockchain projects are experimenting with quantum-resistant consensus mechanisms that don't rely on traditional cryptographic assumptions",
                "How quantum computing could potentially optimize certain blockchain operations like transaction ordering and network routing",
                "The development of quantum-safe smart contracts that would remain secure even against future quantum computer attacks",
                "Why the crypto community is closely watching NIST's post-quantum cryptography standardization process for guidance on future security",
                "How quantum-resistant blockchains might require larger transaction sizes and different performance characteristics than current networks",
                "The potential for quantum entanglement to create new types of cryptographic protocols for decentralized applications",
                "Why preparation for the quantum threat is happening gradually across the crypto ecosystem rather than through sudden protocol changes",
                "How quantum computing could impact the security assumptions underlying various DeFi protocols and smart contract platforms",
                "The possibility that quantum computers could be used to optimize cryptocurrency mining operations or discover new consensus algorithms",
                "Why some researchers are exploring quantum blockchain concepts that would leverage quantum properties for consensus and security",
                "How the transition to post-quantum cryptography will likely happen through gradual protocol upgrades rather than complete system replacements",
                "The potential for quantum communication networks to enable new types of decentralized applications with quantum-enhanced security",
                "Why quantum computing could change the game theory underlying blockchain consensus by altering the computational capabilities of different actors",
                "How crypto projects are balancing the need for quantum resistance with the performance and efficiency requirements of blockchain networks",
                "The role that quantum computing might play in analyzing blockchain data and detecting patterns in decentralized finance markets",
                "Why the development of quantum-resistant cryptography often involves trade-offs between security, performance, and implementation complexity",
                "How quantum technologies could enable new types of zero-knowledge proofs and privacy-preserving protocols for blockchain applications",
                "The potential for quantum computing to transform cryptocurrency portfolio optimization and automated trading strategies",
                "Why the intersection of quantum computing and blockchain technology represents one of the most fascinating areas of emerging technology research"
            ],
        
            'blockchain_tech': [
                # Basic concepts with conversational style (30 points)
                "How zero-knowledge proofs allow you to prove you know something without revealing what that something is - it's like proving you know the password without actually saying it",
                "Why sharding is basically the blockchain equivalent of 'divide and conquer' - splitting the network into manageable pieces to process transactions in parallel",
                "The key differences between optimistic rollups and ZK rollups - one assumes transactions are valid and waits for challenges, the other mathematically proves validity",
                "How Layer 2 solutions are essentially 'traffic bypasses' that alleviate congestion on the main blockchain highway",
                "The evolution of consensus mechanisms beyond simple proof of work, creating more energy-efficient and democratic ways to agree on the state of the blockchain",
                "Why immutability in blockchains is both a feature and a bug - transactions can't be reversed, which prevents fraud but also makes mistakes permanent",
                "How cryptographic hashing creates unique 'fingerprints' for blocks of data, making it virtually impossible to alter records without detection",
                "The concept of distributed ledgers versus centralized databases - why spreading control across many computers creates trust without requiring trusted intermediaries",
                "Why public and private keys work like mathematical locks and keys, where you can share your public key freely but must guard your private key carefully",
                "How mining and validation processes turn computational work into network security, creating economic incentives for honest behavior",
                "The difference between permissioned and permissionless networks - one requires approval to participate, the other welcomes anyone with the technical means",
                "Why blockchain transactions are pseudonymous rather than anonymous - addresses are public but not necessarily linked to real-world identities",
                "How smart contracts execute automatically when predefined conditions are met, removing the need for trusted third parties in many agreements",
                "The concept of gas fees as a mechanism to prevent spam and allocate scarce computational resources on blockchain networks",
                "Why block time and block size create fundamental trade-offs between transaction speed, network security, and decentralization",
                "How digital signatures prove that transactions were authorized by the rightful owner of funds without revealing private keys",
                "The principle of longest chain rule in proof-of-work systems and why it helps resolve conflicts when multiple valid blocks are created simultaneously",
                "Why blockchain networks need careful economic design to balance security incentives with usability and transaction costs",
                "How multi-signature wallets require multiple approvals for transactions, providing enhanced security for high-value accounts",
                "The concept of finality in blockchain transactions - the point after which a transaction becomes practically irreversible",
                "Why fork events in blockchain networks can create entirely new cryptocurrencies when communities disagree on protocol changes",
                "How address generation works to create virtually unlimited unique identifiers without requiring central coordination",
                "The difference between soft forks and hard forks in blockchain protocol upgrades - one is backward compatible, the other isn't",
                "Why blockchain nodes maintain complete copies of transaction history, enabling anyone to verify the entire network state independently",
                "How timestamp servers in blockchain networks create verifiable chronological ordering of transactions without relying on external time sources",
                "The concept of UTXO (Unspent Transaction Output) model versus account-based models for tracking blockchain balances",
                "Why nonce values in blockchain mining represent the random element that makes proof-of-work computationally challenging but easy to verify",
                "How blockchain networks achieve Byzantine fault tolerance, remaining functional even when some participants act maliciously or fail",
                "The principle of deterministic wallet generation, where a single seed phrase can recreate an entire sequence of private keys",
                "Why blockchain transaction pools (mempools) create markets for transaction priority based on fee levels and network congestion",

                # More technical aspects with natural language (30 points)
                "The fascinating world of cross-chain bridges that connect different blockchain ecosystems, though they've unfortunately been prime targets for hackers",
                "How state channels allow for near-instantaneous off-chain transactions that only settle on the main chain when necessary - like running a tab at a bar",
                "Why smart contracts are so revolutionary - they're like self-executing agreements that can't be altered once deployed, creating new trustless interactions",
                "The critical role of oracles in connecting blockchains to external data, though they create their own 'oracle problem' of ensuring that data is accurate",
                "How different approaches to blockchain governance create fundamentally different ecosystems - from benevolent dictatorships to fully decentralized voting",
                "The technical architecture of plasma chains, which create hierarchical structures of blockchains to scale transaction processing",
                "Why atomic swaps enable trustless trading between different cryptocurrencies without requiring centralized exchanges",
                "How recursive zero-knowledge proofs (like SNARKs and STARKs) compress entire computation histories into small, easily verifiable proofs",
                "The challenge of data availability in rollup systems and why it's crucial for maintaining security while scaling transaction throughput",
                "Why validator rotation mechanisms in proof-of-stake systems help prevent long-term collusion while maintaining network security",
                "How commit-reveal schemes prevent front-running attacks by separating the commitment to an action from its revelation",
                "The technical details of how threshold signatures distribute cryptographic signing power across multiple parties",
                "Why MEV (Maximal Extractable Value) creates complex incentive dynamics for block producers and transaction ordering",
                "How verifiable random functions (VRFs) provide cryptographically secure randomness for blockchain applications",
                "The architecture of hybrid consensus mechanisms that combine multiple approaches to achieve optimal security and performance",
                "Why slashing conditions in proof-of-stake systems create economic penalties for malicious or negligent validator behavior",
                "How fraud proofs in optimistic systems enable anyone to challenge invalid state transitions and maintain network integrity",
                "The technical implementation of ring signatures and how they provide transaction privacy by obscuring the true sender",
                "Why committee-based consensus algorithms balance scalability with decentralization through representative validation",
                "How time-locked contracts enable complex multi-party agreements with automatic execution based on temporal conditions",
                "The cryptographic construction of accumulators that allow compact proofs of set membership without revealing the entire set",
                "Why blockchain virtual machines create standardized execution environments for smart contracts across different platforms",
                "How cryptographic commitments enable users to lock in choices before revealing them, preventing manipulation in various protocols",
                "The technical challenges of cross-chain communication protocols that must maintain security while bridging different consensus mechanisms",
                "Why light clients enable resource-constrained devices to interact with blockchains without downloading the entire transaction history",
                "How merkle proofs allow efficient verification of specific transactions within large blocks without downloading complete block data",
                "The implementation of multi-party computation protocols that enable collaborative computation without revealing private inputs",
                "Why blockchain state rent models address the problem of ever-growing state size by charging for long-term storage",
                "How confidential transactions use cryptographic techniques to hide transaction amounts while maintaining auditability",
                "The technical architecture of decentralized identity systems that give users control over their personal data and credentials",

                # Practical implications with authentic perspective (30 points)
                "The real-world challenges of blockchain scalability that new users experience as high fees and slow transactions during peak usage",
                "Why enterprise blockchain applications often look nothing like public cryptocurrencies, focusing instead on private networks with known participants",
                "How blockchain interoperability is slowly moving us from isolated 'blockchain islands' to a connected 'blockchain internet'",
                "The tension between decentralization ideals and the practical benefits of some centralization for user experience and development speed",
                "Why some of the most promising blockchain applications might be in supply chain, digital identity, and record-keeping rather than just financial transactions",
                "How blockchain energy consumption varies dramatically between different consensus mechanisms and network architectures",
                "The practical challenges of key management that prevent mainstream adoption - losing a private key means permanently losing access to funds",
                "Why blockchain transaction irreversibility creates both security benefits and user experience challenges compared to traditional payment systems",
                "How regulatory uncertainty affects blockchain development and adoption across different industries and geographical regions",
                "The real costs of blockchain networks beyond transaction fees, including infrastructure, development, and governance expenses",
                "Why blockchain user interfaces often lag behind traditional applications in usability and user experience design",
                "How different blockchain networks create varying levels of censorship resistance and why this matters for different use cases",
                "The practical limitations of smart contracts - they can only access on-chain data and execute predetermined logic",
                "Why blockchain networks face fundamental trade-offs between decentralization, security, and scalability that can't be fully optimized simultaneously",
                "How the environmental impact of different blockchain consensus mechanisms affects public perception and regulatory approaches",
                "The challenge of blockchain education and why technical complexity limits adoption among non-technical users",
                "Why blockchain networks require careful community coordination for protocol upgrades and feature development",
                "How blockchain transparency creates both benefits for auditability and challenges for privacy in various applications",
                "The practical implications of blockchain finality times for different use cases from payments to smart contract interactions",
                "Why blockchain developer tooling and infrastructure still lag behind traditional software development environments",
                "How blockchain network effects create winner-take-all dynamics that can lead to centralization despite decentralized technology",
                "The real-world performance characteristics of different blockchain networks under various load conditions",
                "Why blockchain applications often require hybrid architectures that combine on-chain and off-chain components",
                "How blockchain governance mechanisms affect the pace of innovation and adaptation to changing requirements",
                "The practical challenges of blockchain integration with existing enterprise systems and regulatory compliance frameworks",
                "Why blockchain networks must balance innovation with stability to maintain trust and usability for production applications",
                "How the pseudonymous nature of blockchain transactions creates both privacy benefits and compliance challenges",
                "The economic sustainability models for blockchain networks and how they affect long-term viability",
                "Why blockchain standardization efforts are crucial for interoperability but difficult to achieve across competing platforms",
                "How blockchain technology adoption patterns differ across industries based on specific regulatory and operational requirements",

                # Specialized technical concepts with varied structure (30 points)
                "How Merkle trees enable efficient and secure verification of large data structures on blockchains",
                "The economic incentive structures that keep blockchains secure and why they need careful balance",
                "Why the 'nothing at stake' problem creates different security considerations for proof-of-stake compared to proof-of-work",
                "How sidechains offer alternative environments with different rules while still maintaining connection to a parent blockchain",
                "The concept of tokenomics and how different token distribution and utility models dramatically affect project outcomes",
                "How cryptographic accumulators enable compact proofs about set membership and operations without revealing the entire set contents",
                "Why validator economics in proof-of-stake systems require careful design to prevent centralization and maintain network security",
                "The technical implementation of payment channels and how they enable high-frequency micropayments with minimal on-chain footprint",
                "How blockchain finality gadgets provide additional assurance that transactions won't be reversed after a certain point",
                "Why different hash functions used in blockchain networks have varying security properties and performance characteristics",
                "The architecture of decentralized autonomous organizations (DAOs) and how they encode governance rules in smart contracts",
                "How cryptographic proofs of space-time create alternative consensus mechanisms based on storage rather than computation",
                "Why blockchain virtual machines need to balance expressiveness with security and deterministic execution across all nodes",
                "The technical details of how atomic cross-chain swaps ensure that either both parties receive their assets or neither does",
                "How different approaches to blockchain privacy, from mixing services to zero-knowledge protocols, provide varying levels of anonymity",
                "Why blockchain state channels require careful dispute resolution mechanisms to handle cases where parties disagree",
                "The implementation of verifiable delay functions (VDFs) that provide publicly verifiable proof that time has passed",
                "How blockchain consensus algorithms must handle network partitions and maintain safety and liveness properties",
                "Why different approaches to blockchain sharding create trade-offs between throughput, security, and cross-shard communication complexity",
                "The technical architecture of blockchain-based prediction markets and how they aggregate information through token prices",
                "How cryptographic multi-party computation enables collaborative blockchain applications without revealing private inputs",
                "Why blockchain oracle networks require multiple data sources and aggregation mechanisms to resist manipulation",
                "The implementation of blockchain-based voting systems and the challenges of maintaining both verifiability and privacy",
                "How different blockchain fee market mechanisms affect transaction prioritization and network resource allocation",
                "Why blockchain light client protocols enable resource-constrained devices to verify network state without full node requirements",
                "The technical details of how blockchain rollups compress transaction data while maintaining the ability to verify state transitions",
                "How cryptographic commitments in blockchain protocols enable fair exchange and prevent front-running attacks",
                "Why blockchain networks require careful parameter tuning for block times, sizes, and difficulty adjustments to maintain stability",
                "The architecture of blockchain-based content delivery networks that incentivize distributed storage and bandwidth sharing",
                "How different approaches to blockchain governance tokens create varying mechanisms for protocol upgrades and parameter changes"
            ],
        
            'advanced_computing': [
                # Basic concepts with conversational style (30 points)
                "How neuromorphic computing tries to mimic the brain's architecture, creating chips that process information more like neurons than traditional processors",
                "The reality of edge computing, which brings processing power closer to where data is created instead of sending everything to the cloud",
                "Why we're hitting the physical limits of Moore's Law and the innovative approaches being developed to keep computing power growing",
                "How specialized hardware accelerators are creating massive efficiency gains for specific workloads like AI, graphics, or cryptography",
                "The rise of heterogeneous computing architectures that combine different types of processors optimized for different tasks",
                "Why parallel processing fundamentally changed how we think about computation - from sequential steps to simultaneous operations across multiple cores",
                "How field-programmable gate arrays (FPGAs) can be reconfigured on the fly to optimize for specific computational tasks",
                "The concept of von Neumann architecture and why moving beyond it might be necessary for future computing breakthroughs",
                "Why graphics processing units (GPUs) turned out to be perfect for AI workloads despite being designed for rendering images",
                "How chiplet architecture allows companies to combine different semiconductor technologies into single packages for optimal performance",
                "The emergence of tensor processing units (TPUs) specifically designed for machine learning matrix operations",
                "Why memory hierarchy from cache to RAM to storage creates fundamental trade-offs between speed, capacity, and cost",
                "How superscalar processors execute multiple instructions simultaneously by analyzing code for parallelizable operations",
                "The principle of speculative execution, where processors guess what operations will be needed next to stay ahead of program flow",
                "Why branch prediction algorithms help modern processors maintain performance by anticipating program execution paths",
                "How out-of-order execution allows processors to rearrange instruction sequences for maximum efficiency while maintaining correct results",
                "The concept of instruction pipelining, where different stages of instruction execution overlap to increase throughput",
                "Why cache coherency protocols ensure that multiple processor cores see consistent views of shared memory",
                "How vector processing units handle single instructions across multiple data elements simultaneously for massive parallel efficiency",
                "The emergence of domain-specific languages that compile directly to optimized hardware rather than general-purpose processors",
                "Why thermal design power (TDP) limits force careful balance between performance and heat generation in modern chips",
                "How dynamic frequency scaling adjusts processor speed based on workload demands to optimize power consumption",
                "The principle of locality of reference that makes caching effective - programs tend to access nearby memory locations repeatedly",
                "Why simultaneous multithreading (hyperthreading) allows single processor cores to handle multiple instruction streams",
                "How application-specific integrated circuits (ASICs) provide maximum efficiency for specialized tasks at the cost of flexibility",
                "The concept of computational complexity and why some problems fundamentally require exponentially more resources as they scale",
                "Why distributed computing systems must carefully handle consistency, availability, and partition tolerance trade-offs",
                "How load balancing algorithms distribute computational work across multiple processors or machines for optimal resource utilization",
                "The principle of Amdahl's Law, which limits how much parallel processing can speed up programs with sequential components",
                "Why emerging memory technologies like 3D XPoint blur the traditional boundaries between volatile and non-volatile storage",

                # More technical aspects with natural language (30 points)
                "Why in-memory computing is challenging the traditional separation between storage and processing, potentially eliminating a major bottleneck",
                "The fascinating potential of optical computing to use light instead of electricity, potentially enabling much faster and more efficient processing",
                "How quantum-inspired algorithms are applying principles from quantum computing to run on classical hardware with impressive results",
                "The growing focus on energy efficiency in computing, both for environmental reasons and practical concerns like battery life and heat",
                "Why cloud computing infrastructure is becoming increasingly specialized with custom silicon for different types of workloads",
                "How near-data computing architectures move processing capabilities closer to storage to reduce data movement overhead",
                "The technical challenge of memory wall, where processor speed improvements outpace memory access speed improvements",
                "Why approximate computing deliberately introduces small errors to achieve significant energy and performance improvements",
                "How photonic interconnects could replace electrical connections in future high-performance computing systems",
                "The architecture of dataflow computing, where program execution is driven by data availability rather than instruction sequences",
                "Why persistent memory technologies create new programming models that blur the distinction between memory and storage",
                "How chiplet-based designs enable mixing different process technologies and IP blocks in single packages",
                "The technical implementation of processing-in-memory (PIM) that performs computation directly within memory arrays",
                "Why advanced packaging technologies like 2.5D and 3D integration enable higher bandwidth and lower latency interconnects",
                "How machine learning accelerators use specialized matrix multiplication units and reduced precision arithmetic",
                "The technical challenge of dark silicon, where power constraints prevent activating all transistors on modern chips simultaneously",
                "Why software-defined hardware allows reconfigurable computing resources to adapt to changing workload requirements",
                "How advanced compiler optimizations automatically parallelize and vectorize code for modern hardware architectures",
                "The implementation of disaggregated computing architectures that separate compute, memory, and storage into independent scalable resources",
                "Why emerging non-volatile memory technologies require new file system and database architectures",
                "How advanced cooling technologies from liquid cooling to phase-change materials enable higher performance densities",
                "The technical details of how modern CPUs handle speculative execution while maintaining security against side-channel attacks",
                "Why network-on-chip architectures provide high-bandwidth, low-latency communication between cores in multi-core processors",
                "How compute express link (CXL) creates coherent connections between processors and accelerators for optimal performance",
                "The architecture of streaming multiprocessors in GPUs that enable massive parallel computation with thousands of threads",
                "Why advanced error correction codes protect against soft errors and hardware faults in large-scale computing systems",
                "How heterogeneous system architecture (HSA) enables unified programming models across different processor types",
                "The technical implementation of hardware transactional memory that simplifies parallel programming",
                "Why advanced interconnect technologies like silicon photonics enable high-bandwidth communication in data centers",
                "How computational storage devices perform processing directly on stored data to reduce data movement",

                # Practical implications with authentic perspective (30 points)
                "The surprising ways advanced computing is enabling real-time processing of complex data streams like video analytics and sensor networks",
                "How high-performance computing is becoming more accessible through cloud services, democratizing access to supercomputer-level resources",
                "The tension between general-purpose computing and specialized accelerators, and how it's reshaping hardware development",
                "Why the future probably isn't a single dominant computing paradigm but rather a diverse ecosystem of specialized approaches",
                "How advanced computing architectures are enabling new applications in fields from medicine to climate modeling to financial analysis",
                "The real-world impact of edge computing on applications from autonomous vehicles to smart manufacturing systems",
                "Why latency requirements in modern applications are driving fundamental changes in computing infrastructure design",
                "How advanced computing enables real-time personalization and recommendation systems that process massive data streams",
                "The practical challenges of programming heterogeneous systems that combine different types of processors and accelerators",
                "Why energy efficiency has become as important as raw performance for data center operations and mobile computing",
                "How advanced computing architectures enable new forms of scientific simulation that were previously impossible",
                "The way specialized hardware is transforming industries from genomics to financial modeling to autonomous systems",
                "Why the democratization of high-performance computing through cloud services is enabling innovation in smaller organizations",
                "How advanced computing enables real-time analysis of massive IoT sensor networks for smart city applications",
                "The practical implications of quantum-resistant computing for preparing classical systems for the post-quantum era",
                "Why advanced memory architectures are enabling new database and analytics applications with unprecedented performance",
                "How edge AI computing enables privacy-preserving machine learning by keeping sensitive data local",
                "The real-world applications of neuromorphic computing in robotics and sensor processing for ultra-low power operation",
                "Why advanced computing infrastructure is essential for handling the exponential growth in data generation",
                "How specialized computing hardware enables new forms of augmented and virtual reality applications",
                "The practical benefits of in-memory computing for real-time fraud detection and risk analysis systems",
                "Why advanced computing architectures are crucial for next-generation autonomous systems and robotics",
                "How high-bandwidth memory technologies enable new approaches to large-scale graph processing and network analysis",
                "The real-world impact of advanced computing on drug discovery and personalized medicine applications",
                "Why the convergence of computing and communication technologies is enabling new distributed application architectures",
                "How advanced computing enables real-time optimization of complex systems from traffic networks to power grids",
                "The practical challenges of maintaining performance while improving energy efficiency across different computing workloads",
                "Why advanced computing infrastructure is essential for handling the computational demands of modern AI applications",
                "How specialized hardware enables new approaches to cryptographic operations and security applications",
                "The real-world applications of advanced computing in climate modeling and environmental monitoring systems",

                # Connections to crypto with varied approaches (30 points)
                "The evolution of mining hardware from CPUs to GPUs to ASICs, and what might come next in the computing arms race",
                "How specialized blockchain processors might eventually optimize for cryptographic operations and consensus processes",
                "Why distributed computing networks might eventually merge with blockchain incentive structures to create new forms of shared computing resources",
                "The way advanced memory technologies could potentially transform blockchain node operation and validation processes",
                "How ultra-low-power computing could make truly tiny IoT devices capable of participating in blockchain networks",
                "Why FPGA-based mining represents a middle ground between general-purpose GPUs and specialized ASICs for cryptocurrency mining",
                "How advanced computing architectures could enable more sophisticated consensus mechanisms that require complex computations",
                "The potential for neuromorphic computing to create ultra-efficient cryptocurrency mining and validation systems",
                "Why edge computing infrastructure could enable new types of distributed blockchain applications with local processing",
                "How advanced cryptographic accelerators could improve the performance of privacy-preserving blockchain protocols",
                "The way high-bandwidth memory could enable blockchain nodes to process larger transaction volumes and complex smart contracts",
                "Why optical computing could potentially revolutionize blockchain performance by enabling faster cryptographic operations",
                "How advanced parallel processing architectures could optimize blockchain transaction validation and state management",
                "The potential for quantum-resistant hardware to provide built-in protection for cryptocurrency operations",
                "Why specialized DeFi processors could optimize for the specific computational patterns of decentralized finance protocols",
                "How advanced computing infrastructure enables sophisticated blockchain analytics and compliance monitoring systems",
                "The way heterogeneous computing could optimize different aspects of blockchain operations from mining to validation",
                "Why persistent memory technologies could improve blockchain node synchronization and state storage efficiency",
                "How advanced network processing units could optimize blockchain communication and consensus protocols",
                "The potential for computational storage to enable new types of blockchain applications that process data where it's stored",
                "Why energy-efficient computing architectures are crucial for sustainable blockchain networks and mining operations",
                "How advanced computing enables sophisticated trading algorithms and automated market-making systems in DeFi",
                "The way specialized security processors could provide hardware-level protection for cryptocurrency wallets and exchanges",
                "Why advanced computing infrastructure enables real-time blockchain monitoring and anomaly detection systems",
                "How machine learning accelerators could optimize blockchain network parameters and consensus algorithms",
                "The potential for advanced computing to enable new types of decentralized autonomous organizations with complex decision-making",
                "Why high-performance computing resources could be tokenized and traded as blockchain-based computational commodities",
                "How advanced computing enables sophisticated risk assessment and portfolio optimization in cryptocurrency markets",
                "The way emerging computing paradigms could enable new blockchain scalability solutions and Layer 2 protocols",
                "Why the intersection of advanced computing and blockchain technology represents a frontier for next-generation decentralized applications"
            ],
        }
    
        # Fallback if category isn't in our expanded set
        default_points = [
            "The fundamental principles behind this technology and why they matter for cryptocurrency",
            "Current applications and real-world use cases that are already showing promise",
            "Future challenges and opportunities that could shape development in this space",
            "How this technology could potentially interact with and enhance blockchain systems",
            "The importance of this technology for the broader digital ecosystem and innovation landscape",
            "The underlying technical architecture that makes this technology unique and why it represents a departure from traditional approaches",
            "How market forces and economic incentives are driving adoption and development in this emerging field",
            "The regulatory landscape and policy considerations that could significantly impact the trajectory of this technology",
            "Why certain industries are particularly well-positioned to benefit from early adoption of these technological advances",
            "The convergence opportunities where this technology intersects with other emerging trends like AI, blockchain, and advanced computing",
            "How this technology addresses specific pain points that existing solutions haven't been able to solve effectively",
            "The scalability considerations and infrastructure requirements needed for widespread implementation",
            "Why the timing is right for this technology to gain mainstream traction and what factors are accelerating adoption",
            "The competitive dynamics between different approaches and implementations within this technological space",
            "How developer communities and open-source initiatives are shaping the evolution and standardization of this technology",
            "The investment landscape and venture capital trends that are fueling innovation and development in this sector",
            "Why security and privacy considerations are particularly important for this technology and how they're being addressed",
            "The global perspective on how different regions are approaching development and regulation of this technology",
            "How this technology could potentially democratize access to capabilities that were previously only available to large organizations",
            "The environmental and sustainability implications of widespread adoption and how they're being mitigated",
            "Why interoperability and standardization efforts are crucial for the long-term success of this technology",
            "The talent acquisition challenges and educational initiatives needed to support continued development in this field",
            "How this technology is creating new business models and economic opportunities that didn't exist before",
            "The technical limitations and engineering challenges that still need to be overcome for optimal performance",
            "Why partnerships and collaboration between traditional industries and technology companies are essential for adoption",
            "The user experience improvements needed to make this technology accessible to non-technical audiences",
            "How this technology could potentially reshape entire industries by changing fundamental operational assumptions",
            "The role of academic research and institutional involvement in advancing the theoretical foundations of this technology",
            "Why the network effects and ecosystem development are critical factors in determining which implementations will succeed",
            "The long-term societal implications and transformative potential that make this technology worth monitoring closely"
        ]
    
        # Get points for this category or use default
        category_points = key_points.get(tech_category, default_points)
    
        # Filter out recently used points for this category to avoid repetition
        recent_points = self._recent_tech_key_points.get(tech_category, [])
        available_points = [p for p in category_points if p not in recent_points]
    
        # If we've filtered too many, reset and use all points
        if len(available_points) < 3:
            logger.logger.debug(f"Resetting recently used points for {tech_category} due to insufficient remaining points")
            self._recent_tech_key_points[tech_category] = []
            available_points = category_points
    
        # Select 3 random points without replacement
        selected_points = []
        if len(available_points) >= 3:
            # Randomly select 3 points
            selected_points = random.sample(available_points, 3)
        else:
            # Use all available and fill in with random ones if needed
            selected_points = available_points.copy()
            # Add random points from the full set to reach 3 total
            additional_needed = 3 - len(selected_points)
            if additional_needed > 0:
                remaining_points = [p for p in category_points if p not in selected_points]
                if remaining_points:
                    selected_points.extend(random.sample(remaining_points, min(additional_needed, len(remaining_points))))
    
        # Fill to 3 points with default if somehow we don't have enough
        while len(selected_points) < 3:
            selected_points.append("How this technology impacts the future of digital assets and decentralized systems")
    
        # Update our tracking of recently used points
        for point in selected_points:
            if point not in self._recent_tech_key_points[tech_category]:
                self._recent_tech_key_points[tech_category].append(point)
    
        # Keep only the 10 most recent points to avoid unbounded growth
        self._recent_tech_key_points[tech_category] = self._recent_tech_key_points[tech_category][-10:]
    
        return selected_points
    def _generate_learning_objective(self, tech_category: str) -> str:
        """
        Generate a learning objective for educational tech content
    
        Args:
            tech_category: Tech category
        
        Returns:
            Learning objective string
        """
        # Define learning objectives by category
        objectives = {
            'ai': [
                "how AI technologies are transforming the crypto landscape",
                "the core principles behind modern AI systems",
                "how AI and blockchain technologies can complement each other",
                "the key limitations and challenges of current AI approaches",
                "how AI is being used to enhance trading, security, and analytics in crypto"
            ],
            'quantum': [
                "how quantum computing affects blockchain security",
                "the fundamentals of quantum computing in accessible terms",
                "the timeline and implications of quantum advances for cryptography",
                "how the crypto industry is preparing for quantum computing",
                "the difference between quantum threats and opportunities for blockchain"
            ],
            'blockchain_tech': [
                "how advanced blockchain technologies are addressing scalability",
                "the technical foundations of modern blockchain systems",
                "the trade-offs between different blockchain scaling approaches",
                "how blockchain privacy technologies actually work",
                "the evolution of blockchain architecture beyond first-generation systems"
            ],
            'advanced_computing': [
                "how specialized computing hardware is changing crypto mining",
                "the next generation of computing technologies on the horizon",
                "how computing advances are enabling new blockchain capabilities",
                "the relationship between energy efficiency and blockchain sustainability",
                "how distributed computing and blockchain share foundational principles"
            ]
        }
    
        # Get objectives for this category
        category_objectives = objectives.get(tech_category, [
            "the fundamentals of this technology in accessible terms",
            "how this technology relates to blockchain and cryptocurrency",
            "the potential future impact of this technological development",
            "the current state and challenges of this technology",
            "how this technology might transform digital finance"
        ])
    
        # Return random objective
        return random.choice(category_objectives)

    def get_tokens_with_recent_data_by_market_cap(self, hours: int = 24, limit: int = 25) -> List[str]:
        """
        Get top tokens by market cap that have recent data in the database
        
        Args:
            hours: Number of hours to look back for recent data (default: 24)
            limit: Maximum number of tokens to return (default: 25)
            
        Returns:
            List[str]: Token symbols sorted by market cap (highest first)
        """
        logger.logger.info(f"🔍 Getting top {limit} tokens by market cap with recent data")
        
        # Check if database is initialized
        if not hasattr(self, 'db') or self.db is None:
            logger.logger.error("❌ FALLBACK: Database not initialized for get_tokens_with_recent_data_by_market_cap")
            from database import CryptoDatabase
            self.db = CryptoDatabase()
            logger.logger.info("✅ Created new database connection")
        
        try:
            # Try to reconnect if the database is closed
            try:
                # Get database connection using self.db
                conn, cursor = self.db._get_connection()
                # Test connection with a simple query
                cursor.execute("SELECT 1")
                cursor.fetchone()
            except Exception as conn_error:
                if "Cannot operate on a closed database" in str(conn_error) or "database is closed" in str(conn_error):
                    logger.logger.warning(f"⚠️ Database connection closed: {str(conn_error)}")
                    # Reinitialize the database connection
                    del self.db
                    from database import CryptoDatabase
                    self.db = CryptoDatabase()
                    logger.logger.info("✅ Reconnected to database successfully")
                    conn, cursor = self.db._get_connection()
                else:
                    raise conn_error  # Re-raise if it's not a closed database error
            
            # Step 1: Get all distinct tokens with recent data
            logger.logger.debug(f"Step 1: Finding tokens with data in last {hours} hours...")
            cursor.execute("""
                SELECT DISTINCT chain
                FROM market_data
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """, (hours,))
            
            recent_tokens = [row['chain'] for row in cursor.fetchall()]
            logger.logger.info(f"📊 Found {len(recent_tokens)} tokens with recent data: {recent_tokens[:10]}...")
            
            if not recent_tokens:
                logger.logger.warning("⚠️ No tokens found with recent data")
                return []
                
            # Step 2: Get latest market cap for each token and filter valid ones
            logger.logger.debug("Step 2: Getting market cap data for tokens...")
            
            # Use placeholders for all tokens in the query
            placeholders = ','.join(['?'] * len(recent_tokens))
            query = f"""
                SELECT chain, market_cap, timestamp
                FROM market_data
                WHERE chain IN ({placeholders})
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                AND market_cap > 0
                ORDER BY timestamp DESC
            """
            
            cursor.execute(query, recent_tokens + [hours])
            
            # Get the most recent market cap for each token
            token_market_caps = {}
            for row in cursor.fetchall():
                token = row['chain']
                if token not in token_market_caps:
                    token_market_caps[token] = row['market_cap']
            
            # Sort tokens by market cap (highest first)
            sorted_tokens = sorted(token_market_caps.keys(), 
                                key=lambda t: token_market_caps.get(t, 0), 
                                reverse=True)
            
            # Limit the number of results
            top_tokens = sorted_tokens[:limit]
            logger.logger.info(f"✅ Top tokens by market cap: {top_tokens[:10]}...")
            
            return top_tokens
            
        except Exception as e:
            logger.logger.error(f"❌ FALLBACK: Error getting tokens by market cap: {str(e)}")
            return []

    @ensure_naive_datetimes     
    def _select_best_token_for_timeframe(self, market_data: Dict[str, Any], timeframe: str) -> Optional[str]:
        """
        Select the best token to use for a specific timeframe post
        NEW: Uses database query to get top 25 tokens by market cap instead of hardcoded list
        
        Args:
            market_data: Market data dictionary from API calls
            timeframe: Timeframe to select for
            
        Returns:
            Best token symbol for the timeframe or None if no suitable token found
        """
        logger.logger.info(f"🎯 Starting token selection for {timeframe} timeframe")
        
        try:
            # Step 1: Get top 25 tokens by market cap from database (last 24 hours)
            logger.logger.info("Step 1: Querying database for top tokens by market cap...")
            
            try:
                database_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=25)
                logger.logger.info(f"📊 Database returned {len(database_tokens)} tokens: {database_tokens}")
            except Exception as db_error:
                logger.logger.error(f"❌ Database query failed: {str(db_error)}")
                logger.log_error("Database Token Query", str(db_error))
                return None
            
            if not database_tokens:
                logger.logger.warning("⚠️ No tokens found in database with recent data")
                return None
            
            # Step 2: Filter to only tokens present in market_data parameter
            logger.logger.info("Step 2: Filtering to tokens present in market_data...")
            
            available_tokens = []
            for token in database_tokens:
                if token in market_data:
                    available_tokens.append(token)
                    logger.logger.debug(f"   ✅ {token}: Present in market_data")
                else:
                    logger.logger.debug(f"   ❌ {token}: Missing from market_data")
            
            logger.logger.info(f"🔍 {len(available_tokens)} tokens available in market_data: {available_tokens}")
            
            if not available_tokens:
                logger.logger.warning("⚠️ No database tokens found in market_data parameter")
                return None
            
            # Step 3: Check historical data sufficiency
            logger.logger.info("Step 3: Validating historical data sufficiency...")
            
            qualified_tokens = []
            
            for token in available_tokens:
                logger.logger.debug(f"   📋 Checking {token} historical data...")
                
                # Check prediction history (>= 5 predictions)
                try:
                    perf_stats = self.db.get_prediction_performance(token=token, timeframe=timeframe)
                    prediction_count = 0
                    if perf_stats and len(perf_stats) > 0:
                        prediction_count = perf_stats[0].get('total_predictions', 0)
                    
                    logger.logger.debug(f"      🔮 {token}: {prediction_count} predictions")
                except Exception as pred_error:
                    logger.logger.debug(f"      ❌ {token}: Prediction check failed - {str(pred_error)}")
                    prediction_count = 0
                
                # Check price history (>= 24 hours of data)
                try:
                    recent_data = self.db.get_recent_market_data(token, hours=24)
                    price_data_points = len(recent_data)
                    
                    logger.logger.debug(f"      📊 {token}: {price_data_points} price data points (24h)")
                except Exception as price_error:
                    logger.logger.debug(f"      ❌ {token}: Price data check failed - {str(price_error)}")
                    price_data_points = 0
                
                # Determine if token qualifies
                prediction_sufficient = prediction_count >= 5
                price_data_sufficient = price_data_points >= 1  # At least some recent data
                
                if prediction_sufficient and price_data_sufficient:
                    qualified_tokens.append(token)
                    logger.logger.debug(f"      ✅ {token}: QUALIFIED (predictions: {prediction_count}, data points: {price_data_points})")
                else:
                    logger.logger.debug(f"      ❌ {token}: INSUFFICIENT (predictions: {prediction_count}, data points: {price_data_points})")
            
            logger.logger.info(f"✅ {len(qualified_tokens)} tokens passed historical data validation: {qualified_tokens}")
            
            if not qualified_tokens:
                logger.logger.warning("⚠️ No tokens have sufficient historical data for reliable scoring")
                return None
            
            # Step 4: Score each qualified token using existing logic
            logger.logger.info("Step 4: Scoring qualified tokens...")
            
            candidates = []
            
            for token in qualified_tokens:
                logger.logger.debug(f"   🧮 Scoring {token}...")
                
                try:
                    # Calculate momentum score
                    momentum_score = self._calculate_momentum_score(token, market_data, timeframe)
                    logger.logger.debug(f"      📈 Momentum: {momentum_score:.2f}")
                    
                    # Calculate activity score based on recent volume and price changes
                    token_data = market_data.get(token, {})
                    volume = token_data.get('volume', 0)
                    price_change = abs(token_data.get('price_change_percentage_24h', 0))
                    logger.logger.debug(f"      💰 Price change: {price_change:.2f}%, Volume: {volume:,.0f}")
                    
                    # Get volume trend
                    historical_volumes = self._get_historical_volume_data(token, timeframe=timeframe)
                    volume_trend, _ = self._analyze_volume_trend(volume, historical_volumes, timeframe=timeframe)
                    logger.logger.debug(f"      📊 Volume trend: {volume_trend:.2f}")
                    
                    # Get historical prediction accuracy
                    perf_stats = self.db.get_prediction_performance(token=token, timeframe=timeframe)
                    
                    # Calculate accuracy score
                    accuracy_score = 0
                    if perf_stats:
                        accuracy = perf_stats[0].get('accuracy_rate', 0)
                        total_preds = perf_stats[0].get('total_predictions', 0)
                        
                        # Only consider accuracy if we have enough data
                        if total_preds >= 5:
                            accuracy_score = accuracy * (min(total_preds, 20) / 20)  # Scale by number of predictions up to 20
                    
                    logger.logger.debug(f"      🎯 Accuracy: {accuracy_score:.2f}")
                    
                    # Calculate recency score - prefer tokens we haven't posted about recently
                    recency_score = 0
                    
                    # Check when this token was last posted for this timeframe
                    recent_posts = self.db.get_recent_posts(hours=48, timeframe=timeframe)
                    
                    token_posts = [p for p in recent_posts if token.upper() in p.get('content', '')]
                    
                    if not token_posts:
                        # Never posted - maximum recency score
                        recency_score = 100
                        logger.logger.debug(f"      🕐 Recency: {recency_score:.0f} (never posted)")
                    else:
                        # Calculate hours since last post
                        last_posts_times = [strip_timezone(datetime.fromisoformat(p.get('timestamp', datetime.min.isoformat()))) for p in token_posts]
                        if last_posts_times:
                            last_post_time = max(last_posts_times)
                            hours_since = safe_datetime_diff(datetime.now(), last_post_time) / 3600
                            
                            # Scale recency score based on timeframe
                            if timeframe == "1h":
                                recency_score = min(100, hours_since * 10)  # Max score after 10 hours
                            elif timeframe == "24h":
                                recency_score = min(100, hours_since * 2)   # Max score after 50 hours
                            else:  # 7d
                                recency_score = min(100, hours_since * 0.5)  # Max score after 200 hours
                            
                            logger.logger.debug(f"      🕐 Recency: {recency_score:.0f} ({hours_since:.1f}h since last post)")
                    
                    # Combine scores with timeframe-specific weightings
                    if timeframe == "1h":
                        # For hourly, momentum and price action matter most
                        total_score = (
                            momentum_score * 0.5 +
                            price_change * 3.0 +
                            volume_trend * 0.7 +
                            accuracy_score * 0.3 +
                            recency_score * 0.4
                        )
                    elif timeframe == "24h":
                        # For daily, balance between momentum, accuracy and recency
                        total_score = (
                            momentum_score * 0.4 +
                            price_change * 2.0 +
                            volume_trend * 0.8 +
                            accuracy_score * 0.5 +
                            recency_score * 0.6
                        )
                    else:  # 7d
                        # For weekly, accuracy and longer-term views matter more
                        total_score = (
                            momentum_score * 0.3 +
                            price_change * 1.0 +
                            volume_trend * 1.0 +
                            accuracy_score * 0.8 +
                            recency_score * 0.8
                        )
                    
                    candidates.append((token, total_score))
                    
                    logger.logger.info(f"      🏆 {token} TOTAL SCORE: {total_score:.2f}")
                    logger.logger.info(f"         Components: M:{momentum_score:.1f} P:{price_change:.1f} V:{volume_trend:.1f} A:{accuracy_score:.1f} R:{recency_score:.1f}")
                    
                except Exception as scoring_error:
                    logger.logger.error(f"      ❌ {token}: Scoring failed - {str(scoring_error)}")
                    logger.log_error(f"Token Scoring - {token}", str(scoring_error))
                    continue
            
            # Step 5: Select highest-scoring token
            if not candidates:
                logger.logger.warning("⚠️ No tokens could be successfully scored")
                return None
            
            # Sort by total score descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            logger.logger.info(f"📊 Final candidate rankings for {timeframe}:")
            for i, (token, score) in enumerate(candidates[:5], 1):
                logger.logger.info(f"   #{i} {token}: {score:.2f}")
            
            selected_token = candidates[0][0]
            selected_score = candidates[0][1]
            
            logger.logger.info(f"🏆 SELECTED TOKEN: {selected_token} with score {selected_score:.2f}")
            logger.logger.info(f"✅ Token selection completed successfully for {timeframe}")
            
            return selected_token
            
        except Exception as e:
            logger.logger.error(f"❌ Token selection failed for {timeframe}: {str(e)}")
            logger.log_error(f"Select Best Token - {timeframe}", str(e))
            return None

    def _check_for_posts_to_reply(self, market_data: Dict[str, Any]) -> bool:
        """
        Check for posts to reply to and generate replies
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating if any replies were posted
        """
        now = strip_timezone(datetime.now())
    
        # Check if it's time to look for posts to reply to
        time_since_last_check = safe_datetime_diff(now, self.last_reply_check) / 60
        if time_since_last_check < self.reply_check_interval:
            logger.logger.debug(f"Skipping reply check, {time_since_last_check:.1f} minutes since last check (interval: {self.reply_check_interval})")
            return False
        
        # Also check cooldown period
        time_since_last_reply = safe_datetime_diff(now, self.last_reply_time) / 60
        if time_since_last_reply < self.reply_cooldown:
            logger.logger.debug(f"In reply cooldown period, {time_since_last_reply:.1f} minutes since last reply (cooldown: {self.reply_cooldown})")
            return False
        
        logger.logger.info("Starting check for posts to reply to")
        self.last_reply_check = now
    
        try:
            # Scrape timeline for posts
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 2)  # Get more to filter
            logger.logger.info(f"Timeline scraping completed - found {len(posts) if posts else 0} posts")
        
            if not posts:
                logger.logger.warning("No posts found during timeline scraping")
                return False

            # Log sample posts for debugging
            for i, post in enumerate(posts[:3]):  # Log first 3 posts
                logger.logger.info(f"Sample post {i}: {post.get('text', '')[:100]}...")

            # Find market-related posts
            logger.logger.info(f"Finding market-related posts among {len(posts)} scraped posts")
            market_posts = self.content_analyzer.find_market_related_posts(posts)
            logger.logger.info(f"Found {len(market_posts)} market-related posts, checking which ones need replies")
            
            # Filter out posts we've already replied to
            unreplied_posts = self.timeline_scraper.filter_already_replied_posts(market_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied market-related posts")
            if unreplied_posts:
                for i, post in enumerate(unreplied_posts[:3]):
                    logger.logger.info(f"Sample unreplied post {i}: {post.get('text', '')[:100]}...")
            
            if not unreplied_posts:
                return False
                
            # Prioritize posts (engagement, relevance, etc.)
            prioritized_posts = self.timeline_scraper.prioritize_posts(unreplied_posts)
            
            # Limit to max replies per cycle
            posts_to_reply = prioritized_posts[:self.max_replies_per_cycle]
            
            # Generate and post replies
            logger.logger.info(f"Starting to reply to {len(posts_to_reply)} prioritized posts")
            successful_replies = self.reply_handler.reply_to_posts(posts_to_reply, market_data, max_replies=self.max_replies_per_cycle)
            
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies")
                self.last_reply_time = now                    
                return True
            else:
                logger.logger.info("No replies were successfully posted")
                return False
                
        except Exception as e:
            logger.log_error("Check For Posts To Reply", str(e))
            return False
    @ensure_naive_datetimes
    def _get_daily_tech_post_count(self) -> Dict[str, Any]:
        """
        Get tech post count for the current calendar day with proper midnight reset
        Respects existing datetime handling and normalization functions
    
        Returns:
            Dictionary containing tech post counts and limits for today
        """
        try:
            # Get the current date and time with proper timezone handling
            now = strip_timezone(datetime.now())
    
            # Calculate the start of the current day (midnight)
            today_start = strip_timezone(datetime(now.year, now.month, now.day, 0, 0, 0))
    
            # Get configured maximum daily tech posts
            max_daily_posts = config.TECH_CONTENT_CONFIG.get('max_daily_tech_posts', 6)
    
            tech_posts = {}
            tech_posts_today = 0
            last_tech_post = None
    
            # Query database only for posts from the current calendar day
            if self.db:
                try:
                    # Calculate hours since midnight for database query
                    hours_since_midnight = safe_datetime_diff(now, today_start) / 3600
            
                    # Get posts only from the current day
                    recent_posts = self.db.get_recent_posts(hours=hours_since_midnight)
            
                    # Filter to tech-related posts and verify they're from today
                    for post in recent_posts:
                        if 'tech_category' in post:
                            # Verify post is from today by checking the timestamp
                            post_time = strip_timezone(datetime.fromisoformat(post['timestamp']))
                            if post_time >= today_start:  # Only count posts from today
                                category = post['tech_category']
                                if category not in tech_posts:
                                    tech_posts[category] = []
                                tech_posts[category].append(post)
                        
                                # Track the most recent tech post timestamp
                                if last_tech_post is None or post_time > last_tech_post:
                                    last_tech_post = post_time
            
                    # Count today's tech posts
                    tech_posts_today = sum(len(posts) for posts in tech_posts.values())
            
                    logger.logger.debug(
                        f"Daily tech posts: {tech_posts_today}/{max_daily_posts} since midnight " 
                        f"({hours_since_midnight:.1f} hours ago)"
                    )
            
                except Exception as db_err:
                    logger.logger.warning(f"Error retrieving tech posts: {str(db_err)}")
                    tech_posts_today = 0
                    last_tech_post = today_start  # Default to start of day if error
            
            # If no posts found today, set default last post to start of day
            if last_tech_post is None:
                last_tech_post = today_start
    
            # Check if maximum posts for today has been reached
            max_reached = tech_posts_today >= max_daily_posts
            if max_reached:
                logger.logger.info(
                    f"Maximum daily tech posts reached for today: {tech_posts_today}/{max_daily_posts}"
                )
            else:
                logger.logger.debug(
                    f"Daily tech post count: {tech_posts_today}/{max_daily_posts} - " 
                    f"additional posts allowed today"
                )
        
            # Return comprehensive stats
            return {
                'tech_posts_today': tech_posts_today,
                'max_daily_posts': max_daily_posts,
                'last_tech_post': last_tech_post,
                'day_start': today_start,
                'max_reached': max_reached,
                'categories_posted': list(tech_posts.keys()),
                'posts_by_category': {k: len(v) for k, v in tech_posts.items()}
            }
    
        except Exception as e:
            logger.log_error("Daily Tech Post Count", str(e))
            # The now variable needs to be defined before using it in exception handling
            current_now = strip_timezone(datetime.now())
            # Return safe defaults
            return {
                'tech_posts_today': 0,
                'max_daily_posts': config.TECH_CONTENT_CONFIG.get('max_daily_posts', 6),
                'last_tech_post': strip_timezone(current_now - timedelta(hours=24)),
                'day_start': strip_timezone(datetime(current_now.year, current_now.month, current_now.day, 0, 0, 0)),
                'max_reached': False,
                'categories_posted': [],
                'posts_by_category': {}
            }
    @ensure_naive_datetimes
    def get_posts_since_timestamp(self, timestamp: str) -> List[Dict[str, Any]]:
        """
        Get all posts since a specific timestamp
    
        Args:
            timestamp: ISO format timestamp string
    
        Returns:
            List of posts
        """
        try:
            # Use the database from the config instead of trying to get a connection directly
            if not hasattr(self, 'db') or not self.db:
                logger.logger.error("Database not initialized")
                return []
            
            # Get a database connection and cursor
            conn = None
            cursor = None
            try:
                # Try to access database using the same pattern used elsewhere in the code
                if hasattr(self.db, 'conn'):
                    conn = self.db.conn
                    cursor = conn.cursor()
                elif hasattr(self.db, '_get_connection'):
                    conn, cursor = self.db._get_connection()
                else:
                    # As a last resort, check if config has a db
                    if hasattr(self, 'config') and hasattr(self.config, 'db'):
                        if hasattr(self.config.db, 'conn'):
                            conn = self.config.db.conn
                            cursor = conn.cursor()
                        elif hasattr(self.config.db, '_get_connection'):
                            conn, cursor = self.config.db._get_connection()
            except AttributeError as ae:
                logger.logger.error(f"Database attribute error: {str(ae)}")
            except Exception as conn_err:
                logger.logger.error(f"Failed to get database connection: {str(conn_err)}")
                
            if not conn or not cursor:
                logger.logger.error("Could not obtain database connection or cursor")
                return []
    
            # Ensure timestamp is properly formatted
            # Check if timestamp is already a datetime
            if isinstance(timestamp, datetime):
                # Convert to string using the same format as used elsewhere
                timestamp_str = strip_timezone(timestamp).isoformat()
            else:
                # Assume it's a string, but verify it's in the expected format
                try:
                    # Parse timestamp string to ensure it's valid
                    dt = datetime.fromisoformat(timestamp)
                    # Ensure timezone handling is consistent
                    timestamp_str = strip_timezone(dt).isoformat()
                except ValueError:
                    logger.logger.error(f"Invalid timestamp format: {timestamp}")
                    return []
    
            # Execute the query
            try:
                query = """
                    SELECT * FROM posted_content
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """
                cursor.execute(query, (timestamp_str,))
            
                # Fetch all results
                results = cursor.fetchall()
            
                # Convert rows to dictionaries if needed
                if results:
                    if not isinstance(results[0], dict):
                        # Get column names from cursor description
                        columns = [desc[0] for desc in cursor.description]
                        # Convert each row to a dictionary
                        dict_results = []
                        for row in results:
                            row_dict = {columns[i]: value for i, value in enumerate(row)}
                            dict_results.append(row_dict)
                        results = dict_results
                
                    # Process datetime fields to ensure consistent handling
                    for post in results:
                        if 'timestamp' in post and post['timestamp']:
                            # Convert timestamp strings to datetime objects with consistent handling
                            try:
                                if isinstance(post['timestamp'], str):
                                    post['timestamp'] = strip_timezone(datetime.fromisoformat(post['timestamp']))
                            except ValueError:
                                # If conversion fails, leave as string
                                pass
            
                return results
            
            except Exception as query_err:
                logger.logger.error(f"Query execution error: {str(query_err)}")
                return []
            
        except Exception as e:
            logger.log_error("Get Posts Since Timestamp", str(e))
            return []
    
    @ensure_naive_datetimes
    def _should_post_tech_content(self, market_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if tech content should be posted, and select a topic
        Uses calendar day-based counters with midnight reset
        Respects daily limits with clear logging

        Args:
            market_data: Optional market data for context

        Returns:
            Tuple of (should_post, topic_data)
        """
        try:
            # Check if tech content is enabled
            if not config.TECH_CONTENT_CONFIG.get('enabled', False):
                logger.logger.debug("Tech content posting disabled in configuration")
                return False, {}
    
            # Get daily post metrics with calendar day reset
            daily_metrics = self._get_daily_tech_post_count()
    
            # Check if we've hit daily maximum for today - log clearly
            if daily_metrics['max_reached']:
                logger.logger.info(
                    f"Maximum daily tech posts reached ({daily_metrics['tech_posts_today']}/"
                    f"{daily_metrics['max_daily_posts']})"
                )
                return False, {}
    
            # Analyze tech topics
            tech_analysis = self._analyze_tech_topics(market_data)
    
            if not tech_analysis.get('enabled', False):
                logger.logger.debug("Tech analysis not enabled or failed")
                return False, {}
    
            # Check if we have candidate topics
            candidates = tech_analysis.get('candidate_topics', [])
            if not candidates:
                logger.logger.debug("No candidate tech topics available")
                return False, {}
    
            # Select top candidate
            selected_topic = candidates[0]
    
            # Check if enough time has passed since last tech post
            last_tech_post = daily_metrics['last_tech_post']
            hours_since_last = safe_datetime_diff(strip_timezone(datetime.now()), last_tech_post) / 3600
            post_frequency = config.TECH_CONTENT_CONFIG.get('post_frequency', 4)
    
            if hours_since_last < post_frequency:
                logger.logger.info(
                    f"Not enough time since last tech post ({hours_since_last:.1f}h < {post_frequency}h)"
                )
                return False, selected_topic
    
            # At this point, we should post tech content
            logger.logger.info(
                f"Will post tech content about {selected_topic['category']} related to "
                f"{selected_topic['selected_token']} (Day count: {daily_metrics['tech_posts_today']}/"
                f"{daily_metrics['max_daily_posts']})"
            )
            return True, selected_topic
    
        except Exception as e:
            logger.log_error("Tech Content Decision", str(e))
            # On error, return False to prevent posting
            return False, {}

    def _post_tech_educational_content(self, market_data: Dict[str, Any]) -> bool:
        """
        Generate and post tech educational content
    
        Args:
            market_data: Market data for context
        
        Returns:
            Boolean indicating if content was successfully posted
        """
        try:
            # Check if we should post tech content
            should_post, topic_data = self._should_post_tech_content(market_data)
        
            if not should_post:
                return False
            
            # Generate tech content
            tech_category = topic_data.get('category', 'ai')  # Default to AI if not specified
            token = topic_data.get('selected_token', 'BTC')   # Default to BTC if not specified
        
            content, metadata = self._generate_tech_content(tech_category, token, market_data)
        
            # Post the content
            return self._post_tech_content(content, metadata)
        
        except Exception as e:
            logger.log_error("Tech Educational Content", str(e))
            return False
    @ensure_naive_datetimes
    def _cleanup(self) -> None:
        """Cleanup resources and save state"""
        try:
            # Stop prediction thread if running
            if self.prediction_thread_running:
                self.prediction_thread_running = False
                if self.prediction_thread and self.prediction_thread.is_alive():
                    self.prediction_thread.join(timeout=5)
                logger.logger.info("Stopped prediction thread")
           
            # Close browser
            if self.browser:
                logger.logger.info("Closing browser...")
                try:
                    self.browser.close_browser()
                    time.sleep(1)
                except Exception as e:
                    logger.logger.warning(f"Error during browser close: {str(e)}")
           
            # Save timeframe prediction data to database for persistence
            try:
                timeframe_state = {
                    "predictions": self.timeframe_predictions,
                    "last_post": {tf: ts.isoformat() for tf, ts in self.timeframe_last_post.items()},
                    "next_scheduled": {tf: ts.isoformat() for tf, ts in self.next_scheduled_posts.items()},
                    "accuracy": self.prediction_accuracy
                }
               
                # Store using the generic JSON data storage
                self.db._store_json_data(
                    data_type="timeframe_state",
                    data=timeframe_state
                )
                logger.logger.info("Saved timeframe state to database")
            except Exception as e:
                logger.logger.warning(f"Failed to save timeframe state: {str(e)}")
           
            # Close database connection
            if self.config:
                self.config.cleanup()
               
            logger.log_shutdown()
        except Exception as e:
            logger.log_error("Cleanup", str(e))

    @ensure_naive_datetimes
    def _ensure_datetime(self, value) -> datetime:
        """
        Convert value to datetime if it's a string, ensuring timezone-naive datetime
        
        Args:
            value: Value to convert
            
        Returns:
            Datetime object (timezone-naive)
        """
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return strip_timezone(dt)
            except ValueError:
                logger.logger.warning(f"Could not parse datetime string: {value}")
                return strip_timezone(datetime.min)
        elif isinstance(value, datetime):
            return strip_timezone(value)
        return strip_timezone(datetime.min)

    def _get_crypto_data(self) -> Optional[Dict[str, Any]]:
        """
        Enhanced M4 optimized crypto data fetching with aggressive performance optimization
        while respecting API free tier limitations.

        M4 Optimizations:
        - Smart request batching for free tier efficiency
        - Parallel processing where rate limits allow
        - Polars vectorized data transformation
        - Intelligent retry with exponential backoff
        - Partial data resilience with detailed logging
        - Dynamic API provider selection for reliability

        Returns:
            Dictionary of token data (100% backward compatible format)
        """
        import time
        import asyncio

        try:
            start_time = time.time()
            logger.logger.info("🚀 M4 Enhanced crypto data pipeline starting...")
        
            # SMART BATCHING STRATEGY - Optimize for free tier
            # CoinGecko free tier: ~10-30 calls/minute, so batch everything into minimal calls
            try:
                # Get all required token IDs in one efficient call
                token_ids = list(self.target_chains.values())  # ['bitcoin', 'ethereum', etc.]
            
                # Build optimized parameters for single batch call
                params = {
                    **config.get_coingecko_params(),
                    'ids': ','.join(token_ids), 
                    'sparkline': True,
                    'price_change_percentage': '1h,24h,7d',  # Get all timeframes at once
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': len(token_ids),
                    'page': 1
                }
            
                logger.logger.debug(f"📡 Fetching {len(token_ids)} tokens in single optimized call")
            
                # ENHANCED API CALL with retry strategy
                raw_data = self._fetch_with_smart_retry(params)
            
                if not raw_data:
                    logger.logger.error("❌ Failed to fetch any market data")
                    return None
                
                # Store original format for later processing
                is_dict_format = isinstance(raw_data, dict)
                
                # Convert dict to list for Polars processing if needed
                if is_dict_format:
                    logger.logger.debug("🔄 Converting dictionary data to list format for processing")
                    list_data = []
                    for key, value in raw_data.items():
                        if isinstance(value, dict):
                            list_data.append(value)
                    
                    if list_data:
                        logger.logger.info(f"✅ Raw data converted: {len(list_data)} items")
                        # Use the converted list for Polars processing
                        polars_input = list_data
                    else:
                        logger.logger.warning("⚠️ Could not convert dictionary to list format")
                        # FIX: Force standardization instead of returning raw dict
                        logger.logger.info("🔄 Forcing standardization of dictionary data")
                        return self._standardize_market_data([])  # Return empty standardized format
                else:
                    # Use the raw list directly
                    logger.logger.info(f"✅ Raw data received: {len(raw_data)} items")
                    polars_input = raw_data
                
            except Exception as api_error:
                logger.logger.error(f"❌ API call failed: {str(api_error)}")
                return None
        
            # M4 POLARS OPTIMIZATION - Lightning-fast data processing
            if POLARS_AVAILABLE and pl is not None and len(polars_input) > 3:
                try:
                    logger.logger.debug("⚡ Using M4 Polars optimization for data processing")
                    processed_data = self._process_data_with_polars(polars_input)
                
                    if processed_data:
                        fetch_time = time.time() - start_time
                        # Count unique tokens by checking for duplicate references
                        unique_token_count = len(set(id(data) for data in processed_data.values()))
                        logger.logger.info(f"🎯 M4 Pipeline completed: {unique_token_count} tokens in {fetch_time:.3f}s")
                        return processed_data
                    else:
                        logger.logger.warning("⚠️ Polars processing returned empty data, falling back")
                except Exception as polars_error:
                    logger.logger.warning(f"⚠️ Polars optimization failed: {str(polars_error)}")
            
            # FIXED: Always ensure proper standardization
            if is_dict_format:
                # Force standardization of dictionary data
                logger.logger.debug("🔄 Forcing standardization of dictionary format data")
                # Convert dict back to list for standardization
                list_for_standardization = []
                for key, value in raw_data.items():
                    if isinstance(value, dict):
                        list_for_standardization.append(value)
                return self._standardize_market_data(list_for_standardization)
            else:
                # Standardize list data
                return self._standardize_market_data(raw_data)
            
        except Exception as e:
            logger.log_error("Enhanced Crypto Data Pipeline", str(e))
            return None

    def _fetch_with_smart_retry(self, params: Dict[str, Any], max_retries: int = 3) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Smart retry mechanism optimized for API free tier limitations
        Uses robust API manager that intelligently selects between providers
        
        Returns:
            Raw data in original format from the API provider 
            (List[Dict[str, Any]] for CoinGecko, Dict[str, Any] for API Manager)
        """
        import time
        import random

        for attempt in range(max_retries):
            try:
                # Add jitter to prevent thundering herd on retries
                if attempt > 0:
                    jitter = random.uniform(0.5, 2.0)
                    backoff_time = (2 ** attempt) + jitter  # Exponential backoff with jitter
                    logger.logger.debug(f"🔄 Retry {attempt + 1}/{max_retries} after {backoff_time:.1f}s")
                    time.sleep(backoff_time)
            
                # Respect rate limits with intelligent timing
                if hasattr(self, '_last_api_call_time'):
                    time_since_last = time.time() - self._last_api_call_time
                    min_interval = 60.0  # 60 seconds between calls for free tier safety
                
                    if time_since_last < min_interval:
                        sleep_time = min_interval - time_since_last
                        logger.logger.debug(f"⏱️ Rate limit protection: sleeping {sleep_time:.1f}s")
                        time.sleep(sleep_time)
            
                # Record call time for rate limiting
                self._last_api_call_time = time.time()
                
                # ROBUST API CALL - Uses intelligent provider selection and automatic failover
                data = self.api_manager.get_market_data(
                    params=params,
                    timeframe="24h",
                    priority_tokens=getattr(self, 'priority_tokens', None),
                    include_price_history=True
                )
                
                if data:
                    provider_used = data[0].get('_provider', 'unknown') if data else 'unknown'
                    fallback_used = data[0].get('_provider_fallback_used', False) if data else False
                    
                    if fallback_used:
                        logger.logger.info(f"✅ Data retrieved via failover to {provider_used} on attempt {attempt + 1}")
                    else:
                        logger.logger.debug(f"✅ Data retrieved from {provider_used} on attempt {attempt + 1}")
                    
                    return data
                else:
                    logger.logger.warning(f"⚠️ Empty data received on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    
            except Exception as api_error:
                error_msg = str(api_error).lower()
            
                # Check for rate limit indicators
                if any(indicator in error_msg for indicator in ['rate limit', '429', 'too many requests']):
                    logger.logger.warning(f"🚫 Rate limit detected on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        # Longer wait for rate limits
                        rate_limit_wait = 60 + random.uniform(10, 30)
                        logger.logger.info(f"⏳ Rate limit cooldown: {rate_limit_wait:.1f}s")
                        time.sleep(rate_limit_wait)
                        continue
            
                # Check for temporary server issues
                elif any(indicator in error_msg for indicator in ['502', '503', '504', 'timeout', 'connection']):
                    logger.logger.warning(f"🌐 Server issue detected on attempt {attempt + 1}: {str(api_error)}")
                    if attempt < max_retries - 1:
                        continue
            
                # For other errors, log and potentially retry
                else:
                    logger.logger.error(f"❌ API error on attempt {attempt + 1}: {str(api_error)}")
                    if attempt < max_retries - 1:
                        continue

        logger.logger.error(f"❌ All {max_retries} retry attempts failed")
        return None

    def _process_data_with_polars(self, raw_data: List[Dict[str, Any]], timeframe: str = "24h") -> Optional[Dict[str, Any]]:
        """
        M4 Polars optimization for lightning-fast data processing with timeframe-aware sparkline handling
        """
        try:
            start_time = time.time()
        
            # Validate input data
            if not raw_data or not isinstance(raw_data, list):
                logger.logger.warning("⚠️ Invalid raw data for Polars processing")
                return None
        
            # Filter valid items
            valid_items = []
            for item in raw_data:
                if isinstance(item, dict) and 'id' in item and 'symbol' in item:
                    valid_items.append(item)
                else:
                    logger.logger.debug(f"⚠️ Skipping invalid item: {type(item)}")
        
            if not valid_items:
                logger.logger.warning("⚠️ No valid items found for processing")
                return None
        
            logger.logger.debug(f"⚡ Processing {len(valid_items)} valid items with Polars for {timeframe} analysis")
        
            # Check if Polars is actually available before creating DataFrame
            if pl is None:
                logger.logger.error("⚠️ Polars not available - cannot process data")
                return None        
                        
            # Create Polars DataFrame
            df = pl.DataFrame(valid_items)
        
            # Enhanced symbol mapping with comprehensive coverage
            symbol_mapping = {
                'BTC': 'bitcoin', 
                'ETH': 'ethereum',
                'SOL': 'solana',
                'XRP': 'ripple',
                'BNB': 'binancecoin',
                'AVAX': 'avalanche-2',
                'DOT': 'polkadot',
                'UNI': 'uniswap',
                'NEAR': 'near',
                'AAVE': 'aave',
                'FIL': 'filecoin',
                'POL': 'matic-network',
                'TRUMP': 'official-trump',
                'KAITO': 'kaito'
            }
        
            # Add standard symbol column using mapping
            df = df.with_columns([
                pl.col('id').map_elements(lambda x: symbol_mapping.get(x, str(x).upper())).alias('standard_symbol')
            ])
        
            # ================================================================
            # 🎯 TIMEFRAME-AWARE SPARKLINE PROCESSING 🎯
            # ================================================================
            
            # Extract sparkline data safely with timeframe-specific handling
            if pl is not None:
                try:
                    # Handle sparkline extraction based on timeframe needs
                    if timeframe == "1h":
                        # For 1h analysis, we want the most recent portion of sparkline data
                        # CoinGecko sparkline_in_7d provides ~168 hourly points, we'll use last 24-48 hours
                        df = df.with_columns([
                            pl.when(pl.col('sparkline_in_7d').is_not_null())
                            .then(
                                pl.col('sparkline_in_7d').struct.field('price').list.tail(48)  # Last 48 hours for 1h analysis
                            )
                            .otherwise(pl.lit([]))
                            .alias('sparkline')
                        ])
                        logger.logger.debug(f"🕐 Configured sparkline for {timeframe}: last 48 hours of data")
                        
                    elif timeframe == "24h":
                        # For 24h analysis, use more data points but with some filtering
                        df = df.with_columns([
                            pl.when(pl.col('sparkline_in_7d').is_not_null())
                            .then(
                                pl.col('sparkline_in_7d').struct.field('price').list.tail(120)  # Last ~5 days for 24h analysis
                            )
                            .otherwise(pl.lit([]))
                            .alias('sparkline')
                        ])
                        logger.logger.debug(f"📅 Configured sparkline for {timeframe}: last 120 data points")
                        
                    elif timeframe == "7d":
                        # For 7d analysis, use the full sparkline dataset
                        df = df.with_columns([
                            pl.when(pl.col('sparkline_in_7d').is_not_null())
                            .then(pl.col('sparkline_in_7d').struct.field('price'))
                            .otherwise(pl.lit([]))
                            .alias('sparkline')
                        ])
                        logger.logger.debug(f"📊 Configured sparkline for {timeframe}: full 7-day dataset")
                        
                    else:
                        # Default to 24h configuration
                        df = df.with_columns([
                            pl.when(pl.col('sparkline_in_7d').is_not_null())
                            .then(pl.col('sparkline_in_7d').struct.field('price'))
                            .otherwise(pl.lit([]))
                            .alias('sparkline')
                        ])
                        logger.logger.warning(f"Unknown timeframe {timeframe}, using default sparkline configuration")
                        
                except Exception as sparkline_error:
                    logger.logger.debug(f"Sparkline processing error for {timeframe}: {str(sparkline_error)}")
                    df = df.with_columns([pl.lit([]).alias('sparkline')])
            else:
                logger.logger.warning("Polars not available for sparkline processing")
                return None
        
            # ================================================================
            # 🔧 VECTORIZED DATA TRANSFORMATION 🔧
            # ================================================================
            
            # Convert DataFrame to structured dictionary format (backward compatible)
            result = {}
            successful_tokens = []
            failed_tokens = []
        
            for row in df.iter_rows(named=True):
                try:
                    symbol = row['standard_symbol']
                    token_id = row['id']
                
                    # Build standardized token data structure with timeframe-appropriate sparkline
                    token_data = {
                        'current_price': float(row['current_price']),
                        'volume': float(row['total_volume']),
                        'price_change_percentage_24h': float(row['price_change_percentage_24h']),
                        'sparkline': row.get('sparkline', []),  # Timeframe-appropriate sparkline data
                        'market_cap': float(row['market_cap']),
                        'market_cap_rank': int(row['market_cap_rank']),
                        'total_supply': row.get('total_supply'),
                        'max_supply': row.get('max_supply'),
                        'circulating_supply': float(row.get('circulating_supply', 0)),
                        'ath': float(row.get('ath', 0)),
                        'ath_change_percentage': float(row.get('ath_change_percentage', 0)),
                        '_processed_timeframe': timeframe,  # Add metadata about processing timeframe
                        '_sparkline_points': len(row.get('sparkline', []))  # Add metadata about data points
                    }
                
                    # Add timeframe-specific price changes if available
                    if 'price_change_percentage_1h_in_currency' in row:
                        token_data['price_change_percentage_1h_in_currency'] = float(row['price_change_percentage_1h_in_currency'])
                    if 'price_change_percentage_7d_in_currency' in row:
                        token_data['price_change_percentage_7d_in_currency'] = float(row['price_change_percentage_7d_in_currency'])
                
                    # Store with both symbol and ID for backward compatibility
                    result[symbol] = token_data
                    result[token_id] = token_data  # Duplicate reference for ID-based access
                
                    successful_tokens.append(symbol)
                
                except Exception as row_error:
                    failed_tokens.append(row.get('id', 'unknown'))
                    logger.logger.debug(f"⚠️ Row processing error for {row.get('id', 'unknown')}: {str(row_error)}")
                    continue
        
            # ================================================================
            # 💾 CONCURRENT DATABASE STORAGE 💾
            # ================================================================
            
            # Store all data in parallel
            if result:
                try:
                    self._store_market_data_batch(result, successful_tokens)
                except Exception as storage_error:
                    logger.logger.warning(f"⚠️ Batch storage failed: {str(storage_error)}")
                    # Continue anyway - data fetching succeeded
        
            # ================================================================
            # 📊 LOG PROCESSING RESULTS 📊
            # ================================================================
            
            processing_time = time.time() - start_time
            logger.logger.info(f"⚡ Polars processing completed in {processing_time:.3f}s for {timeframe} analysis")
            logger.logger.info(f"✅ Successfully processed: {len(successful_tokens)} tokens")
        
            if failed_tokens:
                logger.logger.warning(f"⚠️ Failed to process: {len(failed_tokens)} tokens: {', '.join(failed_tokens[:5])}")
        
            # Apply final symbol corrections for known issues
            result = self._apply_symbol_corrections(result)
            
            # Add processing metadata to result
            if result:
                # Add global metadata
                processing_metadata = {
                    '_processing_timeframe': timeframe,
                    '_processing_time': processing_time,
                    '_successful_count': len(successful_tokens),
                    '_failed_count': len(failed_tokens),
                    '_total_tokens': len(successful_tokens) + len(failed_tokens)
                }
                
                # Add metadata to each token for debugging
                for token_key, token_data in result.items():
                    if isinstance(token_data, dict):
                        token_data.update(processing_metadata)
        
            return result
        
        except Exception as e:
            logger.log_error("Polars Data Processing", str(e))
            return None

    def _store_market_data_batch(self, market_data: Dict[str, Any], successful_tokens: List[str]) -> None:
        """
        Batch store market data for better performance
        """
        try:
            # Store data for each successful token
            for symbol in successful_tokens:
                if symbol in market_data:
                    self.db.store_market_data(symbol, market_data[symbol])
        
            logger.logger.debug(f"💾 Batch stored {len(successful_tokens)} tokens to database")
        
        except Exception as e:
            logger.logger.warning(f"⚠️ Batch storage error: {str(e)}")
            # Try individual storage as fallback
            for symbol in successful_tokens:
                try:
                    if symbol in market_data:
                        self.db.store_market_data(symbol, market_data[symbol])
                except Exception as individual_error:
                    logger.logger.debug(f"⚠️ Individual storage failed for {symbol}: {str(individual_error)}")

    def _apply_symbol_corrections(self, formatted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply final symbol corrections for known mapping issues
        """
        try:
            # Known corrections
            corrections = {'MATIC': 'POL'}
        
            for old_symbol, new_symbol in corrections.items():
                if old_symbol in formatted_data and new_symbol not in formatted_data:
                    formatted_data[new_symbol] = formatted_data[old_symbol]
                    logger.logger.debug(f"🔄 Applied symbol correction: {old_symbol} → {new_symbol}")
        
            return formatted_data
        
        except Exception as e:
            logger.logger.debug(f"⚠️ Symbol correction error: {str(e)}")
            return formatted_data

    @ensure_naive_datetimes
    def _load_saved_timeframe_state(self) -> None:
        """Load previously saved timeframe state from database with enhanced datetime handling"""
        try:
            # Query the latest timeframe state
            conn, cursor = self.db._get_connection()
        
            cursor.execute("""
                SELECT data 
                FROM generic_json_data 
                WHERE data_type = 'timeframe_state'
                ORDER BY timestamp DESC
                LIMIT 1
            """)
        
            result = cursor.fetchone()
        
            if not result:
                logger.logger.info("No saved timeframe state found")
                return
            
            # Parse the saved state
            state_json = result[0]
            state = json.loads(state_json)
        
            # Restore timeframe predictions
            for timeframe, predictions in state.get("predictions", {}).items():
                self.timeframe_predictions[timeframe] = predictions
        
            # Restore last post times with proper datetime handling
            for timeframe, timestamp in state.get("last_post", {}).items():
                try:
                    # Convert string to datetime and ensure it's timezone-naive
                    dt = datetime.fromisoformat(timestamp)
                    self.timeframe_last_post[timeframe] = strip_timezone(dt)
                    logger.logger.debug(f"Restored last post time for {timeframe}: {self.timeframe_last_post[timeframe]}")
                except (ValueError, TypeError) as e:
                    # If timestamp can't be parsed, use a safe default
                    logger.logger.warning(f"Could not parse timestamp for {timeframe} last post: {str(e)}")
                    self.timeframe_last_post[timeframe] = strip_timezone(datetime.now() - timedelta(hours=3))
        
            # Restore next scheduled posts with proper datetime handling
            for timeframe, timestamp in state.get("next_scheduled", {}).items():
                try:
                    # Convert string to datetime and ensure it's timezone-naive
                    dt = datetime.fromisoformat(timestamp)
                    scheduled_time = strip_timezone(dt)
                
                    # If scheduled time is in the past, reschedule
                    now = strip_timezone(datetime.now())
                    if scheduled_time < now:
                        delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                        self.next_scheduled_posts[timeframe] = now + timedelta(hours=delay_hours)
                        logger.logger.debug(f"Rescheduled {timeframe} post for {self.next_scheduled_posts[timeframe]}")
                    else:
                        self.next_scheduled_posts[timeframe] = scheduled_time
                        logger.logger.debug(f"Restored next scheduled time for {timeframe}: {self.next_scheduled_posts[timeframe]}")
                except (ValueError, TypeError) as e:
                    # If timestamp can't be parsed, set a default
                    logger.logger.warning(f"Could not parse timestamp for {timeframe} next scheduled post: {str(e)}")
                    delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                    self.next_scheduled_posts[timeframe] = strip_timezone(datetime.now() + timedelta(hours=delay_hours))
        
            # Restore accuracy tracking
            self.prediction_accuracy = state.get("accuracy", {timeframe: {'correct': 0, 'total': 0} for timeframe in self.timeframes})
        
            # Debug log the restored state
            logger.logger.debug("Restored timeframe state:")
            for tf in self.timeframes:
                last_post = self.timeframe_last_post.get(tf)
                next_post = self.next_scheduled_posts.get(tf)
                logger.logger.debug(f"  {tf}: last={last_post}, next={next_post}")
        
            logger.logger.info("Restored timeframe state from database")
        
        except Exception as e:
            logger.log_error("Load Timeframe State", str(e))
            # Create safe defaults for all timing data
            now = strip_timezone(datetime.now())
            for timeframe in self.timeframes:
                self.timeframe_last_post[timeframe] = now - timedelta(hours=3)
                delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                self.next_scheduled_posts[timeframe] = now + timedelta(hours=delay_hours)
        
            logger.logger.warning("Using default timeframe state due to error")
    

    def _get_historical_price_data(self, token: str, hours: int, timeframe: Optional[str] = None) -> Union[List[Dict[str, Any]], str]:
        """
        Get historical price data directly from price_history table with robust token identifier handling
        Enhanced method with TokenMappingManager integration and database-driven token management
        Updated July 31, 2025 - No hardcoded mappings, pure database-driven approach
        
        Args:
            token: Token identifier (symbol like 'BTC' or CoinGecko ID like 'bitcoin')
            hours: Number of hours of historical data to retrieve
            timeframe: Optional timeframe parameter (for compatibility and optimization)
        
        Returns:
            List of dictionaries with price data, or "Never" if no data found
        """
        try:
            # Validate parameters
            if not token or not isinstance(token, str):
                logger.logger.warning("Invalid token parameter for historical price data")
                return "Never"
                
            if hours <= 0:
                logger.logger.warning(f"Invalid hours parameter: {hours}")
                hours = 24
                
            # Check if database is initialized
            if not hasattr(self, 'db') or self.db is None:
                logger.logger.error("❌ Database not initialized for _get_historical_price_data")
                from database import CryptoDatabase
                self.db = CryptoDatabase()
                logger.logger.info("✅ Created new database connection")
                
            logger.logger.debug(f"Getting historical price data for {token} (hours: {hours})")
            
            import traceback
            print(f"CALLING _get_connection from: {traceback.format_stack()[-2].strip()}")

            # Get database connection
            conn, cursor = self.db._get_connection()
            
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(hours=hours)
            
            # Step 1: Try original token first
            cursor.execute("""
                SELECT 
                    token,
                    timestamp,
                    price,
                    volume,
                    market_cap,
                    total_supply,
                    circulating_supply,
                    high_price,
                    low_price
                FROM price_history
                WHERE token = ? 
                AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (token, time_threshold.isoformat()))
            
            results = cursor.fetchall()
            
            # Step 2: If no results, try TokenMappingManager for symbol-to-coingecko mapping
            if not results and hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                logger.logger.debug(f"No results for {token}, trying TokenMappingManager")
                try:
                    coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(token)
                    if coingecko_id:
                        logger.logger.debug(f"✅ TokenMappingManager: Mapped {token} to {coingecko_id}")
                        cursor.execute("""
                            SELECT 
                                token,
                                timestamp,
                                price,
                                volume,
                                market_cap,
                                total_supply,
                                circulating_supply,
                                high_price,
                                low_price
                            FROM price_history
                            WHERE token = ? 
                            AND timestamp >= ?
                            ORDER BY timestamp ASC
                        """, (coingecko_id, time_threshold.isoformat()))
                        
                        results = cursor.fetchall()
                except Exception as token_mapper_error:
                    logger.logger.debug(f"TokenMappingManager lookup failed for {token}: {str(token_mapper_error)}")
            
            # Step 3: If still no results, try database lookup for symbol-to-coingecko mapping
            if not results:
                logger.logger.debug(f"No results from TokenMappingManager, trying database lookup for {token}")
                try:
                    cursor.execute("""
                        SELECT coin_id FROM coingecko_market_data 
                        WHERE UPPER(symbol) = ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (token.upper(),))
                    
                    result = cursor.fetchone()
                    if result and result['coin_id']:
                        coingecko_id = result['coin_id']
                        logger.logger.debug(f"✅ Database lookup: Mapped {token} to {coingecko_id}")
                        
                        cursor.execute("""
                            SELECT 
                                token,
                                timestamp,
                                price,
                                volume,
                                market_cap,
                                total_supply,
                                circulating_supply,
                                high_price,
                                low_price
                            FROM price_history
                            WHERE token = ? 
                            AND timestamp >= ?
                            ORDER BY timestamp ASC
                        """, (coingecko_id, time_threshold.isoformat()))
                        
                        results = cursor.fetchall()
                    else:
                        logger.logger.debug(f"No mapping found in database for {token}")
                except Exception as db_lookup_error:
                    logger.logger.debug(f"Database lookup failed for {token}: {str(db_lookup_error)}")
            
            # Step 4: Try uppercase token as final attempt
            if not results:
                normalized_token = token.upper()
                if normalized_token != token:
                    logger.logger.debug(f"Trying uppercase version: {normalized_token}")
                    cursor.execute("""
                        SELECT 
                            token,
                            timestamp,
                            price,
                            volume,
                            market_cap,
                            total_supply,
                            circulating_supply,
                            high_price,
                            low_price
                        FROM price_history
                        WHERE token = ? 
                        AND timestamp >= ?
                        ORDER BY timestamp ASC
                    """, (normalized_token, time_threshold.isoformat()))
                    
                    results = cursor.fetchall()
            
            logger.logger.debug(f"Found {len(results)} historical records for {token}")
            
            # If no results, return "Never" (as expected by extract_prices function)
            if not results:
                logger.logger.warning(f"No historical data found for {token} in last {hours} hours")
                return "Never"
            
            # Convert to list of dictionaries format expected by extract_prices
            historical_data = []
            for row in results:
                try:
                    row_dict = dict(row)
                    
                    # Ensure all values are properly typed
                    price_entry = {
                        'token': row_dict['token'],
                        'timestamp': row_dict['timestamp'],
                        'price': float(row_dict['price']) if row_dict['price'] is not None else 0.0,
                        'volume': float(row_dict['volume']) if row_dict['volume'] is not None else 0.0,
                        'market_cap': float(row_dict['market_cap']) if row_dict['market_cap'] is not None else 0.0,
                        'total_supply': float(row_dict['total_supply']) if row_dict['total_supply'] is not None else 0.0,
                        'circulating_supply': float(row_dict['circulating_supply']) if row_dict['circulating_supply'] is not None else 0.0,
                        'high_price': float(row_dict['high_price']) if row_dict['high_price'] is not None else 0.0,
                        'low_price': float(row_dict['low_price']) if row_dict['low_price'] is not None else 0.0
                    }
                    
                    historical_data.append(price_entry)
                    
                except (ValueError, TypeError, KeyError) as conversion_error:
                    logger.logger.debug(f"Error converting row data: {str(conversion_error)}")
                    continue
            
            logger.logger.debug(f"Successfully retrieved {len(historical_data)} price data points for {token}")
            
            # Log sample of data for debugging if we have data
            if historical_data:
                first_entry = historical_data[0]
                last_entry = historical_data[-1]
                logger.logger.debug(f"Price range for {token}: ${first_entry['price']:.6f} - ${last_entry['price']:.6f}")
                
                # Timeframe-specific optimization logging
                if timeframe:
                    logger.logger.debug(f"Historical data retrieved for {timeframe} analysis")
            
            return historical_data
            
        except Exception as e:
            logger.log_error(f"Get Historical Price Data - {token}", str(e))
            return "Never"
   
    @ensure_naive_datetimes
    def _get_token_timeframe_performance(self, token: str) -> Dict[str, Dict[str, Any]]:
        """
        Get prediction performance statistics for a token across all timeframes
        
        Args:
            token: Token symbol
            
        Returns:
            Dictionary of performance statistics by timeframe
        """
        try:
            result = {}
            
            # Gather performance for each timeframe
            for timeframe in self.timeframes:
                perf_stats = self.db.get_prediction_performance(token=token, timeframe=timeframe)
                
                if perf_stats:
                    result[timeframe] = {
                        "accuracy": perf_stats[0].get("accuracy_rate", 0),
                        "total": perf_stats[0].get("total_predictions", 0),
                        "correct": perf_stats[0].get("correct_predictions", 0),
                        "avg_deviation": perf_stats[0].get("avg_deviation", 0)
                    }
                else:
                    result[timeframe] = {
                        "accuracy": 0,
                        "total": 0,
                        "correct": 0,
                        "avg_deviation": 0
                    }
            
            # Get cross-timeframe comparison
            cross_comparison = self.db.get_prediction_comparison_across_timeframes(token)
            
            if cross_comparison:
                result["best_timeframe"] = cross_comparison.get("best_timeframe", {}).get("timeframe", "1h")
                result["overall"] = cross_comparison.get("overall", {})
            
            return result
            
        except Exception as e:
            logger.log_error(f"Get Token Timeframe Performance - {token}", str(e))
            return {tf: {"accuracy": 0, "total": 0, "correct": 0, "avg_deviation": 0} for tf in self.timeframes}
   
    @ensure_naive_datetimes
    def _get_all_active_predictions(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get all active predictions organized by timeframe and token
        
        Returns:
            Dictionary of active predictions by timeframe and token
        """
        try:
            result = {tf: {} for tf in self.timeframes}
            
            # Get active predictions from the database
            active_predictions = self.db.get_active_predictions()
            
            for prediction in active_predictions:
                timeframe = prediction.get("timeframe", "1h")
                token = prediction.get("token", "")
                
                if timeframe in result and token:
                    result[timeframe][token] = prediction
            
            # Merge with in-memory predictions which might be more recent
            for timeframe, predictions in self.timeframe_predictions.items():
                for token, prediction in predictions.items():
                    result.setdefault(timeframe, {})[token] = prediction
            
            return result
            
        except Exception as e:
            logger.log_error("Get All Active Predictions", str(e))
            return {tf: {} for tf in self.timeframes}

    @ensure_naive_datetimes
    def _evaluate_expired_timeframe_predictions(self) -> Dict[str, int]:
        """
        Find and evaluate expired predictions across all timeframes
        REBUILT with TokenMappingManager integration, database-driven token lookup,
        and volume data standardization
        
        Returns:
            Dictionary with count of evaluated predictions by timeframe
        """
        try:
            # Get expired unevaluated predictions
            all_expired = self.db.get_expired_unevaluated_predictions()
            
            if not all_expired:
                logger.logger.debug("No expired predictions to evaluate")
                return {tf: 0 for tf in self.timeframes}
                
            # Group by timeframe
            expired_by_timeframe = {tf: [] for tf in self.timeframes}
            
            for prediction in all_expired:
                timeframe = prediction.get("timeframe", "1h")
                if timeframe in expired_by_timeframe:
                    expired_by_timeframe[timeframe].append(prediction)
            
            # Get current market data for evaluation
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data for prediction evaluation")
                return {tf: 0 for tf in self.timeframes}
            
            # Track evaluated counts
            evaluated_counts = {tf: 0 for tf in self.timeframes}
            
            # Evaluate each prediction by timeframe
            for timeframe, predictions in expired_by_timeframe.items():
                for prediction in predictions:
                    token = prediction["token"]
                    prediction_id = prediction["id"]
                    
                    # ================================================================
                    # 🔥 GRACEFUL TOKEN/CHAIN LOOKUP WITH TOKEN MAPPING MANAGER 🔥
                    # ================================================================
                    token_data = None
                    lookup_attempts = []
                    
                    # ATTEMPT 1: Direct lookup (token as stored)
                    if token in market_data:
                        token_data = market_data[token]
                        lookup_attempts.append(f"✅ Direct lookup: '{token}' found")
                    else:
                        lookup_attempts.append(f"❌ Direct lookup: '{token}' not found")
                    
                    # ATTEMPT 2: Use TokenMappingManager to convert symbol → coingecko_id
                    if not token_data and hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                        try:
                            coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(token)
                            if coingecko_id and coingecko_id in market_data:
                                token_data = market_data[coingecko_id]
                                lookup_attempts.append(f"✅ TokenMappingManager: '{token}' → '{coingecko_id}' found")
                            else:
                                lookup_attempts.append(f"❌ TokenMappingManager: '{token}' → '{coingecko_id}' not found")
                        except Exception as token_mapper_error:
                            logger.logger.debug(f"TokenMappingManager lookup failed for {token}: {str(token_mapper_error)}")
                            lookup_attempts.append(f"❌ TokenMappingManager error: {str(token_mapper_error)}")
                    
                    # ATTEMPT 3: Try database lookup for symbol-to-coingecko mapping
                    if not token_data:
                        try:
                            conn, cursor = self.db._get_connection()
                            cursor.execute("""
                                SELECT coin_id FROM coingecko_market_data 
                                WHERE UPPER(symbol) = ?
                                ORDER BY timestamp DESC
                                LIMIT 1
                            """, (token.upper(),))
                            
                            result = cursor.fetchone()
                            if result and result['coin_id']:
                                coingecko_id = result['coin_id']
                                token_data = market_data.get(coingecko_id)
                                if token_data:
                                    lookup_attempts.append(f"✅ Database lookup: '{token}' → '{coingecko_id}' found")
                                else:
                                    lookup_attempts.append(f"❌ Database lookup: '{token}' → '{coingecko_id}' not in market data")
                            else:
                                lookup_attempts.append(f"❌ Database lookup: No mapping found for '{token}'")
                        except Exception as db_error:
                            logger.logger.warning(f"Database lookup for {token} failed: {str(db_error)}")
                            lookup_attempts.append(f"❌ Database lookup error: {str(db_error)}")
                    
                    # GRACEFUL HANDLING: If no lookup worked, skip this prediction
                    if not token_data:
                        logger.logger.warning(f"Token {token} not found in market data - skipping prediction evaluation (ID: {prediction_id})")
                        logger.logger.debug(f"Lookup attempts for {token}: {', '.join(lookup_attempts)}")
                        continue  # Skip to next prediction instead of crashing

                    # ================================================================
                    # 💰 GET CURRENT PRICE AND EVALUATE PREDICTION 💰
                    # ================================================================
                    current_price = token_data.get("current_price", 0)
                    
                    # Use _safe_get_volume method for consistent volume access if needed
                    if "volume" in token_data or "total_volume" in token_data or "volume_24h" in token_data:
                        volume = self._safe_get_volume(token_data)
                        logger.logger.debug(f"Volume for {token}: {volume}")
                    
                    if current_price == 0:
                        logger.logger.warning(f"Zero price for {token}, skipping evaluation")
                        continue
                        
                    # Record the outcome
                    result = self.db.record_prediction_outcome(prediction_id, current_price)
                    
                    if result:
                        logger.logger.debug(f"Evaluated {timeframe} prediction {prediction_id} for {token}")
                        evaluated_counts[timeframe] += 1
                    else:
                        logger.logger.error(f"Failed to evaluate {timeframe} prediction {prediction_id} for {token}")
            
            # Log evaluation summaries
            for timeframe, count in evaluated_counts.items():
                if count > 0:
                    logger.logger.info(f"Evaluated {count} expired {timeframe} predictions")
            
            # Update prediction performance metrics
            self._update_prediction_performance_metrics()
            
            return evaluated_counts
            
        except Exception as e:
            logger.log_error("Evaluate Expired Timeframe Predictions", str(e))
            return {tf: 0 for tf in self.timeframes}

    def _safe_get_volume(self, token_data):
        """
        Safely extract volume data from token dictionary using multiple possible keys
        
        Args:
            token_data: Token data dictionary
            
        Returns:
            Volume as float, or 0.0 if not found
        """
        # Try multiple volume keys in order of preference
        volume_keys = ['volume', 'total_volume', 'volume_24h', '24h_volume']
        
        for volume_key in volume_keys:
            volume = token_data.get(volume_key)
            if volume is not None:
                try:
                    return float(volume)
                except (ValueError, TypeError):
                    # Skip invalid values
                    continue
        
        # No valid volume found
        return 0.0

    @ensure_naive_datetimes
    def _update_prediction_performance_metrics(self) -> None:
        """Update in-memory prediction performance metrics from database"""
        try:
            # Get overall performance by timeframe
            for timeframe in self.timeframes:
                performance = self.db.get_prediction_performance(timeframe=timeframe)
                
                total_correct = sum(p.get("correct_predictions", 0) for p in performance)
                total_predictions = sum(p.get("total_predictions", 0) for p in performance)
                
                # Update in-memory tracking
                self.prediction_accuracy[timeframe] = {
                    'correct': total_correct,
                    'total': total_predictions
                }
            
            # Log overall performance
            for timeframe, stats in self.prediction_accuracy.items():
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    logger.logger.info(f"{timeframe} prediction accuracy: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                    
        except Exception as e:
            logger.log_error("Update Prediction Performance Metrics", str(e))

    def _analyze_volume_trend(self, current_volume: float, historical_volumes: List[float], timeframe: str = "1h") -> Tuple[float, str]:
        """
        Analyze volume trend over the window period, adjusted for timeframe
        
        Args:
            current_volume: Current volume value
            historical_data: Historical volume data
            timeframe: Timeframe for analysis
            
        Returns:
            Tuple of (percentage_change, trend_description)
        """
        historical_data = historical_volumes
        if not historical_data:
            return 0.0, "insufficient_data"
            
        try:
            # Adjust trend thresholds based on timeframe
            if timeframe == "1h":
                SIGNIFICANT_THRESHOLD = config.VOLUME_TREND_THRESHOLD  # Default (usually 15%)
                MODERATE_THRESHOLD = 5.0
            elif timeframe == "24h":
                SIGNIFICANT_THRESHOLD = 20.0  # Higher threshold for daily predictions
                MODERATE_THRESHOLD = 10.0
            else:  # 7d
                SIGNIFICANT_THRESHOLD = 30.0  # Even higher for weekly predictions
                MODERATE_THRESHOLD = 15.0
            
            # Calculate average volume excluding the current volume
            historical_data = historical_volumes
            avg_volume = statistics.mean(historical_volumes) if historical_volumes else current_volume
            
            # Calculate percentage change
            volume_change = ((current_volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
            
            # Determine trend based on timeframe-specific thresholds
            if volume_change >= SIGNIFICANT_THRESHOLD:
                trend = "significant_increase"
            elif volume_change <= -SIGNIFICANT_THRESHOLD:
                trend = "significant_decrease"
            elif volume_change >= MODERATE_THRESHOLD:
                trend = "moderate_increase"
            elif volume_change <= -MODERATE_THRESHOLD:
                trend = "moderate_decrease"
            else:
                trend = "stable"
                
            logger.logger.debug(
                f"Volume trend analysis ({timeframe}): {volume_change:.2f}% change from average. "
                f"Current: {current_volume:,.0f}, Avg: {avg_volume:,.0f}, "
                f"Trend: {trend}"
            )
            
            return volume_change, trend
            
        except Exception as e:
            logger.log_error(f"Volume Trend Analysis - {timeframe}", str(e))
            return 0.0, "error"

    @ensure_naive_datetimes
    def _generate_weekly_summary(self) -> bool:
        """
        Generate and post a weekly summary of predictions and performance across all timeframes
        
        Returns:
            Boolean indicating if summary was successfully posted
        """
        try:
            # Check if it's Sunday (weekday 6) and around midnight
            now = strip_timezone(datetime.now())
            if now.weekday() != 6 or now.hour != 0:
                return False
                
            # Get performance stats for all timeframes
            overall_stats = {}
            for timeframe in self.timeframes:
                performance_stats = self.db.get_prediction_performance(timeframe=timeframe)
                
                if not performance_stats:
                    continue
                    
                # Calculate overall stats for this timeframe
                total_correct = sum(p["correct_predictions"] for p in performance_stats)
                total_predictions = sum(p["total_predictions"] for p in performance_stats)
                
                if total_predictions > 0:
                    overall_accuracy = (total_correct / total_predictions) * 100
                    overall_stats[timeframe] = {
                        "accuracy": overall_accuracy,
                        "total": total_predictions,
                        "correct": total_correct
                    }
                    
                    # Get token-specific stats
                    token_stats = {}
                    for stat in performance_stats:
                        token = stat["token"]
                        if stat["total_predictions"] > 0:
                            token_stats[token] = {
                                "accuracy": stat["accuracy_rate"],
                                "total": stat["total_predictions"]
                            }
                    
                    # Sort tokens by accuracy
                    sorted_tokens = sorted(token_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True)
                    overall_stats[timeframe]["top_tokens"] = sorted_tokens[:3]
                    overall_stats[timeframe]["bottom_tokens"] = sorted_tokens[-3:] if len(sorted_tokens) >= 3 else []
            
            if not overall_stats:
                return False
                
            # Generate report
            report = "📊 WEEKLY PREDICTION SUMMARY 📊\n\n"
            
            # Add summary for each timeframe
            for timeframe, stats in overall_stats.items():
                if timeframe == "1h":
                    display_tf = "1 HOUR"
                elif timeframe == "24h":
                    display_tf = "24 HOUR"
                else:  # 7d
                    display_tf = "7 DAY"
                    
                report += f"== {display_tf} PREDICTIONS ==\n"
                report += f"Overall Accuracy: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})\n\n"
                
                if stats.get("top_tokens"):
                    report += "Top Performers:\n"
                    for token, token_stats in stats["top_tokens"]:
                        report += f"#{token}: {token_stats['accuracy']:.1f}% ({token_stats['total']} predictions)\n"
                        
                if stats.get("bottom_tokens"):
                    report += "\nBottom Performers:\n"
                    for token, token_stats in stats["bottom_tokens"]:
                        report += f"#{token}: {token_stats['accuracy']:.1f}% ({token_stats['total']} predictions)\n"
                        
                report += "\n"
                
            # Ensure report isn't too long
            max_length = config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
            if len(report) > max_length:
                # Truncate report intelligently
                sections = report.split("==")
                shortened_report = sections[0]  # Keep header
                
                # Add as many sections as will fit
                for section in sections[1:]:
                    if len(shortened_report + "==" + section) <= max_length:
                        shortened_report += "==" + section
                    else:
                        break
                        
                report = shortened_report
            
            # Post the weekly summary
            return self._post_analysis(report, timeframe="summary")
            
        except Exception as e:
            logger.log_error("Weekly Summary", str(e))
            return False

    def _prioritize_tokens(self, available_tokens: List[str], market_data: Dict[str, Any]) -> List[str]:
        """
        Enhanced M4 optimized token prioritization with aggressive alpha-seeking scoring
        for multi-million futures trading. Implements neural engine decision-making with:

        - Dynamic timeframe weighting based on market conditions
        - Non-linear factor interactions and momentum confluence
        - Persistence & acceleration scoring
        - Enhanced smart money detection
        - Market regime awareness
        - Aggressive scoring for maximum opportunity capture

        M4 Optimizations:
        - Polars vectorized calculations (10-50x faster)
        - Numba JIT compilation for math operations
        - Batched database queries
        - Parallel timeframe processing

        Args:
            available_tokens: List of available token symbols
            market_data: Market data dictionary
        
        Returns:
            Prioritized list of token symbols (highest alpha potential first)
        """
        import math
        import statistics

        try:
            # Early return for small datasets - simple prioritization
            if len(available_tokens) < 3:
                # Sort by absolute 24h price change (most active first)
                token_changes = []
                for token in available_tokens:
                    if token in market_data and isinstance(market_data[token], dict):
                        change = market_data[token].get('price_change_percentage_24h', 0)
                        change = float(change) if isinstance(change, (int, float)) else 0.0
                        token_changes.append((token, abs(change)))
                    else:
                        token_changes.append((token, 0.0))
                
                # Sort by absolute change (most volatile first for alpha opportunities)
                token_changes.sort(key=lambda x: x[1], reverse=True)
                return [token for token, _ in token_changes]

            logger.logger.info(f"Enhanced alpha prioritization for {len(available_tokens)} tokens")

            # M4 POLARS OPTIMIZATION - Vectorized Data Preparation
            if POLARS_AVAILABLE and pl is not None and len(available_tokens) >= 3:
                start_time = time.time()
            
                # Build comprehensive token dataset for vectorized processing
                token_records = []
                for token in available_tokens:
                    if token in market_data and isinstance(market_data[token], dict):
                        data = market_data[token]
                    
                        # Safely extract and convert all market metrics
                        try:
                            record = {
                                'token': token,
                                'current_price': float(data.get('current_price', 0)),
                                'price_change_1h': float(data.get('price_change_percentage_1h', 0)),
                                'price_change_24h': float(data.get('price_change_percentage_24h', 0)),
                                'price_change_7d': float(data.get('price_change_percentage_7d', 0)),
                                'volume': float(data.get('total_volume', data.get('volume', 0))),
                                'market_cap': float(data.get('market_cap', 0)),
                                'market_cap_rank': int(data.get('market_cap_rank', 999)),
                                'ath_change': float(data.get('ath_change_percentage', 0)),
                                'circulating_supply': float(data.get('circulating_supply', 0)),
                                'total_supply': float(data.get('total_supply', 0)),
                                'max_supply': float(data.get('max_supply', 0))
                            }
                            token_records.append(record)
                        except (ValueError, TypeError) as conversion_error:
                            logger.logger.debug(f"Data conversion error for {token}: {str(conversion_error)}")
                            # Skip tokens with bad data
                            continue
            
                if not token_records:
                    logger.logger.error("No valid token records could be created")
                    return []
            
                # Create Polars DataFrame for vectorized operations
                df = pl.DataFrame(token_records)
            
                # BULLETPROOF MARKET REGIME DETECTION
                try:
                    # Extract and rigorously clean price changes
                    price_changes_raw = df['price_change_24h'].to_list()
                    
                    # Filter to only valid numeric values
                    price_changes_clean = []
                    for value in price_changes_raw:
                        try:
                            if value is not None:
                                numeric_value = float(value)
                                # Only include finite, reasonable numbers
                                if (not math.isnan(numeric_value) and 
                                    not math.isinf(numeric_value) and
                                    -100 <= numeric_value <= 100):  # Reasonable price change range
                                    price_changes_clean.append(numeric_value)
                        except (ValueError, TypeError, OverflowError):
                            continue
                    
                    # Calculate market regime with clean data
                    if len(price_changes_clean) > 1:
                        market_volatility = statistics.stdev(price_changes_clean)
                        market_trend_strength = statistics.mean(price_changes_clean)
                    else:
                        market_volatility = 5.0
                        market_trend_strength = 0.0
                        
                except Exception as regime_error:
                    logger.logger.debug(f"Market regime detection error: {str(regime_error)}")
                    market_volatility = 5.0
                    market_trend_strength = 0.0
            
                # Determine market regime (now bulletproof)
                if market_volatility > 8.0:
                    regime = "high_volatility"
                    regime_multiplier = 1.4  # Aggressive in volatile markets
                elif market_volatility < 3.0:
                    regime = "low_volatility" 
                    regime_multiplier = 1.1  # Moderate in stable markets
                else:
                    regime = "normal"
                    regime_multiplier = 1.25  # Standard aggressive
            
                bull_market = market_trend_strength > 2.0
                bear_market = market_trend_strength < -2.0
            
                logger.logger.debug(f"Market regime: {regime}, trend: {market_trend_strength:.2f}%, volatility: {market_volatility:.2f}%")
            
                # Dynamic timeframe weighting based on regime
                if regime == "high_volatility":
                    tf_weights = {'1h': 0.5, '24h': 0.35, '7d': 0.15}
                elif regime == "low_volatility":
                    tf_weights = {'1h': 0.25, '24h': 0.35, '7d': 0.4}
                else:
                    tf_weights = {'1h': 0.4, '24h': 0.4, '7d': 0.2}
            
                # ENHANCED MOMENTUM SCORING - Bulletproof with null handling
                df = df.with_columns([
                    # Raw momentum components with safe math
                    (pl.col('price_change_1h').fill_null(0).abs() * 2.0).alias('momentum_1h_raw'),
                    (pl.col('price_change_24h').fill_null(0).abs() * 1.5).alias('momentum_24h_raw'), 
                    (pl.col('price_change_7d').fill_null(0).abs() * 1.0).alias('momentum_7d_raw'),
            
                    # Acceleration detection with division protection
                    (pl.col('price_change_1h').fill_null(0) / 
                    pl.max_horizontal([pl.col('price_change_24h').fill_null(0).abs(), pl.lit(0.01)])).alias('acceleration_factor'),
            
                    # Trend persistence (multi-timeframe alignment)
                    ((pl.col('price_change_1h').fill_null(0) > 0).cast(pl.Int32) + 
                    (pl.col('price_change_24h').fill_null(0) > 0).cast(pl.Int32) + 
                    (pl.col('price_change_7d').fill_null(0) > 0).cast(pl.Int32)).alias('trend_persistence'),
                ])
            
                # SMART MONEY & VOLUME ANALYSIS - Bulletproof
                df = df.with_columns([
                    # Volume momentum with log protection
                    (pl.max_horizontal([pl.col('volume').fill_null(1), pl.lit(1)]).log10() * 8.0).alias('volume_momentum'),
            
                    # Market cap efficiency (smaller caps = higher efficiency)
                    (1000.0 / pl.max_horizontal([pl.col('market_cap_rank').fill_null(999), pl.lit(1)])).alias('mcap_efficiency'),
            
                    # Supply scarcity factor with safe division
                    pl.when(pl.col('max_supply').fill_null(0) > 0)
                    .then((pl.col('max_supply').fill_null(0) - pl.col('circulating_supply').fill_null(0)) / 
                        pl.max_horizontal([pl.col('max_supply').fill_null(1), pl.lit(1)]) * 100)
                    .otherwise(0).alias('scarcity_factor'),
            
                    # ATH recovery potential for beaten down assets
                    pl.when(pl.col('ath_change').fill_null(0) < -20)
                    .then((pl.col('ath_change').fill_null(0).abs() - 20) * 0.5)
                    .otherwise(0).alias('recovery_potential')
                ])
            
                # NON-LINEAR FACTOR INTERACTIONS (Alpha confluence detection)
                df = df.with_columns([
                    # Price-Volume confluence (when momentum aligns)
                    (pl.col('momentum_24h_raw') * pl.col('volume_momentum') / 100.0).alias('pv_confluence'),
            
                    # Multi-timeframe alignment bonus (explosive setups)
                    pl.when(pl.col('trend_persistence') >= 2)
                    .then(pl.col('trend_persistence') * 15.0)
                    .otherwise(pl.col('trend_persistence') * 5.0).alias('alignment_bonus'),
            
                    # Explosive momentum detection
                    (pl.col('acceleration_factor').abs().clip(0, 5) * pl.col('trend_persistence') * 10.0).alias('explosive_score'),
            
                    # Liquidity efficiency score
                    (pl.col('mcap_efficiency') * pl.col('volume_momentum') / 50.0).alias('liquidity_score')
                ])
            
                # MARKET REGIME SPECIFIC ADJUSTMENTS
                if bull_market:
                    # Bull market: Amplify momentum and breakouts
                    df = df.with_columns([
                        (pl.col('momentum_24h_raw') * 1.3).alias('regime_adjusted_momentum'),
                        (pl.col('explosive_score') * 1.4).alias('regime_adjusted_explosive')
                    ])
                elif bear_market:
                    # Bear market: Focus on recovery plays and oversold bounces
                    df = df.with_columns([
                        (pl.col('recovery_potential') * 1.5).alias('regime_adjusted_momentum'),
                        (pl.col('alignment_bonus') * 0.8).alias('regime_adjusted_explosive')
                    ])
                else:
                    # Sideways market: Mean reversion and scarcity plays
                    df = df.with_columns([
                        (pl.col('pv_confluence') * 1.2).alias('regime_adjusted_momentum'),
                        (pl.col('scarcity_factor') * 1.3).alias('regime_adjusted_explosive')
                    ])
            
                # AGGRESSIVE ALPHA SCORING (30-100 range for maximum opportunity capture)
                df = df.with_columns([
                    # Weighted timeframe momentum
                    (pl.col('momentum_1h_raw') * tf_weights['1h'] + 
                    pl.col('momentum_24h_raw') * tf_weights['24h'] + 
                    pl.col('momentum_7d_raw') * tf_weights['7d']).alias('weighted_momentum'),
            
                    # Combined opportunity score
                    (pl.col('pv_confluence') * 0.25 +
                    pl.col('alignment_bonus') * 0.2 +
                    pl.col('explosive_score') * 0.2 +
                    pl.col('liquidity_score') * 0.15 +
                    pl.col('recovery_potential') * 0.1 +
                    pl.col('scarcity_factor') * 0.1).alias('opportunity_score')
                ])
            
                # FINAL ALPHA SCORE (Aggressive 30-100 range)
                df = df.with_columns([
                    # Base score from weighted momentum
                    (30 + pl.col('weighted_momentum').clip(0, 50)).alias('base_score'),
            
                    # Opportunity bonus
                    (pl.col('opportunity_score').clip(0, 20)).alias('opportunity_bonus')
                ])
            
                # Apply regime multiplier and finalize
                df = df.with_columns([
                    ((pl.col('base_score') + pl.col('opportunity_bonus')) * regime_multiplier).clip(30, 100).alias('alpha_score')
                ])
            
                # SORT BY ALPHA POTENTIAL (Highest first)
                df = df.sort('alpha_score', descending=True)
            
                # Extract prioritized token list
                prioritized_tokens = df['token'].to_list()
        
                polars_time = time.time() - start_time
                logger.logger.info(f"🚀 M4 Neural Engine completed: {len(prioritized_tokens)} tokens prioritized in {polars_time:.3f}s")
            
                # Log top alpha opportunities
                try:
                    top_performers = df.head(5)
                    for i, row in enumerate(top_performers.iter_rows(named=True)):
                        logger.logger.info(f"#{i+1} ALPHA TARGET: {row['token']} "
                                        f"(🎯Score: {row['alpha_score']:.1f}, "
                                        f"⚡Momentum: {row['weighted_momentum']:.1f}, "
                                        f"💎Opportunity: {row['opportunity_score']:.1f})")
                except Exception as log_error:
                    logger.logger.debug(f"Logging error: {str(log_error)}")
            
                return prioritized_tokens
            
            # No Polars available - fail fast
            logger.logger.error("🚨 CRITICAL: Polars library required for token prioritization is not available")
            return []

        except Exception as e:
            logger.log_error("Critical Error in Token Prioritization", str(e))
            logger.logger.error(f"🚨 CRITICAL: Token prioritization failed: {str(e)}")
            # Return empty list to signal failure
            return []

    def _generate_predictions(self, token: str, market_data: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Generate market predictions for a specific token at a specific timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for prediction
            
        Returns:
            Prediction data dictionary
        """
        try:
            logger.logger.info(f"Generating {timeframe} predictions for {token}")
        
            # Fix: Add try/except to handle the max() arg is an empty sequence error
            try:
                # Generate prediction for the specified timeframe
                prediction = self.prediction_engine._generate_predictions(
                    token=token,
                    market_data=market_data,
                    timeframe=timeframe
                )
            except ValueError as ve:
                # Handle the empty sequence error specifically
                if "max() arg is an empty sequence" in str(ve):
                    logger.logger.warning(f"Empty sequence error for {token} ({timeframe}), using fallback prediction")
                    # Create a basic fallback prediction
                    token_data = market_data.get(token, {})
                    current_price = token_data.get('current_price', 0)
                
                    # Adjust fallback values based on timeframe
                    if timeframe == "1h":
                        change_pct = 0.5
                        confidence = 60
                        range_factor = 0.01
                    elif timeframe == "24h":
                        change_pct = 1.2
                        confidence = 55
                        range_factor = 0.025
                    else:  # 7d
                        change_pct = 2.5
                        confidence = 50
                        range_factor = 0.05
                
                    prediction = {
                        "prediction": {
                            "price": current_price * (1 + change_pct/100),
                            "confidence": confidence,
                            "lower_bound": current_price * (1 - range_factor),
                            "upper_bound": current_price * (1 + range_factor),
                            "percent_change": change_pct,
                            "timeframe": timeframe
                        },
                        "rationale": f"Technical analysis based on recent price action for {token} over the {timeframe} timeframe.",
                        "sentiment": "NEUTRAL",
                        "key_factors": ["Technical analysis", "Recent price action", "Market conditions"],
                        "timestamp": strip_timezone(datetime.now())
                    }
                else:
                    # Re-raise other ValueError exceptions
                    raise
        
            # Store prediction in database
            prediction_id = self.db.store_prediction(token, prediction, timeframe=timeframe)
            logger.logger.info(f"Stored {token} {timeframe} prediction with ID {prediction_id}")
        
            return prediction
        
        except Exception as e:
            logger.log_error(f"Generate Predictions - {token} ({timeframe})", str(e))
            return {}
        
    def _normalize_prediction_for_analysis(self, prediction: Dict[str, Any], token: str) -> Dict[str, Any]:
        """
        Normalize trading prediction format back to analysis-compatible format
        
        This handles the format mismatch between trading features and analysis methods.
        Trading predictions have entry_price, stop_loss, take_profit while analysis 
        methods expect current_price, lower_bound, upper_bound.
        
        Args:
            prediction: Prediction in trading format
            token: Token symbol
            
        Returns:
            Prediction in analysis-compatible format
        """
        try:
            # Create a working copy
            normalized = prediction.copy()
            
            # Handle nested prediction structure
            if 'prediction' in prediction and isinstance(prediction['prediction'], dict):
                pred_data = prediction['prediction']
            else:
                pred_data = prediction
            
            # ================================================================
            # CONVERT TRADING FIELDS TO ANALYSIS FIELDS
            # ================================================================
            
            # 1. Current Price Mapping
            if 'current_price' not in normalized:
                current_price = (
                    pred_data.get('entry_price') or 
                    pred_data.get('current_price') or 
                    pred_data.get('price', 0)
                )
                normalized['current_price'] = current_price
            
            # 2. Price Prediction Mapping  
            if 'price' not in pred_data or pred_data.get('price') == pred_data.get('take_profit'):
                # If price field is just take_profit, recalculate for analysis
                entry_price = pred_data.get('entry_price', normalized['current_price'])
                take_profit = pred_data.get('take_profit', entry_price * 1.02)
                
                # For analysis, use a more conservative prediction between entry and take_profit
                analysis_price = entry_price + ((take_profit - entry_price) * 0.6)
                normalized['price'] = analysis_price
                
                # Update nested structure if it exists
                if 'prediction' in normalized:
                    normalized['prediction']['price'] = analysis_price
            
            # 3. Range Fields - Convert trading stops to analysis bounds
            entry_price = normalized['current_price']
            confidence = pred_data.get('confidence', 70)
            
            # Calculate analysis-appropriate bounds (not trading stops)
            confidence_factor = confidence / 100.0
            base_range = abs(pred_data.get('percent_change', 2.0)) / 100.0
            
            # Tighter bounds for analysis (not risk management)
            range_factor = base_range * (1 - confidence_factor * 0.5) * 0.3
            
            analysis_lower = entry_price * (1 - range_factor)
            analysis_upper = entry_price * (1 + range_factor)
            
            # Override trading bounds with analysis bounds
            normalized['lower_bound'] = analysis_lower
            normalized['upper_bound'] = analysis_upper
            
            # Update nested structure if it exists
            if 'prediction' in normalized:
                normalized['prediction']['lower_bound'] = analysis_lower
                normalized['prediction']['upper_bound'] = analysis_upper
            
            # 4. Percentage Change - Use analysis price, not take_profit
            if 'percent_change' in pred_data:
                analysis_percent = ((normalized['price'] - entry_price) / entry_price) * 100
                normalized['percent_change'] = analysis_percent
                
                if 'prediction' in normalized:
                    normalized['prediction']['percent_change'] = analysis_percent
            
            # ================================================================
            # PRESERVE TRADING FIELDS FOR COMPATIBILITY
            # ================================================================
            
            # Keep original trading fields for any trading-specific logic
            trading_fields = ['action', 'entry_price', 'stop_loss', 'take_profit', 'position_size']
            for field in trading_fields:
                if field in pred_data:
                    normalized[field] = pred_data[field]
            
            # Add format indicator
            normalized['_format_normalized'] = True
            normalized['_original_format'] = 'trading_enhanced'
            
            logger.logger.debug(f"✅ Normalized {token} prediction: analysis_price=${normalized['price']:.4f}, bounds=${analysis_lower:.4f}-${analysis_upper:.4f}")
            
            return normalized
            
        except Exception as e:
            logger.log_error(f"Prediction Normalization - {token}", str(e))
            # Return original prediction if normalization fails
            return prediction    

    @ensure_naive_datetimes
    def _run_analysis_cycle(self) -> None:
        """Run analysis and posting cycle for all tokens with multi-timeframe prediction integration
           and improved correlation reporting - M4 OPTIMIZED"""
        try:
            # M4 OPTIMIZATION: Performance monitoring
            cycle_start_time = time.time()
            logger.logger.debug("🚀 Starting M4-optimized analysis cycle")
        
            # First, evaluate any expired predictions
            prediction_start = time.time()
            self._evaluate_expired_timeframe_predictions()
            prediction_time = time.time() - prediction_start
            logger.logger.debug(f"⚡ Prediction evaluation: {prediction_time:.3f}s")
        
            logger.logger.debug("TIMEFRAME DEBUGGING INFO:")
            for tf in self.timeframes:
                logger.logger.debug(f"Timeframe: {tf}")
                last_post = self.timeframe_last_post.get(tf)
                next_scheduled = self.next_scheduled_posts.get(tf)
                logger.logger.debug(f"  last_post type: {type(last_post)}, value: {last_post}")
                logger.logger.debug(f"  next_scheduled type: {type(next_scheduled)}, value: {next_scheduled}")

            # Get market data
            fetch_start = time.time()
            raw_market_data = self._get_crypto_data()
            fetch_time = time.time() - fetch_start
            logger.logger.debug(f"⚡ Market data fetch: {fetch_time:.3f}s")

            # Debug raw data
            logger.logger.debug(f"🔍 _get_crypto_data returned type: {type(raw_market_data)}, length: {len(raw_market_data) if raw_market_data else 0}")
            if raw_market_data and len(raw_market_data) > 0:
                if isinstance(raw_market_data, list):
                    # Explicitly cast to list to help Pylance understand the type
                    market_list = list(raw_market_data)
                    first_item = market_list[0]
                    logger.logger.debug(f"🔍 First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}")
                elif isinstance(raw_market_data, dict):
                    # Explicitly cast to dict to help Pylance understand the type
                    market_dict = dict(raw_market_data)
                    logger.logger.debug(f"🔍 Dictionary keys: {list(market_dict.keys())[:3]}")
                else:
                    logger.logger.debug(f"🔍 Unexpected data type: {type(raw_market_data)}")
            
            # Standardize market data format
            standardize_start = time.time()
            if isinstance(raw_market_data, list):
                market_data = self._standardize_market_data(raw_market_data)
                logger.logger.debug(f"🔄 Standardized list to dict: {len(market_data)} tokens")
            else:
                market_data = raw_market_data
                logger.logger.debug(f"🔄 Data already in dict format: {len(market_data) if market_data else 0} tokens")

            standardize_time = time.time() - standardize_start
            logger.logger.debug(f"⚡ Data standardization: {standardize_time:.3f}s")

            # Debug standardized data
            if market_data and len(market_data) > 0:
                sample_token = list(market_data.keys())[0]
                sample_data = market_data[sample_token]
                current_price = sample_data.get('current_price', 'MISSING') if isinstance(sample_data, dict) else 'NOT_DICT'
                logger.logger.debug(f"🔍 Sample token {sample_token}: current_price = {current_price}")

            if not market_data:
                logger.logger.error("Failed to fetch market data")
                return
            
            # Add this debugging code right before the intersection check
            logger.logger.info(f"🔍 DEBUG: reference_tokens = {self.reference_tokens}")
            logger.logger.info(f"🔍 DEBUG: market_data keys = {list(market_data.keys())}")
            logger.logger.info(f"🔍 DEBUG: reference_tokens type = {type(self.reference_tokens)}")
            logger.logger.info(f"🔍 DEBUG: market_data type = {type(market_data)}")

            # Your existing intersection check
            common_tokens = [t for t in self.reference_tokens if t in market_data]

            # M4 OPTIMIZATION: Vectorized token filtering with Polars
            filter_start = time.time()

            # Check if database is initialized
            if not hasattr(self, 'db') or self.db is None:
                logger.logger.error("❌ Database not initialized for token filtering")
                from database import CryptoDatabase
                self.db = CryptoDatabase()
                logger.logger.info("✅ Created new database connection")

            # Ensure reference_tokens is initialized
            if not hasattr(self, 'reference_tokens') or not self.reference_tokens:
                logger.logger.warning("⚠️ Reference tokens not initialized, getting from database")
                try:
                    # Get tokens from database
                    database_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=25)
                    self.reference_tokens = database_tokens if database_tokens else ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
                    logger.logger.info(f"✅ Set reference_tokens from database: {self.reference_tokens[:5]}...")
                except Exception as db_error:
                    logger.logger.warning(f"❌ Database token query failed: {str(db_error)}")
                    self.reference_tokens = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
                    logger.logger.info(f"✅ Set fallback reference_tokens: {self.reference_tokens}")

            # Add diagnostic logging for market_data
            if not market_data:
                logger.logger.error("❌ CRITICAL: market_data is empty or None")
                return
            else:
                logger.logger.info(f"📊 Market data contains {len(market_data)} entries")
                # Log a few sample keys to verify format
                sample_keys = list(market_data.keys())[:5]
                logger.logger.info(f"📊 Sample market data keys: {sample_keys}")

            # Add diagnostic logging for reference tokens
            logger.logger.info(f"📊 Reference tokens: {len(self.reference_tokens)} tokens available")
            logger.logger.info(f"📊 Sample reference tokens: {self.reference_tokens[:5]}")

            # Check for any intersection between market_data keys and reference_tokens
            common_tokens = [t for t in self.reference_tokens if t in market_data]
            logger.logger.info(f"📊 Common tokens between reference_tokens and market_data: {len(common_tokens)}")
            if common_tokens:
                logger.logger.info(f"📊 Sample common tokens: {common_tokens[:5]}")
            else:
                logger.logger.error("❌ CRITICAL: No intersection between reference_tokens and market_data keys")
                # Try to debug case sensitivity issues
                reference_upper = [t.upper() for t in self.reference_tokens]
                market_upper = [k.upper() for k in market_data.keys()]
                common_upper = [t for t in reference_upper if t in market_upper]
                if common_upper:
                    logger.logger.warning(f"⚠️ Case sensitivity issue detected - {len(common_upper)} matches when case-insensitive")

            # Decide what type of content to prioritize
            decision_start = time.time()
            post_priority = self._decide_post_type(market_data)
            decision_time = time.time() - decision_start
            logger.logger.debug(f"⚡ Post type decision: {decision_time:.3f}s -> {post_priority}")
        
            post_success = False  # Track if any posting was successful

            # Act based on the decision
            if post_priority == "reply":
                # Prioritize finding and replying to posts
                reply_start = time.time()
                post_success = self._check_for_reply_opportunities(market_data)
                reply_time = time.time() - reply_start
                logger.logger.debug(f"⚡ Reply processing: {reply_time:.3f}s")
            
                if post_success:
                    logger.logger.info("Successfully posted replies based on priority decision")
                    return
                # Fall through to other post types if no reply opportunities

            elif post_priority == "prediction":
                # Prioritize prediction posts (try timeframe rotation first)
                pred_start = time.time()
                post_success = self._post_timeframe_rotation(market_data)
                pred_time = time.time() - pred_start
                logger.logger.debug(f"⚡ Prediction posting: {pred_time:.3f}s")
            
                if post_success:
                    logger.logger.info("Posted scheduled timeframe prediction based on priority decision")
                    return
                # Fall through to token-specific predictions for 1h timeframe

            elif post_priority == "correlation":
                # Generate and post correlation report using new format
                corr_start = time.time()
            
                # First, determine which timeframe to use for the report
                # Change pattern from using hour of day to something more strategic
                current_hour = datetime.now().hour
                market_volatility = self._calculate_market_volatility(market_data)
        
                # Select timeframe based on market conditions and time of day
                if market_volatility > 3.0:
                    # During high volatility, focus on shorter timeframes
                    report_timeframe = "1h"
                    logger.logger.info("Using 1h timeframe for correlation report due to high volatility")
                elif market_volatility < 1.0:
                    # During low volatility, use longer timeframes
                    report_timeframe = "7d"
                    logger.logger.info("Using 7d timeframe for correlation report due to low volatility")
                else:
                    # During normal volatility, rotate timeframes
                    if 0 <= current_hour < 8:
                        report_timeframe = "7d"  # Overnight, focus on weekly patterns
                    elif 8 <= current_hour < 16:
                        report_timeframe = "24h"  # During main trading hours, use daily
                    else:
                        report_timeframe = "1h"  # Evening, use hourly for recent activity
            
                    logger.logger.info(f"Using {report_timeframe} timeframe for correlation report based on time of day")
            
                # Check if enough time has passed since the last correlation report for this timeframe
                min_hours_between_reports = 4 if report_timeframe == "1h" else 12 if report_timeframe == "24h" else 48
        
                # Try to get last report time from memory or database
                last_report_time = None
        
                # If not found in memory, check database
                if not last_report_time and hasattr(self.db, 'get_last_correlation_report'):
                    try:
                        last_report = self.db.get_last_correlation_report(report_timeframe)
                        if last_report and 'timestamp' in last_report:
                            last_report_time = strip_timezone(datetime.fromisoformat(last_report['timestamp']))
                    except Exception as db_err:
                        logger.logger.debug(f"Error retrieving last correlation report from database: {str(db_err)}")

                if not last_report_time and hasattr(self.db, 'get_last_correlation_report'):
                    try:
                        # Add diagnostic logging
                        logger.logger.info(f"Attempting to retrieve last correlation report for {report_timeframe}")
                        
                        last_report = self.db.get_last_correlation_report(report_timeframe)  # type: ignore
                        
                        # Log the result
                        if last_report:
                            logger.logger.info(f"Found correlation report: {last_report.get('id', 'unknown')}, timestamp: {last_report.get('timestamp', 'unknown')}")
                        else:
                            logger.logger.info(f"No correlation report found for {report_timeframe}")
                            
                        if last_report and 'timestamp' in last_report:
                            last_report_time = strip_timezone(datetime.fromisoformat(last_report['timestamp']))
                    except Exception as db_err:
                        logger.logger.debug(f"Error retrieving last correlation report from database: {str(db_err)}")        
        
                # Default to old timestamp if still not found
                if not last_report_time:
                    last_report_time = strip_timezone(datetime.now() - timedelta(hours=min_hours_between_reports * 2))
        
                # Calculate hours since last report
                hours_since_last = safe_datetime_diff(datetime.now(), last_report_time) / 3600
        
                # Check if enough time has passed
                if hours_since_last >= min_hours_between_reports:
                    # Generate the correlation report
                    correlation_report = self._generate_correlation_report(market_data, timeframe=report_timeframe)
            
                    # Only post if report isn't empty and isn't a duplicate
                    if correlation_report and len(correlation_report) > 0:
                        if self._post_analysis(correlation_report, timeframe=report_timeframe):
                            corr_time = time.time() - corr_start
                            logger.logger.info(f"Posted {report_timeframe} correlation report successfully ({corr_time:.3f}s)")
                            post_success = True
                            return
                        else:
                            logger.logger.error(f"Failed to post {report_timeframe} correlation report")
                    else:
                        logger.logger.warning(f"Empty or duplicate {report_timeframe} correlation report, skipping")
                        # Try another timeframe if the first choice was a duplicate
                        alternate_timeframes = [tf for tf in self.timeframes if tf != report_timeframe]
                        # Pick a timeframe that hasn't been used recently
                        for alt_tf in alternate_timeframes:
                            alt_report = self._generate_correlation_report(market_data, timeframe=alt_tf)
                            if alt_report and len(alt_report) > 0:
                                if self._post_analysis(alt_report, timeframe=alt_tf):
                                    corr_time = time.time() - corr_start
                                    logger.logger.info(f"Posted alternate {alt_tf} correlation report successfully ({corr_time:.3f}s)")
                                    post_success = True
                                    return
                                else:
                                    logger.logger.error(f"Failed to post alternate {alt_tf} correlation report")
                else:
                    logger.logger.info(f"Not enough time has passed for {report_timeframe} correlation report "
                                    f"({hours_since_last:.1f}/{min_hours_between_reports}h)")
            
            elif post_priority == "tech":
                # Prioritize posting tech educational content
                tech_start = time.time()
                post_success = self._post_tech_educational_content(market_data)
                tech_time = time.time() - tech_start
                logger.logger.debug(f"⚡ Tech content processing: {tech_time:.3f}s")
            
                if post_success:
                    logger.logger.info("Posted tech educational content based on priority decision")
                    return
                # Fall through to other post types if tech posting failed

            # Initialize trigger_type with a default value to prevent NoneType errors
            trigger_type = "regular_interval"

            # If we haven't had any successful posts yet, try 1h predictions
            if not post_success:
                # M4 OPTIMIZATION: Enhanced token prioritization timing
                prioritize_start = time.time()
                
                # Ensure reference_tokens is properly initialized
                if not hasattr(self, 'reference_tokens') or not self.reference_tokens:
                    logger.logger.error("❌ CRITICAL: reference_tokens missing or empty for prioritization")
                    # Check if database is initialized
                    if not hasattr(self, 'db') or self.db is None:
                        logger.logger.error("❌ Database not initialized for token prioritization")
                        from database import CryptoDatabase
                        self.db = CryptoDatabase()
                        logger.logger.info("✅ Created new database connection")
                        
                    # Try to get tokens from database
                    try:
                        database_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=25)
                        if database_tokens:
                            self.reference_tokens = database_tokens
                            logger.logger.info(f"✅ Set reference_tokens from database: {self.reference_tokens[:5]}...")
                        else:
                            self.reference_tokens = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
                            logger.logger.warning("⚠️ Using fallback token list - database returned no tokens")
                    except Exception as db_error:
                        logger.logger.error(f"❌ Database token query failed: {str(db_error)}")
                        self.reference_tokens = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
                        logger.logger.warning("⚠️ Using fallback token list due to database error")
                
                # Now prioritize with validated reference_tokens
                available_tokens = self._prioritize_tokens(self.reference_tokens, market_data)
                prioritize_time = time.time() - prioritize_start
                logger.logger.debug(f"⚡ Token prioritization: {prioritize_time:.3f}s")
                
                # Check if we have tokens after prioritization
                if not available_tokens:
                    logger.logger.error("❌ No tokens available after prioritization - check market data")

                # For 1h predictions and regular updates, try each token until we find one that's suitable
                token_processing_start = time.time()
            
                for token_to_analyze in available_tokens:
                    token_start = time.time()
                
                    should_post, token_trigger_type = self._should_post_update(token_to_analyze, market_data, timeframe="1h")
    
                    if should_post:
                        # NEW: Add trigger validation here
                        trigger_worthy, worthiness_reason = self._validate_trigger_worthiness(
                            token_to_analyze, market_data, token_trigger_type, timeframe="1h"
                        )

                        if not trigger_worthy:
                            logger.logger.debug(f"Trigger not worthy for {token_to_analyze}: {worthiness_reason}")
                            continue

                        # Update the main trigger_type variable
                        trigger_type = token_trigger_type
                        logger.logger.info(f"Starting {token_to_analyze} analysis cycle - Trigger: {trigger_type}")

                        # Generate prediction for this token with 1h timeframe
                        prediction = self._generate_predictions(token_to_analyze, market_data, timeframe="1h")

                        if not prediction:
                            logger.logger.error(f"Failed to generate 1h prediction for {token_to_analyze}")
                            continue

                        # Get both standard analysis and prediction-focused content 
                        standard_analysis, storage_data = self._analyze_market_sentiment(
                            token_to_analyze, market_data, trigger_type, timeframe="1h"
                        )
                        prediction_tweet = self._format_prediction_tweet(token_to_analyze, prediction, market_data, timeframe="1h")

                        # Choose which type of content to post based on trigger and past posts
                        should_post_prediction = (
                            "prediction" in trigger_type or 
                            random.random() < 0.35  # 35% chance of posting prediction instead of analysis
                        )

                        if should_post_prediction:
                            analysis_to_post = prediction_tweet
                            if storage_data:
                                storage_data['is_prediction'] = True
                                storage_data['prediction_data'] = prediction
                        else:
                            analysis_to_post = standard_analysis
                            if storage_data:
                                storage_data['is_prediction'] = False

                        if not analysis_to_post:
                            logger.logger.error(f"Failed to generate content for {token_to_analyze}")
                            continue

                        # NEW: Validate content quality
                        content_worthy, final_content = self._validate_content_quality(analysis_to_post, token_to_analyze, "1h")

                        if not content_worthy:
                            self._handle_insufficient_content(token_to_analyze, "1h", final_content)
                            continue

                        # Check for duplicates
                        last_posts = self._get_last_posts_by_timeframe(timeframe="1h")
                        if not self._is_duplicate_analysis(final_content, last_posts, timeframe="1h"):
                            if self._post_analysis(final_content, timeframe="1h"):
                                # Only store in database after successful posting
                                if storage_data:
                                    storage_data['content'] = final_content  # Add the final content
                                    self.db.store_posted_content(**storage_data)

                                    token_time = time.time() - token_start
                                    logger.logger.info(
                                        f"Successfully posted {token_to_analyze} "
                                        f"{'prediction' if should_post_prediction else 'analysis'} - "
                                        f"Trigger: {trigger_type} ({token_time:.3f}s)"
                                    )

                                    # Store additional smart money metrics (existing code)
                                    if token_to_analyze in market_data:
                                        smart_money = self._analyze_smart_money_indicators(
                                            token_to_analyze, market_data[token_to_analyze], timeframe="1h"
                                        )
                                        self.db.store_smart_money_indicators(token_to_analyze, smart_money)

                                    # Store market comparison data
                                    vs_market = self._analyze_token_vs_market(token_to_analyze, market_data, timeframe="1h")
                                    if vs_market:
                                        self.db.store_token_market_comparison(
                                            token_to_analyze,
                                            vs_market.get('vs_market_avg_change', 0),
                                            vs_market.get('vs_market_volume_growth', 0),
                                            vs_market.get('outperforming_market', False),
                                            vs_market.get('correlations', {})
                                        )

                                post_success = True
                                return  # Successfully posted, exit the method
                            else:
                                logger.logger.error(f"Failed to post {token_to_analyze} {'prediction' if should_post_prediction else 'analysis'}")
                                continue
                        else:
                            logger.logger.info(f"Content still duplicate after validation for {token_to_analyze}")
                            continue
                    else:
                        logger.logger.debug(f"No significant {token_to_analyze} changes detected, trying another token")

                token_processing_time = time.time() - token_processing_start
                logger.logger.debug(f"⚡ Token processing loop: {token_processing_time:.3f}s")

            # If we've tried everything and still haven't posted anything, try tech content
            if not post_success and post_priority != "tech":  # Only if we haven't already tried tech
                tech_fallback_start = time.time()
                if self._post_tech_educational_content(market_data):
                    tech_fallback_time = time.time() - tech_fallback_start
                    logger.logger.info(f"Posted tech educational content as fallback ({tech_fallback_time:.3f}s)")
                    post_success = True
                    return

            # Alternatively try correlation reports
            if not post_success:
                corr_fallback_start = time.time()
            
                # Try all timeframes in order of priority based on time of day
                current_hour = datetime.now().hour
        
                # Determine priority order based on time of day
                if 0 <= current_hour < 8:
                    timeframe_priority = ["7d", "24h", "1h"]  # Overnight - longer timeframes first
                elif 8 <= current_hour < 16:
                    timeframe_priority = ["24h", "1h", "7d"]  # Trading hours - medium timeframes first
                else:
                    timeframe_priority = ["1h", "24h", "7d"]  # Evening - short timeframes first
        
                for report_timeframe in timeframe_priority:
                    correlation_report = self._generate_correlation_report(market_data, timeframe=report_timeframe)
            
                    if correlation_report and len(correlation_report) > 0:
                        if self._post_analysis(correlation_report, timeframe=report_timeframe):
                            corr_fallback_time = time.time() - corr_fallback_start
                            logger.logger.info(f"Posted {report_timeframe} correlation report as fallback ({corr_fallback_time:.3f}s)")
                            post_success = True
                            return

            # FINAL FALLBACK: If still no post, try reply opportunities as a last resort
            if not post_success:
                final_fallback_start = time.time()
                logger.logger.info("Checking for reply opportunities as ultimate fallback")
                if self._check_for_reply_opportunities(market_data):
                    final_fallback_time = time.time() - final_fallback_start
                    logger.logger.info(f"Successfully posted replies as fallback ({final_fallback_time:.3f}s)")
                    post_success = True
                    return

            # If we get here, we tried all tokens but couldn't post anything
            if not post_success:
                logger.logger.warning("Tried all available tokens but couldn't post any analysis or replies")

            # M4 OPTIMIZATION: Final performance summary
            total_cycle_time = time.time() - cycle_start_time
            logger.logger.info(f"🎯 M4 analysis cycle completed in {total_cycle_time:.3f}s (success: {post_success})")

        except Exception as e:
            logger.log_error("Token Analysis Cycle", str(e))
    
    def _is_tech_related_post(self, post):
        """
        Determine if a post is related to technology topics we're tracking
    
        Args:
            post: Post dictionary containing post content and metadata
        
        Returns:
            Boolean indicating if the post is tech-related
        """
        try:
            # Get post text
            post_text = post.get('text', '')
            if not post_text:
                return False
            
            # Convert to lowercase for case-insensitive matching
            post_text = post_text.lower()
        
            # Tech keywords to check for
            tech_keywords = [
                'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
                'llm', 'large language model', 'gpt', 'claude', 'chatgpt',
                'quantum', 'computing', 'blockchain technology', 'neural network',
                'transformer', 'computer vision', 'nlp', 'generative ai'
            ]
        
            # Check if any tech keyword is in the post text
            return any(keyword in post_text for keyword in tech_keywords)
        
        except Exception as e:
            logger.log_error("Tech Related Post Check", str(e))
            return False

    @ensure_naive_datetimes
    def _decide_post_type(self, market_data: Dict[str, Any]) -> str:
        """
        Make a strategic decision on what type of post to prioritize: prediction, analysis, reply, tech, or correlation
    
        Args:
            market_data: Market data dictionary
            
        Returns:
            String indicating the recommended action: "prediction", "analysis", "reply", "tech", or "correlation"
        """
        try:
            now = strip_timezone(datetime.now())
    
            # Initialize decision factors
            decision_factors = {
                'prediction': 0.0,
                'analysis': 0.0,
                'reply': 0.0,
                'correlation': 0.0,
                'tech': 0.0  # Added tech as a new decision factor
            }
    
            # Factor 1: Time since last post of each type
            # Use existing database methods instead of get_last_post_time
            try:
                # Get recent posts from the database
                recent_posts = self.db.get_recent_posts(hours=24)
        
                # Find the most recent posts of each type
                last_analysis_time = None
                last_prediction_time = None
                last_correlation_time = None
                last_tech_time = None  # Added tech time tracking
        
                for post in recent_posts:
                    # Convert timestamp to datetime if it's a string
                    post_timestamp = post.get('timestamp')
                    if isinstance(post_timestamp, str):
                        try:
                            post_timestamp = strip_timezone(datetime.fromisoformat(post_timestamp))
                        except ValueError:
                            continue
            
                    # Check if it's a prediction post
                    if post.get('is_prediction', False):
                        if post_timestamp is not None and (last_prediction_time is None or post_timestamp > last_prediction_time):
                            last_prediction_time = post_timestamp
                    # Check if it's a correlation post
                    elif 'CORRELATION' in post.get('content', '').upper():
                        if post_timestamp is not None and (last_correlation_time is None or post_timestamp > last_correlation_time):
                            last_correlation_time = post_timestamp
                    # Check if it's a tech post
                    elif post.get('tech_category', False) or post.get('tech_metadata', False) or 'tech_' in post.get('trigger_type', ''):
                        if post_timestamp is not None and (last_tech_time is None or post_timestamp > last_tech_time):
                            last_tech_time = post_timestamp
                    # Otherwise it's an analysis post
                    else:
                        if post_timestamp is not None and (last_analysis_time is None or post_timestamp > last_analysis_time):
                            last_analysis_time = post_timestamp

            except Exception as db_err:
                logger.logger.warning(f"Error retrieving recent posts: {str(db_err)}")
                last_analysis_time = now - timedelta(hours=12)  # Default fallback
                last_prediction_time = now - timedelta(hours=12)  # Default fallback
                last_correlation_time = now - timedelta(hours=48)  # Default fallback
                last_tech_time = now - timedelta(hours=24)  # Default fallback for tech
    
            # Set default values if no posts found
            if last_analysis_time is None:
                last_analysis_time = now - timedelta(hours=24)
            if last_prediction_time is None:
                last_prediction_time = now - timedelta(hours=24)
            if last_correlation_time is None:
                last_correlation_time = now - timedelta(hours=48)
            if last_tech_time is None:
                last_tech_time = now - timedelta(hours=24)
            
            # Calculate hours since each type of post using safe_datetime_diff
            hours_since_analysis = safe_datetime_diff(now, last_analysis_time) / 3600
            hours_since_prediction = safe_datetime_diff(now, last_prediction_time) / 3600
            hours_since_correlation = safe_datetime_diff(now, last_correlation_time) / 3600
            hours_since_tech = safe_datetime_diff(now, last_tech_time) / 3600
    
            # Check time since last reply (using our sanitized datetime)
            last_reply_time = strip_timezone(self._ensure_datetime(self.last_reply_time))
            hours_since_reply = safe_datetime_diff(now, last_reply_time) / 3600
    
            # Add time factors to decision weights (more time = higher weight)
            decision_factors['prediction'] += min(5.0, hours_since_prediction * 0.5)  # Cap at 5.0
            decision_factors['analysis'] += min(5.0, hours_since_analysis * 0.5)  # Cap at 5.0
            decision_factors['reply'] += min(5.0, hours_since_reply * 0.8)  # Higher weight for replies
            decision_factors['correlation'] += min(3.0, hours_since_correlation * 0.1)  # Lower weight for correlations
            decision_factors['tech'] += min(4.0, hours_since_tech * 0.6)  # Medium weight for tech content
    
            # Factor 2: Time of day considerations - adjust to audience activity patterns
            current_hour = now.hour
    
            # Morning hours (6-10 AM): Favor analyses, predictions and tech content for day traders
            if 6 <= current_hour <= 10:
                decision_factors['prediction'] += 2.0
                decision_factors['analysis'] += 1.5
                decision_factors['tech'] += 1.5  # Good time for educational content
                decision_factors['reply'] += 0.5
        
            # Mid-day (11-15): Balanced approach, slight favor to replies
            elif 11 <= current_hour <= 15:
                decision_factors['prediction'] += 1.0
                decision_factors['analysis'] += 1.0
                decision_factors['tech'] += 1.2  # Still good for tech content
                decision_factors['reply'] += 1.5
        
            # Evening hours (16-22): Strong favor to replies to engage with community
            elif 16 <= current_hour <= 22:
                decision_factors['prediction'] += 0.5
                decision_factors['analysis'] += 1.0
                decision_factors['tech'] += 0.8  # Lower priority but still relevant
                decision_factors['reply'] += 2.5
        
            # Late night (23-5): Favor analyses, tech content, deprioritize replies
            else:
                decision_factors['prediction'] += 1.0
                decision_factors['analysis'] += 2.0
                decision_factors['tech'] += 2.0  # Great for tech content when audience is more global
                decision_factors['reply'] += 0.5
                decision_factors['correlation'] += 1.5  # Good time for correlation reports
    
            # Factor 3: Market volatility - in volatile markets, predictions and analyses are more valuable
            market_volatility = self._calculate_market_volatility(market_data)
    
            # High volatility boosts prediction and analysis priority
            if market_volatility > 3.0:  # High volatility
                decision_factors['prediction'] += 2.0
                decision_factors['analysis'] += 1.5
                decision_factors['tech'] -= 0.5  # Less focus on educational content during high volatility
            elif market_volatility > 1.5:  # Moderate volatility
                decision_factors['prediction'] += 1.0
                decision_factors['analysis'] += 1.0
            else:  # Low volatility, good time for educational content
                decision_factors['tech'] += 1.0
    
            # Factor 4: Community engagement level - check for active discussions
            active_discussions = self._check_for_active_discussions(market_data)
            if active_discussions:
                # If there are active discussions, favor replies
                decision_factors['reply'] += len(active_discussions) * 0.5  # More discussions = higher priority
                
                # Check if there are tech-related discussions
                tech_discussions = [d for d in active_discussions if self._is_tech_related_post(d)]
                if tech_discussions:
                    # If tech discussions are happening, boost tech priority
                    decision_factors['tech'] += len(tech_discussions) * 0.8
                    
                logger.logger.debug(f"Found {len(active_discussions)} active discussions ({len(tech_discussions)} tech-related), boosting reply priority")
    
            # Factor 5: Check scheduled timeframe posts - these get high priority
            due_timeframes = [tf for tf in self.timeframes if self._should_post_timeframe_now(tf)]
            if due_timeframes:
                decision_factors['prediction'] += 3.0  # High priority for scheduled predictions
                logger.logger.debug(f"Scheduled timeframe posts due: {due_timeframes}")
    
            # Factor 6: Day of week considerations
            weekday = now.weekday()  # 0=Monday, 6=Sunday
    
            # Weekends: More casual engagement (replies), less formal analysis
            if weekday >= 5:  # Saturday or Sunday
                decision_factors['reply'] += 1.5
                decision_factors['tech'] += 1.0  # Good for educational content on weekends
                decision_factors['correlation'] += 0.5
            # Mid-week: Focus on predictions and analysis
            elif 1 <= weekday <= 3:  # Tuesday to Thursday
                decision_factors['prediction'] += 1.0
                decision_factors['analysis'] += 0.5
                decision_factors['tech'] += 0.5  # Steady tech content through the week
    
            # Factor 7: Tech content readiness
            tech_analysis = self._analyze_tech_topics(market_data)
            if tech_analysis.get('enabled', False) and tech_analysis.get('candidate_topics', []):
                # Boost tech priority if we have ready topics
                decision_factors['tech'] += 2.0
                logger.logger.debug(f"Tech topics ready: {len(tech_analysis.get('candidate_topics', []))}")
            
            # Log decision factors for debugging
            logger.logger.debug(f"Post type decision factors: {decision_factors}")
    
            # Determine highest priority action
            highest_priority = max(decision_factors.items(), key=lambda x: x[1])
            action = highest_priority[0]
    
            # Special case: If correlation has reasonable score and it's been a while, prioritize it
            if hours_since_correlation > 48 and decision_factors['correlation'] > 2.0:
                action = 'correlation'
                logger.logger.debug(f"Overriding to correlation post ({hours_since_correlation}h since last one)")
    
            logger.logger.info(f"Decided post type: {action} (score: {highest_priority[1]:.2f})")
            return action
    
        except Exception as e:
            logger.log_error("Decide Post Type", str(e))
            # Default to analysis as a safe fallback
            return "analysis"
        
    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate overall market volatility score based on price movements
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Volatility score (0.0-5.0)
        """
        try:
            if not market_data:
                return 1.0  # Default moderate volatility
            
            # Extract price changes for major tokens
            major_tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
            changes = []
        
            for token in major_tokens:
                if token in market_data:
                    change = abs(market_data[token].get('price_change_percentage_24h', 0))
                    changes.append(change)
        
            if not changes:
                return 1.0
            
            # Calculate average absolute price change
            avg_change = sum(changes) / len(changes)
        
            # Calculate volatility score (normalized to a 0-5 scale)
            # <1% = Very Low, 1-2% = Low, 2-3% = Moderate, 3-5% = High, >5% = Very High
            if avg_change < 1.0:
                return 0.5  # Very low volatility
            elif avg_change < 2.0:
                return 1.0  # Low volatility
            elif avg_change < 3.0:
                return 2.0  # Moderate volatility
            elif avg_change < 5.0:
                return 3.0  # High volatility
            else:
                return 5.0  # Very high volatility
    
        except Exception as e:
            logger.log_error("Calculate Market Volatility", str(e))
            return 1.0  # Default to moderate volatility on error

    def _check_for_active_discussions(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for active token discussions that might warrant replies
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            List of posts representing active discussions
        """
        try:
            # Get recent timeline posts
            recent_posts = self.timeline_scraper.scrape_timeline(count=15)
            if not recent_posts:
                return []
            
            # Filter for posts with engagement (replies, likes)
            engaged_posts = []
            for post in recent_posts:
                # Simple engagement check
                has_engagement = (
                    post.get('reply_count', 0) > 0 or
                    post.get('like_count', 0) > 2 or
                    post.get('retweet_count', 0) > 0
                )
            
                if has_engagement:
                    # Analyze the post content
                    analysis = self.content_analyzer.analyze_post(post)
                    post['content_analysis'] = analysis
                
                    # Check if it's a market-related post with sufficient reply score
                    if analysis.get('reply_worthy', False):
                        engaged_posts.append(post)
        
            return engaged_posts
    
        except Exception as e:
            logger.log_error("Check Active Discussions", str(e))
            return []    
            
    def _analyze_smart_money_indicators(self, token: str, token_data: Dict[str, Any], 
                                      timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze potential smart money movements in a token
        Adjusted for different timeframes
        
        Args:
            token: Token symbol
            token_data: Token market data
            timeframe: Timeframe for analysis
            
        Returns:
            Smart money analysis results
        """
        try:
            # Get historical data over multiple timeframes - adjusted based on prediction timeframe
            if timeframe == "1h":
                hourly_data = self._get_historical_volume_data(token, timeframe=timeframe)
                daily_data = self._get_historical_volume_data(token, timeframe="24h")
                # For 1h predictions, we care about recent volume patterns
                short_term_focus = True
            elif timeframe == "24h":
                # For 24h predictions, we want more data
                hourly_data = self._get_historical_volume_data(token, timeframe="1h")
                daily_data = self._get_historical_volume_data(token, timeframe="7d")
                short_term_focus = False
            else:  # 7d
                # For weekly predictions, we need even more historical context
                hourly_data = self._get_historical_volume_data(token, timeframe="24h")
                daily_data = self._get_historical_volume_data(token, timeframe="7d")
                short_term_focus = False

            current_volume = token_data['volume']
            current_price = token_data['current_price']
            
            # Volume anomaly detection
            hourly_volumes = hourly_data
            daily_volumes = daily_data
            
            # Calculate baselines
            avg_hourly_volume = statistics.mean(hourly_volumes) if hourly_volumes else current_volume
            avg_daily_volume = statistics.mean(daily_volumes) if daily_volumes else current_volume
            
            # Volume Z-score (how many standard deviations from mean)
            hourly_std = statistics.stdev(hourly_volumes) if len(hourly_volumes) > 1 else 1
            volume_z_score = (current_volume - avg_hourly_volume) / hourly_std if hourly_std != 0 else 0
            
            # Price-volume divergence
            # (Price going down while volume increasing suggests accumulation)
            price_direction = 1 if token_data['price_change_percentage_24h'] > 0 else -1
            volume_direction = 1 if current_volume > avg_daily_volume else -1
            
            # Divergence detected when price and volume move in opposite directions
            divergence = (price_direction != volume_direction)
            
            # Adjust accumulation thresholds based on timeframe
            if timeframe == "1h":
                price_change_threshold = 2.0
                volume_multiplier = 1.5
            elif timeframe == "24h":
                price_change_threshold = 3.0
                volume_multiplier = 1.8
            else:  # 7d
                price_change_threshold = 5.0
                volume_multiplier = 2.0
            
            # Check for abnormal volume with minimal price movement (potential accumulation)
            stealth_accumulation = (abs(token_data['price_change_percentage_24h']) < price_change_threshold and 
                                  (current_volume > avg_daily_volume * volume_multiplier))
            
            # Calculate volume profile - percentage of volume in each hour
            volume_profile = {}
            
            # Adjust volume profiling based on timeframe
            if timeframe == "1h":
                # For 1h predictions, look at hourly volume distribution over the day
                hours_to_analyze = 24
            elif timeframe == "24h":
                # For 24h predictions, look at volume by day over the week 
                hours_to_analyze = 7 * 24
            else:  # 7d
                # For weekly, look at entire month
                hours_to_analyze = 30 * 24
            
            if hourly_data:
                # Get timestamped volume data directly from database
                conn, cursor = self.db._get_connection()
                cursor.execute("""
                    SELECT timestamp, volume FROM price_history 
                    WHERE token = ? AND timestamp >= datetime('now', '-24 hours')
                    ORDER BY timestamp DESC
                """, (token,))
                timestamped_data = cursor.fetchall()
                
                for i in range(min(hours_to_analyze, 24)):
                    hour_window = strip_timezone(datetime.now() - timedelta(hours=i+1))
                    hour_volume = sum(row[1] for row in timestamped_data 
                                    if hour_window <= datetime.fromisoformat(row[0]) <= hour_window + timedelta(hours=1))
                    volume_profile[f"hour_{i+1}"] = hour_volume
            
            # Detect unusual trading hours (potential institutional activity)
            total_volume = sum(volume_profile.values()) if volume_profile else 0
            unusual_hours = []
            
            # Adjust unusual hour threshold based on timeframe
            unusual_hour_threshold = 15 if timeframe == "1h" else 20 if timeframe == "24h" else 25
            
            if total_volume > 0:
                for hour, vol in volume_profile.items():
                    hour_percentage = (vol / total_volume) * 100
                    if hour_percentage > unusual_hour_threshold:  # % threshold varies by timeframe
                        unusual_hours.append(hour)
            
            # Detect volume clusters (potential accumulation zones)
            volume_cluster_detected = False
            min_cluster_size = 3 if timeframe == "1h" else 2 if timeframe == "24h" else 2
            cluster_threshold = 1.3 if timeframe == "1h" else 1.5 if timeframe == "24h" else 1.8
            
            if len(hourly_volumes) >= min_cluster_size:
                for i in range(len(hourly_volumes)-min_cluster_size+1):
                    if all(vol > avg_hourly_volume * cluster_threshold for vol in hourly_volumes[i:i+min_cluster_size]):
                        volume_cluster_detected = True
                        break           
            
            # Calculate additional metrics for longer timeframes
            pattern_metrics = {}
            
            if timeframe in ["24h", "7d"]:
                # Calculate volume trends over different periods
                if len(daily_volumes) >= 7:
                    week1_avg = statistics.mean(daily_volumes[:7])
                    week2_avg = statistics.mean(daily_volumes[7:14]) if len(daily_volumes) >= 14 else week1_avg
                    week3_avg = statistics.mean(daily_volumes[14:21]) if len(daily_volumes) >= 21 else week1_avg
                    
                    pattern_metrics["volume_trend_week1_to_week2"] = ((week1_avg / week2_avg) - 1) * 100 if week2_avg > 0 else 0
                    pattern_metrics["volume_trend_week2_to_week3"] = ((week2_avg / week3_avg) - 1) * 100 if week3_avg > 0 else 0
                
                # Check for volume breakout patterns
                if len(hourly_volumes) >= 48:
                    recent_max = max(hourly_volumes[:24])
                    previous_max = max(hourly_volumes[24:48])
                    
                    pattern_metrics["volume_breakout"] = recent_max > previous_max * 1.5
                
                # Check for consistent high volume days
                if len(daily_volumes) >= 14:
                    high_volume_days = [vol > avg_daily_volume * 1.3 for vol in daily_volumes[:14]]
                    pattern_metrics["consistent_high_volume"] = sum(high_volume_days) >= 5
            
            # Results
            results = {
                'volume_z_score': volume_z_score,
                'price_volume_divergence': divergence,
                'stealth_accumulation': stealth_accumulation,
                'abnormal_volume': abs(volume_z_score) > self.SMART_MONEY_ZSCORE_THRESHOLD,
                'volume_vs_hourly_avg': (current_volume / avg_hourly_volume) - 1,
                'volume_vs_daily_avg': (current_volume / avg_daily_volume) - 1,
                'unusual_trading_hours': unusual_hours,
                'volume_cluster_detected': volume_cluster_detected,
                'timeframe': timeframe
            }
            
            # Add pattern metrics for longer timeframes
            if pattern_metrics:
                results['pattern_metrics'] = pattern_metrics
            
            # Store in database
            self.db.store_smart_money_indicators(token, results)
            
            return results
        except Exception as e:
            logger.log_error(f"Smart Money Analysis - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}
    
    def _analyze_volume_profile(self, token: str, market_data: Dict[str, Any], 
                              timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze volume distribution and patterns for a token
        Returns different volume metrics based on timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
            
        Returns:
            Volume profile analysis results
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {}
            
            current_volume = token_data.get('volume', 0)
            
            # Adjust analysis window based on timeframe
            if timeframe == "1h":
                hours_to_analyze = 24
                days_to_analyze = 1
            elif timeframe == "24h":
                hours_to_analyze = 7 * 24
                days_to_analyze = 7
            else:  # 7d
                hours_to_analyze = 30 * 24
                days_to_analyze = 30
            
            # Get historical data
            historical_data = self._get_historical_volume_data(token, timeframe=timeframe)
            
            # Create volume profile by hour of day
            hourly_profile = {}
            for hour in range(24):
                hourly_profile[hour] = 0
            
            # Fill the profile using current hour pattern
            current_hour = datetime.now().hour
            for i, volume in enumerate(historical_data):
                # Distribute volumes across hours based on data point index
                hour = (current_hour - i) % 24
                hourly_profile[hour] += volume
            
            # Calculate daily pattern
            total_volume = sum(hourly_profile.values())
            if total_volume > 0:
                hourly_percentage = {hour: (volume / total_volume) * 100 for hour, volume in hourly_profile.items()}
            else:
                hourly_percentage = {hour: 0 for hour in range(24)}
            
            # Find peak volume hours
            peak_hours = sorted(hourly_percentage.items(), key=lambda x: x[1], reverse=True)[:3]
            low_hours = sorted(hourly_percentage.items(), key=lambda x: x[1])[:3]
            
            # Check for consistent daily patterns
            historical_volumes = historical_data
            avg_volume = statistics.mean(historical_volumes) if historical_volumes else current_volume
            
            # Create day of week profile for longer timeframes
            day_of_week_profile = {}
            if timeframe in ["24h", "7d"] and len(historical_data) >= 7 * 24:
                for day in range(7):
                    day_of_week_profile[day] = 0
                
                # Fill the profile using current day pattern
                current_day = datetime.now().weekday()
                for i, volume in enumerate(historical_data):
                    # Distribute volumes across days based on data point index
                    day = (current_day - (i // 24)) % 7  # Assuming roughly 24 data points per day
                    day_of_week_profile[day] += volume
                
                # Calculate percentages
                dow_total = sum(day_of_week_profile.values())
                if dow_total > 0:
                    day_of_week_percentage = {day: (volume / dow_total) * 100 
                                           for day, volume in day_of_week_profile.items()}
                else:
                    day_of_week_percentage = {day: 0 for day in range(7)}
                
                # Find peak trading days
                peak_days = sorted(day_of_week_percentage.items(), key=lambda x: x[1], reverse=True)[:2]
                low_days = sorted(day_of_week_percentage.items(), key=lambda x: x[1])[:2]
            else:
                day_of_week_percentage = {}
                peak_days = []
                low_days = []
            
            # Calculate volume consistency
            if len(historical_volumes) > 0:
                volume_std = statistics.stdev(historical_volumes) if len(historical_volumes) > 1 else 0
                volume_variability = (volume_std / avg_volume) * 100 if avg_volume > 0 else 0
                
                # Volume consistency score (0-100)
                volume_consistency = max(0, 100 - volume_variability)
            else:
                volume_consistency = 50  # Default if not enough data
            
            # Calculate volume trend over the period
            if len(historical_volumes) >= 2:
                earliest_volume = historical_volumes[0]
                latest_volume = historical_volumes[-1]
                period_change = ((latest_volume - earliest_volume) / earliest_volume) * 100 if earliest_volume > 0 else 0
            else:
                period_change = 0
            
            # Assemble results
            volume_profile_results = {
                'hourly_profile': hourly_percentage,
                'peak_hours': peak_hours,
                'low_hours': low_hours,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'current_vs_avg': ((current_volume / avg_volume) - 1) * 100 if avg_volume > 0 else 0,
                'volume_consistency': volume_consistency,
                'period_change': period_change,
                'timeframe': timeframe
            }
            
            # Add day of week profile for longer timeframes
            if day_of_week_percentage:
                volume_profile_results['day_of_week_profile'] = day_of_week_percentage
                volume_profile_results['peak_days'] = peak_days
                volume_profile_results['low_days'] = low_days
            
            return volume_profile_results
            
        except Exception as e:
            logger.log_error(f"Volume Profile Analysis - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}

    def _detect_volume_anomalies(self, token: str, market_data: Dict[str, Any], 
                              timeframe: str = "1h") -> Dict[str, Any]:
        """
        Detect volume anomalies and unusual patterns
        Adjust detection thresholds based on timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
            
        Returns:
            Volume anomaly detection results
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {}
            
            # Adjust anomaly detection window and thresholds based on timeframe
            if timeframe == "1h":
                detection_window = 24  # 24 hours for hourly predictions
                z_score_threshold = 2.0
                volume_spike_threshold = 3.0
                volume_drop_threshold = 0.3
            elif timeframe == "24h":
                detection_window = 7 * 24  # 7 days for daily predictions
                z_score_threshold = 2.5
                volume_spike_threshold = 4.0
                volume_drop_threshold = 0.25
            else:  # 7d
                detection_window = 30 * 24  # 30 days for weekly predictions
                z_score_threshold = 3.0
                volume_spike_threshold = 5.0
                volume_drop_threshold = 0.2
            
            # Get historical data
            volume_data = self._get_historical_volume_data(token, timeframe=timeframe)
            
            volumes = volume_data
            if len(volumes) < 5:
                return {'insufficient_data': True, 'timeframe': timeframe}

            current_volume = token_data.get('volume', 0)
            
            # Calculate metrics
            avg_volume = statistics.mean(volumes)
            if len(volumes) > 1:
                vol_std = statistics.stdev(volumes)
                # Z-score: how many standard deviations from the mean
                volume_z_score = (current_volume - avg_volume) / vol_std if vol_std > 0 else 0
            else:
                volume_z_score = 0
            
            # Moving average calculation
            if len(volumes) >= 10:
                ma_window = 5 if timeframe == "1h" else 7 if timeframe == "24h" else 10
                moving_avgs = []
                
                for i in range(len(volumes) - ma_window + 1):
                    window = volumes[i:i+ma_window]
                    moving_avgs.append(sum(window) / len(window))
                
                # Calculate rate of change in moving average
                if len(moving_avgs) >= 2:
                    ma_change = ((moving_avgs[-1] / moving_avgs[0]) - 1) * 100
                else:
                    ma_change = 0
            else:
                ma_change = 0
            
            # Volume spike detection
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            has_volume_spike = volume_ratio > volume_spike_threshold
            
            # Volume drop detection
            has_volume_drop = volume_ratio < volume_drop_threshold
            
            # Detect sustained high/low volume
            if len(volumes) >= 5:
                recent_volumes = volumes[-5:]
                avg_recent_volume = sum(recent_volumes) / len(recent_volumes)
                sustained_high_volume = avg_recent_volume > avg_volume * 1.5
                sustained_low_volume = avg_recent_volume < avg_volume * 0.5
            else:
                sustained_high_volume = False
                sustained_low_volume = False
            
            # Detect volume patterns for longer timeframes
            pattern_detection = {}
            
            if timeframe in ["24h", "7d"] and len(volumes) >= 14:
                # Check for "volume climax" pattern (increasing volumes culminating in a spike)
                vol_changes = [volumes[i]/volumes[i-1] if volumes[i-1] > 0 else 1 for i in range(1, len(volumes))]
                
                if len(vol_changes) >= 5:
                    recent_changes = vol_changes[-5:]
                    climax_pattern = (sum(1 for change in recent_changes if change > 1.1) >= 3) and has_volume_spike
                    pattern_detection["volume_climax"] = climax_pattern
                
                # Check for "volume exhaustion" pattern (decreasing volumes after a spike)
                if len(volumes) >= 10:
                    peak_idx = volumes.index(max(volumes[-10:]))
                    if peak_idx < len(volumes) - 3:
                        post_peak = volumes[peak_idx+1:]
                        exhaustion_pattern = all(post_peak[i] < post_peak[i-1] for i in range(1, len(post_peak)))
                        pattern_detection["volume_exhaustion"] = exhaustion_pattern
            
            # Assemble results
            anomaly_results = {
                'volume_z_score': volume_z_score,
                'volume_ratio': volume_ratio,
                'has_volume_spike': has_volume_spike,
                'has_volume_drop': has_volume_drop,
                'ma_change': ma_change,
                'sustained_high_volume': sustained_high_volume,
                'sustained_low_volume': sustained_low_volume,
                'abnormal_volume': abs(volume_z_score) > z_score_threshold,
                'timeframe': timeframe
            }
            
            # Add pattern detection for longer timeframes
            if pattern_detection:
                anomaly_results['patterns'] = pattern_detection
            
            return anomaly_results
            
        except Exception as e:
            logger.log_error(f"Volume Anomaly Detection - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}   


    def _analyze_token_vs_market(self, token: str, market_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """
        Analyze token performance relative to market with robust token identifier handling
        Enhanced method with extensive error handling, logging, and different timeframe support
        Now uses database-driven token selection instead of hardcoded lists
        
        Args:
            token: Token identifier (symbol like 'BTC' or CoinGecko ID like 'bitcoin')
            market_data: Market data dictionary (supports both symbol and ID keys)
            timeframe: Timeframe for analysis ('1h', '24h', '7d')
        
        Returns:
            Dictionary containing comprehensive market analysis results
        """
        try:
            # Get timeframe-appropriate window for historical data
            if timeframe == "1h":
                hours = 24
                history_hours = 24
            elif timeframe == "24h":
                hours = 7 * 24
                history_hours = 7 * 24
            else:  # 7d
                hours = 30 * 24
                history_hours = 30 * 24
                
            # Check if database is initialized
            if not hasattr(self, 'db') or self.db is None:
                logger.logger.error("❌ FALLBACK: Database not initialized for _analyze_token_vs_market - creating new connection")
                from database import CryptoDatabase
                self.db = CryptoDatabase()
                logger.logger.info("✅ Created new database connection")
                
            # Function to safely extract prices from historical data
            def extract_prices(history):
                if not history or history == "Never":
                    return []
                if isinstance(history, list):
                    prices = []
                    for entry in history:
                        if isinstance(entry, dict) and 'price' in entry:
                            try:
                                price = float(entry['price'])
                                if price > 0:
                                    prices.append(price)
                            except (ValueError, TypeError):
                                continue
                    return prices
                return []
                
            # Try to get token data - check both symbol and CoinGecko ID
            token_data = market_data.get(token)
            if not token_data:
                logger.logger.debug(f"🔍 Token {token} not found directly in market data, trying symbol-to-ID mapping")
                # Try mapping symbol to CoinGecko ID using TokenMappingManager
                try:
                    # Use token_mapper from config if available
                    if hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                        coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(token)
                        if coingecko_id:
                            token_data = market_data.get(coingecko_id)
                            logger.logger.debug(f"✅ TokenMappingManager: Mapped {token} to {coingecko_id}")
                    
                    # Fallback to database lookup if token_mapper not available
                    if not token_data:
                        conn, cursor = self.db._get_connection()
                        cursor.execute("""
                            SELECT coin_id FROM coingecko_market_data 
                            WHERE UPPER(symbol) = ?
                            ORDER BY timestamp DESC
                            LIMIT 1
                        """, (token.upper(),))
                        
                        result = cursor.fetchone()
                        if result and result['coin_id']:
                            coingecko_id = result['coin_id']
                            token_data = market_data.get(coingecko_id)
                            logger.logger.debug(f"✅ Database lookup: Mapped {token} to {coingecko_id}")
                        else:
                            logger.logger.debug(f"ℹ️ No mapping found in database for {token}")
                except Exception as db_error:
                    # Log the lookup failure and continue with next options
                    logger.logger.warning(f"❌ TokenMappingManager/Database lookup for {token} failed: {str(db_error)}")
                    # No hardcoded fallback - we're fully database-driven now
                    
            if not token_data:
                logger.logger.warning(f"❌ Token {token} not found in market data after all lookups")
                return {
                    "vs_market_avg_change": 0.0,
                    "vs_market_percentile": 50.0,
                    "market_correlation": 0.0,
                    "market_sentiment": "neutral",
                    "timeframe": timeframe
                }
                
            # Get reference tokens from database instead of hardcoded lists
            try:
                # Get top tokens by market cap from database
                database_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=50)
                reference_tokens = [t for t in database_tokens if t != token]
                logger.logger.debug(f"✅ Using database-driven reference tokens: {reference_tokens}")
            except Exception as db_error:
                logger.logger.warning(f"❌ FALLBACK: Database-driven token selection failed: {str(db_error)}")
                # Fall back to existing reference tokens if database query fails
                if hasattr(self, 'reference_tokens'):
                    reference_tokens = [t for t in self.reference_tokens if t != token]
                    logger.logger.warning(f"⚠️ FALLBACK: Using hardcoded reference_tokens: {reference_tokens}")
                else:
                    reference_tokens = [t for t in market_data.keys() if t != token]
                    logger.logger.warning(f"⚠️ FALLBACK: Using market_data keys as reference tokens: {reference_tokens[:5]}...")
            
            # Use the database-driven token selection for timeframe filtering
            # instead of hardcoded major tokens lists
            if timeframe == "1h":
                # For short timeframe, use fewer tokens (top 5)
                filtered_ref_tokens = reference_tokens[:5] if len(reference_tokens) >= 5 else reference_tokens
                logger.logger.debug(f"⚠️ Using top {len(filtered_ref_tokens)} tokens for 1h timeframe")
            elif timeframe == "24h":
                # Medium timeframe uses more tokens (top 8)
                filtered_ref_tokens = reference_tokens[:8] if len(reference_tokens) >= 8 else reference_tokens
                logger.logger.debug(f"⚠️ Using top {len(filtered_ref_tokens)} tokens for 24h timeframe")
            else:  # 7d
                # Long timeframe uses all available reference tokens
                filtered_ref_tokens = reference_tokens
            
            # Keep only tokens that exist in market_data
            reference_tokens = []
            for ref_token in filtered_ref_tokens:
                if ref_token in market_data:
                    reference_tokens.append(ref_token)
                else:
                    # Try mapping symbol to CoinGecko ID using TokenMappingManager
                    try:
                        # Use token_mapper from config if available
                        if hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                            coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(ref_token)
                            if coingecko_id and coingecko_id in market_data:
                                reference_tokens.append(ref_token)
                                logger.logger.debug(f"✅ TokenMappingManager: Added {ref_token} (mapped to {coingecko_id}) to reference tokens")
                                continue
                    
                        # Fallback to database lookup
                        conn, cursor = self.db._get_connection()
                        cursor.execute("""
                            SELECT coin_id FROM coingecko_market_data 
                            WHERE UPPER(symbol) = ?
                            ORDER BY timestamp DESC
                            LIMIT 1
                        """, (ref_token.upper(),))
                        
                        result = cursor.fetchone()
                        if result and result['coin_id']:
                            coingecko_id = result['coin_id']
                            if coingecko_id in market_data:
                                reference_tokens.append(ref_token)
                                logger.logger.debug(f"✅ Database lookup: Added {ref_token} (mapped to {coingecko_id}) to reference tokens")
                    except Exception as db_error:
                        # Log the database error but don't fall back to hardcoded lists
                        logger.logger.debug(f"❌ Database lookup for reference token {ref_token} failed: {str(db_error)}")
                        # No hardcoded fallback - fully database-driven token management
        
            if not reference_tokens:
                logger.logger.warning(f"❌ FALLBACK: No reference tokens found for comparison with {token}")
                return {
                    "vs_market_avg_change": 0.0,
                    "vs_market_percentile": 50.0,
                    "market_correlation": 0.0,
                    "market_sentiment": "neutral",
                    "timeframe": timeframe
                }
            
            logger.logger.info(f"🔍 Analyzing {token} vs {len(reference_tokens)} reference tokens for {timeframe} timeframe")

            # Calculate market metrics
            # ----- 1. Price Changes -----
            market_changes = []
        
            for ref_token in reference_tokens:
                ref_data = market_data.get(ref_token)
                if not ref_data:
                    # Try CoinGecko ID lookup using TokenMappingManager
                    try:
                        # Use token_mapper from config if available
                        if hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                            coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(ref_token)
                            if coingecko_id:
                                ref_data = market_data.get(coingecko_id)
                                if ref_data:
                                    logger.logger.debug(f"✅ TokenMappingManager: Found data for {ref_token} via {coingecko_id}")
                                    
                        # Fallback to database lookup
                        if not ref_data:
                            conn, cursor = self.db._get_connection()
                            cursor.execute("""
                                SELECT coin_id FROM coingecko_market_data 
                                WHERE UPPER(symbol) = ?
                                ORDER BY timestamp DESC
                                LIMIT 1
                            """, (ref_token.upper(),))
                            
                            result = cursor.fetchone()
                            if result and result['coin_id']:
                                coingecko_id = result['coin_id']
                                ref_data = market_data.get(coingecko_id)
                                if ref_data:
                                    logger.logger.debug(f"✅ Database lookup: Found data for {ref_token} via {coingecko_id}")
                    except Exception as db_error:
                        # Log database error and continue without hardcoded fallback
                        logger.logger.debug(f"❌ Database lookup for reference data {ref_token} failed: {str(db_error)}")
                        
                if not ref_data or not isinstance(ref_data, dict):
                    logger.logger.debug(f"⚠️ No data found for reference token {ref_token}")
                    continue
                
                # Extract price change based on timeframe
                if timeframe == "1h":
                    change_keys = ['price_change_percentage_1h_in_currency', 'price_change_1h', 'change_1h', '1h_change']
                elif timeframe == "24h":
                    change_keys = ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']
                else:  # 7d
                    change_keys = ['price_change_percentage_7d_in_currency', 'price_change_7d', 'change_7d', '7d_change']
                
                # Try each key
                found_change = False
                for change_key in change_keys:
                    if change_key in ref_data:
                        try:
                            change_value = float(ref_data[change_key])
                            market_changes.append(change_value)
                            found_change = True
                            break  # Found valid value
                        except (ValueError, TypeError):
                            continue  # Try next key
                
                if not found_change:
                    logger.logger.debug(f"⚠️ No valid price change found for {ref_token} with keys {change_keys}")
        
            # Verify we have market changes data
            if not market_changes:
                logger.logger.warning(f"❌ FALLBACK: No market change data available for comparison with {token}")
                return {
                    "vs_market_avg_change": 0.0,
                    "vs_market_percentile": 50.0,
                    "market_correlation": 0.0,
                    "market_sentiment": "neutral",
                    "timeframe": timeframe
                }
            
            # Calculate market average
            market_avg_change = statistics.mean(market_changes) if market_changes else 0.0
            logger.logger.debug(f"📊 Market average change: {market_avg_change:.2f}% from {len(market_changes)} tokens")
        
            # ----- 2. Token Price Change -----
            token_change = 0.0
            if timeframe == "1h":
                change_keys = ['price_change_percentage_1h_in_currency', 'price_change_1h', 'change_1h', '1h_change']
            elif timeframe == "24h":
                change_keys = ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']
            else:  # 7d
                change_keys = ['price_change_percentage_7d_in_currency', 'price_change_7d', 'change_7d', '7d_change']
            
            token_change_found = False
            for change_key in change_keys:
                if change_key in token_data:
                    try:
                        token_change = float(token_data[change_key])
                        token_change_found = True
                        break
                    except (ValueError, TypeError):
                        continue
            
            if not token_change_found:
                logger.logger.warning(f"⚠️ FALLBACK: No valid price change found for {token}, using 0.0 as default")
                        
            # Calculate vs market metrics
            vs_market_change = token_change - market_avg_change
            outperforming = vs_market_change > 0
            
            # Calculate percentile ranking
            if market_changes:
                tokens_outperforming = sum(1 for change in market_changes if token_change > change)
                vs_market_percentile = (tokens_outperforming / len(market_changes)) * 100
            else:
                vs_market_percentile = 50.0
                
            # ----- 3. Volume Analysis -----
            # Use the new _safe_get_volume method for consistent volume access
            token_volume = self._safe_get_volume(token_data)
            if token_volume == 0:
                logger.logger.warning(f"⚠️ FALLBACK: No valid volume found for {token}, using 0 as default")

            market_volumes = []
            for ref_token in reference_tokens:
                ref_data = market_data.get(ref_token)
                if not ref_data:
                    # Try CoinGecko ID lookup using TokenMappingManager
                    try:
                        # Use token_mapper from config if available
                        if hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                            coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(ref_token)
                            if coingecko_id:
                                ref_data = market_data.get(coingecko_id)
                                
                        # Fallback to database lookup
                        if not ref_data:
                            conn, cursor = self.db._get_connection()
                            cursor.execute("""
                                SELECT coin_id FROM coingecko_market_data 
                                WHERE UPPER(symbol) = ?
                                ORDER BY timestamp DESC
                                LIMIT 1
                            """, (ref_token.upper(),))
                            
                            result = cursor.fetchone()
                            if result and result['coin_id']:
                                coingecko_id = result['coin_id']
                                ref_data = market_data.get(coingecko_id)
                                if ref_data:
                                    logger.logger.debug(f"✅ Database lookup: Found data for {ref_token} via {coingecko_id}")
                    except Exception as db_error:
                        # Log database error and continue without hardcoded fallback
                        logger.logger.debug(f"❌ Database lookup for volume data {ref_token} failed: {str(db_error)}")
                        
                if ref_data:
                    # Use the new _safe_get_volume method for consistent volume access
                    ref_volume = self._safe_get_volume(ref_data)
                    if ref_volume > 0:
                        market_volumes.append(ref_volume)
                        
            if not market_volumes:
                logger.logger.warning(f"⚠️ FALLBACK: No valid market volumes found, using default value")
                market_avg_volume = 1
            else:
                market_avg_volume = statistics.mean(market_volumes)
            
            volume_growth_diff = ((token_volume / market_avg_volume) - 1) * 100 if market_avg_volume > 0 else 0
            
            # ----- 4. Correlation Analysis -----
            correlations = {}
            btc_correlation = 0.5  # Default moderate correlation
            
            # Try to get historical data for better correlation
            token_history = self._get_historical_price_data(token, hours=history_hours, timeframe=timeframe)
            token_prices = extract_prices(token_history)
            
            if not token_prices:
                logger.logger.warning(f"⚠️ FALLBACK: No historical price data for {token}, using default correlation")
            
            for ref_token in reference_tokens[:3]:  # Limit to top 3 for performance
                ref_data = market_data.get(ref_token)
                if not ref_data:
                    # Try CoinGecko ID lookup using TokenMappingManager
                    try:
                        # Use token_mapper from config if available
                        if hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                            coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(ref_token)
                            if coingecko_id:
                                ref_data = market_data.get(coingecko_id)
                                
                        # Fallback to database lookup
                        if not ref_data:
                            conn, cursor = self.db._get_connection()
                            cursor.execute("""
                                SELECT coin_id FROM coingecko_market_data 
                                WHERE UPPER(symbol) = ?
                                ORDER BY timestamp DESC
                                LIMIT 1
                            """, (ref_token.upper(),))
                            
                            result = cursor.fetchone()
                            if result and result['coin_id']:
                                coingecko_id = result['coin_id']
                                ref_data = market_data.get(coingecko_id)
                    except Exception as db_error:
                        # Log database error and continue without hardcoded fallback
                        logger.logger.debug(f"❌ Database lookup for correlation data {ref_token} failed: {str(db_error)}")
                        
                if not ref_data:
                    continue
                    
                # Simple direction correlation
                ref_change = 0.0
                for change_key in change_keys:
                    if change_key in ref_data:
                        try:
                            ref_change = float(ref_data[change_key])
                            break
                        except (ValueError, TypeError):
                            continue
                            
                # Calculate correlation
                if ref_token.upper() == "BTC" or (locals().get('coingecko_id') == 'bitcoin'):
                    # Enhanced BTC correlation if we have historical data
                    if token_prices and len(token_prices) >= 5:
                        try:
                            ref_history = self._get_historical_price_data(ref_token, hours=history_hours, timeframe=timeframe)
                            ref_prices = extract_prices(ref_history)
                            
                            if len(ref_prices) >= 5 and len(token_prices) == len(ref_prices):
                                # Calculate price change correlations
                                token_changes = [(token_prices[i] / token_prices[i-1] - 1) for i in range(1, len(token_prices))]
                                ref_changes = [(ref_prices[i] / ref_prices[i-1] - 1) for i in range(1, len(ref_prices))]
                                
                                if len(token_changes) >= 3:
                                    # Use numpy if available, otherwise fallback to a simpler correlation
                                    if 'numpy' in sys.modules:
                                        import numpy as np
                                        correlation = np.corrcoef(token_changes, ref_changes)[0, 1]
                                        btc_correlation = max(-1, min(1, correlation)) if not np.isnan(correlation) else 0.5
                                    else:
                                        # Simple correlation calculation if numpy not available
                                        mean_token = sum(token_changes) / len(token_changes)
                                        mean_ref = sum(ref_changes) / len(ref_changes)
                                        
                                        numerator = sum((token_changes[i] - mean_token) * (ref_changes[i] - mean_ref) 
                                                    for i in range(len(token_changes)))
                                        
                                        denominator = (sum((tc - mean_token)**2 for tc in token_changes) * 
                                                    sum((rc - mean_ref)**2 for rc in ref_changes))**0.5
                                        
                                        if denominator > 0:
                                            btc_correlation = numerator / denominator
                                        else:
                                            btc_correlation = 0.5
                            else:
                                logger.logger.debug(f"⚠️ FALLBACK: Insufficient price data lengths for correlation: token={len(token_prices)}, ref={len(ref_prices) if 'ref_prices' in locals() else 'N/A'}")
                        except Exception as corr_error:
                            logger.logger.warning(f"❌ FALLBACK: BTC correlation calculation failed: {str(corr_error)}")
                            btc_correlation = 0.5
                    else:
                        logger.logger.debug(f"⚠️ FALLBACK: Insufficient token price data for BTC correlation, using default")
                            
                # Store basic correlation
                direction_match = (token_change > 0) == (ref_change > 0)
                correlations[ref_token] = 1.0 if direction_match else -0.5
                
            # Determine market sentiment
            if vs_market_change > 3.0:
                market_sentiment = "strongly outperforming"
            elif vs_market_change > 1.0:
                market_sentiment = "outperforming"
            elif vs_market_change < -3.0:
                market_sentiment = "strongly underperforming"
            elif vs_market_change < -1.0:
                market_sentiment = "underperforming"
            else:
                market_sentiment = "neutral"
                
            # Calculate extended metrics if volatility method is available
            extended_metrics = {}
            if hasattr(self, '_calculate_relative_volatility'):
                try:
                    relative_volatility = self._calculate_relative_volatility(
                        token, reference_tokens, market_data, timeframe
                    )
                    if relative_volatility is not None:
                        extended_metrics['relative_volatility'] = relative_volatility
                except Exception as vol_error:
                    logger.logger.warning(f"❌ FALLBACK: Volatility calculation failed for {token}: {str(vol_error)}")
            
            # ----- Prepare final result dictionary -----
            result = {
                'vs_market_avg_change': vs_market_change,
                'vs_market_percentile': vs_market_percentile,
                'vs_market_volume_growth': volume_growth_diff,
                'market_correlation': btc_correlation,  # Use BTC correlation as primary metric
                'market_sentiment': market_sentiment,
                'correlations': correlations,
                'outperforming_market': outperforming,
                'btc_correlation': btc_correlation,
                'timeframe': timeframe
            }
        
            # Add extended metrics if available
            if extended_metrics:
                result['extended_metrics'] = extended_metrics
            
            logger.logger.info(f"✅ Successfully analyzed {token} vs market ({timeframe}): {market_sentiment}, {vs_market_change:.2f}% diff")
            return result
        
        except Exception as e:
            # Log error with consistent format
            error_details = f"{type(e).__name__}: {str(e)}"
            logger.log_error(f"Token vs Market Analysis - {token} ({timeframe})", error_details)
            
            # Also log to regular logger
            logger.logger.error(f"❌ FALLBACK: Complete analysis failure for {token} ({timeframe}): {error_details}")
            
            # Return default values
            return {
                "vs_market_avg_change": 0.0,
                "vs_market_percentile": 50.0,
                "market_correlation": 0.0,
                "market_sentiment": "neutral",
                "timeframe": timeframe
            }
        
    def _calculate_relative_volatility(self, token: str, reference_tokens: List[str], 
                                    market_data: Dict[str, Any], timeframe: str) -> Optional[float]:
        """
        Calculate token's volatility relative to market average
        Returns a ratio where >1 means more volatile than market, <1 means less volatile
        Now uses database-driven reference tokens and TokenMappingManager
        
        Args:
            token: Token symbol
            reference_tokens: List of reference token symbols
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
        
        Returns:
            Relative volatility ratio or None if insufficient data
        """
        try:
            # Get historical data with appropriate window for the timeframe
            if timeframe == "1h":
                hours = 24
            elif timeframe == "24h":
                hours = 7 * 24
            else:  # 7d
                hours = 30 * 24
                
            # Check if database is initialized
            if not hasattr(self, 'db') or self.db is None:
                logger.logger.error("❌ FALLBACK: Database not initialized for _calculate_relative_volatility")
                from database import CryptoDatabase
                self.db = CryptoDatabase()
                logger.logger.info("✅ Created new database connection")
        
            # If reference_tokens is empty or None, get them from database
            if not reference_tokens:
                logger.logger.warning("⚠️ FALLBACK: No reference tokens provided, fetching from database")
                try:
                    # Get top tokens by market cap from database
                    database_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=50)
                    reference_tokens = [t for t in database_tokens if t != token]
                    logger.logger.debug(f"✅ Using database-driven reference tokens: {reference_tokens}")
                except Exception as db_error:
                    logger.logger.warning(f"❌ FALLBACK: Database-driven token selection failed: {str(db_error)}")
                    # Fall back to market_data keys if database query fails
                    reference_tokens = [t for t in market_data.keys() if t != token]
                    logger.logger.warning(f"⚠️ FALLBACK: Using market_data keys as reference tokens: {reference_tokens[:5]}...")
        
            # Function to safely extract prices from historical data
            def extract_prices(history_data):
                if history_data is None:
                    return []
                
                # Handle case where history is a string (like "Never")
                if isinstance(history_data, str):
                    logger.logger.warning(f"⚠️ FALLBACK: History data is a string: '{history_data}'")
                    return []
                
                # Ensure history_data is iterable
                if not hasattr(history_data, '__iter__'):
                    logger.logger.warning(f"⚠️ FALLBACK: History data is not iterable: {type(history_data)}")
                    return []
                
                prices = []
            
                for entry in history_data:
                    # Skip None entries
                    if entry is None:
                        continue
                    
                    price = None
                
                    try:
                        # Case 1: Dictionary with price property
                        if isinstance(entry, dict):
                            if 'price' in entry and entry['price'] is not None:
                                try:
                                    price = float(entry['price'])
                                except (ValueError, TypeError):
                                    # Price couldn't be converted to float
                                    pass
                                
                        # Case 2: List/tuple with price as first element
                        elif isinstance(entry, (list, tuple)) and len(entry) > 0:
                            if entry[0] is not None:
                                try:
                                    price = float(entry[0])
                                except (ValueError, TypeError):
                                    # First element couldn't be converted to float
                                    pass
                                
                        # Case 3: Entry has price attribute (but NOT for lists/tuples)
                        elif not isinstance(entry, (list, tuple)) and hasattr(entry, 'price'):
                            try:
                                price = float(entry.price)
                            except (ValueError, TypeError, AttributeError):
                                # Attribute access or conversion failed
                                pass
                            
                        # Case 4: Entry itself is a number
                        elif isinstance(entry, (int, float)):
                            price = float(entry)
                        
                        # Add price to list if valid
                        if price is not None and price > 0:
                            prices.append(price)
                        
                    except Exception as extract_error:
                        # Catch any other unexpected errors during extraction
                        logger.logger.debug(f"⚠️ Error extracting price: {extract_error}")
                        continue
                    
                return prices
        
            # Get token price history and extract prices
            try:
                token_history = self._get_historical_price_data(token, hours=hours, timeframe=timeframe)
                token_prices = extract_prices(token_history)
                logger.logger.debug(f"📊 Got {len(token_prices)} price points for {token}")
            except Exception as token_error:
                logger.logger.error(f"❌ FALLBACK: Error getting token history for {token}: {token_error}")
                return None
            
            # Check if we have enough price data
            if len(token_prices) < 5:
                logger.logger.warning(f"❌ FALLBACK: Insufficient price data for {token}: {len(token_prices)} points, need at least 5")
                return None
            
            # Calculate token price changes
            token_changes = []
            for i in range(1, len(token_prices)):
                if token_prices[i-1] > 0:
                    try:
                        pct_change = ((token_prices[i] / token_prices[i-1]) - 1) * 100
                        token_changes.append(pct_change)
                    except (ZeroDivisionError, OverflowError):
                        continue
                    
            if len(token_changes) < 2:
                logger.logger.warning(f"❌ FALLBACK: Insufficient price changes for {token}: {len(token_changes)} changes, need at least 2")
                return None
            
            # Calculate token volatility (standard deviation)
            try:
                token_volatility = statistics.stdev(token_changes)
                logger.logger.debug(f"📊 {token} volatility: {token_volatility:.2f}%")
            except statistics.StatisticsError as stats_error:
                logger.logger.error(f"❌ FALLBACK: Error calculating token volatility: {stats_error}")
                return None
            
            # Calculate market average volatility
            market_volatilities = []
        
            for ref_token in reference_tokens:
                # Try mapping symbol to ID using TokenMappingManager
                ref_data = None
                coingecko_id = None
                
                # First check if token is in market_data directly
                if ref_token in market_data:
                    ref_data = market_data.get(ref_token)
                else:
                    try:
                        # Use token_mapper from config if available
                        if hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                            coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(ref_token)
                            if coingecko_id and coingecko_id in market_data:
                                ref_data = market_data.get(coingecko_id)
                                logger.logger.debug(f"✅ TokenMappingManager: Found {ref_token} via {coingecko_id}")
                                
                        # Fallback to database lookup if needed
                        if not ref_data:
                            conn, cursor = self.db._get_connection()
                            cursor.execute("""
                                SELECT coin_id FROM coingecko_market_data 
                                WHERE UPPER(symbol) = ?
                                ORDER BY timestamp DESC
                                LIMIT 1
                            """, (ref_token.upper(),))
                            
                            result = cursor.fetchone()
                            if result and result['coin_id']:
                                coingecko_id = result['coin_id']
                                if coingecko_id in market_data:
                                    ref_data = market_data.get(coingecko_id)
                                    logger.logger.debug(f"✅ Database lookup: Found {ref_token} via {coingecko_id}")
                    except Exception as db_error:
                        logger.logger.debug(f"❌ Database lookup for {ref_token} failed: {str(db_error)}")
                
                if not ref_data:
                    logger.logger.debug(f"⚠️ Reference token {ref_token} not found in market data")
                    continue
                
                try:
                    # Get reference token price history and extract prices
                    ref_history = self._get_historical_price_data(ref_token, hours=hours, timeframe=timeframe)
                    ref_prices = extract_prices(ref_history)
                    
                    # Log how many price points we got
                    if len(ref_prices) >= 5:
                        logger.logger.debug(f"📊 Got {len(ref_prices)} price points for {ref_token}")
                    else:
                        logger.logger.debug(f"⚠️ Only {len(ref_prices)} price points for {ref_token}, need at least 5")
                
                    # Check if we have enough price data
                    if len(ref_prices) < 5:
                        continue
                    
                    # Calculate reference token price changes
                    ref_changes = []
                    for i in range(1, len(ref_prices)):
                        if ref_prices[i-1] > 0:
                            try:
                                pct_change = ((ref_prices[i] / ref_prices[i-1]) - 1) * 100
                                ref_changes.append(pct_change)
                            except (ZeroDivisionError, OverflowError):
                                continue
                            
                    # Only calculate volatility if we have enough changes
                    if len(ref_changes) >= 2:
                        try:
                            ref_volatility = statistics.stdev(ref_changes)
                            market_volatilities.append(ref_volatility)
                            logger.logger.debug(f"📊 {ref_token} volatility: {ref_volatility:.2f}%")
                        except statistics.StatisticsError as stats_error:
                            logger.logger.debug(f"⚠️ Error calculating volatility for {ref_token}: {stats_error}")
                            continue
                    else:
                        logger.logger.debug(f"⚠️ Not enough price changes for {ref_token}: {len(ref_changes)}")
                        
                except Exception as ref_error:
                    # Continue with other tokens if there's an error with this one
                    logger.logger.debug(f"❌ FALLBACK: Error processing reference token {ref_token}: {ref_error}")
                    continue
        
            # Check if we have enough market volatility data
            if not market_volatilities:
                logger.logger.warning(f"❌ FALLBACK: No market volatilities calculated for comparison")
                return None
            
            # Calculate market average volatility
            market_avg_volatility = statistics.mean(market_volatilities)
            logger.logger.debug(f"📊 Market average volatility: {market_avg_volatility:.2f}% from {len(market_volatilities)} tokens")
        
            # Calculate relative volatility
            if market_avg_volatility > 0:
                relative_volatility = token_volatility / market_avg_volatility
                logger.logger.info(f"✅ Relative volatility for {token}: {relative_volatility:.2f}x market average")
                return relative_volatility
            else:
                logger.logger.warning(f"❌ FALLBACK: Market average volatility is zero for {token}")
                return None
            
        except Exception as e:
            logger.log_error(f"Calculate Relative Volatility - {token} ({timeframe})", str(e))
            logger.logger.error(f"❌ FALLBACK: Complete volatility calculation failure for {token}: {str(e)}")
            return None

    def _calculate_correlations(self, token: str, market_data: Dict[str, Any], 
                                timeframe: str = "1h") -> Dict[str, Any]:
        """
        Calculate token correlations with the market with robust token identifier handling
        Adjust correlation window based on timeframe with enhanced error handling and logging
        Updated July 31, 2025 - No hardcoded mappings, pure TokenMappingManager approach
        
        Args:
            token: Token identifier (symbol like 'BTC' or CoinGecko ID like 'bitcoin')
            market_data: Market data dictionary (supports both symbol and ID keys)
            timeframe: Timeframe for analysis ('1h', '24h', '7d')
        
        Returns:
            Dictionary of comprehensive correlation metrics
        """
        try:
            # Validate timeframe
            if timeframe not in ["1h", "24h", "7d"]:
                logger.logger.warning(f"Invalid timeframe {timeframe}, defaulting to 1h")
                timeframe = "1h"
                
            # Check if database is initialized
            if not hasattr(self, 'db') or self.db is None:
                logger.logger.error("❌ Database not initialized for _calculate_correlations")
                from database import CryptoDatabase
                self.db = CryptoDatabase()
                logger.logger.info("✅ Created new database connection")
                
            # Try to get token data - first direct lookup
            token_data = market_data.get(token)
            
            # If not found, try TokenMappingManager
            if not token_data and hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                try:
                    coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(token)
                    if coingecko_id:
                        token_data = market_data.get(coingecko_id)
                        logger.logger.debug(f"✅ TokenMappingManager: Found {token} as {coingecko_id}")
                except Exception as token_mapper_error:
                    logger.logger.debug(f"TokenMappingManager lookup failed for {token}: {str(token_mapper_error)}")
            
            # If still not found, try database lookup
            if not token_data:
                try:
                    conn, cursor = self.db._get_connection()
                    cursor.execute("""
                        SELECT coin_id FROM coingecko_market_data 
                        WHERE UPPER(symbol) = ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (token.upper(),))
                    
                    result = cursor.fetchone()
                    if result and result['coin_id']:
                        coingecko_id = result['coin_id']
                        token_data = market_data.get(coingecko_id)
                        logger.logger.debug(f"✅ Database lookup: Found {token} as {coingecko_id}")
                except Exception as db_lookup_error:
                    logger.logger.debug(f"Database lookup failed for {token}: {str(db_lookup_error)}")
                    
            if not token_data:
                logger.logger.warning(f"Token {token} not found in market data for correlations")
                return {'timeframe': timeframe}
            
            # Get reference tokens from database instead of hardcoded lists
            try:
                # Get top tokens by market cap from database based on timeframe
                if timeframe == "1h":
                    limit = 30  # For hourly, just use top 3o tokens
                elif timeframe == "24h":
                    limit = 50  # For daily, use top 50 tokens
                else:  # 7d
                    limit = 100  # For weekly, use top 100 tokens
                    
                database_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=limit)
                reference_tokens = [t for t in database_tokens if t != token]
                logger.logger.debug(f"✅ Using database-driven reference tokens for {timeframe}: {reference_tokens}")
                
            except Exception as db_error:
                logger.logger.warning(f"❌ Database-driven token selection failed: {str(db_error)}")
                # If database fails, use market_data keys as last resort
                reference_tokens = [t for t in market_data.keys() if t != token][:5]
                logger.logger.warning(f"⚠️ Using market_data keys as reference tokens: {reference_tokens}")
            
            # Ensure we have valid reference tokens in market_data
            valid_reference_tokens = []
            for ref_token in reference_tokens:
                ref_data = market_data.get(ref_token)
                
                # If not found directly, try TokenMappingManager
                if not ref_data and hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                    try:
                        coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(ref_token)
                        if coingecko_id:
                            ref_data = market_data.get(coingecko_id)
                            logger.logger.debug(f"✅ TokenMappingManager: Found {ref_token} as {coingecko_id}")
                    except Exception as token_mapper_error:
                        logger.logger.debug(f"TokenMappingManager lookup failed for {ref_token}: {str(token_mapper_error)}")
                
                # If still not found, try database lookup
                if not ref_data:
                    try:
                        conn, cursor = self.db._get_connection()
                        cursor.execute("""
                            SELECT coin_id FROM coingecko_market_data 
                            WHERE UPPER(symbol) = ?
                            ORDER BY timestamp DESC
                            LIMIT 1
                        """, (ref_token.upper(),))
                        
                        result = cursor.fetchone()
                        if result and result['coin_id']:
                            coingecko_id = result['coin_id']
                            ref_data = market_data.get(coingecko_id)
                            logger.logger.debug(f"✅ Database lookup: Found {ref_token} as {coingecko_id}")
                    except Exception as db_lookup_error:
                        logger.logger.debug(f"Database lookup failed for {ref_token}: {str(db_lookup_error)}")
                        
                if ref_data:
                    valid_reference_tokens.append(ref_token)
            
            correlations = {}
        
            # Calculate correlation with each valid reference token
            for ref_token in valid_reference_tokens:
                # Get reference token data with the same lookup logic
                ref_data = market_data.get(ref_token)
                
                if not ref_data and hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                    try:
                        coingecko_id = self.config.token_mapper.symbol_to_coingecko_id(ref_token)
                        if coingecko_id:
                            ref_data = market_data.get(coingecko_id)
                    except Exception:
                        pass
                        
                if not ref_data:
                    try:
                        conn, cursor = self.db._get_connection()
                        cursor.execute("""
                            SELECT coin_id FROM coingecko_market_data 
                            WHERE UPPER(symbol) = ?
                            ORDER BY timestamp DESC
                            LIMIT 1
                        """, (ref_token.upper(),))
                        
                        result = cursor.fetchone()
                        if result and result['coin_id']:
                            coingecko_id = result['coin_id']
                            ref_data = market_data.get(coingecko_id)
                    except Exception:
                        pass
                        
                if not ref_data or not isinstance(ref_data, dict):
                    continue
            
                # Time window for correlation calculation based on timeframe
                try:
                    if timeframe == "1h":
                        # Use 24h change for hourly predictions (short-term)
                        price_correlation_metric = abs(token_data.get('price_change_percentage_24h', 0) - 
                                                    ref_data.get('price_change_percentage_24h', 0))
                    elif timeframe == "24h":
                        # For daily, check if we have 7d change data available
                        if ('price_change_percentage_7d' in token_data and 
                            'price_change_percentage_7d' in ref_data):
                            price_correlation_metric = abs(token_data.get('price_change_percentage_7d', 0) - 
                                                        ref_data.get('price_change_percentage_7d', 0))
                        else:
                            # Fall back to 24h change if 7d not available
                            price_correlation_metric = abs(token_data.get('price_change_percentage_24h', 0) - 
                                                        ref_data.get('price_change_percentage_24h', 0))
                    else:  # 7d
                        # For weekly, use historical correlation if available
                        token_history = None
                        ref_history = None
                    
                        try:
                            token_history = self._get_historical_price_data(token, hours=30*24, timeframe=timeframe)
                            if isinstance(token_history, str) or token_history is None:
                                token_history = []
                        except Exception as th_err:
                            logger.logger.debug(f"Error getting token history: {str(th_err)}")
                            token_history = []
                        
                        try:
                            ref_history = self._get_historical_price_data(ref_token, hours=30*24, timeframe=timeframe)
                            if isinstance(ref_history, str) or ref_history is None:
                                ref_history = []
                        except Exception as rh_err:
                            logger.logger.debug(f"Error getting reference history: {str(rh_err)}")
                            ref_history = []
                    
                        # Safely extract prices from histories
                        if (isinstance(token_history, (list, tuple)) and 
                            isinstance(ref_history, (list, tuple)) and 
                            len(token_history) >= 14 and 
                            len(ref_history) >= 14):
                        
                            token_prices = []
                            ref_prices = []
                        
                            # Extract token prices safely
                            for entry in token_history[:14]:
                                if isinstance(entry, dict) and 'price' in entry:
                                    try:
                                        price = float(entry['price'])
                                        if price > 0:
                                            token_prices.append(price)
                                    except (ValueError, TypeError):
                                        pass
                                    
                            # Extract reference prices safely
                            for entry in ref_history[:14]:
                                if isinstance(entry, dict) and 'price' in entry:
                                    try:
                                        price = float(entry['price'])
                                        if price > 0:
                                            ref_prices.append(price)
                                    except (ValueError, TypeError):
                                        pass
                        
                            # Calculate historical correlation if we have enough data
                            if len(token_prices) == len(ref_prices) and len(token_prices) > 2:
                                try:
                                    # Calculate correlation coefficient
                                    historical_corr = np.corrcoef(token_prices, ref_prices)[0, 1] if 'numpy' in sys.modules else 0.5
                                    price_correlation_metric = abs(1 - historical_corr)
                                except Exception:
                                    # Fall back to 24h change if correlation fails
                                    price_correlation_metric = abs(token_data.get('price_change_percentage_24h', 0) - 
                                                                ref_data.get('price_change_percentage_24h', 0))
                            else:
                                price_correlation_metric = abs(token_data.get('price_change_percentage_24h', 0) - 
                                                            ref_data.get('price_change_percentage_24h', 0))
                        else:
                            price_correlation_metric = abs(token_data.get('price_change_percentage_24h', 0) - 
                                                        ref_data.get('price_change_percentage_24h', 0))
                
                    # Calculate price correlation (convert difference to correlation coefficient)
                    # Smaller difference = higher correlation
                    max_diff = 15 if timeframe == "1h" else 25 if timeframe == "24h" else 40
                    price_correlation = 1 - min(1, price_correlation_metric / max_diff)
                
                    # Volume correlation (simplified)
                    volume_correlation = 0.0
                    try:
                        token_volume = float(token_data.get('volume', 0) or token_data.get('total_volume', 0))
                        ref_volume = float(ref_data.get('volume', 0) or ref_data.get('total_volume', 0))
                    
                        if token_volume > 0 and ref_volume > 0:
                            volume_correlation = 1 - abs((token_volume - ref_volume) / max(token_volume, ref_volume))
                    except (ValueError, TypeError, ZeroDivisionError):
                        volume_correlation = 0.0
                
                    correlations[f'price_correlation_{ref_token}'] = price_correlation
                    correlations[f'volume_correlation_{ref_token}'] = volume_correlation
                
                except Exception as token_err:
                    logger.logger.debug(f"Error calculating correlation for {ref_token}: {str(token_err)}")
                    correlations[f'price_correlation_{ref_token}'] = 0.0
                    correlations[f'volume_correlation_{ref_token}'] = 0.0
        
            # Calculate average correlations
            price_correlations = [v for k, v in correlations.items() if 'price_correlation_' in k]
            volume_correlations = [v for k, v in correlations.items() if 'volume_correlation_' in k]
        
            correlations['avg_price_correlation'] = statistics.mean(price_correlations) if price_correlations else 0.0
            correlations['avg_volume_correlation'] = statistics.mean(volume_correlations) if volume_correlations else 0.0
        
            # Add BTC dominance correlation for longer timeframes
            if timeframe in ["24h", "7d"]:
                try:
                    # Try to get BTC data with TokenMappingManager approach
                    btc_data = market_data.get('BTC')
                    
                    if not btc_data and hasattr(self, 'config') and hasattr(self.config, 'token_mapper'):
                        try:
                            coingecko_id = self.config.token_mapper.symbol_to_coingecko_id('BTC')
                            if coingecko_id:
                                btc_data = market_data.get(coingecko_id)
                        except Exception:
                            pass
                    
                    if not btc_data:
                        btc_data = market_data.get('bitcoin')  # Direct fallback to known ID
                        
                    if btc_data and isinstance(btc_data, dict):
                        btc_mc = float(btc_data.get('market_cap', 0))
                        total_mc = 0
                        
                        # Calculate total market cap
                        for data in market_data.values():
                            if isinstance(data, dict):
                                try:
                                    mc = float(data.get('market_cap', 0))
                                    total_mc += mc
                                except (ValueError, TypeError):
                                    continue
                    
                        if total_mc > 0:
                            btc_dominance = (btc_mc / total_mc) * 100
                            btc_change = float(btc_data.get('price_change_percentage_24h', 0))
                            token_change = float(token_data.get('price_change_percentage_24h', 0))
                        
                            # Simple heuristic: if token moves opposite to BTC and dominance is high,
                            # it might be experiencing a rotation from/to BTC
                            btc_rotation_indicator = (btc_change * token_change < 0) and (btc_dominance > 50)
                        
                            correlations['btc_dominance'] = float(btc_dominance)
                            correlations['btc_rotation_indicator'] = bool(btc_rotation_indicator)
                except Exception as btc_err:
                    logger.logger.debug(f"Error calculating BTC dominance: {str(btc_err)}")
        
            # Add timeframe to correlations dictionary
            correlations['timeframe'] = timeframe
        
            # Store correlation data for any token using the generic method
            try:
                self.db.store_token_correlations(token, correlations)
            except Exception as db_err:
                logger.logger.debug(f"Error storing correlations: {str(db_err)}")
        
            logger.logger.debug(
                f"{token} correlations calculated ({timeframe}) - "
                f"Avg Price: {correlations.get('avg_price_correlation', 0.0):.2f}, "
                f"Avg Volume: {correlations.get('avg_volume_correlation', 0.0):.2f}"
            )
        
            return correlations
        
        except Exception as e:
            logger.log_error(f"Correlation Calculation - {token} ({timeframe})", str(e))
            # Return consistent type on error
            return {
                'avg_price_correlation': 0.0,
                'avg_volume_correlation': 0.0,
                'timeframe': timeframe
            }

    @ensure_naive_datetimes
    def _generate_correlation_report(self, market_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Generate a correlation report for the specified timeframe.
        
        Enhanced version that works with the database-driven token selection system
        and properly integrates with recent updates including:
        - Methods Rebuilt (July 31, 2025)
        - Critical Database Specifications
        - Token Management System Refactoring
        - Volume Key Access Fix
        
        Args:
            market_data: Market data dictionary with token market information
            timeframe: Timeframe for analysis ('1h', '24h', '7d')
        
        Returns:
            Formatted correlation report text
        """
        function_start = time.time()
        logger.logger.info(f"Generating correlation report for {timeframe} timeframe")
        
        try:
            # Check if database is initialized
            if not hasattr(self, 'db') or self.db is None:
                logger.logger.error("❌ FALLBACK: Database not initialized")
                from database import CryptoDatabase
                self.db = CryptoDatabase()
                logger.logger.info("✅ Created new database connection")
            
            # Get top tokens by market cap to analyze
            top_tokens = []
            try:
                # Use the database-driven token selection
                top_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=30)
                logger.logger.debug(f"✅ Using database-driven token selection: {top_tokens}")
            except Exception as db_error:
                logger.logger.warning(f"❌ FALLBACK: Database-driven token selection failed: {str(db_error)}")
                # Fall back to market data keys if database query fails
                top_tokens = list(market_data.keys())[:12]
                logger.logger.warning(f"⚠️ FALLBACK: Using market_data keys as tokens: {top_tokens}")
            
            # Ensure we have market data for selected tokens
            tokens_with_data = [t for t in top_tokens if t in market_data]
            if len(tokens_with_data) < 3:
                logger.logger.warning(f"⚠️ Insufficient tokens with data for correlation report")
                return ""
            
            # Build correlation matrix
            matrix = {}
            
            # Track market-wide metrics
            market_metrics = {
                "avg_24h_change": 0.0,
                "avg_volume_change": 0.0,
                "total_tokens": len(tokens_with_data)
            }
            
            # Calculate correlations between each pair of tokens
            total_correlations = 0
            total_correlation_value = 0.0
            
            # Track price changes for all tokens to calculate overall market direction
            price_changes = []
            
            # Process each token
            for token in tokens_with_data:
                token_data = market_data[token]
                
                # Extract price change using the Volume Key Access Fix pattern
                price_change = 0.0
                try:
                    # Try multiple possible keys for price change
                    for change_key in ['price_change_percentage_24h', 'price_change_24h_in_currency', 'usd_24h_change']:
                        if change_key in token_data:
                            try:
                                price_change = float(token_data[change_key])
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    price_changes.append(price_change)
                    
                    # Add token's contribution to market average
                    market_metrics["avg_24h_change"] += price_change
                except Exception as price_error:
                    logger.logger.debug(f"Error extracting price change for {token}: {str(price_error)}")
                
                # Build matrix row for this token
                matrix[token] = {}
                
                # Compare against all other tokens
                for other_token in tokens_with_data:
                    if token == other_token:
                        # Perfect correlation with self
                        matrix[token][other_token] = 1.0
                        continue
                    
                    # Calculate correlation between token pair
                    other_data = market_data[other_token]
                    
                    # Extract other token's price change
                    other_price_change = 0.0
                    for change_key in ['price_change_percentage_24h', 'price_change_24h_in_currency', 'usd_24h_change']:
                        if change_key in other_data:
                            try:
                                other_price_change = float(other_data[change_key])
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    # Simple direction correlation
                    if price_change == 0 or other_price_change == 0:
                        correlation = 0.0  # No change means no correlation
                    elif (price_change > 0 and other_price_change > 0) or (price_change < 0 and other_price_change < 0):
                        # Same direction movement
                        magnitude_ratio = min(abs(price_change), abs(other_price_change)) / max(abs(price_change), abs(other_price_change))
                        correlation = 0.5 + (0.5 * magnitude_ratio)  # 0.5-1.0 for same direction
                    else:
                        # Opposite direction movement
                        magnitude_ratio = min(abs(price_change), abs(other_price_change)) / max(abs(price_change), abs(other_price_change))
                        correlation = -0.5 - (0.5 * magnitude_ratio)  # -0.5 to -1.0 for opposite direction
                    
                    # Store in matrix
                    matrix[token][other_token] = correlation
                    
                    # Track total correlations for average calculation
                    if token != other_token:
                        total_correlations += 1
                        total_correlation_value += abs(correlation)
            
            # Calculate market-wide metrics
            if tokens_with_data:
                market_metrics["avg_24h_change"] /= len(tokens_with_data)
            
            # Determine market mode
            market_direction = "neutral"
            if market_metrics["avg_24h_change"] > 2.0:
                market_direction = "bullish"
            elif market_metrics["avg_24h_change"] < -2.0:
                market_direction = "bearish"
            
            # Calculate average correlation strength
            avg_correlation = 0.0
            if total_correlations > 0:
                avg_correlation = total_correlation_value / total_correlations
            
            # Determine market correlation state
            correlation_state = "mixed"
            if avg_correlation > 0.7:
                correlation_state = "highly correlated"
            elif avg_correlation > 0.4:
                correlation_state = "moderately correlated"
            elif avg_correlation < 0.2:
                correlation_state = "uncorrelated"
            
            # Generate the correlation report text
            report_id = f"corr_{timeframe}_{int(time.time())}"
            
            # Create the report text
            report_lines = [
                f"🔄 MARKET CORRELATION REPORT ({timeframe.upper()})",
                f"📊 Market Direction: {market_direction.upper()} ({market_metrics['avg_24h_change']:.2f}%)",
                f"⚖️ Correlation State: {correlation_state.upper()} ({avg_correlation:.2f})",
                ""
            ]
            
            # Add strongest correlations section
            strong_correlations = []
            for token in matrix:
                for other_token in matrix[token]:
                    if token != other_token:
                        strong_correlations.append((token, other_token, matrix[token][other_token]))
            
            # Sort by correlation strength (absolute value)
            strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Add top positive correlations
            report_lines.append("✅ TOP POSITIVE CORRELATIONS:")
            positive_correlations = [c for c in strong_correlations if c[2] > 0][:5]
            if positive_correlations:
                for token1, token2, corr in positive_correlations:
                    report_lines.append(f"  {token1} & {token2}: +{corr:.2f}")
            else:
                report_lines.append("  No significant positive correlations")
            
            report_lines.append("")
            
            # Add top negative correlations
            report_lines.append("❌ TOP NEGATIVE CORRELATIONS:")
            negative_correlations = [c for c in strong_correlations if c[2] < 0][:5]
            if negative_correlations:
                for token1, token2, corr in negative_correlations:
                    report_lines.append(f"  {token1} & {token2}: {corr:.2f}")
            else:
                report_lines.append("  No significant negative correlations")
            
            # Add summary insights
            report_lines.extend([
                "",
                "💡 MARKET INSIGHTS:",
                f"• Overall market sentiment is {market_direction} with {correlation_state} movement",
            ])
            
            # Add BTC dominance insight if BTC is in the tokens
            if "bitcoin" in market_data:
                btc_data = market_data["bitcoin"]
                btc_dominance = 0.0
                try:
                    if "market_cap_dominance" in btc_data:
                        btc_dominance = float(btc_data["market_cap_dominance"])
                    elif "market_cap" in btc_data:
                        # Calculate manually if we have all market caps
                        btc_mc = float(btc_data.get("market_cap", 0))
                        total_mc = sum(float(market_data[t].get("market_cap", 0)) for t in tokens_with_data)
                        if total_mc > 0:
                            btc_dominance = (btc_mc / total_mc) * 100
                    
                    report_lines.append(f"• BTC Dominance: {btc_dominance:.2f}%")
                except Exception as btc_err:
                    logger.logger.debug(f"Error calculating BTC dominance: {str(btc_err)}")
            
            # Add trading recommendation based on correlation
            if avg_correlation > 0.7:
                report_lines.append("• High correlations suggest selective trading - focus on fundamentals")
            elif avg_correlation < 0.3:
                report_lines.append("• Low correlations suggest opportunity for diversification")
            
            # Finalize report
            report_text = "\n".join(report_lines)
            
            # Store correlation data for all analyzed tokens
            for token in tokens_with_data:
                token_correlations = {
                    "avg_price_correlation": sum(matrix[token].values()) / len(matrix[token]),
                    "correlations": matrix[token],
                    "timeframe": timeframe
                }
                
                try:
                    self.db.store_token_correlations(token, token_correlations)
                except Exception as db_err:
                    logger.logger.debug(f"Error storing correlations: {str(db_err)}")
            
            # Save full correlation report for future reference
            try:
                self._save_correlation_report(report_id, matrix, timeframe, report_text)
            except Exception as save_err:
                logger.logger.debug(f"Error saving correlation report: {str(save_err)}")
            
            # Log performance
            function_time = time.time() - function_start
            logger.logger.info(f"Generated {timeframe} correlation report in {function_time:.3f}s")
            
            return report_text
            
        except Exception as e:
            logger.log_error(f"Correlation Report Generation - {timeframe}", str(e))
            # Return empty string on error to avoid breaking the posting flow
            return ""

    def _is_matrix_duplicate(self, matrix: Dict[str, Dict[str, float]], timeframe: str) -> bool:
        """
        STRICT CORRELATION MATRIX DUPLICATE DETECTION - BILLION DOLLAR OPTIMIZATION
    
        This method now implements extremely strict duplicate detection to prevent
        correlation matrices from posting too frequently and interfering with 
        high-value prediction content.
    
        Args:
            matrix: Correlation matrix to check (mostly ignored now)
            timeframe: Timeframe for analysis
    
        Returns:
            True (always assume duplicate to block correlation posts)
        """
        try:
            # ALWAYS BLOCK CORRELATION MATRICES - They interfere with billion dollar predictions
            logger.info(f"🚫 Correlation matrix blocked for {timeframe} - strict duplicate detection active")
        
            # Check if correlation matrix is disabled globally
            if getattr(self, 'disable_correlation_matrix', True):
                logger.info(f"🚫 Correlation matrix globally disabled for {timeframe}")
                return True
        
            # Even if not globally disabled, implement ultra-strict controls
            try:
                conn, cursor = self.db._get_connection()
            
                # Check for ANY correlation content in the last 48 hours (very strict)
                cursor.execute("""
                    SELECT content, timestamp FROM posted_content
                    WHERE (
                        content LIKE '%CORRELATION%' OR 
                        content LIKE '%correlation%' OR
                        content LIKE '%aligned%' OR
                        content LIKE '%moving together%' OR
                        content LIKE '%inversely%' OR
                        content LIKE '%++ %' OR
                        content LIKE '%-- %'
                    )
                    AND timestamp >= datetime('now', '-48 hours')
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
            
                recent_correlation = cursor.fetchone()
            
                if recent_correlation:
                    post_time = datetime.fromisoformat(recent_correlation['timestamp']) if isinstance(recent_correlation['timestamp'], str) else recent_correlation['timestamp']
                    now = datetime.now()
                    hours_since_post = (now - post_time).total_seconds() / 3600
                
                    logger.warning(f"🚫 Found correlation-like content posted {hours_since_post:.1f} hours ago - blocking")
                    return True
            
                # Check for sufficient recent prediction content
                cursor.execute("""
                    SELECT COUNT(*) as count FROM posted_content
                    WHERE (
                        content LIKE '%prediction%' OR
                        content LIKE '%Prediction%' OR
                        content LIKE '%PREDICTION%' OR
                        content LIKE '%$%' OR
                        content LIKE '%📊%' OR
                        content LIKE '%🎯%'
                    )
                    AND timestamp >= datetime('now', '-12 hours')
                """)
           
                recent_predictions = cursor.fetchone()['count']
           
                if recent_predictions > 0:
                    logger.info(f"🚫 Found {recent_predictions} recent predictions - correlation not needed")
                    return True
           
                # Check for any substantial content in the last 6 hours
                cursor.execute("""
                    SELECT COUNT(*) as count FROM posted_content
                    WHERE timestamp >= datetime('now', '-6 hours')
                    AND LENGTH(content) > 50
                """)
           
                recent_content = cursor.fetchone()['count']
           
                if recent_content > 0:
                    logger.info(f"🚫 Found {recent_content} recent substantial posts - correlation redundant")
                    return True
           
                # Even if no recent content, still be very restrictive
                # Only allow correlation matrix once per week maximum
                cursor.execute("""
                    SELECT COUNT(*) as count FROM posted_content
                    WHERE (
                        content LIKE '%CORRELATION%' OR 
                        content LIKE '%correlation%'
                    )
                    AND timestamp >= datetime('now', '-168 hours')
                """)
           
                weekly_correlations = cursor.fetchone()['count']
           
                if weekly_correlations > 0:
                    logger.warning(f"🚫 Found {weekly_correlations} correlation posts this week - maximum frequency exceeded")
                    return True
           
                # Final check - only allow if it's an absolute content emergency
                # and it's been more than 24 hours since any post
                cursor.execute("""
                    SELECT COUNT(*) as count FROM posted_content
                    WHERE timestamp >= datetime('now', '-24 hours')
                """)
           
                any_recent_posts = cursor.fetchone()['count']
           
                if any_recent_posts > 0:
                    logger.info(f"🚫 Found posts in last 24 hours - correlation not needed as emergency fallback")
                    return True
           
                # If we somehow get here, it means:
                # - No posts in 24 hours
                # - No correlations in a week
                # - This would be absolute emergency fallback
                logger.warning(f"⚠️ Correlation matrix might be allowed for {timeframe} - emergency fallback scenario")
                return False
           
            except Exception as db_error:
                logger.log_error(f"Matrix Duplicate Check DB Error - {timeframe}", str(db_error))
                # On database error, always block to be safe
                return True
       
        except Exception as e:
            logger.log_error(f"Matrix Duplication Check - {timeframe}", str(e))
            # On any error, always assume duplicate to block correlation posts
            return True

    def _calculate_correlation_metrics(self, matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        DISABLED CORRELATION METRICS - BILLION DOLLAR OPTIMIZATION
    
        This method is disabled as part of the correlation matrix removal.
        Returns minimal default metrics to prevent errors in any remaining code
        that might call this method.
    
        Args:
            matrix: Correlation matrix as a nested dict (ignored)
    
        Returns:
            Default metrics dictionary (minimal data)
        """
        try:
            # Log that correlation metrics calculation was attempted
            logger.info("🚫 Correlation metrics calculation disabled - returning defaults")
        
            # Return minimal default metrics to prevent errors
            default_metrics = {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'min_correlation': 0.0,
                'unity_percentage': 0.0,
                'diversity_score': 0.0,
                'token_count': 0,
                'status': 'disabled',
                'message': 'Correlation metrics disabled - focusing on prediction metrics instead'
            }
        
            # If correlation matrix is disabled, return immediately
            if getattr(self, 'disable_correlation_matrix', True):
                logger.debug("🚫 Correlation metrics blocked - matrix disabled")
                return default_metrics
        
            # Even if not disabled, don't perform actual calculations
            # This prevents any computational overhead from correlation analysis
            logger.debug("🚫 Correlation metrics calculation skipped - resource optimization")
        
            return default_metrics
        
        except Exception as e:
            logger.log_error("Disabled Correlation Metrics", str(e))
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'min_correlation': 0.0,
                'unity_percentage': 0.0,
                'diversity_score': 0.0,
                'token_count': 0,
                'status': 'error',
                'message': f'Correlation metrics error: {str(e)[:50]}'
            }     

    def _save_correlation_report(self, report_id: str, matrix: Dict[str, Dict[str, float]], 
                            timeframe: str, report_text: str) -> None:
        """
        Store correlation report in SQL database tables
        
        CRITICAL: This method should ONLY store data, NOT generate additional text
        The report_text parameter contains the already-formatted text from _generate_correlation_report()
        
        Args:
            report_id: Unique ID for the report
            matrix: Correlation matrix data
            timeframe: Timeframe used for analysis  
            report_text: Formatted report text (ALREADY GENERATED - don't create more)
        """
        conn = None
        
        try:
            conn, cursor = self.db._get_connection()
            
            # Extract report metadata FROM the provided report_text (don't generate new text)
            market_direction = "neutral"
            correlation_state = "mixed"
            if "BULLISH" in report_text:
                market_direction = "bullish"
            elif "BEARISH" in report_text:
                market_direction = "bearish"
                
            if "HIGHLY CORRELATED" in report_text:
                correlation_state = "highly correlated"
            elif "MODERATELY CORRELATED" in report_text:
                correlation_state = "moderately correlated"  
            elif "UNCORRELATED" in report_text:
                correlation_state = "uncorrelated"
                
            # Calculate average correlation from matrix data
            all_correlations = []
            for token_correlations in matrix.values():
                all_correlations.extend(token_correlations.values())
            avg_correlation = sum(all_correlations) / len(all_correlations) if all_correlations else 0.0
            
            # Store main report (using the PROVIDED report_text, not generating new text)
            cursor.execute("""
                INSERT INTO correlation_reports (
                    timestamp, report_id, timeframe, report_text,
                    market_direction, correlation_state, avg_correlation, tokens_analyzed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                report_id,
                timeframe, 
                report_text,  # Use the ALREADY FORMATTED text - don't create duplicate content
                market_direction,
                correlation_state,
                avg_correlation,
                len(matrix)
            ))
            
            # Store correlation pairs (data only, no text generation)
            for token_a, correlations in matrix.items():
                for token_b, correlation_value in correlations.items():
                    if token_a != token_b:  # Don't store self-correlations
                        cursor.execute("""
                            INSERT INTO correlation_pairs (
                                report_id, token_a, token_b, correlation_value, timestamp
                            ) VALUES (?, ?, ?, ?, ?)
                        """, (report_id, token_a, token_b, correlation_value, datetime.now()))
            
            conn.commit()
            logger.logger.debug(f"✅ Saved correlation report {report_id} to database")
            
        except Exception as e:
            logger.logger.error(f"❌ Error saving correlation report: {str(e)}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close() 

    def _should_allow_correlation_matrix(self, timeframe: str) -> bool:
        """
        STRICT CORRELATION MATRIX GATE - BILLION DOLLAR PROTECTION
    
        Ultra-strict gate to prevent correlation matrices from interfering
        with high-value prediction content. Only allows correlation in
        absolute emergency scenarios.
    
        Args:
            timeframe: Timeframe being checked
    
        Returns:
            False (almost always blocks correlation matrices)
        """
        try:
            # Primary check - is correlation globally disabled?
            if getattr(self, 'disable_correlation_matrix', True):
                logger.debug(f"🚫 Correlation globally disabled for {timeframe}")
                return False
        
            # Secondary check - database activity
            try:
                conn, cursor = self.db._get_connection()
            
                # Check if we have ANY content in the last 48 hours
                cursor.execute("""
                    SELECT COUNT(*) as count FROM posted_content
                    WHERE timestamp >= datetime('now', '-48 hours')
                """)
            
                recent_content = cursor.fetchone()['count']
            
                if recent_content > 0:
                    logger.info(f"🚫 Found {recent_content} posts in 48h - correlation not needed")
                    return False
            
                # Check if system is actively generating predictions
                cursor.execute("""
                    SELECT COUNT(*) as count FROM predictions
                    WHERE timestamp >= datetime('now', '-24 hours')
                """)
            
                recent_predictions = cursor.fetchone()['count']
            
                if recent_predictions > 0:
                    logger.info(f"🚫 System generated {recent_predictions} predictions in 24h - correlation redundant")
                    return False
            
                # Final emergency check - only if system has been completely silent
                cursor.execute("""
                    SELECT COUNT(*) as count FROM posted_content
                    WHERE timestamp >= datetime('now', '-72 hours')
                """)
            
                very_recent_content = cursor.fetchone()['count']
            
                if very_recent_content > 0:
                    logger.info(f"🚫 Found content in 72h window - correlation still not needed")
                    return False
            
                # Absolute emergency - no content for 72+ hours
                logger.warning(f"⚠️ EMERGENCY: No content for 72+ hours - correlation might be allowed for {timeframe}")
                return True
            
            except Exception as db_error:
                logger.log_error("Correlation Gate DB Check", str(db_error))
                return False
        
        except Exception as e:
            logger.log_error("Correlation Matrix Gate", str(e))
            return False

    def _calculate_momentum_score(self, token: str, market_data: Dict[str, Any], timeframe: str = "1h") -> float:
        """
        Calculate a momentum score (0-100) for a token based on various metrics
        Adjusted for different timeframes
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
            
        Returns:
            Momentum score (0-100)
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return 50.0  # Neutral score
            
            # Get basic metrics
            price_change = token_data.get('price_change_percentage_24h', 0)
            volume = token_data.get('volume', 0)
        
            # Get historical volume for volume change - adjust window based on timeframe
            if timeframe == "1h":
                window_minutes = 60  # Last hour for hourly predictions
            elif timeframe == "24h":
                window_minutes = 24 * 60  # Last day for daily predictions
            else:  # 7d
                window_minutes = 7 * 24 * 60  # Last week for weekly predictions
                
            historical_volume = self._get_historical_volume_data(token, timeframe=timeframe)
            volume_change, _ = self._analyze_volume_trend(volume, historical_volume, timeframe=timeframe)
        
            # Get smart money indicators
            smart_money = self._analyze_smart_money_indicators(token, token_data, timeframe=timeframe)
        
            # Get market comparison
            vs_market = self._analyze_token_vs_market(token, market_data, timeframe=timeframe)
        
            # Calculate score components (0-20 points each)
            # Adjust price score scaling based on timeframe
            if timeframe == "1h":
                price_range = 5.0  # ±5% for hourly
            elif timeframe == "24h":
                price_range = 10.0  # ±10% for daily
            else:  # 7d
                price_range = 20.0  # ±20% for weekly
                
            price_score = min(20, max(0, (price_change + price_range) * (20 / (2 * price_range))))
        
            # Adjust volume score scaling based on timeframe
            if timeframe == "1h":
                volume_range = 10.0  # ±10% for hourly
            elif timeframe == "24h":
                volume_range = 20.0  # ±20% for daily
            else:  # 7d
                volume_range = 40.0  # ±40% for weekly
                
            volume_score = min(20, max(0, (volume_change + volume_range) * (20 / (2 * volume_range))))
        
            # Smart money score - additional indicators for longer timeframes
            smart_money_score = 0
            if smart_money.get('abnormal_volume', False):
                smart_money_score += 5
            if smart_money.get('stealth_accumulation', False):
                smart_money_score += 5
            if smart_money.get('volume_cluster_detected', False):
                smart_money_score += 5
            if smart_money.get('volume_z_score', 0) > 1.0:
                smart_money_score += 5
                
            # Add pattern metrics for longer timeframes
            if timeframe in ["24h", "7d"] and 'pattern_metrics' in smart_money:
                pattern_metrics = smart_money['pattern_metrics']
                if pattern_metrics.get('volume_breakout', False):
                    smart_money_score += 5
                if pattern_metrics.get('consistent_high_volume', False):
                    smart_money_score += 5
                    
            smart_money_score = min(20, smart_money_score)
        
            # Market comparison score
            market_score = 0
            if vs_market.get('outperforming_market', False):
                market_score += 10
            market_score += min(10, max(0, (vs_market.get('vs_market_avg_change', 0) + 5)))
            market_score = min(20, market_score)
        
            # Trend consistency score - higher standards for longer timeframes
            if timeframe == "1h":
                trend_score = 20 if all([price_score > 10, volume_score > 10, smart_money_score > 5, market_score > 10]) else 0
            elif timeframe == "24h":
                trend_score = 20 if all([price_score > 12, volume_score > 12, smart_money_score > 8, market_score > 12]) else 0
            else:  # 7d
                trend_score = 20 if all([price_score > 15, volume_score > 15, smart_money_score > 10, market_score > 15]) else 0
        
            # Calculate total score (0-100)
            # Adjust component weights based on timeframe
            if timeframe == "1h":
                # For hourly, recent price action and smart money more important
                total_score = (
                    price_score * 0.25 +
                    volume_score * 0.2 +
                    smart_money_score * 0.25 +
                    market_score * 0.15 +
                    trend_score * 0.15
                ) * 1.0
            elif timeframe == "24h":
                # For daily, balance factors with more weight to market comparison
                total_score = (
                    price_score * 0.2 +
                    volume_score * 0.2 +
                    smart_money_score * 0.2 +
                    market_score * 0.25 +
                    trend_score * 0.15
                ) * 1.0
            else:  # 7d
                # For weekly, market factors and trend consistency more important
                total_score = (
                    price_score * 0.15 +
                    volume_score * 0.15 +
                    smart_money_score * 0.2 +
                    market_score * 0.3 +
                    trend_score * 0.2
                ) * 1.0
        
            return total_score
        
        except Exception as e:
            logger.log_error(f"Momentum Score - {token} ({timeframe})", str(e))
            return 50.0  # Neutral score on error

    @ensure_naive_datetimes
    def _format_prediction_tweet(self, token: str, prediction: Dict[str, Any], market_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Enhanced prediction tweet formatter with content expansion and comprehensive debugging
        
        Formats prediction data into Twitter-ready content with intelligent expansion for short content.
        Maintains exact same method signature and database compatibility as existing system.
        
        Args:
            token (str): Token symbol (e.g., "BTC", "ETH")
            prediction (Dict[str, Any]): Prediction data from prediction engine
            market_data (Dict[str, Any]): Current market data for all tokens
            timeframe (str): Timeframe identifier ("1h", "24h", "7d")
        
        Returns:
            str: Formatted tweet content ready for posting via _post_analysis()
        """
        # ================================================================
        # 🔍 PRE-PROCESSING DATA FLOW DEBUG
        # ================================================================
        
        logger.logger.debug(f"🚀 ENHANCED FORMAT PREDICTION START: {token} ({timeframe})")
        logger.logger.debug(f"📊 INPUT VALIDATION:")
        logger.logger.debug(f"   • Token: '{token}' (type: {type(token)})")
        logger.logger.debug(f"   • Prediction keys: {list(prediction.keys()) if prediction else 'None'}")
        logger.logger.debug(f"   • Market data available: {token in market_data if market_data else False}")
        logger.logger.debug(f"   • Timeframe: '{timeframe}' (type: {type(timeframe)})")
        
        try:
            # ================================================================
            # 🔧 NORMALIZE TRADING PREDICTION FORMAT (EXISTING LOGIC)
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 1: Normalizing prediction format")
            
            # Handle both flat and nested prediction structures (preserve existing logic)
            if 'prediction' in prediction and isinstance(prediction['prediction'], dict):
                pred_data = prediction['prediction'].copy()
                base_prediction = prediction.copy()
                logger.logger.debug(f"✅ Using nested prediction structure")
            else:
                pred_data = prediction.copy()
                base_prediction = prediction.copy()
                logger.logger.debug(f"✅ Using flat prediction structure")
            
            # Check if this is a trading-enhanced prediction (preserve existing logic)
            is_trading_format = any(field in base_prediction for field in ['action', 'entry_price', 'stop_loss', 'take_profit'])
            
            if is_trading_format:
                logger.logger.debug(f"🔄 Converting trading prediction format for {token} analysis display")
                
                # Get trading-specific fields (existing logic preserved)
                entry_price = base_prediction.get('entry_price', 0)
                take_profit = base_prediction.get('take_profit', 0)
                stop_loss = base_prediction.get('stop_loss', 0)
                action = base_prediction.get('action', 'HOLD')
                
                # Convert to analysis format
                if action in ['BUY', 'LONG']:
                    pred_data['sentiment'] = 'BULLISH'
                    pred_data['price'] = take_profit if take_profit > 0 else entry_price
                    pred_data['lower_bound'] = entry_price
                    pred_data['upper_bound'] = take_profit if take_profit > 0 else entry_price * 1.05
                elif action in ['SELL', 'SHORT']:
                    pred_data['sentiment'] = 'BEARISH'
                    pred_data['price'] = stop_loss if stop_loss > 0 else entry_price
                    pred_data['lower_bound'] = stop_loss if stop_loss > 0 else entry_price * 0.95
                    pred_data['upper_bound'] = entry_price
                else:  # HOLD
                    pred_data['sentiment'] = 'NEUTRAL'
                    pred_data['price'] = entry_price
                    pred_data['lower_bound'] = entry_price * 0.98
                    pred_data['upper_bound'] = entry_price * 1.02
            
            # ================================================================
            # 📊 EXTRACT PREDICTION DATA WITH FALLBACKS
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 2: Extracting prediction data")
            
            # Extract key prediction fields with comprehensive fallbacks
            sentiment = pred_data.get("sentiment", "NEUTRAL")
            rationale = pred_data.get("rationale", "")
            key_factors = pred_data.get("key_factors", [])
            
            # Price data extraction (prioritize normalized data)
            price = pred_data.get("price", 0)
            confidence = pred_data.get("confidence", 70)
            lower_bound = pred_data.get("lower_bound", 0)
            upper_bound = pred_data.get("upper_bound", 0)
            percent_change = pred_data.get("percent_change", 0)
            
            logger.logger.debug(f"📈 PREDICTION DATA EXTRACTED:")
            logger.logger.debug(f"   • Sentiment: {sentiment}")
            logger.logger.debug(f"   • Price: ${price}")
            logger.logger.debug(f"   • Confidence: {confidence}%")
            logger.logger.debug(f"   • Range: ${lower_bound} - ${upper_bound}")
            logger.logger.debug(f"   • Percent change: {percent_change}%")
            logger.logger.debug(f"   • Rationale length: {len(rationale)} chars")
            logger.logger.debug(f"   • Key factors: {len(key_factors)} items")
            
            # ================================================================
            # 💰 GET CURRENT PRICE FROM MARKET DATA (EXISTING LOGIC)
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 3: Extracting current market price")
            
            # Get current price from market data with multiple fallbacks
            token_data = market_data.get(token, {})
            current_price = (
                token_data.get("current_price") or 
                token_data.get("price") or
                base_prediction.get("current_price") or
                base_prediction.get("entry_price") or
                price or
                0
            )
            
            logger.logger.debug(f"💰 CURRENT PRICE SOURCES:")
            logger.logger.debug(f"   • Market data current_price: {token_data.get('current_price', 'N/A')}")
            logger.logger.debug(f"   • Market data price: {token_data.get('price', 'N/A')}")
            logger.logger.debug(f"   • Prediction current_price: {base_prediction.get('current_price', 'N/A')}")
            logger.logger.debug(f"   • Final current_price: ${current_price}")
            
            # ================================================================
            # ⏰ FORMAT TIMEFRAME FOR DISPLAY (EXISTING LOGIC)
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 4: Formatting timeframe display")
            
            if timeframe == "1h":
                display_timeframe = "1HR"
                time_context = "hour"
            elif timeframe == "24h":
                display_timeframe = "24HR"
                time_context = "24 hours"
            else:  # 7d
                display_timeframe = "7DAY"
                time_context = "week"
            
            logger.logger.debug(f"⏰ TIMEFRAME FORMATTING:")
            logger.logger.debug(f"   • Input: '{timeframe}'")
            logger.logger.debug(f"   • Display: '{display_timeframe}'")
            logger.logger.debug(f"   • Context: '{time_context}'")
            
            # ================================================================
            # 🎨 DETERMINE STYLE AND FORMAT INITIAL TWEET
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 5: Determining tweet style and formatting")
            
            # Determine style based on confidence and market conditions (existing logic)
            use_insider_style = self._should_use_insider_style(confidence, sentiment, rationale)
            
            logger.logger.debug(f"🎨 STYLE SELECTION:")
            logger.logger.debug(f"   • Confidence: {confidence}%")
            logger.logger.debug(f"   • Sentiment: {sentiment}")
            logger.logger.debug(f"   • Use insider style: {use_insider_style}")
            
            if use_insider_style:
                initial_tweet = self._format_insider_tip_style(
                    token, sentiment, confidence, price, percent_change,
                    lower_bound, upper_bound, time_context, rationale, key_factors
                )
                style_used = "insider_tip"
            else:
                initial_tweet = self._format_professional_analysis_style(
                    token, sentiment, confidence, price, percent_change,
                    lower_bound, upper_bound, display_timeframe, rationale, key_factors
                )
                style_used = "professional"
            
            logger.logger.debug(f"✅ INITIAL TWEET GENERATED:")
            logger.logger.debug(f"   • Style used: {style_used}")
            logger.logger.debug(f"   • Length: {len(initial_tweet)} chars")
            logger.logger.debug(f"   • Preview: '{initial_tweet[:80]}...'")
            
            # ================================================================
            # 📏 CONTENT LENGTH VALIDATION AND EXPANSION
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 6: Content length validation and expansion")
            
            # Get tweet constraints
            min_length = config.TWEET_CONSTRAINTS['MIN_LENGTH']
            max_length = config.TWEET_CONSTRAINTS['MAX_LENGTH']
            hard_stop = config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
            
            logger.logger.debug(f"📏 TWEET CONSTRAINTS:")
            logger.logger.debug(f"   • MIN_LENGTH: {min_length}")
            logger.logger.debug(f"   • MAX_LENGTH: {max_length}")
            logger.logger.debug(f"   • HARD_STOP_LENGTH: {hard_stop}")
            logger.logger.debug(f"   • Current length: {len(initial_tweet)}")
            
            # Check if content needs expansion
            if len(initial_tweet) < min_length:
                logger.logger.warning(f"⚠️ CONTENT TOO SHORT: {len(initial_tweet)} < {min_length} chars")
                logger.logger.debug(f"🔧 INITIATING CONTENT EXPANSION")
                
                # Expand content intelligently
                expanded_tweet = self._expand_prediction_content(
                    initial_tweet, token, prediction, market_data, timeframe, 
                    min_length, max_length
                )
                
                logger.logger.info(f"✅ CONTENT EXPANDED:")
                logger.logger.debug(f"   • Original: {len(initial_tweet)} chars")
                logger.logger.debug(f"   • Expanded: {len(expanded_tweet)} chars")
                logger.logger.debug(f"   • Expansion: +{len(expanded_tweet) - len(initial_tweet)} chars")
                
                final_tweet = expanded_tweet
            else:
                logger.logger.debug(f"✅ Content length acceptable: {len(initial_tweet)} >= {min_length}")
                final_tweet = initial_tweet
            
            # ================================================================
            # 📈 ADD ACCURACY TRACKING (EXISTING LOGIC PRESERVED)
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 7: Adding accuracy tracking")
            
            # Add accuracy tracking if available and decent (>60%)
            try:
                performance = self.db.get_prediction_performance(token=token, timeframe=timeframe)
                if performance and performance[0]["total_predictions"] > 5:
                    accuracy = performance[0]["accuracy_rate"]
                    if accuracy > 60:  # Only show if accuracy is decent
                        accuracy_addition = f"\n\nTrack record: {accuracy:.1f}% on {performance[0]['total_predictions']} calls"
                        
                        # Only add if it won't exceed hard stop length
                        if len(final_tweet) + len(accuracy_addition) <= hard_stop:
                            final_tweet += accuracy_addition
                            logger.logger.debug(f"✅ Added accuracy tracking: {accuracy:.1f}% on {performance[0]['total_predictions']} calls")
                        else:
                            logger.logger.debug(f"⚠️ Skipped accuracy tracking: would exceed hard stop ({len(final_tweet) + len(accuracy_addition)} > {hard_stop})")
                    else:
                        logger.logger.debug(f"⚠️ Skipped accuracy tracking: too low ({accuracy:.1f}% <= 60%)")
                else:
                    logger.logger.debug(f"⚠️ Skipped accuracy tracking: insufficient data")
            except Exception as perf_error:
                logger.logger.debug(f"⚠️ Could not get performance data for {token}: {str(perf_error)}")
            
            # ================================================================
            # ✅ FINAL VALIDATION AND RETURN
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 8: Final validation")
            
            # Final length check
            final_length = len(final_tweet)
            
            if final_length < min_length:
                logger.logger.warning(f"⚠️ FINAL WARNING: Tweet still too short ({final_length} < {min_length} chars)")
            elif final_length > hard_stop:
                logger.logger.warning(f"⚠️ FINAL WARNING: Tweet too long ({final_length} > {hard_stop} chars)")
                # Emergency truncation
                final_tweet = self._smart_truncate_prediction(final_tweet, hard_stop)
                logger.logger.debug(f"🔧 Emergency truncated to {len(final_tweet)} chars")
            else:
                logger.logger.debug(f"✅ Final length check passed: {final_length} chars")
            
            # Log final tweet details
            logger.logger.info(f"✅ PREDICTION TWEET COMPLETE: {token} ({timeframe})")
            logger.logger.debug(f"📊 FINAL STATS:")
            logger.logger.debug(f"   • Token: {token}")
            logger.logger.debug(f"   • Timeframe: {timeframe}")
            logger.logger.debug(f"   • Final length: {len(final_tweet)} chars")
            logger.logger.debug(f"   • Style: {style_used}")
            logger.logger.debug(f"   • Expanded: {'Yes' if len(final_tweet) > len(initial_tweet) else 'No'}")
            logger.logger.debug(f"   • Preview: '{final_tweet[:100]}...'")
            
            return final_tweet
            
        except Exception as e:
            # ================================================================
            # ❌ COMPREHENSIVE ERROR HANDLING
            # ================================================================
            
            error_msg = str(e)
            logger.logger.error(f"❌ PREDICTION FORMATTING FAILED: {token} ({timeframe})")
            logger.logger.error(f"🔍 ERROR DETAILS:")
            logger.logger.error(f"   • Exception type: {type(e).__name__}")
            logger.logger.error(f"   • Error message: {error_msg}")
            logger.logger.error(f"   • Token: {token}")
            logger.logger.error(f"   • Timeframe: {timeframe}")
            logger.logger.error(f"   • Prediction keys: {list(prediction.keys()) if prediction else 'None'}")
            
            logger.log_error(f"Format Prediction Tweet - {token} ({timeframe})", str(e))
            
            # ================================================================
            # 🚨 EMERGENCY CONTENT GENERATION
            # ================================================================
            
            logger.logger.warning(f"🚨 GENERATING EMERGENCY CONTENT for {token}")
            
            try:
                # Create minimal emergency tweet with comprehensive fallbacks
                emergency_price = (
                    prediction.get('price', 0) or 
                    prediction.get('entry_price', 0) or
                    prediction.get('take_profit', 0) or
                    market_data.get(token, {}).get('current_price', 0)
                )
                emergency_confidence = prediction.get('confidence', 50)
                
                emergency_tweet = f"{token} {timeframe.upper()} prediction: ${emergency_price:.4f} - {emergency_confidence:.0f}% confidence"
                
                logger.logger.warning(f"✅ EMERGENCY CONTENT GENERATED:")
                logger.logger.debug(f"   • Length: {len(emergency_tweet)} chars")
                logger.logger.debug(f"   • Content: '{emergency_tweet}'")
                
                return emergency_tweet
                
            except Exception as emergency_error:
                logger.logger.error(f"💥 EMERGENCY CONTENT GENERATION FAILED: {str(emergency_error)}")
                logger.log_error(f"Emergency Tweet Generation - {token}", str(emergency_error))
                
                # Absolute last resort
                final_emergency = f"{token} {timeframe} analysis - Market conditions under review"
                logger.logger.error(f"🆘 USING ABSOLUTE LAST RESORT: '{final_emergency}'")
                return final_emergency


    def _expand_prediction_content(self, initial_content: str, token: str, prediction: Dict[str, Any], 
                                market_data: Dict[str, Any], timeframe: str, min_length: int, max_length: int) -> str:
        """
        Intelligently expand short prediction content to meet minimum length requirements
        
        Args:
            initial_content (str): Original tweet content that's too short
            token (str): Token symbol
            prediction (Dict[str, Any]): Prediction data
            market_data (Dict[str, Any]): Market data
            timeframe (str): Timeframe
            min_length (int): Minimum required length
            max_length (int): Maximum allowed length
        
        Returns:
            str: Expanded content meeting length requirements
        """
        logger.logger.debug(f"🔧 CONTENT EXPANSION START: {token}")
        logger.logger.debug(f"   • Initial length: {len(initial_content)}")
        logger.logger.debug(f"   • Target: {min_length}-{max_length} chars")
        
        expanded_content = initial_content
        
        try:
            # ================================================================
            # 📊 EXPANSION STRATEGY 1: ADD MARKET CONTEXT
            # ================================================================
            
            if len(expanded_content) < min_length:
                logger.logger.debug(f"🔧 EXPANSION STRATEGY 1: Adding market context")
                
                token_data = market_data.get(token, {})
                price_change_24h = token_data.get('price_change_percentage_24h', 0)
                volume = token_data.get('total_volume', 0)
                
                # Add market context based on data availability
                market_context_additions = []
                
                if price_change_24h != 0:
                    if abs(price_change_24h) > 5:
                        market_context_additions.append(f"Strong 24h momentum: {price_change_24h:+.2f}%")
                    else:
                        market_context_additions.append(f"24h change: {price_change_24h:+.2f}%")
                
                if volume > 0:
                    volume_display = f"${volume:,.0f}" if volume > 1000000 else f"${volume:,.0f}"
                    market_context_additions.append(f"Volume: {volume_display}")
                
                # Add the most relevant market context
                if market_context_additions:
                    addition = f"\n\n{market_context_additions[0]}"
                    if len(expanded_content) + len(addition) <= max_length:
                        expanded_content += addition
                        logger.logger.debug(f"✅ Added market context: +{len(addition)} chars")
            
            # ================================================================
            # 📊 EXPANSION STRATEGY 2: ADD TECHNICAL DETAILS
            # ================================================================
            
            if len(expanded_content) < min_length:
                logger.logger.debug(f"🔧 EXPANSION STRATEGY 2: Adding technical details")
                
                # Extract technical details from prediction
                rationale = prediction.get('rationale', '')
                key_factors = prediction.get('key_factors', [])
                
                technical_additions = []
                
                # Add rationale snippet if available
                if rationale and len(rationale) > 20:
                    rationale_snippet = rationale[:100] + "..." if len(rationale) > 100 else rationale
                    technical_additions.append(f"Analysis: {rationale_snippet}")
                
                # Add key factor if available
                if key_factors and len(key_factors) > 0:
                    key_factor = str(key_factors[0])[:80] + "..." if len(str(key_factors[0])) > 80 else str(key_factors[0])
                    technical_additions.append(f"Key factor: {key_factor}")
                
                # Add the most relevant technical detail
                if technical_additions:
                    addition = f"\n\n{technical_additions[0]}"
                    if len(expanded_content) + len(addition) <= max_length:
                        expanded_content += addition
                        logger.logger.debug(f"✅ Added technical details: +{len(addition)} chars")
            
            # ================================================================
            # 📊 EXPANSION STRATEGY 3: ADD TIMEFRAME CONTEXT
            # ================================================================
            
            if len(expanded_content) < min_length:
                logger.logger.debug(f"🔧 EXPANSION STRATEGY 3: Adding timeframe context")
                
                timeframe_context = {
                    "1h": "Short-term momentum signal",
                    "24h": "Daily trend analysis",
                    "7d": "Weekly outlook assessment"
                }
                
                context = timeframe_context.get(timeframe, "Market analysis")
                addition = f"\n\n{context} for {token}."
                
                if len(expanded_content) + len(addition) <= max_length:
                    expanded_content += addition
                    logger.logger.debug(f"✅ Added timeframe context: +{len(addition)} chars")
            
            # ================================================================
            # 📊 EXPANSION STRATEGY 4: ADD RISK DISCLAIMER
            # ================================================================
            
            if len(expanded_content) < min_length:
                logger.logger.debug(f"🔧 EXPANSION STRATEGY 4: Adding risk disclaimer")
                
                disclaimers = [
                    "Risk management essential.",
                    "DYOR before trading.",
                    "Not financial advice.",
                    "Monitor closely for changes."
                ]
                
                disclaimer = random.choice(disclaimers)
                addition = f"\n\n{disclaimer}"
                
                if len(expanded_content) + len(addition) <= max_length:
                    expanded_content += addition
                    logger.logger.debug(f"✅ Added disclaimer: +{len(addition)} chars")
            
            # ================================================================
            # 📊 FINAL EXPANSION CHECK
            # ================================================================
            
            final_length = len(expanded_content)
            expansion_amount = final_length - len(initial_content)
            
            logger.logger.info(f"✅ CONTENT EXPANSION COMPLETE:")
            logger.logger.debug(f"   • Original: {len(initial_content)} chars")
            logger.logger.debug(f"   • Final: {final_length} chars")
            logger.logger.debug(f"   • Added: +{expansion_amount} chars")
            logger.logger.debug(f"   • Target met: {final_length >= min_length}")
            
            return expanded_content
            
        except Exception as e:
            logger.logger.error(f"❌ CONTENT EXPANSION FAILED: {str(e)}")
            logger.logger.debug(f"🔄 Returning original content")
            return initial_content

    def _should_use_insider_style(self, confidence: float, sentiment: str, rationale: str) -> bool:
        """
        Determine whether to use Insider Tip Style or Professional Analysis Style
    
        Args:
            confidence: Prediction confidence percentage
            sentiment: Bullish/Bearish/Neutral
            rationale: Technical analysis rationale
        
        Returns:
            True for Insider Tip Style, False for Professional Analysis Style
        """
        try:
            # Use current time for style rotation (respects datetime handling)
            now = strip_timezone(datetime.now())
            hour = now.hour
        
            # High confidence strong signals = Insider Tip
            if confidence > 75 and sentiment in ["BULLISH", "BEARISH"]:
                return True
            
            # Strong technical signals = Insider Tip
            strong_signals = ["oversold", "overbought", "breakout", "reversal", "divergence"]
            if any(signal in rationale.lower() for signal in strong_signals) and confidence > 65:
                return True
            
            # Time-based rotation (evening hours favor insider style)
            if 18 <= hour <= 23 and confidence > 60:
                return True
            
            # Random element for variety (30% chance for medium confidence)
            if 55 <= confidence <= 75:
                # Use deterministic randomness based on hour and confidence
                seed_value = (hour + int(confidence)) % 10
                return seed_value < 3  # 30% chance
            
            return False
        
        except Exception:
            # Default to professional style on error
            return False

    def _format_insider_tip_style(self, token: str, sentiment: str, confidence: float,
                                price: float, percent_change: float, lower_bound: float,
                                upper_bound: float, timeframe: str, rationale: str,
                                key_factors: List[str]) -> str:
        """Format prediction in engaging insider tip style"""
    
        # Extract key signal from rationale
        key_signal = self._extract_key_technical_signal(rationale)
    
        # Confidence-based opening hooks
        if confidence > 80:
            openers = [
                f"Hot take on {token}: This {key_signal} is actually a gift.",
                f"Something interesting just caught my eye on {token}...",
                f"Counterintuitive take on {token} right now...",
                f"While everyone is distracted, {token} is setting up perfectly..."
            ]
        elif confidence > 65:
            openers = [
                f"The {token} chart is telling a story...",
                f"Worth noting what {token} is doing here...",
                f"{token} showing some compelling signals...",
                f"Interesting development on {token}..."
            ]
        else:
            openers = [
                f"Watching {token} closely here...",
                f"Early signals emerging on {token}...",
                f"{token} creating some interesting patterns..."
            ]
    
        # Sentiment-based analysis
        if sentiment == "BULLISH" and confidence > 75:
            analysis = f"The {key_signal} - and I mean really {key_signal.lower()}. This is the kind of setup that usually leads to a bounce. My target is sitting around ${price:.4f}, which would be a nice {percent_change:+.2f}% move from here."
        elif sentiment == "BULLISH":
            analysis = f"The technicals are suggesting {key_signal.lower()}. Im seeing a potential move to ${price:.4f} - thats {percent_change:+.2f}% upside over the next {timeframe}."
        elif sentiment == "BEARISH" and confidence > 75:
            analysis = f"{key_signal.title()} showing classic distribution patterns. This screams downside incoming. Im targeting ${price:.4f} - about {percent_change:+.2f}% from current levels."
        elif sentiment == "BEARISH":
            analysis = f"Technical breakdown forming with {key_signal.lower()}. Could see a move to ${price:.4f} ({percent_change:+.2f}%) if this plays out."
        else:
            analysis = f"Mixed signals with {key_signal.lower()}, but could be interesting. Potential move to ${price:.4f} ({percent_change:+.2f}%) over the next {timeframe}."
    
        # Confidence-based closers
        if sentiment == "BULLISH" and confidence > 70:
            closers = [
                f"Range Im watching: ${lower_bound:.4f} to ${upper_bound:.4f}. Smart money tends to accumulate in these exact conditions.",
                f"Risk range is tight: ${lower_bound:.4f} to ${upper_bound:.4f}. This is how reversals start - quietly, not with fanfare.",
                f"Key levels: ${lower_bound:.4f}-${upper_bound:.4f}. Sometimes the best plays happen when no one is looking."
            ]
        elif sentiment == "BEARISH" and confidence > 70:
            closers = [
                f"Risk parameters: ${lower_bound:.4f}-${upper_bound:.4f}. This is what topping patterns look like in real time.",
                f"Range: ${lower_bound:.4f} to ${upper_bound:.4f}. Distribution phases often start quietly like this.",
                f"Watching ${lower_bound:.4f}-${upper_bound:.4f} zone. Smart money exits before the crowd notices."
            ]
        else:
            closers = [
                f"Range: ${lower_bound:.4f}-${upper_bound:.4f}. Markets love to keep us guessing.",
                f"Key levels: ${lower_bound:.4f}-${upper_bound:.4f}. Patience pays in setups like this.",
                f"Watching ${lower_bound:.4f} to ${upper_bound:.4f}. Sometimes the best trade is no trade."
            ]
    
        # Combine sections
        opener = random.choice(openers)
        closer = random.choice(closers)
    
        return f"{opener}\n\n{analysis}\n\n{closer}"

    def _format_professional_analysis_style(self, token: str, sentiment: str, confidence: float,
                                           price: float, percent_change: float, lower_bound: float,
                                           upper_bound: float, timeframe: str, rationale: str,
                                           key_factors: List[str]) -> str:
        """Format prediction in professional analysis style (enhanced but familiar)"""
    
        # Professional header
        tweet = f"{token} {timeframe} ANALYSIS\n\n"
    
        # Enhanced sentiment display
        if sentiment == "BULLISH":
            sentiment_display = "BULLISH OUTLOOK"
        elif sentiment == "BEARISH":
            sentiment_display = "BEARISH OUTLOOK"
        else:
            sentiment_display = "MARKET ANALYSIS"
    
        tweet += f"{sentiment_display}\n"
    
        # Target and range (familiar format but cleaner)
        tweet += f"Target: ${price:.4f} ({percent_change:+.2f}%)\n"
        tweet += f"Range: ${lower_bound:.4f} - ${upper_bound:.4f}\n"
        tweet += f"Confidence: {confidence:.0f}%\n\n"
    
        # Enhanced rationale (more engaging than original)
        if rationale:
            # Clean up technical rationale to be more readable
            clean_rationale = self._enhance_technical_rationale(rationale)
            tweet += f"{clean_rationale}\n\n"
    
        # Key factors if available
        if key_factors and len(key_factors) > 0:
            key_factor = key_factors[0] if isinstance(key_factors[0], str) else str(key_factors[0])
            tweet += f"Key driver: {key_factor}"
    
        return tweet

    def _extract_key_technical_signal(self, rationale: str) -> str:
        """Extract the key technical signal from rationale text"""
        try:
            rationale_lower = rationale.lower()
        
            # Look for key technical terms
            if "oversold" in rationale_lower and "stochastic" in rationale_lower:
                return "oversold stochastic condition"
            elif "oversold" in rationale_lower:
                return "oversold condition"
            elif "overbought" in rationale_lower:
                return "overbought readings"
            elif "breakout" in rationale_lower:
                return "breakout pattern"
            elif "reversal" in rationale_lower and "bullish" in rationale_lower:
                return "bullish reversal signal"
            elif "reversal" in rationale_lower:
                return "reversal signal"
            elif "divergence" in rationale_lower:
                return "divergence setup"
            elif "support" in rationale_lower:
                return "support level test"
            elif "resistance" in rationale_lower:
                return "resistance challenge"
            elif "momentum" in rationale_lower:
                return "momentum shift"
            elif "macd" in rationale_lower:
                return "MACD signal"
            elif "rsi" in rationale_lower:
                return "RSI indication"
            else:
                return "technical setup"
        except Exception:
            return "technical signals"

    def _enhance_technical_rationale(self, rationale: str) -> str:
        """Make technical rationale more engaging while keeping it professional"""
        try:
            # Replace overly technical language with more engaging equivalents
            enhanced = rationale
        
            replacements = {
            # Original replacements
            "The oversold Stochastic Oscillator and potential bullish reversal suggest": 
                "Stochastic oscillator showing oversold conditions with bullish reversal patterns suggesting",
            "Technical analysis indicates": "The technicals are pointing to",
            "Based on current indicators": "Current market signals indicate",
            "Analysis suggests": "The data suggests",
            "Potential for": "Looking at potential for",
   
            # Expanded natural language transformations
            "The technical indicators show": "What I'm seeing in the charts is",
            "Market data reveals": "The numbers are telling me",
            "Price action demonstrates": "The way this thing is moving suggests",
            "Volume analysis indicates": "Volume is painting a picture of",
            "The charts suggest": "If these charts could talk, they'd be saying",
            "Pattern recognition shows": "The patterns I'm tracking are screaming",
            "Momentum indicators reveal": "Momentum-wise, we're looking at",
            "Support and resistance levels indicate": "Key levels are whispering",
   
            # Academic robot speak -> Human personality
            "Statistical analysis demonstrates": "Running the numbers, and here's what's wild:",
            "Quantitative data suggests": "When you crunch the data, it's pretty clear that",
            "Empirical evidence points to": "Real-world data is showing us",
            "Historical patterns indicate": "History has a funny way of rhyming, and right now it's saying",
            "Correlation analysis reveals": "Connecting the dots here, and",
            "Research indicates": "Digging into this, and",
            "Studies have shown": "Smart people discovered that",
            "The data supports": "Yeah, this is basically what I thought would happen",
   
            # Dry technical terms -> Vivid descriptions
            "Bollinger Bands are contracting": "Bollinger Bands squeezing tighter than a HODLer's grip",
            "RSI is approaching oversold territory": "RSI diving into 'maybe this is actually a steal' territory",
            "MACD is showing divergence": "MACD doing that thing where it disagrees with price action",
            "Moving averages are converging": "Moving averages having their little meet-cute moment",
            "Volume is increasing significantly": "Volume cranking up like someone just dropped alpha",
            "Price is testing resistance": "Price knocking on resistance's door like it owns the place",
            "Support levels are holding": "Support holding stronger than diamond hands in a crash",
            "Breakout pattern is forming": "This thing looks ready to absolutely send it",
   
            # Generic predictions -> Specific scenarios
            "Upward movement is likely": "This could leg up harder than expected",
            "Downward pressure is expected": "Gravity might start working on this one",
            "Sideways consolidation probable": "Might just crab around because crypto loves psychological torture",
            "Volatility is anticipated": "Buckle up, this might get spicy",
            "Price discovery is ongoing": "Market's still figuring out what this thing is actually worth",
            "Trend continuation expected": "Looks like this train isn't stopping anytime soon",
            "Reversal signals are present": "Plot twist energy is strong with this one",
            "Accumulation phase detected": "Smart money might be quietly loading bags",
   
            # Robotic conclusions -> Natural insights
            "This analysis concludes": "Bottom line:",
            "The findings indicate": "Here's what I'm thinking:",
            "Results demonstrate": "What we're seeing here is",
            "Evidence suggests": "All signs point to",
            "Assessment shows": "Real talk:",
            "Evaluation indicates": "If I had to call it:",
            "Investigation reveals": "Digging deeper, it's looking like",
            "The outcome shows": "Plot twist:",
   
            # Boring transitions -> Natural flow
            "Furthermore": "And here's the kicker:",
            "Additionally": "Also worth noting:",
            "Moreover": "On top of that:",
            "In conclusion": "So here's the deal:",
            "Therefore": "Which means:",
            "Consequently": "So naturally:",
            "As a result": "The upshot?",
            "Subsequently": "Then what happened was:",
   
            # Overly cautious hedging -> Confident uncertainty
            "May potentially indicate": "Could be signaling",
            "Appears to suggest": "Looking like it might be",
            "Preliminary data shows": "Early signs are pointing to",
            "Initial analysis reveals": "First pass through the data, and",
            "Tentative conclusions": "Reading between the lines here, but",
            "Conditional on market factors": "Assuming the market doesn't go full degen mode,",
            "Subject to further confirmation": "Still gathering intel, but",
            "Pending additional data": "Need to keep watching this one"
        }
        
            for old, new in replacements.items():
                enhanced = enhanced.replace(old, new)
        
            return enhanced
        except Exception:
            return rationale

    def _smart_truncate_prediction(self, text: str, max_length: int) -> str:
        """Intelligently truncate prediction tweet while preserving key information"""
        if len(text) <= max_length:
            return text
    
        # Try to preserve the structure: opener + analysis + closer
        sections = text.split('\n\n')
    
        if len(sections) >= 3:
            # Try to keep opener and analysis, truncate closer
            opener_and_analysis = f"{sections[0]}\n\n{sections[1]}"
            if len(opener_and_analysis) <= max_length - 50:  # Leave room for shorter closer
                remaining_space = max_length - len(opener_and_analysis) - 2
                truncated_closer = sections[2][:remaining_space] + "..."
                return f"{opener_and_analysis}\n\n{truncated_closer}"
    
        # Fallback: find last complete sentence
        truncate_point = max_length - 3
        last_period = text[:truncate_point].rfind('.')
        last_question = text[:truncate_point].rfind('?')
        last_exclamation = text[:truncate_point].rfind('!')
    
        best_break = max(last_period, last_question, last_exclamation)
    
        if best_break > max_length * 0.7:
            return text[:best_break + 1]
        else:
            last_space = text[:truncate_point].rfind(' ')
            if last_space > 0:
                return text[:last_space] + "..."
            else:
                return text[:truncate_point] + "..."

    @ensure_naive_datetimes
    def _track_prediction(self, token: str, prediction: Dict[str, Any], relevant_tokens: List[str], timeframe: str = "1h") -> None:
        """
        Track predictions for future callbacks and analysis
        Supports multiple timeframes (1h, 24h, 7d)
        
        Args:
            token: Token symbol
            prediction: Prediction data dictionary
            relevant_tokens: List of relevant token symbols
            timeframe: Timeframe for the prediction
        """
        MAX_PREDICTIONS = 20  
    
        # Get current prices of relevant tokens from prediction
        current_prices = {chain: prediction.get(f'{chain.upper()}_price', 0) for chain in relevant_tokens if f'{chain.upper()}_price' in prediction}
    
        # Add the prediction to the tracking list with timeframe info
        self.past_predictions.append({
            'timestamp': strip_timezone(datetime.now()),
            'token': token,
            'prediction': prediction['analysis'],
            'prices': current_prices,
            'sentiment': prediction['sentiment'],
            'timeframe': timeframe,
            'outcome': None
        })
    
        # Keep only predictions from the last 24 hours, up to MAX_PREDICTIONS
        self.past_predictions = [p for p in self.past_predictions 
                                 if safe_datetime_diff(datetime.now(), p['timestamp']) < 86400]
    
        # Trim to max predictions if needed
        if len(self.past_predictions) > MAX_PREDICTIONS:
            self.past_predictions = self.past_predictions[-MAX_PREDICTIONS:]
        
        logger.logger.debug(f"Tracked {timeframe} prediction for {token}")

    @ensure_naive_datetimes
    def _validate_past_prediction(self, prediction: Dict[str, Any], current_prices: Dict[str, float]) -> str:
        """
        Check if a past prediction was accurate
        
        Args:
            prediction: Prediction data dictionary
            current_prices: Dictionary of current prices
            
        Returns:
            Evaluation outcome: 'right', 'wrong', or 'undetermined'
        """
        sentiment_map = {
            'bullish': 1,
            'bearish': -1,
            'neutral': 0,
            'volatile': 0,
            'recovering': 0.5
        }
    
        # Apply different thresholds based on the timeframe
        timeframe = prediction.get('timeframe', '1h')
        if timeframe == '1h':
            threshold = 2.0  # 2% for 1-hour predictions
        elif timeframe == '24h':
            threshold = 4.0  # 4% for 24-hour predictions
        else:  # 7d
            threshold = 7.0  # 7% for 7-day predictions
    
        wrong_tokens = []
        for token, old_price in prediction['prices'].items():
            if token in current_prices and old_price > 0:
                price_change = ((current_prices[token] - old_price) / old_price) * 100
            
                # Get sentiment for this token
                token_sentiment_key = token.upper() if token.upper() in prediction['sentiment'] else token
                token_sentiment_value = prediction['sentiment'].get(token_sentiment_key)
            
                # Handle nested dictionary structure
                if isinstance(token_sentiment_value, dict) and 'mood' in token_sentiment_value:
                    token_sentiment = sentiment_map.get(token_sentiment_value['mood'], 0)
                else:
                    token_sentiment = sentiment_map.get(str(token_sentiment_value), 0.0)  # Convert key to string
            
                # A prediction is wrong if:
                # 1. Bullish but price dropped more than threshold%
                # 2. Bearish but price rose more than threshold%
                if (token_sentiment > 0 and price_change < -threshold) or (token_sentiment < 0 and price_change > threshold):
                    wrong_tokens.append(token)
    
        return 'wrong' if wrong_tokens else 'right'
    
    @ensure_naive_datetimes
    def _get_spicy_callback(self, token: str, current_prices: Dict[str, float], timeframe: str = "1h") -> Optional[str]:
        """
        Generate witty callbacks to past terrible predictions
        Supports multiple timeframes
        
        Args:
            token: Token symbol
            current_prices: Dictionary of current prices
            timeframe: Timeframe for the callback
            
        Returns:
            Callback text or None if no suitable callback found
        """
        # Look for the most recent prediction for this token and timeframe
        recent_predictions = [p for p in self.past_predictions 
                             if safe_datetime_diff(datetime.now(), p['timestamp']) < 24*3600
                             and p['token'] == token
                             and p.get('timeframe', '1h') == timeframe]
    
        if not recent_predictions:
            return None
        
        # Evaluate any unvalidated predictions
        for pred in recent_predictions:
            if pred['outcome'] is None:
                pred['outcome'] = self._validate_past_prediction(pred, current_prices)
            
        # Find any wrong predictions
        wrong_predictions = [p for p in recent_predictions if p['outcome'] == 'wrong']
        if wrong_predictions:
            worst_pred = wrong_predictions[-1]
            time_ago = int(safe_datetime_diff(datetime.now(), worst_pred['timestamp']) / 3600)
        
            # If time_ago is 0, set it to 1 to avoid awkward phrasing
            if time_ago == 0:
                time_ago = 1
        
            # Format timeframe for display
            time_unit = "hr" if timeframe in ["1h", "24h"] else "day"
            time_display = f"{time_ago}{time_unit}"
        
            # Token-specific callbacks with natural language variety
            callbacks = [
                # Self-deprecating humor
                f"(Unlike my galaxy-brain take {time_display} ago about {worst_pred['prediction'].split('.')[0]}... this time I'm sure!)",
                f"(Looks like my {time_display} old prediction about {token} aged like milk. But trust me bro!)",
                f"(That awkward moment when your {time_display} old {token} analysis was completely wrong... but this one's different!)",
                f"(My {token} trading bot would be down bad after that {time_display} old take. Good thing I'm just an analyst!)",
                f"(Excuse the {time_display} old miss on {token}. Even the best crypto analysts are wrong sometimes... just not usually THIS wrong!)",
   
                # Humble admissions with humor
                f"(Remember when I said {token} would moon {time_display} ago? Pepperidge Farm remembers... and so does my portfolio.)",
                f"(That {time_display} old {token} call? Yeah, we don't talk about that one. Moving on!)",
                f"(Plot twist: My {time_display} old {token} prediction was actually financial advice... for doing the opposite.)",
                f"(Update: Still recovering from my {time_display} old {token} take. Therapist says I'm making progress.)",
                f"(Fun fact: My {time_display} old {token} analysis is now used as a case study in 'How Not to Trade 101'.)",
   
                # Conversational and relatable
                f"(Yeah, about that {token} call {time_display} ago... we're just gonna pretend that never happened, okay?)",
                f"(My {time_display} old {token} prediction hit different... like a truck. But hey, this one feels right!)",
                f"(Shoutout to everyone who inversed my {token} call {time_display} ago - you're the real MVPs.)",
                f"(That {time_display} old {token} take was my villain origin story. This is my redemption arc.)",
                f"(Note to self: Maybe don't bet the house on my {token} predictions. See: {time_display} ago.)",
   
                # Self-aware and witty
                f"(Breaking: Local analyst still confident despite {token} disaster {time_display} ago. More at 11.)",
                f"(My {time_display} old {token} prediction aged about as well as a banana in the sun. But this one's different!)",
                f"(Friendly reminder that my {time_display} old {token} call is why we can't have nice things.)",
                f"(That {time_display} old {token} analysis? Consider it my contribution to the 'Worst Takes Ever' hall of fame.)",
                f"(My {time_display} old {token} prediction was so bad, even my crystal ball filed a complaint.)",
   
                # Crypto-culture specific
                f"(My {time_display} old {token} call was peak 'this is fine' energy. Narrator: It was not fine.)",
                f"(That {token} prediction {time_display} ago? Pure hopium mixed with copium. This time it's different... right?)",
                f"(My {time_display} old {token} take was so wrong, it broke the simulation. But we're back!)",
                f"(Remember my {token} moon mission {time_display} ago? Turns out it was more of a submarine expedition.)",
                f"(Plot armor didn't save my {time_display} old {token} prediction. Maybe diamond hands will save this one.)",
   
                # Brutally honest
                f"(Let's be real - my {time_display} old {token} call was about as accurate as a weather forecast. But hear me out...)",
                f"(That {token} prediction {time_display} ago was my 'hold my beer' moment. Spoiler: I should've kept the beer.)",
                f"(My {time_display} old {token} analysis was so cursed, it probably caused the dip. Sorry, not sorry.)",
                f"(Historical fact: My {token} call {time_display} ago single-handedly proved markets can stay irrational longer than I can stay solvent.)",
                f"(That {time_display} old {token} take was my audition for 'Crypto Predictions Gone Wrong'. I got the part.)",
   
                # Meta-humor about being wrong
                f"(Update on my {time_display} old {token} prediction: Still wrong, but now with 20% more confidence!)",
                f"(My {token} analysis {time_display} ago was like a broken clock - wrong twice a day, every day.)",
                f"(That {time_display} old {token} call was my masterpiece in the art of being spectacularly incorrect.)",
                f"(My {time_display} old {token} prediction was so bad, it got its own Wikipedia page under 'Notable Market Failures'.)",
                f"(Fun update: My {token} call from {time_display} ago is still aging poorly. Like, really poorly.)",
   
                # Redemption arc vibes
                f"(Character development: Learning from my {time_display} old {token} disaster to bring you this slightly less terrible take.)",
                f"(Sequel to my {time_display} old {token} flop: This time it's personal (and hopefully accurate).)",
                f"(Phoenix rising from the ashes of my {time_display} old {token} prediction. This is my comeback story.)",
                f"(My {time_display} old {token} call was the tutorial level of being wrong. This is the real game.)",
                f"(Lessons learned from my {token} debacle {time_display} ago: Humility is a virtue, and markets are ruthless teachers.)"
            ]
        
            # Select a callback deterministically but with variation
            callback_seed = f"{datetime.now().date()}_{token}_{timeframe}"
            callback_index = hash(callback_seed) % len(callbacks)
        
            return callbacks[callback_index]
        
        return None

    @ensure_naive_datetimes
    def _format_tweet_analysis(self, token: str, analysis: str, market_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Enhanced analysis formatter with dual style options
        Transforms dry technical analysis into engaging insights
        Backward compatible with existing analysis structure
        """
        try:
            # Get token data for context
            token_data = market_data.get(token, {})
            current_price = token_data.get('current_price', 0)
            price_change_24h = token_data.get('price_change_percentage_24h', 0)
        
            # Determine style based on market conditions and timeframe
            use_insider_style = self._should_use_insider_analysis_style(token, analysis, price_change_24h, timeframe)
        
            if use_insider_style:
                formatted_analysis = self._format_insider_analysis_style(token, analysis, current_price, price_change_24h, timeframe)
            else:
                formatted_analysis = self._format_professional_tweet_style(token, analysis, current_price, price_change_24h, timeframe)
        
            # Ensure proper length constraints
            max_length = config.TWEET_CONSTRAINTS['MAX_LENGTH']
            if len(formatted_analysis) > max_length:
                formatted_analysis = self._smart_truncate_analysis(formatted_analysis, max_length)
        
            # Check minimum length
            min_length = config.TWEET_CONSTRAINTS['MIN_LENGTH']
            if len(formatted_analysis) < min_length:
                logger.logger.warning(f"{timeframe} analysis too short ({len(formatted_analysis)} chars)")
        
            return formatted_analysis
        
        except Exception as e:
            logger.log_error(f"Format Analysis Tweet - {token} ({timeframe})", str(e))
            # Fallback to enhanced version of original
            return self._format_analysis_fallback(token, analysis, timeframe)

    def _should_use_insider_analysis_style(self, token: str, analysis: str, price_change: float, timeframe: str) -> bool:
        """Determine whether to use insider or professional analysis style"""
        try:
            # Strong price movements favor insider style
            if abs(price_change) > 5.0:
                return True
            
            # Key technical signals favor insider style
            insider_signals = [
                "breakout", "reversal", "divergence", "squeeze", 
                "oversold", "overbought", "momentum", "support", "resistance"
            ]
        
            if any(signal in analysis.lower() for signal in insider_signals):
                return True
            
            # Time-based preference (evening hours favor insider style)
            now = strip_timezone(datetime.now())
            if 17 <= now.hour <= 23:
                return True
            
            # Random element for variety (40% chance for regular analysis)
            seed_value = (now.hour + len(token)) % 10
            return seed_value < 4
        
        except Exception:
            return False

    def _format_insider_analysis_style(self, token: str, analysis: str, current_price: float, 
                                     price_change: float, timeframe: str) -> str:
        """Format analysis in engaging insider style"""
    
        # Extract key insights from technical analysis
        key_insight = self._extract_key_insight_from_analysis(analysis)
        market_condition = self._determine_market_condition(analysis)
    
        # Dynamic opening hooks
        if abs(price_change) > 3.0:
            openers = [
                f"Something worth noting on {token} right now...",
                f"Interesting development on {token}...",
                f"The {token} chart is telling a story...",
                f"Worth paying attention to what {token} is doing..."
            ]
        else:
            openers = [
                f"While everyone is focused elsewhere, {token} is quietly setting up...",
                f"{token} creating an interesting pattern here...",
                f"Subtle but important move happening on {token}...",
                f"{token} showing some signals worth watching..."
            ]
    
        # Enhanced middle section based on market condition
        if market_condition == "bullish":
            middles = [
                f"The technicals are looking promising - {key_insight.lower()}. This kind of setup usually leads to upside. Im seeing potential for ${current_price * 1.02:.4f}, maybe {price_change + 2:.1f}% from here.",
                f"{key_insight} creating some bullish undertones. My read: this could push toward ${current_price * 1.015:.4f} over the next {timeframe}. The setup screams accumulation."
            ]
        elif market_condition == "bearish":
            middles = [
                f"The technicals are showing weakness - {key_insight.lower()}. This looks like distribution territory. Could see a drop to ${current_price * 0.98:.4f}, about {price_change - 2:.1f}% from current levels.",
                f"{key_insight} suggesting some downside pressure. My read: controlled pullback to ${current_price * 0.985:.4f} over the next {timeframe}."
            ]
        else:
            middles = [
                f"Mixed signals across the board - {key_insight.lower()}. Yeah, the signals are mixed, but thats actually interesting. Could see movement in either direction.",
                f"The technicals are in neutral territory with {key_insight.lower()}. Sometimes uncertainty creates the best opportunities."
            ]
    
        # Engaging closers
        closers = [
            "Sometimes the best setups come from these quiet periods.",
            "Worth watching closely - these patterns often precede bigger moves.",
            "High conviction despite the mixed signals. Markets love to surprise.",
            "Smart money tends to position during uncertainty like this."
        ]
    
        # Combine sections
        opener = random.choice(openers)
        middle = random.choice(middles)
        closer = random.choice(closers)
    
        return f"{opener}\n\n{middle}\n\n{closer}"

    def _format_professional_tweet_style(self, token: str, analysis: str, current_price: float,
                                           price_change: float, timeframe: str) -> str:
        """Format analysis in enhanced professional style"""
    
        # Enhanced but professional opening
        if timeframe == "1h":
            header = f"{token} HOURLY INSIGHT"
        elif timeframe == "24h":
            header = f"{token} DAILY OUTLOOK"
        else:
            header = f"{token} WEEKLY ANALYSIS"
    
        # Extract and enhance the technical content
        enhanced_analysis = self._enhance_technical_language(analysis)
    
        # Add current market context
        if abs(price_change) > 2.0:
            direction = "upward momentum" if price_change > 0 else "downward pressure"
            context = f"Current price action showing {direction} at ${current_price:.4f} ({price_change:+.2f}% 24h)."
        else:
            context = f"Price consolidating around ${current_price:.4f} with minimal 24h movement ({price_change:+.2f}%)."
    
        # Professional but engaging conclusion
        key_takeaway = self._extract_actionable_takeaway(analysis)
    
        return f"{header}\n\n{context}\n\n{enhanced_analysis}\n\n{key_takeaway}"

    def _extract_key_insight_from_analysis(self, analysis: str) -> str:
        """Extract the most important technical insight from analysis text"""
        try:
            analysis_lower = analysis.lower()
        
            # Look for key technical patterns
            if "bollinger band" in analysis_lower and "squeeze" in analysis_lower:
                return "Bollinger Bands squeezing tight"
            elif "oversold" in analysis_lower:
                return "oversold conditions emerging"
            elif "overbought" in analysis_lower:
                return "overbought territory detected"
            elif "breakout" in analysis_lower:
                return "breakout pattern developing"
            elif "reversal" in analysis_lower:
                return "reversal signals appearing"
            elif "support" in analysis_lower:
                return "support level testing"
            elif "resistance" in analysis_lower:
                return "resistance challenge"
            elif "momentum" in analysis_lower:
                return "momentum shift detected"
            elif "divergence" in analysis_lower:
                return "divergence pattern forming"
            elif "mixed" in analysis_lower:
                return "mixed technical signals"
            else:
                return "technical patterns emerging"
        except Exception:
            return "market signals developing"

    def _determine_market_condition(self, analysis: str) -> str:
        """Determine overall market condition from analysis"""
        try:
            analysis_lower = analysis.lower()
        
            bullish_terms = ["bullish", "upward", "positive", "bounce", "reversal", "support"]
            bearish_terms = ["bearish", "downward", "negative", "drop", "decline", "resistance"]
        
            bullish_count = sum(1 for term in bullish_terms if term in analysis_lower)
            bearish_count = sum(1 for term in bearish_terms if term in analysis_lower)
        
            if bullish_count > bearish_count:
                return "bullish"
            elif bearish_count > bullish_count:
                return "bearish"
            else:
                return "neutral"
        except Exception:
            return "neutral"

    def _enhance_technical_language(self, analysis: str) -> str:
        """Transform dry technical language into more engaging analysis"""
        try:
            enhanced = analysis
        
            # Replace boring technical phrases with engaging equivalents
            replacements = {
                "The short-term trend is moderately bearish": "Immediate setup showing some weakness",
                "technical indicators showing mixed signals": "the technicals are sending mixed messages - but thats actually interesting",
                "potential for a small price movement": "something is about to give",
                "Bollinger Band squeeze suggests": "Bollinger Bands squeezing tight, which usually means",
                "moderately bearish": "showing some bearish undertones",
                "moderately bullish": "displaying bullish characteristics", 
                "Analysis indicates": "My read is",
                "Technical analysis suggests": "The technicals are suggesting",
                "Based on current indicators": "Based on what Im seeing",
                "The data shows": "The charts are telling me"
            }
        
            for old_phrase, new_phrase in replacements.items():
                enhanced = enhanced.replace(old_phrase, new_phrase)
        
            # Fix incomplete sentences
            if enhanced.endswith("within"):
                enhanced = enhanced.replace("within", "within the trading range")
        
            return enhanced
        
        except Exception:
            return analysis

    def _extract_actionable_takeaway(self, analysis: str) -> str:
        """Extract or create an actionable takeaway from the analysis"""
        try:
            analysis_lower = analysis.lower()
        
            if "squeeze" in analysis_lower:
                return "Key takeaway: Compressed volatility often precedes significant moves. Stay alert."
            elif "oversold" in analysis_lower:
                return "Key takeaway: Oversold conditions creating potential bounce opportunities."
            elif "overbought" in analysis_lower:
                return "Key takeaway: Overbought readings suggest caution and potential pullback."
            elif "mixed" in analysis_lower:
                return "Key takeaway: Mixed signals require patience. Wait for clearer direction."
            elif "support" in analysis_lower:
                return "Key takeaway: Support level tests create important inflection points."
            elif "resistance" in analysis_lower:
                return "Key takeaway: Resistance challenges often determine next major move."
            else:
                return "Key takeaway: Technical setup warrants close monitoring for opportunities."
        except Exception:
            return "Technical analysis suggests staying attentive to market developments."

    def _smart_truncate_analysis(self, text: str, max_length: int) -> str:
        """Intelligently truncate analysis while preserving key insights"""
        if len(text) <= max_length:
            return text
    
        # Try to preserve structure: opener + analysis + conclusion
        sections = text.split('\n\n')
    
        if len(sections) >= 3:
            # Keep opener and main analysis, truncate conclusion if needed
            core_content = f"{sections[0]}\n\n{sections[1]}"
            if len(core_content) <= max_length - 50:
                remaining_space = max_length - len(core_content) - 2
                truncated_conclusion = sections[2][:remaining_space] + "..."
                return f"{core_content}\n\n{truncated_conclusion}"
    
        # Fallback to sentence-based truncation
        truncate_point = max_length - 3
        last_period = text[:truncate_point].rfind('.')
    
        if last_period > max_length * 0.7:
            return text[:last_period + 1]
        else:
            last_space = text[:truncate_point].rfind(' ')
            return text[:last_space] + "..." if last_space > 0 else text[:truncate_point] + "..."

    def _format_analysis_fallback(self, token: str, analysis: str, timeframe: str) -> str:
        """Fallback analysis format if enhanced formatting fails"""
        try:
            # Simple enhanced version that's guaranteed to work
            enhanced_analysis = analysis.replace(
                "The short-term trend is moderately bearish", 
                "Technical setup showing some weakness"
            ).replace(
                "technical indicators showing mixed signals",
                "mixed signals across the technicals"
            )
        
            # Add timeframe context
            if timeframe == "1h":
                return f"{token} hourly update: {enhanced_analysis}"
            elif timeframe == "24h":
                return f"{token} daily analysis: {enhanced_analysis}"
            else:
                return f"{token} weekly outlook: {enhanced_analysis}"
            
        except Exception:
            return f"{token} {timeframe} analysis: {analysis}"

    def _get_vs_market_analysis(self, token: str, market_data, timeframe: str = "1h"):
        """
        Analyze token performance against overall market
        Returns metrics showing relative performance
        """
        try:
            # Default response if anything fails
            default_response = {
                "vs_market_avg_change": 0.0,
                "vs_market_percentile": 50.0,
                "market_correlation": 0.0,
                "market_sentiment": "neutral"
            }
    
            if isinstance(market_data, list):
                logger.logger.error(f"CRITICAL: market_data is a list with {len(market_data)} items")
                if len(market_data) > 0:
                    first_item = market_data[0]
                    logger.logger.error(f"First item type: {type(first_item)}")
                    if isinstance(first_item, dict):
                        logger.logger.error(f"First item keys: {list(first_item.keys())}")

            # Validate and standardize market_data
            if not isinstance(market_data, dict):
                logger.logger.warning(f"_get_vs_market_analysis received non-dict market_data: {type(market_data)}")
                # Try to standardize
                if isinstance(market_data, list):
                    market_data = self._standardize_market_data(market_data)
                else:
                    return default_response
        
            # If standardization failed or returned empty data
            if not market_data:
                logger.logger.warning(f"Failed to standardize market data for {token}")
                return default_response
             
            # Now safely access token data
            token_data = market_data.get(token, {}) if isinstance(market_data, dict) else {}
        
            # If we couldn't standardize the data, return default response
            if not market_data:
                logger.logger.warning(f"Failed to standardize market data for {token}")
                return default_response
        
            # Get all tokens except the one we're analyzing
            market_tokens = [t for t in market_data.keys() if t != token]
        
            # Validate we have other tokens to compare against
            if not market_tokens:
                logger.logger.warning(f"No other tokens found for comparison with {token}")
                return default_response
        
            # Calculate average market metrics
            market_changes = []
            market_volumes = []
        
            for market_token in market_tokens:
                token_data = market_data.get(market_token, {})
            
                # Check if token_data is a dictionary before using get()
                if not isinstance(token_data, dict):
                    continue
            
                # Extract change data safely based on timeframe
                if timeframe == "1h":
                    change_key = 'price_change_percentage_1h_in_currency'
                elif timeframe == "24h":
                    change_key = 'price_change_percentage_24h'
                else:  # 7d
                    change_key = 'price_change_percentage_7d_in_currency'
            
                # Try alternate keys if the primary key isn't found
                if change_key not in token_data:
                    alternates = {
                        'price_change_percentage_1h_in_currency': ['price_change_1h', 'change_1h', '1h_change'],
                        'price_change_percentage_24h': ['price_change_24h', 'change_24h', '24h_change'],
                        'price_change_percentage_7d_in_currency': ['price_change_7d', 'change_7d', '7d_change']
                    }
                
                    for alt_key in alternates.get(change_key, []):
                        if alt_key in token_data:
                            change_key = alt_key
                            break
            
                # Safely extract change value
                change = token_data.get(change_key)
                if change is not None:
                    try:
                        change_float = float(change)
                        market_changes.append(change_float)
                    except (ValueError, TypeError):
                        # Skip invalid values
                        pass
            
                # Extract volume data safely
                volume_keys = ['total_volume', 'volume', 'volume_24h']
                for volume_key in volume_keys:
                    volume = token_data.get(volume_key)
                    if volume is not None:
                        try:
                            volume_float = float(volume)
                            market_volumes.append(volume_float)
                            break  # Found a valid volume, no need to check other keys
                        except (ValueError, TypeError):
                            # Skip invalid values
                            pass
        
            # If we don't have enough market data, return default analysis
            if not market_changes:
                logger.logger.warning(f"No market change data available for comparison with {token}")
                return default_response
        
            # Calculate average market change
            market_avg_change = sum(market_changes) / len(market_changes)
        
            # Get token data safely
            token_data = market_data.get(token, {})
        
            # Ensure token_data is a dictionary
            if not isinstance(token_data, dict):
                logger.logger.warning(f"Token data for {token} is not a dictionary: {token_data}")
                return default_response
        
            # Get token change based on timeframe
            token_change = 0.0
            if timeframe == "1h":
                # Try primary key first, then alternates
                keys_to_try = ['price_change_percentage_1h_in_currency', 'price_change_1h', 'change_1h', '1h_change']
            elif timeframe == "24h":
                keys_to_try = ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']
            else:  # 7d
                keys_to_try = ['price_change_percentage_7d_in_currency', 'price_change_7d', 'change_7d', '7d_change']
        
            # Try each key until we find a valid value
            for key in keys_to_try:
                if key in token_data:
                    try:
                        token_change = float(token_data[key])
                        break  # Found valid value, exit loop
                    except (ValueError, TypeError):
                        continue  # Try next key
        
            # Calculate performance vs market
            vs_market_change = token_change - market_avg_change
        
            # Calculate token's percentile in market (what percentage of tokens it's outperforming)
            tokens_outperforming = sum(1 for change in market_changes if token_change > change)
            vs_market_percentile = (tokens_outperforming / len(market_changes)) * 100
        
            # Calculate market correlation (simple approach)
            market_correlation = 0.5  # Default to moderate correlation
        
            # Determine market sentiment
            if vs_market_change > 3.0:
                market_sentiment = "strongly outperforming"
            elif vs_market_change > 1.0:
                market_sentiment = "outperforming"
            elif vs_market_change < -3.0:
                market_sentiment = "strongly underperforming"
            elif vs_market_change < -1.0:
                market_sentiment = "underperforming"
            else:
                market_sentiment = "neutral"
        
            # Return analysis
            return {
                "vs_market_avg_change": vs_market_change,
                "vs_market_percentile": vs_market_percentile,
                "market_correlation": market_correlation,
                "market_sentiment": market_sentiment
            }
    
        except Exception as e:
            logger.log_error(f"Token vs Market Analysis - {token} ({timeframe})", str(e))
            return {
                "vs_market_avg_change": 0.0,
                "vs_market_percentile": 50.0,
                "market_correlation": 0.0,
                "market_sentiment": "neutral"
            }

    @ensure_naive_datetimes
    def _analyze_market_sentiment(self, token, market_data, trigger_type=None, timeframe="1h"):
        """
        Analyze market sentiment for a token with enhanced error handling,
        compatibility with _get_vs_market_analysis, and trading format support

        Args:
            token: Token symbol
            market_data: Market data dictionary/list (supports trading-enhanced format)
            trigger_type: The type of trigger that initiated this analysis (optional)
            timeframe: Timeframe for analysis ("1h", "24h", or "7d")

        Returns:
            Tuple of (sentiment, market_context)
        """
        import traceback
        from datetime import datetime, timedelta

        # Initialize defaults
        sentiment = "neutral"
        market_context = ""

        try:
            # ================================================================
            # 🔧 NORMALIZE TRADING FORMAT FOR MARKET ANALYSIS
            # ================================================================
            
            # Check if market_data contains trading-enhanced predictions
            normalized_market_data = market_data
            if isinstance(market_data, dict):
                # Check if any token data contains trading fields
                trading_fields_detected = False
                for token_key, token_data in market_data.items():
                    if isinstance(token_data, dict) and any(field in token_data for field in ['action', 'entry_price', 'stop_loss', 'take_profit']):
                        trading_fields_detected = True
                        break
                
                if trading_fields_detected:
                    logger.logger.debug(f"🔄 Converting trading-enhanced market data for {token} sentiment analysis")
                    
                    # Create normalized copy for sentiment analysis
                    normalized_market_data = {}
                    for token_key, token_data in market_data.items():
                        if isinstance(token_data, dict):
                            normalized_token_data = token_data.copy()
                            
                            # Convert trading fields to analysis fields
                            if 'entry_price' in token_data and 'current_price' not in normalized_token_data:
                                normalized_token_data['current_price'] = token_data['entry_price']
                            
                            # Convert trading price to more conservative analysis price
                            if 'take_profit' in token_data and 'entry_price' in token_data:
                                entry_price = token_data['entry_price']
                                take_profit = token_data['take_profit']
                                # Use 60% of the way to take_profit for sentiment analysis
                                analysis_price = entry_price + ((take_profit - entry_price) * 0.6)
                                normalized_token_data['price'] = analysis_price
                            
                            normalized_market_data[token_key] = normalized_token_data
                        else:
                            normalized_market_data[token_key] = token_data
            
            # Log trigger_type for context if available
            if trigger_type:
                logger.logger.debug(f"Market sentiment analysis for {token} triggered by: {trigger_type}")
            
            # ================================================================
            # 📊 GET VS MARKET ANALYSIS WITH DEFENSIVE ERROR HANDLING
            # ================================================================
            
            try:
                # Initialize vs_market with default values
                vs_market = {
                    "vs_market_avg_change": 0.0,
                    "vs_market_percentile": 50.0,
                    "market_correlation": 0.0,
                    "market_sentiment": "neutral"
                }
        
                # Ensure market_data is standardized if possible
                standardized_data = None
                if hasattr(self, '_standardize_market_data'):
                    try:
                        standardized_data = self._standardize_market_data(normalized_market_data)
                        logger.logger.debug(f"Standardized market data type: {type(standardized_data)}")
                    except Exception as std_error:
                        logger.logger.error(f"Error in standardizing market data for {token}: {str(std_error)}")
                        standardized_data = None
        
                # Only proceed with _get_vs_market_analysis if we have dictionary data
                if standardized_data and isinstance(standardized_data, dict):
                    # Use standardized data
                    if hasattr(self, '_get_vs_market_analysis'):
                        vs_market = self._get_vs_market_analysis(token, standardized_data, timeframe)
                    elif hasattr(self, '_analyze_token_vs_market'):
                        # Fallback to newer method if available
                        vs_market_result = self._analyze_token_vs_market(token, standardized_data, timeframe)
                        # Extract relevant keys for compatibility
                        vs_market = {
                            "vs_market_avg_change": vs_market_result.get("vs_market_avg_change", 0.0),
                            "vs_market_percentile": vs_market_result.get("vs_market_percentile", 50.0),
                            "market_correlation": vs_market_result.get("market_correlation", 0.0),
                            "market_sentiment": vs_market_result.get("market_sentiment", "neutral")
                        }
                elif isinstance(normalized_market_data, dict):
                    # If standardization failed but original data is a dict, use it
                    if hasattr(self, '_get_vs_market_analysis'):
                        vs_market = self._get_vs_market_analysis(token, normalized_market_data, timeframe)
                    elif hasattr(self, '_analyze_token_vs_market'):
                        # Fallback to newer method if available
                        vs_market_result = self._analyze_token_vs_market(token, normalized_market_data, timeframe)
                        # Extract relevant keys for compatibility
                        vs_market = {
                            "vs_market_avg_change": vs_market_result.get("vs_market_avg_change", 0.0),
                            "vs_market_percentile": vs_market_result.get("vs_market_percentile", 50.0),
                            "market_correlation": vs_market_result.get("market_correlation", 0.0),
                            "market_sentiment": vs_market_result.get("market_sentiment", "neutral")
                        }
                else:
                    # Handle the case where market_data is a list and standardization failed
                    logger.logger.warning(f"Cannot analyze market for {token}: market_data is {type(normalized_market_data)} and standardization failed")
            
                    # Create an emergency backup dictionary if needed
                    if isinstance(normalized_market_data, list):
                        temp_dict = {}
                        for item in normalized_market_data:
                            if isinstance(item, dict) and 'symbol' in item:
                                symbol = item['symbol'].upper()
                                temp_dict[symbol] = item
                
                        # If we have a reasonable dictionary now, try to use it
                        if temp_dict:
                            if hasattr(self, '_get_vs_market_analysis'):
                                logger.logger.debug(f"Using emergency backup dictionary for {token} with {len(temp_dict)} items")
                                try:
                                    vs_market = self._get_vs_market_analysis(token, temp_dict, timeframe)
                                except Exception as emergency_error:
                                    logger.logger.error(f"Error in emergency market analysis for {token}: {str(emergency_error)}")
                            elif hasattr(self, '_analyze_token_vs_market'):
                                logger.logger.debug(f"Using emergency backup dictionary with newer method for {token}")
                                try:
                                    vs_market_result = self._analyze_token_vs_market(token, temp_dict, timeframe)
                                    # Extract relevant keys for compatibility
                                    vs_market = {
                                        "vs_market_avg_change": vs_market_result.get("vs_market_avg_change", 0.0),
                                        "vs_market_percentile": vs_market_result.get("vs_market_percentile", 50.0),
                                        "market_correlation": vs_market_result.get("market_correlation", 0.0),
                                        "market_sentiment": vs_market_result.get("market_sentiment", "neutral")
                                    }
                                except Exception as emergency_error:
                                    logger.logger.error(f"Error in emergency market analysis with newer method for {token}: {str(emergency_error)}")
                                
                # Validate result is a dictionary with required keys
                if not isinstance(vs_market, dict):
                    logger.logger.warning(f"VS market analysis returned non-dict for {token}: {type(vs_market)}")
                    vs_market = {
                        "vs_market_avg_change": 0.0,
                        "vs_market_percentile": 50.0,
                        "market_correlation": 0.0,
                        "market_sentiment": "neutral"
                    }
                elif 'vs_market_avg_change' not in vs_market:
                    logger.logger.warning(f"Missing 'vs_market_avg_change' for {token}")
                    vs_market['vs_market_avg_change'] = 0.0
        
                if 'vs_market_sentiment' not in vs_market and 'market_sentiment' in vs_market:
                    # Copy market_sentiment to vs_market_sentiment for compatibility
                    vs_market['vs_market_sentiment'] = vs_market['market_sentiment']
                elif 'vs_market_sentiment' not in vs_market:
                    vs_market['vs_market_sentiment'] = "neutral"
            
            except Exception as vs_error:
                logger.logger.error(f"Error in market analysis for {token}: {str(vs_error)}")
                logger.logger.debug(traceback.format_exc())
                vs_market = {
                    "vs_market_avg_change": 0.0,
                    "vs_market_percentile": 50.0,
                    "market_correlation": 0.0,
                    "market_sentiment": "neutral"
                }

            # ================================================================
            # 📈 GET OVERALL MARKET TREND
            # ================================================================
            
            try:
                market_conditions = {}
                if hasattr(self, 'market_conditions'):
                    market_conditions = self.market_conditions

                market_trend = market_conditions.get('market_trend', 'neutral')
            except Exception as trend_error:
                logger.logger.error(f"Error getting market trend for {token}: {str(trend_error)}")
                market_trend = 'neutral'

            # ================================================================
            # ⏰ HANDLE DATETIME OPERATIONS
            # ================================================================
            
            try:
                # Get current time and appropriate time window based on timeframe
                current_time = datetime.now()

                if timeframe == "1h":
                    time_window = timedelta(hours=1)
                    time_desc = "hourly"
                elif timeframe == "24h":
                    time_window = timedelta(hours=24)
                    time_desc = "daily"
                else:  # 7d
                    time_window = timedelta(days=7)
                    time_desc = "weekly"
        
                # Calculate prediction target time (when this prediction will be evaluated)
                target_time = current_time + time_window

                # Format times appropriately for displaying in context
                target_time_str = target_time.strftime("%H:%M") if timeframe == "1h" else target_time.strftime("%b %d")

                # ================================================================
                # 💹 GET HISTORICAL VOLATILITY
                # ================================================================

                volatility = 0.0
                token_data = None

                # Try multiple ways to get token data (prioritize normalized data)
                if isinstance(normalized_market_data, dict) and token in normalized_market_data:
                    token_data = normalized_market_data[token]
                elif isinstance(normalized_market_data, list):
                    # Try to find token in the list
                    for item in normalized_market_data:
                        if isinstance(item, dict) and item.get('symbol', '').upper() == token:
                            token_data = item
                            break

                # Extract volatility if token_data is available
                if isinstance(token_data, dict) and 'volatility' in token_data:
                    volatility = token_data['volatility']

            except Exception as time_error:
                logger.logger.error(f"Error processing time data for {token}: {str(time_error)}")
                time_desc = timeframe
                target_time_str = "upcoming " + timeframe
                volatility = 0.0

            # ================================================================
            # 🎯 ANALYZE SENTIMENT BASED ON AVAILABLE DATA
            # ================================================================
            
            try:
                # Determine sentiment based on market performance
                vs_sentiment = vs_market.get('vs_market_sentiment', 'neutral')
                vs_change = vs_market.get('vs_market_avg_change', 0.0)

                if vs_sentiment in ['strongly outperforming', 'outperforming']:
                    sentiment = "bullish"
                    market_context = f"\n{token} outperforming market average by {abs(vs_change):.1f}%"
                elif vs_sentiment in ['strongly underperforming', 'underperforming']:
                    sentiment = "bearish"
                    market_context = f"\n{token} underperforming market average by {abs(vs_change):.1f}%"
                else:
                    # Neutral market performance
                    sentiment = "neutral"
                    market_context = f"\n{token} performing close to market average"
        
                # Add market trend context
                if market_trend in ['strongly bullish', 'bullish']:
                    market_context += f"\nOverall market trend: bullish"
                    # In bullish market, amplify token's performance
                    if sentiment == "bullish":
                        sentiment = "strongly bullish"
                elif market_trend in ['strongly bearish', 'bearish']:
                    market_context += f"\nOverall market trend: bearish"
                    # In bearish market, amplify token's performance
                    if sentiment == "bearish":
                        sentiment = "strongly bearish"
                else:
                    market_context += f"\nOverall market trend: neutral"
        
                # Add time-based context
                market_context += f"\nAnalysis for {time_desc} timeframe (until {target_time_str})"

                # Add volatility context if available
                if volatility > 0:
                    market_context += f"\nCurrent volatility: {volatility:.1f}%"
        
                    # Adjust sentiment based on volatility
                    if volatility > 10 and sentiment in ["bullish", "bearish"]:
                        sentiment = f"volatile {sentiment}"
                    
                # ================================================================
                # 🚀 INCORPORATE TRIGGER TYPE INTO ANALYSIS
                # ================================================================
                
                if trigger_type:
                    # Adjust sentiment based on trigger type
                    if 'price_change' in trigger_type:
                        market_context += f"\nTriggered by significant price movement"
                    elif 'volume_change' in trigger_type or 'volume_trend' in trigger_type:
                        market_context += f"\nTriggered by notable volume activity"
                    elif 'smart_money' in trigger_type:
                        market_context += f"\nTriggered by smart money indicators"
                        # Emphasize sentiment for smart money triggers
                        if sentiment in ["bullish", "bearish"]:
                            sentiment = f"smart money {sentiment}"
                    elif 'prediction' in trigger_type:
                        market_context += f"\nBased on predictive model analysis"

            except Exception as analysis_error:
                logger.logger.error(f"Error in sentiment analysis for {token}: {str(analysis_error)}")
                logger.logger.debug(traceback.format_exc())
                sentiment = "neutral"
                market_context = f"\n{token} market sentiment analysis unavailable"

            # ================================================================
            # 🕒 ENSURE TIMEZONE HANDLING
            # ================================================================
            
            try:
                # For any datetime objects that need timezone handling
                if 'strip_timezone' in globals() and callable(strip_timezone):
                    # Apply timezone handling to relevant datetime objects
                    local_vars = locals()
        
                    # Handle 'current_time' if it exists and is a datetime
                    if 'current_time' in local_vars and isinstance(local_vars.get('current_time'), datetime):
                        current_time = strip_timezone(local_vars.get('current_time'))
            
                    # Handle 'target_time' if it exists and is a datetime
                    if 'target_time' in local_vars and isinstance(local_vars.get('target_time'), datetime):
                        target_time = strip_timezone(local_vars.get('target_time'))
            
                # Check if we have the decorator function available
                if 'ensure_naive_datetimes' not in globals() or not callable(ensure_naive_datetimes):
                    logger.logger.debug("ensure_naive_datetimes decorator not available")
        
            except Exception as tz_error:
                logger.logger.error(f"Error handling timezone data for {token}: {str(tz_error)}")

            # ================================================================
            # 💾 PREPARE STORAGE DATA FOR DATABASE
            # ================================================================
            
            storage_data = {
                'content': None,  # Will be filled in by the caller
                'sentiment': {token: sentiment},
                'trigger_type': trigger_type if trigger_type else "regular_interval",
                'timeframe': timeframe,
                'market_context': market_context.strip(),
                'vs_market_change': vs_market.get('vs_market_avg_change', 0),
                'market_sentiment': vs_market.get('market_sentiment', 'neutral'),
                'timestamp': strip_timezone(datetime.now()) if hasattr(self, 'strip_timezone') else datetime.now()
            }

            return sentiment, storage_data

        except Exception as e:
            # Catch-all exception handler
            logger.logger.error(f"Error in _analyze_market_sentiment for {token} ({timeframe}): {str(e)}")
            logger.logger.debug(traceback.format_exc())
            logger.logger.error(f"{timeframe} analysis error details: {str(e)}")
        
            # Return minimal valid data to prevent downstream errors
            default_storage_data = {
                'content': None,
                'sentiment': {token: "neutral"},
                'trigger_type': "error_recovery",
                'timeframe': timeframe,
                'market_context': "",
                'timestamp': strip_timezone(datetime.now()) if hasattr(self, 'strip_timezone') else datetime.now()
            }
            return "neutral", default_storage_data

    @ensure_naive_datetimes
    def _should_post_update(self, token: str, new_data: Dict[str, Any], timeframe: str = "1h") -> Tuple[bool, str]:
        """
        Determine if we should post an update based on market changes for a specific timeframe
        
        Args:
            token: Token symbol
            new_data: Latest market data dictionary
            timeframe: Timeframe for the analysis
            
        Returns:
            Tuple of (should_post, trigger_reason)
        """
        if not self.last_market_data:
            self.last_market_data = new_data
            return True, f"initial_post_{timeframe}"

        trigger_reason = None

        # Check token for significant changes
        if token in new_data and token in self.last_market_data:
            # Get timeframe-specific thresholds
            thresholds = self.timeframe_thresholds.get(timeframe, self.timeframe_thresholds["1h"])
        
            # Calculate immediate price change since last check
            price_change = abs(
                (new_data[token]['current_price'] - self.last_market_data[token]['current_price']) /
                self.last_market_data[token]['current_price'] * 100
            )
        
            # Calculate immediate volume change since last check
            current_volume = self._safe_get_volume(new_data[token])
            last_volume = self._safe_get_volume(self.last_market_data[token])
            immediate_volume_change = abs(
                (current_volume - last_volume) / last_volume * 100
            ) if last_volume > 0 else 0.0

            logger.logger.debug(
                f"{token} immediate changes ({timeframe}) - "
                f"Price: {price_change:.2f}%, Volume: {immediate_volume_change:.2f}%"
            )

            # Check immediate price change against timeframe threshold
            price_threshold = thresholds["price_change"]
            if price_change >= price_threshold:
                trigger_reason = f"price_change_{token.lower()}_{timeframe}"
                logger.logger.info(
                    f"Significant price change detected for {token} ({timeframe}): "
                    f"{price_change:.2f}% (threshold: {price_threshold}%)"
                )
            # Check immediate volume change against timeframe threshold
            else:
                volume_threshold = thresholds["volume_change"]
                if immediate_volume_change >= volume_threshold:
                    trigger_reason = f"volume_change_{token.lower()}_{timeframe}"
                    logger.logger.info(
                        f"Significant immediate volume change detected for {token} ({timeframe}): "
                        f"{immediate_volume_change:.2f}% (threshold: {volume_threshold}%)"
                )
                # Check rolling window volume trend
                else:
                    # Initialize variables with safe defaults BEFORE any conditional code
                    volume_change_pct = 0.0  # Default value
                    trend = "unknown"        # Default value

                    # Get historical volume data
                    historical_volume = self._get_historical_volume_data(token, timeframe=timeframe)

                    # Then try to get actual values if we have historical data
                    if historical_volume:
                        try:
                            current_volume = self._safe_get_volume(new_data[token])
                            volume_change_pct, trend = self._analyze_volume_trend(
                            current_volume,
                            historical_volume,
                            timeframe=timeframe
                        )
                        except Exception as e:
                            # If analysis fails, we already have defaults
                            logger.logger.debug(f"Error analyzing volume trend: {str(e)}")
                    
                        # Log the volume trend - ensure all variables are defined
                        volume_change_pct = 0.0 if 'volume_change_pct' not in locals() else volume_change_pct
                        trend = "unknown" if 'trend' not in locals() else trend 
                        timeframe = "1h" if 'timeframe' not in locals() else timeframe

                    # Log the volume trend
                    logger.logger.debug(
                        f"{token} rolling window volume trend ({timeframe}): {volume_change_pct:.2f}% ({trend})"
                        )

                    # Check if trend is significant enough to trigger
                    if trend in ["significant_increase", "significant_decrease"]:
                        trigger_reason = f"volume_trend_{token.lower()}_{trend}_{timeframe}"
                        logger.logger.info(
                            f"Significant volume trend detected for {token} ({timeframe}): "
                            f"{volume_change_pct:.2f}% - {trend}"
                        )
        
            # Check for smart money indicators
            if not trigger_reason:
                smart_money = self._analyze_smart_money_indicators(token, new_data[token], timeframe=timeframe)
                if smart_money.get('abnormal_volume') or smart_money.get('stealth_accumulation'):
                    trigger_reason = f"smart_money_{token.lower()}_{timeframe}"
                    logger.logger.info(f"Smart money movement detected for {token} ({timeframe})")
                
                # Check for pattern metrics in longer timeframes
                elif timeframe in ["24h", "7d"] and 'pattern_metrics' in smart_money:
                    pattern_metrics = smart_money['pattern_metrics']
                    if pattern_metrics.get('volume_breakout', False) or pattern_metrics.get('consistent_high_volume', False):
                        trigger_reason = f"pattern_metrics_{token.lower()}_{timeframe}"
                        logger.logger.info(f"Advanced pattern metrics detected for {token} ({timeframe})")
        
            # Check for significant outperformance vs market
            if not trigger_reason:
                vs_market = self._analyze_token_vs_market(token, new_data, timeframe=timeframe)
                outperformance_threshold = 3.0 if timeframe == "1h" else 5.0 if timeframe == "24h" else 8.0
            
                if vs_market.get('outperforming_market') and abs(vs_market.get('vs_market_avg_change', 0)) > outperformance_threshold:
                    trigger_reason = f"{token.lower()}_outperforming_market_{timeframe}"
                    logger.logger.info(f"{token} significantly outperforming market ({timeframe})")
                
                # Check if we need to post prediction update
                # Trigger prediction post based on time since last prediction
                if not trigger_reason:
                    # Check when the last prediction was posted
                    last_prediction = self.db.get_active_predictions(token=token, timeframe=timeframe)
                    if not last_prediction:
                        # No recent predictions for this timeframe, should post one
                        trigger_reason = f"prediction_needed_{token.lower()}_{timeframe}"
                        logger.logger.info(f"No recent {timeframe} prediction for {token}, triggering prediction post")

        # Check if regular interval has passed (only for 1h timeframe)
        if not trigger_reason and timeframe == "1h":
            time_since_last = safe_datetime_diff(datetime.now(), self.last_check_time)
            if time_since_last >= config.BASE_INTERVAL:
                trigger_reason = f"regular_interval_{timeframe}"
                logger.logger.debug(f"Regular interval check triggered for {timeframe}")

        should_post = trigger_reason is not None
        if should_post:
            self.last_market_data = new_data
            logger.logger.info(f"Update triggered by: {trigger_reason}")
        else:
            logger.logger.debug(f"No {timeframe} triggers activated for {token}, skipping update")

        return should_post, trigger_reason if trigger_reason is not None else ""

    @ensure_naive_datetimes
    def _validate_trigger_worthiness(self, token: str, market_data: Dict[str, Any], 
                                    trigger_type: str, timeframe: str = "1h") -> Tuple[bool, str]:
        """
        M4-optimized trigger validation with Polars vectorization
        """
        try:
            # Check if we can use M4 optimizations
            if len(market_data) > 5 and POLARS_AVAILABLE and pl is not None:
                # Convert market data to Polars DataFrame for vectorized operations
                market_records = []
                for sym, data in market_data.items():
                    if isinstance(data, dict):
                        market_records.append({
                            'symbol': sym,
                            'price_change_24h': float(data.get('price_change_percentage_24h', 0)),
                            'volume': float(data.get('volume', data.get('total_volume', 0))),
                            'market_cap': float(data.get('market_cap', 0))
                        })
           
                if market_records:
                    df = pl.DataFrame(market_records)
               
                    # Vectorized threshold checks
                    significance_thresholds = {
                        "1h": {"min_price_change": 1.5, "min_volume_change": 5.0},
                        "24h": {"min_price_change": 3.0, "min_volume_change": 10.0}, 
                        "7d": {"min_price_change": 5.0, "min_volume_change": 15.0}
                    }
                    thresholds = significance_thresholds.get(timeframe, significance_thresholds["1h"])
               
                    # Filter for our token
                    token_row = df.filter(pl.col('symbol') == token)
                    if token_row.height == 0:
                        return False, "token_not_found_in_data"
               
                    token_data_row = token_row.row(0, named=True)
               
                    # Vectorized price check
                    price_change = abs(token_data_row['price_change_24h'])
                    if price_change < thresholds["min_price_change"]:
                        return False, f"price_change_insufficient_{price_change:.1f}%"
               
                    # Market context check - is this token moving independently?
                    market_avg_change = df.select(pl.col('price_change_24h').abs().mean()).item()
                    if abs(price_change - market_avg_change) < 1.0:
                        return False, f"too_correlated_with_market_{market_avg_change:.1f}%"
               
                    # Volume significance check using existing method
                    historical_volume = self._get_historical_volume_data(token, timeframe=timeframe)
                    if historical_volume:
                        token_data = market_data.get(token, {})
                        volume_change, trend = self._analyze_volume_trend(token_data.get('volume', 0), historical_volume, timeframe)
                        if abs(volume_change) < thresholds["min_volume_change"] and trend not in ["significant_increase", "significant_decrease"]:
                            return False, f"volume_change_insufficient_{volume_change:.1f}%"
               
                    return True, "trigger_validated_m4"
       
            # Fallback to original implementation for small datasets
            significance_thresholds = {
                "1h": {"min_price_change": 1.5, "min_volume_change": 5.0},
                "24h": {"min_price_change": 3.0, "min_volume_change": 10.0}, 
                "7d": {"min_price_change": 5.0, "min_volume_change": 15.0}
            }
       
            thresholds = significance_thresholds.get(timeframe, significance_thresholds["1h"])
            token_data = market_data.get(token, {})
       
            # Price significance check
            price_change = abs(token_data.get('price_change_percentage_24h', 0))
            if price_change < thresholds["min_price_change"]:
                return False, f"price_change_insufficient_{price_change:.1f}%"
       
            # Volume significance check
            historical_volume = self._get_historical_volume_data(token, timeframe=timeframe)
            if historical_volume:
                volume_change, trend = self._analyze_volume_trend(token_data.get('volume', 0), historical_volume, timeframe)
                if abs(volume_change) < thresholds["min_volume_change"] and trend not in ["significant_increase", "significant_decrease"]:
                    return False, f"volume_change_insufficient_{volume_change:.1f}%"
       
            return True, "trigger_validated"
       
        except Exception as e:
            logger.log_error(f"Trigger Validation - {token} ({timeframe})", str(e))
            return False, f"validation_error_{str(e)[:50]}"

    @ensure_naive_datetimes
    def _validate_content_quality(self, content: str, token: str, timeframe: str) -> Tuple[bool, str]:
        """
        Enhanced content quality validation with smart fixing instead of rejection
        
        Validates content quality and automatically fixes issues when possible, rather than
        just rejecting content. Maintains exact same method signature for compatibility.
        
        Args:
            content (str): Content to validate and potentially fix
            token (str): Token symbol for context-specific fixes
            timeframe (str): Timeframe for targeted improvements
        
        Returns:
            Tuple[bool, str]: (is_valid, fixed_content_or_reason)
                            - True, fixed_content: Content is valid or was successfully fixed
                            - False, reason: Content cannot be salvaged, reason for rejection
        """
        # ================================================================
        # 🔍 PRE-VALIDATION DEBUG LOGGING
        # ================================================================
        
        logger.logger.debug(f"🔍 ENHANCED CONTENT VALIDATION START: {token} ({timeframe})")
        logger.logger.debug(f"📊 INPUT ANALYSIS:")
        logger.logger.debug(f"   • Content length: {len(content) if content else 0} chars")
        logger.logger.debug(f"   • Content type: {type(content)}")
        logger.logger.debug(f"   • Token: '{token}'")
        logger.logger.debug(f"   • Timeframe: '{timeframe}'")
        logger.logger.debug(f"   • Preview: '{content[:100] if content else 'None'}...'")
        
        try:
            # ================================================================
            # 🛡️ BASIC CONTENT EXISTENCE AND LENGTH VALIDATION
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 1: Basic content validation")
            
            # Check for completely empty or null content
            if not content or not content.strip():
                logger.logger.warning(f"❌ VALIDATION FAILED: Empty or null content")
                logger.logger.debug(f"🔍 Content state: content={content}, stripped='{content.strip() if content else 'None'}'")
                return False, "content_empty_or_null"
            
            # Initial length check
            original_length = len(content.strip())
            logger.logger.debug(f"📏 Original content length: {original_length} chars")
            
            # If content is too short, try to fix it instead of rejecting
            if original_length < 5:
                logger.logger.warning(f"⚠️ CONTENT TOO SHORT: {original_length} < 5 chars")
                logger.logger.debug(f"🔧 ATTEMPTING SMART CONTENT EXPANSION")
                
                # Try to expand the content intelligently
                expanded_content = self._smart_expand_short_content(content, token, timeframe)
                
                if len(expanded_content.strip()) >= 275:
                    logger.logger.info(f"✅ CONTENT EXPANSION SUCCESS:")
                    logger.logger.debug(f"   • Original: {original_length} chars")
                    logger.logger.debug(f"   • Expanded: {len(expanded_content)} chars")
                    logger.logger.debug(f"   • Added: +{len(expanded_content) - original_length} chars")
                    content = expanded_content
                else:
                    logger.logger.error(f"❌ EXPANSION FAILED: Still too short ({len(expanded_content)} chars)")
                    return False, f"content_too_short_unfixable_{original_length}_chars"
            
            # ================================================================
            # 🚫 FALLBACK PHRASE DETECTION AND REPLACEMENT
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 2: Fallback phrase detection and fixing")
            
            # Check for the exact fallback phrases we want to eliminate
            fallback_phrases = [
                "big things coming", "to the moon", "hodl strong", "bullish af",
                "buy the dip", "this is the way", "diamond hands", "breaking out",
                "getting ready to pump", "solid fundamentals", "undervalued gem",
                "next 100x potential", "the future is now", "early adopters win",
                "massive potential", "smart money flowing", "not financial advice",
                "lightning in a bottle", "paper hands ngmi", "lfg"
            ]
            
            content_lower = content.lower()
            detected_fallbacks = []
            
            for phrase in fallback_phrases:
                if phrase in content_lower:
                    detected_fallbacks.append(phrase)
            
            if detected_fallbacks:
                logger.logger.warning(f"⚠️ DETECTED FALLBACK PHRASES: {detected_fallbacks}")
                logger.logger.debug(f"🔧 ATTEMPTING FALLBACK PHRASE REPLACEMENT")
                
                # Try to replace fallback phrases with meaningful content
                improved_content = self._replace_fallback_phrases(content, detected_fallbacks, token, timeframe)
                
                if improved_content != content:
                    logger.logger.info(f"✅ FALLBACK REPLACEMENT SUCCESS:")
                    logger.logger.debug(f"   • Replaced phrases: {len(detected_fallbacks)}")
                    logger.logger.debug(f"   • Original: '{content[:60]}...'")
                    logger.logger.debug(f"   • Improved: '{improved_content[:60]}...'")
                    content = improved_content
                else:
                    logger.logger.warning(f"⚠️ FALLBACK REPLACEMENT FAILED: Content unchanged")
                    # Don't reject yet, continue with other fixes
            
            # ================================================================
            # 🔄 GENERIC CONTENT DETECTION AND ENHANCEMENT
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 3: Generic content detection and enhancement")
            
            # Check for generic/repetitive content
            generic_phrases = [
                "market analysis", "technical indicators", "price movement", 
                "trading volume", "market sentiment", "showing mixed signals",
                "moderately bearish", "moderately bullish", "neutral"
            ]
            
            content_lower = content.lower()
            generic_count = sum(1 for phrase in generic_phrases if phrase in content_lower)
            detected_generic = [phrase for phrase in generic_phrases if phrase in content_lower]
            
            logger.logger.debug(f"📊 GENERIC CONTENT ANALYSIS:")
            logger.logger.debug(f"   • Generic phrases found: {generic_count}")
            logger.logger.debug(f"   • Detected phrases: {detected_generic}")
            
            if generic_count >= 3:
                logger.logger.warning(f"⚠️ TOO MANY GENERIC PHRASES: {generic_count} >= 3")
                logger.logger.debug(f"🔧 ATTEMPTING GENERIC CONTENT ENHANCEMENT")
                
                # Try to make content more specific and engaging
                enhanced_content = self._enhance_generic_content(content, token, timeframe, detected_generic)
                
                if enhanced_content != content:
                    logger.logger.info(f"✅ GENERIC ENHANCEMENT SUCCESS:")
                    logger.logger.debug(f"   • Enhanced phrases: {len(detected_generic)}")
                    logger.logger.debug(f"   • Specificity improved: Yes")
                    content = enhanced_content
                else:
                    logger.logger.warning(f"⚠️ GENERIC ENHANCEMENT FAILED: Content unchanged")
                    # Continue with validation, don't reject yet
            
            # ================================================================
            # 📈 TECHNICAL INSIGHT VALIDATION AND ADDITION
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 4: Technical insight validation and enhancement")
            
            # Check for meaningful insights
            insight_indicators = [
                # Original terms
                "breakout", "support", "resistance", "divergence", "accumulation",
                "distribution", "oversold", "overbought", "correlation", "trend",
                "squeeze", "reversal", "momentum", "volume spike", "unusual activity",
                "bull flag", "bear flag", "triangle", "wedge", "consolidation",
                
                # Technical analysis patterns and concepts
                "head and shoulders", "double top", "double bottom", "cup and handle", "falling wedge",
                "rising wedge", "ascending triangle", "descending triangle", "symmetrical triangle", "pennant",
                "flag", "channel", "trendline", "supply zone", "demand zone", "liquidity zone", "order block",
                "fibonacci", "extension", "retracement", "golden ratio", "pivot point", "time cycle", "harmonic pattern",
                "gartley", "butterfly", "bat pattern", "crab pattern", "abcd pattern", "three drives", "elliot wave",
                "wyckoff", "wyckoff accumulation", "wyckoff distribution", "market structure", "higher high", "higher low",
                "lower high", "lower low", "double bottom", "double top", "triple bottom", "triple top", "rounded bottom",
                "rounded top", "saucer", "island reversal", "gap up", "gap down", "parabolic", "blow-off top", "v-bottom",
                "inverted v-top", "capitulation", "consolidation phase", "accumulation phase", "distribution phase", "markup phase",
                "markdown phase", "re-accumulation", "re-distribution", "fakeout", "shakeout", "stop hunt", "liquidity grab",
                
                # Technical indicators
                "moving average", "simple moving average", "exponential moving average", "weighted moving average", "ema", "sma", "wma",
                "macd", "rsi", "relative strength index", "stochastic", "stochastic rsi", "cci", "commodity channel index",
                "bollinger bands", "keltner channel", "atr", "average true range", "adx", "average directional index", "dmi",
                "directional movement index", "parabolic sar", "ichimoku cloud", "ichimoku", "tenkan", "kijun", "senkou span",
                "chikou span", "vwap", "volume weighted average price", "volume profile", "obv", "on-balance volume",
                "money flow index", "mfi", "chaikin money flow", "cmf", "awesome oscillator", "ao", "momentum oscillator",
                "williams %r", "trix", "ultimate oscillator", "stochastic oscillator", "slow stochastic", "fast stochastic",
                "roc", "rate of change", "ppo", "percentage price oscillator", "choppiness index", "supertrend", "heikin ashi",
                "donchian channel", "elder-ray", "force index", "hull moving average", "kama", "kaufman adaptive moving average",
                "psar", "zigzag", "vortex indicator", "elder impulse system", "aroon", "aroon oscillator", "dpo", "detrended price oscillator",
                "standard deviation", "linear regression", "atr bands", "starc bands", "fractal", "alligator indicator", "gator oscillator",
                "market facilitation index", "acceleration bands", "andrew's pitchfork", "renko", "point and figure", "kagi",
                
                # Volume and market dynamics
                "volume", "volume profile", "volume analysis", "relative volume", "volume climax", "volume divergence",
                "buying pressure", "selling pressure", "absorption", "stopping volume", "effort vs result", "volume spread analysis",
                "vsa", "accumulation volume", "distribution volume", "churn", "price rejection", "volume delta", "bid-ask spread",
                "market depth", "order book", "liquidity pool", "smart money", "retail", "institutional", "whale", "whale wallet",
                "whale activity", "market maker", "market participant", "volume spike", "volume expansion", "volume contraction",
                "high volume node", "low volume node", "point of control", "value area", "market profile", "time at price",
                "footprint chart", "volume profile fixed range", "vpfr", "volume profile visible range", "vpvr", "delta volume",
                "absorption volume", "open interest", "oi", "open interest analysis", "aggregated volume", "cumulative volume",
                "volume zone oscillator", "volume imbalance", "institutional order flow", "dark pool", "block trade", "tape reading",
                
                # Price action concepts
                "price action", "candlestick", "bar pattern", "candle pattern", "engulfing", "doji", "hammer", "shooting star",
                "morning star", "evening star", "harami", "piercing line", "dark cloud cover", "three white soldiers", "three black crows",
                "spinning top", "marubozu", "dragonfly doji", "gravestone doji", "long-legged doji", "tweezer top", "tweezer bottom",
                "inside bar", "outside bar", "pin bar", "rejection", "indecision", "momentum candle", "thrusting pattern", "countertrend",
                "with trend", "continuation", "reversal pattern", "breakaway gap", "exhaustion gap", "price discovery", "range bound",
                "range expansion", "range contraction", "mean reversion", "trend following", "counter-trend", "overbought condition",
                "oversold condition", "bullish divergence", "bearish divergence", "hidden divergence", "regular divergence", "exaggerated move",
                "fair value gap", "imbalance", "inefficiency", "premium", "discount", "absorption", "high of day", "low of day",
                
                # Market cycles and conditions
                "market cycle", "bull market", "bear market", "sideways market", "ranging market", "trending market", "volatility",
                "implied volatility", "historical volatility", "volatility contraction", "volatility expansion", "low volatility",
                "high volatility", "fear", "greed", "sentiment", "market sentiment", "euphoria", "despair", "optimism", "pessimism",
                "capitulation phase", "distribution phase", "markup phase", "markdown phase", "disbelief", "complacency", "anxiety",
                "denial", "return to normal", "mean reversion", "seasonality", "market regime", "risk on", "risk off", "risk premium",
                "risk appetite", "risk aversion", "uncertainty", "conviction", "confidence", "fear index", "volatility index", "vix",
                "extreme value", "climax", "exhaustion", "liquidity crisis", "panic selling", "fomo", "fear of missing out",
                
                # Crypto-specific terms
                "blockchain", "distributed ledger", "consensus", "protocol", "token", "coin", "tokenomics", "utility token", "security token",
                "governance token", "stablecoin", "defi", "decentralized finance", "yield farming", "staking", "mining", "hash rate",
                "difficulty adjustment", "block reward", "halving", "fork", "hard fork", "soft fork", "node", "validator", "masternode",
                "smart contract", "dapp", "decentralized application", "dao", "decentralized autonomous organization", "nft", "non-fungible token",
                "layer 1", "layer 2", "scaling", "sharding", "rollup", "sidechain", "cross-chain", "interoperability", "bridge", "oracle",
                "wallet", "cold storage", "hot wallet", "hardware wallet", "private key", "public key", "seed phrase", "gas fee",
                "gas price", "gas limit", "transaction fee", "mempool", "confirmation", "block", "block time", "block height",
                "merkle tree", "hash function", "cryptography", "encryption", "proof of work", "pow", "proof of stake", "pos",
                "delegated proof of stake", "dpos", "proof of authority", "proof of history", "byzantine fault tolerance", "network effect",
                "adoption", "user base", "tokenization", "digital asset", "virtual currency", "cryptoeconomics", "game theory",
                "incentive mechanism", "economic bandwidth", "liquidity mining", "yield", "apy", "apr", "impermanent loss", "slippage",
                "total value locked", "tvl", "market cap", "fully diluted valuation", "fdv", "maximum supply", "circulating supply",
                "token burn", "token unlock", "vesting", "ico", "initial coin offering", "ido", "initial dex offering", "ieo",
                "initial exchange offering", "launchpad", "presale", "private sale", "public sale", "token generation event", "tge",
                "token distribution", "token utility", "token velocity", "token sink", "deflationary", "inflationary", "token economics",
                "token supply", "token demand", "token issuance", "token allocation", "token metrics", "token model", "utility",
                "network security", "network congestion", "network latency", "transaction throughput", "tps", "transactions per second",
                "block size", "scalability", "finality", "censorship resistance", "trustless", "permissionless", "composability",
                "interoperability", "whale address", "development activity", "github commits", "network upgrade", "roadmap",
                "whitepaper", "tokenomics", "team background", "partnership", "integration", "adoption", "use case", "real world application",
                "institutional adoption", "retail adoption", "regulation", "compliance", "legal framework", "kyc", "aml", "custody solution",
                "price discovery", "price equilibrium", "price action", "market inefficiency", "arbitrage", "cross-exchange", "listing",
                "delisting", "exchange volume", "trading pair", "market depth", "onchain metrics", "onchain analysis", "technical signal",
                "fundamental analysis", "chain analysis", "active addresses", "unique addresses", "transaction count", "value transferred",
                "realized value", "network value", "nvt ratio", "stock to flow", "metaverse", "web3", "dex", "amm", "automated market maker",
                "liquidity provider", "lp", "yield aggregator", "lending protocol", "borrowing", "collateral", "collateralization ratio"
            ]
            
            insight_count = sum(1 for indicator in insight_indicators if indicator.lower() in content_lower)
            detected_insights = [indicator for indicator in insight_indicators if indicator.lower() in content_lower]
            
            logger.logger.debug(f"📈 TECHNICAL INSIGHT ANALYSIS:")
            logger.logger.debug(f"   • Technical terms found: {insight_count}")
            logger.logger.debug(f"   • Detected insights: {detected_insights}")
            
            if insight_count == 0:
                logger.logger.warning(f"⚠️ LACKS TECHNICAL INSIGHTS: No technical terms found")
                logger.logger.debug(f"🔧 ATTEMPTING TECHNICAL INSIGHT ADDITION")
                
                # Try to add relevant technical insights
                enriched_content = self._add_technical_insights(content, token, timeframe)
                
                if enriched_content != content:
                    # Re-check for insights after enrichment
                    new_insight_count = sum(1 for indicator in insight_indicators if indicator.lower() in enriched_content.lower())
                    
                    if new_insight_count > 0:
                        logger.logger.info(f"✅ TECHNICAL ENRICHMENT SUCCESS:")
                        logger.logger.debug(f"   • Added insights: {new_insight_count}")
                        logger.logger.debug(f"   • Technical depth improved: Yes")
                        content = enriched_content
                    else:
                        logger.logger.warning(f"⚠️ TECHNICAL ENRICHMENT INEFFECTIVE: Still no insights")
                else:
                    logger.logger.warning(f"⚠️ TECHNICAL ENRICHMENT FAILED: Content unchanged")
            
            # ================================================================
            # 📏 FINAL LENGTH AND QUALITY VALIDATION
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 5: Final validation checks")
            
            final_length = len(content.strip())
            final_content_lower = content.lower()
            
            # Re-check technical insights after all improvements
            final_insight_count = sum(1 for indicator in insight_indicators if indicator.lower() in final_content_lower)
            
            # Re-check generic content after improvements
            final_generic_count = sum(1 for phrase in generic_phrases if phrase in final_content_lower)
            
            # Re-check fallback phrases after improvements
            final_fallback_count = sum(1 for phrase in fallback_phrases if phrase in final_content_lower)
            
            logger.logger.debug(f"📊 FINAL CONTENT ANALYSIS:")
            logger.logger.debug(f"   • Final length: {final_length} chars")
            logger.logger.debug(f"   • Technical insights: {final_insight_count}")
            logger.logger.debug(f"   • Generic phrases: {final_generic_count}")
            logger.logger.debug(f"   • Fallback phrases: {final_fallback_count}")
            
            # ================================================================
            # ✅ FINAL VALIDATION DECISION
            # ================================================================
            
            logger.logger.debug(f"🔍 STEP 6: Making final validation decision")
            
            # Check if content still fails after all fixes
            validation_issues = []
            
            if final_length < 50:
                validation_issues.append(f"length_too_short_{final_length}")
            
            if final_fallback_count > 0:
                validation_issues.append(f"contains_fallback_phrases_{final_fallback_count}")
            
            if final_generic_count >= 3:
                validation_issues.append(f"too_generic_{final_generic_count}_phrases")
            
            if final_insight_count == 0:
                validation_issues.append("lacks_technical_insights")
            
            # Make final decision
            if validation_issues:
                logger.logger.error(f"❌ FINAL VALIDATION FAILED:")
                for issue in validation_issues:
                    logger.logger.error(f"   • Issue: {issue}")
                
                primary_issue = validation_issues[0]  # Use the first/most critical issue
                logger.logger.error(f"💥 CONTENT REJECTED: {primary_issue}")
                return False, primary_issue
            else:
                logger.logger.info(f"✅ FINAL VALIDATION SUCCESS:")
                logger.logger.debug(f"   • All quality checks passed")
                logger.logger.debug(f"   • Content ready for posting")
                logger.logger.debug(f"   • Final preview: '{content[:80]}...'")
                return True, content
        
        except Exception as e:
            # ================================================================
            # ❌ COMPREHENSIVE ERROR HANDLING
            # ================================================================
            
            error_msg = str(e)
            logger.logger.error(f"❌ VALIDATION ERROR: {token} ({timeframe})")
            logger.logger.error(f"🔍 ERROR DETAILS:")
            logger.logger.error(f"   • Exception type: {type(e).__name__}")
            logger.logger.error(f"   • Error message: {error_msg}")
            logger.logger.error(f"   • Content length: {len(content) if content else 0}")
            logger.logger.error(f"   • Content preview: '{content[:50] if content else 'None'}...'")
            
            logger.log_error(f"Content Quality Validation - {token} ({timeframe})", str(e))
            return False, f"validation_error_{str(e)[:50]}"


    def _smart_expand_short_content(self, content: str, token: str, timeframe: str) -> str:
        """
        Intelligently expand short content with relevant market context
        
        Args:
            content (str): Original short content
            token (str): Token symbol for context
            timeframe (str): Timeframe for appropriate context
        
        Returns:
            str: Expanded content with additional relevant information
        """
        logger.logger.debug(f"🔧 SMART EXPANSION: {token} ({timeframe})")
        
        try:
            expanded = content.strip()
            
            # ================================================================
            # 📊 EXPANSION STRATEGY 1: ADD TIMEFRAME CONTEXT
            # ================================================================
            
            timeframe_contexts = {
                "1h": [
                    "Short-term price action suggests",
                    "Hourly momentum indicates",
                    "Near-term technical setup shows",
                    "Immediate price levels point to"
                ],
                "24h": [
                    "Daily analysis reveals",
                    "24-hour chart pattern suggests",
                    "Intraday momentum shows",
                    "Daily technical structure indicates"
                ],
                "7d": [
                    "Weekly outlook suggests",
                    "Medium-term trend analysis shows",
                    "7-day pattern indicates",
                    "Weekly technical setup reveals"
                ]
            }
            
            contexts = timeframe_contexts.get(timeframe, timeframe_contexts["1h"])
            context_addition = f" {random.choice(contexts)} continued development."
            
            if len(expanded + context_addition) <= 280:  # Twitter limit
                expanded += context_addition
                logger.logger.debug(f"✅ Added timeframe context: +{len(context_addition)} chars")
            
            # ================================================================
            # 📊 EXPANSION STRATEGY 2: ADD TOKEN-SPECIFIC CONTEXT
            # ================================================================
            
            if len(expanded) < 100:  # Still need more content
                token_contexts = {
                    "BTC": "Bitcoin's market leadership role",
                    "ETH": "Ethereum's ecosystem strength",
                    "default": f"{token}'s market positioning"
                }
                
                token_context = token_contexts.get(token, token_contexts["default"])
                context_addition = f" {token_context} remains a key factor in this assessment."
                
                if len(expanded + context_addition) <= 280:
                    expanded += context_addition
                    logger.logger.debug(f"✅ Added token context: +{len(context_addition)} chars")
            
            # ================================================================
            # 📊 EXPANSION STRATEGY 3: ADD GENERAL MARKET WISDOM
            # ================================================================
            
            if len(expanded) < 120:  # Still need more content
                market_wisdom = [
                    "Market volatility requires careful position sizing.",
                    "Risk management remains paramount in current conditions.",
                    "Technical levels provide guidance for entry and exit points.",
                    "Volume confirmation adds validity to price movements."
                ]
                
                wisdom = random.choice(market_wisdom)
                
                if len(expanded + " " + wisdom) <= 280:
                    expanded += f" {wisdom}"
                    logger.logger.debug(f"✅ Added market wisdom: +{len(wisdom)} chars")
            
            logger.logger.debug(f"📊 EXPANSION COMPLETE:")
            logger.logger.debug(f"   • Original: {len(content)} chars")
            logger.logger.debug(f"   • Expanded: {len(expanded)} chars")
            logger.logger.debug(f"   • Added: +{len(expanded) - len(content)} chars")
            
            return expanded
            
        except Exception as e:
            logger.logger.error(f"❌ SMART EXPANSION FAILED: {str(e)}")
            return content  # Return original if expansion fails


    def _replace_fallback_phrases(self, content: str, detected_fallbacks: List[str], token: str, timeframe: str) -> str:
        """
        Replace detected fallback phrases with more professional alternatives
        
        Args:
            content (str): Content containing fallback phrases
            detected_fallbacks (List[str]): List of detected fallback phrases
            token (str): Token symbol for context
            timeframe (str): Timeframe for appropriate replacements
        
        Returns:
            str: Content with fallback phrases replaced
        """
        logger.logger.debug(f"🔧 FALLBACK REPLACEMENT: {len(detected_fallbacks)} phrases")
        
        try:
            improved_content = content
            
            # Professional replacements for common fallback phrases
            replacements = {
                "to the moon": f"{token} shows strong upward momentum",
                "hodl strong": f"maintaining {token} positions recommended",
                "bullish af": f"{token} displays bullish technical signals",
                "buy the dip": f"{token} presents potential accumulation opportunity",
                "diamond hands": "strong conviction positions warranted",
                "breaking out": f"{token} approaching key resistance levels",
                "getting ready to pump": f"{token} building momentum for potential move",
                "solid fundamentals": f"{token} maintains strong technical foundation",
                "undervalued gem": f"{token} shows favorable risk/reward profile",
                "massive potential": f"{token} presents significant opportunity",
                "smart money flowing": "institutional interest appears to be increasing",
                "lfg": "momentum building for potential significant move"
            }
            
            # Replace each detected fallback with professional alternative
            for fallback in detected_fallbacks:
                if fallback in replacements:
                    professional_version = replacements[fallback]
                    improved_content = improved_content.replace(fallback, professional_version)
                    logger.logger.debug(f"✅ Replaced '{fallback}' with '{professional_version}'")
                else:
                    # Generic professional replacement
                    generic_replacement = f"{token} technical analysis suggests continued monitoring"
                    improved_content = improved_content.replace(fallback, generic_replacement)
                    logger.logger.debug(f"✅ Generic replacement for '{fallback}'")
            
            return improved_content
            
        except Exception as e:
            logger.logger.error(f"❌ FALLBACK REPLACEMENT FAILED: {str(e)}")
            return content


    def _enhance_generic_content(self, content: str, token: str, timeframe: str, detected_generic: List[str]) -> str:
        """
        Enhance generic content with more specific and engaging language
        
        Args:
            content (str): Generic content to enhance
            token (str): Token symbol for specific context
            timeframe (str): Timeframe for targeted language
            detected_generic (List[str]): List of detected generic phrases
        
        Returns:
            str: Enhanced content with more specific language
        """
        logger.logger.debug(f"🔧 GENERIC ENHANCEMENT: {len(detected_generic)} phrases")
        
        try:
            enhanced_content = content
            
            # Specific replacements for generic phrases
            enhancements = {
                "market analysis": f"{token} {timeframe} technical assessment",
                "technical indicators": f"{token} momentum and trend signals",
                "price movement": f"{token} directional bias analysis",
                "trading volume": f"{token} liquidity and participation levels",
                "market sentiment": f"{token} trader positioning and outlook",
                "showing mixed signals": "presenting conflicting technical indications",
                "moderately bearish": "displaying cautious downside bias",
                "moderately bullish": "exhibiting measured upside potential",
                "neutral": f"{token} consolidating within defined range"
            }
            
            # Replace generic phrases with more specific alternatives
            for generic in detected_generic:
                if generic in enhancements:
                    specific_version = enhancements[generic]
                    enhanced_content = enhanced_content.replace(generic, specific_version)
                    logger.logger.debug(f"✅ Enhanced '{generic}' to '{specific_version}'")
            
            return enhanced_content
            
        except Exception as e:
            logger.logger.error(f"❌ GENERIC ENHANCEMENT FAILED: {str(e)}")
            return content


    def _add_technical_insights(self, content: str, token: str, timeframe: str) -> str:
        """
        Add relevant technical insights to content lacking technical depth
        
        Args:
            content (str): Content lacking technical insights
            token (str): Token symbol for context
            timeframe (str): Timeframe for appropriate technical focus
        
        Returns:
            str: Content enriched with technical insights
        """
        logger.logger.debug(f"🔧 TECHNICAL ENRICHMENT: {token} ({timeframe})")
        
        try:
            # Technical insights appropriate for different timeframes
            technical_additions = {
                "1h": [
                    "RSI levels suggest potential momentum shifts ahead.",
                    "Volume profile indicates active participation at current levels.",
                    "Support/resistance dynamics favor defined risk parameters.",
                    "Moving average convergence suggests directional clarity emerging."
                ],
                "24h": [
                    "Daily chart structure shows key technical level interaction.",
                    "Momentum oscillators indicate potential trend continuation or reversal.",
                    "Volume patterns suggest institutional versus retail participation.",
                    "Fibonacci retracement levels provide strategic entry/exit zones."
                ],
                "7d": [
                    "Weekly trend analysis reveals macro directional bias.",
                    "Long-term moving averages suggest structural market positioning.",
                    "Volume accumulation patterns indicate smart money positioning.",
                    "Cyclical analysis suggests optimal timing for position adjustments."
                ]
            }
            
            # Select appropriate technical addition
            additions = technical_additions.get(timeframe, technical_additions["1h"])
            selected_addition = random.choice(additions)
            
            # Add technical insight if there's room
            enriched_content = content
            if len(content + " " + selected_addition) <= 280:
                enriched_content = f"{content} {selected_addition}"
                logger.logger.debug(f"✅ Added technical insight: '{selected_addition}'")
            else:
                logger.logger.debug(f"⚠️ Skipped technical addition: would exceed length limit")
            
            return enriched_content
            
        except Exception as e:
            logger.logger.error(f"❌ TECHNICAL ENRICHMENT FAILED: {str(e)}")
            return content

    @ensure_naive_datetimes
    def _handle_insufficient_content(self, token: str, timeframe: str, reason: str) -> None:
        """
        Handle cases where content isn't worth posting
        """
        try:
            logger.logger.info(f"Skipping {token} {timeframe} post: {reason}")
        
            # Update next check time to avoid immediate retry
            if timeframe in self.next_scheduled_posts:
                delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * 0.5
                self.next_scheduled_posts[timeframe] = strip_timezone(
                    datetime.now() + timedelta(hours=delay_hours)
                )
        
            # Track skipped posts for analytics (optional - will work even if method doesn't exist)
            try:
                if hasattr(self.db, '_store_json_data'):
                    skip_data = {
                        'token': token,
                        'timeframe': timeframe, 
                        'reason': reason,
                        'timestamp': strip_timezone(datetime.now()).isoformat()
                    }
                    self.db._store_json_data('skipped_post', skip_data)
            except Exception as storage_error:
                # Don't fail the whole method if storage fails
                logger.logger.debug(f"Could not store skip data: {str(storage_error)}")
            
        except Exception as e:
            logger.log_error(f"Handle Insufficient Content - {token}", str(e))

    @ensure_naive_datetimes
    def start(self) -> None:
        """
        Main bot execution loop with multi-timeframe support and reply functionality
        """
        try:
            retry_count = 0
            max_setup_retries = 3
            
            # Start the prediction thread early
            self._start_prediction_thread()
            
            # Load saved timeframe state
            self._load_saved_timeframe_state()
            
            # Initialize the browser and login
            while retry_count < max_setup_retries:
                if not self.browser.initialize_driver():
                    retry_count += 1
                    logger.logger.warning(f"Browser initialization attempt {retry_count} failed, retrying...")
                    time.sleep(10)
                    continue
                    
                if not self._login_to_twitter():
                    retry_count += 1
                    logger.logger.warning(f"Twitter login attempt {retry_count} failed, retrying...")
                    time.sleep(15)
                    continue
                    
                break
            
            if retry_count >= max_setup_retries:
                raise Exception("Failed to initialize bot after maximum retries")

            logger.logger.info("Bot initialized successfully")
            
            # Log the timeframes that will be used
            logger.logger.info(f"Bot configured with timeframes: {', '.join(self.timeframes)}")
            logger.logger.info(f"Timeframe posting frequencies: {self.timeframe_posting_frequency}")
            logger.logger.info(f"Reply checking interval: {self.reply_check_interval} minutes")

            # Pre-queue predictions for all tokens and timeframes - DATABASE DRIVEN
            market_data = self._get_crypto_data()
            if market_data:
                logger.logger.info("🔍 Getting top tokens from database for prediction queueing...")
                
                try:
                    # Get top tokens by market cap from database (no fallbacks)
                    database_tokens = self.get_tokens_with_recent_data_by_market_cap(hours=24, limit=25)
                    if not database_tokens:
                        logger.logger.error("❌ CRITICAL: No tokens from database - cannot queue predictions")
                        raise SystemExit("Database connection lost - no tokens for prediction queueing")
                    
                    # Filter to only tokens present in market_data
                    available_tokens = [token for token in database_tokens if token in market_data]
                    logger.logger.info(f"📊 {len(available_tokens)} database tokens available in market_data: {available_tokens}")
                    
                    if not available_tokens:
                        logger.logger.error("❌ CRITICAL: No database tokens found in market_data")
                        raise SystemExit("Market data incompatible with database tokens")
                    
                    # Only queue predictions for the most important tokens to avoid overloading
                    top_tokens = self._prioritize_tokens(available_tokens, market_data)[:5]
                    
                    logger.logger.info(f"🎯 Pre-queueing predictions for top database tokens: {', '.join(top_tokens)}")
                    for token in top_tokens:
                        self._queue_predictions_for_all_timeframes(token, market_data)
                        
                except Exception as e:
                    logger.logger.error(f"❌ CRITICAL: Database token selection failed: {str(e)}")
                    raise SystemExit(f"Prediction queueing failed: {str(e)}")

            while True:
                try:
                    self._run_analysis_cycle()
                    
                    # Calculate sleep time until next regular check
                    time_since_last = safe_datetime_diff(datetime.now(), self.last_check_time)
                    sleep_time = max(0, config.BASE_INTERVAL - time_since_last)
                    
                    # Check if we should post a weekly summary
                    if self._generate_weekly_summary():
                        logger.logger.info("Posted weekly performance summary")   

                    logger.logger.debug(f"Sleeping for {sleep_time:.1f}s until next check")
                    time.sleep(sleep_time)
                    
                    self.last_check_time = strip_timezone(datetime.now())
                    
                except Exception as e:
                    logger.log_error("Analysis Cycle", str(e), exc_info=True)
                    time.sleep(60)  # Shorter sleep on error
                    continue

        except KeyboardInterrupt:
            logger.logger.info("Bot stopped by user")
        except Exception as e:
            logger.log_error("Bot Execution", str(e))
        finally:
            self._cleanup()

if __name__ == "__main__":
    try:
        # Import necessary components
        from config import config  # This already has the database initialized
        from llm_provider import LLMProvider
        
        # Create LLM provider
        llm_provider = LLMProvider(config)
        
        # Create the bot using the database from config
        bot = CryptoAnalysisBot(
            database=config.db,  # Use the database that's already initialized in config
            llm_provider=llm_provider,
            config=config
        )
        
        # Start the bot
        bot.start()
    except Exception as e:
        from utils.logger import logger
        logger.log_error("Bot Startup", str(e))       
