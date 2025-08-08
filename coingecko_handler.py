#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import requests
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime, timedelta
from datetime_utils import strip_timezone, ensure_naive_datetimes, safe_datetime_diff
import json
import math
from utils.logger import logger

class CoinGeckoDataValidator:
    """
    🔍 STRICT DATA VALIDATION FOR GENERATIONAL WEALTH 🔍
    
    Ensures all market data meets strict quality standards
    NO SYNTHETIC DATA - REAL DATA ONLY
    """
    
    @staticmethod
    def validate_price_data(price_data: Any) -> Tuple[bool, float]:
        """
        Validate price data is real and usable
        
        Returns:
            (is_valid, validated_price)
        """
        try:
            if price_data is None:
                return False, 0.0
            
            price = float(price_data)
            
            # Strict validation rules
            if price <= 0:
                return False, 0.0
            if math.isnan(price) or math.isinf(price):
                return False, 0.0
            if price > 1e12:  # Unrealistic price (> $1 trillion)
                return False, 0.0
            
            return True, price
            
        except (ValueError, TypeError):
            return False, 0.0
    
    @staticmethod
    def validate_sparkline_data(sparkline_data: Any) -> Tuple[bool, List[float]]:
        """
        Validate sparkline data is complete and real
        
        Returns:
            (is_valid, price_array)
        """
        try:
            if not sparkline_data:
                return False, []
            
            # Extract price array based on CoinGecko format
            price_array = []
            
            if isinstance(sparkline_data, dict):
                if 'price' in sparkline_data:
                    price_array = sparkline_data['price']
                elif 'prices' in sparkline_data:
                    price_array = sparkline_data['prices']
            elif isinstance(sparkline_data, list):
                price_array = sparkline_data
            else:
                return False, []
            
            # Strict validation requirements
            if not price_array or len(price_array) < 50:  # Require substantial data
                return False, []
            
            # Validate each price point
            validated_prices = []
            for price in price_array:
                is_valid, validated_price = CoinGeckoDataValidator.validate_price_data(price)
                if not is_valid:
                    return False, []  # ANY invalid price fails the entire array
                validated_prices.append(validated_price)
            
            # Check for data completeness (no long flat lines indicating stale data)
            unique_prices = len(set(validated_prices))
            if unique_prices < len(validated_prices) * 0.1:  # Less than 10% unique prices
                logger.logger.warning("Sparkline data appears stale - too many duplicate prices")
                return False, []
            
            return True, validated_prices
            
        except Exception as e:
            logger.logger.error(f"Sparkline validation failed: {e}")
            return False, []
    
    @staticmethod
    def validate_market_data_entry(entry: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a complete market data entry
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Required fields for trading
            required_fields = ['id', 'symbol', 'current_price', 'market_cap', 'total_volume']
            
            for field in required_fields:
                if field not in entry:
                    return False, f"Missing required field: {field}"
            
            # Validate current price
            is_valid, _ = CoinGeckoDataValidator.validate_price_data(entry['current_price'])
            if not is_valid:
                return False, "Invalid current_price"
            
            # Validate volume (can be 0 but not negative)
            try:
                volume = float(entry['total_volume'])
                if volume < 0 or math.isnan(volume) or math.isinf(volume):
                    return False, "Invalid volume"
            except (ValueError, TypeError):
                return False, "Invalid volume format"
            
            # Validate market cap
            try:
                market_cap = float(entry['market_cap'])
                if market_cap <= 0 or math.isnan(market_cap) or math.isinf(market_cap):
                    return False, "Invalid market_cap"
            except (ValueError, TypeError):
                return False, "Invalid market_cap format"
            
            # Validate sparkline if present
            if 'sparkline_in_7d' in entry:
                sparkline_valid, _ = CoinGeckoDataValidator.validate_sparkline_data(entry['sparkline_in_7d'])
                if not sparkline_valid:
                    return False, "Invalid sparkline data"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_batch_data(data: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        """
        Validate entire batch of market data
        
        Returns:
            (batch_is_valid, valid_entries, error_messages)
        """
        if not data or not isinstance(data, list):
            return False, [], ["No data provided or invalid format"]
        
        valid_entries = []
        error_messages = []
        
        for i, entry in enumerate(data):
            is_valid, error_msg = CoinGeckoDataValidator.validate_market_data_entry(entry)
            if is_valid:
                valid_entries.append(entry)
            else:
                error_messages.append(f"Entry {i} ({entry.get('symbol', 'UNKNOWN')}): {error_msg}")
        
        # Require at least 80% of entries to be valid for batch to pass
        success_rate = len(valid_entries) / len(data)
        batch_valid = success_rate >= 0.8
        
        if not batch_valid:
            error_messages.append(f"Batch validation failed: only {success_rate:.1%} valid entries")
        
        return batch_valid, valid_entries, error_messages

class CoinGeckoQuotaTracker:
    """
    📊 API QUOTA TRACKING FOR FREE TIER 📊
    
    Prevents hitting API limits that could suspend trading
    """
    
    def __init__(self, daily_limit: int = 400):  # Conservative limit under 500
        """
        Initialize quota tracker for CoinGecko API
        
        Args:
            daily_limit: Maximum requests per day (default 400 for free tier)
        """
        self.daily_limit = daily_limit
        self.requests_today = 0
        self.requests_this_minute = 0
        self.last_request_time = 0
        self.minute_start = time.time()
        self.day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Track request history for analysis
        self.request_history = []
        self.failed_requests_today = 0
        
        logger.logger.info(f"📊 CoinGecko Quota Tracker initialized with daily limit: {daily_limit}")
    
    def can_make_request(self) -> Tuple[bool, str]:
        """
        Check if we can make a request without hitting limits
        
        Returns:
            (can_request, reason_if_not)
        """
        current_time = time.time()
        
        # Reset minute counter if needed
        if current_time - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.minute_start = current_time
        
        # Reset daily counter if needed
        current_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if current_day > self.day_start:
            self.requests_today = 0
            self.failed_requests_today = 0
            self.day_start = current_day
            logger.logger.info("🔄 Daily quota reset")
        
        # Check daily limit
        if self.requests_today >= self.daily_limit:
            return False, f"Daily limit reached ({self.requests_today}/{self.daily_limit})"
        
        # Check minute limit (25 per minute for safety)
        if self.requests_this_minute >= 25:
            return False, f"Minute limit reached ({self.requests_this_minute}/25)"
        
        # Check if we need to wait for rate limiting
        time_since_last = current_time - self.last_request_time
        if time_since_last < 3.0:  # Conservative 3-second minimum
            return False, f"Rate limit: need to wait {3.0 - time_since_last:.1f}s"
        
        return True, "OK"
    
    def record_request(self, success: bool) -> None:
        """Record a request for quota tracking"""
        current_time = time.time()
        
        self.requests_today += 1
        self.requests_this_minute += 1
        self.last_request_time = current_time
        
        if not success:
            self.failed_requests_today += 1
        
        # Keep history for last 100 requests
        self.request_history.append({
            'timestamp': current_time,
            'success': success
        })
        if len(self.request_history) > 100:
            self.request_history.pop(0)
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota status"""
        current_time = time.time()
        
        # Calculate success rate from recent history
        recent_requests = [r for r in self.request_history if current_time - r['timestamp'] < 3600]  # Last hour
        success_rate = sum(1 for r in recent_requests if r['success']) / len(recent_requests) if recent_requests else 1.0
        
        return {
            'requests_today': self.requests_today,
            'daily_limit': self.daily_limit,
            'daily_remaining': self.daily_limit - self.requests_today,
            'requests_this_minute': self.requests_this_minute,
            'failed_requests_today': self.failed_requests_today,
            'success_rate_1h': success_rate,
            'time_until_next_allowed': max(0, 3.0 - (current_time - self.last_request_time))
        }

class CoinGeckoHandler:
    """
    🚀 GENERATIONAL WEALTH COINGECKO HANDLER 🚀
    
    Enhanced for serious wealth generation with strict data validation
    Built on the original foundation with critical improvements:
    - Conservative rate limiting for free tier
    - Strict data validation (NO synthetic data)
    - Extended caching to reduce API calls
    - Comprehensive quota tracking
    - Proper failure modes that stop trading on bad data
    """
    
    def __init__(self, base_url: str, cache_duration: int = 300) -> None:  # 5-minute cache
        """
        Initialize the enhanced CoinGecko handler
        
        Args:
            base_url: The base URL for the CoinGecko API
            cache_duration: Cache duration in seconds (default 5 minutes)
        """
        self.base_url = base_url
        self.cache_duration = cache_duration
        self.cache = {}
        self.last_request_time = 0
        self.min_request_interval = 60.0  # Conservative 60 seconds for free tier
        
        # Initialize quota tracker
        self.quota_tracker = CoinGeckoQuotaTracker(daily_limit=400)
        
        # Initialize data validator
        self.validator = CoinGeckoDataValidator()
        
        # Enhanced tracking
        self.daily_requests = 0
        self.daily_requests_reset = datetime.now()
        self.failed_requests = 0
        self.active_retries = 0
        self.max_retries = 2  # Reduced retries to conserve quota
        self.retry_delay = 10  # Longer delay between retries
        
        # Data quality tracking
        self.data_quality_stats = {
            'valid_requests': 0,
            'invalid_data_responses': 0,
            'total_validation_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Enhanced headers for better API compatibility
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        self.token_id_cache = {}  # Cache for token ID lookups
        
        logger.logger.info("🚀 Generational Wealth CoinGecko Handler initialized")
        logger.logger.info(f"📊 Cache duration: {cache_duration}s, Rate limit: {self.min_request_interval}s")
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate a unique cache key for the request"""
        param_str = json.dumps(params, sort_keys=True)
        return f"{endpoint}:{param_str}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_entry = self.cache[cache_key]
        cache_time = cache_entry['timestamp']
        current_time = time.time()
        
        return (current_time - cache_time) < self.cache_duration
    
    def _get_from_cache(self, cache_key: str) -> Any:
        """Get data from cache if available and valid"""
        if self._is_cache_valid(cache_key):
            self.data_quality_stats['cache_hits'] += 1
            logger.logger.debug(f"💾 Cache hit for {cache_key}")
            return self.cache[cache_key]['data']
        
        self.data_quality_stats['cache_misses'] += 1
        return None
    
    def _add_to_cache(self, cache_key: str, data: Any) -> None:
        """Add validated data to cache"""
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': data,
            'validated': True  # Mark as validated data
        }
        logger.logger.debug(f"💾 Added validated data to cache: {cache_key}")
    
    def _clean_cache(self) -> None:
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if (current_time - entry['timestamp']) >= self.cache_duration
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.logger.debug(f"🗑️ Cleaned {len(expired_keys)} expired cache entries")
    
    def _enforce_rate_limit(self) -> bool:
        """
        Enforce strict rate limiting for free tier
        
        Returns:
            True if request can proceed, False if blocked
        """
        # Check quota first
        can_request, reason = self.quota_tracker.can_make_request()
        if not can_request:
            logger.logger.warning(f"🚫 Request blocked: {reason}")
            return False
        
        # Enforce minimum interval
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.logger.debug(f"⏳ Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Reset daily request count if a day has passed
        if safe_datetime_diff(datetime.now(), self.daily_requests_reset) >= 86400:
            self.daily_requests = 0
            self.daily_requests_reset = datetime.now()
            logger.logger.info("🔄 Daily request counter reset")
        
        return True
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, request_type: str = "standard") -> Optional[Any]:
        """
        Make a request to the CoinGecko API with strict validation
        
        🎯 ENHANCED with API Manager coordination and request type tracking
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            request_type: Type of request for load balancing coordination ("bulk", "individual", "historical", "real_time", "standard")
        
        Returns:
            Validated data or None if validation fails
        """
        if params is None:
            params = {}
        
        # 🎯 NEW: Log request type and API Manager coordination
        import inspect
        caller_info = inspect.stack()[1] if len(inspect.stack()) > 1 else None
        is_api_manager_call = caller_info and 'api_manager' in str(caller_info.filename).lower()
        
        if is_api_manager_call:
            logger.logger.debug(f"📊 CoinGecko API Request: {request_type} via API Manager routing")
        else:
            logger.logger.debug(f"📊 CoinGecko API Request: {request_type} via direct handler")
        
        # 🎯 NEW: Log rate limit prevention working
        quota_status = self.quota_tracker.get_quota_status()
        if request_type in ["bulk", "historical"] and quota_status['daily_remaining'] < 50:
            logger.logger.warning(f"📊 Rate Limit Prevention: {request_type} request with low quota ({quota_status['daily_remaining']} remaining)")
            if is_api_manager_call:
                logger.logger.info("📊 API Manager Coordination: Should route this to CoinMarketCap for rate limit prevention")
        
        # Check if we can make the request
        if not self._enforce_rate_limit():
            # 🎯 NEW: Enhanced rate limit logging
            logger.logger.warning(f"📊 Rate Limit Block: {request_type} request blocked by rate limiter")
            if is_api_manager_call:
                logger.logger.info("📊 API Manager Coordination: Rate limit hit - recommend provider switch")
            return None
        
        # 🎯 NEW: Log request characteristics for load balancing analysis
        token_count = len(params.get('ids', '').split(',')) if params.get('ids') else 1
        per_page = params.get('per_page', 0)
        
        logger.logger.debug(f"📊 Request Characteristics: {endpoint} | {token_count} tokens | per_page={per_page} | type={request_type}")
        
        url = f"{self.base_url}/{endpoint}"
        self.last_request_time = time.time()
        self.daily_requests += 1
        
        success = False
        response_data = None
        start_time = time.time()
        
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            }
            
            logger.logger.debug(f"📡 Making API request to {endpoint}")
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                
                # CRITICAL: Validate the response data
                if isinstance(response_data, list):
                    # Validate batch data
                    is_valid, valid_entries, errors = self.validator.validate_batch_data(response_data)
                    if is_valid:
                        success = True
                        response_data = valid_entries
                        self.data_quality_stats['valid_requests'] += 1
                        
                        # 🎯 NEW: Enhanced success logging with request type
                        duration = time.time() - start_time
                        logger.logger.info(f"✅ Validated API response: {len(valid_entries)} valid entries")
                        logger.logger.debug(f"📊 CoinGecko Performance: {request_type} request completed in {duration:.3f}s")
                        
                        # 🎯 NEW: Log specialization effectiveness
                        if request_type in ["real_time", "individual", "standard"] and is_api_manager_call:
                            logger.logger.debug("📊 Specialization Match: ✅ CoinGecko optimal for this request type")
                        elif request_type in ["bulk", "historical"] and is_api_manager_call:
                            logger.logger.debug("📊 Specialization Alert: ⚠️ Non-optimal request type for CoinGecko")
                    else:
                        success = False
                        self.data_quality_stats['invalid_data_responses'] += 1
                        logger.logger.error(f"❌ API response validation failed:")
                        for error in errors[:3]:  # Log first 3 errors
                            logger.logger.error(f"   {error}")
                        response_data = None
                        
                        # 🎯 NEW: Enhanced validation failure logging
                        if is_api_manager_call:
                            logger.logger.error("📊 API Manager Coordination: Validation failure - recommend fallback provider")
                            
                elif isinstance(response_data, dict):
                    # For single coin data, validate differently
                    if 'id' in response_data and 'market_data' in response_data:
                        # This is detailed coin data
                        success = True
                        self.data_quality_stats['valid_requests'] += 1
                        
                        # 🎯 NEW: Single coin success logging
                        duration = time.time() - start_time
                        logger.logger.debug(f"📊 CoinGecko Performance: Single coin {request_type} request completed in {duration:.3f}s")
                    else:
                        # Validate as market data entry
                        is_valid, error_msg = self.validator.validate_market_data_entry(response_data)
                        if is_valid:
                            success = True
                            self.data_quality_stats['valid_requests'] += 1
                            
                            # 🎯 NEW: Market data entry success logging
                            duration = time.time() - start_time
                            logger.logger.debug(f"📊 CoinGecko Performance: Market data {request_type} request completed in {duration:.3f}s")
                        else:
                            success = False
                            self.data_quality_stats['invalid_data_responses'] += 1
                            logger.logger.error(f"❌ Single entry validation failed: {error_msg}")
                            response_data = None
                            
                            # 🎯 NEW: Enhanced single entry failure logging
                            if is_api_manager_call:
                                logger.logger.error("📊 API Manager Coordination: Single entry validation failure")
                else:
                    # Other data types (like coins list) - allow through
                    success = True
                    self.data_quality_stats['valid_requests'] += 1
                    
                    # 🎯 NEW: Other data types logging
                    duration = time.time() - start_time
                    logger.logger.debug(f"📊 CoinGecko Performance: Other data {request_type} request completed in {duration:.3f}s")
                    
            elif response.status_code == 429:
                success = False
                self.failed_requests += 1
                
                # 🎯 NEW: Enhanced rate limit exceeded logging
                logger.logger.error(f"🚫 API rate limit exceeded: {response.status_code}")
                logger.logger.error(f"📊 Rate Limit Context: {request_type} request, {quota_status['daily_remaining']} quota remaining")
                
                if is_api_manager_call:
                    logger.logger.error("📊 API Manager Coordination: Rate limit exceeded - immediate provider switch recommended")
                    
                # Don't retry immediately on rate limits
                return None
            else:
                success = False
                self.failed_requests += 1
                
                # 🎯 NEW: Enhanced general failure logging
                duration = time.time() - start_time
                logger.logger.error(f"❌ API request failed: {response.status_code} - {response.text[:200]}")
                logger.logger.error(f"📊 Request Context: {request_type} request failed after {duration:.3f}s")
                
                if is_api_manager_call:
                    logger.logger.error("📊 API Manager Coordination: Request failure - consider fallback provider")
                    
                response_data = None
                
        except requests.exceptions.RequestException as e:
            success = False
            self.failed_requests += 1
            duration = time.time() - start_time
            
            # 🎯 NEW: Enhanced request exception logging
            logger.logger.error(f"🌐 Request exception: {str(e)}")
            logger.logger.error(f"📊 Exception Context: {request_type} request, network issue after {duration:.3f}s")
            
            if is_api_manager_call:
                logger.logger.error("📊 API Manager Coordination: Network exception - fallback provider recommended")
                
            response_data = None
        except Exception as e:
            success = False
            self.failed_requests += 1
            duration = time.time() - start_time
            
            # 🎯 NEW: Enhanced unexpected error logging
            logger.logger.error(f"💥 Unexpected error in API request: {str(e)}")
            logger.logger.error(f"📊 Error Context: {request_type} request, unexpected error after {duration:.3f}s")
            
            if is_api_manager_call:
                logger.logger.error("📊 API Manager Coordination: Unexpected error - immediate fallback required")
                
            response_data = None
        
        # Record request for quota tracking
        self.quota_tracker.record_request(success)
        
        # 🎯 NEW: Final performance and coordination summary
        total_duration = time.time() - start_time
        if success:
            logger.logger.debug(f"📊 Request Summary: ✅ {request_type} completed successfully in {total_duration:.3f}s")
            if is_api_manager_call:
                logger.logger.debug("📊 Provider Coordination: CoinGecko request successful - maintaining provider preference")
        else:
            logger.logger.debug(f"📊 Request Summary: ❌ {request_type} failed after {total_duration:.3f}s")
            if is_api_manager_call:
                logger.logger.debug("📊 Provider Coordination: CoinGecko request failed - API Manager should switch provider")
        
        return response_data
    
    def get_with_cache(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Get validated data from API with caching"""
        if params is None:
            params = {}
        
        cache_key = self._get_cache_key(endpoint, params)
        
        # Try to get from cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Check quota before making any requests
        quota_status = self.quota_tracker.get_quota_status()
        if quota_status['daily_remaining'] <= 10:  # Reserve last 10 requests
            logger.logger.error(f"🚫 QUOTA CRITICAL: Only {quota_status['daily_remaining']} requests remaining today")
            return None
        
        # Not in cache, make API request with retries
        retry_count = 0
        while retry_count < self.max_retries:
            data = self._make_request(endpoint, params)
            if data is not None:
                self._add_to_cache(cache_key, data)
                return data
            
            retry_count += 1
            if retry_count < self.max_retries:
                wait_time = self.retry_delay * retry_count
                logger.logger.warning(f"🔄 Retrying API request ({retry_count}/{self.max_retries}) in {wait_time}s")
                time.sleep(wait_time)
        
        logger.logger.error(f"❌ CRITICAL: Failed to get validated data after {self.max_retries} retries")
        self.data_quality_stats['total_validation_failures'] += 1
        return None
    
    def get_market_data(self, params: Optional[Dict[str, Any]] = None, timeframe: str = "24h", 
                        priority_tokens: Optional[List[str]] = None, include_price_history: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Get cryptocurrency market data with sophisticated hybrid approach for advanced trading
        
        Optimized for free tier: Batch basic data + selective price history for priority tokens
        
        Args:
            params: Query parameters for the API
            timeframe: Analysis timeframe ("1h", "24h", "7d") - determines price change parameters
            priority_tokens: List of token IDs to fetch detailed price history for
            include_price_history: Whether to fetch price history for priority tokens
        
        Returns:
            List of VALIDATED market data entries with selective price history enhancement
        """
        endpoint = "coins/markets"
        
        # Set default params for all tracked tokens with timeframe-appropriate settings
        if params is None:
            # Determine price change parameters based on timeframe
            if timeframe == "1h":
                price_change_param = "1h,24h"  # Focus on short-term changes
            elif timeframe == "24h":
                price_change_param = "1h,24h,7d"  # Full range of changes
            elif timeframe == "7d":
                price_change_param = "24h,7d,30d"  # Longer-term perspective
            else:
                price_change_param = "1h,24h,7d"  # Default to full range
            
            params = {
                "vs_currency": "usd",
                "ids": "bitcoin,ethereum,solana,ripple,binancecoin,avalanche-2,polkadot,uniswap,near,aave,matic-network,filecoin,kaito",
                "order": "market_cap_desc",
                "per_page": 30,
                "page": 1,
                "sparkline": False,  # Markets endpoint doesn't provide sparkline data
                "price_change_percentage": price_change_param
            }
        else:
            # CRITICAL FIX: Ensure vs_currency is always present
            if 'vs_currency' not in params:
                params['vs_currency'] = 'usd'

            # If params provided, ensure timeframe-appropriate price change parameters
            if 'price_change_percentage' not in params:
                if timeframe == "1h":
                    params['price_change_percentage'] = "1h,24h"
                elif timeframe == "24h":
                    params['price_change_percentage'] = "1h,24h,7d"
                elif timeframe == "7d":
                    params['price_change_percentage'] = "24h,7d,30d"
                else:
                    params['price_change_percentage'] = "1h,24h,7d"
            
            # Ensure sparkline is enabled (even though markets endpoint doesn't provide it)
            if 'sparkline' not in params:
                params['sparkline'] = True
        
        # Add timeframe metadata to params for tracking
        params['_analysis_timeframe'] = timeframe
        
        # ================================================================
        # 🚀 STEP 1: GET BATCH BASIC DATA (EFFICIENT)
        # ================================================================
        
        logger.logger.debug(f"📡 Fetching batch market data for {timeframe} analysis")
        result = self.get_with_cache(endpoint, params)
        
        # Validate basic response
        if result is None:
            logger.logger.error("🚫 TRADING HALT: No validated market data available")
            return None
        
        # Parse string responses if needed
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                logger.logger.error("❌ Failed to parse string result from CoinGecko API")
                return None
        
        if not isinstance(result, list):
            logger.logger.error("❌ Expected list of market data, got different format")
            return None
        
        logger.logger.info(f"✅ Batch data received: {len(result)} tokens for {timeframe} analysis")
        
        # ================================================================
        # 🎯 STEP 2: ENHANCE PRIORITY TOKENS WITH PRICE HISTORY (SELECTIVE)
        # ================================================================
        
        if include_price_history and priority_tokens:
            logger.logger.info(f"🔍 Fetching detailed price history for {len(priority_tokens)} priority tokens")
            
            # Enhance priority tokens with price history
            enhanced_result = []
            api_calls_made = 0
            
            for token_data in result:
                # Safe dictionary access - this should work fine since result comes from JSON API
                token_id = token_data.get('id') if isinstance(token_data, dict) else None
                
                # Check if this token is in priority list
                if token_id and token_id in priority_tokens:
                    try:
                        # Determine optimal timespan based on analysis timeframe
                        if timeframe == "1h":
                            days_param = 2  # 2 days of hourly data for 1h analysis
                        elif timeframe == "24h":
                            days_param = 7  # 7 days of hourly data for 24h analysis  
                        elif timeframe == "7d":
                            days_param = 30  # 30 days for 7d analysis
                        else:
                            days_param = 7  # Default to 7 days
                        
                        # Fetch detailed price history for this priority token
                        chart_endpoint = f"coins/{token_id}/market_chart"
                        chart_params = {
                            "vs_currency": "usd",
                            "days": days_param
                        }
                        
                        logger.logger.debug(f"📊 Fetching {days_param}-day price history for {token_id}")
                        
                        # Use cache-enabled request
                        chart_data = self.get_with_cache(chart_endpoint, chart_params)
                        api_calls_made += 1
                        
                        if chart_data and isinstance(chart_data, dict):
                            prices_data = chart_data.get('prices', [])
                            volumes_data = chart_data.get('total_volumes', [])
                            
                            if prices_data and len(prices_data) >= 20:  # Minimum viable for technical analysis
                                # Extract price history array
                                price_history = [entry[1] for entry in prices_data if len(entry) >= 2]
                                volume_history = [entry[1] for entry in volumes_data if len(entry) >= 2]
                                
                                # Add price history to token data
                                token_data['price_history'] = price_history
                                token_data['volume_history'] = volume_history
                                token_data['price_history_points'] = len(price_history)
                                token_data['price_history_timespan_days'] = days_param
                                token_data['_enhanced_with_history'] = True
                                
                                logger.logger.debug(f"✅ Enhanced {token_id} with {len(price_history)} price points")
                            else:
                                logger.logger.warning(f"⚠️ Insufficient price history for {token_id}: {len(prices_data)} points")
                                token_data['_enhanced_with_history'] = False
                        else:
                            logger.logger.warning(f"⚠️ Failed to fetch price history for {token_id}")
                            token_data['_enhanced_with_history'] = False
                        
                        # Rate limiting between individual calls
                        if api_calls_made < len(priority_tokens):
                            time.sleep(1.2)  # Conservative rate limiting
                            
                    except Exception as e:
                        logger.logger.error(f"❌ Error enhancing {token_id}: {str(e)}")
                        token_data['_enhanced_with_history'] = False
                else:
                    # Not a priority token - no price history enhancement
                    token_data['_enhanced_with_history'] = False
                
                enhanced_result.append(token_data)
            
            result = enhanced_result
            logger.logger.info(f"🎯 Enhanced {api_calls_made} priority tokens with detailed price history")
        
        # ================================================================
        # 🏷️ STEP 3: ADD PROCESSING METADATA
        # ================================================================
        
        # Add metadata to track processing approach
        processing_metadata = {
            '_fetch_timeframe': timeframe,
            '_fetch_timestamp': time.time(),
            '_processing_approach': 'hybrid_batch_selective',
            '_priority_tokens_enhanced': len(priority_tokens) if priority_tokens else 0,
            '_total_tokens': len(result),
            '_includes_price_history': include_price_history
        }
        
        # Add metadata to each token
        for token_data in result:
            if isinstance(token_data, dict):
                token_data.update(processing_metadata)
        
        logger.logger.info(f"✅ Returning {len(result)} validated market data entries with hybrid enhancement")
        return result
    
    def select_priority_tokens(self, market_data: List[Dict[str, Any]], max_tokens: int = 4, 
                            selection_strategy: str = "adaptive") -> List[str]:
        """
        Intelligently select priority tokens for enhanced analysis with price history
        
        Implements a balanced approach that ensures all tokens get analyzed fairly
        while still prioritizing high-scoring tokens for trading opportunities
        
        Args:
            market_data: List of market data from batch call
            max_tokens: Maximum number of tokens to select (free tier constraint)
            selection_strategy: Strategy for selection ("adaptive", "volume", "momentum", "volatility")
        
        Returns:
            List of token IDs selected for priority analysis
        """
        if not market_data or not isinstance(market_data, list):
            logger.logger.warning("⚠️ Invalid market data for priority selection")
            return []
        
        if max_tokens <= 0:
            return []
        
        logger.logger.info(f"🎯 Selecting {max_tokens} priority tokens using balanced '{selection_strategy}' strategy")
        
        # ================================================================
        # 📊 STEP 1: ANALYZE AND SCORE ALL TOKENS
        # ================================================================
        
        scored_tokens = []
        
        # Extract all available token IDs for tracking
        available_token_ids = []
        token_id_to_symbol = {}  # For logging
        
        for token in market_data:
            if not isinstance(token, dict):
                continue
                
            token_id = token.get('id')
            if token_id:
                available_token_ids.append(token_id)
                token_id_to_symbol[token_id] = token.get('symbol', '').upper()
        
        # Get analysis history from cache
        analysis_history_key = 'token_analysis_history'
        token_last_analyzed = self._get_from_cache(analysis_history_key) or {}
        current_time = time.time()
        
        # Initialize missing tokens with timestamp 0 (never analyzed)
        for token_id in available_token_ids:
            if token_id not in token_last_analyzed:
                token_last_analyzed[token_id] = 0
        
        # Calculate scores for all tokens (same as original method)
        for token in market_data:
            if not isinstance(token, dict):
                continue
                
            token_id = token.get('id')
            if not token_id:
                continue
            
            try:
                # Extract key metrics for scoring
                current_price = float(token.get('current_price', 0))
                volume_24h = float(token.get('total_volume', 0))
                market_cap = float(token.get('market_cap', 0))
                price_change_24h = float(token.get('price_change_percentage_24h', 0))
                price_change_1h = float(token.get('price_change_percentage_1h_in_currency', 0))
                price_change_7d = float(token.get('price_change_percentage_7d_in_currency', 0))
                market_cap_rank = int(token.get('market_cap_rank', 999))
                
                # Skip tokens with insufficient data
                if current_price <= 0 or volume_24h <= 0:
                    continue
                
                # Calculate scores (same as original)
                volume_score = min(100, (volume_24h / 1_000_000_000) * 20)
                
                momentum_1h = abs(price_change_1h) * 10
                momentum_24h = abs(price_change_24h) * 2  
                momentum_7d = abs(price_change_7d) * 0.5
                momentum_score = min(100, momentum_1h + momentum_24h + momentum_7d)
                
                volatility_components = [price_change_1h, price_change_24h, price_change_7d]
                volatility_range = max(volatility_components) - min(volatility_components)
                volatility_score = min(100, volatility_range * 2)
                
                if market_cap_rank <= 10:
                    stability_score = 80
                elif market_cap_rank <= 50:
                    stability_score = 60
                elif market_cap_rank <= 100:
                    stability_score = 40
                else:
                    stability_score = 20
                
                liquidity_score = min(100, (market_cap / 100_000_000) * 10)
                
                # Apply scoring strategy (same as original)
                if selection_strategy == "volume":
                    final_score = volume_score * 0.7 + liquidity_score * 0.3
                elif selection_strategy == "momentum":
                    final_score = momentum_score * 0.5 + volume_score * 0.3 + volatility_score * 0.2
                elif selection_strategy == "volatility":
                    final_score = volatility_score * 0.4 + momentum_score * 0.3 + volume_score * 0.3
                else:  # "adaptive" or any other
                    final_score = (
                        volume_score * 0.25 +
                        momentum_score * 0.25 +
                        volatility_score * 0.20 +
                        stability_score * 0.15 +
                        liquidity_score * 0.15
                    )
                
                # Get time since last analysis and add to scoring data
                last_analyzed = token_last_analyzed.get(token_id, 0)
                hours_since_analyzed = (current_time - last_analyzed) / 3600 if last_analyzed > 0 else 24 * 7  # Default to 1 week
                
                # Store scoring data
                scored_tokens.append({
                    'token_id': token_id,
                    'symbol': token.get('symbol', '').upper(),
                    'final_score': final_score,
                    'volume_score': volume_score,
                    'momentum_score': momentum_score,
                    'volatility_score': volatility_score,
                    'stability_score': stability_score,
                    'liquidity_score': liquidity_score,
                    'last_analyzed': last_analyzed,
                    'hours_since_analyzed': hours_since_analyzed
                })
                
            except (ValueError, TypeError, KeyError) as e:
                logger.logger.debug(f"⚠️ Error scoring token {token_id}: {str(e)}")
                continue
        
        # ================================================================
        # 🏆 STEP 2: BALANCED SELECTION APPROACH
        # ================================================================
        
        if not scored_tokens:
            logger.logger.warning("⚠️ No tokens could be scored for priority selection")
            return []
        
        priority_token_ids = []
        
        # 1. Sort tokens by age (hours since last analyzed) - oldest first
        aged_tokens = sorted(scored_tokens, key=lambda x: x['hours_since_analyzed'], reverse=True)
        
        # 2. Always include the oldest token to ensure full coverage over time
        if aged_tokens:
            oldest_token = aged_tokens[0]
            priority_token_ids.append(oldest_token['token_id'])
            logger.logger.info(f"Including oldest token: {oldest_token['symbol']} (last analyzed {oldest_token['hours_since_analyzed']:.1f} hours ago)")
        
        # 3. Sort remaining tokens by final score (highest first)
        remaining_tokens = [t for t in scored_tokens if t['token_id'] not in priority_token_ids]
        remaining_tokens.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 4. Fill remaining slots with highest-scoring tokens
        remaining_slots = max_tokens - len(priority_token_ids)
        for token in remaining_tokens[:remaining_slots]:
            priority_token_ids.append(token['token_id'])
        
        # ================================================================
        # 📝 STEP 3: UPDATE TOKEN ANALYSIS HISTORY
        # ================================================================
        
        # Record that these tokens are about to be analyzed
        for token_id in priority_token_ids:
            token_last_analyzed[token_id] = current_time
        
        # Update cache
        self._add_to_cache(analysis_history_key, token_last_analyzed)
        
        # ================================================================
        # 📊 STEP 4: LOG SELECTION RESULTS
        # ================================================================
        
        # Prepare full selection details for logging
        selected_tokens = []
        for token_id in priority_token_ids:
            token_data = next((t for t in scored_tokens if t['token_id'] == token_id), None)
            if token_data:
                selected_tokens.append(token_data)
        
        logger.logger.info(f"🎯 BALANCED TOKEN SELECTION COMPLETE:")
        logger.logger.info(f"   Strategy: {selection_strategy} with rotation")
        logger.logger.info(f"   Tokens analyzed: {len(scored_tokens)}")
        logger.logger.info(f"   Tokens selected: {len(priority_token_ids)}")
        
        for i, token in enumerate(selected_tokens, 1):
            hours_ago = token['hours_since_analyzed']
            time_str = f"(last analyzed: {hours_ago:.1f}h ago)" if hours_ago < 24*7 else "(never analyzed)"
            
            logger.logger.info(
                f"   #{i} {token['symbol']} (Score: {token['final_score']:.1f}) - "
                f"Vol: {token['volume_score']:.0f}, Mom: {token['momentum_score']:.0f}, "
                f"Vola: {token['volatility_score']:.0f} {time_str}"
            )
        
        # Store selection metadata for analysis
        selection_metadata = {
            'selection_timestamp': current_time,
            'selection_strategy': f"{selection_strategy}_balanced",
            'total_candidates': len(scored_tokens),
            'selected_count': len(priority_token_ids)
        }
        
        # Cache the selection for potential reuse
        cache_key = f"priority_selection_{selection_strategy}_{max_tokens}"
        self._add_to_cache(cache_key, {
            'token_ids': priority_token_ids,
            'metadata': selection_metadata
        })
        
        return priority_token_ids
    
    def get_market_data_batched(self, token_ids: List[str], batch_size: int = 30, timeframe: str = "24h", 
                            priority_tokens: Optional[List[str]] = None, include_price_history: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Get market data for many tokens in batches with hybrid approach integration
        
        Args:
            token_ids: List of CoinGecko token IDs
            batch_size: Maximum number of tokens per request (reduced to 30 for free tier)
            timeframe: Analysis timeframe ("1h", "24h", "7d") - determines price change parameters
            priority_tokens: List of token IDs to fetch detailed price history for
            include_price_history: Whether to fetch price history for priority tokens
            
        Returns:
            Combined list of VALIDATED market data entries with hybrid enhancement
        """
        if not token_ids:
            return []
        
        # Check quota before starting batch operation
        quota_status = self.quota_tracker.get_quota_status()
        estimated_requests = (len(token_ids) + batch_size - 1) // batch_size
        
        # Add extra quota buffer for priority token price history calls
        extra_calls_needed = len(priority_tokens) if (include_price_history and priority_tokens) else 0
        total_estimated_calls = estimated_requests + extra_calls_needed
        
        if quota_status['daily_remaining'] < total_estimated_calls + 10:  # +10 safety buffer
            logger.logger.error(f"🚫 BATCH BLOCKED: Need {total_estimated_calls} requests, only {quota_status['daily_remaining']} remaining")
            return None
        
        logger.logger.info(f"📡 Starting batch operation: {len(token_ids)} tokens, {estimated_requests} batches, {extra_calls_needed} priority enhancements")
        
        all_data = []
        successful_batches = 0
        
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i+batch_size]
            batch_ids = ','.join(batch)
            
            # Determine timeframe-appropriate price change parameters
            if timeframe == "1h":
                price_change_param = "1h,24h"
            elif timeframe == "24h":
                price_change_param = "1h,24h,7d"
            elif timeframe == "7d":
                price_change_param = "24h,7d,30d"
            else:
                price_change_param = "1h,24h,7d"  # Default
            
            params = {
                "vs_currency": "usd",
                "ids": batch_ids,
                "order": "market_cap_desc",
                "per_page": len(batch),
                "page": 1,
                "sparkline": True,
                "price_change_percentage": price_change_param
            }
            
            # Determine which priority tokens are in this batch
            batch_priority_tokens = []
            if priority_tokens:
                batch_priority_tokens = [token_id for token_id in priority_tokens if token_id in batch]
            
            # Call the hybrid get_market_data method with batch-specific parameters
            batch_data = self.get_market_data(
                params=params,
                timeframe=timeframe,
                priority_tokens=batch_priority_tokens,
                include_price_history=include_price_history
            )
            
            if batch_data:
                all_data.extend(batch_data)
                successful_batches += 1
                
                # Log batch results with enhancement info - safe dictionary access
                enhanced_count = 0
                for token in batch_data:
                    if isinstance(token, dict) and token.get('_enhanced_with_history', False):
                        enhanced_count += 1
                
                logger.logger.info(f"✅ Batch {i//batch_size + 1}: {len(batch_data)} entries, {enhanced_count} enhanced with price history")
            else:
                logger.logger.error(f"❌ Batch {i//batch_size + 1} failed validation")
                # Don't continue if we start getting validation failures
                if successful_batches == 0:
                    return None
        
        if not all_data:
            logger.logger.error("🚫 TRADING HALT: No validated data from any batch")
            return None
        
        # Calculate total enhancements across all batches - safe dictionary access
        total_enhanced = 0
        for token in all_data:
            if isinstance(token, dict) and token.get('_enhanced_with_history', False):
                total_enhanced += 1
        
        logger.logger.info(f"✅ Batch operation complete: {len(all_data)} total entries from {successful_batches} batches")
        logger.logger.info(f"🎯 Priority enhancements: {total_enhanced} tokens enhanced with detailed price history")
        
        return all_data
    
    def manage_price_history_cache(self, operation: str = "cleanup", token_id: Optional[str] = None, 
                                cache_duration_hours: int = 6) -> Dict[str, Any]:
        """
        Intelligent cache management for price history data to maximize API efficiency
        
        Manages cache strategically to minimize API calls while ensuring data freshness
        for advanced trading analysis
        
        Args:
            operation: Operation to perform ("cleanup", "check", "invalidate", "stats", "optimize")
            token_id: Specific token to operate on (None for all tokens)
            cache_duration_hours: Cache duration in hours (default 6 hours for price history)
        
        Returns:
            Dictionary with operation results and cache statistics
        """
        current_time = time.time()
        cache_duration_seconds = cache_duration_hours * 3600
        
        if operation == "cleanup":
            return self._cleanup_expired_cache(current_time, cache_duration_seconds)
        elif operation == "check":
            return self._check_cache_status(token_id, current_time, cache_duration_seconds)
        elif operation == "invalidate":
            return self._invalidate_cache(token_id)
        elif operation == "stats":
            return self._get_cache_statistics(current_time, cache_duration_seconds)
        elif operation == "optimize":
            return self._optimize_cache_for_trading(current_time, cache_duration_seconds)
        else:
            logger.logger.error(f"❌ Unknown cache operation: {operation}")
            return {"error": f"Unknown operation: {operation}"}

    def _cleanup_expired_cache(self, current_time: float, cache_duration_seconds: float) -> Dict[str, Any]:
        """Clean up expired cache entries to free memory"""
        try:
            initial_cache_size = len(self.cache)
            expired_keys = []
            price_history_keys_removed = 0
            
            for cache_key, cache_entry in list(self.cache.items()):
                cache_time = cache_entry.get('timestamp', 0)
                age_seconds = current_time - cache_time
                
                # Different expiry rules for different data types
                if 'market_chart' in cache_key:
                    # Price history data - use longer cache duration
                    if age_seconds >= cache_duration_seconds:
                        expired_keys.append(cache_key)
                        price_history_keys_removed += 1
                elif 'coins/markets' in cache_key:
                    # Basic market data - shorter cache duration (1 hour)
                    if age_seconds >= 3600:  # 1 hour
                        expired_keys.append(cache_key)
                else:
                    # Other data - standard cache duration
                    if age_seconds >= self.cache_duration:
                        expired_keys.append(cache_key)
            
            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
            
            final_cache_size = len(self.cache)
            memory_freed = initial_cache_size - final_cache_size
            
            logger.logger.info(f"🧹 Cache cleanup complete: {memory_freed} entries removed ({price_history_keys_removed} price history)")
            
            return {
                "operation": "cleanup",
                "success": True,
                "entries_removed": memory_freed,
                "price_history_removed": price_history_keys_removed,
                "cache_size_before": initial_cache_size,
                "cache_size_after": final_cache_size,
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.logger.error(f"❌ Cache cleanup failed: {str(e)}")
            return {"operation": "cleanup", "success": False, "error": str(e)}

    def _check_cache_status(self, token_id: Optional[str], current_time: float, 
                        cache_duration_seconds: float) -> Dict[str, Any]:
        """Check cache status for specific token or all tokens"""
        try:
            if token_id:
                # Check specific token
                return self._check_single_token_cache(token_id, current_time, cache_duration_seconds)
            else:
                # Check all tokens
                return self._check_all_tokens_cache(current_time, cache_duration_seconds)
                
        except Exception as e:
            logger.logger.error(f"❌ Cache status check failed: {str(e)}")
            return {"operation": "check", "success": False, "error": str(e)}

    def _check_single_token_cache(self, token_id: str, current_time: float, 
                                cache_duration_seconds: float) -> Dict[str, Any]:
        """Check cache status for a specific token"""
        cache_info = {
            "token_id": token_id,
            "has_basic_data": False,
            "has_price_history": False,
            "basic_data_age_hours": None,
            "price_history_age_hours": None,
            "basic_data_fresh": False,
            "price_history_fresh": False,
            "cache_keys_found": []
        }
        
        # Check for basic market data
        for cache_key in self.cache:
            if 'coins/markets' in cache_key and token_id in cache_key:
                cache_entry = self.cache[cache_key]
                cache_time = cache_entry.get('timestamp', 0)
                age_seconds = current_time - cache_time
                age_hours = age_seconds / 3600
                
                cache_info["has_basic_data"] = True
                cache_info["basic_data_age_hours"] = round(age_hours, 2)
                cache_info["basic_data_fresh"] = age_seconds < 3600  # 1 hour freshness
                cache_info["cache_keys_found"].append(cache_key)
                break
        
        # Check for price history data
        price_history_key = f"coins/{token_id}/market_chart"
        for cache_key in self.cache:
            if price_history_key in cache_key:
                cache_entry = self.cache[cache_key]
                cache_time = cache_entry.get('timestamp', 0)
                age_seconds = current_time - cache_time
                age_hours = age_seconds / 3600
                
                cache_info["has_price_history"] = True
                cache_info["price_history_age_hours"] = round(age_hours, 2)
                cache_info["price_history_fresh"] = age_seconds < cache_duration_seconds
                cache_info["cache_keys_found"].append(cache_key)
                break
        
        return {
            "operation": "check",
            "success": True,
            "cache_info": cache_info,
            "timestamp": current_time
        }

    def _check_all_tokens_cache(self, current_time: float, cache_duration_seconds: float) -> Dict[str, Any]:
        """Check cache status for all tokens"""
        basic_data_cached = 0
        price_history_cached = 0
        fresh_basic_data = 0
        fresh_price_history = 0
        
        for cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            cache_time = cache_entry.get('timestamp', 0)
            age_seconds = current_time - cache_time
            
            if 'coins/markets' in cache_key:
                basic_data_cached += 1
                if age_seconds < 3600:  # 1 hour freshness
                    fresh_basic_data += 1
            elif 'market_chart' in cache_key:
                price_history_cached += 1
                if age_seconds < cache_duration_seconds:
                    fresh_price_history += 1
        
        return {
            "operation": "check",
            "success": True,
            "summary": {
                "basic_data_cached": basic_data_cached,
                "price_history_cached": price_history_cached,
                "fresh_basic_data": fresh_basic_data,
                "fresh_price_history": fresh_price_history,
                "total_cache_entries": len(self.cache)
            },
            "timestamp": current_time
        }

    def _invalidate_cache(self, token_id: Optional[str]) -> Dict[str, Any]:
        """Invalidate cache entries for specific token or all cache"""
        try:
            if token_id:
                # Invalidate specific token
                keys_to_remove = []
                for cache_key in self.cache:
                    if token_id in cache_key:
                        keys_to_remove.append(cache_key)
                
                for key in keys_to_remove:
                    del self.cache[key]
                
                logger.logger.info(f"🔄 Invalidated {len(keys_to_remove)} cache entries for {token_id}")
                
                return {
                    "operation": "invalidate",
                    "success": True,
                    "token_id": token_id,
                    "entries_invalidated": len(keys_to_remove),
                    "timestamp": time.time()
                }
            else:
                # Invalidate all cache
                initial_size = len(self.cache)
                self.cache.clear()
                
                logger.logger.info(f"🔄 Invalidated entire cache: {initial_size} entries removed")
                
                return {
                    "operation": "invalidate",
                    "success": True,
                    "entries_invalidated": initial_size,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.logger.error(f"❌ Cache invalidation failed: {str(e)}")
            return {"operation": "invalidate", "success": False, "error": str(e)}

    def _get_cache_statistics(self, current_time: float, cache_duration_seconds: float) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            stats = {
                "total_entries": len(self.cache),
                "basic_market_data": 0,
                "price_history_data": 0,
                "other_data": 0,
                "fresh_entries": 0,
                "stale_entries": 0,
                "memory_usage_estimate": 0,
                "oldest_entry_age_hours": 0.0,
                "newest_entry_age_hours": 0.0,
                "average_entry_age_hours": 0.0,
                "cache_hit_efficiency": 0.0
            }
            
            entry_ages = []
            
            for cache_key, cache_entry in self.cache.items():
                cache_time = cache_entry.get('timestamp', 0)
                age_seconds = current_time - cache_time
                age_hours = age_seconds / 3600
                entry_ages.append(age_hours)
                
                # Categorize entries
                if 'coins/markets' in cache_key:
                    stats["basic_market_data"] += 1
                    if age_seconds < 3600:  # 1 hour freshness for basic data
                        stats["fresh_entries"] += 1
                    else:
                        stats["stale_entries"] += 1
                elif 'market_chart' in cache_key:
                    stats["price_history_data"] += 1
                    if age_seconds < cache_duration_seconds:  # Custom freshness for price history
                        stats["fresh_entries"] += 1
                    else:
                        stats["stale_entries"] += 1
                else:
                    stats["other_data"] += 1
                    if age_seconds < self.cache_duration:
                        stats["fresh_entries"] += 1
                    else:
                        stats["stale_entries"] += 1
                
                # Estimate memory usage (rough approximation)
                try:
                    import sys
                    stats["memory_usage_estimate"] += sys.getsizeof(cache_entry)
                except:
                    stats["memory_usage_estimate"] += 1000  # Rough estimate
            
            # Calculate age statistics with proper type handling
            if entry_ages:
                stats["oldest_entry_age_hours"] = float(round(max(entry_ages), 2))
                stats["newest_entry_age_hours"] = float(round(min(entry_ages), 2))
                stats["average_entry_age_hours"] = float(round(sum(entry_ages) / len(entry_ages), 2))
            
            # Calculate cache efficiency with proper type handling
            total_requests = int(getattr(self, 'daily_requests', 1))  # Ensure int type
            cache_requests = int(stats["total_entries"])  # Ensure int type
            
            if total_requests > 0:
                efficiency_calculation = (cache_requests / total_requests) * 100
                stats["cache_hit_efficiency"] = float(round(efficiency_calculation, 2))
            else:
                stats["cache_hit_efficiency"] = 0.0
            
            return {
                "operation": "stats",
                "success": True,
                "statistics": stats,
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.logger.error(f"❌ Cache statistics failed: {str(e)}")
            return {"operation": "stats", "success": False, "error": str(e)}

    def _optimize_cache_for_trading(self, current_time: float, cache_duration_seconds: float) -> Dict[str, Any]:
        """Optimize cache specifically for trading efficiency"""
        try:
            optimizations_applied = []
            
            # 1. Clean up expired entries
            cleanup_result = self._cleanup_expired_cache(current_time, cache_duration_seconds)
            if cleanup_result["success"]:
                optimizations_applied.append(f"Cleaned {cleanup_result['entries_removed']} expired entries")
            
            # 2. Prioritize price history cache for active trading tokens
            priority_tokens = ['bitcoin', 'ethereum', 'solana', 'uniswap', 'ripple']
            priority_preserved = 0
            
            for cache_key in list(self.cache.keys()):
                if 'market_chart' in cache_key:
                    # Check if this is a priority token
                    is_priority = any(token in cache_key for token in priority_tokens)
                    if is_priority:
                        # Extend cache duration for priority tokens
                        cache_entry = self.cache[cache_key]
                        cache_entry['priority_preserved'] = True
                        priority_preserved += 1
            
            if priority_preserved > 0:
                optimizations_applied.append(f"Priority-preserved {priority_preserved} price history entries")
            
            # 3. Remove redundant basic market data if price history exists
            redundant_removed = 0
            for cache_key in list(self.cache.keys()):
                if 'coins/markets' in cache_key:
                    # Check if we have corresponding price history
                    has_price_history = any('market_chart' in key for key in self.cache.keys())
                    if has_price_history:
                        cache_entry = self.cache[cache_key]
                        cache_time = cache_entry.get('timestamp', 0)
                        age_seconds = current_time - cache_time
                        
                        # Remove old basic data if we have fresh price history
                        if age_seconds > 1800:  # Older than 30 minutes
                            del self.cache[cache_key]
                            redundant_removed += 1
            
            if redundant_removed > 0:
                optimizations_applied.append(f"Removed {redundant_removed} redundant basic data entries")
            
            logger.logger.info(f"🎯 Cache optimization complete: {', '.join(optimizations_applied)}")
            
            return {
                "operation": "optimize",
                "success": True,
                "optimizations_applied": optimizations_applied,
                "final_cache_size": len(self.cache),
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.logger.error(f"❌ Cache optimization failed: {str(e)}")
            return {"operation": "optimize", "success": False, "error": str(e)}
    
    def get_coin_detail(self, coin_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed data for a specific coin with validation
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            
        Returns:
            Validated detailed coin data or None
        """
        endpoint = f"coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "true"
        }
        
        result = self.get_with_cache(endpoint, params)
        
        # Validate the detailed data
        if result and isinstance(result, dict):
            if 'market_data' in result and 'current_price' in result['market_data']:
                try:
                    current_price = result['market_data']['current_price'].get('usd')
                    is_valid, _ = self.validator.validate_price_data(current_price)
                    if is_valid:
                        return result
                    else:
                        logger.logger.error(f"❌ Invalid price data for {coin_id}")
                        return None
                except:
                    logger.logger.error(f"❌ Malformed market data for {coin_id}")
                    return None
        
        logger.logger.error(f"❌ No valid detailed data for {coin_id}")
        return None

    def get_coin_ohlc(self, coin_id: str, days: int = 1) -> Optional[List[List[float]]]:
        """Get OHLC data for a specific coin with validation"""
        if days not in [1, 7, 14, 30, 90, 180, 365]:
            days = 1
            
        endpoint = f"coins/{coin_id}/ohlc"
        url = f"{self.base_url}/{endpoint}"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        
        # Make direct request instead of using get_with_cache (which expects dict responses)
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                result = response.json()
            else:
                logger.logger.error(f"❌ OHLC API failed: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.logger.error(f"❌ OHLC request failed: {e}")
            return None
        
        # Your existing validation logic works perfectly
        if result and isinstance(result, list):
            validated_ohlc = []
            for candle in result:
                if len(candle) == 5:  # [timestamp, open, high, low, close]
                    try:
                        timestamp = float(candle[0])
                        open_price = float(candle[1])
                        high_price = float(candle[2])
                        low_price = float(candle[3])
                        close_price = float(candle[4])
                        
                        # Validate OHLC logic
                        if (open_price > 0 and high_price > 0 and low_price > 0 and close_price > 0 and
                            high_price >= max(open_price, close_price) and
                            low_price <= min(open_price, close_price)):
                            validated_ohlc.append([timestamp, open_price, high_price, low_price, close_price])
                    except (ValueError, TypeError):
                        continue
            
            if len(validated_ohlc) >= len(result) * 0.8:  # Require 80% valid candles
                return validated_ohlc
        
        logger.logger.error(f"❌ OHLC data validation failed for {coin_id}")
        return None
    
    def find_token_id(self, token_symbol: str) -> Optional[str]:
        """
        Find the exact CoinGecko ID for a token by symbol
        Now using database lookup instead of hardcoded mappings
        
        Args:
            token_symbol: Token symbol to look up (e.g. 'BTC')
            
        Returns:
            CoinGecko ID (e.g. 'bitcoin') or None if not found
        """
        # Check in-memory cache first for performance
        token_symbol_lower = token_symbol.lower()
        
        # Check cache
        if token_symbol_lower in self.token_id_cache:
            return self.token_id_cache[token_symbol_lower]
        
        # Not in cache, try to find in database
        try:
            # Use the database module import at function level, not at method level
            # This prevents circular import issues
            import sys
            if 'database' in sys.modules:
                # Database module is already imported, we can use it
                db_module = sys.modules['database']
                db = db_module.CryptoDatabase()
                conn, cursor = db._get_connection()
                
                cursor.execute("""
                    SELECT coin_id FROM coingecko_market_data 
                    WHERE LOWER(symbol) = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (token_symbol_lower,))
                
                result = cursor.fetchone()
                if result and result['coin_id']:
                    logger.logger.info(f"Found {token_symbol} with ID: {result['coin_id']} (from database)")
                    # Add to cache
                    self.token_id_cache[token_symbol_lower] = result['coin_id']
                    return result['coin_id']
        except Exception as e:
            logger.logger.warning(f"Database lookup for {token_symbol} failed: {str(e)}")
        
        # Not found in database, fall back to API call
        endpoint = "coins/list"
        coins_list = self.get_with_cache(endpoint)
        
        if not coins_list:
            return None
        
        # First try exact match on symbol
        for coin in coins_list:
            if coin.get('symbol', '').lower() == token_symbol_lower:
                logger.logger.info(f"Found {token_symbol} with ID: {coin['id']} (from API)")
                # Add to cache
                self.token_id_cache[token_symbol_lower] = coin['id']
                return coin['id']
        
        # If not found, try partial name match
        for coin in coins_list:
            if token_symbol_lower in coin.get('name', '').lower():
                logger.logger.info(f"Found possible {token_symbol} match with ID: {coin['id']} (name: {coin['name']})")
                # Add to cache
                self.token_id_cache[token_symbol_lower] = coin['id']
                return coin['id']
        
        logger.logger.error(f"Could not find {token_symbol} in CoinGecko coin list")
        return None
    
    def get_multiple_tokens_by_symbol(self, symbols: List[str]) -> Dict[str, str]:
        """
        Get CoinGecko IDs for multiple token symbols
        (Enhanced with quota checking)
        """
        # Common token mappings for quick lookups
        common_mappings = {
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
        
        # Initialize result with common mappings
        result = {}
        symbols_to_fetch = []
        
        for symbol in symbols:
            symbol_lower = symbol.lower()
            
            # Check common mappings first
            if symbol_lower in common_mappings:
                result[symbol] = common_mappings[symbol_lower]
                continue
                
            # Check cache next
            if symbol_lower in self.token_id_cache:
                result[symbol] = self.token_id_cache[symbol_lower]
                continue
                
            # Need to fetch from API
            symbols_to_fetch.append(symbol)
        
        if not symbols_to_fetch:
            return result
        
        # Check quota before fetching coins list
        quota_status = self.quota_tracker.get_quota_status()
        if quota_status['daily_remaining'] <= 5:
            logger.logger.error("🚫 Cannot fetch token IDs: quota too low")
            # Return partial results
            for symbol in symbols_to_fetch:
                result[symbol] = None
            return result
        
        # Fetch coins list once for all missing symbols
        endpoint = "coins/list"
        coins_list = self.get_with_cache(endpoint)
        
        if not coins_list:
            # Return partial results
            for symbol in symbols_to_fetch:
                result[symbol] = None
            return result
        
        # Create lookup dictionary from coins list
        symbol_to_id = {}
        for coin in coins_list:
            coin_symbol = coin.get('symbol', '').lower()
            if coin_symbol and coin_symbol not in symbol_to_id:
                symbol_to_id[coin_symbol] = coin['id']
        
        # Assign found IDs to results and update cache
        for symbol in symbols_to_fetch:
            symbol_lower = symbol.lower()
            if symbol_lower in symbol_to_id:
                result[symbol] = symbol_to_id[symbol_lower]
                self.token_id_cache[symbol_lower] = symbol_to_id[symbol_lower]
                logger.logger.debug(f"Found {symbol} with ID: {symbol_to_id[symbol_lower]}")
            else:
                logger.logger.warning(f"Could not find ID for {symbol}")
                result[symbol] = None
        
        return result
    
    def get_trending_tokens(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get trending tokens from CoinGecko with validation
        """
        endpoint = "search/trending"
        result = self.get_with_cache(endpoint)
        
        if result and 'coins' in result:
            # Validate trending data
            validated_trending = []
            for coin_data in result['coins']:
                if isinstance(coin_data, dict) and 'item' in coin_data:
                    item = coin_data['item']
                    if 'id' in item and 'symbol' in item and 'name' in item:
                        validated_trending.append(coin_data)
            
            if validated_trending:
                logger.logger.info(f"✅ Validated {len(validated_trending)} trending tokens")
                return validated_trending
        
        logger.logger.error("❌ No valid trending data available")
        return None
    
    def check_token_exists(self, token_id: str) -> bool:
        """
        Check if a token ID exists in CoinGecko (with quota awareness)
        """
        # Check quota first
        quota_status = self.quota_tracker.get_quota_status()
        if quota_status['daily_remaining'] <= 5:
            logger.logger.warning(f"🚫 Cannot check token existence: quota too low")
            return False
        
        endpoint = f"coins/{token_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        }
        
        try:
            # Use a direct request with minimal data to check existence
            if not self._enforce_rate_limit():
                return False
            
            url = f"{self.base_url}/{endpoint}"
            
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            self.last_request_time = time.time()
            self.daily_requests += 1
            
            success = response.status_code == 200
            self.quota_tracker.record_request(success)
            
            return success
            
        except Exception as e:
            logger.logger.error(f"Error checking if {token_id} exists: {str(e)}")
            self.quota_tracker.record_request(False)
            return False
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive API and validation statistics
        """
        self._clean_cache()
        quota_status = self.quota_tracker.get_quota_status()
        
        return {
            'daily_requests': self.daily_requests,
            'failed_requests': self.failed_requests,
            'cache_size': len(self.cache),
            'token_id_cache_size': len(self.token_id_cache),
            'quota_status': quota_status,
            'data_quality': self.data_quality_stats,
            'validation_success_rate': (
                self.data_quality_stats['valid_requests'] / 
                max(1, self.data_quality_stats['valid_requests'] + self.data_quality_stats['invalid_data_responses'])
            ) * 100,
            'cache_hit_rate': (
                self.data_quality_stats['cache_hits'] / 
                max(1, self.data_quality_stats['cache_hits'] + self.data_quality_stats['cache_misses'])
            ) * 100
        }
    
    def optimize_for_multiple_tokens(self, tokens: List[str]) -> bool:
        """
        Optimize handler for a specific list of tokens (with quota awareness)
        """
        try:
            # Check quota before optimization
            quota_status = self.quota_tracker.get_quota_status()
            estimated_requests = 2  # Token IDs + market data
            
            if quota_status['daily_remaining'] < estimated_requests + 20:  # +20 safety buffer
                logger.logger.error(f"🚫 Cannot optimize: need {estimated_requests} requests, only {quota_status['daily_remaining']} remaining")
                return False
            
            # Pre-fetch and cache token IDs
            token_ids = self.get_multiple_tokens_by_symbol(tokens)
            
            # Pre-fetch market data for valid IDs only
            valid_ids = [id for id in token_ids.values() if id]
            if valid_ids:
                market_data = self.get_market_data_batched(valid_ids)
                if market_data:
                    logger.logger.info(f"✅ Pre-cached validated data for {len(market_data)} tokens")
                    return True
                else:
                    logger.logger.error("❌ Failed to pre-cache market data")
                    return False
            
            logger.logger.warning("⚠️ No valid token IDs found for optimization")
            return False
            
        except Exception as e:
            logger.logger.error(f"💥 Error optimizing for multiple tokens: {str(e)}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status for trading decisions
        """
        stats = self.get_request_stats()
        quota_status = stats['quota_status']
        
        # Determine system health
        validation_rate = stats['validation_success_rate']
        cache_hit_rate = stats['cache_hit_rate']
        quota_remaining_pct = (quota_status['daily_remaining'] / quota_status['daily_limit']) * 100
        
        # Health scoring
        health_score = 0
        health_issues = []
        
        # Validation health (40% of score)
        if validation_rate >= 95:
            health_score += 40
        elif validation_rate >= 80:
            health_score += 30
            health_issues.append(f"Validation rate below 95%: {validation_rate:.1f}%")
        elif validation_rate >= 60:
            health_score += 20
            health_issues.append(f"Poor validation rate: {validation_rate:.1f}%")
        else:
            health_issues.append(f"CRITICAL: Validation rate: {validation_rate:.1f}%")
        
        # Quota health (30% of score)
        if quota_remaining_pct >= 50:
            health_score += 30
        elif quota_remaining_pct >= 25:
            health_score += 20
            health_issues.append(f"Quota below 50%: {quota_remaining_pct:.1f}%")
        elif quota_remaining_pct >= 10:
            health_score += 10
            health_issues.append(f"LOW QUOTA: {quota_remaining_pct:.1f}%")
        else:
            health_issues.append(f"CRITICAL QUOTA: {quota_remaining_pct:.1f}%")
        
        # Cache efficiency (20% of score)
        if cache_hit_rate >= 70:
            health_score += 20
        elif cache_hit_rate >= 50:
            health_score += 15
        elif cache_hit_rate >= 30:
            health_score += 10
        else:
            health_issues.append(f"Poor cache efficiency: {cache_hit_rate:.1f}%")
        
        # Request success rate (10% of score)
        request_success_rate = (1 - (stats['failed_requests'] / max(1, stats['daily_requests']))) * 100
        if request_success_rate >= 95:
            health_score += 10
        elif request_success_rate >= 80:
            health_score += 7
        else:
            health_issues.append(f"High request failure rate: {100-request_success_rate:.1f}%")
        
        # Determine overall status
        if health_score >= 90:
            status = "EXCELLENT"
            trading_recommendation = "FULL_TRADING"
        elif health_score >= 70:
            status = "GOOD"
            trading_recommendation = "NORMAL_TRADING"
        elif health_score >= 50:
            status = "FAIR"
            trading_recommendation = "CAUTIOUS_TRADING"
        elif health_score >= 30:
            status = "POOR"
            trading_recommendation = "MINIMAL_TRADING"
        else:
            status = "CRITICAL"
            trading_recommendation = "HALT_TRADING"
        
        return {
            'status': status,
            'health_score': health_score,
            'trading_recommendation': trading_recommendation,
            'health_issues': health_issues,
            'detailed_stats': stats,
            'recommendations': self._get_health_recommendations(stats, health_issues),
            'last_check': datetime.now().isoformat()
        }
    
    def _get_health_recommendations(self, stats: Dict, issues: List[str]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        quota_status = stats['quota_status']
        
        if quota_status['daily_remaining'] < 50:
            recommendations.append("URGENT: Conserve API quota - reduce trading frequency")
        
        if stats['validation_success_rate'] < 80:
            recommendations.append("Investigate data quality issues with CoinGecko API")
        
        if stats['cache_hit_rate'] < 50:
            recommendations.append("Consider increasing cache duration to improve efficiency")
        
        if stats['failed_requests'] > 10:
            recommendations.append("Check network connectivity and API stability")
        
        if quota_status['success_rate_1h'] < 0.8:
            recommendations.append("Recent request failures detected - monitor API status")
        
        return recommendations
    
    def emergency_quota_conserve(self) -> None:
        """
        Emergency quota conservation mode
        Extends cache duration and increases rate limits
        """
        logger.logger.warning("🚨 EMERGENCY QUOTA CONSERVATION ACTIVATED")
        
        # Extend cache duration significantly
        self.cache_duration = 900  # 15 minutes
        
        # Increase rate limiting
        self.min_request_interval = 10.0  # 10 seconds
        
        # Reduce retry attempts
        self.max_retries = 1
        
        # Update quota tracker to be more conservative
        setattr(self.quota_tracker, 'daily_limit', 200)  # Very conservative
        
        logger.logger.warning("📊 Emergency settings: 15min cache, 5s intervals, 1 retry, 200 daily limit")
    
    def reset_to_normal_mode(self) -> None:
        """Reset to normal operating parameters"""
        logger.logger.info("🔄 Resetting to normal operation mode")
        
        self.cache_duration = 300  # 5 minutes
        self.min_request_interval = 60.0  # 60 seconds
        self.max_retries = 2
        setattr(self.quota_tracker, 'daily_limit', 400)
        
        logger.logger.info("✅ Normal mode restored")
    
    def should_halt_trading(self) -> Tuple[bool, str]:
        """
        Determine if trading should be halted due to data quality or quota issues
        
        Returns:
            (should_halt, reason)
        """
        health = self.get_system_health()
        
        # Critical halt conditions
        if health['trading_recommendation'] == 'HALT_TRADING':
            return True, f"System health critical: {health['status']}"
        
        quota_status = self.quota_tracker.get_quota_status()
        
        # Quota-based halt conditions
        if quota_status['daily_remaining'] <= 5:
            return True, f"API quota critical: {quota_status['daily_remaining']} requests remaining"
        
        # Data quality halt conditions
        if self.data_quality_stats['invalid_data_responses'] > self.data_quality_stats['valid_requests']:
            return True, "Data validation failure rate too high"
        
        # Recent failure rate check
        if quota_status['success_rate_1h'] < 0.5:
            return True, f"Recent API success rate too low: {quota_status['success_rate_1h']:.1%}"
        
        return False, "System healthy for trading"
    
    def __str__(self) -> str:
        """String representation of handler status"""
        stats = self.get_request_stats()
        quota = stats['quota_status']
        return (f"CoinGeckoHandler("
                f"health={self.get_system_health()['status']}, "
                f"quota={quota['daily_remaining']}/{quota['daily_limit']}, "
                f"cache_hit={stats['cache_hit_rate']:.1f}%, "
                f"validation={stats['validation_success_rate']:.1f}%)")
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"CoinGeckoHandler(base_url='{self.base_url}', "
                f"cache_duration={self.cache_duration}, "
                f"min_interval={self.min_request_interval}, "
                f"daily_limit={getattr(self.quota_tracker, 'daily_limit', 400)})")

# ============================================================================
# 🚀 ADDITIONAL UTILITY FUNCTIONS FOR GENERATIONAL WEALTH 🚀
# ============================================================================

def create_wealth_optimized_handler(base_url: str = "https://api.coingecko.com/api/v3") -> CoinGeckoHandler:
    """
    Create a CoinGecko handler optimized for generational wealth creation
    
    Returns:
        Fully configured handler with conservative settings
    """
    handler = CoinGeckoHandler(base_url, cache_duration=300)
    
    # Pre-optimize for common wealth-building tokens
    wealth_tokens = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'AVAX', 'DOT', 'UNI', 'AAVE']
    success = handler.optimize_for_multiple_tokens(wealth_tokens)
    
    if success:
        logger.logger.info("🏆 Wealth-optimized handler ready for generational trading")
    else:
        logger.logger.warning("⚠️ Could not pre-optimize handler - check quota status")
    
    return handler

def validate_trading_readiness(handler: CoinGeckoHandler) -> Dict[str, Any]:
    """
    Comprehensive validation of system readiness for trading
    
    Args:
        handler: CoinGeckoHandler instance
        
    Returns:
        Detailed readiness report
    """
    readiness_report = {
        'ready_for_trading': False,
        'health_status': None,
        'critical_issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    try:
        # Get system health
        health = handler.get_system_health()
        readiness_report['health_status'] = health
        
        # Check if trading should be halted
        should_halt, halt_reason = handler.should_halt_trading()
        
        if should_halt:
            readiness_report['critical_issues'].append(halt_reason)
        else:
            # Additional readiness checks
            stats = handler.get_request_stats()
            quota_status = stats['quota_status']
            
            # Quota checks
            if quota_status['daily_remaining'] < 100:
                readiness_report['warnings'].append(f"Low API quota: {quota_status['daily_remaining']} remaining")
            
            # Cache efficiency check
            if stats['cache_hit_rate'] < 30:
                readiness_report['warnings'].append(f"Poor cache efficiency: {stats['cache_hit_rate']:.1f}%")
            
            # Validation rate check
            if stats['validation_success_rate'] < 90:
                readiness_report['warnings'].append(f"Validation rate below optimal: {stats['validation_success_rate']:.1f}%")
            
            # Overall readiness determination
            if (health['trading_recommendation'] in ['FULL_TRADING', 'NORMAL_TRADING'] and
                quota_status['daily_remaining'] > 50 and
                stats['validation_success_rate'] > 70):
                readiness_report['ready_for_trading'] = True
        
        # Add recommendations
        readiness_report['recommendations'] = health['recommendations']
        
    except Exception as e:
        readiness_report['critical_issues'].append(f"Readiness check failed: {str(e)}")
    
    return readiness_report

# ============================================================================
# 🎯 MODULE EXPORTS AND METADATA 🎯
# ============================================================================

__all__ = [
    'CoinGeckoHandler',
    'CoinGeckoDataValidator', 
    'CoinGeckoQuotaTracker',
    'create_wealth_optimized_handler',
    'validate_trading_readiness'
]

__version__ = "3.0.0"
__author__ = "Generational Wealth Trading System"
__description__ = "Enhanced CoinGecko API handler with strict validation and quota management"

# Log module completion
logger.logger.info("🚀 Generational Wealth CoinGecko Handler v3.0 loaded successfully")
logger.logger.info("✅ Features: Strict validation, quota tracking, emergency modes, trading halt detection")
logger.logger.info("🎯 Ready for serious wealth generation with real data only")
