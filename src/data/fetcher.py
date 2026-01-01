"""
Data Fetcher for all NSE URLs
Handles rate limiting, retries, and caching
"""

import aiohttp
import asyncio
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import urllib.parse
import hashlib

from src.utils.logger import get_logger
from src.data.cache_manager import CacheManager

logger = get_logger(__name__)

class DataFetcher:
    """Main data fetcher for NSE APIs"""
    
    def __init__(self, config):
        self.config = config
        self.session = None
        self.cache = CacheManager(config)
        
        # Rate limiting
        self.request_timestamps = []
        self.request_count = 0
        self.daily_limit = config['fetcher']['rate_limiting']['max_requests_per_hour'] * 24
        
        # Headers and proxies
        self.user_agents = config['fetcher']['headers']['user_agents']
        self.default_headers = config['fetcher']['headers']['default_headers']
        
        # Circuit breaker
        self.circuit_open = False
        self.circuit_open_time = None
        self.failure_count = 0
        self.failure_threshold = 5
        
    async def initialize(self):
        """Initialize the fetcher"""
        try:
            # Create session with custom settings
            timeout = aiohttp.ClientTimeout(
                total=self.config['fetcher']['timeout']['total'],
                connect=self.config['fetcher']['timeout']['connect'],
                sock_read=self.config['fetcher']['timeout']['read']
            )
            
            connector = aiohttp.TCPConnector(
                limit=self.config['fetcher']['rate_limiting']['max_concurrent_requests'],
                ttl_dns_cache=300
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self._get_headers()
            )
            
            logger.info("DataFetcher initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DataFetcher: {e}")
            raise
    
    async def fetch_all(self) -> Dict[str, Any]:
        """
        Fetch data from all enabled sources
        
        Returns:
            Dictionary with data from all sources
        """
        try:
            logger.info("Fetching data from all sources...")
            
            # Check circuit breaker
            if self.circuit_open:
                if time.time() - self.circuit_open_time < 300:  # 5 minutes
                    logger.warning("Circuit breaker is open, using cached data")
                    return await self.get_cached_data()
                else:
                    self.circuit_open = False
                    self.failure_count = 0
            
            # Prepare fetch tasks
            fetch_tasks = []
            for source_name, source_config in self.config['data_sources'].items():
                if source_config.get('enabled', False):
                    task = self._fetch_source_async(source_name, source_config)
                    fetch_tasks.append(task)
            
            # Execute with concurrency limit
            semaphore = asyncio.Semaphore(
                self.config['fetcher']['rate_limiting']['max_concurrent_requests']
            )
            
            async def fetch_with_limit(task):
                async with semaphore:
                    return await task
            
            results = {}
            for task in fetch_tasks:
                try:
                    source_name, data = await fetch_with_limit(task)
                    if data is not None:
                        results[source_name] = data
                        # Update cache
                        await self.cache.set(source_name, data)
                except Exception as e:
                    logger.error(f"Failed to fetch {source_name}: {e}")
                    # Try to get from cache
                    cached_data = await self.cache.get(source_name)
                    if cached_data:
                        results[source_name] = cached_data
                        logger.info(f"Using cached data for {source_name}")
            
            # Add metadata
            results['metadata'] = {
                'fetch_timestamp': datetime.now().isoformat(),
                'sources_fetched': list(results.keys()),
                'request_count': self.request_count,
                'circuit_state': 'open' if self.circuit_open else 'closed'
            }
            
            logger.info(f"Fetched data from {len(results) - 1} sources")
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch all data: {e}")
            self._record_failure()
            return await self.get_cached_data()
    
    async def _fetch_source_async(self, source_name: str, source_config: Dict) -> tuple:
        """Async wrapper for fetching single source"""
        try:
            await self._check_rate_limit()
            
            url = self._build_url(source_name, source_config['url'])
            data = await self._make_request(url, source_name)
            
            self._record_success()
            return source_name, data
            
        except Exception as e:
            logger.error(f"Failed to fetch {source_name}: {e}")
            self._record_failure()
            raise
    
    def _build_url(self, source_name: str, base_url: str) -> str:
        """Build URL with parameters"""
        symbol = self.config['market']['symbols']['primary']
        
        if source_name == 'option_chain_v3':
            # Need to get expiry date
            expiry = self._get_next_expiry()
            return base_url.replace('{symbol}', symbol).replace('{expiry}', expiry)
        
        elif source_name == 'indices.index_chart':
            return base_url.replace('{symbol}', symbol)
        
        return base_url
    
    def _get_next_expiry(self) -> str:
        """Get next expiry date"""
        # This should be fetched dynamically, but for now use hardcoded
        today = datetime.now()
        
        # Nifty options expire on Thursday
        days_ahead = (3 - today.weekday()) % 7  # 3 = Thursday
        if days_ahead == 0:  # Today is Thursday
            days_ahead = 7
        
        expiry_date = today + timedelta(days=days_ahead)
        return expiry_date.strftime('%d-%b-%Y').upper()
    
    async def _make_request(self, url: str, source_name: str) -> Any:
        """Make HTTP request with retry logic"""
        max_retries = self.config['fetcher']['retry_policy']['max_retries']
        
        for attempt in range(max_retries):
            try:
                # Respect rate limit
                await self._check_rate_limit()
                
                # Add jitter
                if attempt > 0:
                    jitter = random.uniform(1, 3) * (2 ** attempt)
                    await asyncio.sleep(jitter)
                    logger.debug(f"Retry {attempt + 1} for {source_name}")
                
                # Make request
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.request_count += 1
                        self.request_timestamps.append(time.time())
                        
                        # Validate data
                        if self._validate_data(source_name, data):
                            return data
                        else:
                            raise ValueError(f"Invalid data from {source_name}")
                    
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited on {source_name}, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    elif response.status in self.config['fetcher']['retry_policy']['status_codes_to_retry']:
                        logger.warning(f"HTTP {response.status} for {source_name}, retrying")
                        continue
                    
                    else:
                        raise Exception(f"HTTP {response.status}: {response.reason}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {source_name}, attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                logger.warning(f"Request failed for {source_name}: {e}")
                if attempt == max_retries - 1:
                    raise
        
        raise Exception(f"All retries failed for {source_name}")
    
    def _validate_data(self, source_name: str, data: Any) -> bool:
        """Validate data based on source type"""
        if source_name.startswith('option_chain'):
            return isinstance(data, dict) and 'records' in data
        
        elif source_name.startswith('market_'):
            return isinstance(data, dict) and 'data' in data
        
        elif source_name.startswith('indices'):
            return isinstance(data, dict) and 'data' in data
        
        elif source_name == 'fii_dii':
            return isinstance(data, dict)
        
        return isinstance(data, (dict, list)) and bool(data)
    
    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        now = time.time()
        
        # Check daily limit
        if self.request_count >= self.daily_limit:
            raise Exception("Daily request limit reached")
        
        # Check requests per minute
        minute_ago = now - 60
        recent_requests = [t for t in self.request_timestamps if t > minute_ago]
        
        if len(recent_requests) >= self.config['fetcher']['rate_limiting']['max_requests_per_minute']:
            wait_time = 60 - (now - min(recent_requests))
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Clean old timestamps
        hour_ago = now - 3600
        self.request_timestamps = [t for t in self.request_timestamps if t > hour_ago]
    
    def _get_headers(self) -> Dict:
        """Get request headers"""
        headers = self.default_headers.copy()
        headers['User-Agent'] = random.choice(self.user_agents)
        return headers
    
    def _record_success(self):
        """Record successful request"""
        self.failure_count = 0
        if self.circuit_open:
            logger.info("Circuit breaker reset")
            self.circuit_open = False
    
    def _record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold and not self.circuit_open:
            self.circuit_open = True
            self.circuit_open_time = time.time()
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    async def get_cached_data(self) -> Dict:
        """Get all cached data"""
        try:
            results = {}
            for source_name in self.config['data_sources']:
                data = await self.cache.get(source_name)
                if data:
                    results[source_name] = data
            
            results['metadata'] = {
                'fetch_timestamp': datetime.now().isoformat(),
                'sources_fetched': list(results.keys()),
                'cached': True,
                'warning': 'Using cached data due to fetch failures'
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get cached data: {e}")
            return {}
    
    async def close(self):
        """Close the fetcher"""
        if self.session:
            await self.session.close()
        await self.cache.close()
        logger.info("DataFetcher closed")