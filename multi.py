# =============================================================================
# 🌐 NEW MULTI-CHAIN MANAGER - CLEAN ARCHITECTURE
# =============================================================================

import asyncio
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from web3 import Web3
from datetime import datetime, timedelta
import logging

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# 💰 PRICE DATA MANAGER - CENTRALIZED API & CACHING
# =============================================================================

class PriceDataManager:
    """
    Centralized price data management with smart caching and rate limiting
    Eliminates redundant API calls and provides fallback pricing
    """
    
    def __init__(self):
        self.token_price_cache = {}
        self.gas_price_cache = {}
        self.cache_duration = 60  # 60 seconds cache
        self.last_api_call = {}
        self.api_call_interval = 2  # 2 seconds between API calls
        
        # Fallback prices (updated periodically)
        self.fallback_prices = {
            "BTC": 108204.00,
            "ETH": 2531.92,
            "SOL": 150.94,
            "XRP": 2.34,
            "BNB": 658.64,
            "AVAX": 18.00,
            "MATIC": 0.8
        }
        
        # Token mapping for CoinGecko API
        self.token_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "SOL": "solana",
            "XRP": "ripple",
            "BNB": "binancecoin",
            "AVAX": "avalanche-2",
            "MATIC": "matic-network"
        }
        
        print("💰 PriceDataManager initialized with centralized caching")
    
    async def get_token_price(self, token_symbol: str) -> float:
        """
        Get token price with smart caching and rate limiting
        
        Args:
            token_symbol: Token symbol (e.g., 'BTC', 'ETH', 'MATIC')
            
        Returns:
            float: Token price in USD
        """
        # Check cache first
        cache_key = token_symbol.upper()
        now = time.time()
        
        if cache_key in self.token_price_cache:
            cached_data = self.token_price_cache[cache_key]
            if now - cached_data['timestamp'] < self.cache_duration:
                return cached_data['price']
        
        # Rate limiting check
        if cache_key in self.last_api_call:
            time_since_last_call = now - self.last_api_call[cache_key]
            if time_since_last_call < self.api_call_interval:
                # Use cached or fallback price if rate limited
                if cache_key in self.token_price_cache:
                    return self.token_price_cache[cache_key]['price']
                return self.fallback_prices.get(cache_key, 100.0)
        
        # Make API call
        try:
            coingecko_id = self.token_mapping.get(cache_key, "")
            if not coingecko_id:
                return self.fallback_prices.get(cache_key, 100.0)
            
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                price = float(data[coingecko_id]["usd"])
                
                # Cache the result
                self.token_price_cache[cache_key] = {
                    'price': price,
                    'timestamp': now
                }
                self.last_api_call[cache_key] = now
                
                return price
            else:
                print(f"⚠️ CoinGecko API returned status {response.status_code} for {token_symbol}")
                
        except Exception as e:
            print(f"⚠️ Price fetch failed for {token_symbol}: {e}")
        
        # Return fallback price
        fallback_price = self.fallback_prices.get(cache_key, 100.0)
        print(f"💡 Using fallback price for {token_symbol}: ${fallback_price}")
        return fallback_price
    
    def cache_gas_price(self, network_name: str, gas_price_gwei: float, duration: int = 30):
        """Cache gas price for a network"""
        self.gas_price_cache[network_name] = {
            'price_gwei': gas_price_gwei,
            'timestamp': time.time(),
            'duration': duration
        }
    
    def get_cached_gas_price(self, network_name: str) -> Optional[float]:
        """Get cached gas price if available and valid"""
        if network_name in self.gas_price_cache:
            cached = self.gas_price_cache[network_name]
            if time.time() - cached['timestamp'] < cached['duration']:
                return cached['price_gwei']
        return None
    
    def update_fallback_prices(self, new_prices: Dict[str, float]):
        """Update fallback prices (call this periodically with latest known good prices)"""
        self.fallback_prices.update(new_prices)
        print(f"📈 Updated fallback prices for {len(new_prices)} tokens")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics for monitoring"""
        return {
            'cached_tokens': len(self.token_price_cache),
            'cached_gas_prices': len(self.gas_price_cache),
            'fallback_tokens': len(self.fallback_prices),
            'cache_duration': self.cache_duration,
            'api_interval': self.api_call_interval
        }

# =============================================================================
# 🔧 NETWORK CONFIGURATION - CLEAN DATA STRUCTURES
# =============================================================================

@dataclass
class NetworkConfiguration:
    """Clean configuration for each blockchain network"""
    name: str
    chain_id: int
    rpc_endpoints: List[str]
    native_token: str
    native_token_symbol: str
    block_explorer: str
    dex_router: str
    common_tokens: Dict[str, str]
    faucet_url: Optional[str] = None
    bridge_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.rpc_endpoints:
            raise ValueError(f"No RPC endpoints provided for {self.name}")
        if self.chain_id <= 0:
            raise ValueError(f"Invalid chain_id for {self.name}")
        if not self.dex_router:
            raise ValueError(f"No DEX router provided for {self.name}")

class NetworkConfigManager:
    """Manages all network configurations with validation"""
    
    def __init__(self):
        self.networks = self._initialize_networks()
        print(f"🔧 NetworkConfigManager initialized with {len(self.networks)} networks")
    
    def _initialize_networks(self) -> Dict[str, NetworkConfiguration]:
        """Initialize configuration for all supported networks with WORKING RPC endpoints"""
        return {
            "polygon": NetworkConfiguration(
                name="Polygon",
                chain_id=137,
                rpc_endpoints=[
                    "https://polygon.llamarpc.com",
                    "https://polygon-rpc.com",
                    "https://rpc.ankr.com/polygon",
                    "https://polygon.blockpi.network/v1/rpc/public",
                    "https://polygon-mainnet.public.blastapi.io"
                ],
                native_token="MATIC",
                native_token_symbol="MATIC",
                block_explorer="https://polygonscan.com",
                dex_router="0xE592427A0AEce92De3Edee1F18E0157C05861564",
                common_tokens={
                    "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                    "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
                    "WBTC": "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6"
                },
                faucet_url="https://faucet.polygon.technology",
                bridge_url="https://wallet.polygon.technology/polygon/bridge"
            ),
            
            "optimism": NetworkConfiguration(
                name="Optimism",
                chain_id=10,
                rpc_endpoints=[
                    "https://optimism.llamarpc.com",
                    "https://mainnet.optimism.io",
                    "https://rpc.ankr.com/optimism",
                    "https://optimism.publicnode.com",
                    "https://optimism-mainnet.public.blastapi.io"
                ],
                native_token="ETH",
                native_token_symbol="ETH",
                block_explorer="https://optimistic.etherscan.io",
                dex_router="0xE592427A0AEce92De3Edee1F18E0157C05861564",
                common_tokens={
                    "USDC": "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
                    "USDT": "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",
                    "WBTC": "0x68f180fcCe6836688e9084f035309E29Bf0A2095"
                },
                bridge_url="https://app.optimism.io/bridge"
            ),
            
            "base": NetworkConfiguration(
                name="Base",
                chain_id=8453,
                rpc_endpoints=[
                    "https://base.llamarpc.com",
                    "https://mainnet.base.org",
                    "https://rpc.ankr.com/base",
                    "https://base.publicnode.com",
                    "https://base-mainnet.public.blastapi.io"
                ],
                native_token="ETH",
                native_token_symbol="ETH",
                block_explorer="https://basescan.org",
                dex_router="0x2626664c2603336E57B271c5C0b26F421741e481",
                common_tokens={
                    "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "USDbC": "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA"
                },
                bridge_url="https://bridge.base.org"
            )
        }
    
    def get_network(self, network_name: str) -> Optional[NetworkConfiguration]:
        """Get network configuration by name"""
        return self.networks.get(network_name)
    
    def get_all_networks(self) -> Dict[str, NetworkConfiguration]:
        """Get all network configurations"""
        return self.networks.copy()
    
    def validate_network(self, network_name: str) -> bool:
        """Validate that a network exists and is properly configured"""
        network = self.get_network(network_name)
        if not network:
            return False
        
        try:
            # Basic validation checks
            assert len(network.rpc_endpoints) > 0
            assert network.chain_id > 0
            assert network.dex_router
            assert len(network.common_tokens) > 0
            return True
        except (AssertionError, AttributeError):
            return False
        
# =============================================================================
# 🌐 NETWORK CONNECTION MANAGER - ROBUST RPC CONNECTIONS
# =============================================================================

class NetworkConnectionManager:
    """
    Manages Web3 connections to all networks with robust fallback logic
    Maintains the working multi-RPC approach that was proven successful
    """
    
    def __init__(self, config_manager: NetworkConfigManager):
        self.config_manager = config_manager
        self.active_connections = {}
        self.connection_status = {}
        self.rpc_health = {}  # Track which RPCs are working
        
        print("🌐 NetworkConnectionManager initialized")
    
    async def connect_to_network(self, network_name: str) -> bool:
        """
        Connect to a specific network with improved error handling and fallback RPCs
        Maintains the working approach from original implementation
        """
        network = self.config_manager.get_network(network_name)
        if not network:
            logger.error(f"Unknown network: {network_name}")
            return False
        
        # Skip if already connected and working
        if network_name in self.active_connections:
            try:
                w3 = self.active_connections[network_name]
                if w3.is_connected() and w3.eth.chain_id == network.chain_id:
                    return True
            except:
                # Connection is stale, remove it
                del self.active_connections[network_name]
        
        print(f"🌐 Connecting to {network.name}...")
        
        # Try each RPC endpoint (the working approach)
        for i, rpc_url in enumerate(network.rpc_endpoints):
            try:
                print(f"   Trying RPC {i+1}/{len(network.rpc_endpoints)}: {rpc_url}")
                
                # Enhanced Web3 configuration with better timeouts
                w3 = Web3(Web3.HTTPProvider(
                    rpc_url, 
                    request_kwargs={
                        'timeout': 10,
                        'headers': {'User-Agent': 'TradingBot/1.0'}
                    }
                ))
                
                # Test connection
                if w3.is_connected():
                    # Verify chain ID with timeout
                    try:
                        chain_id = w3.eth.chain_id
                        print(f"   Connected! Chain ID: {chain_id}")
                        
                        if chain_id == network.chain_id:
                            self.active_connections[network_name] = w3
                            self.connection_status[network_name] = True
                            self._update_rpc_health(network_name, rpc_url, True)
                            
                            logger.info(f"✅ Connected to {network.name} via {rpc_url}")
                            print(f"✅ {network.name} connection successful")
                            return True
                        else:
                            print(f"   ❌ Chain ID mismatch: expected {network.chain_id}, got {chain_id}")
                            self._update_rpc_health(network_name, rpc_url, False)
                    except Exception as e:
                        print(f"   ❌ Chain ID check failed: {e}")
                        self._update_rpc_health(network_name, rpc_url, False)
                else:
                    print(f"   ❌ Connection failed")
                    self._update_rpc_health(network_name, rpc_url, False)
                
            except Exception as e:
                print(f"   ❌ RPC Error: {str(e)[:100]}...")
                self._update_rpc_health(network_name, rpc_url, False)
                continue
        
        # All RPCs failed
        self.connection_status[network_name] = False
        logger.error(f"❌ Failed to connect to {network.name}")
        print(f"❌ All RPCs failed for {network.name}")
        return False
    
    async def connect_to_all_networks(self) -> Dict[str, bool]:
        """
        Connect to all networks in parallel with better error handling
        Maintains the working parallel connection approach
        """
        logger.info("🌐 Connecting to all L2 networks...")
        print("\n🌐 TESTING ALL NETWORK CONNECTIONS...")
        print("=" * 50)
        
        tasks = []
        for network_name in self.config_manager.get_all_networks().keys():
            task = asyncio.create_task(self.connect_to_network(network_name))
            tasks.append((network_name, task))
        
        results = {}
        for network_name, task in tasks:
            try:
                results[network_name] = await task
            except Exception as e:
                logger.error(f"Connection error for {network_name}: {e}")
                results[network_name] = False
        
        # Update connection status
        self.connection_status = results
        
        connected_count = sum(results.values())
        print(f"\n✅ Connected to {connected_count}/{len(self.config_manager.networks)} networks")
        
        # Show connection status
        for network_name, connected in results.items():
            network = self.config_manager.get_network(network_name)
            status = "✅ CONNECTED" if connected else "❌ FAILED"
            print(f"   {network.name if network else network_name}: {status}")
        
        logger.info(f"✅ Connected to {connected_count}/{len(self.config_manager.networks)} networks")
        
        return results
    
    def get_connection(self, network_name: str) -> Optional[Web3]:
        """Get Web3 connection for a network"""
        return self.active_connections.get(network_name)
    
    def is_connected(self, network_name: str) -> bool:
        """Check if a network is connected"""
        return self.connection_status.get(network_name, False)
    
    def get_connected_networks(self) -> List[str]:
        """Get list of connected network names"""
        return [name for name, connected in self.connection_status.items() if connected]
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get status of all network connections"""
        return self.connection_status.copy()
    
    async def refresh_connections(self) -> Dict[str, bool]:
        """Refresh all network connections"""
        try:
            print("🔄 Refreshing network connections...")
            self.active_connections.clear()
            self.connection_status.clear()
            
            # Re-initialize
            return await self.connect_to_all_networks()
            
        except Exception as e:
            print(f"❌ Connection refresh failed: {e}")
            return {}
    
    def _update_rpc_health(self, network_name: str, rpc_url: str, is_healthy: bool):
        """Track RPC endpoint health for future optimization"""
        if network_name not in self.rpc_health:
            self.rpc_health[network_name] = {}
        
        self.rpc_health[network_name][rpc_url] = {
            'healthy': is_healthy,
            'last_check': datetime.now(),
            'success_count': self.rpc_health[network_name].get(rpc_url, {}).get('success_count', 0) + (1 if is_healthy else 0),
            'total_attempts': self.rpc_health[network_name].get(rpc_url, {}).get('total_attempts', 0) + 1
        }
    
    def get_rpc_health_report(self) -> Dict[str, Any]:
        """Get health report for all RPC endpoints"""
        return {
            'rpc_health': self.rpc_health,
            'connected_networks': len(self.get_connected_networks()),
            'total_networks': len(self.config_manager.networks),
            'timestamp': datetime.now()
        }

# =============================================================================
# 💰 BALANCE MANAGER - CENTRALIZED BALANCE TRACKING
# =============================================================================

@dataclass
class BalanceInfo:
    """Balance information for a network"""
    network: str
    native_balance: float
    native_balance_usd: float
    native_token_symbol: str
    last_updated: datetime
    tokens: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.tokens is None:
            self.tokens = {}

class BalanceManager:
    """
    Centralized balance management across all networks
    Eliminates duplicate balance checking and provides unified caching
    """
    
    def __init__(self, config_manager: NetworkConfigManager, 
                 connection_manager: NetworkConnectionManager, 
                 price_manager: PriceDataManager):
        self.config_manager = config_manager
        self.connection_manager = connection_manager
        self.price_manager = price_manager
        
        self.balance_cache = {}
        self.cache_duration = 60  # 60 seconds cache
        self.last_balance_check = {}
        
        print("💰 BalanceManager initialized with centralized caching")
    
    async def get_balance_for_network(self, network_name: str, wallet_address: str) -> Optional[BalanceInfo]:
        """Get balance for a specific network with caching"""
        # Check cache first
        cache_key = f"{network_name}_{wallet_address}"
        if cache_key in self.balance_cache:
            cached_balance = self.balance_cache[cache_key]
            if (datetime.now() - cached_balance.last_updated).seconds < self.cache_duration:
                return cached_balance
        
        # Get fresh balance
        network = self.config_manager.get_network(network_name)
        w3 = self.connection_manager.get_connection(network_name)
        
        if not network or not w3:
            return None
        
        try:
            # Get native token balance
            balance_wei = w3.eth.get_balance(w3.to_checksum_address(wallet_address))
            native_balance = float(w3.from_wei(balance_wei, 'ether'))
            
            # Get native token price in USD
            native_token_price_usd = await self.price_manager.get_token_price(network.native_token_symbol)
            native_balance_usd = native_balance * native_token_price_usd
            
            # Create balance info
            balance_info = BalanceInfo(
                network=network_name,
                native_balance=native_balance,
                native_balance_usd=native_balance_usd,
                native_token_symbol=network.native_token_symbol,
                last_updated=datetime.now()
            )
            
            # Cache the result
            self.balance_cache[cache_key] = balance_info
            
            return balance_info
            
        except Exception as e:
            logger.error(f"Balance check failed for {network_name}: {e}")
            return None
    
    async def get_all_balances(self, wallet_address: str) -> Dict[str, BalanceInfo]:
        """Get balances across all connected networks"""
        print(f"\n💰 CHECKING BALANCES ACROSS ALL NETWORKS...")
        print(f"📍 Wallet: {wallet_address}")
        print("=" * 60)
        
        tasks = []
        connected_networks = self.connection_manager.get_connected_networks()
        
        for network_name in connected_networks:
            task = asyncio.create_task(self.get_balance_for_network(network_name, wallet_address))
            tasks.append((network_name, task))
        
        balances = {}
        total_usd = 0.0
        
        for network_name, task in tasks:
            try:
                balance = await task
                if balance:
                    balances[network_name] = balance
                    total_usd += balance.native_balance_usd
                    
                    # Display balance
                    network = self.config_manager.get_network(network_name)
                    
                    if balance.native_balance > 0:
                        print(f"✅ {network.name if network else network_name:<15}: {balance.native_balance:.6f} {balance.native_token_symbol} (${balance.native_balance_usd:.2f})")
                    else:
                        print(f"⚪ {network.name if network else network_name:<15}: {balance.native_balance:.6f} {balance.native_token_symbol} (${balance.native_balance_usd:.2f})")
                else:
                    network = self.config_manager.get_network(network_name)
                    print(f"❌ {network.name if network else network_name:<15}: Balance check failed")
                    
            except Exception as e:
                network = self.config_manager.get_network(network_name)
                print(f"❌ {network.name if network else network_name:<15}: Error - {e}")
        
        print("=" * 60)
        print(f"💰 Total Portfolio Value: ${total_usd:.2f}")
        
        # Show funding recommendations if no funds
        if total_usd == 0:
            self._show_funding_recommendations()
        
        return balances
    
    async def update_all_balances(self, wallet_address: str) -> bool:
        """Update native balances for all connected networks"""
        try:
            print(f"🔄 Updating balances for all networks...")
            
            tasks = []
            connected_networks = self.connection_manager.get_connected_networks()
            
            for network_name in connected_networks:
                task = asyncio.create_task(self.get_balance_for_network(network_name, wallet_address))
                tasks.append((network_name, task))
            
            updated_count = 0
            for network_name, task in tasks:
                try:
                    balance = await task
                    if balance is not None:
                        updated_count += 1
                        network = self.config_manager.get_network(network_name)
                        print(f"   ✅ {network.name if network else network_name}: {balance.native_balance:.6f} {balance.native_token_symbol}")
                except Exception as e:
                    network = self.config_manager.get_network(network_name)
                    print(f"   ❌ {network.name if network else network_name}: Balance update failed - {e}")
            
            print(f"✅ Updated balances for {updated_count} networks")
            return updated_count > 0
            
        except Exception as e:
            print(f"❌ Balance update failed: {e}")
            return False
    
    def get_total_portfolio_value_usd(self, wallet_address: str) -> float:
        """Get total portfolio value in USD from cached balances"""
        total_usd = 0.0
        
        for cache_key, balance_info in self.balance_cache.items():
            if wallet_address in cache_key:
                # Check if cache is still valid
                if (datetime.now() - balance_info.last_updated).seconds < self.cache_duration:
                    total_usd += balance_info.native_balance_usd
        
        return total_usd
    
    def get_network_balance(self, network_name: str, wallet_address: str) -> float:
        """Get cached native balance for a specific network"""
        cache_key = f"{network_name}_{wallet_address}"
        
        if cache_key in self.balance_cache:
            cached_balance = self.balance_cache[cache_key]
            if (datetime.now() - cached_balance.last_updated).seconds < self.cache_duration:
                return cached_balance.native_balance
        
        return 0.0
    
    def can_afford_amount(self, network_name: str, wallet_address: str, required_amount: float) -> bool:
        """Check if wallet has enough native tokens for a transaction"""
        available_balance = self.get_network_balance(network_name, wallet_address)
        return available_balance >= required_amount
    
    def get_funded_networks(self, wallet_address: str) -> List[str]:
        """Get list of networks that have funds"""
        funded_networks = []
        
        for cache_key, balance_info in self.balance_cache.items():
            if wallet_address in cache_key and balance_info.native_balance > 0:
                # Check if cache is still valid
                if (datetime.now() - balance_info.last_updated).seconds < self.cache_duration:
                    funded_networks.append(balance_info.network)
        
        return funded_networks
    
    def _show_funding_recommendations(self):
        """Show funding recommendations when no funds detected"""
        print("\n⚠️  NO FUNDS DETECTED!")
        print("💡 Fund your wallet on these networks to start trading:")
        
        for network_name, network in self.config_manager.get_all_networks().items():
            if network.bridge_url:
                print(f"   🌉 {network.name}: {network.bridge_url}")
            if network.faucet_url:
                print(f"   🚰 {network.name} Testnet: {network.faucet_url}")
    
    def get_balance_summary(self, wallet_address: str) -> Dict[str, Any]:
        """Get summary of all balances"""
        funded_networks = self.get_funded_networks(wallet_address)
        total_value = self.get_total_portfolio_value_usd(wallet_address)
        
        return {
            'total_usd_value': total_value,
            'funded_networks': funded_networks,
            'funded_network_count': len(funded_networks),
            'total_networks': len(self.config_manager.networks),
            'cache_entries': len(self.balance_cache),
            'last_update': max([b.last_updated for b in self.balance_cache.values()]) if self.balance_cache else None
        }

# =============================================================================
# ⛽ GAS ESTIMATOR - SMART GAS OPTIMIZATION
# =============================================================================

@dataclass
class GasEstimate:
    """Gas cost estimation for a network"""
    network: str
    gas_price_gwei: float
    estimated_cost_usd: float
    estimated_cost_native: float
    native_token_price_usd: float
    liquidity_score: float  # 0-100, higher = better liquidity
    total_cost_score: float  # gas + slippage estimate as % of trade
    gas_units: int
    timestamp: datetime

class GasEstimator:
    """
    Smart gas estimation and optimization across all networks
    Finds the cheapest network for trading and manages gas costs efficiently
    """
    
    def __init__(self, config_manager: NetworkConfigManager, 
                 connection_manager: NetworkConnectionManager, 
                 price_manager: PriceDataManager):
        self.config_manager = config_manager
        self.connection_manager = connection_manager
        self.price_manager = price_manager
        
        self.gas_cache = {}
        self.cache_duration = 30  # 30 seconds cache for gas prices
        
        # Gas unit estimates for different transaction types
        self.gas_unit_estimates = {
            "polygon": {"swap": 200000, "transfer": 21000},
            "optimism": {"swap": 200000, "transfer": 21000},
            "base": {"swap": 200000, "transfer": 21000}
        }
        
        # Liquidity scoring (based on TVL and volume data)
        self.liquidity_scores = {
            "polygon": 85,    # Good liquidity
            "optimism": 75,   # Moderate liquidity
            "base": 70        # Growing liquidity
        }
        
        print("⛽ GasEstimator initialized with smart optimization")
    
    async def get_gas_estimate(self, network_name: str, trade_amount_usd: float, 
                              transaction_type: str = "swap") -> Optional[GasEstimate]:
        """Get comprehensive gas estimate for a network with caching"""
        
        # Check cache first
        cache_key = f"{network_name}_{transaction_type}_{int(time.time() / self.cache_duration)}"
        if cache_key in self.gas_cache:
            cached_estimate = self.gas_cache[cache_key]
            # Recalculate total cost score for this specific trade amount
            cached_estimate.total_cost_score = self._calculate_total_cost_score(
                cached_estimate.estimated_cost_usd, 
                cached_estimate.liquidity_score, 
                trade_amount_usd
            )
            return cached_estimate
        
        w3 = self.connection_manager.get_connection(network_name)
        network = self.config_manager.get_network(network_name)
        
        if not w3 or not network:
            return None
        
        try:
            # Check if we have cached gas price from price manager
            cached_gas_price = self.price_manager.get_cached_gas_price(network_name)
            
            if cached_gas_price:
                gas_price_gwei = cached_gas_price
                gas_price_wei = w3.to_wei(gas_price_gwei, 'gwei')
            else:
                # Get current gas price with error handling
                try:
                    gas_price_wei = w3.eth.gas_price
                    gas_price_gwei = float(w3.from_wei(gas_price_wei, 'gwei'))
                    
                    # Cache the gas price
                    self.price_manager.cache_gas_price(network_name, gas_price_gwei)
                    
                except Exception as e:
                    print(f"⚠️ Gas price fetch failed for {network_name}: {e}")
                    # Use fallback gas prices
                    fallback_gas_prices = {
                        "polygon": 30.0,
                        "optimism": 0.001,
                        "base": 0.001
                    }
                    gas_price_gwei = fallback_gas_prices.get(network_name, 10.0)
                    gas_price_wei = w3.to_wei(gas_price_gwei, 'gwei')
            
            # Estimate gas units based on transaction type
            estimated_gas_units = self._estimate_gas_units(network_name, transaction_type)
            
            # Calculate gas cost in native token
            total_gas_cost_wei = gas_price_wei * estimated_gas_units
            estimated_cost_native = float(w3.from_wei(total_gas_cost_wei, 'ether'))
            
            # Get native token price in USD
            native_token_price_usd = await self.price_manager.get_token_price(network.native_token_symbol)
            
            # Calculate USD cost
            estimated_cost_usd = estimated_cost_native * native_token_price_usd
            
            # Get liquidity score for this network
            liquidity_score = self._estimate_liquidity_score(network_name, trade_amount_usd)
            
            # Calculate total cost score (gas + slippage as % of trade)
            total_cost_score = self._calculate_total_cost_score(estimated_cost_usd, liquidity_score, trade_amount_usd)
            
            estimate = GasEstimate(
                network=network_name,
                gas_price_gwei=gas_price_gwei,
                estimated_cost_usd=estimated_cost_usd,
                estimated_cost_native=estimated_cost_native,
                native_token_price_usd=native_token_price_usd,
                liquidity_score=liquidity_score,
                total_cost_score=total_cost_score,
                gas_units=estimated_gas_units,
                timestamp=datetime.now()
            )
            
            # Cache the result
            self.gas_cache[cache_key] = estimate
            
            return estimate
            
        except Exception as e:
            logger.error(f"Gas estimation failed for {network_name}: {e}")
            print(f"❌ Gas estimation failed for {network_name}: {e}")
            return None
    
    async def find_optimal_network(self, trade_amount_usd: float, 
                                 transaction_type: str = "swap") -> Tuple[Optional[str], Optional[GasEstimate]]:
        """Find the optimal network for a trade based on gas costs and liquidity"""
        print(f"\n🔍 FINDING OPTIMAL NETWORK FOR ${trade_amount_usd:.2f} TRADE...")
        
        # Get gas estimates for all connected networks
        estimates = {}
        tasks = []
        
        connected_networks = self.connection_manager.get_connected_networks()
        for network_name in connected_networks:
            task = asyncio.create_task(self.get_gas_estimate(network_name, trade_amount_usd, transaction_type))
            tasks.append((network_name, task))
        
        for network_name, task in tasks:
            try:
                estimate = await task
                if estimate:
                    estimates[network_name] = estimate
                    
                    # Display estimate
                    network = self.config_manager.get_network(network_name)
                    print(f"⛽ {network.name if network else network_name:<15}: ${estimate.estimated_cost_usd:.3f} gas ({estimate.total_cost_score:.2f}% of trade)")
            except Exception as e:
                print(f"❌ {network_name}: Gas estimation failed - {e}")
        
        if not estimates:
            print("❌ No gas estimates available")
            return None, None
        
        # Find the best network (lowest total cost score)
        best_network = min(estimates.keys(), key=lambda n: estimates[n].total_cost_score)
        best_estimate = estimates[best_network]
        
        # Only return if gas cost is reasonable (less than 10% of trade)
        if best_estimate.total_cost_score <= 10.0:
            network = self.config_manager.get_network(best_network)
            print(f"✅ Optimal network: {network.name if network else best_network} (${best_estimate.estimated_cost_usd:.3f} gas)")
            return best_network, best_estimate
        else:
            print(f"❌ All networks too expensive (best: {best_estimate.total_cost_score:.2f}% of trade)")
            return None, None
    
    def can_afford_gas(self, network_name: str, wallet_address: str, gas_estimate: GasEstimate, 
                      balance_manager: 'BalanceManager') -> bool:
        """Check if wallet can afford gas on a specific network"""
        try:
            # Get network config with proper error handling
            network = self.config_manager.get_network(network_name)
            if not network:
                print(f"❌ Network configuration not found for {network_name}")
                return False
        
            # Get available balance
            available_balance = balance_manager.get_network_balance(network_name, wallet_address)
            required_amount = gas_estimate.estimated_cost_native * 1.2  # 20% safety buffer
        
            can_afford = available_balance >= required_amount
        
            if not can_afford:
                print(f"💸 Insufficient {network.native_token_symbol} on {network.name}")
                print(f"   Required: {required_amount:.6f} {network.native_token_symbol}")
                print(f"   Available: {available_balance:.6f} {network.native_token_symbol}")
            
                # Provide helpful links if available
                if hasattr(network, 'bridge_url') and network.bridge_url:
                    print(f"   🌉 Bridge tokens: {network.bridge_url}")
                if hasattr(network, 'faucet_url') and network.faucet_url:
                    print(f"   🚰 Get testnet tokens: {network.faucet_url}")
            else:
                print(f"✅ Sufficient {network.native_token_symbol} on {network.name}")
                print(f"   Required: {required_amount:.6f} {network.native_token_symbol}")
                print(f"   Available: {available_balance:.6f} {network.native_token_symbol}")
        
            return can_afford
        
        except Exception as e:
            print(f"❌ Gas affordability check failed for {network_name}: {e}")
            return False
    
    def _estimate_gas_units(self, network_name: str, transaction_type: str) -> int:
        """Estimate gas units needed for different transaction types"""
        return self.gas_unit_estimates.get(network_name, {}).get(transaction_type, 200000)
    
    def _estimate_liquidity_score(self, network_name: str, trade_amount_usd: float) -> float:
        """Estimate liquidity score (0-100) based on network and trade size"""
        base_score = self.liquidity_scores.get(network_name, 50)
        
        # Reduce score for larger trades (more slippage expected)
        if trade_amount_usd > 10000:
            base_score *= 0.7
        elif trade_amount_usd > 1000:
            base_score *= 0.85
        elif trade_amount_usd > 100:
            base_score *= 0.95
        
        return min(100, base_score)
    
    def _calculate_total_cost_score(self, gas_cost_usd: float, liquidity_score: float, trade_amount_usd: float) -> float:
        """Calculate total cost score including gas and expected slippage"""
        # Calculate gas as percentage of trade
        gas_percentage = (gas_cost_usd / trade_amount_usd) * 100 if trade_amount_usd > 0 else 0
        
        # Estimate slippage cost based on liquidity
        slippage_factor = (100 - liquidity_score) / 1000  # Convert to percentage
        estimated_slippage_usd = trade_amount_usd * slippage_factor
        slippage_percentage = (estimated_slippage_usd / trade_amount_usd) * 100 if trade_amount_usd > 0 else 0
        
        # Total cost as percentage of trade
        return gas_percentage + slippage_percentage
    
    def get_gas_statistics(self) -> Dict[str, Any]:
        """Get gas estimation statistics for monitoring"""
        return {
            'cached_estimates': len(self.gas_cache),
            'cache_duration': self.cache_duration,
            'supported_networks': list(self.gas_unit_estimates.keys()),
            'liquidity_scores': self.liquidity_scores.copy(),
            'last_estimates': {
                cache_key: estimate.timestamp 
                for cache_key, estimate in self.gas_cache.items()
            }
        }

# =============================================================================
# 🎯 MULTI-CHAIN MANAGER - MAIN ORCHESTRATOR CLASS
# =============================================================================

class MultiChainManager:
    """
    Main orchestrator for multi-chain operations
    Provides clean unified interface for all blockchain interactions
    Integrates all component managers for seamless operation
    """
    
    def __init__(self):
        """Initialize the complete multi-chain system"""
        print("🚀 Initializing Enhanced Multi-Chain Manager...")
        
        # Initialize all component managers
        self.price_manager = PriceDataManager()
        self.config_manager = NetworkConfigManager()
        self.connection_manager = NetworkConnectionManager(self.config_manager)
        self.balance_manager = BalanceManager(self.config_manager, self.connection_manager, self.price_manager)
        self.gas_estimator = GasEstimator(self.config_manager, self.connection_manager, self.price_manager)
        
        # Legacy compatibility properties
        self.networks = self.config_manager.networks
        self.active_connections = self.connection_manager.active_connections
        
        # System state
        self.multi_chain_initialized = False
        self.initialization_time = None
        
        print("🎯 MultiChainManager initialized successfully")
        logger.info("🎯 Enhanced Multi-Chain Manager system ready")
    
    # =============================================================================
    # 🚀 CORE INITIALIZATION METHODS
    # =============================================================================
    
    async def connect_to_all_networks(self) -> Dict[str, bool]:
        """Initialize connections to all networks"""
        print("🌐 Initializing multi-chain connections...")
        
        connection_results = await self.connection_manager.connect_to_all_networks()
        
        if any(connection_results.values()):
            self.multi_chain_initialized = True
            self.initialization_time = datetime.now()
            
            connected_count = sum(connection_results.values())
            print(f"✅ Multi-chain initialized: {connected_count}/3 networks connected")
            logger.info(f"✅ Multi-chain system initialized with {connected_count} networks")
        else:
            print("❌ No networks connected - multi-chain features disabled")
            logger.error("❌ Multi-chain initialization failed - no networks connected")
        
        return connection_results
    
    async def initialize_system(self, wallet_address: Optional[str] = None) -> bool:
        """Complete system initialization with optional balance update"""
        try:
            # Connect to all networks
            connection_results = await self.connect_to_all_networks()
            
            if not any(connection_results.values()):
                return False
            
            # Update balances if wallet address provided
            if wallet_address:
                print("💰 Updating initial balance cache...")
                await self.balance_manager.update_all_balances(wallet_address)
                print("✅ Balance cache initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            logger.error(f"System initialization error: {e}")
            return False
    
    # =============================================================================
    # 💰 BALANCE MANAGEMENT INTERFACE
    # =============================================================================
    
    async def get_all_balances(self, wallet_address: str) -> Dict[str, BalanceInfo]:
        """Get balances across all connected networks"""
        return await self.balance_manager.get_all_balances(wallet_address)
    
    async def get_balance_for_network(self, network_name: str, wallet_address: str) -> Optional[BalanceInfo]:
        """Get balance for a specific network"""
        return await self.balance_manager.get_balance_for_network(network_name, wallet_address)
    
    async def update_all_balances(self, wallet_address: str) -> bool:
        """Update balances for all networks"""
        return await self.balance_manager.update_all_balances(wallet_address)
    
    def get_total_portfolio_value(self, wallet_address: str) -> float:
        """Get total portfolio value in USD"""
        return self.balance_manager.get_total_portfolio_value_usd(wallet_address)
    
    def get_funded_networks(self, wallet_address: str) -> List[str]:
        """Get list of networks with funds"""
        return self.balance_manager.get_funded_networks(wallet_address)
    
    # =============================================================================
    # ⛽ GAS OPTIMIZATION INTERFACE
    # =============================================================================
    
    async def find_optimal_network(self, trade_amount_usd: float) -> Tuple[Optional[str], Optional[GasEstimate]]:
        """Find optimal network for trading based on gas costs"""
        return await self.gas_estimator.find_optimal_network(trade_amount_usd)
    
    async def get_gas_estimate(self, network_name: str, trade_amount_usd: float) -> Optional[GasEstimate]:
        """Get gas estimate for a specific network"""
        return await self.gas_estimator.get_gas_estimate(network_name, trade_amount_usd)
    
    def can_afford_trade(self, network_name: str, gas_estimate: GasEstimate, wallet_address: str) -> bool:
        """Check if wallet can afford gas for a trade"""
        return self.gas_estimator.can_afford_gas(network_name, wallet_address, gas_estimate, self.balance_manager)
    
    # =============================================================================
    # 💸 PRICE DATA INTERFACE
    # =============================================================================
    
    async def _get_token_price(self, token_symbol: str) -> float:
        """Get token price (unified method for compatibility)"""
        return await self.price_manager.get_token_price(token_symbol)
    
    def update_fallback_prices(self, new_prices: Dict[str, float]):
        """Update fallback token prices"""
        self.price_manager.update_fallback_prices(new_prices)
    
    # =============================================================================
    # 🔗 CONNECTION MANAGEMENT INTERFACE
    # =============================================================================
    
    def get_connection(self, network_name: str) -> Optional[Web3]:
        """Get Web3 connection for a network"""
        return self.connection_manager.get_connection(network_name)
    
    def is_network_connected(self, network_name: str) -> bool:
        """Check if a network is connected"""
        return self.connection_manager.is_connected(network_name)
    
    def get_connected_networks(self) -> List[str]:
        """Get list of connected networks"""
        return self.connection_manager.get_connected_networks()
    
    async def refresh_connections(self) -> Dict[str, bool]:
        """Refresh all network connections"""
        self.multi_chain_initialized = False
        return await self.connection_manager.refresh_connections()
    
    # =============================================================================
    # 📊 SYSTEM MONITORING & DIAGNOSTICS
    # =============================================================================
    
    async def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        print("\n🔧 RUNNING SYSTEM DIAGNOSTICS...")
        print("=" * 50)
        
        diagnostics = {
            'system_initialized': self.multi_chain_initialized,
            'initialization_time': self.initialization_time,
            'connected_networks': self.connection_manager.get_connected_networks(),
            'connection_status': self.connection_manager.get_connection_status(),
            'price_cache_stats': self.price_manager.get_cache_stats(),
            'gas_statistics': self.gas_estimator.get_gas_statistics(),
            'rpc_health': self.connection_manager.get_rpc_health_report(),
            'timestamp': datetime.now()
        }
        
        # Display key metrics
        print(f"✅ System Status: {'OPERATIONAL' if self.multi_chain_initialized else 'NOT INITIALIZED'}")
        print(f"🌐 Connected Networks: {len(diagnostics['connected_networks'])}/3")
        print(f"💰 Price Cache: {diagnostics['price_cache_stats']['cached_tokens']} tokens")
        print(f"⛽ Gas Cache: {diagnostics['gas_statistics']['cached_estimates']} estimates")
        
        if self.initialization_time:
            uptime = datetime.now() - self.initialization_time
            print(f"⏱️  System Uptime: {uptime}")
        
        print("=" * 50)
        
        return diagnostics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'initialized': self.multi_chain_initialized,
            'connected_networks': len(self.get_connected_networks()),
            'total_networks': len(self.config_manager.networks),
            'uptime_seconds': (datetime.now() - self.initialization_time).total_seconds() if self.initialization_time else 0,
            'components': {
                'price_manager': bool(self.price_manager),
                'config_manager': bool(self.config_manager),
                'connection_manager': bool(self.connection_manager),
                'balance_manager': bool(self.balance_manager),
                'gas_estimator': bool(self.gas_estimator)
            }
        }
    
    # =============================================================================
    # 🛠️ UTILITY & COMPATIBILITY METHODS
    # =============================================================================
    
    async def check_network_funding_status(self, wallet_address: str) -> Tuple[int, int]:
        """Check funding status across networks (returns funded_count, unfunded_count)"""
        all_networks = list(self.config_manager.networks.keys())
        funded_networks = self.get_funded_networks(wallet_address)
        
        funded_count = len(funded_networks)
        unfunded_count = len(all_networks) - funded_count
        
        if unfunded_count > 0:
            print(f"\n💡 FUNDING RECOMMENDATIONS:")
            print(f"✅ Funded networks: {', '.join(funded_networks) if funded_networks else 'None'}")
            
            unfunded = [name for name in all_networks if name not in funded_networks]
            print(f"⚠️  Unfunded networks: {', '.join(unfunded)}")
            
            print(f"\n🌉 BRIDGE/FUND THESE NETWORKS:")
            for network_name in unfunded:
                network = self.config_manager.get_network(network_name)
                if network and network.bridge_url:
                    print(f"   {network.name}: {network.bridge_url}")
        
        return funded_count, unfunded_count
    
    async def suggest_optimal_funding(self, target_trade_amount_usd: float, wallet_address: str):
        """Suggest optimal funding strategy for trading"""
        print(f"\n💡 FUNDING OPTIMIZATION FOR ${target_trade_amount_usd:.2f} TRADES")
        print("=" * 60)
        
        if not self.multi_chain_initialized:
            print("❌ System not initialized")
            return
        
        # Get gas estimates for all networks
        estimates = {}
        for network_name in self.get_connected_networks():
            estimate = await self.gas_estimator.get_gas_estimate(network_name, target_trade_amount_usd)
            if estimate:
                estimates[network_name] = estimate
        
        if not estimates:
            print("❌ No gas estimates available")
            return
        
        # Sort by cost efficiency
        sorted_networks = sorted(estimates.items(), key=lambda x: x[1].total_cost_score)
        
        print("🏆 RECOMMENDED FUNDING PRIORITY:")
        for i, (network_name, estimate) in enumerate(sorted_networks, 1):
            network = self.config_manager.get_network(network_name)
            current_balance = self.balance_manager.get_network_balance(network_name, wallet_address)
            
            # Estimate funding needed (gas for ~10 trades)
            recommended_funding = estimate.estimated_cost_native * 10
            needs_funding = current_balance < recommended_funding
            
            status = "🔴 NEEDS FUNDING" if needs_funding else "✅ FUNDED"
            
            print(f"{i}. {network.name if network else network_name:<15} - ${estimate.estimated_cost_usd:.3f} gas ({estimate.total_cost_score:.1f}%) {status}")
            
            if needs_funding:
                print(f"   💰 Current: {current_balance:.6f} {network.native_token_symbol if network else 'tokens'}")
                print(f"   💡 Recommend: {recommended_funding:.6f} {network.native_token_symbol if network else 'tokens'} for 10 trades")
                
                # Show bridge URL
                if network and hasattr(network, 'bridge_url') and network.bridge_url:
                    print(f"   🌉 Bridge: {network.bridge_url}")
    
    def __str__(self) -> str:
        """String representation of the system"""
        status = "INITIALIZED" if self.multi_chain_initialized else "NOT INITIALIZED"
        connected = len(self.get_connected_networks()) if self.multi_chain_initialized else 0
        return f"MultiChainManager(status={status}, connected_networks={connected}/3)"
    
    def __repr__(self) -> str:
        """Detailed representation of the system"""
        return (f"MultiChainManager("
                f"initialized={self.multi_chain_initialized}, "
                f"networks={list(self.config_manager.networks.keys())}, "
                f"connected={self.get_connected_networks()}, "
                f"components=['price_manager', 'config_manager', 'connection_manager', 'balance_manager', 'gas_estimator']"
                f")")

# =============================================================================
# 🧪 TESTING AND DIAGNOSTICS FUNCTIONS
# =============================================================================

async def test_multi_chain_manager():
    """Test function to verify all networks and functionality"""
    print("🚀 TESTING ENHANCED MULTI-CHAIN MANAGER")
    print("=" * 60)
    
    manager = MultiChainManager()
    
    # Test system initialization
    success = await manager.initialize_system()
    
    if success:
        # Run diagnostics
        await manager.run_system_diagnostics()
        
        # Test gas optimization
        print(f"\n⛽ TESTING GAS OPTIMIZATION...")
        test_trade_amount = 15.0
        
        optimal_network, gas_estimate = await manager.find_optimal_network(test_trade_amount)
        
        if optimal_network:
            network = manager.config_manager.get_network(optimal_network)
            network_name = network.name if network else optimal_network
            print(f"✅ Gas optimization successful: {network_name} selected")
        else:
            print("❌ No suitable network found")
    else:
        print("❌ System initialization failed")
    
    return manager

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_multi_chain_manager())    
