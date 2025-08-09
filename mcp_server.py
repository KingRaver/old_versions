#!/usr/bin/env python3
"""
CoinGecko MCP Server for Tokenetics
Provides Claude AI with real-time cryptocurrency data through Model Context Protocol
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coingecko-mcp")

# CoinGecko API configuration
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

class CoinGeckoMCP:
    def __init__(self):
        self.server = Server("coingecko-mcp")
        self.http_client = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.http_client:
            await self.http_client.aclose()

    async def make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Make HTTP request to CoinGecko API with error handling"""
            if self.http_client is None:
                raise Exception("HTTP client not initialized")
                
            url = f"{COINGECKO_BASE_URL}/{endpoint.lstrip('/')}"
            
            try:
                response = await self.http_client.get(url, params=params or {})
                response.raise_for_status()
                
                # Log successful request
                logger.info(f"CoinGecko API request successful: {endpoint}")
                
                return response.json()
                
            except httpx.HTTPStatusError as e:
                logger.error(f"CoinGecko API HTTP error: {e.response.status_code} - {e.response.text}")
                raise Exception(f"CoinGecko API error: {e.response.status_code}")
            except httpx.RequestError as e:
                logger.error(f"CoinGecko API request error: {str(e)}")
                raise Exception(f"Failed to connect to CoinGecko API: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in CoinGecko request: {str(e)}")
                raise Exception(f"Unexpected error: {str(e)}")

    def setup_tools(self):
        """Register all MCP tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available tools for Claude"""
            return [
                types.Tool(
                    name="get_crypto_prices",
                    description="Get current prices and basic market data for cryptocurrencies",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "coins": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of cryptocurrency IDs (e.g., ['bitcoin', 'ethereum']) or symbols (e.g., ['btc', 'eth'])"
                            },
                            "vs_currency": {
                                "type": "string",
                                "default": "usd",
                                "description": "Target currency (usd, eur, btc, etc.)"
                            },
                            "include_24hr_change": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include 24-hour price change percentage"
                            }
                        },
                        "required": ["coins"]
                    }
                ),
                types.Tool(
                    name="get_trending_coins",
                    description="Get currently trending cryptocurrencies",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "default": 10,
                                "description": "Number of trending coins to return (max 10)"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="get_market_data",
                    description="Get detailed market data for cryptocurrencies",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "coins": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of cryptocurrency IDs"
                            },
                            "vs_currency": {
                                "type": "string",
                                "default": "usd",
                                "description": "Target currency"
                            }
                        },
                        "required": ["coins"]
                    }
                ),
                types.Tool(
                    name="search_coins",
                    description="Search for cryptocurrencies by name or symbol",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (coin name or symbol)"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="get_top_coins",
                    description="Get top cryptocurrencies by market capitalization",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "default": 10,
                                "description": "Number of top coins to return (max 250)"
                            },
                            "vs_currency": {
                                "type": "string",
                                "default": "usd",
                                "description": "Target currency"
                            }
                        }
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls from Claude"""
            try:
                if name == "get_crypto_prices":
                    result = await self._get_crypto_prices(**arguments)
                elif name == "get_trending_coins":
                    result = await self._get_trending_coins(**arguments)
                elif name == "get_market_data":
                    result = await self._get_market_data(**arguments)
                elif name == "search_coins":
                    result = await self._search_coins(**arguments)
                elif name == "get_top_coins":
                    result = await self._get_top_coins(**arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                error_response = {
                    "error": str(e),
                    "tool": name,
                    "arguments": arguments
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

    # Tool implementation methods
    async def _get_crypto_prices(self, coins: List[str], vs_currency: str = "usd", include_24hr_change: bool = True) -> Dict[str, Any]:
        """Get current prices for specified cryptocurrencies"""
        # Convert symbols to IDs if needed
        coin_ids = ",".join(coins)
        
        params = {
            "ids": coin_ids,
            "vs_currencies": vs_currency,
            "include_24hr_change": include_24hr_change,
            "include_market_cap": True,
            "include_24hr_vol": True
        }
        
        data = await self.make_request("simple/price", params)
        
        return {
            "prices": data,
            "currency": vs_currency,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def _get_trending_coins(self, limit: int = 10) -> Dict[str, Any]:
        """Get trending cryptocurrencies"""
        data = await self.make_request("search/trending")
        
        trending_coins = []
        for item in data.get("coins", [])[:limit]:
            coin = item.get("item", {})
            trending_coins.append({
                "id": coin.get("id"),
                "name": coin.get("name"),
                "symbol": coin.get("symbol"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "price_btc": coin.get("price_btc"),
                "thumb": coin.get("thumb")
            })
        
        return {
            "trending_coins": trending_coins,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def _get_market_data(self, coins: List[str], vs_currency: str = "usd") -> Dict[str, Any]:
        """Get detailed market data for cryptocurrencies"""
        coin_ids = ",".join(coins)
        
        params = {
            "ids": coin_ids,
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": len(coins),
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h,7d,30d"
        }
        
        data = await self.make_request("coins/markets", params)
        
        return {
            "market_data": data,
            "currency": vs_currency,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def _search_coins(self, query: str) -> Dict[str, Any]:
        """Search for cryptocurrencies"""
        params = {"query": query}
        data = await self.make_request("search", params)
        
        # Format search results
        coins = []
        for coin in data.get("coins", [])[:10]:  # Limit to top 10 results
            coins.append({
                "id": coin.get("id"),
                "name": coin.get("name"),
                "symbol": coin.get("symbol"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "thumb": coin.get("thumb")
            })
        
        return {
            "search_results": coins,
            "query": query,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def _get_top_coins(self, limit: int = 10, vs_currency: str = "usd") -> Dict[str, Any]:
        """Get top cryptocurrencies by market cap"""
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": min(limit, 250),  # CoinGecko API limit
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h"
        }
        
        data = await self.make_request("coins/markets", params)
        
        return {
            "top_coins": data,
            "currency": vs_currency,
            "limit": limit,
            "timestamp": asyncio.get_event_loop().time()
        }

async def main():
    """Main entry point for the MCP server"""
    logger.info("Starting CoinGecko MCP Server for Tokenetics...")
    
    async with CoinGeckoMCP() as coingecko_mcp:
        # Setup tools
        coingecko_mcp.setup_tools()
        
        # Run the server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("CoinGecko MCP Server is running and ready for connections")
            await coingecko_mcp.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="coingecko-mcp",
                    server_version="1.0.0",
                    capabilities=coingecko_mcp.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )

if __name__ == "__main__":
    asyncio.run(main())
