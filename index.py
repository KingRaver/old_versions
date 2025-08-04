# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
import os
import httpx
import asyncio
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="Tokenetics API",
    description="AI Crypto Trading Assistant with CoinGecko MCP Integration",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tokenetics.space", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    context: Dict[str, Any] = {}

class ChatResponse(BaseModel):
    answer: str
    market_data: Dict[str, Any] = {}
    confidence: float = 0.0

# Root endpoints
@app.get("/")
async def root():
    return {"message": "Tokenetics API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "tokenetics-api"}

# Market data endpoints
@app.get("/api/market/prices")
async def get_crypto_prices():
    """Get current prices for Bitcoin and Ethereum"""
    try:
        # Using CoinGecko API directly (free tier)
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": "bitcoin,ethereum",
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_market_cap": "true",
                    "include_24hr_vol": "true"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format the response for frontend
                formatted_data = {
                    "bitcoin": {
                        "price": data["bitcoin"]["usd"],
                        "change_24h": data["bitcoin"]["usd_24h_change"],
                        "market_cap": data["bitcoin"]["usd_market_cap"],
                        "volume_24h": data["bitcoin"]["usd_24h_vol"]
                    },
                    "ethereum": {
                        "price": data["ethereum"]["usd"],
                        "change_24h": data["ethereum"]["usd_24h_change"],
                        "market_cap": data["ethereum"]["usd_market_cap"],
                        "volume_24h": data["ethereum"]["usd_24h_vol"]
                    },
                    "timestamp": int(asyncio.get_event_loop().time())
                }
                
                return formatted_data
            else:
                raise HTTPException(status_code=503, detail="CoinGecko API unavailable")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prices: {str(e)}")

@app.get("/api/market/trending")
async def get_trending_coins():
    """Get trending coins from CoinGecko"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.coingecko.com/api/v3/search/trending")
            
            if response.status_code == 200:
                data = response.json()
                
                # Format trending coins data
                trending_coins = []
                for coin in data['coins'][:5]:  # Top 5 trending
                    coin_data = coin['item']
                    trending_coins.append({
                        "id": coin_data['id'],
                        "name": coin_data['name'],
                        "symbol": coin_data['symbol'],
                        "market_cap_rank": coin_data.get('market_cap_rank'),
                        "thumb": coin_data.get('thumb'),
                        "price_btc": coin_data.get('price_btc')
                    })
                    
                return {"trending_coins": trending_coins}
            else:
                raise HTTPException(status_code=503, detail="CoinGecko API unavailable")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trending coins: {str(e)}")

@app.get("/api/market/coin/{coin_id}")
async def get_coin_details(coin_id: str):
    """Get detailed information about a specific coin"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "true",
                    "community_data": "false",
                    "developer_data": "false"
                }
            )
            
            if response.status_code == 200:
                coin_data = response.json()
                
                # Extract relevant market data
                market_data = coin_data.get('market_data', {})
                
                formatted_data = {
                    "id": coin_data['id'],
                    "name": coin_data['name'],
                    "symbol": coin_data['symbol'],
                    "current_price": market_data.get('current_price', {}).get('usd'),
                    "market_cap": market_data.get('market_cap', {}).get('usd'),
                    "market_cap_rank": market_data.get('market_cap_rank'),
                    "total_volume": market_data.get('total_volume', {}).get('usd'),
                    "price_change_24h": market_data.get('price_change_24h'),
                    "price_change_percentage_24h": market_data.get('price_change_percentage_24h'),
                    "circulating_supply": market_data.get('circulating_supply'),
                    "total_supply": market_data.get('total_supply'),
                    "ath": market_data.get('ath', {}).get('usd'),
                    "atl": market_data.get('atl', {}).get('usd'),
                    "description": coin_data.get('description', {}).get('en', '')[:500] + "..." if coin_data.get('description', {}).get('en') else ""
                }
                
                return formatted_data
            else:
                raise HTTPException(status_code=404, detail="Coin not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching coin details: {str(e)}")

# Chat endpoints
@app.post("/api/chat/ask", response_model=ChatResponse)
async def ask_crypto_question(request: ChatRequest):
    """
    Process crypto-related questions using Claude AI
    For now, this is a simplified version - full MCP integration coming next
    """
    try:
        # Import Anthropic here to avoid issues if API key is missing
        try:
            from anthropic import Anthropic
            anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except Exception:
            return ChatResponse(
                answer="AI chat is currently unavailable. Please check back later.",
                market_data={},
                confidence=0.0
            )
        
        system_prompt = """You are Tokenetics, an AI crypto trading assistant. You help users understand cryptocurrency markets and make informed trading decisions using natural language.

Key guidelines:
- Provide clear, beginner-friendly explanations
- Include relevant market context when possible
- Suggest actionable insights
- Always mention this is not financial advice
- Be conversational and helpful

User question: {question}"""

        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system=system_prompt.format(question=request.question),
            messages=[
                {
                    "role": "user",
                    "content": request.question
                }
            ]
        )
        
        # Handle different types of response content blocks
        answer = ""
        for content_block in response.content:
            if content_block.type == "text":
                answer += content_block.text
            elif content_block.type == "tool_use":
                # Handle tool use blocks (for future MCP integration)
                answer += f"[Tool used: {content_block.name}]"
            else:
                answer += str(content_block)

        # Fallback if no content
        if not answer:
            answer = "I'm sorry, I couldn't process your question. Please try again."
        
        return ChatResponse(
            answer=answer,
            market_data={},  # Will be populated with MCP data later
            confidence=0.8   # Placeholder confidence score
        )
        
    except Exception as e:
        return ChatResponse(
            answer="Sorry, I encountered an error processing your question. Please try again.",
            market_data={},
            confidence=0.0
        )

@app.get("/api/chat/health")
async def chat_health():
    """Health check for chat service"""
    try:
        # Check if Anthropic API key is available
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return {
                "status": "healthy",
                "claude_api": "configured",
                "model": "claude-3-sonnet-20240229"
            }
        else:
            return {
                "status": "warning",
                "claude_api": "no_api_key",
                "message": "Anthropic API key not configured"
            }
    except Exception as e:
        return {
            "status": "error",
            "claude_api": "error",
            "error": str(e)
        }

# Test endpoint
@app.get("/api/test")
async def test_endpoint():
    return {"test": "working", "status": "ok", "message": "All systems operational"}

# Vercel ASGI handler - REQUIRED for Vercel deployment
handler = Mangum(app)

# Local development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
