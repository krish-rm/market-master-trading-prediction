"""
Simple LLM module for Market Master.
Provides mock trading coach functionality.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TradingPersona:
    """Trading persona enum."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class MockTradingCoach:
    """Mock trading coach for demo purposes."""
    
    def __init__(self, persona: TradingPersona = TradingPersona.MODERATE):
        """Initialize the trading coach."""
        self.persona = persona
        logger.info(f"Mock trading coach initialized with persona: {persona}")
    
    def get_advice(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get trading advice based on market data."""
        # Simple mock advice
        advice = {
            'timestamp': datetime.now().isoformat(),
            'persona': self.persona,
            'recommendation': 'HOLD',
            'confidence': 0.7,
            'reasoning': f'Market conditions suggest {self.persona} approach',
            'risk_level': 'medium'
        }
        
        logger.info(f"Generated trading advice: {advice['recommendation']}")
        return advice


def get_trading_advice(market_data: Dict[str, Any], persona: TradingPersona = TradingPersona.MODERATE) -> Dict[str, Any]:
    """Get trading advice."""
    coach = MockTradingCoach(persona)
    return coach.get_advice(market_data) 