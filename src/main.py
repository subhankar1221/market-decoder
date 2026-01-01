#!/usr/bin/env python3
"""
Market Decoder Pro - Main Entry Point
Complete market analysis system with Greeks calculation
"""

import asyncio
import signal
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.scheduler import MarketScheduler
from src.core.state_manager import SystemState
from src.data.fetcher import DataFetcher
from src.analytics.options.greeks import GreeksCalculator
from src.analytics.options.chain_analyzer import OptionChainAnalyzer
from src.analytics.market.sentiment import MarketSentimentAnalyzer
from src.storage.database import MarketDatabase
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

class MarketDecoderPro:
    """Main Market Decoder Pro Application"""
    
    def __init__(self, config_path="configs/defaults.yaml"):
        self.config_path = config_path
        self.state = None
        self.fetcher = None
        self.scheduler = None
        self.database = None
        self.analyzers = {}
        self.running = False
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing Market Decoder Pro...")
            
            # 1. Initialize system state
            self.state = SystemState(self.config_path)
            await self.state.load_config()
            
            # 2. Setup database
            self.database = MarketDatabase(self.state.config)
            await self.database.initialize()
            
            # 3. Initialize data fetcher
            self.fetcher = DataFetcher(self.state.config)
            await self.fetcher.initialize()
            
            # 4. Initialize analyzers
            await self.initialize_analyzers()
            
            # 5. Initialize scheduler
            self.scheduler = MarketScheduler(self.state, self)
            
            logger.info("Market Decoder Pro initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    async def initialize_analyzers(self):
        """Initialize all analysis modules"""
        try:
            # Greeks Calculator
            self.analyzers['greeks'] = GreeksCalculator(self.state.config)
            
            # Option Chain Analyzer
            self.analyzers['options'] = OptionChainAnalyzer(self.state.config)
            
            # Market Sentiment Analyzer
            self.analyzers['sentiment'] = MarketSentimentAnalyzer(self.state.config)
            
            # Risk Analyzer
            from src.analytics.risk.risk_metrics import RiskAnalyzer
            self.analyzers['risk'] = RiskAnalyzer(self.state.config)
            
            # Technical Analyzer
            from src.analytics.technical.support_resistance import SupportResistanceAnalyzer
            self.analyzers['technical'] = SupportResistanceAnalyzer(self.state.config)
            
            logger.info(f"Initialized {len(self.analyzers)} analyzers")
            
        except Exception as e:
            logger.error(f"Failed to initialize analyzers: {e}")
            raise
    
    async def run_full_analysis(self):
        """Run complete market analysis cycle"""
        try:
            logger.info("Starting full market analysis cycle...")
            
            # 1. Fetch all market data
            market_data = await self.fetcher.fetch_all()
            if not market_data:
                logger.warning("No market data fetched")
                return
            
            # 2. Process option chain data
            option_analysis = await self.analyze_options(market_data)
            
            # 3. Calculate Greeks for all strikes
            greeks_analysis = await self.calculate_greeks(market_data, option_analysis)
            
            # 4. Analyze market sentiment
            sentiment_analysis = await self.analyze_sentiment(market_data)
            
            # 5. Calculate risk metrics
            risk_analysis = await self.analyze_risk(market_data, option_analysis, greeks_analysis)
            
            # 6. Generate trading signals
            signals = await self.generate_signals(market_data, option_analysis, sentiment_analysis)
            
            # 7. Compile comprehensive report
            report = await self.compile_report(
                market_data, 
                option_analysis, 
                greeks_analysis, 
                sentiment_analysis, 
                risk_analysis, 
                signals
            )
            
            # 8. Save to database and output files
            await self.save_analysis(report)
            
            # 9. Display results
            await self.display_results(report)
            
            logger.info("Full analysis cycle completed")
            return report
            
        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}")
            return None
    
    async def analyze_options(self, market_data):
        """Analyze option chain data"""
        try:
            option_data = market_data.get('option_chain', {})
            if not option_data:
                return {}
            
            analysis = await self.analyzers['options'].analyze(option_data)
            return analysis
            
        except Exception as e:
            logger.error(f"Option analysis failed: {e}")
            return {}
    
    async def calculate_greeks(self, market_data, option_analysis):
        """Calculate Greeks for option chain"""
        try:
            spot_price = market_data.get('spot_price', 0)
            option_chain = market_data.get('option_chain', {}).get('records', {}).get('data', [])
            
            if not option_chain or spot_price <= 0:
                return {}
            
            greeks = await self.analyzers['greeks'].calculate_all(
                option_chain=option_chain,
                spot_price=spot_price,
                risk_free_rate=self.state.config['analysis']['greeks']['risk_free_rate'],
                dividend_yield=self.state.config['analysis']['greeks']['dividend_yield']
            )
            
            return greeks
            
        except Exception as e:
            logger.error(f"Greeks calculation failed: {e}")
            return {}
    
    async def analyze_sentiment(self, market_data):
        """Analyze market sentiment"""
        try:
            sentiment = await self.analyzers['sentiment'].analyze(market_data)
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {}
    
    async def analyze_risk(self, market_data, option_analysis, greeks_analysis):
        """Analyze market risk"""
        try:
            risk = await self.analyzers['risk'].analyze(
                market_data=market_data,
                option_analysis=option_analysis,
                greeks_analysis=greeks_analysis
            )
            return risk
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {}
    
    async def generate_signals(self, market_data, option_analysis, sentiment_analysis):
        """Generate trading signals"""
        try:
            from src.analytics.technical.signals import SignalGenerator
            signal_generator = SignalGenerator(self.state.config)
            
            signals = await signal_generator.generate(
                market_data=market_data,
                option_analysis=option_analysis,
                sentiment=sentiment_analysis
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return []
    
    async def compile_report(self, market_data, option_analysis, greeks_analysis, 
                            sentiment_analysis, risk_analysis, signals):
        """Compile comprehensive analysis report"""
        try:
            report = {
                'timestamp': market_data.get('timestamp'),
                'symbol': self.state.config['market']['symbols']['primary'],
                
                'market_data': {
                    'spot_price': market_data.get('spot_price'),
                    'indices': market_data.get('indices', {}),
                    'statistics': market_data.get('statistics', {}),
                    'turnover': market_data.get('turnover', {})
                },
                
                'option_analysis': option_analysis,
                'greeks_analysis': greeks_analysis,
                'sentiment_analysis': sentiment_analysis,
                'risk_analysis': risk_analysis,
                'signals': signals,
                
                'metadata': {
                    'analysis_version': '1.0',
                    'processing_time': None,  # Will be set later
                    'data_sources': list(market_data.keys())
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report compilation failed: {e}")
            return {}
    
    async def save_analysis(self, report):
        """Save analysis to database and files"""
        try:
            if self.database and report:
                await self.database.save_analysis(report)
            
            # Save to JSON file
            output_path = self.state.config['storage']['files']['output_path']
            os.makedirs(output_path, exist_ok=True)
            
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_path}/analysis_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Analysis saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
    
    async def display_results(self, report):
        """Display analysis results"""
        try:
            from src.utils.formatters import ConsoleFormatter
            formatter = ConsoleFormatter(self.state.config)
            
            await formatter.display_report(report)
            
        except Exception as e:
            logger.error(f"Failed to display results: {e}")
    
    async def start(self):
        """Start the market decoder"""
        try:
            # Initialize
            if not await self.initialize():
                logger.error("Initialization failed")
                return
            
            self.running = True
            logger.info("Market Decoder Pro started")
            
            # Start scheduler
            await self.scheduler.start()
            
            # Keep running
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Market Decoder Pro...")
        
        self.running = False
        
        # Stop scheduler
        if self.scheduler:
            await self.scheduler.stop()
        
        # Close database
        if self.database:
            await self.database.close()
        
        # Close fetcher
        if self.fetcher:
            await self.fetcher.close()
        
        logger.info("Market Decoder Pro shutdown complete")

def main():
    """Main function"""
    # Setup logging
    setup_logger()
    
    # Create application
    decoder = MarketDecoderPro()
    
    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(decoder.shutdown())
        )
    
    try:
        # Run application
        loop.run_until_complete(decoder.start())
    except KeyboardInterrupt:
        loop.run_until_complete(decoder.shutdown())
    finally:
        loop.close()

if __name__ == "__main__":
    main()