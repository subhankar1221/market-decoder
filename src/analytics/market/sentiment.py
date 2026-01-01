"""
Market Sentiment Analyzer - Analyzes overall market sentiment from multiple data sources
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
from collections import defaultdict

from src.utils.logger import get_logger
from src.utils.math_utils import calculate_sma, calculate_ema, calculate_rsi, calculate_macd

logger = get_logger(__name__)

class MarketSentimentAnalyzer:
    """Comprehensive market sentiment analysis from multiple indicators"""
    
    def __init__(self, config):
        self.config = config
        self.sentiment_config = config['analysis']['market']['sentiment']
        
        # Sentiment weights
        self.weights = {
            'pcr': 0.25,
            'market_breadth': 0.20,
            'vix_analysis': 0.20,
            'iv_percentile': 0.15,
            'fii_dii': 0.10,
            'price_action': 0.10
        }
        
        # Historical data storage (simplified - in production would use database)
        self.historical_data = {
            'pcr_history': [],
            'vix_history': [],
            'advance_decline_history': [],
            'timestamp_history': []
        }
        
    async def analyze(self, market_data: Dict) -> Dict:
        """
        Analyze market sentiment from all available data
        
        Args:
            market_data: Dictionary containing all market data sources
            
        Returns:
            Comprehensive sentiment analysis
        """
        try:
            logger.info("Starting market sentiment analysis...")
            
            # Extract data from different sources
            extracted_data = self._extract_market_data(market_data)
            
            if not extracted_data:
                logger.warning("No market data available for sentiment analysis")
                return self._get_default_sentiment()
            
            # Update historical data
            self._update_historical_data(extracted_data)
            
            # Calculate sentiment components
            sentiment_components = await self._calculate_sentiment_components(extracted_data)
            
            # Calculate overall sentiment
            overall_sentiment = await self._calculate_overall_sentiment(sentiment_components)
            
            # Generate sentiment signals
            signals = await self._generate_sentiment_signals(sentiment_components, overall_sentiment)
            
            # Compile final analysis
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'overall_sentiment': overall_sentiment,
                'components': sentiment_components,
                'signals': signals,
                'confidence': self._calculate_confidence(sentiment_components),
                'trend_analysis': await self._analyze_sentiment_trends(),
                'extreme_indicators': await self._check_extreme_indicators(sentiment_components),
                'metadata': {
                    'data_sources_used': list(extracted_data.keys()),
                    'weights_applied': self.weights
                }
            }
            
            logger.info("Market sentiment analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return self._get_default_sentiment()
    
    def _extract_market_data(self, market_data: Dict) -> Dict:
        """Extract relevant data from market data dictionary"""
        extracted = {}
        
        # Extract from option chain
        if 'option_chain' in market_data:
            option_data = market_data['option_chain']
            extracted['spot_price'] = option_data.get('spot_price', 0)
            extracted['vix'] = option_data.get('vix', {}).get('value', 0)
            extracted['vix_change'] = option_data.get('vix', {}).get('change', 0)
            
            # Extract PCR from option chain summary
            summary = option_data.get('summary', {})
            extracted['pcr_oi'] = summary.get('pcr_oi', 1.0)
            extracted['pcr_volume'] = summary.get('pcr_volume', 1.0)
        
        # Extract from market statistics
        if 'market_statistics' in market_data:
            stats = market_data['market_statistics']
            extracted['advances'] = stats.get('advances', 0)
            extracted['declines'] = stats.get('declines', 0)
            extracted['unchanged'] = stats.get('unchanged', 0)
            extracted['total_securities'] = stats.get('total_securities', 0)
            
            # Extract market PCR
            pcr_data = stats.get('pcr', {})
            extracted['market_pcr_total'] = pcr_data.get('total', 0)
            extracted['market_pcr_index'] = pcr_data.get('index', 0)
            extracted['market_pcr_stock'] = pcr_data.get('stock', 0)
            
            # Calculate advance/decline ratio
            if extracted['declines'] > 0:
                extracted['advance_decline_ratio'] = extracted['advances'] / extracted['declines']
            else:
                extracted['advance_decline_ratio'] = 0
        
        # Extract from indices
        if 'indices' in market_data:
            indices = market_data['indices']
            
            # Find NIFTY index
            nifty_data = None
            for index in indices.get('indices', []):
                if index.get('name') == 'NIFTY 50':
                    nifty_data = index
                    break
            
            if nifty_data:
                extracted['nifty_price'] = nifty_data.get('last_price', 0)
                extracted['nifty_change'] = nifty_data.get('change', 0)
                extracted['nifty_percent_change'] = nifty_data.get('percent_change', 0)
                extracted['nifty_pe'] = nifty_data.get('pe_ratio', 0)
                extracted['nifty_pb'] = nifty_data.get('pb_ratio', 0)
                extracted['nifty_div_yield'] = nifty_data.get('dividend_yield', 0)
        
        # Extract from FII/DII data
        if 'fii_dii' in market_data:
            fii_dii = market_data['fii_dii']
            extracted['fii_net'] = fii_dii.get('fii', {}).get('activity', {}).get('total', {}).get('net', 0)
            extracted['dii_net'] = fii_dii.get('dii', {}).get('activity', {}).get('total', {}).get('net', 0)
            extracted['fii_dii_sentiment'] = fii_dii.get('summary', {}).get('sentiment', 'neutral')
        
        # Extract from turnover data
        if 'market_turnover' in market_data:
            turnover = market_data['market_turnover']
            extracted['total_turnover'] = turnover.get('total_turnover', 0)
            extracted['fo_turnover'] = turnover.get('segments', {}).get('f&o', {}).get('turnover', 0)
            
            # Calculate turnover ratios
            if extracted['total_turnover'] > 0:
                extracted['fo_turnover_ratio'] = extracted['fo_turnover'] / extracted['total_turnover']
            else:
                extracted['fo_turnover_ratio'] = 0
        
        return extracted
    
    def _update_historical_data(self, current_data: Dict):
        """Update historical data storage"""
        timestamp = datetime.now()
        
        # Store current data point
        self.historical_data['pcr_history'].append({
            'timestamp': timestamp,
            'pcr_oi': current_data.get('pcr_oi', 1.0),
            'pcr_volume': current_data.get('pcr_volume', 1.0)
        })
        
        self.historical_data['vix_history'].append({
            'timestamp': timestamp,
            'vix': current_data.get('vix', 0),
            'vix_change': current_data.get('vix_change', 0)
        })
        
        self.historical_data['advance_decline_history'].append({
            'timestamp': timestamp,
            'advances': current_data.get('advances', 0),
            'declines': current_data.get('declines', 0),
            'ratio': current_data.get('advance_decline_ratio', 0)
        })
        
        self.historical_data['timestamp_history'].append(timestamp)
        
        # Keep only last 100 data points to manage memory
        for key in self.historical_data:
            if isinstance(self.historical_data[key], list):
                self.historical_data[key] = self.historical_data[key][-100:]
    
    async def _calculate_sentiment_components(self, data: Dict) -> Dict:
        """Calculate all sentiment components"""
        components = {}
        
        # 1. PCR Analysis
        components['pcr_analysis'] = await self._analyze_pcr(data)
        
        # 2. Market Breadth Analysis
        components['market_breadth'] = await self._analyze_market_breadth(data)
        
        # 3. VIX Analysis
        components['vix_analysis'] = await self._analyze_vix(data)
        
        # 4. IV Percentile Analysis
        components['iv_percentile'] = await self._analyze_iv_percentile(data)
        
        # 5. FII/DII Analysis
        components['fii_dii'] = await self._analyze_fii_dii(data)
        
        # 6. Price Action Analysis
        components['price_action'] = await self._analyze_price_action(data)
        
        # 7. Additional Indicators
        components['additional_indicators'] = await self._analyze_additional_indicators(data)
        
        return components
    
    async def _analyze_pcr(self, data: Dict) -> Dict:
        """Analyze Put-Call Ratio sentiment"""
        try:
            pcr_oi = data.get('pcr_oi', 1.0)
            pcr_volume = data.get('pcr_volume', 1.0)
            market_pcr = data.get('market_pcr_total', 0)
            
            # Calculate PCR scores
            pcr_oi_score = self._calculate_pcr_score(pcr_oi, 'oi')
            pcr_volume_score = self._calculate_pcr_score(pcr_volume, 'volume')
            market_pcr_score = self._calculate_pcr_score(market_pcr, 'market') if market_pcr > 0 else 0
            
            # Weighted average score
            total_weight = 0.5 + 0.3 + (0.2 if market_pcr > 0 else 0)
            weighted_score = (pcr_oi_score * 0.5 + pcr_volume_score * 0.3 + market_pcr_score * 0.2) / total_weight
            
            # Determine sentiment
            sentiment, strength = self._score_to_sentiment(weighted_score)
            
            # Calculate trend
            trend = self._calculate_pcr_trend(pcr_oi)
            
            return {
                'pcr_oi': pcr_oi,
                'pcr_volume': pcr_volume,
                'market_pcr': market_pcr,
                'pcr_oi_score': pcr_oi_score,
                'pcr_volume_score': pcr_volume_score,
                'market_pcr_score': market_pcr_score,
                'weighted_score': weighted_score,
                'sentiment': sentiment,
                'strength': strength,
                'trend': trend,
                'interpretation': self._interpret_pcr(pcr_oi),
                'confidence': self._calculate_pcr_confidence(data)
            }
            
        except Exception as e:
            logger.error(f"PCR analysis failed: {e}")
            return self._get_default_pcr_analysis()
    
    def _calculate_pcr_score(self, pcr: float, pcr_type: str) -> float:
        """Convert PCR value to sentiment score (-1 to +1)"""
        # Different thresholds for different PCR types
        if pcr_type == 'oi':
            if pcr < 0.7:
                return 1.0  # Very bullish
            elif pcr < 0.9:
                return 0.5  # Bullish
            elif pcr < 1.1:
                return 0.0  # Neutral
            elif pcr < 1.3:
                return -0.5  # Bearish
            else:
                return -1.0  # Very bearish
        elif pcr_type == 'volume':
            if pcr < 0.8:
                return 1.0
            elif pcr < 1.0:
                return 0.5
            elif pcr < 1.2:
                return 0.0
            elif pcr < 1.4:
                return -0.5
            else:
                return -1.0
        else:  # market
            if pcr < 0.9:
                return 1.0
            elif pcr < 1.1:
                return 0.5
            elif pcr < 1.3:
                return 0.0
            elif pcr < 1.5:
                return -0.5
            else:
                return -1.0
    
    def _calculate_pcr_trend(self, current_pcr: float) -> str:
        """Calculate PCR trend"""
        if len(self.historical_data['pcr_history']) < 2:
            return 'neutral'
        
        # Get recent PCR values
        recent_pcrs = [point['pcr_oi'] for point in self.historical_data['pcr_history'][-5:]]
        
        if len(recent_pcrs) < 2:
            return 'neutral'
        
        # Calculate slope
        x = list(range(len(recent_pcrs)))
        y = recent_pcrs
        
        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 'neutral'
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        if slope < -0.05:
            return 'falling'
        elif slope > 0.05:
            return 'rising'
        else:
            return 'stable'
    
    def _interpret_pcr(self, pcr: float) -> str:
        """Interpret PCR value"""
        if pcr < 0.7:
            return "Extreme Bullish - Market overly optimistic"
        elif pcr < 0.9:
            return "Bullish - More calls than puts"
        elif pcr < 1.1:
            return "Neutral - Balanced market"
        elif pcr < 1.3:
            return "Bearish - More puts than calls"
        elif pcr < 1.5:
            return "Very Bearish - Heavy put buying"
        else:
            return "Extreme Bearish - Panic put buying"
    
    def _calculate_pcr_confidence(self, data: Dict) -> float:
        """Calculate confidence in PCR analysis"""
        confidence_factors = []
        
        # Data availability
        if data.get('pcr_oi') and data.get('pcr_volume'):
            confidence_factors.append(0.8)
        elif data.get('pcr_oi'):
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Historical consistency
        if len(self.historical_data['pcr_history']) > 10:
            recent_pcrs = [point['pcr_oi'] for point in self.historical_data['pcr_history'][-10:]]
            if len(recent_pcrs) > 5:
                std_dev = statistics.stdev(recent_pcrs) if len(recent_pcrs) > 1 else 0
                if std_dev < 0.2:
                    confidence_factors.append(0.9)
                elif std_dev < 0.4:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.5)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    async def _analyze_market_breadth(self, data: Dict) -> Dict:
        """Analyze market breadth sentiment"""
        try:
            advances = data.get('advances', 0)
            declines = data.get('declines', 0)
            unchanged = data.get('unchanged', 0)
            total = data.get('total_securities', 0)
            ad_ratio = data.get('advance_decline_ratio', 0)
            
            if total == 0:
                return self._get_default_breadth_analysis()
            
            # Calculate breadth metrics
            advance_percent = (advances / total) * 100
            decline_percent = (declines / total) * 100
            unchanged_percent = (unchanged / total) * 100
            
            # Calculate breadth score
            breadth_score = self._calculate_breadth_score(advance_percent, decline_percent, ad_ratio)
            
            # Determine sentiment
            sentiment, strength = self._score_to_sentiment(breadth_score)
            
            # Calculate additional breadth indicators
            trin_index = self._calculate_trin_index(advances, declines, data)
            arms_index = self._calculate_arms_index(advances, declines, data)
            
            return {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'total_securities': total,
                'advance_percent': advance_percent,
                'decline_percent': decline_percent,
                'advance_decline_ratio': ad_ratio,
                'breadth_score': breadth_score,
                'sentiment': sentiment,
                'strength': strength,
                'trin_index': trin_index,
                'arms_index': arms_index,
                'interpretation': self._interpret_breadth(advance_percent, decline_percent, ad_ratio),
                'confidence': self._calculate_breadth_confidence(data)
            }
            
        except Exception as e:
            logger.error(f"Market breadth analysis failed: {e}")
            return self._get_default_breadth_analysis()
    
    def _calculate_breadth_score(self, advance_percent: float, decline_percent: float, ad_ratio: float) -> float:
        """Calculate market breadth score (-1 to +1)"""
        # Score based on advance percentage
        if advance_percent > 70:
            score1 = 1.0
        elif advance_percent > 60:
            score1 = 0.7
        elif advance_percent > 50:
            score1 = 0.3
        elif advance_percent > 40:
            score1 = 0.0
        elif advance_percent > 30:
            score1 = -0.3
        elif advance_percent > 20:
            score1 = -0.7
        else:
            score1 = -1.0
        
        # Score based on advance/decline ratio
        if ad_ratio > 2.0:
            score2 = 1.0
        elif ad_ratio > 1.5:
            score2 = 0.7
        elif ad_ratio > 1.2:
            score2 = 0.3
        elif ad_ratio > 0.8:
            score2 = 0.0
        elif ad_ratio > 0.5:
            score2 = -0.3
        elif ad_ratio > 0.3:
            score2 = -0.7
        else:
            score2 = -1.0
        
        # Combine scores
        return (score1 * 0.6 + score2 * 0.4)
    
    def _calculate_trin_index(self, advances: int, declines: int, data: Dict) -> float:
        """Calculate TRIN (Trading Index) aka Arms Index"""
        # TRIN = (Advances/Declines) / (Advance Volume/Decline Volume)
        # Simplified version without volume
        if declines == 0:
            return 0
        
        advance_volume = data.get('advance_volume', advances * 1000)  # Placeholder
        decline_volume = data.get('decline_volume', declines * 1000)   # Placeholder
        
        if decline_volume == 0:
            return 0
        
        advance_decline_ratio = advances / declines
        volume_ratio = advance_volume / decline_volume
        
        if volume_ratio == 0:
            return 0
        
        return advance_decline_ratio / volume_ratio
    
    def _calculate_arms_index(self, advances: int, declines: int, data: Dict) -> float:
        """Calculate Arms Index (same as TRIN)"""
        # Same calculation as TRIN
        return self._calculate_trin_index(advances, declines, data)
    
    def _interpret_breadth(self, advance_percent: float, decline_percent: float, ad_ratio: float) -> str:
        """Interpret market breadth"""
        if advance_percent > 70 and ad_ratio > 2.0:
            return "Extremely Broad Rally - Strong bullish participation"
        elif advance_percent > 60 and ad_ratio > 1.5:
            return "Broad Rally - Healthy bullish breadth"
        elif advance_percent > 50:
            return "Positive Breadth - More advancing stocks"
        elif advance_percent < 30 and ad_ratio < 0.5:
            return "Broad Decline - Widespread selling"
        elif advance_percent < 40:
            return "Negative Breadth - More declining stocks"
        else:
            return "Neutral Breadth - Mixed market participation"
    
    def _calculate_breadth_confidence(self, data: Dict) -> float:
        """Calculate confidence in breadth analysis"""
        confidence_factors = []
        
        # Data completeness
        if all(k in data for k in ['advances', 'declines', 'total_securities']):
            confidence_factors.append(0.9)
        elif 'advances' in data and 'declines' in data:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # Market coverage
        total = data.get('total_securities', 0)
        if total > 1000:
            confidence_factors.append(0.9)
        elif total > 500:
            confidence_factors.append(0.7)
        elif total > 100:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    async def _analyze_vix(self, data: Dict) -> Dict:
        """Analyze VIX (Fear Index) sentiment"""
        try:
            vix = data.get('vix', 0)
            vix_change = data.get('vix_change', 0)
            
            if vix == 0:
                return self._get_default_vix_analysis()
            
            # Calculate VIX score
            vix_score = self._calculate_vix_score(vix, vix_change)
            
            # Determine sentiment
            sentiment, strength = self._score_to_sentiment(vix_score)
            
            # Calculate VIX trend
            trend = self._calculate_vix_trend(vix, vix_change)
            
            # Calculate VIX percentile (simplified)
            vix_percentile = self._calculate_vix_percentile(vix)
            
            return {
                'vix': vix,
                'vix_change': vix_change,
                'vix_score': vix_score,
                'sentiment': sentiment,
                'strength': strength,
                'trend': trend,
                'vix_percentile': vix_percentile,
                'interpretation': self._interpret_vix(vix, vix_change),
                'confidence': self._calculate_vix_confidence(data)
            }
            
        except Exception as e:
            logger.error(f"VIX analysis failed: {e}")
            return self._get_default_vix_analysis()
    
    def _calculate_vix_score(self, vix: float, vix_change: float) -> float:
        """Calculate VIX sentiment score (-1 to +1)"""
        # Lower VIX = bullish (complacency), Higher VIX = bearish (fear)
        # VIX change adds dynamic component
        
        # Base score from VIX level
        if vix < 12:
            base_score = 1.0  # Very bullish (low fear)
        elif vix < 15:
            base_score = 0.5  # Bullish
        elif vix < 20:
            base_score = 0.0  # Neutral
        elif vix < 25:
            base_score = -0.5  # Bearish
        else:
            base_score = -1.0  # Very bearish (high fear)
        
        # Adjustment from VIX change
        if vix_change > 10:
            adjustment = -0.3  # Fear increasing rapidly
        elif vix_change > 5:
            adjustment = -0.2
        elif vix_change > 2:
            adjustment = -0.1
        elif vix_change < -10:
            adjustment = 0.3  # Fear decreasing rapidly
        elif vix_change < -5:
            adjustment = 0.2
        elif vix_change < -2:
            adjustment = 0.1
        else:
            adjustment = 0.0
        
        return max(-1, min(1, base_score + adjustment))
    
    def _calculate_vix_trend(self, current_vix: float, vix_change: float) -> str:
        """Calculate VIX trend"""
        if len(self.historical_data['vix_history']) < 2:
            return 'neutral' if abs(vix_change) < 2 else 'rising' if vix_change > 0 else 'falling'
        
        # Get recent VIX values
        recent_vix = [point['vix'] for point in self.historical_data['vix_history'][-5:]]
        
        if len(recent_vix) < 2:
            return 'neutral'
        
        # Calculate slope
        x = list(range(len(recent_vix)))
        y = recent_vix
        
        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 'neutral'
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        if slope > 0.2:
            return 'rising'
        elif slope < -0.2:
            return 'falling'
        else:
            return 'stable'
    
    def _calculate_vix_percentile(self, current_vix: float) -> float:
        """Calculate VIX percentile (simplified)"""
        # In production, would use historical VIX data
        # Simplified version based on typical ranges
        
        if current_vix < 12:
            return 20  # Low VIX
        elif current_vix < 15:
            return 40  # Below average
        elif current_vix < 20:
            return 60  # Average
        elif current_vix < 25:
            return 75  # Above average
        elif current_vix < 30:
            return 85  # High
        else:
            return 95  # Very high
    
    def _interpret_vix(self, vix: float, vix_change: float) -> str:
        """Interpret VIX value"""
        if vix < 12:
            if vix_change > 5:
                return "Extreme Complacency with Rising Fear"
            else:
                return "Extreme Complacency - Market overly confident"
        
        elif vix < 15:
            if vix_change > 5:
                return "Complacency with Fear Building"
            else:
                return "Complacent Market - Low fear levels"
        
        elif vix < 20:
            if vix_change > 5:
                return "Normal Volatility with Rising Fear"
            elif vix_change < -5:
                return "Normal Volatility with Declining Fear"
            else:
                return "Normal Market Volatility"
        
        elif vix < 25:
            if vix_change > 5:
                return "Elevated Fear - Caution advised"
            else:
                return "Elevated Fear Levels"
        
        elif vix < 30:
            if vix_change > 5:
                return "High Fear - Panic building"
            else:
                return "High Fear - Market nervous"
        
        else:
            if vix_change > 5:
                return "Extreme Panic - Market in fear"
            else:
                return "Extreme Fear - Panic levels"
    
    def _calculate_vix_confidence(self, data: Dict) -> float:
        """Calculate confidence in VIX analysis"""
        confidence_factors = []
        
        # VIX data availability
        if data.get('vix') and data.get('vix_change'):
            confidence_factors.append(0.9)
        elif data.get('vix'):
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Historical data availability
        if len(self.historical_data['vix_history']) > 20:
            confidence_factors.append(0.8)
        elif len(self.historical_data['vix_history']) > 10:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    async def _analyze_iv_percentile(self, data: Dict) -> Dict:
        """Analyze Implied Volatility percentile sentiment"""
        try:
            # Get IV percentile from option data
            # This would come from option chain analysis
            iv_percentile = data.get('iv_percentile', 50)  # Default 50%
            
            # Calculate IV score
            iv_score = self._calculate_iv_percentile_score(iv_percentile)
            
            # Determine sentiment
            sentiment, strength = self._score_to_sentiment(iv_score)
            
            return {
                'iv_percentile': iv_percentile,
                'iv_score': iv_score,
                'sentiment': sentiment,
                'strength': strength,
                'interpretation': self._interpret_iv_percentile(iv_percentile),
                'confidence': 0.7  # Fixed confidence for now
            }
            
        except Exception as e:
            logger.error(f"IV percentile analysis failed: {e}")
            return self._get_default_iv_analysis()
    
    def _calculate_iv_percentile_score(self, iv_percentile: float) -> float:
        """Convert IV percentile to sentiment score (-1 to +1)"""
        # High IV percentile = expensive options = bearish for option buyers
        # Low IV percentile = cheap options = bullish for option buyers
        
        if iv_percentile > 80:
            return -0.8  # Very bearish (options expensive)
        elif iv_percentile > 70:
            return -0.4  # Bearish
        elif iv_percentile > 60:
            return -0.2  # Mildly bearish
        elif iv_percentile > 40:
            return 0.0   # Neutral
        elif iv_percentile > 30:
            return 0.2   # Mildly bullish
        elif iv_percentile > 20:
            return 0.4   # Bullish
        else:
            return 0.8   # Very bullish (options cheap)
    
    def _interpret_iv_percentile(self, iv_percentile: float) -> str:
        """Interpret IV percentile"""
        if iv_percentile > 90:
            return "Extreme IV - Options extremely expensive, high fear premium"
        elif iv_percentile > 80:
            return "Very High IV - Options very expensive, significant fear"
        elif iv_percentile > 70:
            return "High IV - Options expensive, elevated fear"
        elif iv_percentile > 60:
            return "Above Average IV - Options somewhat expensive"
        elif iv_percentile > 40:
            return "Average IV - Options fairly priced"
        elif iv_percentile > 30:
            return "Below Average IV - Options somewhat cheap"
        elif iv_percentile > 20:
            return "Low IV - Options cheap, complacency"
        elif iv_percentile > 10:
            return "Very Low IV - Options very cheap, high complacency"
        else:
            return "Extreme Low IV - Options extremely cheap, extreme complacency"
    
    async def _analyze_fii_dii(self, data: Dict) -> Dict:
        """Analyze FII/DII flows sentiment"""
        try:
            fii_net = data.get('fii_net', 0)
            dii_net = data.get('dii_net', 0)
            sentiment = data.get('fii_dii_sentiment', 'neutral')
            
            # Calculate FII/DII score
            fii_dii_score = self._calculate_fii_dii_score(fii_net, dii_net)
            
            # Determine sentiment from score
            score_sentiment, strength = self._score_to_sentiment(fii_dii_score)
            
            # Use provided sentiment if available, otherwise use calculated
            final_sentiment = sentiment if sentiment != 'neutral' else score_sentiment
            
            return {
                'fii_net': fii_net,
                'dii_net': dii_net,
                'total_net': fii_net + dii_net,
                'fii_dii_score': fii_dii_score,
                'sentiment': final_sentiment,
                'strength': strength,
                'interpretation': self._interpret_fii_dii(fii_net, dii_net),
                'confidence': self._calculate_fii_dii_confidence(data)
            }
            
        except Exception as e:
            logger.error(f"FII/DII analysis failed: {e}")
            return self._get_default_fii_dii_analysis()
    
    def _calculate_fii_dii_score(self, fii_net: float, dii_net: float) -> float:
        """Calculate FII/DII sentiment score (-1 to +1)"""
        total_net = fii_net + dii_net
        
        # Score based on total net flow in Crores
        if total_net > 2000:
            return 1.0  # Very bullish
        elif total_net > 1000:
            return 0.7  # Bullish
        elif total_net > 500:
            return 0.3  # Mildly bullish
        elif total_net > -500:
            return 0.0  # Neutral
        elif total_net > -1000:
            return -0.3  # Mildly bearish
        elif total_net > -2000:
            return -0.7  # Bearish
        else:
            return -1.0  # Very bearish
    
    def _interpret_fii_dii(self, fii_net: float, dii_net: float) -> str:
        """Interpret FII/DII flows"""
        total_net = fii_net + dii_net
        
        if fii_net > 1000 and dii_net > 1000:
            return "Strong Institutional Buying - Both FIIs and DIIs buying"
        elif fii_net > 1000:
            return "FII Buying - Foreign investors bullish"
        elif dii_net > 1000:
            return "DII Buying - Domestic investors bullish"
        elif fii_net < -1000 and dii_net < -1000:
            return "Strong Institutional Selling - Both FIIs and DIIs selling"
        elif fii_net < -1000:
            return "FII Selling - Foreign investors bearish"
        elif dii_net < -1000:
            return "DII Selling - Domestic investors bearish"
        elif total_net > 500:
            return "Net Institutional Inflow - Overall buying"
        elif total_net < -500:
            return "Net Institutional Outflow - Overall selling"
        else:
            return "Balanced Institutional Activity - No clear direction"
    
    def _calculate_fii_dii_confidence(self, data: Dict) -> float:
        """Calculate confidence in FII/DII analysis"""
        confidence_factors = []
        
        # Data availability
        if data.get('fii_net') is not None and data.get('dii_net') is not None:
            confidence_factors.append(0.9)
        elif data.get('fii_net') is not None or data.get('dii_net') is not None:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Data magnitude (more reliable for larger flows)
        total_net = abs(data.get('fii_net', 0)) + abs(data.get('dii_net', 0))
        if total_net > 2000:
            confidence_factors.append(0.8)
        elif total_net > 500:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    async def _analyze_price_action(self, data: Dict) -> Dict:
        """Analyze price action sentiment"""
        try:
            nifty_price = data.get('nifty_price', 0)
            nifty_change = data.get('nifty_change', 0)
            nifty_percent_change = data.get('nifty_percent_change', 0)
            
            if nifty_price == 0:
                return self._get_default_price_action_analysis()
            
            # Calculate price action score
            price_score = self._calculate_price_action_score(nifty_percent_change)
            
            # Determine sentiment
            sentiment, strength = self._score_to_sentiment(price_score)
            
            # Calculate trend
            trend = self._calculate_price_trend(nifty_percent_change)
            
            return {
                'nifty_price': nifty_price,
                'nifty_change': nifty_change,
                'nifty_percent_change': nifty_percent_change,
                'price_score': price_score,
                'sentiment': sentiment,
                'strength': strength,
                'trend': trend,
                'interpretation': self._interpret_price_action(nifty_percent_change),
                'confidence': self._calculate_price_action_confidence(data)
            }
            
        except Exception as e:
            logger.error(f"Price action analysis failed: {e}")
            return self._get_default_price_action_analysis()
    
    def _calculate_price_action_score(self, percent_change: float) -> float:
        """Calculate price action sentiment score (-1 to +1)"""
        # Based on daily percent change
        if percent_change > 2.0:
            return 1.0  # Very bullish
        elif percent_change > 1.0:
            return 0.7  # Bullish
        elif percent_change > 0.5:
            return 0.3  # Mildly bullish
        elif percent_change > -0.5:
            return 0.0  # Neutral
        elif percent_change > -1.0:
            return -0.3  # Mildly bearish
        elif percent_change > -2.0:
            return -0.7  # Bearish
        else:
            return -1.0  # Very bearish
    
    def _calculate_price_trend(self, percent_change: float) -> str:
        """Calculate price trend"""
        if percent_change > 1.0:
            return 'strong_uptrend'
        elif percent_change > 0.5:
            return 'uptrend'
        elif percent_change > 0.1:
            return 'mild_uptrend'
        elif percent_change > -0.1:
            return 'sideways'
        elif percent_change > -0.5:
            return 'mild_downtrend'
        elif percent_change > -1.0:
            return 'downtrend'
        else:
            return 'strong_downtrend'
    
    def _interpret_price_action(self, percent_change: float) -> str:
        """Interpret price action"""
        if percent_change > 2.0:
            return "Strong Rally - Bullish momentum"
        elif percent_change > 1.0:
            return "Rally - Positive momentum"
        elif percent_change > 0.5:
            return "Moderate Gains - Upward bias"
        elif percent_change > 0.1:
            return "Slight Gains - Mildly positive"
        elif percent_change > -0.1:
            return "Range Bound - Neutral action"
        elif percent_change > -0.5:
            return "Slight Decline - Mildly negative"
        elif percent_change > -1.0:
            return "Decline - Negative momentum"
        elif percent_change > -2.0:
            return "Strong Decline - Bearish momentum"
        else:
            return "Sharp Decline - Heavy selling"
    
    def _calculate_price_action_confidence(self, data: Dict) -> float:
        """Calculate confidence in price action analysis"""
        confidence_factors = []
        
        # Data availability
        if all(k in data for k in ['nifty_price', 'nifty_change', 'nifty_percent_change']):
            confidence_factors.append(0.9)
        elif 'nifty_price' in data and 'nifty_percent_change' in data:
            confidence_factors.append(0.7)
        elif 'nifty_price' in data:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Market hours (more reliable during market hours)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 15:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    async def _analyze_additional_indicators(self, data: Dict) -> Dict:
        """Analyze additional market indicators"""
        try:
            indicators = {}
            
            # Turnover analysis
            total_turnover = data.get('total_turnover', 0)
            fo_turnover_ratio = data.get('fo_turnover_ratio', 0)
            
            if total_turnover > 0:
                turnover_score = self._calculate_turnover_score(fo_turnover_ratio)
                indicators['turnover'] = {
                    'total_turnover': total_turnover,
                    'fo_turnover_ratio': fo_turnover_ratio,
                    'score': turnover_score,
                    'interpretation': self._interpret_turnover(fo_turnover_ratio)
                }
            
            # PE Ratio analysis
            nifty_pe = data.get('nifty_pe', 0)
            if nifty_pe > 0:
                pe_score = self._calculate_pe_score(nifty_pe)
                indicators['pe_ratio'] = {
                    'pe': nifty_pe,
                    'score': pe_score,
                    'interpretation': self._interpret_pe_ratio(nifty_pe)
                }
            
            # Dividend Yield analysis
            div_yield = data.get('nifty_div_yield', 0)
            if div_yield > 0:
                dy_score = self._calculate_dividend_yield_score(div_yield)
                indicators['dividend_yield'] = {
                    'yield': div_yield,
                    'score': dy_score,
                    'interpretation': self._interpret_dividend_yield(div_yield)
                }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Additional indicators analysis failed: {e}")
            return {}
    
    def _calculate_turnover_score(self, fo_turnover_ratio: float) -> float:
        """Calculate turnover sentiment score"""
        # Higher F&O turnover ratio indicates speculative activity
        if fo_turnover_ratio > 0.8:
            return -0.5  # Bearish (excessive speculation)
        elif fo_turnover_ratio > 0.6:
            return 0.0   # Neutral
        elif fo_turnover_ratio > 0.4:
            return 0.3   # Mildly bullish (healthy activity)
        else:
            return 0.0   # Neutral
    
    def _interpret_turnover(self, fo_turnover_ratio: float) -> str:
        """Interpret turnover ratio"""
        if fo_turnover_ratio > 0.8:
            return "High F&O Turnover - Speculative activity elevated"
        elif fo_turnover_ratio > 0.6:
            return "Elevated F&O Activity - Increased speculation"
        elif fo_turnover_ratio > 0.4:
            return "Normal F&O Activity - Healthy market participation"
        else:
            return "Low F&O Activity - Limited speculative interest"
    
    def _calculate_pe_score(self, pe_ratio: float) -> float:
        """Calculate PE ratio sentiment score"""
        # Lower PE = bullish (cheap), Higher PE = bearish (expensive)
        if pe_ratio > 25:
            return -0.8  # Very bearish (expensive)
        elif pe_ratio > 22:
            return -0.4  # Bearish
        elif pe_ratio > 20:
            return -0.2  # Mildly bearish
        elif pe_ratio > 18:
            return 0.0   # Neutral
        elif pe_ratio > 16:
            return 0.2   # Mildly bullish
        elif pe_ratio > 14:
            return 0.4   # Bullish
        else:
            return 0.8   # Very bullish (cheap)
    
    def _interpret_pe_ratio(self, pe_ratio: float) -> str:
        """Interpret PE ratio"""
        if pe_ratio > 25:
            return "Extremely High PE - Market very expensive"
        elif pe_ratio > 22:
            return "High PE - Market expensive"
        elif pe_ratio > 20:
            return "Above Average PE - Market somewhat expensive"
        elif pe_ratio > 18:
            return "Average PE - Market fairly valued"
        elif pe_ratio > 16:
            return "Below Average PE - Market somewhat cheap"
        elif pe_ratio > 14:
            return "Low PE - Market cheap"
        else:
            return "Very Low PE - Market very cheap"
    
    def _calculate_dividend_yield_score(self, dividend_yield: float) -> float:
        """Calculate dividend yield sentiment score"""
        # Higher dividend yield = bullish (cheap), Lower = bearish (expensive)
        if dividend_yield > 2.0:
            return 0.8   # Very bullish (high yield)
        elif dividend_yield > 1.5:
            return 0.4   # Bullish
        elif dividend_yield > 1.2:
            return 0.2   # Mildly bullish
        elif dividend_yield > 1.0:
            return 0.0   # Neutral
        elif dividend_yield > 0.8:
            return -0.2  # Mildly bearish
        elif dividend_yield > 0.6:
            return -0.4  # Bearish
        else:
            return -0.8  # Very bearish (low yield)
    
    def _interpret_dividend_yield(self, dividend_yield: float) -> str:
        """Interpret dividend yield"""
        if dividend_yield > 2.0:
            return "High Dividend Yield - Attractive valuations"
        elif dividend_yield > 1.5:
            return "Above Average Yield - Reasonable valuations"
        elif dividend_yield > 1.2:
            return "Average Dividend Yield - Fair valuations"
        elif dividend_yield > 1.0:
            return "Below Average Yield - Somewhat expensive"
        elif dividend_yield > 0.8:
            return "Low Dividend Yield - Expensive market"
        else:
            return "Very Low Dividend Yield - Very expensive market"
    
    async def _calculate_overall_sentiment(self, components: Dict) -> Dict:
        """Calculate overall market sentiment"""
        try:
            # Collect scores from all components
            scores = []
            weights = []
            
            # PCR Analysis
            if 'pcr_analysis' in components:
                pcr_data = components['pcr_analysis']
                scores.append(pcr_data.get('weighted_score', 0))
                weights.append(self.weights['pcr'])
            
            # Market Breadth
            if 'market_breadth' in components:
                breadth_data = components['market_breadth']
                scores.append(breadth_data.get('breadth_score', 0))
                weights.append(self.weights['market_breadth'])
            
            # VIX Analysis
            if 'vix_analysis' in components:
                vix_data = components['vix_analysis']
                scores.append(vix_data.get('vix_score', 0))
                weights.append(self.weights['vix_analysis'])
            
            # IV Percentile
            if 'iv_percentile' in components:
                iv_data = components['iv_percentile']
                scores.append(iv_data.get('iv_score', 0))
                weights.append(self.weights['iv_percentile'])
            
            # FII/DII
            if 'fii_dii' in components:
                fii_dii_data = components['fii_dii']
                scores.append(fii_dii_data.get('fii_dii_score', 0))
                weights.append(self.weights['fii_dii'])
            
            # Price Action
            if 'price_action' in components:
                price_data = components['price_action']
                scores.append(price_data.get('price_score', 0))
                weights.append(self.weights['price_action'])
            
            # Additional indicators (if available)
            if 'additional_indicators' in components:
                additional = components['additional_indicators']
                
                # Add turnover score if available
                if 'turnover' in additional:
                    scores.append(additional['turnover'].get('score', 0))
                    weights.append(0.05)  # Small weight
                
                # Add PE ratio score if available
                if 'pe_ratio' in additional:
                    scores.append(additional['pe_ratio'].get('score', 0))
                    weights.append(0.05)  # Small weight
            
            # Calculate weighted average
            if scores and weights:
                weighted_sum = sum(s * w for s, w in zip(scores, weights))
                total_weight = sum(weights)
                overall_score = weighted_sum / total_weight
            else:
                overall_score = 0.0
            
            # Determine sentiment and strength
            sentiment, strength = self._score_to_sentiment(overall_score)
            
            # Calculate consistency
            consistency = self._calculate_sentiment_consistency(scores)
            
            return {
                'overall_score': overall_score,
                'sentiment': sentiment,
                'strength': strength,
                'consistency': consistency,
                'interpretation': self._interpret_overall_sentiment(overall_score, sentiment, strength),
                'component_count': len(scores),
                'weighted_component_count': len([w for w in weights if w > 0])
            }
            
        except Exception as e:
            logger.error(f"Overall sentiment calculation failed: {e}")
            return self._get_default_overall_sentiment()
    
    def _score_to_sentiment(self, score: float) -> Tuple[str, str]:
        """Convert score to sentiment and strength"""
        # Determine sentiment
        if score > 0.7:
            sentiment = "very_bullish"
        elif score > 0.3:
            sentiment = "bullish"
        elif score > -0.3:
            sentiment = "neutral"
        elif score > -0.7:
            sentiment = "bearish"
        else:
            sentiment = "very_bearish"
        
        # Determine strength
        abs_score = abs(score)
        if abs_score > 0.7:
            strength = "very_strong"
        elif abs_score > 0.5:
            strength = "strong"
        elif abs_score > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        return sentiment, strength
    
    def _calculate_sentiment_consistency(self, scores: List[float]) -> float:
        """Calculate consistency across sentiment components"""
        if len(scores) < 2:
            return 1.0  # Perfect consistency with single component
        
        # Count how many scores are in the same direction
        positive_count = sum(1 for s in scores if s > 0.1)
        negative_count = sum(1 for s in scores if s < -0.1)
        neutral_count = len(scores) - positive_count - negative_count
        
        # Calculate consistency ratio
        max_count = max(positive_count, negative_count, neutral_count)
        consistency = max_count / len(scores)
        
        return consistency
    
    def _interpret_overall_sentiment(self, score: float, sentiment: str, strength: str) -> str:
        """Interpret overall sentiment"""
        strength_map = {
            'very_strong': 'Extremely',
            'strong': 'Strongly',
            'moderate': 'Moderately',
            'weak': 'Slightly'
        }
        
        sentiment_map = {
            'very_bullish': 'Bullish',
            'bullish': 'Bullish',
            'neutral': 'Neutral',
            'bearish': 'Bearish',
            'very_bearish': 'Bearish'
        }
        
        strength_word = strength_map.get(strength, '')
        sentiment_word = sentiment_map.get(sentiment, 'Neutral')
        
        if sentiment == 'neutral':
            return f"Market is {sentiment_word} - No clear directional bias"
        else:
            return f"Market is {strength_word} {sentiment_word}"
    
    async def _generate_sentiment_signals(self, components: Dict, overall_sentiment: Dict) -> List[Dict]:
        """Generate sentiment-based trading signals"""
        signals = []
        
        # Check for extreme sentiment signals
        overall_score = overall_sentiment.get('overall_score', 0)
        
        # Extreme bullish signal
        if overall_score > 0.8:
            signals.append({
                'type': 'sentiment_extreme',
                'signal': 'CAUTION_BULLISH',
                'reason': 'Extreme bullish sentiment, potential reversal risk',
                'confidence': 0.7
            })
        
        # Extreme bearish signal
        elif overall_score < -0.8:
            signals.append({
                'type': 'sentiment_extreme',
                'signal': 'CAUTION_BEARISH',
                'reason': 'Extreme bearish sentiment, potential bounce opportunity',
                'confidence': 0.7
            })
        
        # Contrarian signals based on PCR extremes
        pcr_analysis = components.get('pcr_analysis', {})
        pcr_oi = pcr_analysis.get('pcr_oi', 1.0)
        
        if pcr_oi < 0.7:
            signals.append({
                'type': 'contrarian',
                'signal': 'SELL_CALLS_BUY_PUTS',
                'reason': f'Extremely low PCR(OI): {pcr_oi:.2f} indicates overbought conditions',
                'confidence': 0.6
            })
        
        elif pcr_oi > 1.5:
            signals.append({
                'type': 'contrarian',
                'signal': 'BUY_CALLS_SELL_PUTS',
                'reason': f'Extremely high PCR(OI): {pcr_oi:.2f} indicates oversold conditions',
                'confidence': 0.6
            })
        
        # VIX-based signals
        vix_analysis = components.get('vix_analysis', {})
        vix = vix_analysis.get('vix', 0)
        
        if vix < 12:
            signals.append({
                'type': 'volatility',
                'signal': 'BUY_VOLATILITY',
                'reason': f'Low VIX: {vix:.1f} suggests complacency, volatility likely to increase',
                'confidence': 0.65
            })
        
        elif vix > 25:
            signals.append({
                'type': 'volatility',
                'signal': 'SELL_VOLATILITY',
                'reason': f'High VIX: {vix:.1f} suggests fear, volatility likely to decrease',
                'confidence': 0.65
            })
        
        return signals[:3]  # Return top 3 signals
    
    def _calculate_confidence(self, components: Dict) -> float:
        """Calculate overall confidence in sentiment analysis"""
        confidence_factors = []
        
        # Component confidence
        for component_name, component_data in components.items():
            if 'confidence' in component_data:
                confidence_factors.append(component_data['confidence'])
        
        # Data completeness
        component_count = len(components)
        if component_count >= 5:
            completeness = 0.9
        elif component_count >= 3:
            completeness = 0.7
        elif component_count >= 1:
            completeness = 0.5
        else:
            completeness = 0.3
        
        confidence_factors.append(completeness)
        
        # Consistency
        overall_sentiment = self._calculate_overall_sentiment_sync(components)
        consistency = overall_sentiment.get('consistency', 0.5)
        confidence_factors.append(consistency)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_overall_sentiment_sync(self, components: Dict) -> Dict:
        """Synchronous version for confidence calculation"""
        # Simplified version without async
        scores = []
        
        for component_name, component_data in components.items():
            if 'score' in component_data:
                scores.append(component_data['score'])
            elif 'weighted_score' in component_data:
                scores.append(component_data['weighted_score'])
        
        if not scores:
            return {'consistency': 0.5}
        
        # Calculate consistency
        if len(scores) < 2:
            consistency = 1.0
        else:
            positive_count = sum(1 for s in scores if s > 0.1)
            negative_count = sum(1 for s in scores if s < -0.1)
            max_count = max(positive_count, negative_count, len(scores) - positive_count - negative_count)
            consistency = max_count / len(scores)
        
        return {'consistency': consistency}
    
    async def _analyze_sentiment_trends(self) -> Dict:
        """Analyze sentiment trends over time"""
        try:
            if len(self.historical_data['timestamp_history']) < 2:
                return {'trend': 'insufficient_data', 'duration': 0}
            
            # Calculate sentiment trend from recent data
            recent_pcrs = [point['pcr_oi'] for point in self.historical_data['pcr_history'][-10:]]
            recent_vix = [point['vix'] for point in self.historical_data['vix_history'][-10:]]
            
            if len(recent_pcrs) < 2 or len(recent_vix) < 2:
                return {'trend': 'insufficient_data', 'duration': 0}
            
            # Calculate trends
            pcr_trend = self._calculate_series_trend(recent_pcrs)
            vix_trend = self._calculate_series_trend(recent_vix)
            
            # Determine overall trend
            if pcr_trend == 'falling' and vix_trend == 'falling':
                overall_trend = 'improving'  # PCR falling (bullish), VIX falling (bullish)
            elif pcr_trend == 'rising' and vix_trend == 'rising':
                overall_trend = 'deteriorating'  # PCR rising (bearish), VIX rising (bearish)
            else:
                overall_trend = 'mixed'
            
            return {
                'trend': overall_trend,
                'pcr_trend': pcr_trend,
                'vix_trend': vix_trend,
                'duration_days': min(10, len(self.historical_data['timestamp_history'])),
                'data_points': len(recent_pcrs)
            }
            
        except Exception as e:
            logger.error(f"Sentiment trend analysis failed: {e}")
            return {'trend': 'analysis_failed', 'duration': 0}
    
    def _calculate_series_trend(self, series: List[float]) -> str:
        """Calculate trend of a series"""
        if len(series) < 2:
            return 'stable'
        
        # Simple slope calculation
        x = list(range(len(series)))
        y = series
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 'stable'
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Determine trend based on slope
        if slope > 0.1:
            return 'rising'
        elif slope < -0.1:
            return 'falling'
        else:
            return 'stable'
    
    async def _check_extreme_indicators(self, components: Dict) -> Dict:
        """Check for extreme sentiment indicators"""
        extremes = {
            'pcr_extreme': False,
            'vix_extreme': False,
            'breadth_extreme': False,
            'iv_extreme': False,
            'total_extremes': 0
        }
        
        # Check PCR extremes
        pcr_analysis = components.get('pcr_analysis', {})
        pcr_oi = pcr_analysis.get('pcr_oi', 1.0)
        if pcr_oi < 0.7 or pcr_oi > 1.5:
            extremes['pcr_extreme'] = True
            extremes['total_extremes'] += 1
        
        # Check VIX extremes
        vix_analysis = components.get('vix_analysis', {})
        vix = vix_analysis.get('vix', 0)
        if vix < 12 or vix > 25:
            extremes['vix_extreme'] = True
            extremes['total_extremes'] += 1
        
        # Check breadth extremes
        breadth_analysis = components.get('market_breadth', {})
        advance_percent = breadth_analysis.get('advance_percent', 50)
        if advance_percent > 70 or advance_percent < 30:
            extremes['breadth_extreme'] = True
            extremes['total_extremes'] += 1
        
        # Check IV extremes
        iv_analysis = components.get('iv_percentile', {})
        iv_percentile = iv_analysis.get('iv_percentile', 50)
        if iv_percentile > 80 or iv_percentile < 20:
            extremes['iv_extreme'] = True
            extremes['total_extremes'] += 1
        
        # Determine overall extreme level
        if extremes['total_extremes'] >= 3:
            extremes['overall_extreme'] = 'high'
        elif extremes['total_extremes'] >= 2:
            extremes['overall_extreme'] = 'medium'
        elif extremes['total_extremes'] >= 1:
            extremes['overall_extreme'] = 'low'
        else:
            extremes['overall_extreme'] = 'none'
        
        return extremes
    
    # Default analysis methods for error handling
    
    def _get_default_sentiment(self) -> Dict:
        """Get default sentiment analysis when data is unavailable"""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_sentiment': {
                'overall_score': 0.0,
                'sentiment': 'neutral',
                'strength': 'weak',
                'consistency': 0.5,
                'interpretation': 'Insufficient data for sentiment analysis',
                'component_count': 0
            },
            'components': {},
            'signals': [],
            'confidence': 0.3,
            'trend_analysis': {'trend': 'insufficient_data'},
            'extreme_indicators': {'total_extremes': 0, 'overall_extreme': 'none'},
            'metadata': {'data_sources_used': [], 'weights_applied': self.weights}
        }
    
    def _get_default_pcr_analysis(self) -> Dict:
        return {
            'pcr_oi': 1.0,
            'pcr_volume': 1.0,
            'market_pcr': 0,
            'pcr_oi_score': 0.0,
            'pcr_volume_score': 0.0,
            'market_pcr_score': 0.0,
            'weighted_score': 0.0,
            'sentiment': 'neutral',
            'strength': 'weak',
            'trend': 'stable',
            'interpretation': 'PCR data unavailable',
            'confidence': 0.3
        }
    
    def _get_default_breadth_analysis(self) -> Dict:
        return {
            'advances': 0,
            'declines': 0,
            'unchanged': 0,
            'total_securities': 0,
            'advance_percent': 0,
            'decline_percent': 0,
            'advance_decline_ratio': 0,
            'breadth_score': 0.0,
            'sentiment': 'neutral',
            'strength': 'weak',
            'trin_index': 0,
            'arms_index': 0,
            'interpretation': 'Market breadth data unavailable',
            'confidence': 0.3
        }
    
    def _get_default_vix_analysis(self) -> Dict:
        return {
            'vix': 0,
            'vix_change': 0,
            'vix_score': 0.0,
            'sentiment': 'neutral',
            'strength': 'weak',
            'trend': 'stable',
            'vix_percentile': 50,
            'interpretation': 'VIX data unavailable',
            'confidence': 0.3
        }
    
    def _get_default_iv_analysis(self) -> Dict:
        return {
            'iv_percentile': 50,
            'iv_score': 0.0,
            'sentiment': 'neutral',
            'strength': 'weak',
            'interpretation': 'IV percentile data unavailable',
            'confidence': 0.5
        }
    
    def _get_default_fii_dii_analysis(self) -> Dict:
        return {
            'fii_net': 0,
            'dii_net': 0,
            'total_net': 0,
            'fii_dii_score': 0.0,
            'sentiment': 'neutral',
            'strength': 'weak',
            'interpretation': 'FII/DII data unavailable',
            'confidence': 0.3
        }
    
    def _get_default_price_action_analysis(self) -> Dict:
        return {
            'nifty_price': 0,
            'nifty_change': 0,
            'nifty_percent_change': 0,
            'price_score': 0.0,
            'sentiment': 'neutral',
            'strength': 'weak',
            'trend': 'sideways',
            'interpretation': 'Price action data unavailable',
            'confidence': 0.3
        }
    
    def _get_default_overall_sentiment(self) -> Dict:
        return {
            'overall_score': 0.0,
            'sentiment': 'neutral',
            'strength': 'weak',
            'consistency': 0.5,
            'interpretation': 'Unable to calculate overall sentiment',
            'component_count': 0,
            'weighted_component_count': 0
        }