"""
Option Chain Analyzer - Comprehensive analysis of option chain data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import statistics

from src.utils.logger import get_logger
from src.utils.math_utils import calculate_percentile, calculate_z_score

logger = get_logger(__name__)

class OptionChainAnalyzer:
    """Analyzes option chain data for trading insights"""
    
    def __init__(self, config):
        self.config = config
        self.analysis_config = config['analysis']['options']
        
    async def analyze(self, option_data: Dict) -> Dict:
        """
        Perform comprehensive option chain analysis
        
        Args:
            option_data: Normalized option chain data
            
        Returns:
            Comprehensive analysis results
        """
        try:
            if not option_data or 'data' not in option_data:
                return {}
            
            logger.info("Starting option chain analysis...")
            
            # Extract basic data
            spot_price = option_data.get('spot_price', 0)
            options = option_data.get('data', [])
            
            if not options or spot_price <= 0:
                return {}
            
            # Perform all analyses
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'spot_price': spot_price,
                'total_options': len(options),
                'oi_analysis': await self._analyze_oi(options, spot_price),
                'volume_analysis': await self._analyze_volume(options, spot_price),
                'iv_analysis': await self._analyze_iv(options, spot_price),
                'price_analysis': await self._analyze_prices(options, spot_price),
                'sentiment_analysis': await self._analyze_sentiment(options, spot_price),
                'risk_analysis': await self._analyze_risk(options, spot_price),
                'levels_analysis': await self._analyze_key_levels(options, spot_price),
                'max_pain': await self._calculate_max_pain(options, spot_price),
                'summary': {}
            }
            
            # Generate summary
            analysis['summary'] = await self._generate_summary(analysis)
            
            logger.info(f"Option chain analysis completed: {len(options)} options analyzed")
            return analysis
            
        except Exception as e:
            logger.error(f"Option chain analysis failed: {e}")
            return {}
    
    async def _analyze_oi(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze Open Interest patterns"""
        try:
            # Separate calls and puts
            calls = []
            puts = []
            
            for option in options:
                strike = option.get('strike', 0)
                
                if 'CE' in option:
                    ce_data = option['CE']
                    calls.append({
                        'strike': strike,
                        'oi': ce_data.get('oi', 0),
                        'change_oi': ce_data.get('change_in_oi', 0),
                        'iv': ce_data.get('iv', 0)
                    })
                
                if 'PE' in option:
                    pe_data = option['PE']
                    puts.append({
                        'strike': strike,
                        'oi': pe_data.get('oi', 0),
                        'change_oi': pe_data.get('change_in_oi', 0),
                        'iv': pe_data.get('iv', 0)
                    })
            
            # Calculate totals
            total_call_oi = sum(c['oi'] for c in calls)
            total_put_oi = sum(p['oi'] for p in puts)
            total_call_change = sum(c['change_oi'] for c in calls)
            total_put_change = sum(p['change_oi'] for p in puts)
            
            # Find top OI strikes
            top_call_oi = sorted(calls, key=lambda x: x['oi'], reverse=True)[:10]
            top_put_oi = sorted(puts, key=lambda x: x['oi'], reverse=True)[:10]
            
            # Find largest OI changes
            top_call_change = sorted(calls, key=lambda x: x['change_oi'], reverse=True)[:10]
            top_put_change = sorted(puts, key=lambda x: x['change_oi'], reverse=True)[:10]
            
            # Analyze OI concentration
            oi_concentration = self._analyze_oi_concentration(calls, puts, spot_price)
            
            # Calculate OI ratios
            pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            pcr_change = total_put_change / total_call_change if total_call_change > 0 else 0
            
            return {
                'totals': {
                    'calls': total_call_oi,
                    'puts': total_put_oi,
                    'total': total_call_oi + total_put_oi,
                    'call_change': total_call_change,
                    'put_change': total_put_change,
                    'net_change': total_call_change + total_put_change
                },
                'ratios': {
                    'pcr_oi': pcr_oi,
                    'pcr_change': pcr_change,
                    'call_put_ratio': 1 / pcr_oi if pcr_oi > 0 else 0
                },
                'top_strikes': {
                    'calls_by_oi': top_call_oi,
                    'puts_by_oi': top_put_oi,
                    'calls_by_change': top_call_change,
                    'puts_by_change': top_put_change
                },
                'concentration': oi_concentration,
                'buildup_analysis': self._analyze_oi_buildup(calls, puts, spot_price)
            }
            
        except Exception as e:
            logger.error(f"OI analysis failed: {e}")
            return {}
    
    async def _analyze_volume(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze trading volume patterns"""
        try:
            total_call_volume = 0
            total_put_volume = 0
            volume_by_strike = {}
            
            for option in options:
                strike = option.get('strike', 0)
                
                if 'CE' in option:
                    volume = option['CE'].get('volume', 0)
                    total_call_volume += volume
                    
                    if strike not in volume_by_strike:
                        volume_by_strike[strike] = {'call': 0, 'put': 0}
                    volume_by_strike[strike]['call'] += volume
                
                if 'PE' in option:
                    volume = option['PE'].get('volume', 0)
                    total_put_volume += volume
                    
                    if strike not in volume_by_strike:
                        volume_by_strike[strike] = {'call': 0, 'put': 0}
                    volume_by_strike[strike]['put'] += volume
            
            # Calculate volume ratios
            total_volume = total_call_volume + total_put_volume
            pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0
            
            # Find high volume strikes
            high_volume_strikes = []
            for strike, volumes in volume_by_strike.items():
                total_strike_volume = volumes['call'] + volumes['put']
                if total_strike_volume > (total_volume * 0.01):  # More than 1% of total
                    high_volume_strikes.append({
                        'strike': strike,
                        'call_volume': volumes['call'],
                        'put_volume': volumes['put'],
                        'total_volume': total_strike_volume,
                        'percent_of_total': (total_strike_volume / total_volume) * 100
                    })
            
            # Sort by total volume
            high_volume_strikes.sort(key=lambda x: x['total_volume'], reverse=True)
            
            # Analyze volume spikes
            volume_spikes = self._analyze_volume_spikes(options)
            
            return {
                'totals': {
                    'calls': total_call_volume,
                    'puts': total_put_volume,
                    'total': total_volume
                },
                'ratios': {
                    'pcr_volume': pcr_volume,
                    'call_put_volume_ratio': 1 / pcr_volume if pcr_volume > 0 else 0,
                    'call_percentage': (total_call_volume / total_volume) * 100 if total_volume > 0 else 0,
                    'put_percentage': (total_put_volume / total_volume) * 100 if total_volume > 0 else 0
                },
                'high_volume_strikes': high_volume_strikes[:10],
                'volume_spikes': volume_spikes,
                'volume_profile': self._analyze_volume_profile(options, spot_price)
            }
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return {}
    
    async def _analyze_iv(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze Implied Volatility patterns"""
        try:
            call_ivs = []
            put_ivs = []
            iv_by_strike = {}
            
            for option in options:
                strike = option.get('strike', 0)
                distance_percent = abs(strike - spot_price) / spot_price * 100
                
                if 'CE' in option:
                    iv = option['CE'].get('iv', 0)
                    if iv > 0:
                        call_ivs.append(iv)
                        if strike not in iv_by_strike:
                            iv_by_strike[strike] = {'call': 0, 'put': 0, 'distance': distance_percent}
                        iv_by_strike[strike]['call'] = iv
                
                if 'PE' in option:
                    iv = option['PE'].get('iv', 0)
                    if iv > 0:
                        put_ivs.append(iv)
                        if strike not in iv_by_strike:
                            iv_by_strike[strike] = {'call': 0, 'put': 0, 'distance': distance_percent}
                        iv_by_strike[strike]['put'] = iv
            
            # Calculate IV statistics
            call_iv_stats = self._calculate_iv_stats(call_ivs, 'call')
            put_iv_stats = self._calculate_iv_stats(put_ivs, 'put')
            
            # Analyze IV skew
            iv_skew = self._analyze_iv_skew(iv_by_strike, spot_price)
            
            # Analyze IV term structure (would need multiple expiries)
            iv_term_structure = self._analyze_iv_term_structure(options)
            
            # Calculate IV percentile and rank
            iv_percentile = self._calculate_iv_percentile(call_ivs + put_ivs)
            
            return {
                'call_iv': call_iv_stats,
                'put_iv': put_iv_stats,
                'skew_analysis': iv_skew,
                'term_structure': iv_term_structure,
                'percentile_analysis': iv_percentile,
                'iv_surface': self._analyze_iv_surface(iv_by_strike, spot_price)
            }
            
        except Exception as e:
            logger.error(f"IV analysis failed: {e}")
            return {}
    
    async def _analyze_prices(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze option prices"""
        try:
            call_prices = []
            put_prices = []
            price_ratios = []
            
            for option in options:
                strike = option.get('strike', 0)
                
                if 'CE' in option:
                    price = option['CE'].get('last_price', 0)
                    if price > 0:
                        call_prices.append({
                            'strike': strike,
                            'price': price,
                            'intrinsic': max(spot_price - strike, 0),
                            'extrinsic': max(price - max(spot_price - strike, 0), 0)
                        })
                
                if 'PE' in option:
                    price = option['PE'].get('last_price', 0)
                    if price > 0:
                        put_prices.append({
                            'strike': strike,
                            'price': price,
                            'intrinsic': max(strike - spot_price, 0),
                            'extrinsic': max(price - max(strike - spot_price, 0), 0)
                        })
            
            # Calculate price statistics
            call_price_stats = self._calculate_price_stats(call_prices, 'call')
            put_price_stats = self._calculate_price_stats(put_prices, 'put')
            
            # Analyze premium decay
            premium_decay = self._analyze_premium_decay(options, spot_price)
            
            # Analyze cost of carry
            cost_of_carry = self._analyze_cost_of_carry(options, spot_price)
            
            return {
                'call_prices': call_price_stats,
                'put_prices': put_price_stats,
                'premium_decay': premium_decay,
                'cost_of_carry': cost_of_carry,
                'arbitrage_opportunities': self._find_arbitrage_opportunities(options, spot_price)
            }
            
        except Exception as e:
            logger.error(f"Price analysis failed: {e}")
            return {}
    
    async def _analyze_sentiment(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze market sentiment from options"""
        try:
            # Calculate PCR from different perspectives
            pcr_oi = await self._calculate_pcr(options, 'oi')
            pcr_volume = await self._calculate_pcr(options, 'volume')
            pcr_change = await self._calculate_pcr(options, 'change_oi')
            
            # Analyze OI buildup direction
            oi_buildup = self._analyze_buildup_direction(options)
            
            # Analyze max pain vs spot
            max_pain = await self._calculate_max_pain(options, spot_price)
            max_pain_distance = abs(spot_price - max_pain.get('max_pain_strike', spot_price))
            max_pain_percent = (max_pain_distance / spot_price) * 100
            
            # Analyze skew sentiment
            skew_sentiment = self._analyze_skew_sentiment(options, spot_price)
            
            # Calculate overall sentiment score
            sentiment_score = self._calculate_sentiment_score(
                pcr_oi, pcr_volume, oi_buildup, max_pain_percent, skew_sentiment
            )
            
            return {
                'pcr_analysis': {
                    'oi': pcr_oi,
                    'volume': pcr_volume,
                    'change': pcr_change,
                    'interpretation': self._interpret_pcr(pcr_oi)
                },
                'oi_buildup': oi_buildup,
                'max_pain_analysis': {
                    'max_pain_strike': max_pain.get('max_pain_strike', 0),
                    'distance_from_spot': max_pain_distance,
                    'percent_from_spot': max_pain_percent,
                    'sentiment': 'bullish' if spot_price > max_pain.get('max_pain_strike', spot_price) else 'bearish'
                },
                'skew_sentiment': skew_sentiment,
                'overall_sentiment': {
                    'score': sentiment_score,
                    'level': self._get_sentiment_level(sentiment_score),
                    'confidence': self._calculate_sentiment_confidence(options)
                }
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {}
    
    async def _analyze_risk(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze risk metrics"""
        try:
            # Calculate gamma exposure
            gamma_exposure = self._calculate_gamma_exposure(options, spot_price)
            
            # Calculate vega exposure
            vega_exposure = self._calculate_vega_exposure(options)
            
            # Calculate theta decay
            theta_decay = self._calculate_theta_decay(options)
            
            # Analyze pin risk
            pin_risk = self._analyze_pin_risk(options, spot_price)
            
            # Calculate Value at Risk (simplified)
            var = self._calculate_var(options, spot_price)
            
            # Analyze concentration risk
            concentration_risk = self._analyze_concentration_risk(options)
            
            return {
                'greeks_exposure': {
                    'gamma': gamma_exposure,
                    'vega': vega_exposure,
                    'theta': theta_decay
                },
                'pin_risk': pin_risk,
                'value_at_risk': var,
                'concentration_risk': concentration_risk,
                'liquidity_risk': self._analyze_liquidity_risk(options),
                'volatility_risk': self._analyze_volatility_risk(options)
            }
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {}
    
    async def _analyze_key_levels(self, options: List[Dict], spot_price: float) -> Dict:
        """Identify key support and resistance levels"""
        try:
            # Support from Put OI
            puts = []
            for option in options:
                if 'PE' in option and option['strike'] < spot_price:
                    puts.append({
                        'strike': option['strike'],
                        'oi': option['PE'].get('oi', 0),
                        'change_oi': option['PE'].get('change_in_oi', 0)
                    })
            
            # Resistance from Call OI
            calls = []
            for option in options:
                if 'CE' in option and option['strike'] > spot_price:
                    calls.append({
                        'strike': option['strike'],
                        'oi': option['CE'].get('oi', 0),
                        'change_oi': option['CE'].get('change_in_oi', 0)
                    })
            
            # Sort by OI and find key levels
            puts.sort(key=lambda x: x['oi'], reverse=True)
            calls.sort(key=lambda x: x['oi'], reverse=True)
            
            # Identify support levels (highest put OI)
            supports = []
            for i, put in enumerate(puts[:5]):
                strength = put['oi'] / max(puts[0]['oi'], 1) if puts else 0
                supports.append({
                    'level': put['strike'],
                    'strength': strength,
                    'oi': put['oi'],
                    'change_oi': put['change_oi'],
                    'rank': i + 1
                })
            
            # Identify resistance levels (highest call OI)
            resistances = []
            for i, call in enumerate(calls[:5]):
                strength = call['oi'] / max(calls[0]['oi'], 1) if calls else 0
                resistances.append({
                    'level': call['strike'],
                    'strength': strength,
                    'oi': call['oi'],
                    'change_oi': call['change_oi'],
                    'rank': i + 1
                })
            
            # Identify psychological levels
            psychological_levels = self._identify_psychological_levels(spot_price)
            
            # Identify recent highs/lows (would need historical data)
            recent_levels = self._identify_recent_levels(options)
            
            return {
                'supports': supports,
                'resistances': resistances,
                'psychological_levels': psychological_levels,
                'recent_levels': recent_levels,
                'pivot_points': self._calculate_pivot_points(spot_price)
            }
            
        except Exception as e:
            logger.error(f"Key levels analysis failed: {e}")
            return {}
    
    async def _calculate_max_pain(self, options: List[Dict], spot_price: float) -> Dict:
        """Calculate Max Pain point"""
        try:
            strike_payoffs = {}
            
            for option in options:
                strike = option.get('strike', 0)
                
                if strike not in strike_payoffs:
                    strike_payoffs[strike] = 0
                
                # Calculate payoff for calls
                if 'CE' in option:
                    call_oi = option['CE'].get('oi', 0)
                    # For each call OI contract, payoff = max(0, strike - spot) at expiry
                    # But for max pain, we calculate total pain for option writers
                    strike_payoffs[strike] += call_oi * max(0, strike - spot_price)
                
                # Calculate payoff for puts
                if 'PE' in option:
                    put_oi = option['PE'].get('oi', 0)
                    strike_payoffs[strike] += put_oi * max(0, spot_price - strike)
            
            # Find strike with minimum total payoff (max pain)
            if strike_payoffs:
                min_pain_strike = min(strike_payoffs, key=strike_payoffs.get)
                min_pain_value = strike_payoffs[min_pain_strike]
            else:
                min_pain_strike = spot_price
                min_pain_value = 0
            
            return {
                'max_pain_strike': min_pain_strike,
                'max_pain_value': min_pain_value,
                'distance_from_spot': abs(spot_price - min_pain_strike),
                'percent_from_spot': (abs(spot_price - min_pain_strike) / spot_price) * 100,
                'all_payoffs': strike_payoffs
            }
            
        except Exception as e:
            logger.error(f"Max pain calculation failed: {e}")
            return {'max_pain_strike': spot_price, 'max_pain_value': 0}
    
    async def _generate_summary(self, analysis: Dict) -> Dict:
        """Generate summary of all analyses"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'market_status': 'open',  # Would check market hours
                'overall_sentiment': analysis.get('sentiment_analysis', {}).get('overall_sentiment', {}),
                'key_findings': [],
                'trading_signals': [],
                'risk_assessment': 'medium',
                'confidence_score': 0.0
            }
            
            # Extract key findings
            findings = self._extract_key_findings(analysis)
            summary['key_findings'] = findings
            
            # Generate trading signals
            signals = self._generate_trading_signals(analysis)
            summary['trading_signals'] = signals
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(analysis)
            summary['confidence_score'] = confidence
            
            # Assess overall risk
            risk = self._assess_overall_risk(analysis)
            summary['risk_assessment'] = risk
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {}
    
    # Helper methods for detailed analysis
    
    def _analyze_oi_concentration(self, calls: List[Dict], puts: List[Dict], spot_price: float) -> Dict:
        """Analyze OI concentration"""
        # Calculate concentration metrics
        total_call_oi = sum(c['oi'] for c in calls)
        total_put_oi = sum(p['oi'] for p in puts)
        
        # Find strikes with high OI concentration
        concentration_threshold = self.analysis_config['oi_concentration']['threshold_percent'] / 100
        
        high_concentration_calls = []
        high_concentration_puts = []
        
        for call in calls:
            if total_call_oi > 0 and call['oi'] / total_call_oi > concentration_threshold:
                high_concentration_calls.append({
                    'strike': call['strike'],
                    'oi': call['oi'],
                    'percent_of_total': (call['oi'] / total_call_oi) * 100
                })
        
        for put in puts:
            if total_put_oi > 0 and put['oi'] / total_put_oi > concentration_threshold:
                high_concentration_puts.append({
                    'strike': put['strike'],
                    'oi': put['oi'],
                    'percent_of_total': (put['oi'] / total_put_oi) * 100
                })
        
        return {
            'call_concentration': sorted(high_concentration_calls, key=lambda x: x['percent_of_total'], reverse=True),
            'put_concentration': sorted(high_concentration_puts, key=lambda x: x['percent_of_total'], reverse=True),
            'concentration_score': len(high_concentration_calls) + len(high_concentration_puts)
        }
    
    def _analyze_oi_buildup(self, calls: List[Dict], puts: List[Dict], spot_price: float) -> Dict:
        """Analyze OI buildup patterns"""
        # Separate by moneyness
        itm_calls = [c for c in calls if c['strike'] < spot_price * 0.98]
        atm_calls = [c for c in calls if spot_price * 0.98 <= c['strike'] <= spot_price * 1.02]
        otm_calls = [c for c in calls if c['strike'] > spot_price * 1.02]
        
        itm_puts = [p for p in puts if p['strike'] > spot_price * 1.02]
        atm_puts = [p for p in puts if spot_price * 0.98 <= p['strike'] <= spot_price * 1.02]
        otm_puts = [p for p in puts if p['strike'] < spot_price * 0.98]
        
        # Calculate buildup by zone
        buildup = {
            'calls': {
                'itm': sum(c['change_oi'] for c in itm_calls),
                'atm': sum(c['change_oi'] for c in atm_calls),
                'otm': sum(c['change_oi'] for c in otm_calls)
            },
            'puts': {
                'itm': sum(p['change_oi'] for p in itm_puts),
                'atm': sum(p['change_oi'] for p in atm_puts),
                'otm': sum(p['change_oi'] for p in otm_puts)
            }
        }
        
        # Interpret buildup
        interpretation = []
        if buildup['calls']['otm'] > 0 and abs(buildup['calls']['otm']) > abs(buildup['puts']['otm']):
            interpretation.append('Call writing at OTM strikes')
        if buildup['puts']['otm'] > 0 and abs(buildup['puts']['otm']) > abs(buildup['calls']['otm']):
            interpretation.append('Put writing at OTM strikes')
        
        return {
            'buildup_by_zone': buildup,
            'interpretation': interpretation,
            'net_buildup': sum(buildup['calls'].values()) + sum(buildup['puts'].values())
        }
    
    def _analyze_volume_spikes(self, options: List[Dict]) -> List[Dict]:
        """Identify volume spikes"""
        spikes = []
        
        for option in options:
            strike = option.get('strike', 0)
            
            # Check for unusual volume (simplified)
            if 'CE' in option:
                volume = option['CE'].get('volume', 0)
                avg_volume = option['CE'].get('avg_volume', 0)  # Would need historical
                if avg_volume > 0 and volume > avg_volume * 2:
                    spikes.append({
                        'strike': strike,
                        'type': 'CE',
                        'volume': volume,
                        'avg_volume': avg_volume,
                        'spike_ratio': volume / avg_volume
                    })
            
            if 'PE' in option:
                volume = option['PE'].get('volume', 0)
                avg_volume = option['PE'].get('avg_volume', 0)
                if avg_volume > 0 and volume > avg_volume * 2:
                    spikes.append({
                        'strike': strike,
                        'type': 'PE',
                        'volume': volume,
                        'avg_volume': avg_volume,
                        'spike_ratio': volume / avg_volume
                    })
        
        return sorted(spikes, key=lambda x: x['spike_ratio'], reverse=True)[:10]
    
    def _analyze_volume_profile(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze volume profile by strike"""
        volume_profile = []
        
        for option in options:
            strike = option.get('strike', 0)
            call_volume = option.get('CE', {}).get('volume', 0)
            put_volume = option.get('PE', {}).get('volume', 0)
            total_volume = call_volume + put_volume
            
            if total_volume > 0:
                volume_profile.append({
                    'strike': strike,
                    'call_volume': call_volume,
                    'put_volume': put_volume,
                    'total_volume': total_volume,
                    'distance_from_spot': abs(strike - spot_price),
                    'percent_from_spot': (abs(strike - spot_price) / spot_price) * 100
                })
        
        # Sort by total volume
        volume_profile.sort(key=lambda x: x['total_volume'], reverse=True)
        
        return {
            'profile': volume_profile[:20],
            'high_volume_nodes': [v for v in volume_profile if v['total_volume'] > statistics.mean([vp['total_volume'] for vp in volume_profile]) * 2],
            'volume_value_area': self._calculate_volume_value_area(volume_profile)
        }
    
    def _calculate_iv_stats(self, ivs: List[float], option_type: str) -> Dict:
        """Calculate IV statistics"""
        if not ivs:
            return {}
        
        return {
            'type': option_type,
            'mean': statistics.mean(ivs),
            'median': statistics.median(ivs),
            'std': statistics.stdev(ivs) if len(ivs) > 1 else 0,
            'min': min(ivs),
            'max': max(ivs),
            'range': max(ivs) - min(ivs),
            'count': len(ivs)
        }
    
    def _analyze_iv_skew(self, iv_by_strike: Dict, spot_price: float) -> Dict:
        """Analyze IV skew across strikes"""
        if not iv_by_strike:
            return {}
        
        # Calculate skew for different moneyness zones
        itm_call_ivs = []
        atm_call_ivs = []
        otm_call_ivs = []
        itm_put_ivs = []
        atm_put_ivs = []
        otm_put_ivs = []
        
        for strike, data in iv_by_strike.items():
            distance = data.get('distance', 0)
            
            if data['call'] > 0:
                if strike < spot_price * 0.98:
                    itm_call_ivs.append(data['call'])
                elif strike > spot_price * 1.02:
                    otm_call_ivs.append(data['call'])
                else:
                    atm_call_ivs.append(data['call'])
            
            if data['put'] > 0:
                if strike > spot_price * 1.02:
                    itm_put_ivs.append(data['put'])
                elif strike < spot_price * 0.98:
                    otm_put_ivs.append(data['put'])
                else:
                    atm_put_ivs.append(data['put'])
        
        # Calculate average IVs
        avg_itm_call = statistics.mean(itm_call_ivs) if itm_call_ivs else 0
        avg_atm_call = statistics.mean(atm_call_ivs) if atm_call_ivs else 0
        avg_otm_call = statistics.mean(otm_call_ivs) if otm_call_ivs else 0
        avg_itm_put = statistics.mean(itm_put_ivs) if itm_put_ivs else 0
        avg_atm_put = statistics.mean(atm_put_ivs) if atm_put_ivs else 0
        avg_otm_put = statistics.mean(otm_put_ivs) if otm_put_ivs else 0
        
        # Calculate skew ratios
        call_skew = avg_otm_call / avg_itm_call if avg_itm_call > 0 else 0
        put_skew = avg_otm_put / avg_itm_put if avg_itm_put > 0 else 0
        put_call_skew = avg_atm_put / avg_atm_call if avg_atm_call > 0 else 0
        
        return {
            'call_skew': call_skew,
            'put_skew': put_skew,
            'put_call_skew': put_call_skew,
            'skew_type': self._determine_skew_type(call_skew, put_skew, put_call_skew),
            'iv_by_moneyness': {
                'calls': {'itm': avg_itm_call, 'atm': avg_atm_call, 'otm': avg_otm_call},
                'puts': {'itm': avg_itm_put, 'atm': avg_atm_put, 'otm': avg_otm_put}
            }
        }
    
    def _analyze_iv_term_structure(self, options: List[Dict]) -> Dict:
        """Analyze IV term structure (would need multiple expiries)"""
        # Placeholder - needs multiple expiry data
        return {
            'backwardation': False,
            'contango': True,
            'term_structure': 'normal',
            'near_term_iv': 0,
            'far_term_iv': 0
        }
    
    def _calculate_iv_percentile(self, ivs: List[float]) -> Dict:
        """Calculate IV percentile and rank"""
        if not ivs:
            return {}
        
        current_iv = statistics.mean(ivs) if ivs else 0
        
        # Would need historical IV data for accurate percentile
        # For now, use simplified calculation
        iv_percentile = calculate_percentile(ivs, current_iv) if ivs else 50
        
        return {
            'current_iv': current_iv,
            'percentile': iv_percentile,
            'rank': 'high' if iv_percentile > 70 else 'low' if iv_percentile < 30 else 'medium',
            'interpretation': self._interpret_iv_percentile(iv_percentile)
        }
    
    def _analyze_iv_surface(self, iv_by_strike: Dict, spot_price: float) -> Dict:
        """Analyze IV surface"""
        # Calculate IV surface metrics
        strikes = sorted(iv_by_strike.keys())
        
        if len(strikes) < 3:
            return {}
        
        # Fit polynomial to IV surface
        try:
            x = [(strike - spot_price) / spot_price for strike in strikes]
            y_call = [iv_by_strike[strike]['call'] for strike in strikes if iv_by_strike[strike]['call'] > 0]
            y_put = [iv_by_strike[strike]['put'] for strike in strikes if iv_by_strike[strike]['put'] > 0]
            
            if len(y_call) > 2 and len(y_put) > 2:
                # Simple curvature calculation
                call_curvature = np.polyfit(range(len(y_call)), y_call, 2)[0]
                put_curvature = np.polyfit(range(len(y_put)), y_put, 2)[0]
            else:
                call_curvature = 0
                put_curvature = 0
        except:
            call_curvature = 0
            put_curvature = 0
        
        return {
            'call_curvature': call_curvature,
            'put_curvature': put_curvature,
            'surface_type': 'smile' if call_curvature > 0 else 'skew' if call_curvature < 0 else 'flat',
            'complexity': 'high' if abs(call_curvature) > 0.1 else 'medium' if abs(call_curvature) > 0.05 else 'low'
        }
    
    def _calculate_price_stats(self, prices: List[Dict], option_type: str) -> Dict:
        """Calculate price statistics"""
        if not prices:
            return {}
        
        price_values = [p['price'] for p in prices]
        intrinsic_values = [p['intrinsic'] for p in prices]
        extrinsic_values = [p['extrinsic'] for p in prices]
        
        return {
            'type': option_type,
            'price_stats': {
                'mean': statistics.mean(price_values),
                'median': statistics.median(price_values),
                'min': min(price_values),
                'max': max(price_values)
            },
            'intrinsic_stats': {
                'mean': statistics.mean(intrinsic_values),
                'total': sum(intrinsic_values)
            },
            'extrinsic_stats': {
                'mean': statistics.mean(extrinsic_values),
                'total': sum(extrinsic_values),
                'percent_of_price': (statistics.mean(extrinsic_values) / statistics.mean(price_values)) * 100 if statistics.mean(price_values) > 0 else 0
            }
        }
    
    def _analyze_premium_decay(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze premium decay patterns"""
        # Would need time series data for accurate analysis
        # For now, calculate average extrinsic value
        extrinsic_values = []
        
        for option in options:
            strike = option.get('strike', 0)
            
            if 'CE' in option:
                price = option['CE'].get('last_price', 0)
                intrinsic = max(spot_price - strike, 0)
                extrinsic = max(price - intrinsic, 0)
                extrinsic_values.append(extrinsic)
            
            if 'PE' in option:
                price = option['PE'].get('last_price', 0)
                intrinsic = max(strike - spot_price, 0)
                extrinsic = max(price - intrinsic, 0)
                extrinsic_values.append(extrinsic)
        
        avg_extrinsic = statistics.mean(extrinsic_values) if extrinsic_values else 0
        
        return {
            'avg_extrinsic': avg_extrinsic,
            'decay_rate': 'high' if avg_extrinsic > spot_price * 0.02 else 'medium' if avg_extrinsic > spot_price * 0.01 else 'low',
            'time_value_risk': 'high' if avg_extrinsic > spot_price * 0.03 else 'medium'
        }
    
    def _analyze_cost_of_carry(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze cost of carry (put-call parity)"""
        # Check put-call parity violations
        violations = []
        
        for option in options:
            strike = option.get('strike', 0)
            
            if 'CE' in option and 'PE' in option:
                call_price = option['CE'].get('last_price', 0)
                put_price = option['PE'].get('last_price', 0)
                
                # Put-call parity: C - P = S - K * e^(-rT)
                # Simplified: C - P ≈ S - K
                parity_diff = call_price - put_price
                fair_diff = spot_price - strike
                
                if abs(parity_diff - fair_diff) > spot_price * 0.01:  # 1% threshold
                    violations.append({
                        'strike': strike,
                        'call_price': call_price,
                        'put_price': put_price,
                        'parity_diff': parity_diff,
                        'fair_diff': fair_diff,
                        'arbitrage_opportunity': True
                    })
        
        return {
            'violations': violations,
            'arbitrage_opportunities': len(violations),
            'market_efficiency': 'high' if len(violations) == 0 else 'medium' if len(violations) < 3 else 'low'
        }
    
    def _find_arbitrage_opportunities(self, options: List[Dict], spot_price: float) -> List[Dict]:
        """Find arbitrage opportunities"""
        opportunities = []
        
        # Check for box spreads, conversions, reversals
        # Simplified check for put-call parity violations
        for option in options:
            strike = option.get('strike', 0)
            
            if 'CE' in option and 'PE' in option:
                call_price = option['CE'].get('last_price', 0)
                put_price = option['PE'].get('last_price', 0)
                
                # Synthetic long: long call + short put = long stock
                synthetic_cost = call_price - put_price
                
                # If synthetic cost differs significantly from spot-strike
                if abs(synthetic_cost - (spot_price - strike)) > max(call_price, put_price) * 0.1:
                    opportunities.append({
                        'type': 'conversion_reversal',
                        'strike': strike,
                        'call_price': call_price,
                        'put_price': put_price,
                        'synthetic_cost': synthetic_cost,
                        'fair_cost': spot_price - strike,
                        'arbitrage_potential': abs(synthetic_cost - (spot_price - strike))
                    })
        
        return opportunities
    
    async def _calculate_pcr(self, options: List[Dict], metric: str) -> float:
        """Calculate Put-Call Ratio for given metric"""
        total_calls = 0
        total_puts = 0
        
        for option in options:
            if 'CE' in option:
                total_calls += option['CE'].get(metric, 0)
            if 'PE' in option:
                total_puts += option['PE'].get(metric, 0)
        
        return total_puts / total_calls if total_calls > 0 else 0
    
    def _analyze_buildup_direction(self, options: List[Dict]) -> Dict:
        """Analyze OI buildup direction"""
        call_buildup = 0
        put_buildup = 0
        
        for option in options:
            if 'CE' in option:
                call_buildup += option['CE'].get('change_in_oi', 0)
            if 'PE' in option:
                put_buildup += option['PE'].get('change_in_oi', 0)
        
        total_buildup = call_buildup + put_buildup
        
        return {
            'call_buildup': call_buildup,
            'put_buildup': put_buildup,
            'net_buildup': total_buildup,
            'direction': 'call' if call_buildup > put_buildup else 'put',
            'strength': abs(call_buildup - put_buildup) / max(abs(call_buildup), abs(put_buildup), 1) * 100
        }
    
    def _analyze_skew_sentiment(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze skew for sentiment"""
        # Calculate IV skew for calls and puts
        call_skew = 0
        put_skew = 0
        
        otm_call_ivs = []
        itm_call_ivs = []
        otm_put_ivs = []
        itm_put_ivs = []
        
        for option in options:
            strike = option.get('strike', 0)
            
            if 'CE' in option:
                iv = option['CE'].get('iv', 0)
                if iv > 0:
                    if strike > spot_price * 1.02:
                        otm_call_ivs.append(iv)
                    elif strike < spot_price * 0.98:
                        itm_call_ivs.append(iv)
            
            if 'PE' in option:
                iv = option['PE'].get('iv', 0)
                if iv > 0:
                    if strike < spot_price * 0.98:
                        otm_put_ivs.append(iv)
                    elif strike > spot_price * 1.02:
                        itm_put_ivs.append(iv)
        
        # Calculate average IVs
        avg_otm_call = statistics.mean(otm_call_ivs) if otm_call_ivs else 0
        avg_itm_call = statistics.mean(itm_call_ivs) if itm_call_ivs else 0
        avg_otm_put = statistics.mean(otm_put_ivs) if otm_put_ivs else 0
        avg_itm_put = statistics.mean(itm_put_ivs) if itm_put_ivs else 0
        
        # Calculate skew
        call_skew = avg_otm_call / avg_itm_call if avg_itm_call > 0 else 1
        put_skew = avg_otm_put / avg_itm_put if avg_itm_put > 0 else 1
        
        return {
            'call_skew': call_skew,
            'put_skew': put_skew,
            'sentiment': 'fear' if put_skew > 1.2 else 'greed' if call_skew > 1.2 else 'neutral',
            'skew_strength': 'strong' if max(call_skew, put_skew) > 1.5 else 'moderate' if max(call_skew, put_skew) > 1.2 else 'weak'
        }
    
    def _calculate_sentiment_score(self, pcr_oi: float, pcr_volume: float, 
                                 oi_buildup: Dict, max_pain_percent: float, 
                                 skew_sentiment: Dict) -> float:
        """Calculate overall sentiment score (-1 to +1)"""
        score = 0.0
        
        # PCR contribution (40%)
        if pcr_oi < 0.7:
            score += 0.4  # Very bullish
        elif pcr_oi < 1.0:
            score += 0.2  # Mildly bullish
        elif pcr_oi > 1.5:
            score -= 0.4  # Very bearish
        elif pcr_oi > 1.2:
            score -= 0.2  # Mildly bearish
        
        # OI buildup contribution (30%)
        buildup_direction = oi_buildup.get('direction', '')
        if buildup_direction == 'call':
            score += 0.3 * (oi_buildup.get('strength', 0) / 100)
        elif buildup_direction == 'put':
            score -= 0.3 * (oi_buildup.get('strength', 0) / 100)
        
        # Max pain contribution (20%)
        # If spot is above max pain, bullish; below max pain, bearish
        # This would need actual max pain vs spot comparison
        
        # Skew sentiment contribution (10%)
        skew = skew_sentiment.get('sentiment', 'neutral')
        if skew == 'fear':
            score -= 0.1
        elif skew == 'greed':
            score += 0.1
        
        return max(-1, min(1, score))
    
    def _get_sentiment_level(self, score: float) -> str:
        """Convert sentiment score to level"""
        if score > 0.7:
            return 'very_bullish'
        elif score > 0.3:
            return 'bullish'
        elif score > -0.3:
            return 'neutral'
        elif score > -0.7:
            return 'bearish'
        else:
            return 'very_bearish'
    
    def _calculate_sentiment_confidence(self, options: List[Dict]) -> float:
        """Calculate confidence in sentiment analysis"""
        # Based on data quality and consistency
        total_options = len(options)
        if total_options == 0:
            return 0.0
        
        # Check data completeness
        complete_data = 0
        for option in options:
            if 'CE' in option and 'PE' in option:
                ce_data = option['CE']
                pe_data = option['PE']
                if all(k in ce_data for k in ['oi', 'volume', 'iv']) and \
                   all(k in pe_data for k in ['oi', 'volume', 'iv']):
                    complete_data += 1
        
        completeness = complete_data / total_options
        
        # Check consistency (would need time series)
        consistency = 0.8  # Placeholder
        
        return (completeness + consistency) / 2
    
    def _calculate_gamma_exposure(self, options: List[Dict], spot_price: float) -> Dict:
        """Calculate gamma exposure"""
        # Simplified gamma calculation
        gamma_exposure = 0
        gamma_by_strike = {}
        
        for option in options:
            strike = option.get('strike', 0)
            
            # Gamma is highest for ATM options
            distance = abs(strike - spot_price) / spot_price
            
            if distance < 0.02:  # ATM
                gamma_weight = 0.8
            elif distance < 0.05:  # Near ATM
                gamma_weight = 0.5
            else:
                gamma_weight = 0.2
            
            total_oi = 0
            if 'CE' in option:
                total_oi += option['CE'].get('oi', 0)
            if 'PE' in option:
                total_oi += option['PE'].get('oi', 0)
            
            gamma_contribution = total_oi * gamma_weight
            gamma_exposure += gamma_contribution
            
            if gamma_contribution > 0:
                gamma_by_strike[strike] = gamma_contribution
        
        return {
            'total_gamma': gamma_exposure,
            'gamma_by_strike': gamma_by_strike,
            'gamma_risk': 'high' if gamma_exposure > 1000000 else 'medium' if gamma_exposure > 500000 else 'low'
        }
    
    def _calculate_vega_exposure(self, options: List[Dict]) -> Dict:
        """Calculate vega exposure"""
        vega_exposure = 0
        
        for option in options:
            total_oi = 0
            avg_iv = 0
            iv_count = 0
            
            if 'CE' in option:
                total_oi += option['CE'].get('oi', 0)
                iv = option['CE'].get('iv', 0)
                if iv > 0:
                    avg_iv += iv
                    iv_count += 1
            
            if 'PE' in option:
                total_oi += option['PE'].get('oi', 0)
                iv = option['PE'].get('iv', 0)
                if iv > 0:
                    avg_iv += iv
                    iv_count += 1
            
            if iv_count > 0:
                avg_iv = avg_iv / iv_count
                # Simplified vega: OI * IV
                vega_exposure += total_oi * avg_iv
        
        return {
            'total_vega': vega_exposure,
            'vega_risk': 'high' if vega_exposure > 5000000 else 'medium' if vega_exposure > 1000000 else 'low'
        }
    
    def _calculate_theta_decay(self, options: List[Dict]) -> Dict:
        """Calculate theta decay"""
        total_theta = 0
        
        for option in options:
            # Simplified theta: higher for ATM, higher IV
            strike = option.get('strike', 0)
            
            # Would need actual theta calculation from Greeks
            # For now, use placeholder
            theta_contribution = 0
            
            if 'CE' in option:
                oi = option['CE'].get('oi', 0)
                iv = option['CE'].get('iv', 0)
                theta_contribution += oi * iv * 0.01  # Placeholder
            
            if 'PE' in option:
                oi = option['PE'].get('oi', 0)
                iv = option['PE'].get('iv', 0)
                theta_contribution += oi * iv * 0.01  # Placeholder
            
            total_theta += theta_contribution
        
        return {
            'total_theta': total_theta,
            'daily_decay': total_theta / 365,
            'decay_risk': 'high' if total_theta > 100000 else 'medium' if total_theta > 50000 else 'low'
        }
    
    def _analyze_pin_risk(self, options: List[Dict], spot_price: float) -> Dict:
        """Analyze pin risk (risk of pinning at strike)"""
        pin_risk_strikes = []
        
        for option in options:
            strike = option.get('strike', 0)
            
            # Check if strike is near spot and has high OI
            if abs(strike - spot_price) / spot_price < 0.01:  # Within 1%
                total_oi = 0
                if 'CE' in option:
                    total_oi += option['CE'].get('oi', 0)
                if 'PE' in option:
                    total_oi += option['PE'].get('oi', 0)
                
                if total_oi > 100000:  # High OI threshold
                    pin_risk_strikes.append({
                        'strike': strike,
                        'total_oi': total_oi,
                        'distance_from_spot': abs(strike - spot_price),
                        'pin_risk': 'high' if total_oi > 500000 else 'medium'
                    })
        
        return {
            'pin_risk_strikes': sorted(pin_risk_strikes, key=lambda x: x['total_oi'], reverse=True),
            'overall_pin_risk': 'high' if any(s['pin_risk'] == 'high' for s in pin_risk_strikes) else 'medium' if pin_risk_strikes else 'low'
        }
    
    def _calculate_var(self, options: List[Dict], spot_price: float) -> Dict:
        """Calculate Value at Risk (simplified)"""
        # Simplified VAR calculation based on IV
        total_iv_exposure = 0
        
        for option in options:
            if 'CE' in option:
                iv = option['CE'].get('iv', 0)
                oi = option['CE'].get('oi', 0)
                total_iv_exposure += iv * oi
            
            if 'PE' in option:
                iv = option['PE'].get('iv', 0)
                oi = option['PE'].get('oi', 0)
                total_iv_exposure += iv * oi
        
        # Convert to VAR (1-day, 95% confidence)
        var_1d_95 = total_iv_exposure * 1.645 / np.sqrt(252)  # Simplified
        
        return {
            'var_1d_95': var_1d_95,
            'var_1d_99': var_1d_95 * 2.33 / 1.645,
            'risk_level': 'high' if var_1d_95 > spot_price * 0.05 else 'medium' if var_1d_95 > spot_price * 0.02 else 'low'
        }
    
    def _analyze_concentration_risk(self, options: List[Dict]) -> Dict:
        """Analyze concentration risk"""
        # Check if OI is concentrated in few strikes
        total_oi = 0
        oi_by_strike = {}
        
        for option in options:
            strike = option.get('strike', 0)
            strike_oi = 0
            
            if 'CE' in option:
                strike_oi += option['CE'].get('oi', 0)
            if 'PE' in option:
                strike_oi += option['PE'].get('oi', 0)
            
            total_oi += strike_oi
            oi_by_strike[strike] = strike_oi
        
        # Calculate concentration
        sorted_oi = sorted(oi_by_strike.values(), reverse=True)
        
        if total_oi > 0:
            top_5_percent = sum(sorted_oi[:5]) / total_oi * 100
            top_10_percent = sum(sorted_oi[:10]) / total_oi * 100
        else:
            top_5_percent = 0
            top_10_percent = 0
        
        return {
            'top_5_strikes_percent': top_5_percent,
            'top_10_strikes_percent': top_10_percent,
            'concentration_risk': 'high' if top_5_percent > 50 else 'medium' if top_5_percent > 30 else 'low',
            'strikes_with_high_oi': len([oi for oi in sorted_oi if oi > statistics.mean(sorted_oi) * 2])
        }
    
    def _analyze_liquidity_risk(self, options: List[Dict]) -> Dict:
        """Analyze liquidity risk"""
        # Check bid-ask spreads and volume
        wide_spreads = 0
        low_volume = 0
        
        for option in options:
            if 'CE' in option:
                bid = option['CE'].get('bid', 0)
                ask = option['CE'].get('ask', 0)
                volume = option['CE'].get('volume', 0)
                
                if bid > 0 and ask > 0:
                    spread = (ask - bid) / ((bid + ask) / 2) * 100
                    if spread > 10:  # Wide spread
                        wide_spreads += 1
                
                if volume < 100:  # Low volume
                    low_volume += 1
            
            if 'PE' in option:
                bid = option['PE'].get('bid', 0)
                ask = option['PE'].get('ask', 0)
                volume = option['PE'].get('volume', 0)
                
                if bid > 0 and ask > 0:
                    spread = (ask - bid) / ((bid + ask) / 2) * 100
                    if spread > 10:
                        wide_spreads += 1
                
                if volume < 100:
                    low_volume += 1
        
        total_options = len(options) * 2  # Each strike has call and put
        
        return {
            'wide_spread_percent': (wide_spreads / total_options) * 100 if total_options > 0 else 0,
            'low_volume_percent': (low_volume / total_options) * 100 if total_options > 0 else 0,
            'liquidity_risk': 'high' if wide_spreads > total_options * 0.3 else 'medium' if wide_spreads > total_options * 0.1 else 'low'
        }
    
    def _analyze_volatility_risk(self, options: List[Dict]) -> Dict:
        """Analyze volatility risk"""
        ivs = []
        
        for option in options:
            if 'CE' in option:
                iv = option['CE'].get('iv', 0)
                if iv > 0:
                    ivs.append(iv)
            
            if 'PE' in option:
                iv = option['PE'].get('iv', 0)
                if iv > 0:
                    ivs.append(iv)
        
        if not ivs:
            return {'volatility_risk': 'low', 'avg_iv': 0, 'iv_std': 0}
        
        avg_iv = statistics.mean(ivs)
        iv_std = statistics.stdev(ivs) if len(ivs) > 1 else 0
        
        return {
            'avg_iv': avg_iv,
            'iv_std': iv_std,
            'volatility_risk': 'high' if avg_iv > 30 or iv_std > 10 else 'medium' if avg_iv > 20 or iv_std > 5 else 'low'
        }
    
    def _identify_psychological_levels(self, spot_price: float) -> List[Dict]:
        """Identify psychological support/resistance levels"""
        # Round numbers near spot price
        levels = []
        
        base = round(spot_price / 100) * 100  # Nearest 100
        
        for offset in [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]:
            level = base + offset
            distance = abs(level - spot_price)
            percent = (distance / spot_price) * 100
            
            if percent < 5:  # Within 5%
                levels.append({
                    'level': level,
                    'type': 'psychological',
                    'distance': distance,
                    'percent_distance': percent,
                    'strength': 1 - (percent / 5)  # Closer = stronger
                })
        
        return sorted(levels, key=lambda x: x['strength'], reverse=True)
    
    def _identify_recent_levels(self, options: List[Dict]) -> List[Dict]:
        """Identify recent highs/lows (placeholder)"""
        # Would need historical data
        return []
    
    def _calculate_pivot_points(self, spot_price: float) -> Dict:
        """Calculate pivot points"""
        # Simplified pivot points
        # Would need previous day's high, low, close
        pivot = spot_price  # Placeholder
        r1 = pivot * 1.01
        s1 = pivot * 0.99
        r2 = pivot * 1.02
        s2 = pivot * 0.98
        
        return {
            'pivot': pivot,
            'resistance1': r1,
            'resistance2': r2,
            'support1': s1,
            'support2': s2
        }
    
    def _extract_key_findings(self, analysis: Dict) -> List[str]:
        """Extract key findings from analysis"""
        findings = []
        
        # Check PCR
        pcr_oi = analysis.get('sentiment_analysis', {}).get('pcr_analysis', {}).get('oi', 1)
        if pcr_oi < 0.7:
            findings.append(f"Extremely low PCR(OI) of {pcr_oi:.2f} indicates bullish sentiment")
        elif pcr_oi > 1.5:
            findings.append(f"High PCR(OI) of {pcr_oi:.2f} indicates bearish sentiment")
        
        # Check OI buildup
        buildup = analysis.get('oi_analysis', {}).get('buildup_analysis', {})
        net_buildup = buildup.get('net_buildup', 0)
        if abs(net_buildup) > 100000:
            direction = 'calls' if net_buildup > 0 else 'puts'
            findings.append(f"Significant OI buildup in {direction}: {abs(net_buildup):,} contracts")
        
        # Check max pain
        max_pain = analysis.get('max_pain', {})
        max_pain_strike = max_pain.get('max_pain_strike', 0)
        spot = analysis.get('spot_price', 0)
        if abs(max_pain_strike - spot) / spot > 0.02:
            direction = 'above' if max_pain_strike > spot else 'below'
            findings.append(f"Max pain {direction} spot by {(abs(max_pain_strike - spot)/spot*100):.1f}%")
        
        # Check IV
        iv_percentile = analysis.get('iv_analysis', {}).get('percentile_analysis', {}).get('percentile', 50)
        if iv_percentile > 70:
            findings.append(f"High IV percentile: {iv_percentile:.0f} (expensive options)")
        elif iv_percentile < 30:
            findings.append(f"Low IV percentile: {iv_percentile:.0f} (cheap options)")
        
        # Check risk
        risk = analysis.get('risk_analysis', {})
        if risk.get('pin_risk', {}).get('overall_pin_risk') == 'high':
            findings.append("High pin risk detected near current spot")
        
        return findings[:5]  # Return top 5 findings
    
    def _generate_trading_signals(self, analysis: Dict) -> List[Dict]:
        """Generate trading signals"""
        signals = []
        
        spot = analysis.get('spot_price', 0)
        sentiment = analysis.get('sentiment_analysis', {}).get('overall_sentiment', {})
        sentiment_level = sentiment.get('level', 'neutral')
        sentiment_score = sentiment.get('score', 0)
        
        # Generate signals based on analysis
        if sentiment_level == 'very_bullish' and sentiment_score > 0.7:
            signals.append({
                'type': 'bullish',
                'strength': 'strong',
                'signal': 'BUY_CALLS',
                'reason': 'Strong bullish sentiment with high confidence',
                'confidence': sentiment.get('confidence', 0.5)
            })
        
        elif sentiment_level == 'very_bearish' and sentiment_score < -0.7:
            signals.append({
                'type': 'bearish',
                'strength': 'strong',
                'signal': 'BUY_PUTS',
                'reason': 'Strong bearish sentiment with high confidence',
                'confidence': sentiment.get('confidence', 0.5)
            })
        
        # Check for mean reversion signals
        pcr_oi = analysis.get('sentiment_analysis', {}).get('pcr_analysis', {}).get('oi', 1)
        if pcr_oi < 0.7:
            signals.append({
                'type': 'mean_reversion',
                'strength': 'medium',
                'signal': 'SELL_CALLS_BUY_PUTS',
                'reason': f'Extreme low PCR(OI): {pcr_oi:.2f} suggests overbought conditions',
                'confidence': 0.6
            })
        elif pcr_oi > 1.5:
            signals.append({
                'type': 'mean_reversion',
                'strength': 'medium',
                'signal': 'BUY_CALLS_SELL_PUTS',
                'reason': f'Extreme high PCR(OI): {pcr_oi:.2f} suggests oversold conditions',
                'confidence': 0.6
            })
        
        # Check IV signals
        iv_percentile = analysis.get('iv_analysis', {}).get('percentile_analysis', {}).get('percentile', 50)
        if iv_percentile > 70:
            signals.append({
                'type': 'volatility',
                'strength': 'medium',
                'signal': 'SELL_VOLATILITY',
                'reason': f'High IV percentile: {iv_percentile:.0f} (sell premium)',
                'confidence': 0.7
            })
        elif iv_percentile < 30:
            signals.append({
                'type': 'volatility',
                'strength': 'medium',
                'signal': 'BUY_VOLATILITY',
                'reason': f'Low IV percentile: {iv_percentile:.0f} (buy premium)',
                'confidence': 0.7
            })
        
        return signals[:3]  # Return top 3 signals
    
    def _calculate_confidence_score(self, analysis: Dict) -> float:
        """Calculate overall confidence score"""
        confidence_factors = []
        
        # Data completeness
        total_options = analysis.get('total_options', 0)
        if total_options > 100:
            confidence_factors.append(0.9)
        elif total_options > 50:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Sentiment consistency
        sentiment_confidence = analysis.get('sentiment_analysis', {}).get('overall_sentiment', {}).get('confidence', 0.5)
        confidence_factors.append(sentiment_confidence)
        
        # Risk assessment completeness
        risk_factors = len(analysis.get('risk_analysis', {}))
        if risk_factors > 5:
            confidence_factors.append(0.8)
        elif risk_factors > 3:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    def _assess_overall_risk(self, analysis: Dict) -> str:
        """Assess overall risk level"""
        risk_factors = []
        
        pin_risk = analysis.get('risk_analysis', {}).get('pin_risk', {}).get('overall_pin_risk', 'low')
        if pin_risk == 'high':
            risk_factors.append(3)
        elif pin_risk == 'medium':
            risk_factors.append(2)
        else:
            risk_factors.append(1)
        
        var_risk = analysis.get('risk_analysis', {}).get('value_at_risk', {}).get('risk_level', 'low')
        if var_risk == 'high':
            risk_factors.append(3)
        elif var_risk == 'medium':
            risk_factors.append(2)
        else:
            risk_factors.append(1)
        
        volatility_risk = analysis.get('risk_analysis', {}).get('volatility_risk', {}).get('volatility_risk', 'low')
        if volatility_risk == 'high':
            risk_factors.append(3)
        elif volatility_risk == 'medium':
            risk_factors.append(2)
        else:
            risk_factors.append(1)
        
        avg_risk = statistics.mean(risk_factors) if risk_factors else 1
        
        if avg_risk > 2.5:
            return 'very_high'
        elif avg_risk > 2:
            return 'high'
        elif avg_risk > 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _interpret_pcr(self, pcr: float) -> str:
        """Interpret PCR value"""
        if pcr < 0.7:
            return "Extremely Bullish"
        elif pcr < 0.9:
            return "Bullish"
        elif pcr < 1.1:
            return "Neutral"
        elif pcr < 1.3:
            return "Mildly Bearish"
        elif pcr < 1.5:
            return "Bearish"
        else:
            return "Extremely Bearish"
    
    def _interpret_iv_percentile(self, percentile: float) -> str:
        """Interpret IV percentile"""
        if percentile > 80:
            return "Options are expensive (high IV)"
        elif percentile > 60:
            return "Options are relatively expensive"
        elif percentile > 40:
            return "IV is at average levels"
        elif percentile > 20:
            return "Options are relatively cheap"
        else:
            return "Options are cheap (low IV)"
    
    def _determine_skew_type(self, call_skew: float, put_skew: float, put_call_skew: float) -> str:
        """Determine type of volatility skew"""
        if call_skew > 1.2 and put_skew > 1.2:
            return "volatility_smile"
        elif call_skew > 1.2:
            return "reverse_skew"
        elif put_skew > 1.2:
            return "forward_skew"
        elif put_call_skew > 1.1:
            return "put_skew"
        elif put_call_skew < 0.9:
            return "call_skew"
        else:
            return "flat"
    
    def _calculate_volume_value_area(self, volume_profile: List[Dict]) -> Dict:
        """Calculate Volume Value Area (simplified)"""
        if not volume_profile:
            return {}
        
        total_volume = sum(vp['total_volume'] for vp in volume_profile)
        if total_volume == 0:
            return {}
        
        # Calculate cumulative volume
        sorted_profile = sorted(volume_profile, key=lambda x: x['strike'])
        cumulative = 0
        vwap_candidates = []
        
        for vp in sorted_profile:
            cumulative += vp['total_volume']
            if cumulative <= total_volume * 0.7:  # 70% of volume
                vwap_candidates.append(vp)
        
        if vwap_candidates:
            vah = max(vp['strike'] for vp in vwap_candidates)  # Value Area High
            val = min(vp['strike'] for vp in vwap_candidates)  # Value Area Low
            poc = max(vwap_candidates, key=lambda x: x['total_volume'])['strike']  # Point of Control
            
            return {
                'value_area_high': vah,
                'value_area_low': val,
                'point_of_control': poc,
                'value_area_width': vah - val,
                'percent_of_total_volume': (sum(vp['total_volume'] for vp in vwap_candidates) / total_volume) * 100
            }
        
        return {}