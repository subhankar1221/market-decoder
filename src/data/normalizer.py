"""
Data Normalizer - Standardizes data from different sources
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

from src.utils.logger import get_logger
from src.utils.date_utils import parse_nse_date, convert_to_datetime

logger = get_logger(__name__)

class DataNormalizer:
    """Normalizes data from various NSE sources"""
    
    def __init__(self, config):
        self.config = config
        
        # Option chain column mappings
        self.option_columns = {
            'strikePrice': 'strike',
            'expiryDate': 'expiry',
            'openInterest': 'oi',
            'changeinOpenInterest': 'change_in_oi',
            'totalTradedVolume': 'volume',
            'impliedVolatility': 'iv',
            'lastPrice': 'last_price',
            'change': 'change',
            'pChange': 'percent_change',
            'bidprice': 'bid',
            'askPrice': 'ask',
            'bidQty': 'bid_qty',
            'askQty': 'ask_qty',
            'underlyingValue': 'underlying'
        }
        
        # Index mappings
        self.index_columns = {
            'index': 'name',
            'last': 'last_price',
            'variation': 'change',
            'percentChange': 'percent_change',
            'yearHigh': 'year_high',
            'yearLow': 'year_low',
            'pe': 'pe_ratio',
            'pb': 'pb_ratio',
            'dy': 'dividend_yield'
        }
    
    def normalize_option_chain(self, raw_data: Dict) -> Dict:
        """Normalize option chain data"""
        try:
            if not raw_data or 'records' not in raw_data:
                return {}
            
            records = raw_data['records']
            
            # Extract basic information
            normalized = {
                'timestamp': datetime.now().isoformat(),
                'underlying': records.get('underlyingValue', 0),
                'spot_price': records.get('underlyingValue', 0),
                'expiry_dates': records.get('expiryDates', []),
                'current_expiry': records.get('expiryDates', [])[0] if records.get('expiryDates') else None,
                'strike_prices': records.get('strikePrices', []),
                'timestamp_original': records.get('timestamp', ''),
                'vix': self._extract_vix(records),
                'data': []
            }
            
            # Process each option
            for option in records.get('data', []):
                normalized_option = self._normalize_option(option)
                if normalized_option:
                    normalized['data'].append(normalized_option)
            
            # Calculate summary statistics
            normalized.update(self._calculate_option_summary(normalized['data']))
            
            logger.debug(f"Normalized {len(normalized['data'])} option records")
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize option chain: {e}")
            return {}
    
    def _normalize_option(self, option: Dict) -> Optional[Dict]:
        """Normalize a single option record"""
        try:
            strike = option.get('strikePrice', 0)
            
            normalized = {
                'strike': strike,
                'timestamp': datetime.now().isoformat()
            }
            
            # Process Call option
            if 'CE' in option and option['CE']:
                ce_data = option['CE']
                normalized['CE'] = self._normalize_option_data(ce_data, 'CE', strike)
            
            # Process Put option
            if 'PE' in option and option['PE']:
                pe_data = option['PE']
                normalized['PE'] = self._normalize_option_data(pe_data, 'PE', strike)
            
            return normalized if 'CE' in normalized or 'PE' in normalized else None
            
        except Exception as e:
            logger.error(f"Failed to normalize option: {e}")
            return None
    
    def _normalize_option_data(self, data: Dict, option_type: str, strike: float) -> Dict:
        """Normalize option data for a specific type"""
        normalized = {
            'type': option_type,
            'strike': strike,
            'timestamp': datetime.now().isoformat()
        }
        
        # Map columns
        for raw_key, norm_key in self.option_columns.items():
            if raw_key in data:
                value = data[raw_key]
                
                # Convert string numbers to float
                if isinstance(value, str) and self._is_numeric(value):
                    value = float(value.replace(',', ''))
                
                normalized[norm_key] = value
        
        # Calculate additional fields
        normalized.update(self._calculate_option_fields(normalized, option_type))
        
        return normalized
    
    def _calculate_option_fields(self, option_data: Dict, option_type: str) -> Dict:
        """Calculate additional option fields"""
        calculated = {}
        
        # Calculate moneyness
        underlying = option_data.get('underlying', 0)
        strike = option_data.get('strike', 0)
        
        if underlying > 0 and strike > 0:
            if option_type == 'CE':
                if strike < underlying * 0.98:
                    moneyness = 'ITM'
                elif strike > underlying * 1.02:
                    moneyness = 'OTM'
                else:
                    moneyness = 'ATM'
            else:  # PE
                if strike > underlying * 1.02:
                    moneyness = 'ITM'
                elif strike < underlying * 0.98:
                    moneyness = 'OTM'
                else:
                    moneyness = 'ATM'
            
            calculated['moneyness'] = moneyness
            
            # Calculate intrinsic and extrinsic value
            last_price = option_data.get('last_price', 0)
            if option_type == 'CE':
                intrinsic = max(underlying - strike, 0)
            else:
                intrinsic = max(strike - underlying, 0)
            
            extrinsic = max(last_price - intrinsic, 0)
            
            calculated['intrinsic_value'] = intrinsic
            calculated['extrinsic_value'] = extrinsic
            calculated['time_value'] = extrinsic
        
        return calculated
    
    def _calculate_option_summary(self, options: List[Dict]) -> Dict:
        """Calculate summary statistics for options"""
        summary = {
            'total_oi_calls': 0,
            'total_oi_puts': 0,
            'total_volume_calls': 0,
            'total_volume_puts': 0,
            'total_change_oi_calls': 0,
            'total_change_oi_puts': 0,
            'strikes_with_data': len(options),
            'atm_strike': 0,
            'max_oi_strike_call': {'strike': 0, 'oi': 0},
            'max_oi_strike_put': {'strike': 0, 'oi': 0}
        }
        
        if not options:
            return summary
        
        # Calculate totals
        for option in options:
            if 'CE' in option:
                ce = option['CE']
                summary['total_oi_calls'] += ce.get('oi', 0)
                summary['total_volume_calls'] += ce.get('volume', 0)
                summary['total_change_oi_calls'] += ce.get('change_in_oi', 0)
                
                if ce.get('oi', 0) > summary['max_oi_strike_call']['oi']:
                    summary['max_oi_strike_call'] = {'strike': option['strike'], 'oi': ce['oi']}
            
            if 'PE' in option:
                pe = option['PE']
                summary['total_oi_puts'] += pe.get('oi', 0)
                summary['total_volume_puts'] += pe.get('volume', 0)
                summary['total_change_oi_puts'] += pe.get('change_in_oi', 0)
                
                if pe.get('oi', 0) > summary['max_oi_strike_put']['oi']:
                    summary['max_oi_strike_put'] = {'strike': option['strike'], 'oi': pe['oi']}
        
        # Calculate PCR
        if summary['total_oi_calls'] > 0:
            summary['pcr_oi'] = summary['total_oi_puts'] / summary['total_oi_calls']
        else:
            summary['pcr_oi'] = 0
        
        if summary['total_volume_calls'] > 0:
            summary['pcr_volume'] = summary['total_volume_puts'] / summary['total_volume_calls']
        else:
            summary['pcr_volume'] = 0
        
        return summary
    
    def _extract_vix(self, records: Dict) -> Dict:
        """Extract VIX data"""
        vix_data = records.get('VIX', {})
        
        return {
            'value': vix_data.get('value', 0),
            'change': vix_data.get('change', 0),
            'percent_change': vix_data.get('percentChange', 0)
        }
    
    def normalize_market_stats(self, raw_data: Dict) -> Dict:
        """Normalize market statistics"""
        try:
            if not raw_data or 'data' not in raw_data:
                return {}
            
            data = raw_data['data']
            
            normalized = {
                'timestamp': datetime.now().isoformat(),
                'advances': data.get('advances', 0),
                'declines': data.get('declines', 0),
                'unchanged': data.get('unchanged', 0),
                'total_securities': data.get('totalSecurities', 0)
            }
            
            # Extract PCR data
            pcr_data = data.get('putCallRatio', {})
            if pcr_data:
                normalized['pcr'] = {
                    'total': pcr_data.get('total', 0),
                    'index': pcr_data.get('index', 0),
                    'stock': pcr_data.get('stock', 0)
                }
            
            # Extract additional stats
            normalized.update(self._extract_additional_stats(data))
            
            # Calculate ratios
            if normalized['declines'] > 0:
                normalized['advance_decline_ratio'] = normalized['advances'] / normalized['declines']
            else:
                normalized['advance_decline_ratio'] = 0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize market stats: {e}")
            return {}
    
    def _extract_additional_stats(self, data: Dict) -> Dict:
        """Extract additional market statistics"""
        additional = {}
        
        # Market cap statistics
        market_cap = data.get('marketCap', {})
        if market_cap:
            additional['market_cap'] = {
                'total': market_cap.get('totalMktCap', 0),
                'advances': market_cap.get('advancesMktCap', 0),
                'declines': market_cap.get('declinesMktCap', 0)
            }
        
        # 52-week statistics
        week_stats = data.get('52week', {})
        if week_stats:
            additional['52week'] = {
                'high': week_stats.get('high', 0),
                'low': week_stats.get('low', 0),
                'high_percent': week_stats.get('highPer', 0),
                'low_percent': week_stats.get('lowPer', 0)
            }
        
        return additional
    
    def normalize_indices(self, raw_data: Dict) -> Dict:
        """Normalize indices data"""
        try:
            if not raw_data or 'data' not in raw_data:
                return {}
            
            indices_data = raw_data['data']
            
            normalized = {
                'timestamp': datetime.now().isoformat(),
                'indices': [],
                'summary': {}
            }
            
            # Process each index
            for index in indices_data:
                normalized_index = self._normalize_index(index)
                if normalized_index:
                    normalized['indices'].append(normalized_index)
            
            # Calculate summary
            normalized['summary'] = self._calculate_indices_summary(normalized['indices'])
            
            logger.debug(f"Normalized {len(normalized['indices'])} indices")
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize indices: {e}")
            return {}
    
    def _normalize_index(self, index: Dict) -> Optional[Dict]:
        """Normalize a single index"""
        try:
            normalized = {
                'timestamp': datetime.now().isoformat()
            }
            
            # Map columns
            for raw_key, norm_key in self.index_columns.items():
                if raw_key in index:
                    value = index[raw_key]
                    
                    # Convert string numbers
                    if isinstance(value, str) and self._is_numeric(value):
                        value = float(value.replace(',', ''))
                    
                    normalized[norm_key] = value
            
            # Calculate additional fields
            normalized.update(self._calculate_index_fields(normalized))
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize index: {e}")
            return None
    
    def _calculate_index_fields(self, index: Dict) -> Dict:
        """Calculate additional index fields"""
        calculated = {}
        
        # Calculate market status
        last_price = index.get('last_price', 0)
        year_high = index.get('year_high', 0)
        year_low = index.get('year_low', 0)
        
        if last_price > 0 and year_high > 0 and year_low > 0:
            # Distance from 52-week high/low
            calculated['distance_from_high'] = ((year_high - last_price) / year_high) * 100
            calculated['distance_from_low'] = ((last_price - year_low) / year_low) * 100
            
            # Market position
            if calculated['distance_from_high'] < 5:
                calculated['market_position'] = 'near_high'
            elif calculated['distance_from_low'] < 5:
                calculated['market_position'] = 'near_low'
            else:
                calculated['market_position'] = 'mid_range'
        
        return calculated
    
    def _calculate_indices_summary(self, indices: List[Dict]) -> Dict:
        """Calculate summary for all indices"""
        summary = {
            'total_indices': len(indices),
            'advancing': 0,
            'declining': 0,
            'unchanged': 0,
            'top_gainers': [],
            'top_losers': []
        }
        
        if not indices:
            return summary
        
        # Count advancing/declining
        for index in indices:
            change = index.get('percent_change', 0)
            if change > 0:
                summary['advancing'] += 1
            elif change < 0:
                summary['declining'] += 1
            else:
                summary['unchanged'] += 1
        
        # Sort by performance
        sorted_by_gain = sorted(indices, key=lambda x: x.get('percent_change', 0), reverse=True)
        sorted_by_loss = sorted(indices, key=lambda x: x.get('percent_change', 0))
        
        summary['top_gainers'] = sorted_by_gain[:5]
        summary['top_losers'] = sorted_by_loss[:5]
        
        return summary
    
    def normalize_turnover(self, raw_data: Dict) -> Dict:
        """Normalize market turnover data"""
        try:
            if not raw_data or 'data' not in raw_data:
                return {}
            
            data = raw_data['data']
            
            normalized = {
                'timestamp': datetime.now().isoformat(),
                'segments': {}
            }
            
            # Process each segment
            segments = ['Cash', 'F&O', 'SLB']
            for segment in segments:
                if segment in data:
                    segment_data = data[segment]
                    normalized['segments'][segment.lower()] = {
                        'turnover': segment_data.get('turnoverInCrore', 0),
                        'percent_change': segment_data.get('percentChange', 0),
                        'market_share': segment_data.get('marketShare', 0)
                    }
            
            # Calculate totals
            total_turnover = sum(seg['turnover'] for seg in normalized['segments'].values())
            normalized['total_turnover'] = total_turnover
            
            # Calculate segment percentages
            for segment_name, segment_data in normalized['segments'].items():
                if total_turnover > 0:
                    segment_data['percentage_of_total'] = (segment_data['turnover'] / total_turnover) * 100
                else:
                    segment_data['percentage_of_total'] = 0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize turnover: {e}")
            return {}
    
    def normalize_fii_dii(self, raw_data: Dict) -> Dict:
        """Normalize FII/DII data"""
        try:
            if not raw_data:
                return {}
            
            normalized = {
                'timestamp': datetime.now().isoformat(),
                'fii': {},
                'dii': {},
                'summary': {}
            }
            
            # Extract FII data
            if 'data' in raw_data and 'fii' in raw_data['data']:
                fii_data = raw_data['data']['fii']
                normalized['fii'] = self._normalize_fii_dii_segment(fii_data, 'FII')
            
            # Extract DII data
            if 'data' in raw_data and 'dii' in raw_data['data']:
                dii_data = raw_data['data']['dii']
                normalized['dii'] = self._normalize_fii_dii_segment(dii_data, 'DII')
            
            # Calculate summary
            normalized['summary'] = self._calculate_fii_dii_summary(normalized)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize FII/DII data: {e}")
            return {}
    
    def _normalize_fii_dii_segment(self, data: Dict, segment_type: str) -> Dict:
        """Normalize FII/DII segment data"""
        normalized = {
            'type': segment_type,
            'timestamp': datetime.now().isoformat(),
            'activity': {}
        }
        
        # Extract activity by type
        activity_types = ['equity', 'debt', 'total']
        for act_type in activity_types:
            if act_type in data:
                act_data = data[act_type]
                normalized['activity'][act_type] = {
                    'buy': act_data.get('buy', 0),
                    'sell': act_data.get('sell', 0),
                    'net': act_data.get('net', 0)
                }
        
        return normalized
    
    def _calculate_fii_dii_summary(self, data: Dict) -> Dict:
        """Calculate FII/DII summary"""
        summary = {
            'net_inflow': 0,
            'sentiment': 'neutral'
        }
        
        # Calculate total net inflow
        fii_net = data.get('fii', {}).get('activity', {}).get('total', {}).get('net', 0)
        dii_net = data.get('dii', {}).get('activity', {}).get('total', {}).get('net', 0)
        
        summary['net_inflow'] = fii_net + dii_net
        
        # Determine sentiment
        if summary['net_inflow'] > 1000:  # > 1000 Cr
            summary['sentiment'] = 'very_bullish'
        elif summary['net_inflow'] > 500:
            summary['sentiment'] = 'bullish'
        elif summary['net_inflow'] < -1000:
            summary['sentiment'] = 'very_bearish'
        elif summary['net_inflow'] < -500:
            summary['sentiment'] = 'bearish'
        else:
            summary['sentiment'] = 'neutral'
        
        return summary
    
    def _is_numeric(self, value: str) -> bool:
        """Check if string is numeric"""
        try:
            float(value.replace(',', ''))
            return True
        except:
            return False
    
    def normalize_all(self, raw_data: Dict) -> Dict:
        """Normalize all data sources"""
        try:
            normalized = {
                'timestamp': datetime.now().isoformat(),
                'sources': {}
            }
            
            # Normalize each source
            for source_name, data in raw_data.items():
                if source_name == 'metadata':
                    continue
                
                if source_name.startswith('option_chain'):
                    normalized['sources'][source_name] = self.normalize_option_chain(data)
                elif source_name == 'market_statistics':
                    normalized['sources'][source_name] = self.normalize_market_stats(data)
                elif source_name == 'market_turnover':
                    normalized['sources'][source_name] = self.normalize_turnover(data)
                elif source_name.startswith('indices'):
                    normalized['sources'][source_name] = self.normalize_indices(data)
                elif source_name == 'fii_dii':
                    normalized['sources'][source_name] = self.normalize_fii_dii(data)
                else:
                    # Keep as-is for unknown sources
                    normalized['sources'][source_name] = data
            
            # Add metadata
            if 'metadata' in raw_data:
                normalized['metadata'] = raw_data['metadata']
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize all data: {e}")
            return {}