"""
Output formatters for Market Decoder
Console, JSON, and file formatting utilities
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from tabulate import tabulate
from colorama import init, Fore, Back, Style

from src.utils.logger import get_logger
from src.utils.date_utils import get_ist_now, format_duration

logger = get_logger(__name__)

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class ConsoleFormatter:
    """Formatter for console output with colors and formatting"""
    
    # Color mapping for sentiment
    SENTIMENT_COLORS = {
        'very_bullish': Fore.GREEN + Style.BRIGHT,
        'bullish': Fore.GREEN,
        'neutral': Fore.YELLOW,
        'bearish': Fore.RED,
        'very_bearish': Fore.RED + Style.BRIGHT
    }
    
    # Color mapping for strength
    STRENGTH_COLORS = {
        'very_strong': Style.BRIGHT,
        'strong': '',
        'moderate': Style.DIM,
        'weak': Style.DIM + Fore.WHITE
    }
    
    # Color mapping for signals
    SIGNAL_COLORS = {
        'bullish': Fore.GREEN,
        'bearish': Fore.RED,
        'neutral': Fore.YELLOW,
        'caution': Fore.MAGENTA,
        'warning': Fore.RED + Style.BRIGHT
    }
    
    # Unicode symbols
    SYMBOLS = {
        'up': '📈',
        'down': '📉',
        'bull': '🐂',
        'bear': '🐻',
        'neutral': '➖',
        'warning': '⚠️',
        'success': '✅',
        'error': '❌',
        'info': 'ℹ️',
        'clock': '⏰',
        'money': '💰',
        'chart': '📊',
        'risk': '⚡',
        'signal': '🚦',
        'alert': '🔔',
        'level': '🔑',
        'volume': '📈',
        'volatility': '🌊',
        'greeks': '𝛿𝛾𝜃𝜈𝜌'
    }
    
    def __init__(self, config):
        self.config = config
        self.output_config = config.get('output', {}).get('console', {})
        self.use_colors = self.output_config.get('color_codes', True)
        self.use_unicode = self.output_config.get('unicode_symbols', True)
        
    async def display_report(self, report: Dict):
        """Display complete market analysis report"""
        try:
            print("\n" + "="*80)
            await self._display_header(report)
            await self._display_summary(report)
            await self._display_sentiment(report)
            await self._display_option_analysis(report)
            await self._display_greeks_analysis(report)
            await self._display_key_levels(report)
            await self._display_signals(report)
            await self._display_alerts(report)
            await self._display_footer(report)
            print("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to display report: {e}")
            print(f"{Fore.RED}Error displaying report: {e}{Style.RESET_ALL}")
    
    async def _display_header(self, report: Dict):
        """Display report header"""
        timestamp = report.get('timestamp', get_ist_now())
        symbol = report.get('symbol', 'NIFTY')
        
        # Format timestamp
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S IST')
            except:
                timestamp_str = timestamp
        else:
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S IST')
        
        title = f"{self.SYMBOLS['chart']} MARKET DECODER PRO - {symbol}"
        
        if self.use_colors:
            print(f"{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Timestamp: {timestamp_str}{Style.RESET_ALL}")
        else:
            print(title)
            print(f"Timestamp: {timestamp_str}")
        
        print("-"*80)
    
    async def _display_summary(self, report: Dict):
        """Display market summary"""
        market_data = report.get('market_data', {})
        
        if not market_data:
            return
        
        spot_price = market_data.get('spot_price', 0)
        change = market_data.get('change', 0)
        percent_change = market_data.get('percent_change', 0)
        
        # Determine color for change
        if self.use_colors:
            if change > 0:
                change_color = Fore.GREEN
                change_symbol = self.SYMBOLS['up']
            elif change < 0:
                change_color = Fore.RED
                change_symbol = self.SYMBOLS['down']
            else:
                change_color = Fore.YELLOW
                change_symbol = self.SYMBOLS['neutral']
        else:
            change_color = ""
            change_symbol = "▲" if change > 0 else "▼" if change < 0 else "➖"
        
        # Format spot price
        spot_formatted = f"{spot_price:,.2f}"
        change_formatted = f"{change:+,.2f}"
        percent_formatted = f"{percent_change:+,.2f}%"
        
        # Display
        if self.use_colors:
            print(f"{Fore.WHITE}{self.SYMBOLS['money']} Spot Price: {Fore.YELLOW}{spot_formatted}{Style.RESET_ALL}")
            print(f"   Change: {change_color}{change_symbol} {change_formatted} ({percent_formatted}){Style.RESET_ALL}")
        else:
            print(f"{self.SYMBOLS['money']} Spot Price: {spot_formatted}")
            print(f"   Change: {change_symbol} {change_formatted} ({percent_formatted})")
        
        # Display additional market stats
        if 'vix' in market_data:
            vix = market_data['vix']
            vix_change = market_data.get('vix_change', 0)
            
            # Determine VIX color
            if self.use_colors:
                if vix > 25:
                    vix_color = Fore.RED + Style.BRIGHT
                elif vix > 20:
                    vix_color = Fore.RED
                elif vix > 15:
                    vix_color = Fore.YELLOW
                else:
                    vix_color = Fore.GREEN
            else:
                vix_color = ""
            
            vix_symbol = self.SYMBOLS['volatility']
            vix_change_symbol = "▲" if vix_change > 0 else "▼" if vix_change < 0 else "➖"
            
            if self.use_colors:
                print(f"{Fore.WHITE}{vix_symbol} VIX: {vix_color}{vix:.2f}{Style.RESET_ALL} "
                      f"({vix_change_symbol} {vix_change:+.2f})")
            else:
                print(f"{vix_symbol} VIX: {vix:.2f} ({vix_change_symbol} {vix_change:+.2f})")
        
        print()
    
    async def _display_sentiment(self, report: Dict):
        """Display market sentiment"""
        sentiment_analysis = report.get('sentiment_analysis', {})
        
        if not sentiment_analysis:
            return
        
        overall = sentiment_analysis.get('overall_sentiment', {})
        sentiment = overall.get('sentiment', 'neutral')
        strength = overall.get('strength', 'weak')
        score = overall.get('overall_score', 0)
        confidence = sentiment_analysis.get('confidence', 0)
        
        # Get sentiment symbol
        if sentiment == 'very_bullish' or sentiment == 'bullish':
            sentiment_symbol = self.SYMBOLS['bull']
        elif sentiment == 'very_bearish' or sentiment == 'bearish':
            sentiment_symbol = self.SYMBOLS['bear']
        else:
            sentiment_symbol = self.SYMBOLS['neutral']
        
        # Get colors
        if self.use_colors:
            sentiment_color = self.SENTIMENT_COLORS.get(sentiment, Fore.YELLOW)
            strength_color = self.STRENGTH_COLORS.get(strength, '')
        else:
            sentiment_color = ""
            strength_color = ""
        
        # Format sentiment text
        sentiment_text = sentiment.replace('_', ' ').title()
        strength_text = strength.replace('_', ' ').title()
        
        # Display sentiment
        if self.use_colors:
            print(f"{Fore.CYAN}{self.SYMBOLS['chart']} MARKET SENTIMENT{Style.RESET_ALL}")
            print(f"   {sentiment_symbol} {sentiment_color}{sentiment_text}{Style.RESET_ALL} "
                  f"({strength_color}{strength_text}{Style.RESET_ALL})")
            print(f"   Score: {score:+.2f} | Confidence: {confidence:.0%}")
        else:
            print(f"{self.SYMBOLS['chart']} MARKET SENTIMENT")
            print(f"   {sentiment_symbol} {sentiment_text} ({strength_text})")
            print(f"   Score: {score:+.2f} | Confidence: {confidence:.0%}")
        
        # Display PCR if available
        components = sentiment_analysis.get('components', {})
        pcr_analysis = components.get('pcr_analysis', {})
        
        if pcr_analysis:
            pcr_oi = pcr_analysis.get('pcr_oi', 0)
            interpretation = pcr_analysis.get('interpretation', '')
            
            # Color for PCR
            if self.use_colors:
                if pcr_oi < 0.9:
                    pcr_color = Fore.GREEN
                elif pcr_oi > 1.3:
                    pcr_color = Fore.RED
                else:
                    pcr_color = Fore.YELLOW
            else:
                pcr_color = ""
            
            if self.use_colors:
                print(f"   {Fore.WHITE}PCR(OI):{Style.RESET_ALL} {pcr_color}{pcr_oi:.2f}{Style.RESET_ALL} - {interpretation}")
            else:
                print(f"   PCR(OI): {pcr_oi:.2f} - {interpretation}")
        
        print()
    
    async def _display_option_analysis(self, report: Dict):
        """Display option chain analysis"""
        option_analysis = report.get('option_analysis', {})
        
        if not option_analysis:
            return
        
        oi_analysis = option_analysis.get('oi_analysis', {})
        totals = oi_analysis.get('totals', {})
        volume_analysis = option_analysis.get('volume_analysis', {})
        
        # Display OI totals
        total_call_oi = totals.get('calls', 0)
        total_put_oi = totals.get('puts', 0)
        total_oi = totals.get('total', 0)
        
        # Calculate percentages
        if total_oi > 0:
            call_percent = (total_call_oi / total_oi) * 100
            put_percent = (total_put_oi / total_oi) * 100
        else:
            call_percent = put_percent = 0
        
        # Display OI analysis
        if self.use_colors:
            print(f"{Fore.CYAN}{self.SYMBOLS['volume']} OPEN INTEREST ANALYSIS{Style.RESET_ALL}")
            print(f"   Calls: {Fore.BLUE}{total_call_oi:,}{Style.RESET_ALL} ({call_percent:.1f}%) | "
                  f"Puts: {Fore.MAGENTA}{total_put_oi:,}{Style.RESET_ALL} ({put_percent:.1f}%) | "
                  f"Total: {Fore.WHITE}{total_oi:,}{Style.RESET_ALL}")
        else:
            print(f"{self.SYMBOLS['volume']} OPEN INTEREST ANALYSIS")
            print(f"   Calls: {total_call_oi:,} ({call_percent:.1f}%) | "
                  f"Puts: {total_put_oi:,} ({put_percent:.1f}%) | "
                  f"Total: {total_oi:,}")
        
        # Display PCR
        ratios = oi_analysis.get('ratios', {})
        pcr_oi = ratios.get('pcr_oi', 0)
        
        if self.use_colors:
            if pcr_oi < 0.9:
                pcr_color = Fore.GREEN
            elif pcr_oi > 1.3:
                pcr_color = Fore.RED
            else:
                pcr_color = Fore.YELLOW
        else:
            pcr_color = ""
        
        if self.use_colors:
            print(f"   {Fore.WHITE}PCR(OI):{Style.RESET_ALL} {pcr_color}{pcr_oi:.2f}{Style.RESET_ALL}")
        else:
            print(f"   PCR(OI): {pcr_oi:.2f}")
        
        # Display Max Pain if available
        max_pain = option_analysis.get('max_pain', {})
        max_pain_strike = max_pain.get('max_pain_strike', 0)
        
        if max_pain_strike > 0:
            spot_price = report.get('market_data', {}).get('spot_price', 0)
            distance = abs(max_pain_strike - spot_price)
            percent_distance = (distance / spot_price) * 100 if spot_price > 0 else 0
            
            # Color for Max Pain
            if self.use_colors:
                if percent_distance > 2:
                    mp_color = Fore.YELLOW
                else:
                    mp_color = Fore.WHITE
            else:
                mp_color = ""
            
            if self.use_colors:
                print(f"   {Fore.WHITE}Max Pain:{Style.RESET_ALL} {mp_color}{max_pain_strike:,.0f}{Style.RESET_ALL} "
                      f"(±{percent_distance:.1f}% from spot)")
            else:
                print(f"   Max Pain: {max_pain_strike:,.0f} (±{percent_distance:.1f}% from spot)")
        
        print()
    
    async def _display_greeks_analysis(self, report: Dict):
        """Display Greeks analysis"""
        greeks_analysis = report.get('greeks_analysis', {})
        
        if not greeks_analysis:
            return
        
        summary = greeks_analysis.get('summary', {})
        delta_summary = summary.get('delta', {})
        gamma_summary = summary.get('gamma', {})
        theta_summary = summary.get('theta', {})
        
        # Display Greeks summary
        if self.use_colors:
            print(f"{Fore.CYAN}{self.SYMBOLS['greeks']} GREEKS ANALYSIS{Style.RESET_ALL}")
        else:
            print(f"{self.SYMBOLS['greeks']} GREEKS ANALYSIS")
        
        # Delta
        delta_mean = delta_summary.get('mean', 0)
        delta_risk = summary.get('risk_assessment', {}).get('delta_risk', 'low')
        
        if self.use_colors:
            if delta_risk == 'high':
                delta_color = Fore.RED
            elif delta_risk == 'medium':
                delta_color = Fore.YELLOW
            else:
                delta_color = Fore.GREEN
        else:
            delta_color = ""
        
        if self.use_colors:
            print(f"   {Fore.WHITE}Δ Delta:{Style.RESET_ALL} {delta_color}{delta_mean:+.4f}{Style.RESET_ALL} "
                  f"(Risk: {delta_risk})")
        else:
            print(f"   Δ Delta: {delta_mean:+.4f} (Risk: {delta_risk})")
        
        # Gamma
        gamma_mean = gamma_summary.get('mean', 0)
        max_gamma = gamma_summary.get('max_gamma', 0)
        
        if self.use_colors:
            if max_gamma > 0.2:
                gamma_color = Fore.RED
            elif max_gamma > 0.1:
                gamma_color = Fore.YELLOW
            else:
                gamma_color = Fore.GREEN
        else:
            gamma_color = ""
        
        if self.use_colors:
            print(f"   {Fore.WHITE}Γ Gamma:{Style.RESET_ALL} {gamma_color}{gamma_mean:.6f}{Style.RESET_ALL} "
                  f"(Max: {max_gamma:.6f})")
        else:
            print(f"   Γ Gamma: {gamma_mean:.6f} (Max: {max_gamma:.6f})")
        
        # Theta
        total_theta = theta_summary.get('total_decay', 0)
        
        if self.use_colors:
            if total_theta > 100000:
                theta_color = Fore.RED
            elif total_theta > 50000:
                theta_color = Fore.YELLOW
            else:
                theta_color = Fore.GREEN
        else:
            theta_color = ""
        
        if self.use_colors:
            print(f"   {Fore.WHITE}Θ Theta:{Style.RESET_ALL} {theta_color}{total_theta:,.0f}{Style.RESET_ALL} "
                  f"(Daily decay: {total_theta/365:,.0f})")
        else:
            print(f"   Θ Theta: {total_theta:,.0f} (Daily decay: {total_theta/365:,.0f})")
        
        # Vega
        vega_summary = summary.get('vega', {})
        total_vega = vega_summary.get('total_vega', 0)
        
        if self.use_colors:
            if total_vega > 5000000:
                vega_color = Fore.RED
            elif total_vega > 1000000:
                vega_color = Fore.YELLOW
            else:
                vega_color = Fore.GREEN
        else:
            vega_color = ""
        
        if self.use_colors:
            print(f"   {Fore.WHITE}ν Vega:{Style.RESET_ALL} {vega_color}{total_vega:,.0f}{Style.RESET_ALL} "
                  f"(1% IV change impact)")
        else:
            print(f"   ν Vega: {total_vega:,.0f} (1% IV change impact)")
        
        print()
    
    async def _display_key_levels(self, report: Dict):
        """Display key support and resistance levels"""
        option_analysis = report.get('option_analysis', {})
        levels_analysis = option_analysis.get('levels_analysis', {})
        
        if not levels_analysis:
            return
        
        supports = levels_analysis.get('supports', [])
        resistances = levels_analysis.get('resistances', [])
        spot_price = report.get('market_data', {}).get('spot_price', 0)
        
        if self.use_colors:
            print(f"{Fore.CYAN}{self.SYMBOLS['level']} KEY SUPPORT/RESISTANCE LEVELS{Style.RESET_ALL}")
        else:
            print(f"{self.SYMBOLS['level']} KEY SUPPORT/RESISTANCE LEVELS")
        
        # Display supports
        if supports:
            if self.use_colors:
                print(f"   {Fore.GREEN}Supports:{Style.RESET_ALL}")
            else:
                print(f"   Supports:")
            
            for i, support in enumerate(supports[:3]):  # Top 3 supports
                level = support.get('level', 0)
                strength = support.get('strength', 0)
                oi = support.get('oi', 0)
                distance = spot_price - level
                percent = (distance / spot_price) * 100 if spot_price > 0 else 0
                
                # Create strength bar
                strength_bar = "█" * int(strength * 10)
                
                if self.use_colors:
                    print(f"     {level:,.0f} [{Fore.GREEN}{strength_bar:<10}{Style.RESET_ALL}] "
                          f"-{percent:.1f}% | OI: {oi:,}")
                else:
                    print(f"     {level:,.0f} [{strength_bar:<10}] -{percent:.1f}% | OI: {oi:,}")
        
        # Display resistances
        if resistances:
            if self.use_colors:
                print(f"   {Fore.RED}Resistances:{Style.RESET_ALL}")
            else:
                print(f"   Resistances:")
            
            for i, resistance in enumerate(resistances[:3]):  # Top 3 resistances
                level = resistance.get('level', 0)
                strength = resistance.get('strength', 0)
                oi = resistance.get('oi', 0)
                distance = level - spot_price
                percent = (distance / spot_price) * 100 if spot_price > 0 else 0
                
                # Create strength bar
                strength_bar = "█" * int(strength * 10)
                
                if self.use_colors:
                    print(f"     {level:,.0f} [{Fore.RED}{strength_bar:<10}{Style.RESET_ALL}] "
                          f"+{percent:.1f}% | OI: {oi:,}")
                else:
                    print(f"     {level:,.0f} [{strength_bar:<10}] +{percent:.1f}% | OI: {oi:,}")
        
        print()
    
    async def _display_signals(self, report: Dict):
        """Display trading signals"""
        signals = report.get('signals', [])
        
        if not signals:
            return
        
        if self.use_colors:
            print(f"{Fore.CYAN}{self.SYMBOLS['signal']} TRADING SIGNALS{Style.RESET_ALL}")
        else:
            print(f"{self.SYMBOLS['signal']} TRADING SIGNALS")
        
        for signal in signals[:3]:  # Top 3 signals
            signal_type = signal.get('type', 'neutral')
            strength = signal.get('strength', 'medium')
            description = signal.get('description', '')
            confidence = signal.get('confidence', 0)
            
            # Get signal symbol and color
            if signal_type == 'bullish':
                signal_symbol = self.SYMBOLS['bull']
                if self.use_colors:
                    signal_color = self.SIGNAL_COLORS['bullish']
            elif signal_type == 'bearish':
                signal_symbol = self.SYMBOLS['bear']
                if self.use_colors:
                    signal_color = self.SIGNAL_COLORS['bearish']
            elif signal_type == 'warning' or signal_type == 'caution':
                signal_symbol = self.SYMBOLS['warning']
                if self.use_colors:
                    signal_color = self.SIGNAL_COLORS['warning']
            else:
                signal_symbol = self.SYMBOLS['neutral']
                if self.use_colors:
                    signal_color = self.SIGNAL_COLORS['neutral']
            
            # Format strength
            strength_text = strength.upper()
            
            # Format confidence
            confidence_text = f"{confidence:.0%}"
            
            # Display signal
            if self.use_colors:
                print(f"   {signal_symbol} {signal_color}{signal_type.upper()}{Style.RESET_ALL} "
                      f"[{strength_text}] - {description}")
                print(f"      Confidence: {confidence_text}")
            else:
                print(f"   {signal_symbol} {signal_type.upper()} [{strength_text}] - {description}")
                print(f"      Confidence: {confidence_text}")
        
        print()
    
    async def _display_alerts(self, report: Dict):
        """Display alerts and warnings"""
        alerts = []
        
        # Collect alerts from different sources
        sentiment_analysis = report.get('sentiment_analysis', {})
        if sentiment_analysis:
            sentiment_alerts = sentiment_analysis.get('extreme_indicators', {})
            if sentiment_alerts.get('total_extremes', 0) > 0:
                alerts.append({
                    'type': 'warning',
                    'message': f"Extreme sentiment indicators detected: {sentiment_alerts['total_extremes']}"
                })
        
        option_analysis = report.get('option_analysis', {})
        if option_analysis:
            risk_analysis = option_analysis.get('risk_analysis', {})
            if risk_analysis.get('pin_risk', {}).get('overall_pin_risk') == 'high':
                alerts.append({
                    'type': 'warning',
                    'message': "High pin risk detected near current spot price"
                })
        
        # Add any explicit alerts
        if 'alerts' in report:
            alerts.extend(report['alerts'])
        
        if not alerts:
            return
        
        if self.use_colors:
            print(f"{Fore.CYAN}{self.SYMBOLS['alert']} ALERTS & WARNINGS{Style.RESET_ALL}")
        else:
            print(f"{self.SYMBOLS['alert']} ALERTS & WARNINGS")
        
        for alert in alerts[:5]:  # Limit to 5 alerts
            alert_type = alert.get('type', 'info')
            message = alert.get('message', '')
            
            # Get alert symbol and color
            if alert_type == 'warning' or alert_type == 'caution':
                alert_symbol = self.SYMBOLS['warning']
                if self.use_colors:
                    alert_color = Fore.RED + Style.BRIGHT
            elif alert_type == 'error':
                alert_symbol = self.SYMBOLS['error']
                if self.use_colors:
                    alert_color = Fore.RED
            elif alert_type == 'success':
                alert_symbol = self.SYMBOLS['success']
                if self.use_colors:
                    alert_color = Fore.GREEN
            else:
                alert_symbol = self.SYMBOLS['info']
                if self.use_colors:
                    alert_color = Fore.CYAN
            
            if self.use_colors:
                print(f"   {alert_symbol} {alert_color}{message}{Style.RESET_ALL}")
            else:
                print(f"   {alert_symbol} {message}")
        
        print()
    
    async def _display_footer(self, report: Dict):
        """Display report footer with metadata"""
        metadata = report.get('metadata', {})
        
        if not metadata:
            return
        
        # Get metadata values
        analysis_version = metadata.get('analysis_version', '1.0')
        processing_time = metadata.get('processing_time', None)
        data_sources = metadata.get('data_sources', [])
        
        # Format processing time
        if processing_time:
            if isinstance(processing_time, (int, float)):
                processing_str = f"{processing_time:.2f}s"
            else:
                processing_str = str(processing_time)
        else:
            processing_str = "N/A"
        
        # Get current time
        current_time = get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')
        
        # Display footer
        if self.use_colors:
            print(f"{Fore.CYAN}{self.SYMBOLS['clock']} REPORT METADATA{Style.RESET_ALL}")
            print(f"   Analysis Version: {Fore.YELLOW}{analysis_version}{Style.RESET_ALL}")
            print(f"   Processing Time: {Fore.YELLOW}{processing_str}{Style.RESET_ALL}")
            print(f"   Data Sources: {Fore.YELLOW}{len(data_sources)}{Style.RESET_ALL} sources")
            print(f"   Report Generated: {Fore.YELLOW}{current_time}{Style.RESET_ALL}")
            
            # Display quick status
            risk_assessment = report.get('risk_analysis', {}).get('overall_risk', 'medium')
            sentiment = report.get('sentiment_analysis', {}).get('overall_sentiment', {}).get('sentiment', 'neutral')
            
            # Color for risk
            if risk_assessment == 'high' or risk_assessment == 'very_high':
                risk_color = Fore.RED + Style.BRIGHT
            elif risk_assessment == 'medium':
                risk_color = Fore.YELLOW
            else:
                risk_color = Fore.GREEN
            
            # Color for sentiment
            sentiment_color = self.SENTIMENT_COLORS.get(sentiment, Fore.YELLOW)
            sentiment_text = sentiment.replace('_', ' ').title()
            
            print(f"\n{Fore.CYAN}QUICK STATUS{Style.RESET_ALL}")
            print(f"   Overall Risk: {risk_color}{risk_assessment.upper()}{Style.RESET_ALL}")
            print(f"   Market Sentiment: {sentiment_color}{sentiment_text}{Style.RESET_ALL}")
        else:
            print(f"{self.SYMBOLS['clock']} REPORT METADATA")
            print(f"   Analysis Version: {analysis_version}")
            print(f"   Processing Time: {processing_str}")
            print(f"   Data Sources: {len(data_sources)} sources")
            print(f"   Report Generated: {current_time}")
            
            # Display quick status
            risk_assessment = report.get('risk_analysis', {}).get('overall_risk', 'medium')
            sentiment = report.get('sentiment_analysis', {}).get('overall_sentiment', {}).get('sentiment', 'neutral')
            sentiment_text = sentiment.replace('_', ' ').title()
            
            print(f"\nQUICK STATUS")
            print(f"   Overall Risk: {risk_assessment.upper()}")
            print(f"   Market Sentiment: {sentiment_text}")

class JSONFormatter:
    """Formatter for JSON output"""
    
    def __init__(self, config):
        self.config = config
        self.output_config = config.get('output', {}).get('files', {}).get('json', {})
        self.pretty_print = self.output_config.get('pretty_print', True)
        self.include_raw = self.output_config.get('include_raw', False)
    
    async def format_report(self, report: Dict) -> str:
        """Format report as JSON string"""
        try:
            # Create a clean version of the report
            formatted_report = self._clean_report_for_json(report)
            
            # Convert to JSON
            if self.pretty_print:
                json_str = json.dumps(formatted_report, indent=2, default=self._json_serializer)
            else:
                json_str = json.dumps(formatted_report, default=self._json_serializer)
            
            return json_str
            
        except Exception as e:
            logger.error(f"Failed to format report as JSON: {e}")
            return json.dumps({'error': str(e)})
    
    async def save_report(self, report: Dict, filepath: str):
        """Save report as JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Format report
            json_str = await self.format_report(report)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            logger.info(f"Report saved as JSON: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
            raise
    
    def _clean_report_for_json(self, report: Dict) -> Dict:
        """Clean report for JSON serialization"""
        # Remove any non-serializable objects
        cleaned = {}
        
        for key, value in report.items():
            if isinstance(value, dict):
                cleaned[key] = self._clean_report_for_json(value)
            elif isinstance(value, list):
                cleaned[key] = self._clean_list_for_json(value)
            elif isinstance(value, (str, int, float, bool, type(None))):
                cleaned[key] = value
            else:
                # Try to convert to string
                cleaned[key] = str(value)
        
        return cleaned
    
    def _clean_list_for_json(self, lst: List) -> List:
        """Clean list for JSON serialization"""
        cleaned = []
        
        for item in lst:
            if isinstance(item, dict):
                cleaned.append(self._clean_report_for_json(item))
            elif isinstance(item, list):
                cleaned.append(self._clean_list_for_json(item))
            elif isinstance(item, (str, int, float, bool, type(None))):
                cleaned.append(item)
            else:
                cleaned.append(str(item))
        
        return cleaned
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

class CSVFormatter:
    """Formatter for CSV output"""
    
    def __init__(self, config):
        self.config = config
        self.output_config = config.get('output', {}).get('files', {}).get('csv', {})
        self.delimiter = self.output_config.get('delimiter', ',')
        self.include_timestamp = self.output_config.get('include_timestamp', True)
    
    async def format_report(self, report: Dict) -> List[List[str]]:
        """Format report as CSV rows"""
        try:
            rows = []
            
            # Add header
            header = await self._get_headers(report)
            rows.append(header)
            
            # Add data rows
            data_rows = await self._get_data_rows(report)
            rows.extend(data_rows)
            
            return rows
            
        except Exception as e:
            logger.error(f"Failed to format report as CSV: {e}")
            return []
    
    async def save_report(self, report: Dict, filepath: str):
        """Save report as CSV file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Get CSV rows
            rows = await self.format_report(report)
            
            if not rows:
                raise ValueError("No data to save")
            
            # Write to file
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=self.delimiter)
                writer.writerows(rows)
            
            logger.info(f"Report saved as CSV: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV report: {e}")
            raise
    
    async def _get_headers(self, report: Dict) -> List[str]:
        """Get CSV headers"""
        headers = []
        
        # Basic metadata
        if self.include_timestamp:
            headers.append('timestamp')
        
        headers.extend([
            'symbol',
            'spot_price',
            'change',
            'percent_change',
            'vix',
            'vix_change'
        ])
        
        # Sentiment
        headers.extend([
            'sentiment',
            'sentiment_score',
            'pcr_oi',
            'pcr_volume'
        ])
        
        # Option chain
        headers.extend([
            'total_call_oi',
            'total_put_oi',
            'total_oi',
            'max_pain_strike',
            'max_pain_distance'
        ])
        
        # Greeks
        headers.extend([
            'delta_mean',
            'gamma_mean',
            'theta_total',
            'vega_total',
            'delta_risk',
            'gamma_risk'
        ])
        
        # Risk
        headers.extend([
            'overall_risk',
            'pin_risk',
            'var_1d_95'
        ])
        
        # Support/Resistance
        headers.extend([
            'support_1',
            'support_2',
            'support_3',
            'resistance_1',
            'resistance_2',
            'resistance_3'
        ])
        
        # Signals
        headers.extend([
            'signal_count',
            'primary_signal',
            'primary_signal_confidence'
        ])
        
        return headers
    
    async def _get_data_rows(self, report: Dict) -> List[List[str]]:
        """Get CSV data rows"""
        rows = []
        
        # Extract data from report
        row_data = []
        
        # Basic metadata
        if self.include_timestamp:
            timestamp = report.get('timestamp', get_ist_now())
            if isinstance(timestamp, str):
                row_data.append(timestamp)
            else:
                row_data.append(timestamp.isoformat())
        
        symbol = report.get('symbol', 'NIFTY')
        row_data.append(symbol)
        
        # Market data
        market_data = report.get('market_data', {})
        spot_price = market_data.get('spot_price', 0)
        change = market_data.get('change', 0)
        percent_change = market_data.get('percent_change', 0)
        vix_data = market_data.get('vix', {})
        vix = vix_data.get('value', 0) if isinstance(vix_data, dict) else 0
        vix_change = vix_data.get('change', 0) if isinstance(vix_data, dict) else 0
        
        row_data.extend([
            f"{spot_price:.2f}",
            f"{change:.2f}",
            f"{percent_change:.2f}",
            f"{vix:.2f}",
            f"{vix_change:.2f}"
        ])
        
        # Sentiment
        sentiment_analysis = report.get('sentiment_analysis', {})
        overall_sentiment = sentiment_analysis.get('overall_sentiment', {})
        sentiment = overall_sentiment.get('sentiment', 'neutral')
        sentiment_score = overall_sentiment.get('score', 0)
        
        pcr_analysis = sentiment_analysis.get('pcr_analysis', {})
        pcr_oi = pcr_analysis.get('oi', 0)
        pcr_volume = pcr_analysis.get('volume', 0)
        
        row_data.extend([
            sentiment,
            f"{sentiment_score:.3f}",
            f"{pcr_oi:.3f}",
            f"{pcr_volume:.3f}"
        ])
        
        # Option chain
        option_analysis = report.get('option_analysis', {})
        oi_analysis = option_analysis.get('oi_analysis', {})
        totals = oi_analysis.get('totals', {})
        total_call_oi = totals.get('calls', 0)
        total_put_oi = totals.get('puts', 0)
        total_oi = totals.get('total', 0)
        
        max_pain = option_analysis.get('max_pain', {})
        max_pain_strike = max_pain.get('max_pain_strike', 0)
        max_pain_distance = max_pain.get('distance_from_spot', 0)
        
        row_data.extend([
            f"{total_call_oi:.0f}",
            f"{total_put_oi:.0f}",
            f"{total_oi:.0f}",
            f"{max_pain_strike:.0f}",
            f"{max_pain_distance:.2f}"
        ])
        
        # Greeks
        greeks_analysis = report.get('greeks_analysis', {})
        summary = greeks_analysis.get('summary', {})
        delta_summary = summary.get('delta', {})
        gamma_summary = summary.get('gamma', {})
        theta_summary = summary.get('theta', {})
        vega_summary = summary.get('vega', {})
        
        delta_mean = delta_summary.get('mean', 0)
        gamma_mean = gamma_summary.get('mean', 0)
        theta_total = theta_summary.get('total_decay', 0)
        vega_total = vega_summary.get('total_vega', 0)
        
        risk_assessment = summary.get('risk_assessment', {})
        delta_risk = risk_assessment.get('delta_risk', 'low')
        gamma_risk = risk_assessment.get('gamma_risk', 'low')
        
        row_data.extend([
            f"{delta_mean:.6f}",
            f"{gamma_mean:.6f}",
            f"{theta_total:.0f}",
            f"{vega_total:.0f}",
            delta_risk,
            gamma_risk
        ])
        
        # Risk
        risk_analysis = report.get('risk_analysis', {})
        overall_risk = risk_analysis.get('overall_risk', 'medium')
        pin_risk = risk_analysis.get('pin_risk', {}).get('overall_pin_risk', 'low')
        var = risk_analysis.get('value_at_risk', {}).get('var_1d_95', 0)
        
        row_data.extend([
            overall_risk,
            pin_risk,
            f"{var:.0f}"
        ])
        
        # Support/Resistance
        levels_analysis = option_analysis.get('levels_analysis', {})
        supports = levels_analysis.get('supports', [])
        resistances = levels_analysis.get('resistances', [])
        
        # Get top 3 supports
        support_1 = supports[0].get('level', 0) if len(supports) > 0 else 0
        support_2 = supports[1].get('level', 0) if len(supports) > 1 else 0
        support_3 = supports[2].get('level', 0) if len(supports) > 2 else 0
        
        # Get top 3 resistances
        resistance_1 = resistances[0].get('level', 0) if len(resistances) > 0 else 0
        resistance_2 = resistances[1].get('level', 0) if len(resistances) > 1 else 0
        resistance_3 = resistances[2].get('level', 0) if len(resistances) > 2 else 0
        
        row_data.extend([
            f"{support_1:.0f}",
            f"{support_2:.0f}",
            f"{support_3:.0f}",
            f"{resistance_1:.0f}",
            f"{resistance_2:.0f}",
            f"{resistance_3:.0f}"
        ])
        
        # Signals
        signals = report.get('signals', [])
        signal_count = len(signals)
        primary_signal = signals[0].get('type', 'none') if signals else 'none'
        primary_signal_confidence = signals[0].get('confidence', 0) if signals else 0
        
        row_data.extend([
            f"{signal_count}",
            primary_signal,
            f"{primary_signal_confidence:.2f}"
        ])
        
        rows.append(row_data)
        
        return rows

class ExcelFormatter:
    """Formatter for Excel output"""
    
    def __init__(self, config):
        self.config = config
        self.output_config = config.get('output', {}).get('files', {}).get('excel', {})
    
    async def save_report(self, report: Dict, filepath: str):
        """Save report as Excel file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create Excel writer
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Save summary sheet
                await self._save_summary_sheet(report, writer)
                
                # Save option chain sheet
                await self._save_option_chain_sheet(report, writer)
                
                # Save Greeks sheet
                await self._save_greeks_sheet(report, writer)
                
                # Save signals sheet
                await self._save_signals_sheet(report, writer)
            
            logger.info(f"Report saved as Excel: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save Excel report: {e}")
            raise
    
    async def _save_summary_sheet(self, report: Dict, writer: pd.ExcelWriter):
        """Save summary sheet"""
        try:
            summary_data = []
            
            # Basic info
            summary_data.append(['Symbol', report.get('symbol', 'NIFTY')])
            summary_data.append(['Timestamp', report.get('timestamp', get_ist_now())])
            
            # Market data
            market_data = report.get('market_data', {})
            summary_data.append(['Spot Price', market_data.get('spot_price', 0)])
            summary_data.append(['Change', market_data.get('change', 0)])
            summary_data.append(['% Change', market_data.get('percent_change', 0)])
            
            # VIX
            vix_data = market_data.get('vix', {})
            if isinstance(vix_data, dict):
                summary_data.append(['VIX', vix_data.get('value', 0)])
                summary_data.append(['VIX Change', vix_data.get('change', 0)])
            else:
                summary_data.append(['VIX', 0])
                summary_data.append(['VIX Change', 0])
            
            # Sentiment
            sentiment_analysis = report.get('sentiment_analysis', {})
            overall_sentiment = sentiment_analysis.get('overall_sentiment', {})
            summary_data.append(['Sentiment', overall_sentiment.get('sentiment', 'neutral')])
            summary_data.append(['Sentiment Score', overall_sentiment.get('score', 0)])
            summary_data.append(['Confidence', sentiment_analysis.get('confidence', 0)])
            
            # PCR
            pcr_analysis = sentiment_analysis.get('pcr_analysis', {})
            summary_data.append(['PCR (OI)', pcr_analysis.get('oi', 0)])
            summary_data.append(['PCR (Volume)', pcr_analysis.get('volume', 0)])
            
            # Risk
            risk_analysis = report.get('risk_analysis', {})
            summary_data.append(['Overall Risk', risk_analysis.get('overall_risk', 'medium')])
            
            # Create DataFrame
            df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            
            # Save to Excel
            df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Summary']
            for column in df:
                column_width = max(df[column].astype(str).map(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                worksheet.column_dimensions[chr(65 + col_idx)].width = column_width + 2
            
        except Exception as e:
            logger.error(f"Failed to save summary sheet: {e}")
    
    async def _save_option_chain_sheet(self, report: Dict, writer: pd.ExcelWriter):
        """Save option chain sheet"""
        try:
            option_analysis = report.get('option_analysis', {})
            
            if not option_analysis:
                return
            
            # Create DataFrames for different sections
            
            # OI Analysis
            oi_analysis = option_analysis.get('oi_analysis', {})
            totals = oi_analysis.get('totals', {})
            oi_data = {
                'Metric': ['Total Call OI', 'Total Put OI', 'Total OI', 'Call %', 'Put %'],
                'Value': [
                    totals.get('calls', 0),
                    totals.get('puts', 0),
                    totals.get('total', 0),
                    (totals.get('calls', 0) / totals.get('total', 1) * 100) if totals.get('total', 0) > 0 else 0,
                    (totals.get('puts', 0) / totals.get('total', 1) * 100) if totals.get('total', 0) > 0 else 0
                ]
            }
            df_oi = pd.DataFrame(oi_data)
            
            # Volume Analysis
            volume_analysis = option_analysis.get('volume_analysis', {})
            volume_totals = volume_analysis.get('totals', {})
            volume_data = {
                'Metric': ['Call Volume', 'Put Volume', 'Total Volume', 'Call %', 'Put %'],
                'Value': [
                    volume_totals.get('calls', 0),
                    volume_totals.get('puts', 0),
                    volume_totals.get('total', 0),
                    (volume_totals.get('calls', 0) / volume_totals.get('total', 1) * 100) if volume_totals.get('total', 0) > 0 else 0,
                    (volume_totals.get('puts', 0) / volume_totals.get('total', 1) * 100) if volume_totals.get('total', 0) > 0 else 0
                ]
            }
            df_volume = pd.DataFrame(volume_data)
            
            # Max Pain
            max_pain = option_analysis.get('max_pain', {})
            max_pain_data = {
                'Metric': ['Max Pain Strike', 'Max Pain Value', 'Distance from Spot', '% from Spot'],
                'Value': [
                    max_pain.get('max_pain_strike', 0),
                    max_pain.get('max_pain_value', 0),
                    max_pain.get('distance_from_spot', 0),
                    max_pain.get('percent_from_spot', 0)
                ]
            }
            df_max_pain = pd.DataFrame(max_pain_data)
            
            # Combine all DataFrames
            df_combined = pd.concat([df_oi, df_volume, df_max_pain], ignore_index=True)
            
            # Save to Excel
            df_combined.to_excel(writer, sheet_name='OptionChain', index=False)
            
        except Exception as e:
            logger.error(f"Failed to save option chain sheet: {e}")
    
    async def _save_greeks_sheet(self, report: Dict, writer: pd.ExcelWriter):
        """Save Greeks sheet"""
        try:
            greeks_analysis = report.get('greeks_analysis', {})
            
            if not greeks_analysis:
                return
            
            summary = greeks_analysis.get('summary', {})
            
            # Create DataFrames for different Greeks
            greeks_data = []
            
            # Delta
            delta_summary = summary.get('delta', {})
            greeks_data.append(['Delta Mean', delta_summary.get('mean', 0)])
            greeks_data.append(['Delta Std', delta_summary.get('std', 0)])
            greeks_data.append(['Delta Min', delta_summary.get('min', 0)])
            greeks_data.append(['Delta Max', delta_summary.get('max', 0)])
            
            # Gamma
            gamma_summary = summary.get('gamma', {})
            greeks_data.append(['Gamma Mean', gamma_summary.get('mean', 0)])
            greeks_data.append(['Gamma Std', gamma_summary.get('std', 0)])
            greeks_data.append(['Max Gamma', gamma_summary.get('max_gamma', 0)])
            
            # Theta
            theta_summary = summary.get('theta', {})
            greeks_data.append(['Total Theta', theta_summary.get('total_decay', 0)])
            greeks_data.append(['Mean Theta', theta_summary.get('mean_decay', 0)])
            
            # Vega
            vega_summary = summary.get('vega', {})
            greeks_data.append(['Total Vega', vega_summary.get('total_vega', 0)])
            greeks_data.append(['Mean Vega', vega_summary.get('mean_vega', 0)])
            
            # Risk Assessment
            risk_assessment = summary.get('risk_assessment', {})
            greeks_data.append(['Delta Risk', risk_assessment.get('delta_risk', 'low')])
            greeks_data.append(['Gamma Risk', risk_assessment.get('gamma_risk', 'low')])
            greeks_data.append(['Vega Risk', risk_assessment.get('vega_risk', 'low')])
            
            # Create DataFrame
            df = pd.DataFrame(greeks_data, columns=['Greek', 'Value'])
            
            # Save to Excel
            df.to_excel(writer, sheet_name='Greeks', index=False)
            
        except Exception as e:
            logger.error(f"Failed to save Greeks sheet: {e}")
    
    async def _save_signals_sheet(self, report: Dict, writer: pd.ExcelWriter):
        """Save signals sheet"""
        try:
            signals = report.get('signals', [])
            
            if not signals:
                return
            
            # Create signals data
            signals_data = []
            
            for i, signal in enumerate(signals):
                signals_data.append([
                    i + 1,
                    signal.get('type', ''),
                    signal.get('strength', ''),
                    signal.get('description', ''),
                    signal.get('confidence', 0),
                    signal.get('timestamp', '')
                ])
            
            # Create DataFrame
            df = pd.DataFrame(signals_data, columns=[
                'Signal #', 'Type', 'Strength', 'Description', 'Confidence', 'Timestamp'
            ])
            
            # Save to Excel
            df.to_excel(writer, sheet_name='Signals', index=False)
            
        except Exception as e:
            logger.error(f"Failed to save signals sheet: {e}")

class OutputManager:
    """Manager for all output formats"""
    
    def __init__(self, config):
        self.config = config
        self.console_formatter = ConsoleFormatter(config)
        self.json_formatter = JSONFormatter(config)
        self.csv_formatter = CSVFormatter(config)
        self.excel_formatter = ExcelFormatter(config)
        
    async def output_report(self, report: Dict):
        """Output report in all configured formats"""
        try:
            # Get output configuration
            output_config = self.config.get('output', {})
            
            # Console output
            if output_config.get('console', {}).get('enabled', False):
                await self.console_formatter.display_report(report)
            
            # File outputs
            files_config = output_config.get('files', {})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON output
            if files_config.get('json', {}).get('enabled', False):
                json_path = self.config['storage']['files']['output_path'] + f'/analysis_{timestamp}.json'
                await self.json_formatter.save_report(report, json_path)
            
            # CSV output
            if files_config.get('csv', {}).get('enabled', False):
                csv_path = self.config['storage']['files']['output_path'] + f'/analysis_{timestamp}.csv'
                await self.csv_formatter.save_report(report, csv_path)
            
            # Excel output (if enabled)
            if files_config.get('excel', {}).get('enabled', False):
                excel_path = self.config['storage']['files']['output_path'] + f'/analysis_{timestamp}.xlsx'
                await self.excel_formatter.save_report(report, excel_path)
            
            logger.info(f"Report output completed: {timestamp}")
            
        except Exception as e:
            logger.error(f"Failed to output report: {e}")

# Utility functions for formatting specific data types

def format_price(price: float, precision: int = 2) -> str:
    """Format price with commas and precision"""
    if price >= 1000:
        return f"{price:,.{precision}f}"
    else:
        return f"{price:.{precision}f}"

def format_percentage(value: float, precision: int = 2) -> str:
    """Format percentage value"""
    return f"{value:.{precision}f}%"

def format_large_number(number: float) -> str:
    """Format large number with K, M, B suffixes"""
    if abs(number) >= 1_000_000_000:
        return f"{number/1_000_000_000:.1f}B"
    elif abs(number) >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif abs(number) >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return f"{number:.0f}"

def format_risk_level(risk: str) -> str:
    """Format risk level with emoji"""
    risk_emojis = {
        'very_high': '🔥',
        'high': '⚠️',
        'medium': '⚡',
        'low': '✅',
        'very_low': '🟢'
    }
    return f"{risk_emojis.get(risk, '⚪')} {risk.replace('_', ' ').title()}"

def format_sentiment(sentiment: str) -> str:
    """Format sentiment with emoji"""
    sentiment_emojis = {
        'very_bullish': '🐂🔥',
        'bullish': '🐂',
        'neutral': '➖',
        'bearish': '🐻',
        'very_bearish': '🐻🔥'
    }
    return f"{sentiment_emojis.get(sentiment, '⚪')} {sentiment.replace('_', ' ').title()}"

def format_timestamp(timestamp, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Format timestamp"""
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime(format_str)
        except:
            return timestamp
    elif isinstance(timestamp, datetime):
        return timestamp.strftime(format_str)
    else:
        return str(timestamp)

def format_greek_value(value: float, greek: str) -> str:
    """Format Greek value with appropriate precision"""
    precisions = {
        'delta': 4,
        'gamma': 6,
        'theta': 6,
        'vega': 6,
        'rho': 6
    }
    precision = precisions.get(greek, 4)
    return f"{value:.{precision}f}"

def format_option_strike(strike: float) -> str:
    """Format option strike price"""
    if strike >= 1000:
        return f"{strike:,.0f}"
    else:
        return f"{strike:.0f}"

def format_oi_change(change: float) -> str:
    """Format OI change with sign"""
    if change > 0:
        return f"+{format_large_number(change)}"
    elif change < 0:
        return f"-{format_large_number(abs(change))}"
    else:
        return "0"

def format_iv(iv: float) -> str:
    """Format implied volatility"""
    return f"{iv:.1f}%"

def create_progress_bar(value: float, max_value: float = 100, width: int = 20) -> str:
    """Create a progress bar"""
    if max_value == 0:
        return "[" + " " * width + "]"
    
    filled = int((value / max_value) * width)
    return "[" + "█" * filled + " " * (width - filled) + "]"

def create_strength_bar(strength: float, width: int = 10) -> str:
    """Create a strength bar"""
    filled = int(strength * width)
    return "[" + "█" * filled + " " * (width - filled) + "]"

# Example usage
if __name__ == "__main__":
    # Test formatters
    import yaml
    
    # Load sample config
    with open("configs/defaults.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample report
    sample_report = {
        'timestamp': datetime.now().isoformat(),
        'symbol': 'NIFTY',
        'market_data': {
            'spot_price': 22000.50,
            'change': 150.25,
            'percent_change': 0.68,
            'vix': {
                'value': 15.75,
                'change': -0.25
            }
        },
        'sentiment_analysis': {
            'overall_sentiment': {
                'sentiment': 'bullish',
                'score': 0.65,
                'strength': 'moderate'
            },
            'pcr_analysis': {
                'oi': 0.85,
                'volume': 0.92,
                'interpretation': 'Bullish'
            },
            'confidence': 0.75
        },
        'option_analysis': {
            'oi_analysis': {
                'totals': {
                    'calls': 12500000,
                    'puts': 10500000,
                    'total': 23000000
                }
            },
            'max_pain': {
                'max_pain_strike': 21950,
                'distance_from_spot': 50.5,
                'percent_from_spot': 0.23
            }
        },
        'greeks_analysis': {
            'summary': {
                'delta': {'mean': 0.48, 'std': 0.12},
                'gamma': {'mean': 0.025, 'max_gamma': 0.085},
                'theta': {'total_decay': 85000},
                'vega': {'total_vega': 1200000},
                'risk_assessment': {
                    'delta_risk': 'low',
                    'gamma_risk': 'medium',
                    'vega_risk': 'low'
                }
            }
        },
        'risk_analysis': {
            'overall_risk': 'medium',
            'pin_risk': {'overall_pin_risk': 'low'},
            'value_at_risk': {'var_1d_95': 45000}
        },
        'signals': [
            {
                'type': 'bullish',
                'strength': 'moderate',
                'description': 'Positive OI buildup in calls',
                'confidence': 0.65,
                'timestamp': datetime.now().isoformat()
            }
        ],
        'metadata': {
            'analysis_version': '1.0',
            'processing_time': 2.5,
            'data_sources': ['option_chain', 'market_stats']
        }
    }
    
    # Test console formatter
    formatter = ConsoleFormatter(config)
    
    # Test formatting utilities
    print("Formatting utilities test:")
    print(f"Price: {format_price(22000.50)}")
    print(f"Percentage: {format_percentage(0.68)}")
    print(f"Large number: {format_large_number(12500000)}")
    print(f"Risk level: {format_risk_level('medium')}")
    print(f"Sentiment: {format_sentiment('bullish')}")
    print(f"Greek value: {format_greek_value(0.025, 'gamma')}")
    print(f"Progress bar: {create_progress_bar(65, 100)}")
    print(f"Strength bar: {create_strength_bar(0.75)}")