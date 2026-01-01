"""
Greeks Calculator for Options
Calculates Delta, Gamma, Theta, Vega, Rho using Black-Scholes and other models
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import asyncio

from src.models.black_scholes import BlackScholesModel
from src.utils.logger import get_logger
from src.utils.date_utils import days_to_expiry, trading_days_to_expiry

logger = get_logger(__name__)

class GreeksCalculator:
    """Calculator for option Greeks"""
    
    def __init__(self, config):
        self.config = config
        self.model = BlackScholesModel()
        
        # Greeks calculation parameters
        self.greeks_config = config['analysis']['greeks']
        self.risk_free_rate = self.greeks_config['risk_free_rate']
        self.dividend_yield = self.greeks_config['dividend_yield']
        self.days_in_year = self.greeks_config['days_in_year']
        
        # Precision
        self.delta_precision = self.greeks_config['delta_precision']
        self.gamma_precision = self.greeks_config['gamma_precision']
        self.theta_precision = self.greeks_config['theta_precision']
        self.vega_precision = self.greeks_config['vega_precision']
        self.rho_precision = self.greeks_config['rho_precision']
    
    async def calculate_all(self, option_chain: List[Dict], spot_price: float, 
                          risk_free_rate: Optional[float] = None,
                          dividend_yield: Optional[float] = None) -> Dict:
        """
        Calculate all Greeks for entire option chain
        
        Args:
            option_chain: List of option contracts
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate (optional)
            dividend_yield: Dividend yield (optional)
            
        Returns:
            Dictionary with Greeks for each strike
        """
        try:
            logger.info(f"Calculating Greeks for {len(option_chain)} options...")
            
            # Use provided rates or defaults
            r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            q = dividend_yield if dividend_yield is not None else self.dividend_yield
            
            greeks_by_strike = {}
            
            # Process each option
            for option in option_chain:
                strike = option.get('strikePrice')
                if not strike:
                    continue
                
                # Calculate for calls
                if 'CE' in option:
                    call_greeks = await self.calculate_option_greeks(
                        spot_price=spot_price,
                        strike_price=strike,
                        time_to_expiry=self._get_time_to_expiry(option['CE']),
                        iv=option['CE'].get('impliedVolatility', 0) / 100,  # Convert % to decimal
                        option_type='call',
                        risk_free_rate=r,
                        dividend_yield=q
                    )
                    
                    if strike not in greeks_by_strike:
                        greeks_by_strike[strike] = {}
                    greeks_by_strike[strike]['CE'] = call_greeks
                
                # Calculate for puts
                if 'PE' in option:
                    put_greeks = await self.calculate_option_greeks(
                        spot_price=spot_price,
                        strike_price=strike,
                        time_to_expiry=self._get_time_to_expiry(option['PE']),
                        iv=option['PE'].get('impliedVolatility', 0) / 100,
                        option_type='put',
                        risk_free_rate=r,
                        dividend_yield=q
                    )
                    
                    if strike not in greeks_by_strike:
                        greeks_by_strike[strike] = {}
                    greeks_by_strike[strike]['PE'] = put_greeks
            
            # Calculate Greeks surface and summary
            greeks_surface = await self.calculate_greeks_surface(greeks_by_strike, spot_price)
            greeks_summary = await self.calculate_greeks_summary(greeks_by_strike)
            
            result = {
                'by_strike': greeks_by_strike,
                'surface': greeks_surface,
                'summary': greeks_summary,
                'metadata': {
                    'spot_price': spot_price,
                    'risk_free_rate': r,
                    'dividend_yield': q,
                    'calculation_time': datetime.now().isoformat()
                }
            }
            
            logger.info("Greeks calculation completed")
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate Greeks: {e}")
            return {}
    
    async def calculate_option_greeks(self, spot_price: float, strike_price: float, 
                                    time_to_expiry: float, iv: float, 
                                    option_type: str, risk_free_rate: float,
                                    dividend_yield: float) -> Dict:
        """
        Calculate Greeks for a single option
        
        Args:
            spot_price: Current spot price
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            iv: Implied volatility (decimal)
            option_type: 'call' or 'put'
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            
        Returns:
            Dictionary with all Greeks
        """
        try:
            # Calculate using Black-Scholes
            greeks = self.model.calculate_greeks(
                S=spot_price,
                K=strike_price,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=iv,
                q=dividend_yield,
                option_type=option_type
            )
            
            # Add additional metrics
            greeks.update({
                'moneyness': self._calculate_moneyness(spot_price, strike_price, option_type),
                'breakeven': self._calculate_breakeven(spot_price, strike_price, option_type),
                'probability_itm': self._calculate_probability_itm(spot_price, strike_price, time_to_expiry, iv, option_type),
                'probability_otm': 1 - self._calculate_probability_itm(spot_price, strike_price, time_to_expiry, iv, option_type),
                'expected_move': self._calculate_expected_move(spot_price, iv, time_to_expiry),
                'leverage': self._calculate_leverage(spot_price, greeks.get('delta', 0))
            })
            
            # Round values for readability
            greeks = self._round_greeks(greeks)
            
            return greeks
            
        except Exception as e:
            logger.error(f"Failed to calculate option Greeks: {e}")
            return {}
    
    async def calculate_greeks_surface(self, greeks_by_strike: Dict, spot_price: float) -> Dict:
        """
        Calculate Greeks surface metrics
        
        Args:
            greeks_by_strike: Greeks by strike
            spot_price: Current spot price
            
        Returns:
            Greeks surface analysis
        """
        try:
            strikes = sorted(greeks_by_strike.keys())
            if not strikes:
                return {}
            
            # Extract deltas for calls and puts
            call_deltas = []
            put_deltas = []
            call_gammas = []
            put_gammas = []
            call_vegas = []
            put_vegas = []
            
            for strike in strikes:
                if 'CE' in greeks_by_strike[strike]:
                    call_greeks = greeks_by_strike[strike]['CE']
                    call_deltas.append(call_greeks.get('delta', 0))
                    call_gammas.append(call_greeks.get('gamma', 0))
                    call_vegas.append(call_greeks.get('vega', 0))
                
                if 'PE' in greeks_by_strike[strike]:
                    put_greeks = greeks_by_strike[strike]['PE']
                    put_deltas.append(put_greeks.get('delta', 0))
                    put_gammas.append(put_greeks.get('gamma', 0))
                    put_vegas.append(put_greeks.get('vega', 0))
            
            # Calculate surface metrics
            surface = {
                'delta_surface': {
                    'call_delta_skew': np.mean(call_deltas) if call_deltas else 0,
                    'put_delta_skew': np.mean(put_deltas) if put_deltas else 0,
                    'atm_delta': self._find_atm_delta(greeks_by_strike, spot_price),
                    'delta_exposure': self._calculate_delta_exposure(greeks_by_strike)
                },
                
                'gamma_surface': {
                    'max_gamma_strike': self._find_max_gamma_strike(greeks_by_strike),
                    'gamma_exposure': self._calculate_gamma_exposure(greeks_by_strike),
                    'gamma_risk': self._calculate_gamma_risk(greeks_by_strike, spot_price)
                },
                
                'vega_surface': {
                    'vega_exposure': self._calculate_vega_exposure(greeks_by_strike),
                    'vega_skew': self._calculate_vega_skew(greeks_by_strike),
                    'volatility_smile': self._analyze_volatility_smile(greeks_by_strike)
                },
                
                'theta_surface': {
                    'theta_decay': self._calculate_theta_decay(greeks_by_strike),
                    'time_decay_profile': self._analyze_time_decay(greeks_by_strike)
                }
            }
            
            return surface
            
        except Exception as e:
            logger.error(f"Failed to calculate Greeks surface: {e}")
            return {}
    
    async def calculate_greeks_summary(self, greeks_by_strike: Dict) -> Dict:
        """
        Calculate summary statistics for Greeks
        
        Args:
            greeks_by_strike: Greeks by strike
            
        Returns:
            Summary statistics
        """
        try:
            # Aggregate Greeks values
            all_deltas = []
            all_gammas = []
            all_thetas = []
            all_vegas = []
            all_rhos = []
            
            for strike, greeks_dict in greeks_by_strike.items():
                for option_type, greeks in greeks_dict.items():
                    all_deltas.append(greeks.get('delta', 0))
                    all_gammas.append(greeks.get('gamma', 0))
                    all_thetas.append(greeks.get('theta', 0))
                    all_vegas.append(greeks.get('vega', 0))
                    all_rhos.append(greeks.get('rho', 0))
            
            summary = {
                'delta': {
                    'mean': np.mean(all_deltas) if all_deltas else 0,
                    'std': np.std(all_deltas) if all_deltas else 0,
                    'min': min(all_deltas) if all_deltas else 0,
                    'max': max(all_deltas) if all_deltas else 0
                },
                
                'gamma': {
                    'mean': np.mean(all_gammas) if all_gammas else 0,
                    'std': np.std(all_gammas) if all_gammas else 0,
                    'max_gamma': max(all_gammas) if all_gammas else 0
                },
                
                'theta': {
                    'total_decay': sum(all_thetas) if all_thetas else 0,
                    'mean_decay': np.mean(all_thetas) if all_thetas else 0
                },
                
                'vega': {
                    'total_vega': sum(all_vegas) if all_vegas else 0,
                    'mean_vega': np.mean(all_vegas) if all_vegas else 0
                },
                
                'risk_assessment': {
                    'delta_risk': self._assess_delta_risk(all_deltas),
                    'gamma_risk': self._assess_gamma_risk(all_gammas),
                    'vega_risk': self._assess_vega_risk(all_vegas)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to calculate Greeks summary: {e}")
            return {}
    
    # Helper methods
    
    def _get_time_to_expiry(self, option_data: Dict) -> float:
        """Calculate time to expiry in years"""
        try:
            expiry_date_str = option_data.get('expiryDate')
            if not expiry_date_str:
                # If no expiry date, use default
                return 30 / self.days_in_year  # 30 days
            
            # Parse expiry date
            from datetime import datetime
            expiry_date = datetime.strptime(expiry_date_str, '%d-%b-%Y')
            current_date = datetime.now()
            
            # Calculate days to expiry
            days_to_expiry = (expiry_date - current_date).days
            
            # Use trading days for more accuracy
            trading_days = trading_days_to_expiry(current_date, expiry_date)
            
            # Convert to years
            time_to_expiry = max(trading_days / 252, 1/252)  # At least 1 trading day
            
            return time_to_expiry
            
        except Exception:
            # Default to 30 days if calculation fails
            return 30 / self.days_in_year
    
    def _calculate_moneyness(self, spot: float, strike: float, option_type: str) -> str:
        """Calculate moneyness of option"""
        if option_type == 'call':
            if strike < spot * 0.95:
                return 'deep_itm'
            elif strike < spot * 0.98:
                return 'itm'
            elif abs(strike - spot) / spot < 0.02:
                return 'atm'
            elif strike > spot * 1.02:
                return 'otm'
            else:
                return 'deep_otm'
        else:  # put
            if strike > spot * 1.05:
                return 'deep_itm'
            elif strike > spot * 1.02:
                return 'itm'
            elif abs(strike - spot) / spot < 0.02:
                return 'atm'
            elif strike < spot * 0.98:
                return 'otm'
            else:
                return 'deep_otm'
    
    def _calculate_breakeven(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate breakeven price"""
        # Simplified - actual breakeven depends on premium
        if option_type == 'call':
            return strike  # For zero premium
        else:
            return strike  # For zero premium
    
    def _calculate_probability_itm(self, spot: float, strike: float, 
                                 time_to_expiry: float, iv: float, 
                                 option_type: str) -> float:
        """Calculate probability of finishing in the money"""
        try:
            d2 = self.model.calculate_d2(spot, strike, time_to_expiry, 
                                        self.risk_free_rate, iv, self.dividend_yield)
            
            if option_type == 'call':
                probability = norm.cdf(d2)
            else:  # put
                probability = norm.cdf(-d2)
            
            return probability
            
        except Exception:
            return 0.5  # Default probability
    
    def _calculate_expected_move(self, spot: float, iv: float, time_to_expiry: float) -> Dict:
        """Calculate expected move based on IV"""
        try:
            # Expected move = spot * IV * sqrt(time_to_expiry)
            expected_move = spot * iv * math.sqrt(time_to_expiry)
            
            return {
                'absolute': expected_move,
                'percent': (expected_move / spot) * 100,
                'one_sd': {
                    'lower': spot - expected_move,
                    'upper': spot + expected_move
                }
            }
            
        except Exception:
            return {'absolute': 0, 'percent': 0, 'one_sd': {'lower': spot, 'upper': spot}}
    
    def _calculate_leverage(self, spot: float, delta: float) -> float:
        """Calculate leverage (Delta * Spot / Premium) - simplified"""
        try:
            # Simplified leverage calculation
            if abs(delta) > 0:
                leverage = delta * spot / (spot * 0.05)  # Assuming 5% premium
                return leverage
            return 0
        except Exception:
            return 0
    
    def _round_greeks(self, greeks: Dict) -> Dict:
        """Round Greeks values for readability"""
        rounded = {}
        for key, value in greeks.items():
            if key == 'delta':
                rounded[key] = round(value, self.delta_precision)
            elif key == 'gamma':
                rounded[key] = round(value, self.gamma_precision)
            elif key == 'theta':
                rounded[key] = round(value, self.theta_precision)
            elif key == 'vega':
                rounded[key] = round(value, self.vega_precision)
            elif key == 'rho':
                rounded[key] = round(value, self.rho_precision)
            else:
                rounded[key] = value
        return rounded
    
    # Surface calculation helper methods
    
    def _find_atm_delta(self, greeks_by_strike: Dict, spot_price: float) -> float:
        """Find delta for ATM strike"""
        try:
            # Find strike closest to spot
            closest_strike = min(greeks_by_strike.keys(), 
                               key=lambda x: abs(x - spot_price))
            
            if 'CE' in greeks_by_strike[closest_strike]:
                return greeks_by_strike[closest_strike]['CE'].get('delta', 0.5)
            
            return 0.5  # Default ATM delta
            
        except Exception:
            return 0.5
    
    def _calculate_delta_exposure(self, greeks_by_strike: Dict) -> float:
        """Calculate total delta exposure"""
        total_delta = 0
        for strike, greeks_dict in greeks_by_strike.items():
            for option_type, greeks in greeks_dict.items():
                delta = greeks.get('delta', 0)
                # Calls have positive delta, Puts have negative delta
                if option_type == 'CE':
                    total_delta += delta
                else:
                    total_delta -= abs(delta)  # Put delta is negative
        return total_delta
    
    def _find_max_gamma_strike(self, greeks_by_strike: Dict) -> Dict:
        """Find strike with maximum gamma"""
        max_gamma = 0
        max_strike = 0
        max_option_type = ''
        
        for strike, greeks_dict in greeks_by_strike.items():
            for option_type, greeks in greeks_dict.items():
                gamma = greeks.get('gamma', 0)
                if gamma > max_gamma:
                    max_gamma = gamma
                    max_strike = strike
                    max_option_type = option_type
        
        return {
            'strike': max_strike,
            'gamma': max_gamma,
            'option_type': max_option_type
        }
    
    def _calculate_gamma_exposure(self, greeks_by_strike: Dict) -> float:
        """Calculate total gamma exposure"""
        total_gamma = 0
        for strike, greeks_dict in greeks_by_strike.items():
            for option_type, greeks in greeks_dict.items():
                total_gamma += abs(greeks.get('gamma', 0))
        return total_gamma
    
    def _calculate_gamma_risk(self, greeks_by_strike: Dict, spot_price: float) -> Dict:
        """Calculate gamma risk profile"""
        # Gamma is highest near ATM
        atm_gamma = self._find_atm_delta(greeks_by_strike, spot_price)
        
        return {
            'atm_gamma_concentration': atm_gamma,
            'gamma_skew': self._calculate_gamma_skew(greeks_by_strike, spot_price),
            'risk_level': 'high' if atm_gamma > 0.1 else 'medium' if atm_gamma > 0.05 else 'low'
        }
    
    def _calculate_gamma_skew(self, greeks_by_strike: Dict, spot_price: float) -> float:
        """Calculate gamma skew"""
        # Compare gamma of OTM calls vs OTM puts
        otm_call_gamma = 0
        otm_put_gamma = 0
        
        for strike, greeks_dict in greeks_by_strike.items():
            if strike > spot_price * 1.02:  # OTM calls
                if 'CE' in greeks_dict:
                    otm_call_gamma = max(otm_call_gamma, greeks_dict['CE'].get('gamma', 0))
            elif strike < spot_price * 0.98:  # OTM puts
                if 'PE' in greeks_dict:
                    otm_put_gamma = max(otm_put_gamma, greeks_dict['PE'].get('gamma', 0))
        
        if otm_put_gamma > 0:
            return otm_call_gamma / otm_put_gamma
        return 1.0
    
    def _calculate_vega_exposure(self, greeks_by_strike: Dict) -> float:
        """Calculate total vega exposure"""
        total_vega = 0
        for strike, greeks_dict in greeks_by_strike.items():
            for option_type, greeks in greeks_dict.items():
                total_vega += abs(greeks.get('vega', 0))
        return total_vega
    
    def _calculate_vega_skew(self, greeks_by_strike: Dict) -> Dict:
        """Calculate vega skew"""
        # Vega is typically highest for ATM options
        vegas = []
        for strike, greeks_dict in greeks_by_strike.items():
            for option_type, greeks in greeks_dict.items():
                vegas.append(greeks.get('vega', 0))
        
        if vegas:
            return {
                'mean': np.mean(vegas),
                'std': np.std(vegas),
                'skewness': float(np.mean((np.array(vegas) - np.mean(vegas))**3) / (np.std(vegas)**3)) if np.std(vegas) > 0 else 0
            }
        
        return {'mean': 0, 'std': 0, 'skewness': 0}
    
    def _analyze_volatility_smile(self, greeks_by_strike: Dict) -> Dict:
        """Analyze volatility smile/skew"""
        # This would typically use IV data
        # For now, return placeholder
        return {
            'smile_present': True,
            'skew_direction': 'positive',  # or 'negative', 'flat'
            'smile_strength': 'moderate'
        }
    
    def _calculate_theta_decay(self, greeks_by_strike: Dict) -> float:
        """Calculate total theta decay"""
        total_theta = 0
        for strike, greeks_dict in greeks_by_strike.items():
            for option_type, greeks in greeks_dict.items():
                # Theta is negative (decay), take absolute value
                total_theta += abs(greeks.get('theta', 0))
        return total_theta
    
    def _analyze_time_decay(self, greeks_by_strike: Dict) -> Dict:
        """Analyze time decay profile"""
        thetas = []
        for strike, greeks_dict in greeks_by_strike.items():
            for option_type, greeks in greeks_dict.items():
                thetas.append(abs(greeks.get('theta', 0)))
        
        if thetas:
            return {
                'max_decay': max(thetas) if thetas else 0,
                'avg_decay': np.mean(thetas) if thetas else 0,
                'decay_distribution': 'exponential'  # or 'linear', 'accelerated'
            }
        
        return {'max_decay': 0, 'avg_decay': 0, 'decay_distribution': 'unknown'}
    
    # Risk assessment methods
    
    def _assess_delta_risk(self, deltas: List[float]) -> str:
        """Assess delta risk level"""
        if not deltas:
            return 'low'
        
        total_delta = sum(deltas)
        if abs(total_delta) > 1000:
            return 'very_high'
        elif abs(total_delta) > 500:
            return 'high'
        elif abs(total_delta) > 100:
            return 'medium'
        else:
            return 'low'
    
    def _assess_gamma_risk(self, gammas: List[float]) -> str:
        """Assess gamma risk level"""
        if not gammas:
            return 'low'
        
        max_gamma = max(gammas)
        if max_gamma > 0.2:
            return 'very_high'
        elif max_gamma > 0.1:
            return 'high'
        elif max_gamma > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def _assess_vega_risk(self, vegas: List[float]) -> str:
        """Assess vega risk level"""
        if not vegas:
            return 'low'
        
        total_vega = sum(vegas)
        if total_vega > 5000:
            return 'very_high'
        elif total_vega > 2000:
            return 'high'
        elif total_vega > 500:
            return 'medium'
        else:
            return 'low'