"""
Black-Scholes Option Pricing Model
Includes Greeks calculation
"""

import math
from scipy.stats import norm
import numpy as np
from typing import Dict, Tuple

class BlackScholesModel:
    """Black-Scholes option pricing model implementation"""
    
    def calculate_price(self, S: float, K: float, T: float, r: float, 
                       sigma: float, q: float = 0, option_type: str = 'call') -> float:
        """
        Calculate option price using Black-Scholes
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility (decimal)
            q: Dividend yield (decimal)
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        d1 = self.calculate_d1(S, K, T, r, sigma, q)
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        
        return price
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float,
                        sigma: float, q: float = 0, option_type: str = 'call') -> Dict:
        """
        Calculate all Greeks
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility (decimal)
            q: Dividend yield (decimal)
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with all Greeks
        """
        d1 = self.calculate_d1(S, K, T, r, sigma, q)
        d2 = d1 - sigma * math.sqrt(T)
        
        # Common calculations
        sqrt_T = math.sqrt(T)
        exp_neg_qT = math.exp(-q * T)
        exp_neg_rT = math.exp(-r * T)
        nd1 = norm.pdf(d1)
        
        greeks = {}
        
        # Delta
        if option_type == 'call':
            greeks['delta'] = exp_neg_qT * norm.cdf(d1)
        else:  # put
            greeks['delta'] = exp_neg_qT * (norm.cdf(d1) - 1)
        
        # Gamma (same for calls and puts)
        greeks['gamma'] = (exp_neg_qT * nd1) / (S * sigma * sqrt_T)
        
        # Theta
        if option_type == 'call':
            term1 = - (S * exp_neg_qT * nd1 * sigma) / (2 * sqrt_T)
            term2 = r * K * exp_neg_rT * norm.cdf(d2)
            term3 = q * S * exp_neg_qT * norm.cdf(d1)
            greeks['theta'] = (term1 - term2 + term3) / 365  # Per day
        else:  # put
            term1 = - (S * exp_neg_qT * nd1 * sigma) / (2 * sqrt_T)
            term2 = r * K * exp_neg_rT * norm.cdf(-d2)
            term3 = q * S * exp_neg_qT * norm.cdf(-d1)
            greeks['theta'] = (term1 + term2 - term3) / 365  # Per day
        
        # Vega (same for calls and puts)
        greeks['vega'] = (S * exp_neg_qT * nd1 * sqrt_T) / 100  # For 1% change in vol
        
        # Rho
        if option_type == 'call':
            greeks['rho'] = (K * T * exp_neg_rT * norm.cdf(d2)) / 100  # For 1% change in rate
        else:  # put
            greeks['rho'] = (-K * T * exp_neg_rT * norm.cdf(-d2)) / 100
        
        return greeks
    
    def calculate_d1(self, S: float, K: float, T: float, r: float, 
                    sigma: float, q: float = 0) -> float:
        """
        Calculate d1 parameter for Black-Scholes
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility (decimal)
            q: Dividend yield (decimal)
            
        Returns:
            d1 parameter
        """
        if T <= 0 or sigma <= 0:
            return 0
        
        numerator = math.log(S / K) + (r - q + 0.5 * sigma**2) * T
        denominator = sigma * math.sqrt(T)
        
        return numerator / denominator
    
    def calculate_d2(self, S: float, K: float, T: float, r: float,
                    sigma: float, q: float = 0) -> float:
        """
        Calculate d2 parameter for Black-Scholes
        
        Args:
            Same as calculate_d1
            
        Returns:
            d2 parameter
        """
        d1 = self.calculate_d1(S, K, T, r, sigma, q)
        return d1 - sigma * math.sqrt(T)
    
    def calculate_implied_volatility(self, price: float, S: float, K: float, 
                                    T: float, r: float, q: float = 0, 
                                    option_type: str = 'call', 
                                    max_iterations: int = 100, 
                                    tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            price: Market price of option
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            q: Dividend yield (decimal)
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility (decimal)
        """
        # Initial guess
        sigma = 0.3  # Start with 30% volatility
        
        for i in range(max_iterations):
            # Calculate price with current sigma
            calculated_price = self.calculate_price(S, K, T, r, sigma, q, option_type)
            
            # Calculate vega
            d1 = self.calculate_d1(S, K, T, r, sigma, q)
            sqrt_T = math.sqrt(T)
            vega = S * math.exp(-q * T) * norm.pdf(d1) * sqrt_T
            
            # Check if vega is too small
            if vega < tolerance:
                break
            
            # Calculate error
            error = calculated_price - price
            
            # Update sigma using Newton-Raphson
            sigma -= error / vega
            
            # Check convergence
            if abs(error) < tolerance:
                break
        
        return max(0.01, min(2.0, sigma))  # Bound between 1% and 200%
    
    def calculate_probability_itm(self, S: float, K: float, T: float, 
                                 r: float, sigma: float, q: float = 0, 
                                 option_type: str = 'call') -> float:
        """
        Calculate probability of finishing in the money
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility (decimal)
            q: Dividend yield (decimal)
            option_type: 'call' or 'put'
            
        Returns:
            Probability (0 to 1)
        """
        d2 = self.calculate_d2(S, K, T, r, sigma, q)
        
        if option_type == 'call':
            return norm.cdf(d2)
        else:  # put
            return norm.cdf(-d2)
    
    def calculate_expected_move(self, S: float, sigma: float, T: float, 
                               confidence: float = 0.68) -> Dict:
        """
        Calculate expected move based on volatility
        
        Args:
            S: Spot price
            sigma: Volatility (decimal)
            T: Time to expiry (years)
            confidence: Confidence level (0.68 for 1SD, 0.95 for 2SD)
            
        Returns:
            Dictionary with expected move information
        """
        # Convert confidence to z-score
        if confidence == 0.68:
            z = 1.0
        elif confidence == 0.95:
            z = 2.0
        elif confidence == 0.997:
            z = 3.0
        else:
            # Calculate z-score from confidence
            z = norm.ppf((1 + confidence) / 2)
        
        # Expected move = spot * vol * sqrt(time) * z
        expected_move = S * sigma * math.sqrt(T) * z
        
        return {
            'confidence': confidence,
            'z_score': z,
            'absolute_move': expected_move,
            'percent_move': (expected_move / S) * 100,
            'lower_bound': S - expected_move,
            'upper_bound': S + expected_move
        }