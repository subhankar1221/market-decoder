#!/usr/bin/env python3
"""
Simple starter for Market Decoder
Run this to test the system
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_system():
    """Test the system components"""
    print("🧪 Testing Market Decoder System...")
    
    try:
        # Test 1: Import modules
        print("1. Importing modules...")
        from src.utils.logger import setup_logger
        setup_logger()
        
        from src.data.fetcher import DataFetcher
        from src.models.black_scholes import BlackScholesModel
        from src.analytics.options.greeks import GreeksCalculator
        
        print("✅ Modules imported successfully")
        
        # Test 2: Load config
        print("\n2. Loading configuration...")
        import yaml
        
        with open("configs/defaults.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded for {config['market']['symbols']['primary']}")
        
        # Test 3: Test Black-Scholes model
        print("\n3. Testing Black-Scholes model...")
        bs = BlackScholesModel()
        
        # Test calculation
        price = bs.calculate_price(
            S=22000, K=22100, T=30/365,
            r=0.05, sigma=0.15, option_type='call'
        )
        
        greeks = bs.calculate_greeks(
            S=22000, K=22100, T=30/365,
            r=0.05, sigma=0.15, option_type='call'
        )
        
        print(f"   Option Price: {price:.2f}")
        print(f"   Delta: {greeks['delta']:.4f}")
        print(f"   Gamma: {greeks['gamma']:.6f}")
        print(f"   Theta: {greeks['theta']:.4f} per day")
        print("✅ Black-Scholes working")
        
        # Test 4: Test Greeks calculator
        print("\n4. Testing Greeks calculator...")
        greeks_calc = GreeksCalculator(config)
        
        # Create sample option chain
        sample_chain = [
            {
                'strikePrice': 21900,
                'CE': {
                    'impliedVolatility': 15.5,
                    'expiryDate': '25-Jan-2024'
                },
                'PE': {
                    'impliedVolatility': 16.2,
                    'expiryDate': '25-Jan-2024'
                }
            },
            {
                'strikePrice': 22000,
                'CE': {
                    'impliedVolatility': 14.8,
                    'expiryDate': '25-Jan-2024'
                },
                'PE': {
                    'impliedVolatility': 15.5,
                    'expiryDate': '25-Jan-2024'
                }
            }
        ]
        
        all_greeks = await greeks_calc.calculate_all(
            option_chain=sample_chain,
            spot_price=22000
        )
        
        print(f"   Calculated Greeks for {len(sample_chain)} strikes")
        print(f"   Delta exposure: {all_greeks.get('summary', {}).get('delta', {}).get('mean', 0):.4f}")
        print("✅ Greeks calculator working")
        
        print("\n🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Run 'python src/main.py' to start full system")
        print("2. Check configs/defaults.yaml for configuration")
        print("3. Monitor logs in outputs/logs/")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def create_project_structure():
    """Create necessary directories"""
    directories = [
        'configs/symbols',
        'outputs/json',
        'outputs/csv',
        'outputs/logs',
        'outputs/reports',
        'data/cache',
        'data/historical'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    print("\n📁 Project structure created")

if __name__ == "__main__":
    print("🚀 Market Decoder Pro - Setup Helper")
    print("=" * 50)
    
    # Create directories
    create_project_structure()
    
    # Run tests
    asyncio.run(test_system())