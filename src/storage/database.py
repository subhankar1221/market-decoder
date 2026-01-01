"""
Database storage for Market Decoder
Uprises SQLite for local storage
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.date_utils import get_ist_now

logger = get_logger(__name__)

class MarketDatabase:
    """Database manager for Market Decoder"""
    
    def __init__(self, config):
        self.config = config
        self.db_config = config['storage']['database']
        self.db_path = self.db_config['path']
        self.connection = None
        self.cursor = None
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            logger.info(f"Initializing database at {self.db_path}")
            
            # Connect to database
            self.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            
            # Enable WAL mode for better concurrency
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("PRAGMA cache_size=-2000")  # 2MB cache
            self.connection.execute("PRAGMA foreign_keys=ON")
            
            self.cursor = self.connection.cursor()
            
            # Create tables
            await self._create_tables()
            
            # Create indexes
            await self._create_indexes()
            
            # Run maintenance
            await self._run_maintenance()
            
            logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    async def _create_tables(self):
        """Create all database tables"""
        
        # 1. Market Data Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            data_type TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 2. Option Chain Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS option_chain (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            expiry_date DATE NOT NULL,
            strike_price REAL NOT NULL,
            option_type TEXT NOT NULL,  -- 'CE' or 'PE'
            open_interest INTEGER,
            change_in_oi INTEGER,
            volume INTEGER,
            iv REAL,
            last_price REAL,
            bid_price REAL,
            ask_price REAL,
            bid_qty INTEGER,
            ask_qty INTEGER,
            underlying_price REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol, expiry_date, strike_price, option_type)
        )
        ''')
        
        # 3. Greeks Data Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS greeks_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            expiry_date DATE NOT NULL,
            strike_price REAL NOT NULL,
            option_type TEXT NOT NULL,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            rho REAL,
            iv REAL,
            theoretical_price REAL,
            moneyness TEXT,
            probability_itm REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol, expiry_date, strike_price, option_type)
        )
        ''')
        
        # 4. Analysis Results Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            results_json TEXT NOT NULL,
            summary_text TEXT,
            confidence REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 5. Market Sentiment Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            overall_score REAL,
            sentiment TEXT,
            strength TEXT,
            components_json TEXT,
            signals_json TEXT,
            confidence REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol)
        )
        ''')
        
        # 6. Trading Signals Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            signal_strength TEXT,
            description TEXT,
            confidence REAL,
            parameters_json TEXT,
            triggered_by TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active',
            closed_at DATETIME,
            pnl REAL
        )
        ''')
        
        # 7. Risk Metrics Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            var_95 REAL,
            var_99 REAL,
            expected_shortfall REAL,
            max_drawdown REAL,
            volatility REAL,
            beta REAL,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol)
        )
        ''')
        
        # 8. Historical Prices Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            symbol TEXT NOT NULL,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER,
            vix REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, symbol)
        )
        ''')
        
        # 9. Backtest Results Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            backtest_id TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            initial_capital REAL,
            final_capital REAL,
            total_return REAL,
            annual_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            win_rate REAL,
            total_trades INTEGER,
            results_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(backtest_id)
        )
        ''')
        
        # 10. System Logs Table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            log_level TEXT NOT NULL,
            module TEXT,
            message TEXT,
            data_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.connection.commit()
        logger.info("Database tables created successfully")
    
    async def _create_indexes(self):
        """Create database indexes for performance"""
        
        # Indexes for market_data
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_market_data_timestamp 
        ON market_data(timestamp, symbol, data_type)
        ''')
        
        # Indexes for option_chain
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_option_chain_timestamp 
        ON option_chain(timestamp, symbol, expiry_date)
        ''')
        
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_option_chain_strike 
        ON option_chain(strike_price, option_type)
        ''')
        
        # Indexes for greeks_data
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_greeks_timestamp 
        ON greeks_data(timestamp, symbol)
        ''')
        
        # Indexes for analysis_results
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_analysis_timestamp 
        ON analysis_results(timestamp, symbol, analysis_type)
        ''')
        
        # Indexes for market_sentiment
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp 
        ON market_sentiment(timestamp, symbol)
        ''')
        
        # Indexes for trading_signals
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_signals_timestamp 
        ON trading_signals(timestamp, symbol, signal_type, status)
        ''')
        
        # Indexes for historical_prices
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_prices_date 
        ON historical_prices(date, symbol)
        ''')
        
        self.connection.commit()
        logger.info("Database indexes created successfully")
    
    async def _run_maintenance(self):
        """Run database maintenance tasks"""
        try:
            # Vacuum database to reclaim space
            self.cursor.execute("VACUUM")
            
            # Analyze database for query optimization
            self.cursor.execute("ANALYZE")
            
            # Set auto vacuum
            self.cursor.execute("PRAGMA auto_vacuum = INCREMENTAL")
            
            logger.info("Database maintenance completed")
            
        except Exception as e:
            logger.warning(f"Database maintenance failed: {e}")
    
    async def save_market_data(self, data_type: str, symbol: str, data: Dict):
        """Save market data to database"""
        try:
            timestamp = get_ist_now()
            data_json = json.dumps(data, default=str)
            
            self.cursor.execute('''
            INSERT OR REPLACE INTO market_data 
            (timestamp, symbol, data_type, data_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, symbol, data_type, data_json, timestamp))
            
            self.connection.commit()
            logger.debug(f"Saved {data_type} data for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save market data: {e}")
            self.connection.rollback()
    
    async def save_option_chain(self, symbol: str, option_data: Dict):
        """Save option chain data to database"""
        try:
            timestamp = get_ist_now()
            
            for option in option_data.get('data', []):
                strike = option.get('strike')
                expiry = option_data.get('current_expiry')
                underlying = option_data.get('spot_price', 0)
                
                # Save Call option
                if 'CE' in option:
                    ce_data = option['CE']
                    self.cursor.execute('''
                    INSERT OR REPLACE INTO option_chain 
                    (timestamp, symbol, expiry_date, strike_price, option_type,
                     open_interest, change_in_oi, volume, iv, last_price,
                     bid_price, ask_price, bid_qty, ask_qty, underlying_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp, symbol, expiry, strike, 'CE',
                        ce_data.get('oi'), ce_data.get('change_in_oi'),
                        ce_data.get('volume'), ce_data.get('iv'),
                        ce_data.get('last_price'), ce_data.get('bid'),
                        ce_data.get('ask'), ce_data.get('bid_qty'),
                        ce_data.get('ask_qty'), underlying
                    ))
                
                # Save Put option
                if 'PE' in option:
                    pe_data = option['PE']
                    self.cursor.execute('''
                    INSERT OR REPLACE INTO option_chain 
                    (timestamp, symbol, expiry_date, strike_price, option_type,
                     open_interest, change_in_oi, volume, iv, last_price,
                     bid_price, ask_price, bid_qty, ask_qty, underlying_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp, symbol, expiry, strike, 'PE',
                        pe_data.get('oi'), pe_data.get('change_in_oi'),
                        pe_data.get('volume'), pe_data.get('iv'),
                        pe_data.get('last_price'), pe_data.get('bid'),
                        pe_data.get('ask'), pe_data.get('bid_qty'),
                        pe_data.get('ask_qty'), underlying
                    ))
            
            self.connection.commit()
            logger.debug(f"Saved option chain for {symbol}: {len(option_data.get('data', []))} strikes")
            
        except Exception as e:
            logger.error(f"Failed to save option chain: {e}")
            self.connection.rollback()
    
    async def save_greeks(self, symbol: str, greeks_data: Dict):
        """Save Greeks data to database"""
        try:
            timestamp = get_ist_now()
            
            for strike, strike_data in greeks_data.get('by_strike', {}).items():
                for option_type, greeks in strike_data.items():
                    self.cursor.execute('''
                    INSERT OR REPLACE INTO greeks_data 
                    (timestamp, symbol, expiry_date, strike_price, option_type,
                     delta, gamma, theta, vega, rho, iv, theoretical_price,
                     moneyness, probability_itm)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp, symbol, 
                        greeks_data.get('metadata', {}).get('expiry_date', ''),
                        strike, option_type,
                        greeks.get('delta'), greeks.get('gamma'),
                        greeks.get('theta'), greeks.get('vega'),
                        greeks.get('rho'), greeks.get('iv'),
                        greeks.get('theoretical_price'),
                        greeks.get('moneyness'), greeks.get('probability_itm')
                    ))
            
            self.connection.commit()
            logger.debug(f"Saved Greeks data for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save Greeks data: {e}")
            self.connection.rollback()
    
    async def save_analysis(self, symbol: str, analysis_type: str, results: Dict, 
                          summary: str = "", confidence: float = 0.0):
        """Save analysis results to database"""
        try:
            timestamp = get_ist_now()
            results_json = json.dumps(results, default=str)
            
            self.cursor.execute('''
            INSERT INTO analysis_results 
            (timestamp, symbol, analysis_type, results_json, summary_text, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, symbol, analysis_type, results_json, summary, confidence))
            
            self.connection.commit()
            logger.debug(f"Saved {analysis_type} analysis for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            self.connection.rollback()
    
    async def save_sentiment(self, symbol: str, sentiment_data: Dict):
        """Save market sentiment to database"""
        try:
            timestamp = get_ist_now()
            components_json = json.dumps(sentiment_data.get('components', {}), default=str)
            signals_json = json.dumps(sentiment_data.get('signals', []), default=str)
            
            overall = sentiment_data.get('overall_sentiment', {})
            
            self.cursor.execute('''
            INSERT OR REPLACE INTO market_sentiment 
            (timestamp, symbol, overall_score, sentiment, strength,
             components_json, signals_json, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, symbol,
                overall.get('overall_score', 0),
                overall.get('sentiment', 'neutral'),
                overall.get('strength', 'weak'),
                components_json, signals_json,
                sentiment_data.get('confidence', 0.5)
            ))
            
            self.connection.commit()
            logger.debug(f"Saved market sentiment for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save sentiment: {e}")
            self.connection.rollback()
    
    async def save_signal(self, symbol: str, signal_data: Dict):
        """Save trading signal to database"""
        try:
            timestamp = get_ist_now()
            parameters_json = json.dumps(signal_data.get('parameters', {}), default=str)
            
            self.cursor.execute('''
            INSERT INTO trading_signals 
            (timestamp, symbol, signal_type, signal_strength, description,
             confidence, parameters_json, triggered_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, symbol,
                signal_data.get('type', 'unknown'),
                signal_data.get('strength', 'medium'),
                signal_data.get('description', ''),
                signal_data.get('confidence', 0.5),
                parameters_json,
                signal_data.get('trigger', 'system')
            ))
            
            self.connection.commit()
            logger.info(f"Saved trading signal for {symbol}: {signal_data.get('type')}")
            
            return self.cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            self.connection.rollback()
            return None
    
    async def save_risk_metrics(self, symbol: str, risk_data: Dict):
        """Save risk metrics to database"""
        try:
            timestamp = get_ist_now()
            
            self.cursor.execute('''
            INSERT OR REPLACE INTO risk_metrics 
            (timestamp, symbol, var_95, var_99, expected_shortfall,
             max_drawdown, volatility, beta, sharpe_ratio, sortino_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, symbol,
                risk_data.get('var_95', 0),
                risk_data.get('var_99', 0),
                risk_data.get('expected_shortfall', 0),
                risk_data.get('max_drawdown', 0),
                risk_data.get('volatility', 0),
                risk_data.get('beta', 0),
                risk_data.get('sharpe_ratio', 0),
                risk_data.get('sortino_ratio', 0)
            ))
            
            self.connection.commit()
            logger.debug(f"Saved risk metrics for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save risk metrics: {e}")
            self.connection.rollback()
    
    async def save_historical_price(self, symbol: str, price_data: Dict):
        """Save historical price data to database"""
        try:
            date = price_data.get('date', datetime.now().date())
            
            self.cursor.execute('''
            INSERT OR REPLACE INTO historical_prices 
            (date, symbol, open_price, high_price, low_price, 
             close_price, volume, vix)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date, symbol,
                price_data.get('open', 0),
                price_data.get('high', 0),
                price_data.get('low', 0),
                price_data.get('close', 0),
                price_data.get('volume', 0),
                price_data.get('vix', 0)
            ))
            
            self.connection.commit()
            logger.debug(f"Saved historical price for {symbol} on {date}")
            
        except Exception as e:
            logger.error(f"Failed to save historical price: {e}")
            self.connection.rollback()
    
    async def save_backtest_result(self, backtest_data: Dict):
        """Save backtest results to database"""
        try:
            backtest_id = backtest_data.get('backtest_id', str(datetime.now().timestamp()))
            results_json = json.dumps(backtest_data.get('results', {}), default=str)
            
            self.cursor.execute('''
            INSERT OR REPLACE INTO backtest_results 
            (backtest_id, start_date, end_date, symbol, strategy,
             initial_capital, final_capital, total_return, annual_return,
             sharpe_ratio, max_drawdown, win_rate, total_trades, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                backtest_id,
                backtest_data.get('start_date'),
                backtest_data.get('end_date'),
                backtest_data.get('symbol', 'NIFTY'),
                backtest_data.get('strategy', 'unknown'),
                backtest_data.get('initial_capital', 0),
                backtest_data.get('final_capital', 0),
                backtest_data.get('total_return', 0),
                backtest_data.get('annual_return', 0),
                backtest_data.get('sharpe_ratio', 0),
                backtest_data.get('max_drawdown', 0),
                backtest_data.get('win_rate', 0),
                backtest_data.get('total_trades', 0),
                results_json
            ))
            
            self.connection.commit()
            logger.info(f"Saved backtest result: {backtest_id}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest result: {e}")
            self.connection.rollback()
    
    async def log_system_event(self, level: str, message: str, module: str = "", data: Dict = None):
        """Log system event to database"""
        try:
            timestamp = get_ist_now()
            data_json = json.dumps(data, default=str) if data else None
            
            self.cursor.execute('''
            INSERT INTO system_logs 
            (timestamp, log_level, module, message, data_json)
            VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, level, module, message, data_json))
            
            self.connection.commit()
            
        except Exception as e:
            # Don't log database errors to database to avoid infinite loop
            print(f"Failed to log system event: {e}")
    
    # Query Methods
    
    async def get_market_data(self, symbol: str, data_type: str, 
                            start_time: datetime = None, 
                            end_time: datetime = None,
                            limit: int = 100) -> List[Dict]:
        """Get market data from database"""
        try:
            query = '''
            SELECT timestamp, data_json 
            FROM market_data 
            WHERE symbol = ? AND data_type = ?
            '''
            params = [symbol, data_type]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            results = []
            for timestamp, data_json in rows:
                data = json.loads(data_json)
                data['timestamp'] = timestamp
                results.append(data)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return []
    
    async def get_option_chain(self, symbol: str, expiry_date: str = None,
                             timestamp: datetime = None,
                             limit: int = 100) -> List[Dict]:
        """Get option chain data from database"""
        try:
            if timestamp:
                # Get specific timestamp
                query = '''
                SELECT * FROM option_chain 
                WHERE symbol = ? AND timestamp = ?
                ORDER BY strike_price
                '''
                params = [symbol, timestamp]
            else:
                # Get latest data
                query = '''
                WITH latest AS (
                    SELECT MAX(timestamp) as max_timestamp 
                    FROM option_chain 
                    WHERE symbol = ?
                )
                SELECT oc.* FROM option_chain oc
                JOIN latest ON oc.timestamp = latest.max_timestamp
                WHERE oc.symbol = ?
                '''
                params = [symbol, symbol]
                
                if expiry_date:
                    query += " AND oc.expiry_date = ?"
                    params.append(expiry_date)
                
                query += " ORDER BY oc.strike_price LIMIT ?"
                params.append(limit)
            
            self.cursor.execute(query, params)
            columns = [desc[0] for desc in self.cursor.description]
            rows = self.cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get option chain: {e}")
            return []
    
    async def get_greeks_data(self, symbol: str, timestamp: datetime = None,
                            expiry_date: str = None) -> List[Dict]:
        """Get Greeks data from database"""
        try:
            query = '''
            SELECT * FROM greeks_data 
            WHERE symbol = ?
            '''
            params = [symbol]
            
            if timestamp:
                query += " AND timestamp = ?"
                params.append(timestamp)
            else:
                query += " AND timestamp = (SELECT MAX(timestamp) FROM greeks_data WHERE symbol = ?)"
                params.append(symbol)
            
            if expiry_date:
                query += " AND expiry_date = ?"
                params.append(expiry_date)
            
            query += " ORDER BY strike_price, option_type"
            
            self.cursor.execute(query, params)
            columns = [desc[0] for desc in self.cursor.description]
            rows = self.cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get Greeks data: {e}")
            return []
    
    async def get_sentiment_history(self, symbol: str, 
                                  days: int = 7) -> List[Dict]:
        """Get sentiment history from database"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            
            query = '''
            SELECT timestamp, overall_score, sentiment, strength, 
                   components_json, signals_json, confidence
            FROM market_sentiment
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            '''
            
            self.cursor.execute(query, (symbol, start_time))
            rows = self.cursor.fetchall()
            
            results = []
            for row in rows:
                result = {
                    'timestamp': row[0],
                    'overall_score': row[1],
                    'sentiment': row[2],
                    'strength': row[3],
                    'components': json.loads(row[4]) if row[4] else {},
                    'signals': json.loads(row[5]) if row[5] else [],
                    'confidence': row[6]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get sentiment history: {e}")
            return []
    
    async def get_active_signals(self, symbol: str = None) -> List[Dict]:
        """Get active trading signals from database"""
        try:
            query = '''
            SELECT * FROM trading_signals 
            WHERE status = 'active'
            '''
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY timestamp DESC"
            
            self.cursor.execute(query, params)
            columns = [desc[0] for desc in self.cursor.description]
            rows = self.cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get active signals: {e}")
            return []
    
    async def get_historical_prices(self, symbol: str, 
                                  start_date: datetime,
                                  end_date: datetime = None) -> List[Dict]:
        """Get historical prices from database"""
        try:
            if end_date is None:
                end_date = datetime.now()
            
            query = '''
            SELECT date, open_price, high_price, low_price, 
                   close_price, volume, vix
            FROM historical_prices
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
            '''
            
            self.cursor.execute(query, (symbol, start_date, end_date))
            rows = self.cursor.fetchall()
            
            results = []
            for row in rows:
                result = {
                    'date': row[0],
                    'open': row[1],
                    'high': row[2],
                    'low': row[3],
                    'close': row[4],
                    'volume': row[5],
                    'vix': row[6]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get historical prices: {e}")
            return []
    
    async def get_analysis_results(self, symbol: str, analysis_type: str,
                                 limit: int = 10) -> List[Dict]:
        """Get analysis results from database"""
        try:
            query = '''
            SELECT timestamp, results_json, summary_text, confidence
            FROM analysis_results
            WHERE symbol = ? AND analysis_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
            '''
            
            self.cursor.execute(query, (symbol, analysis_type, limit))
            rows = self.cursor.fetchall()
            
            results = []
            for row in rows:
                result = {
                    'timestamp': row[0],
                    'results': json.loads(row[1]) if row[1] else {},
                    'summary': row[2],
                    'confidence': row[3]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get analysis results: {e}")
            return []
    
    async def get_risk_metrics_history(self, symbol: str,
                                     days: int = 30) -> List[Dict]:
        """Get risk metrics history from database"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            
            query = '''
            SELECT timestamp, var_95, var_99, expected_shortfall,
                   max_drawdown, volatility, sharpe_ratio
            FROM risk_metrics
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            '''
            
            self.cursor.execute(query, (symbol, start_time))
            rows = self.cursor.fetchall()
            
            results = []
            for row in rows:
                result = {
                    'timestamp': row[0],
                    'var_95': row[1],
                    'var_99': row[2],
                    'expected_shortfall': row[3],
                    'max_drawdown': row[4],
                    'volatility': row[5],
                    'sharpe_ratio': row[6]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get risk metrics history: {e}")
            return []
    
    async def get_backtest_results(self, strategy: str = None,
                                 limit: int = 20) -> List[Dict]:
        """Get backtest results from database"""
        try:
            query = '''
            SELECT backtest_id, start_date, end_date, symbol, strategy,
                   initial_capital, final_capital, total_return, annual_return,
                   sharpe_ratio, max_drawdown, win_rate, total_trades,
                   results_json, created_at
            FROM backtest_results
            '''
            params = []
            
            if strategy:
                query += " WHERE strategy = ?"
                params.append(strategy)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            results = []
            for row in rows:
                result = {
                    'backtest_id': row[0],
                    'start_date': row[1],
                    'end_date': row[2],
                    'symbol': row[3],
                    'strategy': row[4],
                    'initial_capital': row[5],
                    'final_capital': row[6],
                    'total_return': row[7],
                    'annual_return': row[8],
                    'sharpe_ratio': row[9],
                    'max_drawdown': row[10],
                    'win_rate': row[11],
                    'total_trades': row[12],
                    'results': json.loads(row[13]) if row[13] else {},
                    'created_at': row[14]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get backtest results: {e}")
            return []
    
    async def get_system_logs(self, level: str = None, 
                            module: str = None,
                            limit: int = 100) -> List[Dict]:
        """Get system logs from database"""
        try:
            query = '''
            SELECT timestamp, log_level, module, message, data_json
            FROM system_logs
            WHERE 1=1
            '''
            params = []
            
            if level:
                query += " AND log_level = ?"
                params.append(level)
            
            if module:
                query += " AND module = ?"
                params.append(module)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            results = []
            for row in rows:
                result = {
                    'timestamp': row[0],
                    'level': row[1],
                    'module': row[2],
                    'message': row[3],
                    'data': json.loads(row[4]) if row[4] else {}
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get system logs: {e}")
            return []
    
    # Statistical Methods
    
    async def get_statistics(self, symbol: str, days: int = 30) -> Dict:
        """Get market statistics from database"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            
            # Get price statistics
            price_query = '''
            SELECT close_price FROM historical_prices
            WHERE symbol = ? AND date >= ?
            ORDER BY date DESC
            LIMIT ?
            '''
            
            self.cursor.execute(price_query, (symbol, start_time.date(), days))
            prices = [row[0] for row in self.cursor.fetchall() if row[0]]
            
            # Get sentiment statistics
            sentiment_query = '''
            SELECT overall_score FROM market_sentiment
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp DESC
            '''
            
            self.cursor.execute(sentiment_query, (symbol, start_time))
            sentiments = [row[0] for row in self.cursor.fetchall() if row[0]]
            
            # Get volatility statistics
            volatility_query = '''
            SELECT volatility FROM risk_metrics
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp DESC
            '''
            
            self.cursor.execute(volatility_query, (symbol, start_time))
            volatilities = [row[0] for row in self.cursor.fetchall() if row[0]]
            
            # Calculate statistics
            stats = {
                'price_stats': {
                    'current': prices[0] if prices else 0,
                    'average': np.mean(prices) if prices else 0,
                    'std_dev': np.std(prices) if len(prices) > 1 else 0,
                    'min': min(prices) if prices else 0,
                    'max': max(prices) if prices else 0,
                    'count': len(prices)
                },
                'sentiment_stats': {
                    'current': sentiments[0] if sentiments else 0,
                    'average': np.mean(sentiments) if sentiments else 0,
                    'std_dev': np.std(sentiments) if len(sentiments) > 1 else 0,
                    'count': len(sentiments)
                },
                'volatility_stats': {
                    'current': volatilities[0] if volatilities else 0,
                    'average': np.mean(volatilities) if volatilities else 0,
                    'std_dev': np.std(volatilities) if len(volatilities) > 1 else 0,
                    'count': len(volatilities)
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def get_correlation_matrix(self, symbols: List[str], 
                                   days: int = 30) -> pd.DataFrame:
        """Get correlation matrix between symbols"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            
            # Get prices for all symbols
            all_prices = {}
            
            for symbol in symbols:
                query = '''
                SELECT date, close_price FROM historical_prices
                WHERE symbol = ? AND date >= ?
                ORDER BY date ASC
                '''
                
                self.cursor.execute(query, (symbol, start_time.date()))
                rows = self.cursor.fetchall()
                
                prices = {row[0]: row[1] for row in rows}
                all_prices[symbol] = prices
            
            # Align dates
            common_dates = set()
            for symbol in symbols:
                if not common_dates:
                    common_dates = set(all_prices[symbol].keys())
                else:
                    common_dates = common_dates.intersection(set(all_prices[symbol].keys()))
            
            common_dates = sorted(common_dates)
            
            # Create DataFrame
            data = {}
            for symbol in symbols:
                data[symbol] = [all_prices[symbol].get(date, np.nan) for date in common_dates]
            
            df = pd.DataFrame(data, index=common_dates)
            
            # Calculate correlation matrix
            correlation_matrix = df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Failed to get correlation matrix: {e}")
            return pd.DataFrame()
    
    async def get_performance_metrics(self, symbol: str, 
                                    start_date: datetime,
                                    end_date: datetime = None) -> Dict:
        """Get performance metrics for a symbol"""
        try:
            if end_date is None:
                end_date = datetime.now()
            
            # Get prices
            prices = await self.get_historical_prices(symbol, start_date, end_date)
            
            if not prices:
                return {}
            
            # Calculate returns
            price_series = [p['close'] for p in prices]
            returns = []
            for i in range(1, len(price_series)):
                if price_series[i-1] > 0:
                    ret = (price_series[i] - price_series[i-1]) / price_series[i-1]
                    returns.append(ret)
            
            if not returns:
                return {}
            
            # Calculate metrics
            total_return = (price_series[-1] - price_series[0]) / price_series[0]
            
            # Annualized return
            days = (end_date - start_date).days
            if days > 0:
                annual_return = (1 + total_return) ** (365 / days) - 1
            else:
                annual_return = 0
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio (assuming 5% risk-free rate)
            risk_free_rate = 0.05
            excess_returns = [r - risk_free_rate/252 for r in returns]
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            # Maximum drawdown
            peak = price_series[0]
            max_dd = 0
            for price in price_series:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak
                if dd > max_dd:
                    max_dd = dd
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns if r < risk_free_rate/252]
            downside_dev = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
            sortino_ratio = (annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0
            
            # Win rate (based on positive returns)
            positive_returns = [r for r in returns if r > 0]
            win_rate = len(positive_returns) / len(returns) if returns else 0
            
            # Average win/loss
            avg_win = np.mean(positive_returns) if positive_returns else 0
            negative_returns = [r for r in returns if r < 0]
            avg_loss = np.mean(negative_returns) if negative_returns else 0
            
            # Profit factor
            gross_profit = sum(positive_returns) if positive_returns else 0
            gross_loss = abs(sum(negative_returns)) if negative_returns else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_trades': len(returns),
                'positive_trades': len(positive_returns),
                'negative_trades': len(negative_returns)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    # Maintenance Methods
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from database"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old market data
            self.cursor.execute('''
            DELETE FROM market_data 
            WHERE timestamp < ?
            ''', (cutoff_date,))
            
            # Delete old option chain data (keep only last 7 days for details)
            option_cutoff = datetime.now() - timedelta(days=7)
            self.cursor.execute('''
            DELETE FROM option_chain 
            WHERE timestamp < ?
            ''', (option_cutoff,))
            
            # Delete old Greeks data
            self.cursor.execute('''
            DELETE FROM greeks_data 
            WHERE timestamp < ?
            ''', (option_cutoff,))
            
            # Delete old analysis results (keep only last 30 days)
            self.cursor.execute('''
            DELETE FROM analysis_results 
            WHERE timestamp < ?
            ''', (cutoff_date,))
            
            # Delete old system logs (keep only last 14 days)
            log_cutoff = datetime.now() - timedelta(days=14)
            self.cursor.execute('''
            DELETE FROM system_logs 
            WHERE timestamp < ?
            ''', (log_cutoff,))
            
            self.connection.commit()
            
            rows_deleted = self.cursor.rowcount
            logger.info(f"Cleaned up {rows_deleted} old records from database")
            
            # Vacuum to reclaim space
            self.cursor.execute("VACUUM")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            self.connection.rollback()
    
    async def backup_database(self, backup_path: str = None):
        """Create a backup of the database"""
        try:
            if backup_path is None:
                backup_dir = "backups"
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"market_decoder_backup_{timestamp}.db")
            
            # Use SQLite backup API
            backup_conn = sqlite3.connect(backup_path)
            self.connection.backup(backup_conn)
            backup_conn.close()
            
            logger.info(f"Database backup created at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return None
    
    async def restore_database(self, backup_path: str):
        """Restore database from backup"""
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Close current connection
            await self.close()
            
            # Replace current database with backup
            import shutil
            shutil.copy2(backup_path, self.db_path)
            
            # Reinitialize
            await self.initialize()
            
            logger.info(f"Database restored from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            return False
    
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}
            
            # Table row counts
            tables = [
                'market_data', 'option_chain', 'greeks_data',
                'analysis_results', 'market_sentiment', 'trading_signals',
                'risk_metrics', 'historical_prices', 'backtest_results',
                'system_logs'
            ]
            
            for table in tables:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cursor.fetchone()[0]
                stats[f"{table}_count"] = count
            
            # Database size
            if os.path.exists(self.db_path):
                stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
            # Oldest and newest records
            self.cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM market_data")
            min_max = self.cursor.fetchone()
            if min_max and min_max[0]:
                stats['oldest_record'] = min_max[0]
                stats['newest_record'] = min_max[1]
                stats['data_age_days'] = (datetime.now() - datetime.fromisoformat(min_max[0])).days
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def close(self):
        """Close database connection"""
        try:
            if self.connection:
                # Run cleanup before closing
                await self.cleanup_old_data()
                
                # Close connection
                self.connection.close()
                logger.info("Database connection closed")
                
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    async def save_analysis(self, analysis_report: Dict):
        """Save complete analysis report to database"""
        try:
            timestamp = analysis_report.get('timestamp', get_ist_now())
            symbol = analysis_report.get('symbol', 'NIFTY')
            
            # Save different components
            if 'option_analysis' in analysis_report:
                await self.save_analysis(
                    symbol, 'option_chain',
                    analysis_report['option_analysis'],
                    "Option chain analysis",
                    0.8
                )
            
            if 'greeks_analysis' in analysis_report:
                await self.save_analysis(
                    symbol, 'greeks',
                    analysis_report['greeks_analysis'],
                    "Greeks analysis",
                    0.7
                )
            
            if 'sentiment_analysis' in analysis_report:
                await self.save_sentiment(symbol, analysis_report['sentiment_analysis'])
            
            if 'risk_analysis' in analysis_report:
                await self.save_risk_metrics(symbol, analysis_report['risk_analysis'])
            
            if 'signals' in analysis_report:
                for signal in analysis_report['signals']:
                    await self.save_signal(symbol, signal)
            
            # Save overall report
            await self.save_analysis(
                symbol, 'full_report',
                analysis_report,
                "Complete market analysis",
                analysis_report.get('metadata', {}).get('confidence', 0.5)
            )
            
            logger.info(f"Saved complete analysis for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")