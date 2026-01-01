"""
Logging configuration for Market Decoder
"""

import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import colorlog

class LoggerConfig:
    """Logger configuration class"""
    
    # Log format with colors
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    COLOR_LOG_FORMAT = '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Log colors
    LOG_COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
    
    @staticmethod
    def setup_logger(name='MarketDecoder', log_level=logging.INFO, log_to_file=True):
        """
        Setup logger with console and file handlers
        
        Args:
            name: Logger name
            log_level: Logging level
            log_to_file: Whether to log to file
            
        Returns:
            Configured logger
        """
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create formatters
        color_formatter = colorlog.ColoredFormatter(
            LoggerConfig.COLOR_LOG_FORMAT,
            log_colors=LoggerConfig.LOG_COLORS,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            LoggerConfig.LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if log_to_file:
            # Create logs directory if it doesn't exist
            log_dir = 'outputs/logs'
            os.makedirs(log_dir, exist_ok=True)
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file = os.path.join(log_dir, f'market_decoder_{timestamp}.log')
            
            # Rotating file handler (10MB per file, keep 5 backup files)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Add error handler for uncaught exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Call the default handler for KeyboardInterrupt
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        sys.excepthook = handle_exception
        
        return logger
    
    @staticmethod
    def get_logger(name=None):
        """
        Get logger instance
        
        Args:
            name: Logger name (optional)
            
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(name)
        else:
            # Return root logger if no name specified
            return logging.getLogger()
    
    @staticmethod
    def set_log_level(level):
        """
        Set log level for all handlers
        
        Args:
            level: Logging level
        """
        logger = logging.getLogger()
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

# Global setup function
def setup_logger(log_level=logging.INFO, log_to_file=True):
    """
    Setup the default logger
    
    Args:
        log_level: Logging level
        log_to_file: Whether to log to file
        
    Returns:
        Configured logger
    """
    return LoggerConfig.setup_logger('MarketDecoder', log_level, log_to_file)

# Global getter function
def get_logger(name=None):
    """
    Get logger instance
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    return LoggerConfig.get_logger(name)

# Module level logger
logger = get_logger(__name__)

# Example usage
if __name__ == "__main__":
    # Setup logger
    setup_logger(logging.DEBUG)
    
    # Get logger
    log = get_logger()
    
    # Test logging
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    log.critical("Critical message")