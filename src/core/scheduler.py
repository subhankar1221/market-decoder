"""
Market Scheduler - Manages timed execution of analysis tasks
"""

import asyncio
import time
from datetime import datetime, time as dt_time
from typing import Dict, List, Callable, Optional
import schedule
import threading

from src.utils.logger import get_logger
from src.core.state_manager import SystemState

logger = get_logger(__name__)

class MarketScheduler:
    """Scheduler for market analysis tasks"""
    
    def __init__(self, state: SystemState, decoder):
        self.state = state
        self.decoder = decoder
        self.config = state.config['scheduler']
        
        self.running = False
        self.tasks = {}
        self.schedule_thread = None
        
        # Job definitions
        self.jobs = {
            'option_chain_update': {
                'function': self._update_option_chain,
                'interval': self.config['intervals']['high_frequency'],
                'enabled': True,
                'last_run': None,
                'next_run': None
            },
            'market_stats_update': {
                'function': self._update_market_stats,
                'interval': self.config['intervals']['medium_frequency'],
                'enabled': True,
                'last_run': None,
                'next_run': None
            },
            'full_analysis': {
                'function': self._run_full_analysis,
                'interval': self.config['intervals']['low_frequency'],
                'enabled': True,
                'last_run': None,
                'next_run': None
            },
            'risk_assessment': {
                'function': self._run_risk_assessment,
                'interval': 900,  # 15 minutes
                'enabled': True,
                'last_run': None,
                'next_run': None
            },
            'report_generation': {
                'function': self._generate_reports,
                'interval': 3600,  # 1 hour
                'enabled': True,
                'last_run': None,
                'next_run': None
            }
        }
    
    async def start(self):
        """Start the scheduler"""
        try:
            self.running = True
            logger.info("Market scheduler started")
            
            # Start background schedule thread
            self.schedule_thread = threading.Thread(target=self._run_schedule_loop, daemon=True)
            self.schedule_thread.start()
            
            # Run initial tasks
            await self._run_initial_tasks()
            
            # Main loop
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Scheduler start failed: {e}")
            raise
    
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        
        # Cancel all tasks
        for task_name, task_info in self.tasks.items():
            if task_info.get('task'):
                task_info['task'].cancel()
        
        logger.info("Market scheduler stopped")
    
    async def _run_initial_tasks(self):
        """Run initial tasks on startup"""
        try:
            logger.info("Running initial tasks...")
            
            # Run full analysis immediately
            await self._run_full_analysis()
            
            # Update market stats
            await self._update_market_stats()
            
            logger.info("Initial tasks completed")
            
        except Exception as e:
            logger.error(f"Initial tasks failed: {e}")
    
    def _run_schedule_loop(self):
        """Run schedule loop in background thread"""
        # Configure schedule based on config
        for job_config in self.config.get('jobs', []):
            job_name = job_config['name']
            if job_name in self.jobs and job_config['enabled']:
                interval = job_config['interval']
                
                # Schedule the job
                schedule.every(interval).seconds.do(
                    self._schedule_job_wrapper, job_name
                )
                logger.info(f"Scheduled job '{job_name}' every {interval} seconds")
        
        # Main schedule loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Schedule loop error: {e}")
                time.sleep(5)
    
    def _schedule_job_wrapper(self, job_name: str):
        """Wrapper to run scheduled jobs in asyncio"""
        asyncio.run_coroutine_threadsafe(
            self._execute_job(job_name),
            asyncio.get_event_loop()
        )
    
    async def _execute_job(self, job_name: str):
        """Execute a scheduled job"""
        if job_name not in self.jobs:
            logger.error(f"Unknown job: {job_name}")
            return
        
        job = self.jobs[job_name]
        
        # Check if job is enabled
        if not job['enabled']:
            return
        
        # Check if market is open
        if not self._is_market_open():
            logger.debug(f"Market closed, skipping {job_name}")
            return
        
        try:
            logger.info(f"Executing job: {job_name}")
            
            # Update job status
            job['last_run'] = datetime.now()
            job['next_run'] = datetime.now().timestamp() + job['interval']
            
            # Execute job
            await job['function']()
            
            logger.info(f"Job completed: {job_name}")
            
        except Exception as e:
            logger.error(f"Job {job_name} failed: {e}")
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            now = datetime.now()
            current_time = now.time()
            
            # Check if weekday
            if now.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Check market hours
            market_hours = self.state.config['market']['trading_hours']
            
            # Parse times
            market_open = dt_time.fromisoformat(market_hours['market_open'])
            market_close = dt_time.fromisoformat(market_hours['market_close'])
            
            # Check if within market hours
            return market_open <= current_time <= market_close
            
        except Exception as e:
            logger.error(f"Failed to check market hours: {e}")
            return False
    
    async def _update_option_chain(self):
        """Update option chain data"""
        try:
            # This would fetch and process option chain
            # For now, just trigger full analysis
            await self._run_full_analysis()
            
        except Exception as e:
            logger.error(f"Option chain update failed: {e}")
    
    async def _update_market_stats(self):
        """Update market statistics"""
        try:
            # Fetch and update market stats
            # This could be a lighter version of full analysis
            logger.debug("Updating market statistics")
            
        except Exception as e:
            logger.error(f"Market stats update failed: {e}")
    
    async def _run_full_analysis(self):
        """Run complete market analysis"""
        try:
            logger.info("Starting full market analysis...")
            
            # Run decoder's full analysis
            await self.decoder.run_full_analysis()
            
            logger.info("Full analysis completed")
            
        except Exception as e:
            logger.error(f"Full analysis failed: {e}")
    
    async def _run_risk_assessment(self):
        """Run risk assessment"""
        try:
            logger.info("Running risk assessment...")
            
            # Placeholder for risk assessment
            # This would analyze portfolio risk, margin requirements, etc.
            
            logger.info("Risk assessment completed")
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
    
    async def _generate_reports(self):
        """Generate reports"""
        try:
            logger.info("Generating reports...")
            
            # Placeholder for report generation
            # This would create daily/weekly reports
            
            logger.info("Reports generated")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
    
    def get_job_status(self) -> Dict:
        """Get status of all jobs"""
        status = {}
        for job_name, job in self.jobs.items():
            status[job_name] = {
                'enabled': job['enabled'],
                'last_run': job['last_run'],
                'next_run': job['next_run'],
                'interval': job['interval']
            }
        return status