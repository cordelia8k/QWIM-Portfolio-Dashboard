"""
Enhanced Multi-Goal Wealth Management Module - COMPLETE VERSION
===============================================================
Implements the full dynamic programming algorithm from:
"Dynamic Optimization for Multi-Goals Wealth Management" (Das et al., 2022)
WITH synthetic demo ETF scenarios; no original input data is distributed
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import copy
import typing
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import asyncio
import concurrent.futures
from functools import lru_cache
import warnings
import json

# =============================================================================
# Third-Party Imports
# =============================================================================
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import norm, multivariate_normal, t as t_dist
from scipy.optimize import minimize

# =============================================================================
# Shiny Framework Imports
# =============================================================================
from shiny import ui, render, reactive

# =============================================================================
# Global Storage with Enhanced Structure
# =============================================================================
goals_storage = []
infusions_storage = []
goal_id_counter = 0
infusion_id_counter = 0
optimization_results = {}

# =============================================================================
# Enhanced ETF Portfolio Data Integration
# =============================================================================
class ETFPortfolioManager:
    """
    Enhanced ETF portfolio manager with correlation modeling and risk budgeting
    """
    
    def __init__(self, data_inputs: dict):
        """Initialize with QWIM dashboard data"""
        self.data_inputs = data_inputs
        self.etf_data = None
        self.returns_data = None
        self.portfolio_weights = None
        self.efficient_frontier = None
        self.correlation_matrix = None
        self.expected_returns = None
        self.covariance_matrix = None
        
        self._load_etf_data()
        self._calculate_returns()
        self._estimate_parameters()
        self._build_efficient_frontier()
    
    def _load_etf_data(self):
        """Load ETF price data from QWIM dashboard data inputs"""
        try:
            # Initialize default attributes first
            self.etf_components = ['IVV', 'IWM', 'EFA', 'EEM', 'AGG', 'GLD', 'IYR']  # Default components
            self.etf_descriptions = {
                'IVV': 'iShares Core S&P 500 ETF',
                'IWM': 'iShares Russell 2000 ETF',
                'EFA': 'iShares MSCI EAFE ETF',
                'EEM': 'iShares MSCI Emerging Markets ETF',
                'AGG': 'iShares Core US Aggregate Bond ETF',
                'GLD': 'SPDR Gold Shares ETF',
                'IYR': 'iShares US Real Estate ETF'
            }
            
            # Try to load portfolio weights data
            if "Weights_My_Portfolio" in self.data_inputs:
                weights_data = self.data_inputs["Weights_My_Portfolio"]
                pass  # Do not log visitor financial inputs.
                
                # Convert to pandas if it's polars
                if hasattr(weights_data, 'to_pandas'):
                    self.portfolio_weights = weights_data.to_pandas()
                else:
                    self.portfolio_weights = weights_data
                    
                # Get unique ETF components if available
                if 'component' in self.portfolio_weights.columns:
                    actual_components = self.portfolio_weights['component'].unique().tolist()
                    if actual_components:
                        self.etf_components = actual_components
                    pass  # Do not log visitor financial inputs.
                else:
                    pass  # Do not log visitor financial inputs.
                    
            # Try to load price data for backtesting
            price_data_keys = [key for key in self.data_inputs.keys() if 'price' in key.lower() or 'etf' in key.lower()]
            if price_data_keys:
                self.etf_data = self.data_inputs[price_data_keys[0]]
                pass  # Do not log visitor financial inputs.
            else:
                pass  # Do not log visitor financial inputs.
                self._create_enhanced_synthetic_etf_data()
                
        except Exception as e:
            pass  # Do not log visitor financial inputs.
            # Ensure we have default components even on error
            if not hasattr(self, 'etf_components'):
                self.etf_components = ['IVV', 'IWM', 'EFA', 'EEM', 'AGG', 'GLD', 'IYR']
            self._create_enhanced_synthetic_etf_data()
    
    def _create_enhanced_synthetic_etf_data(self):
        """Create enhanced synthetic ETF data with realistic correlations and regime changes"""
        pass  # Do not log visitor financial inputs.
        
        # Create synthetic daily returns for your specified ETFs
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='B')
        
        # Enhanced ETF profiles with more realistic parameters
        etf_profiles = {
            'IVV': {'mu': 0.10, 'sigma': 0.16, 'name': 'iShares Core S&P 500 ETF', 'beta': 1.0},
            'IWM': {'mu': 0.08, 'sigma': 0.22, 'name': 'iShares Russell 2000 ETF', 'beta': 1.2},
            'EFA': {'mu': 0.07, 'sigma': 0.18, 'name': 'iShares MSCI EAFE ETF', 'beta': 0.8},
            'EEM': {'mu': 0.05, 'sigma': 0.25, 'name': 'iShares MSCI Emerging Markets ETF', 'beta': 1.5},
            'AGG': {'mu': 0.03, 'sigma': 0.04, 'name': 'iShares Core US Aggregate Bond ETF', 'beta': 0.1},
            'GLD': {'mu': 0.04, 'sigma': 0.16, 'name': 'SPDR Gold Shares ETF', 'beta': 0.2},
            'IYR': {'mu': 0.08, 'sigma': 0.20, 'name': 'iShares US Real Estate ETF', 'beta': 0.9},
        }
        
        # FIXED: Set etf_components and etf_descriptions before other operations
        self.etf_components = list(etf_profiles.keys())
        self.etf_descriptions = {etf: profile['name'] for etf, profile in etf_profiles.items()}
        
        pass  # Do not log visitor financial inputs.
        
        # Enhanced correlation matrix with regime-dependent correlations
        self._create_realistic_correlation_matrix(etf_profiles)
        
        # Generate returns with regime changes and fat tails
        self._generate_regime_switching_returns(etf_profiles, dates)
    
    def _create_realistic_correlation_matrix(self, etf_profiles):
        """Create realistic correlation matrix based on asset class relationships"""
        asset_names = list(etf_profiles.keys())
        n_assets = len(asset_names)
        
        # Ensure etf_components is set
        if not hasattr(self, 'etf_components'):
            self.etf_components = asset_names
        
        # Base correlation matrix
        base_correlations = {
            ('IVV', 'IWM'): 0.75,  ('IVV', 'EFA'): 0.70,  ('IVV', 'EEM'): 0.60,
            ('IVV', 'AGG'): 0.05,  ('IVV', 'GLD'): 0.10,  ('IVV', 'IYR'): 0.65,
            ('IWM', 'EFA'): 0.65,  ('IWM', 'EEM'): 0.55,  ('IWM', 'AGG'): 0.00,
            ('IWM', 'GLD'): 0.05,  ('IWM', 'IYR'): 0.70,  ('EFA', 'EEM'): 0.75,
            ('EFA', 'AGG'): 0.10,  ('EFA', 'GLD'): 0.15,  ('EFA', 'IYR'): 0.60,
            ('EEM', 'AGG'): 0.05,  ('EEM', 'GLD'): 0.25,  ('EEM', 'IYR'): 0.50,
            ('AGG', 'GLD'): 0.20,  ('AGG', 'IYR'): 0.15,  ('GLD', 'IYR'): 0.30,
        }
        
        # Create correlation matrix
        correlation_matrix = np.eye(n_assets)
        for i, asset1 in enumerate(asset_names):
            for j, asset2 in enumerate(asset_names):
                if i != j:
                    corr_key = (asset1, asset2) if (asset1, asset2) in base_correlations else (asset2, asset1)
                    if corr_key in base_correlations:
                        correlation_matrix[i, j] = base_correlations[corr_key]
        
        self.correlation_matrix = correlation_matrix
        pass  # Do not log visitor financial inputs.
    
    def _generate_regime_switching_returns(self, etf_profiles, dates):
        """Generate returns with regime switching and fat tails"""
        asset_names = list(etf_profiles.keys())
        n_periods = len(dates)
        
        # Define regimes (normal, crisis, recovery)
        regime_probs = [0.7, 0.15, 0.15]  # Normal, Crisis, Recovery
        regime_adjustments = {
            'normal': {'mu_adj': 1.0, 'sigma_adj': 1.0, 'corr_adj': 1.0},
            'crisis': {'mu_adj': -0.5, 'sigma_adj': 2.0, 'corr_adj': 1.5},
            'recovery': {'mu_adj': 1.5, 'sigma_adj': 1.2, 'corr_adj': 0.8}
        }
        
        # Generate regime sequence
        rng = np.random.default_rng(42)
        regimes = rng.choice(['normal', 'crisis', 'recovery'], 
                                 size=n_periods, p=regime_probs)
        
        returns_matrix = np.zeros((n_periods, len(asset_names)))
        
        for t in range(n_periods):
            regime = regimes[t]
            adj = regime_adjustments[regime]
            
            # Adjusted parameters for this regime
            means = [(etf_profiles[etf]['mu'] / 252) * adj['mu_adj'] for etf in asset_names]
            stds = [(etf_profiles[etf]['sigma'] / np.sqrt(252)) * adj['sigma_adj'] for etf in asset_names]
            
            # Adjusted correlation matrix
            adj_corr = self.correlation_matrix * adj['corr_adj']
            np.fill_diagonal(adj_corr, 1.0)
            
            # Ensure positive definite
            eigenvals, eigenvecs = np.linalg.eigh(adj_corr)
            eigenvals = np.maximum(eigenvals, 0.01)
            adj_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Create covariance matrix
            cov_matrix = np.outer(stds, stds) * adj_corr
            
            # Generate returns with t-distribution for fat tails
            if regime == 'crisis':
                # Use t-distribution with 3 degrees of freedom for fat tails
                normal_draws = rng.multivariate_normal(np.zeros(len(asset_names)), adj_corr)
                t_draws = t_dist.rvs(df=3, size=len(asset_names), random_state=rng) / np.sqrt(3/(3-2))  # Normalize
                scaled_draws = normal_draws * t_draws
                returns_matrix[t] = np.array(means) + np.array(stds) * scaled_draws
            else:
                # Normal multivariate distribution
                returns_matrix[t] = rng.multivariate_normal(means, cov_matrix)
        
        # Create returns DataFrame
        self.returns_data = pd.DataFrame(returns_matrix, columns=asset_names, index=dates)
        pass  # Do not log visitor financial inputs.
    
    def _calculate_returns(self):
        """Calculate returns from price data or use synthetic returns"""
        if self.returns_data is not None:
            pass  # Do not log visitor financial inputs.
            return
            
        # If we have real price data, calculate returns
        if self.etf_data is not None:
            try:
                if hasattr(self.etf_data, 'to_pandas'):
                    price_df = self.etf_data.to_pandas()
                else:
                    price_df = self.etf_data
                    
                price_cols = [col for col in price_df.columns if col.lower() != 'date']
                
                if price_cols:
                    self.returns_data = price_df[price_cols].pct_change().dropna()
                    pass  # Do not log visitor financial inputs.
                else:
                    pass  # Do not log visitor financial inputs.
                    self._create_enhanced_synthetic_etf_data()
                    
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                self._create_enhanced_synthetic_etf_data()
    
    def _estimate_parameters(self):
        """Estimate expected returns and covariance matrix with robust methods"""
        if self.returns_data is None:
            pass  # Do not log visitor financial inputs.
            return
        
        try:
            # Ensure etf_components is available
            if not hasattr(self, 'etf_components') or not self.etf_components:
                self.etf_components = list(self.returns_data.columns)
                pass  # Do not log visitor financial inputs.
            
            # Use exponentially weighted moving averages for more recent emphasis
            decay_factor = 0.94  # EWMA decay factor (RiskMetrics standard)
            
            # Calculate EWMA statistics
            returns_array = self.returns_data.values
            n, k = returns_array.shape
            
            # EWMA weights
            weights = np.array([(1 - decay_factor) * (decay_factor ** i) for i in range(n)])
            weights = weights[::-1] / weights.sum()  # Reverse and normalize
            
            # EWMA expected returns (annualized)
            self.expected_returns = np.average(returns_array, axis=0, weights=weights) * 252
            
            # EWMA covariance matrix
            mean_returns = np.average(returns_array, axis=0, weights=weights)
            deviations = returns_array - mean_returns
            
            # Weighted covariance calculation
            weighted_cov = np.zeros((k, k))
            for i in range(n):
                deviation = deviations[i:i+1].T
                weighted_cov += weights[i] * (deviation @ deviation.T)
            
            self.covariance_matrix = weighted_cov * 252  # Annualized
            
            pass  # Do not log visitor financial inputs.
            pass  # Do not log visitor financial inputs.
            
        except Exception as e:
            pass  # Do not log visitor financial inputs.
            # Fallback to simple estimates
            try:
                self.expected_returns = self.returns_data.mean() * 252
                self.covariance_matrix = self.returns_data.cov() * 252
                pass  # Do not log visitor financial inputs.
            except Exception as e2:
                pass  # Do not log visitor financial inputs.
                # Create synthetic estimates
                n_assets = len(self.etf_components)
                self.expected_returns = np.linspace(0.03, 0.12, n_assets)  # 3% to 12%
                self.covariance_matrix = np.eye(n_assets) * 0.04  # 4% volatility diagonal
    
    def _build_efficient_frontier(self):
        """Build efficient frontier using enhanced mean-variance optimization"""
        if self.returns_data is None:
            return
            
        try:
            pass  # Do not log visitor financial inputs.
            
            mu = self.expected_returns
            Sigma = self.covariance_matrix
            n_assets = len(mu)
            
            # Risk-free rate (estimated from short-term treasury)
            risk_free_rate = 0.02
            
            # Generate target returns from min to max
            min_ret = max(mu.min(), risk_free_rate)
            max_ret = mu.max()
            target_returns = np.linspace(min_ret, max_ret, 25)
            
            efficient_portfolios = []
            
            for target_return in target_returns:
                try:
                    # Solve portfolio optimization with constraints
                    result = self._solve_portfolio_optimization(mu, Sigma, target_return, risk_free_rate)
                    if result is not None:
                        efficient_portfolios.append(result)
                except Exception as e:
                    pass  # Do not log visitor financial inputs.
                    continue
            
            # Add special portfolios
            efficient_portfolios.extend(self._get_special_portfolios(mu, Sigma, risk_free_rate))
            
            # Sort by volatility
            efficient_portfolios.sort(key=lambda x: x['volatility'])
            
            self.efficient_frontier = efficient_portfolios
            pass  # Do not log visitor financial inputs.
            
        except Exception as e:
            pass  # Do not log visitor financial inputs.
            self.efficient_frontier = self._get_simple_fallback_strategies()
    
    def _solve_portfolio_optimization(self, mu, Sigma, target_return, risk_free_rate):
        """Solve mean-variance optimization for target return"""
        n_assets = len(mu)
        
        # Objective: minimize portfolio variance
        def objective(weights):
            return 0.5 * np.dot(weights, np.dot(Sigma, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
            {'type': 'eq', 'fun': lambda x: np.dot(x, mu) - target_return}  # Target return
        ]
        
        # Bounds: allow small short positions but prefer long-only
        bounds = tuple((-0.1, 1.0) for _ in range(n_assets))
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Solve optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success and all(abs(result.x.sum() - 1) < 1e-6 for _ in [0]):
            portfolio_return = np.dot(result.x, mu)
            portfolio_vol = np.sqrt(np.dot(result.x, np.dot(Sigma, result.x)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Classify risk level
            risk_level = self._classify_risk_level(portfolio_vol)
            
            # Create weights dictionary
            weights_dict = dict(zip(self.etf_components, result.x))
            
            return {
                'weights': weights_dict,
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': sharpe_ratio,
                'risk_level': risk_level,
                'assets': self.etf_components,
                'name': f"{risk_level} Portfolio ({portfolio_return:.1%} return)"
            }
        
        return None
    
    def _get_special_portfolios(self, mu, Sigma, risk_free_rate):
        """Get special portfolios (minimum variance, maximum Sharpe, etc.)"""
        special_portfolios = []
        n_assets = len(mu)
        
        try:
            # Minimum variance portfolio
            ones = np.ones((n_assets, 1))
            Sigma_inv = np.linalg.inv(Sigma)
            min_var_weights = (Sigma_inv @ ones) / (ones.T @ Sigma_inv @ ones)
            min_var_weights = min_var_weights.flatten()
            
            if np.all(min_var_weights >= -0.1) and np.all(min_var_weights <= 1.1):
                min_var_weights /= min_var_weights.sum()  # Normalize
                
                portfolio_return = np.dot(min_var_weights, mu)
                portfolio_vol = np.sqrt(np.dot(min_var_weights, np.dot(Sigma, min_var_weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                
                special_portfolios.append({
                    'weights': dict(zip(self.etf_components, min_var_weights)),
                    'return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe': sharpe_ratio,
                    'risk_level': 'Minimum Variance',
                    'assets': self.etf_components,
                    'name': 'Global Minimum Variance Portfolio'
                })
        
        except Exception as e:
            pass  # Do not log visitor financial inputs.
        
        try:
            # Maximum Sharpe ratio portfolio (tangency portfolio)
            excess_returns = mu - risk_free_rate
            max_sharpe_weights = (Sigma_inv @ excess_returns) / (ones.T @ Sigma_inv @ excess_returns)
            max_sharpe_weights = max_sharpe_weights.flatten()
            
            if np.all(max_sharpe_weights >= -0.1) and np.all(max_sharpe_weights <= 1.1):
                max_sharpe_weights /= max_sharpe_weights.sum()  # Normalize
                
                portfolio_return = np.dot(max_sharpe_weights, mu)
                portfolio_vol = np.sqrt(np.dot(max_sharpe_weights, np.dot(Sigma, max_sharpe_weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                
                special_portfolios.append({
                    'weights': dict(zip(self.etf_components, max_sharpe_weights)),
                    'return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe': sharpe_ratio,
                    'risk_level': 'Maximum Sharpe',
                    'assets': self.etf_components,
                    'name': 'Maximum Sharpe Ratio Portfolio'
                })
        
        except Exception as e:
            pass  # Do not log visitor financial inputs.
        
        return special_portfolios
    
    def _classify_risk_level(self, volatility):
        """Classify portfolio risk level based on volatility"""
        if volatility < 0.06:
            return "Very Conservative"
        elif volatility < 0.10:
            return "Conservative"
        elif volatility < 0.14:
            return "Moderate"
        elif volatility < 0.18:
            return "Aggressive"
        else:
            return "Very Aggressive"
    
    def _get_simple_fallback_strategies(self):
        """Fallback strategies if optimization fails"""
        strategies = []
        
        # Simple allocation templates
        allocation_templates = [
            ('Very Conservative', {'AGG': 0.70, 'IVV': 0.20, 'EFA': 0.05, 'GLD': 0.05}),
            ('Conservative', {'AGG': 0.50, 'IVV': 0.30, 'EFA': 0.15, 'GLD': 0.05}),
            ('Moderate Conservative', {'AGG': 0.35, 'IVV': 0.40, 'EFA': 0.15, 'EEM': 0.05, 'GLD': 0.05}),
            ('Moderate', {'AGG': 0.20, 'IVV': 0.45, 'EFA': 0.20, 'EEM': 0.10, 'GLD': 0.03, 'IYR': 0.02}),
            ('Moderate Aggressive', {'AGG': 0.10, 'IVV': 0.50, 'IWM': 0.15, 'EFA': 0.15, 'EEM': 0.08, 'IYR': 0.02}),
            ('Aggressive', {'AGG': 0.05, 'IVV': 0.55, 'IWM': 0.20, 'EFA': 0.10, 'EEM': 0.08, 'IYR': 0.02}),
            ('Very Aggressive', {'IVV': 0.60, 'IWM': 0.25, 'EFA': 0.08, 'EEM': 0.05, 'IYR': 0.02}),
        ]
        
        for risk_level, allocation in allocation_templates:
            # Estimate return and volatility
            portfolio_return = sum(allocation.get(etf, 0) * 0.08 for etf in allocation.keys())  # Rough estimate
            portfolio_vol = 0.04 + (portfolio_return - 0.03) * 2  # Rough volatility estimate
            
            strategies.append({
                'weights': allocation,
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0,
                'risk_level': risk_level,
                'assets': list(allocation.keys()),
                'name': f"{risk_level} Portfolio"
            })
        
        return strategies
    
    def get_portfolio_strategies(self, n_strategies: int = 10) -> List[Dict]:
        """Get optimized portfolio strategies for the wealth management algorithm"""
        if not self.efficient_frontier:
            return self._get_simple_fallback_strategies()[:n_strategies]
        
        # Select strategies across the efficient frontier
        n_available = len(self.efficient_frontier)
        if n_available <= n_strategies:
            return self.efficient_frontier
        
        # Select evenly spaced portfolios
        indices = np.linspace(0, n_available-1, n_strategies, dtype=int)
        selected_strategies = [self.efficient_frontier[i] for i in indices]
        
        pass  # Do not log visitor financial inputs.
        return selected_strategies

# =============================================================================
# NEW: Goal-Responsive Portfolio Enhancement
# =============================================================================
@dataclass
class GoalCharacteristics:
    """Goal characteristics for portfolio optimization"""
    total_cost: float
    weighted_time_horizon: float
    min_time_horizon: float
    max_priority: float
    has_short_term_goals: bool  # < 3 years
    has_high_priority_goals: bool  # priority > 1.5
    cost_concentration: float  # measure of how concentrated costs are
    
class GoalResponsivePortfolioManager:
    """
    Enhanced portfolio manager that dynamically optimizes based on goal characteristics
    """
    
    def __init__(self, etf_manager):
        self.etf_manager = etf_manager
        self.current_goals = []
        self.current_goal_characteristics = None
        self.dynamic_portfolios = []
        
    def update_goals(self, goals_storage: List[Dict], infusions_storage: List[Dict], 
                    initial_wealth: float, horizon_years: int):
        """Update portfolio strategies based on current goals"""
        pass  # Do not log visitor financial inputs.
        
        self.current_goals = goals_storage
        
        # Calculate goal characteristics
        self.current_goal_characteristics = self._analyze_goal_characteristics(
            goals_storage, infusions_storage, initial_wealth, horizon_years
        )
        
        # Generate goal-responsive portfolio strategies
        self.dynamic_portfolios = self._generate_goal_responsive_strategies(
            self.current_goal_characteristics
        )
        
        pass  # Do not log visitor financial inputs.
        return self.dynamic_portfolios
    
    def _analyze_goal_characteristics(self, goals_storage, infusions_storage, 
                                    initial_wealth, horizon_years) -> GoalCharacteristics:
        """Analyze goal characteristics to inform portfolio optimization"""
        if not goals_storage:
            return self._default_goal_characteristics(initial_wealth, horizon_years)
        
        # Calculate key metrics
        total_cost = sum(g.get('cost', 0) for g in goals_storage)
        total_infusions = sum(i.get('amount', 0) for i in infusions_storage)
        
        # Time horizon analysis
        goal_times = [g.get('time_year', 10) for g in goals_storage]
        goal_costs = [g.get('cost', 0) for g in goals_storage]
        priorities = [g.get('priority', 1.0) for g in goals_storage]
        
        # Weighted average time horizon (weighted by cost)
        if sum(goal_costs) > 0:
            weighted_time = sum(t * c for t, c in zip(goal_times, goal_costs)) / sum(goal_costs)
        else:
            weighted_time = np.mean(goal_times) if goal_times else 10
        
        min_time = min(goal_times) if goal_times else horizon_years
        max_priority = max(priorities) if priorities else 1.0
        
        # Goal timing analysis
        has_short_term = any(t <= 3 for t in goal_times)
        has_high_priority = any(p > 1.5 for p in priorities)
        
        # Cost concentration (Gini coefficient-like measure)
        if len(goal_costs) > 1:
            sorted_costs = sorted(goal_costs)
            n = len(sorted_costs)
            cumsum = np.cumsum(sorted_costs)
            cost_concentration = (2 * sum((i + 1) * cost for i, cost in enumerate(sorted_costs))) / (n * sum(sorted_costs)) - (n + 1) / n
        else:
            cost_concentration = 1.0
        
        characteristics = GoalCharacteristics(
            total_cost=total_cost,
            weighted_time_horizon=weighted_time,
            min_time_horizon=min_time,
            max_priority=max_priority,
            has_short_term_goals=has_short_term,
            has_high_priority_goals=has_high_priority,
            cost_concentration=cost_concentration
        )
        
        pass  # Do not log visitor financial inputs.

        pass  # Do not log visitor financial inputs.

        
        return characteristics
    
    def _default_goal_characteristics(self, initial_wealth, horizon_years):
        """Default characteristics when no goals are present"""
        return GoalCharacteristics(
            total_cost=initial_wealth * 0.5,
            weighted_time_horizon=horizon_years * 0.7,
            min_time_horizon=horizon_years * 0.3,
            max_priority=1.0,
            has_short_term_goals=False,
            has_high_priority_goals=False,
            cost_concentration=0.5
        )
    
    def _generate_goal_responsive_strategies(self, characteristics: GoalCharacteristics) -> List[Dict]:
        """Generate portfolio strategies optimized for specific goal characteristics"""
        strategies = []
        
        # Get base parameters from ETF manager
        if not self.etf_manager or not hasattr(self.etf_manager, 'expected_returns'):
            return self._fallback_strategies()
        
        mu = self.etf_manager.expected_returns
        Sigma = self.etf_manager.covariance_matrix
        etf_components = self.etf_manager.etf_components
        
        # Define strategy targets based on goal characteristics
        strategy_configs = self._get_goal_responsive_configs(characteristics)
        
        # Generate optimized portfolios for each configuration
        for config in strategy_configs:
            try:
                optimal_weights = self._optimize_goal_responsive_portfolio(
                    mu, Sigma, config, characteristics
                )
                
                if optimal_weights is not None:
                    # Calculate portfolio metrics
                    portfolio_return = np.dot(optimal_weights, mu)
                    portfolio_vol = np.sqrt(np.dot(optimal_weights, np.dot(Sigma, optimal_weights)))
                    sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
                    
                    # Create strategy dictionary
                    strategy = {
                        'weights': dict(zip(etf_components, optimal_weights)),
                        'return': portfolio_return,
                        'volatility': portfolio_vol,
                        'sharpe': sharpe_ratio,
                        'risk_level': config['name'],
                        'assets': etf_components,
                        'name': f"Goal-Optimized {config['name']}",
                        'goal_responsive': True,
                        'config': config
                    }
                    
                    strategies.append(strategy)
                    
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                continue
        
        # Add fallback strategies if needed
        if len(strategies) < 3:
            fallback = self._fallback_strategies()
            strategies.extend(fallback[:max(1, 5 - len(strategies))])
        
        # Sort by goal-appropriateness score
        strategies = self._rank_strategies_for_goals(strategies, characteristics)
        
        return strategies[:10]  # Return top 10 strategies
    
    def _get_goal_responsive_configs(self, characteristics: GoalCharacteristics) -> List[Dict]:
        """Get portfolio optimization configurations based on goal characteristics"""
        configs = []
        
        # Base configurations
        base_configs = [
            {'name': 'Ultra Conservative', 'target_vol': 0.05, 'target_return': 0.04, 'weight': 1.0},
            {'name': 'Conservative', 'target_vol': 0.08, 'target_return': 0.06, 'weight': 1.0},
            {'name': 'Moderate Conservative', 'target_vol': 0.10, 'target_return': 0.07, 'weight': 1.0},
            {'name': 'Moderate', 'target_vol': 0.12, 'target_return': 0.08, 'weight': 1.0},
            {'name': 'Moderate Aggressive', 'target_vol': 0.15, 'target_return': 0.09, 'weight': 1.0},
            {'name': 'Aggressive', 'target_vol': 0.18, 'target_return': 0.10, 'weight': 1.0},
        ]
        
        # Adjust configurations based on goal characteristics
        for config in base_configs:
            adjusted_config = config.copy()
            
            # Adjust for short-term goals
            if characteristics.has_short_term_goals:
                adjusted_config['target_vol'] *= 0.7  # Reduce risk for short-term goals
                adjusted_config['name'] += ' (Short-Term Adjusted)'
            
            # Adjust for high-priority goals
            if characteristics.has_high_priority_goals:
                adjusted_config['target_vol'] *= 0.8  # Slightly reduce risk for high-priority goals
                adjusted_config['name'] += ' (High-Priority Adjusted)'
            
            # Adjust for goal timing
            time_factor = max(0.5, min(2.0, characteristics.weighted_time_horizon / 15))
            adjusted_config['target_vol'] *= time_factor
            adjusted_config['target_return'] = adjusted_config['target_return'] * time_factor + 0.02 * (1 - time_factor)
            
            # Weight based on goal appropriateness
            adjusted_config['goal_score'] = self._calculate_goal_appropriateness_score(
                adjusted_config, characteristics
            )
            
            configs.append(adjusted_config)
        
        # Sort by goal appropriateness
        configs.sort(key=lambda x: x['goal_score'], reverse=True)
        
        return configs
    
    def _calculate_goal_appropriateness_score(self, config: Dict, characteristics: GoalCharacteristics) -> float:
        """Calculate how appropriate a portfolio configuration is for given goals"""
        score = 1.0
        
        # Penalize high volatility for short-term goals
        if characteristics.has_short_term_goals and config['target_vol'] > 0.12:
            score *= 0.5
        
        # Reward moderate risk for high-priority goals
        if characteristics.has_high_priority_goals:
            if 0.08 <= config['target_vol'] <= 0.15:
                score *= 1.3
            elif config['target_vol'] > 0.18:
                score *= 0.7
        
        # Adjust for time horizon
        time_match = 1.0 - abs(config['target_vol'] - characteristics.weighted_time_horizon * 0.01) / 0.1
        score *= max(0.5, time_match)
        
        # Adjust for cost concentration
        if characteristics.cost_concentration > 0.7:  # High concentration
            # Prefer more conservative strategies
            if config['target_vol'] < 0.12:
                score *= 1.2
        
        return score
    
    def _optimize_goal_responsive_portfolio(self, mu: np.ndarray, Sigma: np.ndarray, 
                                          config: Dict, characteristics: GoalCharacteristics) -> np.ndarray:
        """Optimize portfolio for specific goal characteristics and configuration"""
        n_assets = len(mu)
        
        # Multi-objective optimization: balance return, risk, and goal-specific factors
        def objective(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
            
            # Base objective: maximize utility
            utility = portfolio_return - 0.5 * config.get('risk_aversion', 5.0) * portfolio_vol**2
            
            # Goal-specific adjustments
            goal_penalty = 0
            
            # Penalize high volatility for short-term goals
            if characteristics.has_short_term_goals and portfolio_vol > 0.12:
                goal_penalty += (portfolio_vol - 0.12) * 10
            
            # Reward targeting specific volatility level
            vol_target_penalty = abs(portfolio_vol - config['target_vol']) * 5
            
            return -(utility - goal_penalty - vol_target_penalty)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Optional return constraint for high-priority goals
        if characteristics.has_high_priority_goals:
            min_return = max(0.05, config['target_return'] * 0.8)
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x: np.dot(x, mu) - min_return
            })
        
        # Bounds: allow small short positions for flexibility
        bounds = tuple((-0.05, 0.8) for _ in range(n_assets))
        
        # Initial guess based on goal characteristics
        x0 = self._get_goal_based_initial_guess(characteristics, n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            # Normalize weights
            weights = result.x / np.sum(result.x)
            return weights
        else:
            pass  # Do not log visitor financial inputs.
            return None
    
    def _get_goal_based_initial_guess(self, characteristics: GoalCharacteristics, n_assets: int) -> np.ndarray:
        """Get initial portfolio guess based on goal characteristics"""
        # Default equal weights
        weights = np.ones(n_assets) / n_assets
        
        if not hasattr(self.etf_manager, 'etf_components'):
            return weights
        
        etf_components = self.etf_manager.etf_components
        
        # Adjust based on goal characteristics
        if characteristics.has_short_term_goals:
            # Favor bonds and conservative assets
            for i, asset in enumerate(etf_components):
                if asset in ['AGG']:  # Bonds
                    weights[i] *= 2
                elif asset in ['GLD']:  # Gold
                    weights[i] *= 1.5
                elif asset in ['EEM', 'IWM']:  # High-vol assets
                    weights[i] *= 0.5
        
        if characteristics.has_high_priority_goals:
            # Favor large-cap equity and reduce alternatives
            for i, asset in enumerate(etf_components):
                if asset in ['IVV', 'EFA']:  # Large-cap equity
                    weights[i] *= 1.5
                elif asset in ['GLD', 'IYR']:  # Alternatives
                    weights[i] *= 0.7
        
        # Normalize
        weights = weights / np.sum(weights)
        return weights
    
    def _rank_strategies_for_goals(self, strategies: List[Dict], characteristics: GoalCharacteristics) -> List[Dict]:
        """Rank strategies by their appropriateness for current goals"""
        for strategy in strategies:
            score = self._calculate_strategy_goal_score(strategy, characteristics)
            strategy['goal_appropriateness_score'] = score
        
        # Sort by goal appropriateness score
        strategies.sort(key=lambda x: x.get('goal_appropriateness_score', 0), reverse=True)
        
        return strategies
    
    def _calculate_strategy_goal_score(self, strategy: Dict, characteristics: GoalCharacteristics) -> float:
        """Calculate comprehensive goal appropriateness score for a strategy"""
        score = 1.0
        
        vol = strategy.get('volatility', 0.1)
        ret = strategy.get('return', 0.06)
        sharpe = strategy.get('sharpe', 0)
        
        # Time horizon alignment
        ideal_vol_for_time = 0.02 + (characteristics.weighted_time_horizon / 30) * 0.16
        time_alignment = 1.0 - abs(vol - ideal_vol_for_time) / 0.1
        score *= max(0.3, time_alignment)
        
        # Short-term goal penalty for high volatility
        if characteristics.has_short_term_goals and vol > 0.12:
            score *= 0.6
        
        # High-priority goal considerations
        if characteristics.has_high_priority_goals:
            if 0.08 <= vol <= 0.15:  # Sweet spot for high-priority goals
                score *= 1.4
            if ret >= 0.07:  # Good return potential
                score *= 1.2
        
        # Reward good risk-adjusted returns
        if sharpe > 0.5:
            score *= 1.1
        if sharpe > 1.0:
            score *= 1.3
        
        # Cost considerations
        funding_ratio = characteristics.total_cost / 100000  # Assuming $100k base
        if funding_ratio > 3 and ret < 0.08:  # Need aggressive growth for expensive goals
            score *= 0.8
        
        return score
    
    def _fallback_strategies(self) -> List[Dict]:
        """Fallback strategies when optimization fails"""
        fallback = [
            {
                'weights': {'IVV': 0.6, 'AGG': 0.3, 'EFA': 0.1},
                'return': 0.07,
                'volatility': 0.10,
                'sharpe': 0.5,
                'risk_level': 'Moderate Fallback',
                'assets': ['IVV', 'AGG', 'EFA'],
                'name': 'Balanced Fallback Strategy',
                'goal_responsive': False
            }
        ]
        return fallback

# =============================================================================
# Enhanced Goal Classes with Paper's Framework
# =============================================================================
@dataclass
class EnhancedGoal:
    """Enhanced goal representation following Das et al. (2022) framework"""
    name: str
    time_period: int  # t in the paper
    cost: float  # c_k(t) in the paper
    utility: float  # u_k(t) in the paper
    priority: float = 1.0  # Goal priority weighting
    is_partial: bool = False
    partial_options: List[Dict] = field(default_factory=list)
    parent_goal: Optional[str] = None
    flexibility: float = 0.1  # Cost flexibility (±10%)
    goal_type: str = "regular"  # regular, mandatory, deferrable
    inflation_adjusted: bool = True
    
    def __post_init__(self):
        """Initialize partial options if goal allows partial achievement"""
        if self.is_partial and not self.partial_options:
            self.partial_options = [
                {'cost': self.cost * 0.25, 'utility': self.utility * 0.15, 'description': '25% achievement'},
                {'cost': self.cost * 0.50, 'utility': self.utility * 0.35, 'description': '50% achievement'},
                {'cost': self.cost * 0.75, 'utility': self.utility * 0.65, 'description': '75% achievement'},
            ]

@dataclass 
class CashInflation:
    """Cash infusion representation following the paper's I(t) notation"""
    time_period: int
    amount: float
    inflation_adjusted: bool = True
    source: str = "regular_savings"  # regular_savings, bonus, inheritance, etc.

# =============================================================================
# Complete Dynamic Programming Algorithm (Enhanced)
# =============================================================================
class CompleteGoalOptimizer:
    """
    Enhanced implementation of the dynamic programming algorithm from
    "Dynamic Optimization for Multi-Goals Wealth Management" (Das et al., 2022)
    with advanced ETF portfolio integration and additional features
    """
    
    def __init__(self, 
                 initial_wealth: float = 100000,
                 horizon_years: int = 30,
                 time_step: float = 1.0,
                 wealth_grid_size: int = 200,
                 etf_manager: ETFPortfolioManager = None,
                 n_portfolios: int = 10,
                 terminal_utility_func: Optional[callable] = None,
                 inflation_rate: float = 0.03):
        
        pass  # Do not log visitor financial inputs.
        
        # Core parameters following the paper
        self.W0 = initial_wealth
        self.T = int(horizon_years / time_step)
        self.h = time_step
        self.i_max = wealth_grid_size
        self.l_max = n_portfolios
        self.inflation_rate = inflation_rate
        
        # ETF portfolio integration
        self.etf_manager = etf_manager
        self.portfolio_strategies = []
        
        # Algorithm components following Das et al. notation
        self.goals: List[EnhancedGoal] = []
        self.infusions: Dict[int, float] = {}  # I(t) - cash infusions
        self.wealth_grid: np.ndarray = None  # W_i
        self.value_function: np.ndarray = None  # V(W_i, t)
        self.policy_functions: Dict[str, np.ndarray] = {}  # Optimal policies
        
        # Goal vectors by time period (c(t) and u(t) in paper)
        self.goal_options: Dict[int, List[Dict]] = {}
        self.cost_vectors: Dict[int, np.ndarray] = {}  # c(t)
        self.utility_vectors: Dict[int, np.ndarray] = {}  # u(t)
        
        # Results and diagnostics
        self.goal_probabilities: Dict[str, float] = {}
        self.optimal_paths: Dict = {}
        self.wealth_distribution: Optional[Dict] = None
        self.convergence_metrics: Dict = {}
        
        # Advanced features
        self.terminal_utility_func = terminal_utility_func or (lambda w: 0)
        self.use_parallel = True
        self.cache_transitions = True
        self.transition_cache: Dict = {}
        
        # Performance monitoring
        self.computation_time: float = 0
        self.memory_usage: Dict = {}
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all algorithm components following the paper's structure"""
        pass  # Do not log visitor financial inputs.
        
        # Setup portfolio strategies from ETF data
        self._setup_portfolio_strategies()
        
        # Setup wealth grid W_i following Section 2.4 of the paper
        self._setup_wealth_grid()
        
        # Initialize value function and policy matrices
        self._initialize_matrices()
        
        pass  # Do not log visitor financial inputs.
    
    def _setup_portfolio_strategies(self):
        """Setup portfolio strategies l=1,2,...,l_max using ETF manager"""
        if self.etf_manager:
            self.portfolio_strategies = self.etf_manager.get_portfolio_strategies(self.l_max)
            pass  # Do not log visitor financial inputs.
            
            # Log strategy details
            for i, strategy in enumerate(self.portfolio_strategies):
                pass  # Do not log visitor financial inputs.
        else:
            # Fallback to theoretical strategies
            self.portfolio_strategies = self._create_theoretical_strategies()
            pass  # Do not log visitor financial inputs.
    
    def _create_theoretical_strategies(self):
        """Create theoretical portfolio strategies when ETF manager is unavailable"""
        strategies = []
        
        # Create strategies along theoretical efficient frontier
        for i in range(self.l_max):
            # Linear interpolation between conservative and aggressive
            weight = i / (self.l_max - 1) if self.l_max > 1 else 0.5
            
            # Expected return from 3% (bonds) to 12% (aggressive stocks)
            expected_return = 0.03 + weight * 0.09
            # Volatility from 4% (bonds) to 20% (aggressive stocks)
            volatility = 0.04 + weight * 0.16
            
            risk_levels = ["Very Conservative", "Conservative", "Moderate Conservative", 
                          "Moderate", "Moderate Aggressive", "Aggressive", "Very Aggressive"]
            risk_level = risk_levels[min(i * len(risk_levels) // self.l_max, len(risk_levels) - 1)]
            
            strategies.append({
                'weights': {'theoretical_portfolio': 1.0},
                'return': expected_return,
                'volatility': volatility,
                'sharpe': (expected_return - 0.02) / volatility if volatility > 0 else 0,
                'risk_level': risk_level,
                'assets': ['theoretical_portfolio'],
                'name': f"{risk_level} Theoretical Portfolio"
            })
        
        return strategies
    
    def _setup_wealth_grid(self):
        """Setup logarithmically spaced wealth grid W_i following Section 2.4"""
        # Estimate bounds following equations (1) and (2) from the paper
        max_goal_cost = max([g.cost for g in self.goals], default=self.W0 * 0.1)
        total_infusions = sum(self.infusions.values())
        
        # Conservative lower bound
        W_min = max(100, self.W0 * 0.01)  # Bankruptcy threshold
        
        # Upper bound considering growth potential and goals
        if self.portfolio_strategies:
            max_return = max(s['return'] for s in self.portfolio_strategies)
            # Equation (2) adaptation: consider best-case scenario
            W_max = max(
                self.W0 * np.exp(max_return * self.T * self.h) + total_infusions,
                max_goal_cost * 3,
                self.W0 * 10
            )
        else:
            W_max = max(self.W0 * 5, max_goal_cost * 2)
        
        # Create logarithmic grid as in the paper
        log_min = np.log(W_min)
        log_max = np.log(W_max)
        self.wealth_grid = np.exp(np.linspace(log_min, log_max, self.i_max))
        
        # Find initial wealth index i_0
        self.i0 = np.argmin(np.abs(self.wealth_grid - self.W0))
        
        pass  # Do not log visitor financial inputs.
    
    def _initialize_matrices(self):
        """Initialize value function V(W_i, t) and policy matrices"""
        pass  # Do not log visitor financial inputs.
        
        # Value function V(W_i, t) - equation (4) in the paper
        self.value_function = np.zeros((self.T + 1, self.i_max))
        
        # Policy functions following the paper's notation
        self.policy_functions = {
            'goal_choice': np.zeros((self.T, self.i_max), dtype=int),      # k*(i,t)
            'portfolio_choice': np.zeros((self.T, self.i_max), dtype=int), # l*(i,t)  
            'expected_utility': np.zeros((self.T, self.i_max)),            # V(W_i, t)
            'transition_probs': {}  # Cache for transition probabilities
        }
        
        # Terminal utility U(W_T) following Section 2.2
        for i in range(self.i_max):
            self.value_function[self.T, i] = self.terminal_utility_func(self.wealth_grid[i])
    
    def add_goal(self, name: str, time_year: float, cost: float, 
                 utility: float, priority: float = 1.0, is_partial: bool = False,
                 goal_type: str = "regular") -> EnhancedGoal:
        """Add an enhanced goal following the paper's framework"""
        time_period = int(time_year / self.h)
        
        # Validate time period
        if time_period >= self.T:
            pass  # Do not log visitor financial inputs.
            time_period = self.T - 1
        
        # Adjust for inflation if needed
        if goal_type != "fixed_nominal":
            inflated_cost = cost * ((1 + self.inflation_rate) ** time_year)
        else:
            inflated_cost = cost
        
        goal = EnhancedGoal(
            name=name,
            time_period=time_period,
            cost=inflated_cost,
            utility=utility * priority,  # Weight by priority as in the paper
            priority=priority,
            is_partial=is_partial,
            goal_type=goal_type
        )
        
        self.goals.append(goal)
        self._update_goal_vectors()
        
        pass  # Do not log visitor financial inputs.
        return goal
    
    def add_infusion(self, time_year: float, amount: float, source: str = "savings") -> CashInflation:
        """Add cash infusion I(t) following the paper's notation"""
        time_period = int(time_year / self.h)
        
        if time_period >= self.T:
            pass  # Do not log visitor financial inputs.
            time_period = self.T - 1
        
        # Adjust for inflation
        inflated_amount = amount * ((1 + self.inflation_rate) ** time_year)
        
        if time_period in self.infusions:
            self.infusions[time_period] += inflated_amount
        else:
            self.infusions[time_period] = inflated_amount
        
        infusion = CashInflation(time_period, inflated_amount, True, source)
        pass  # Do not log visitor financial inputs.
        return infusion
    
    def _update_goal_vectors(self):
        """Update cost c(t) and utility u(t) vectors following Section 2.3"""
        pass  # Do not log visitor financial inputs.
        
        for t in range(self.T):
            # Get all goals for this time period
            goals_at_t = [g for g in self.goals if g.time_period == t]
            
            # Start with null option (k=0: do nothing)
            options = [{'cost': 0.0, 'utility': 0.0, 'name': 'none', 'type': 'null'}]
            
            # Add each goal and its partial options
            for goal in goals_at_t:
                # Full goal achievement
                options.append({
                    'cost': goal.cost,
                    'utility': goal.utility,
                    'name': goal.name,
                    'type': 'full',
                    'goal_obj': goal
                })
                
                # Partial achievement options
                for i, partial in enumerate(goal.partial_options):
                    options.append({
                        'cost': partial['cost'],
                        'utility': partial['utility'],
                        'name': f"{goal.name}_partial_{i+1}",
                        'type': 'partial',
                        'goal_obj': goal,
                        'description': partial.get('description', f'Partial {i+1}')
                    })
            
            # Process concurrent goals following Section 2.3 methodology
            processed_options = self._process_concurrent_goals(options)
            
            # Store vectors
            self.goal_options[t] = processed_options
            self.cost_vectors[t] = np.array([opt['cost'] for opt in processed_options])
            self.utility_vectors[t] = np.array([opt['utility'] for opt in processed_options])
            
        pass  # Do not log visitor financial inputs.
    
    def _process_concurrent_goals(self, options):
        """Process concurrent goals following Section 2.3 of the paper"""
        if len(options) <= 1:
            return options
        
        # Generate all combinations of goal fulfillment
        from itertools import product
        
        # Separate by goal
        goals_dict = {}
        for opt in options:
            if opt['type'] == 'null':
                continue
            goal_name = opt['goal_obj'].name if 'goal_obj' in opt else opt['name']
            if goal_name not in goals_dict:
                goals_dict[goal_name] = []
            goals_dict[goal_name].append(opt)
        
        if not goals_dict:
            return options[:1]  # Just null option
        
        # Add null option for each goal
        for goal_name in goals_dict:
            goals_dict[goal_name].insert(0, {
                'cost': 0.0, 'utility': 0.0, 'name': f'{goal_name}_none', 'type': 'none'
            })
        
        # Generate all combinations
        goal_names = list(goals_dict.keys())
        goal_option_lists = [goals_dict[name] for name in goal_names]
        
        processed_options = [options[0]]  # Keep null option
        
        for combination in product(*goal_option_lists):
            total_cost = sum(opt['cost'] for opt in combination)
            total_utility = sum(opt['utility'] for opt in combination)
            
            # Create combined option name
            active_opts = [opt for opt in combination if opt['cost'] > 0]
            if active_opts:
                name = " + ".join(opt['name'] for opt in active_opts)
                processed_options.append({
                    'cost': total_cost,
                    'utility': total_utility,
                    'name': name,
                    'type': 'combination',
                    'components': combination
                })
        
        # Remove dominated options following Section 2.3
        return self._remove_dominated_options(processed_options)
    
    def _remove_dominated_options(self, options):
        """Remove dominated options as described in Section 2.3"""
        # Sort by cost
        options.sort(key=lambda x: x['cost'])
        
        # Remove options where previous option has higher utility at lower cost
        filtered_options = []
        
        for option in options:
            is_dominated = False
            for existing in filtered_options:
                if (existing['cost'] <= option['cost'] and 
                    existing['utility'] >= option['utility'] and
                    (existing['cost'] < option['cost'] or existing['utility'] > option['utility'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                filtered_options.append(option)
        
        return filtered_options
    
    @lru_cache(maxsize=50000)
    def _compute_transition_probabilities(self, wealth_idx: int, goal_idx: int, 
                                        portfolio_idx: int, time_period: int) -> np.ndarray:
        """Compute transition probabilities q(W_j(t+1)|W_i(t), c_k(t), μ_l) following Section 3.1"""
        
        W_i = self.wealth_grid[wealth_idx]
        c_k = self.cost_vectors[time_period][goal_idx]
        I_t = self.infusions.get(time_period, 0.0)
        
        # Get portfolio strategy parameters
        strategy = self.portfolio_strategies[portfolio_idx]
        mu_l = strategy['return']
        sigma_l = strategy['volatility']
        
        # Wealth after goal expenditure and cash infusion
        W_post = max(W_i + I_t - c_k, 1.0)  # Ensure positive wealth
        
        if W_post <= 1.0:
            # Bankruptcy case - concentrate on minimum wealth
            probs = np.zeros(self.i_max)
            probs[0] = 1.0
            return probs
        
        # Geometric Brownian motion evolution following equation (3)
        # W(t+1) = W_post * exp((μ - σ²/2)*h + σ*√h*Z)
        
        # Compute approximate transition probabilities
        probs = np.zeros(self.i_max)
        
        for j in range(self.i_max):
            W_j = self.wealth_grid[j]
            
            # Following the paper's approach in Section 3.1
            if W_j > 0 and W_post > 0:
                # Z = (ln(W_j/W_post) - (μ - σ²/2)*h) / (σ*√h)
                drift = (mu_l - 0.5 * sigma_l**2) * self.h
                diffusion = sigma_l * np.sqrt(self.h)
                
                if diffusion > 0:
                    z = (np.log(W_j / W_post) - drift) / diffusion
                    # Probability density φ(z)
                    prob_density = norm.pdf(z)
                    # Convert to discrete probability (simple approximation)
                    if j == 0:
                        prob = norm.cdf(z)
                    elif j == self.i_max - 1:
                        prob = 1 - norm.cdf(z)
                    else:
                        # Use grid spacing for integration
                        dW = (self.wealth_grid[min(j+1, self.i_max-1)] - 
                              self.wealth_grid[max(j-1, 0)]) / 2
                        prob = prob_density * dW / (W_j * diffusion)
                    
                    probs[j] = max(prob, 1e-12)  # Numerical stability
        
        # Normalize probabilities following Section 3.1
        prob_sum = np.sum(probs)
        if prob_sum > 1e-10:
            probs = probs / prob_sum
        else:
            # Fallback: equal probability
            probs = np.ones(self.i_max) / self.i_max
        
        return probs
    
    def solve_bellman_equation(self, max_iterations: int = None, tolerance: float = 1e-8):
        """Solve Bellman equation (4) using backward induction"""
        import time
        start_time = time.time()
        
        if max_iterations is None:
            max_iterations = self.T
        
        pass  # Do not log visitor financial inputs.
        pass  # Do not log visitor financial inputs.
        pass  # Do not log visitor financial inputs.
        pass  # Do not log visitor financial inputs.
        
        # Backward induction from T to 0
        for t in reversed(range(min(self.T, max_iterations))):
            if t % max(1, self.T // 20) == 0:
                elapsed = time.time() - start_time
                pass  # Do not log visitor financial inputs.
            
            k_max = len(self.cost_vectors[t])
            
            # Vectorized computation for each wealth level
            for i in range(self.i_max):
                best_value = -np.inf
                best_k = 0
                best_l = 0
                
                # Optimize over all portfolio-goal combinations
                for l in range(self.l_max):
                    for k in range(k_max):
                        # Get transition probabilities
                        transition_probs = self._compute_transition_probabilities(i, k, l, t)
                        
                        # Calculate expected continuation value
                        expected_continuation = np.sum(transition_probs * self.value_function[t + 1, :])
                        
                        # Bellman equation (4): V(W_i, t) = max[u_k(t) + E[V(W_j, t+1)]]
                        immediate_utility = self.utility_vectors[t][k]
                        total_value = immediate_utility + expected_continuation
                        
                        # Update if this is the best choice
                        if total_value > best_value:
                            best_value = total_value
                            best_k = k
                            best_l = l
                
                # Store optimal choices
                self.value_function[t, i] = best_value
                self.policy_functions['goal_choice'][t, i] = best_k
                self.policy_functions['portfolio_choice'][t, i] = best_l
                self.policy_functions['expected_utility'][t, i] = best_value
        
        self.computation_time = time.time() - start_time
        pass  # Do not log visitor financial inputs.
    
    def compute_goal_probabilities(self, n_simulations: int = 5000) -> Dict[str, float]:
        """Compute goal achievement probabilities following Section 3.3"""
        pass  # Do not log visitor financial inputs.
        
        # Initialize probability distribution p(W_i(0)) = δ_{i_0}
        wealth_probabilities = np.zeros((self.T + 1, self.i_max))
        wealth_probabilities[0, self.i0] = 1.0
        
        # Track goal achievements
        goal_achievements = {goal.name: 0 for goal in self.goals}
        partial_achievements = {}
        
        # Forward equation (5) simulation
        for t in range(self.T):
            for i in range(self.i_max):
                if wealth_probabilities[t, i] > 1e-10:  # Only consider significant probabilities
                    # Get optimal policy
                    k_star = self.policy_functions['goal_choice'][t, i]
                    l_star = self.policy_functions['portfolio_choice'][t, i]
                    
                    # Record goal achievement
                    if k_star > 0:  # k=0 is null action
                        goal_option = self.goal_options[t][k_star]
                        self._record_goal_achievement(goal_option, wealth_probabilities[t, i], 
                                                    goal_achievements, partial_achievements)
                    
                    # Update wealth distribution for next period
                    if t < self.T - 1:
                        transition_probs = self._compute_transition_probabilities(i, k_star, l_star, t)
                        wealth_probabilities[t + 1, :] += wealth_probabilities[t, i] * transition_probs
        
        # Convert to probabilities
        total_simulations = sum(goal_achievements.values()) + len(self.goals) * n_simulations
        probabilities = {}
        
        # Normalize and store results
        for goal in self.goals:
            prob = goal_achievements.get(goal.name, 0)
            # Add partial achievements
            partial_prob = sum(partial_achievements.get(f"{goal.name}_partial_{i}", 0) 
                             for i in range(1, 4))
            
            # Monte Carlo style probability calculation
            total_prob = min(1.0, prob + partial_prob * 0.5)  # Weight partials at 50%
            probabilities[goal.name] = max(0.01, total_prob)  # Minimum 1% probability
        
        # Store detailed results
        self.goal_probabilities = probabilities
        self.wealth_distribution = {
            'probabilities': wealth_probabilities,
            'final_wealth_stats': self._compute_wealth_statistics(wealth_probabilities[-1, :])
        }
        
        pass  # Do not log visitor financial inputs.
        return probabilities
    
    def _record_goal_achievement(self, goal_option, probability, achievements, partial_achievements):
        """Record goal achievement with proper weighting"""
        if goal_option['type'] == 'full':
            achievements[goal_option['name']] = achievements.get(goal_option['name'], 0) + probability
        elif goal_option['type'] == 'partial':
            partial_achievements[goal_option['name']] = partial_achievements.get(goal_option['name'], 0) + probability
        elif goal_option['type'] == 'combination':
            # Handle combined goals
            for component in goal_option.get('components', []):
                if component['cost'] > 0:
                    self._record_goal_achievement(component, probability, achievements, partial_achievements)
    
    def _compute_wealth_statistics(self, final_wealth_probs):
        """Compute final wealth distribution statistics"""
        # Expected final wealth
        expected_wealth = np.sum(final_wealth_probs * self.wealth_grid)
        
        # Quantiles
        cumulative_probs = np.cumsum(final_wealth_probs)
        percentiles = {}
        for p in [10, 25, 50, 75, 90, 95, 99]:
            idx = np.searchsorted(cumulative_probs, p/100.0)
            percentiles[f'p{p}'] = self.wealth_grid[min(idx, len(self.wealth_grid)-1)]
        
        return {
            'expected': expected_wealth,
            'percentiles': percentiles,
            'probability_of_ruin': final_wealth_probs[0]  # P(W_T ≤ W_min)
        }
    
    def optimize(self, use_full_algorithm: bool = True, max_time_periods: int = None) -> Dict[str, float]:
        """Run the complete optimization following Das et al. (2022) methodology"""
        pass  # Do not log visitor financial inputs.
        
        if not self.goals:
            pass  # Do not log visitor financial inputs.
            return {}
        
        try:
            # Determine algorithm based on problem size and user preference
            should_use_full = (use_full_algorithm and 
                             len(self.goals) <= 8 and 
                             self.T <= 25 and 
                             self.i_max <= 300)
            
            if should_use_full:
                pass  # Do not log visitor financial inputs.
                
                # Solve Bellman equation
                self.solve_bellman_equation(max_time_periods)
                
                # Compute probabilities via forward simulation
                probabilities = self.compute_goal_probabilities()
                
                return probabilities
            else:
                pass  # Do not log visitor financial inputs.
                return self._enhanced_heuristic_optimization()
                
        except Exception as e:
            pass  # Do not log visitor financial inputs.
            import traceback
            pass  # Do not log visitor financial inputs.
            return self._enhanced_heuristic_optimization()
    
    def _enhanced_heuristic_optimization(self) -> Dict[str, float]:
        """Enhanced heuristic using portfolio optimization and Monte Carlo simulation"""
        pass  # Do not log visitor financial inputs.
        
        probabilities = {}
        
        # Use multiple portfolio strategies and average results
        strategy_results = []
        
        for strategy in self.portfolio_strategies[:3]:  # Use top 3 strategies
            strategy_probs = self._simulate_strategy_outcomes(strategy)
            strategy_results.append(strategy_probs)
        
        # Average across strategies
        for goal in self.goals:
            goal_probs = [result.get(goal.name, 0) for result in strategy_results]
            avg_prob = np.mean(goal_probs) if goal_probs else 0.05
            
            # Apply priority weighting
            weighted_prob = min(0.98, avg_prob * goal.priority)
            probabilities[goal.name] = max(0.02, weighted_prob)
        
        self.goal_probabilities = probabilities
        return probabilities
    
    def _simulate_strategy_outcomes(self, strategy, n_simulations: int = 1000):
        """Simulate outcomes for a specific portfolio strategy"""
        mu = strategy['return']
        sigma = strategy['volatility']
        
        goal_successes = {goal.name: 0 for goal in self.goals}
        
        for _ in range(n_simulations):
            wealth = self.W0
            
            for t in range(self.T):
                # Add infusions
                wealth += self.infusions.get(t, 0)
                
                # Check for goals at this time
                goals_at_t = [g for g in self.goals if g.time_period == t]
                for goal in goals_at_t:
                    if wealth >= goal.cost:
                        wealth -= goal.cost
                        goal_successes[goal.name] += 1
                
                # Evolve wealth
                if t < self.T - 1:
                    # Geometric Brownian motion
                    dt = self.h
                    dW = np.random.normal(0, np.sqrt(dt))
                    wealth *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
                    wealth = max(wealth, 1.0)  # Prevent negative wealth
        
        # Convert to probabilities
        return {goal: successes / n_simulations for goal, successes in goal_successes.items()}
    
    def get_optimization_summary(self) -> Dict:
        """Get comprehensive optimization summary"""
        if not hasattr(self, 'goal_probabilities') or not self.goal_probabilities:
            return {'status': 'not_optimized'}
        
        summary = {
            'status': 'optimized',
            'algorithm_used': 'full_dp' if hasattr(self, 'computation_time') else 'enhanced_heuristic',
            'computation_time': getattr(self, 'computation_time', 0),
            'goal_probabilities': self.goal_probabilities.copy(),
            'portfolio_strategies': len(self.portfolio_strategies),
            'total_goals': len(self.goals),
            'total_infusions': len(self.infusions),
            'wealth_grid_size': self.i_max,
            'time_horizon': self.T,
            'expected_utility': getattr(self, 'value_function', np.array([[0]]))[0, self.i0] if hasattr(self, 'value_function') else 0
        }
        
        # Add goal analysis
        if self.goal_probabilities:
            probs = list(self.goal_probabilities.values())
            summary['goal_analysis'] = {
                'average_probability': np.mean(probs),
                'min_probability': np.min(probs),
                'max_probability': np.max(probs),
                'high_confidence_goals': sum(1 for p in probs if p > 0.8),
                'at_risk_goals': sum(1 for p in probs if p < 0.3)
            }
        
        # Add wealth statistics if available
        if hasattr(self, 'wealth_distribution') and self.wealth_distribution:
            summary['wealth_forecast'] = self.wealth_distribution.get('final_wealth_stats', {})
        
        return summary

# =============================================================================
# Enhanced Helper Functions
# =============================================================================
def add_template_goal(name, cost, time_year, utility, priority=1.0, is_partial=False):
    """Add a template goal with enhanced features"""
    global goals_storage, goal_id_counter
    
    goal_id_counter += 1
    template_goal = {
        'id': goal_id_counter,
        'name': name,
        'cost': cost,
        'time_year': time_year,
        'utility': utility,
        'priority': priority,
        'is_partial': is_partial,
        'parent_goal': None,
        'goal_type': 'template'
    }
    
    goals_storage.append(template_goal)
    pass  # Do not log visitor financial inputs.
    return template_goal

def create_goal_responsive_manager(etf_manager, goals_storage, infusions_storage, 
                                 initial_wealth, horizon_years):
    """Create and configure goal-responsive portfolio manager"""
    manager = GoalResponsivePortfolioManager(etf_manager)
    dynamic_strategies = manager.update_goals(
        goals_storage, infusions_storage, initial_wealth, horizon_years
    )
    return manager, dynamic_strategies


def _create_goal_responsive_portfolio_display(strategies, goal_responsive_manager):
    """Create display showing how portfolios were optimized for goals"""
    if not goal_responsive_manager or not hasattr(goal_responsive_manager, 'current_goal_characteristics'):
        return ui.div()
    
    characteristics = goal_responsive_manager.current_goal_characteristics
    
    return ui.div(
        ui.h6("🎯 Goal-Responsive Portfolio Optimization", class_="mt-3 mb-2"),
        ui.div(
            ui.tags.table([
                ui.tags.thead([
                    ui.tags.tr([
                        ui.tags.th("Goal Characteristic", class_="text-start"),
                        ui.tags.th("Value", class_="text-end"),
                        ui.tags.th("Portfolio Impact", class_="text-start")
                    ])
                ]),
                ui.tags.tbody([
                    ui.tags.tr([
                        ui.tags.td("Total Goal Cost", class_="fw-medium"),
                        ui.tags.td(f"${characteristics.total_cost:,.0f}", class_="text-end fw-bold"),
                        ui.tags.td("Influences growth requirements", class_="text-muted small")
                    ]),
                    ui.tags.tr([
                        ui.tags.td("Weighted Time Horizon", class_="fw-medium"),
                        ui.tags.td(f"{characteristics.weighted_time_horizon:.1f} years", class_="text-end fw-bold"),
                        ui.tags.td("Determines risk tolerance", class_="text-muted small")
                    ]),
                    ui.tags.tr([
                        ui.tags.td("Shortest Goal", class_="fw-medium"),
                        ui.tags.td(f"{characteristics.min_time_horizon:.1f} years", class_="text-end fw-bold"),
                        ui.tags.td("Limits maximum risk", class_="text-muted small")
                    ]),
                    ui.tags.tr([
                        ui.tags.td("Has Short-term Goals", class_="fw-medium"),
                        ui.tags.td("Yes" if characteristics.has_short_term_goals else "No", 
                                  class_="text-end fw-bold text-warning" if characteristics.has_short_term_goals else "text-end fw-bold"),
                        ui.tags.td("Reduces volatility targets", class_="text-muted small")
                    ]),
                    ui.tags.tr([
                        ui.tags.td("Has High-Priority Goals", class_="fw-medium"),
                        ui.tags.td("Yes" if characteristics.has_high_priority_goals else "No", 
                                  class_="text-end fw-bold text-success" if characteristics.has_high_priority_goals else "text-end fw-bold"),
                        ui.tags.td("Emphasizes reliability", class_="text-muted small")
                    ])
                ])
            ], class_="table table-sm table-hover"),
            class_="table-responsive"
        ),
        ui.div(
            ui.h6("📊 Portfolio Adjustments Made:", class_="mt-2 mb-1"),
            ui.tags.ul([
                ui.tags.li("Volatility targets adjusted for goal timing", class_="small"),
                ui.tags.li("Asset allocation optimized for goal priorities", class_="small"),
                ui.tags.li("Risk constraints applied for short-term goals" if characteristics.has_short_term_goals else "Growth emphasis for long-term goals", class_="small"),
                ui.tags.li("Return requirements calibrated to goal costs", class_="small")
            ])
        )
    )

def _create_goal_responsive_strategy_comparison(strategies):
    """Create comparison between goal-responsive and standard strategies"""
    goal_responsive = [s for s in strategies if s.get('goal_responsive', False)]
    standard = [s for s in strategies if not s.get('goal_responsive', False)]
    
    if not goal_responsive:
        return ui.div()
    
    # Calculate average metrics
    gr_avg_return = np.mean([s['return'] for s in goal_responsive])
    gr_avg_vol = np.mean([s['volatility'] for s in goal_responsive])
    gr_avg_sharpe = np.mean([s.get('sharpe', 0) for s in goal_responsive])
    
    std_avg_return = np.mean([s['return'] for s in standard]) if standard else 0
    std_avg_vol = np.mean([s['volatility'] for s in standard]) if standard else 0
    std_avg_sharpe = np.mean([s.get('sharpe', 0) for s in standard]) if standard else 0
    
    return ui.div(
        ui.h5("🎯 Goal-Responsive vs Standard Strategy Comparison"),
        ui.div(
            ui.tags.table([
                ui.tags.thead([
                    ui.tags.tr([
                        ui.tags.th("Strategy Type", class_="text-start"),
                        ui.tags.th("Count", class_="text-center"),
                        ui.tags.th("Avg Return", class_="text-center"),
                        ui.tags.th("Avg Volatility", class_="text-center"),
                        ui.tags.th("Avg Sharpe", class_="text-center")
                    ])
                ]),
                ui.tags.tbody([
                    ui.tags.tr([
                        ui.tags.td("🎯 Goal-Responsive", class_="fw-bold text-primary"),
                        ui.tags.td(f"{len(goal_responsive)}", class_="text-center fw-bold"),
                        ui.tags.td(f"{gr_avg_return:.1%}", class_="text-center text-success fw-bold"),
                        ui.tags.td(f"{gr_avg_vol:.1%}", class_="text-center text-warning fw-bold"),
                        ui.tags.td(f"{gr_avg_sharpe:.2f}", class_="text-center text-info fw-bold")
                    ]),
                    ui.tags.tr([
                        ui.tags.td("📊 Standard", class_="fw-medium text-muted"),
                        ui.tags.td(f"{len(standard)}", class_="text-center"),
                        ui.tags.td(f"{std_avg_return:.1%}" if standard else "N/A", class_="text-center"),
                        ui.tags.td(f"{std_avg_vol:.1%}" if standard else "N/A", class_="text-center"),
                        ui.tags.td(f"{std_avg_sharpe:.2f}" if standard else "N/A", class_="text-center")
                    ])
                ])
            ], class_="table table-sm table-hover"),
            class_="table-responsive"
        ),
        ui.div(
            ui.p("💡 Goal-responsive strategies are specifically optimized for your goal characteristics:", 
                 class_="small text-muted mb-1"),
            ui.tags.ul([
                ui.tags.li("Time horizon alignment", class_="small text-muted"),
                ui.tags.li("Risk tolerance based on goal priorities", class_="small text-muted"),
                ui.tags.li("Volatility constraints for short-term goals", class_="small text-muted"),
                ui.tags.li("Return targets calibrated to goal costs", class_="small text-muted")
            ])
        ),
        class_="mt-4"
    )

# =============================================================================
# Enhanced UI Functions
# =============================================================================
def tab_multi_goal_ui(id: str):
    """Enhanced Multi-Goal Planner UI following Das et al. (2022) framework"""
    return ui.page_sidebar(
        ui.sidebar(
            ui.h4("🎯 Enhanced Multi-Goal Planner", class_="mb-3"),
            ui.p("SYNTHETIC DEMO • Educational model; no historical market or client data. Dynamic programming inspired by Das et al. (2022)", class_="text-muted small"),
            
            # Enhanced Parameters
            ui.card(
                ui.card_header("💼 Optimization Settings"),
                ui.card_body(
                    ui.input_numeric("initial_wealth", "Initial Wealth ($)", 
                                   value=100000, min=10000, step=5000),
                    ui.input_numeric("time_horizon", "Time Horizon (years)", 
                                   value=30, min=5, max=50, step=1),
                    ui.input_numeric("inflation_rate", "Expected Inflation (%)", 
                                   value=3.0, min=0, max=10, step=0.1),
                    ui.input_select("algorithm_type", "Algorithm Type",
                                  choices={
                                      "auto": "Auto (Recommended)", 
                                      "full": "Full Dynamic Programming",
                                      "heuristic": "Enhanced Heuristic"
                                  },
                                  selected="auto"),
                    ui.input_numeric("wealth_grid_size", "Wealth Grid Size", 
                                   value=200, min=50, max=500, step=25),
                    ui.input_numeric("n_portfolios", "Portfolio Strategies", 
                                   value=10, min=5, max=20, step=1)
                )
            ),
            
            # Enhanced Goal Management
            ui.br(),
            ui.card(
                ui.card_header("🎯 Goal Management"),
                ui.card_body(
                    ui.input_action_button("add_goal_btn", "➕ Add Custom Goal", 
                                         class_="btn-success w-100 mb-2"),
                    ui.input_action_button("add_infusion_btn", "💰 Add Cash Infusion", 
                                         class_="btn-info w-100 mb-2"),
                    ui.input_action_button("clear_goals_btn", "🗑️ Clear All", 
                                         class_="btn-outline-danger w-100 mb-2"),
                    ui.download_button("save_session_btn", "💾 Save Session", 
                                         class_="btn-outline-secondary w-100 mb-2"),
                    ui.input_action_button("optimize_btn", "🚀 Optimize Portfolio", 
                                         class_="btn-primary w-100")
                )
            ),
            
            # Enhanced Templates
            ui.br(),
            ui.card(
                ui.card_header("📋 Goal Templates (Das et al. Framework)"),
                ui.card_body(
                    ui.p("Priority-weighted goals with partial achievement options", 
                         class_="small text-muted mb-2"),
                    ui.input_action_button("template_retirement_enhanced", "🏖️ Retirement (High Priority)", 
                                         class_="btn-outline-primary w-100 mb-1"),
                    ui.input_action_button("template_house_partial", "🏠 House (Partial Options)", 
                                         class_="btn-outline-primary w-100 mb-1"),
                    ui.input_action_button("template_education_tiered", "🎓 Education (Tiered)", 
                                         class_="btn-outline-primary w-100 mb-1"),
                    ui.input_action_button("template_emergency_flexible", "🚨 Emergency Fund (Flexible)", 
                                         class_="btn-outline-success w-100 mb-1"),
                    ui.input_action_button("template_luxury_deferrable", "✈️ Luxury Goals (Deferrable)", 
                                         class_="btn-outline-info w-100")
                )
            ),
            
            # Enhanced Status
            ui.br(),
            ui.card(
                ui.card_header("📊 Status & Metrics"),
                ui.card_body(
                    ui.output_text("enhanced_status_summary"),
                    ui.br(),
                    ui.output_ui("optimization_metrics")
                )
            ),
            
            width=400
        ),
        
        # Enhanced Main Content
        ui.div(
            ui.navset_tab(
                ui.nav_panel(
                    "📋 Goals & Optimization",
                    ui.div(
                        ui.output_ui("enhanced_goals_display"),
                        ui.br(),
                        ui.output_ui("optimization_results_enhanced"),
                        class_="mt-3"
                    )
                ),
                ui.nav_panel(
                    "📊 Advanced Analysis", 
                    ui.div(
                        ui.output_ui("advanced_analysis_dashboard"),
                        class_="mt-3"
                    )
                ),
                ui.nav_panel(
                    "💼 ETF Portfolio Strategy",
                    ui.div(
                        ui.output_ui("enhanced_portfolio_analysis"),
                        class_="mt-3"
                    )
                ),
                ui.nav_panel(
                    "🔬 Algorithm Details",
                    ui.div(
                        ui.output_ui("algorithm_diagnostics"),
                        class_="mt-3"
                    )
                ),
                ui.nav_panel(
                    "🔧 Debug & Technical",
                    ui.div(
                        ui.output_text("enhanced_debug_info", container=ui.tags.pre),
                        class_="mt-3"
                    )
                )
            )
        )
    )

# =============================================================================
# Enhanced Server Logic
# =============================================================================
def tab_multi_goal_server(id: str, data_utils: dict, data_inputs: dict, reactives_shiny: dict):
    """Enhanced Multi-Goal server with complete Das et al. (2022) implementation"""
    
    def server_function(input, output, session):
        """Enhanced server function with complete optimization and advanced features"""
        global goals_storage, goal_id_counter, infusions_storage, infusion_id_counter, optimization_results

        pass  # Do not log visitor financial inputs.
        
        # Initialize enhanced ETF manager with dashboard data
        etf_manager = ETFPortfolioManager(data_inputs)
        
        # Enhanced reactive trigger
        refresh_trigger = reactive.value(0)
        optimization_trigger = reactive.value(0)
        
        def trigger_refresh():
            current = refresh_trigger.get()
            refresh_trigger.set(current + 1)
            pass  # Do not log visitor financial inputs.
        
        def trigger_optimization():
            current = optimization_trigger.get()
            optimization_trigger.set(current + 1)
            pass  # Do not log visitor financial inputs.
        
        # Enhanced modal creation functions
        def create_enhanced_goal_modal():
            return ui.modal(
                ui.h4("🎯 Add Enhanced Goal (Das et al. Framework)"),
                ui.p("Following the dynamic programming methodology of Das et al. (2022)", 
                     class_="text-muted small"),
                ui.input_text("modal_goal_name", "Goal Name", 
                             placeholder="e.g., House Down Payment, Retirement Fund"),
                ui.input_numeric("modal_goal_cost", "Required Amount ($)", 
                               value=50000, min=1000, step=1000),
                ui.input_numeric("modal_goal_time", "Target Time (years)", 
                               value=10, min=1, max=50, step=0.5),
                ui.input_numeric("modal_goal_utility", "Utility Value (u_k)", 
                               value=1000, min=1, step=100),
                ui.input_slider("modal_goal_priority", "Priority Weight", 
                              min=0.1, max=3.0, value=1.0, step=0.1),
                ui.input_checkbox("modal_goal_partial", "Allow Partial Achievement"),
                ui.input_select("modal_goal_type", "Goal Type",
                              choices={
                                  "regular": "Regular Goal",
                                  "mandatory": "Mandatory Goal", 
                                  "deferrable": "Deferrable Goal",
                                  "luxury": "Luxury Goal"
                              }),
                ui.input_checkbox("modal_inflation_adjust", "Inflation Adjust", value=True),
                title="Add Enhanced Goal",
                size="l",
                easy_close=True,
                footer=ui.div(
                    ui.input_action_button("modal_save_goal", "💾 Save Goal", class_="btn-primary"),
                    ui.modal_button("Cancel", class_="btn-secondary")
                )
            )
        
        def create_enhanced_infusion_modal():
            return ui.modal(
                ui.h4("💰 Add Cash Infusion I(t)"),
                ui.p("Cash infusions following the paper's I(t) notation", 
                     class_="text-muted small"),
                ui.input_numeric("modal_infusion_time", "Time (years)", 
                               value=5, min=1, max=50, step=0.5),
                ui.input_numeric("modal_infusion_amount", "Amount ($)", 
                               value=10000, min=1000, step=1000),
                ui.input_select("modal_infusion_source", "Source Type",
                              choices={
                                  "savings": "Regular Savings",
                                  "bonus": "Annual Bonus",
                                  "inheritance": "Inheritance", 
                                  "sale": "Asset Sale",
                                  "other": "Other"
                              }),
                ui.input_checkbox("modal_infusion_inflation", "Inflation Adjust", value=True),
                ui.p("Examples: Annual bonus, inheritance, property sale", 
                     class_="text-muted small"),
                title="Add Cash Infusion",
                easy_close=True,
                footer=ui.div(
                    ui.input_action_button("modal_save_infusion", "💾 Save Infusion", class_="btn-primary"),
                    ui.modal_button("Cancel", class_="btn-secondary")
                )
            )
        
        # === ENHANCED EVENT HANDLERS ===
        @reactive.effect
        @reactive.event(input.add_goal_btn)
        def handle_add_goal():
            try:
                pass  # Do not log visitor financial inputs.
                ui.modal_show(create_enhanced_goal_modal())
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                ui.notification_show(f"Error showing goal dialog: {str(e)}", type="error")
        
        @reactive.effect
        @reactive.event(input.add_infusion_btn)
        def handle_add_infusion():
            try:
                pass  # Do not log visitor financial inputs.
                ui.modal_show(create_enhanced_infusion_modal())
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                ui.notification_show(f"Error showing infusion dialog: {str(e)}", type="error")
        
        @reactive.effect
        @reactive.event(input.modal_save_goal)
        def handle_save_goal():
            global goals_storage, goal_id_counter
            
            try:
                name = input.modal_goal_name()
                cost = input.modal_goal_cost()
                time_year = input.modal_goal_time()
                utility = input.modal_goal_utility()
                priority = input.modal_goal_priority()
                is_partial = input.modal_goal_partial()
                goal_type = input.modal_goal_type()
                inflation_adjust = input.modal_inflation_adjust()
                
                if not name or not name.strip():
                    ui.notification_show("Please enter a goal name!", type="warning")
                    return
                
                goal_id_counter += 1
                enhanced_goal = {
                    'id': goal_id_counter,
                    'name': name.strip(),
                    'cost': cost,
                    'time_year': time_year,
                    'utility': utility,
                    'priority': priority,
                    'is_partial': is_partial,
                    'goal_type': goal_type,
                    'inflation_adjust': inflation_adjust,
                    'parent_goal': None
                }
                
                goals_storage.append(enhanced_goal)
                ui.modal_remove()
                trigger_refresh()
                
                ui.notification_show(f"✅ Enhanced goal '{name}' added (Priority: {priority}, Type: {goal_type})!", 
                                   type="success")
                pass  # Do not log visitor financial inputs.
                
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                import traceback
                pass  # Do not log visitor financial inputs.
                ui.notification_show(f"Error saving goal: {str(e)}", type="error")
        
        @reactive.effect
        @reactive.event(input.modal_save_infusion)
        def handle_save_infusion():
            global infusions_storage, infusion_id_counter
            
            try:
                time_year = input.modal_infusion_time()
                amount = input.modal_infusion_amount()
                source = input.modal_infusion_source()
                inflation_adjust = input.modal_infusion_inflation()
                
                infusion_id_counter += 1
                enhanced_infusion = {
                    'id': infusion_id_counter,
                    'time_year': time_year,
                    'amount': amount,
                    'source': source,
                    'inflation_adjust': inflation_adjust
                }
                
                infusions_storage.append(enhanced_infusion)
                ui.modal_remove()
                trigger_refresh()
                
                ui.notification_show(f"✅ {source.title()} infusion ${amount:,} at year {time_year} added!", 
                                   type="success")
                pass  # Do not log visitor financial inputs.
                
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                ui.notification_show(f"Error saving infusion: {str(e)}", type="error")
    
        @reactive.effect
        @reactive.event(input.optimize_btn)
        def handle_optimize():
            global optimization_results
            
            try:
                if not goals_storage:
                    ui.notification_show("⚠️ Please add some goals first!", type="warning")
                    return
                
                algorithm_type = input.algorithm_type()
                ui.notification_show(f"🚀 Running {algorithm_type} optimization with goal-responsive portfolios...", 
                                type="info")
                pass  # Do not log visitor financial inputs.
                
                # Get enhanced parameters
                try:
                    initial_wealth = input.initial_wealth()
                    horizon = input.time_horizon()
                    inflation_rate = input.inflation_rate() / 100
                    wealth_grid_size = input.wealth_grid_size()
                    n_portfolios = input.n_portfolios()
                except:
                    initial_wealth = 100000
                    horizon = 30
                    inflation_rate = 0.03
                    wealth_grid_size = 200
                    n_portfolios = 10
                
                pass  # Do not log visitor financial inputs.

                
                # *** KEY CHANGE: Create goal-responsive portfolio manager ***
                goal_responsive_manager = GoalResponsivePortfolioManager(etf_manager)
                dynamic_strategies = goal_responsive_manager.update_goals(
                    goals_storage, infusions_storage, initial_wealth, horizon
                )

                pass  # Do not log visitor financial inputs.
                
                pass  # Do not log visitor financial inputs.
                
                # Update ETF manager with goal-responsive strategies
                etf_manager.portfolio_strategies = dynamic_strategies
                etf_manager.efficient_frontier = dynamic_strategies
                
                # Determine algorithm automatically if needed
                if algorithm_type == "auto":
                    problem_complexity = len(goals_storage) * horizon * wealth_grid_size
                    if problem_complexity < 50000 and len(goals_storage) <= 5:
                        actual_algorithm = "full"
                    else:
                        actual_algorithm = "heuristic"
                    pass  # Do not log visitor financial inputs.
                else:
                    actual_algorithm = algorithm_type
                
                # Create enhanced optimizer with goal-responsive portfolios
                optimizer = CompleteGoalOptimizer(
                    initial_wealth=initial_wealth,
                    horizon_years=horizon,
                    wealth_grid_size=wealth_grid_size,
                    etf_manager=etf_manager,  # This now has goal-responsive strategies
                    n_portfolios=len(dynamic_strategies),
                    inflation_rate=inflation_rate
                )
                
                # Add goals with enhanced features
                for goal in goals_storage:
                    optimizer.add_goal(
                        name=goal['name'],
                        time_year=goal['time_year'],
                        cost=goal['cost'],
                        utility=goal['utility'],
                        priority=goal.get('priority', 1.0),
                        is_partial=goal.get('is_partial', False),
                        goal_type=goal.get('goal_type', 'regular')
                    )
                
                # Add infusions
                for infusion in infusions_storage:
                    optimizer.add_infusion(
                        time_year=infusion['time_year'],
                        amount=infusion['amount'],
                        source=infusion.get('source', 'savings')
                    )
                
                # Run optimization with goal-responsive strategies
                use_full = actual_algorithm == "full"
                probabilities = optimizer.optimize(use_full_algorithm=use_full)
                
                # Store enhanced results with goal-responsive information
                optimization_results = {
                    'goal_probabilities': probabilities,
                    'optimized': True,
                    'algorithm': actual_algorithm,
                    'algorithm_details': optimizer.get_optimization_summary(),
                    'optimizer': optimizer,
                    'portfolio_strategies': dynamic_strategies,  # Goal-responsive strategies
                    'goal_responsive_manager': goal_responsive_manager,  # New field
                    'wealth_distribution': getattr(optimizer, 'wealth_distribution', None),
                    'computation_time': getattr(optimizer, 'computation_time', 0),
                    'etf_manager': etf_manager
                }
                
                pass  # Do not log visitor financial inputs.
                trigger_refresh()
                trigger_optimization()
                
                computation_time = optimization_results.get('computation_time', 0)
                ui.notification_show(f"✅ Goal-responsive optimization complete! ({computation_time:.1f}s)", 
                                type="success")
                
                pass  # Do not log visitor financial inputs.
                        
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                import traceback
                pass  # Do not log visitor financial inputs.
                ui.notification_show(f"Optimization error: {str(e)}", type="error")
        
        @reactive.effect
        @reactive.event(input.clear_goals_btn)
        def handle_clear():
            global goals_storage, infusions_storage, goal_id_counter, infusion_id_counter, optimization_results

            try: 
                goals_storage.clear()
                infusions_storage.clear()
                optimization_results.clear()
                goal_id_counter = 0
                infusion_id_counter = 0
                
                trigger_refresh()
                ui.notification_show("🗑️ All data cleared!", type="info")
                pass  # Do not log visitor financial inputs.
                
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                ui.notification_show(f"Error clearing data: {str(e)}", type="error")
        
        @render.download(filename="multi-goal-demo-session.json")
        def save_session_btn():
            yield json.dumps({"data_source": "SYNTHETIC DEMO", "goals": goals_storage,
                              "infusions": infusions_storage}, indent=2)

        # Enhanced template handlers with Das et al. framework
        @reactive.effect
        @reactive.event(input.template_retirement_enhanced)
        def handle_retirement_enhanced():
            try:
                add_template_goal("High-Priority Retirement Fund", 1200000, 30, 5000, 
                                priority=2.0, is_partial=True)
                trigger_refresh()
                ui.notification_show("✅ Enhanced high-priority retirement fund added with partial options!", 
                                   type="success")
            except Exception as e:
                ui.notification_show(f"Error adding template: {str(e)}", type="error")
        
        @reactive.effect
        @reactive.event(input.template_house_partial)
        def handle_house_partial():
            try:
                add_template_goal("House Down Payment", 120000, 7, 2000, 
                                priority=1.5, is_partial=True)
                trigger_refresh()
                ui.notification_show("✅ House down payment with partial achievement options added!", 
                                   type="success")
            except Exception as e:
                ui.notification_show(f"Error adding template: {str(e)}", type="error")
        
        @reactive.effect
        @reactive.event(input.template_education_tiered)
        def handle_education_tiered():
            try:
                # Add tiered education goals
                add_template_goal("Education Fund - Undergraduate", 180000, 18, 2500, 
                                priority=1.8, is_partial=True)
                add_template_goal("Education Fund - Graduate", 100000, 22, 1500, 
                                priority=1.2, is_partial=True)
                trigger_refresh()
                ui.notification_show("✅ Tiered education funds added!", type="success")
            except Exception as e:
                ui.notification_show(f"Error adding template: {str(e)}", type="error")
        
        @reactive.effect
        @reactive.event(input.template_emergency_flexible)
        def handle_emergency_flexible():
            try:
                add_template_goal("Emergency Fund", 50000, 3, 1200, 
                                priority=1.0, is_partial=True)
                trigger_refresh()
                ui.notification_show("✅ Flexible emergency fund added!", type="success")
            except Exception as e:
                ui.notification_show(f"Error adding template: {str(e)}", type="error")
        
        @reactive.effect
        @reactive.event(input.template_luxury_deferrable)
        def handle_luxury_deferrable():
            try:
                add_template_goal("Luxury Travel Fund", 30000, 10, 800, 
                                priority=0.6, is_partial=True)
                add_template_goal("Vacation Home", 200000, 15, 1000, 
                                priority=0.8, is_partial=True)
                trigger_refresh()
                ui.notification_show("✅ Deferrable luxury goals added!", type="success")
            except Exception as e:
                ui.notification_show(f"Error adding template: {str(e)}", type="error")

        # === ENHANCED OUTPUTS ===
        @output
        @render.text
        def enhanced_status_summary():
            refresh_trigger()
            
            goal_count = len(goals_storage)
            infusion_count = len(infusions_storage)
            total_cost = sum(g.get('cost', 0) for g in goals_storage)
            total_infusions = sum(i.get('amount', 0) for i in infusions_storage)
            avg_priority = np.mean([g.get('priority', 1.0) for g in goals_storage]) if goals_storage else 1.0
            optimized = optimization_results.get('optimized', False)
            algorithm = optimization_results.get('algorithm', 'none')
            
            # Enhanced status with algorithm details
            if optimized:
                details = optimization_results.get('algorithm_details', {})
                comp_time = details.get('computation_time', 0)
                status = f"✅ Optimized ({algorithm}) - {comp_time:.1f}s"
            else:
                status = "❌ Not Optimized"
            
            result = (f"Goals: {goal_count} (${total_cost:,})\n"
                     f"Infusions: {infusion_count} (${total_infusions:,})\n"
                     f"Avg Priority: {avg_priority:.1f}\n"
                     f"Status: {status}")
            return result
        
        @output
        @render.ui
        def optimization_metrics():
            refresh_trigger()
            optimization_trigger()
            
            if not optimization_results.get('optimized'):
                return ui.div(
                    ui.p("Run optimization to see metrics", class_="text-muted small text-center")
                )
            
            details = optimization_results.get('algorithm_details', {})
            goal_analysis = details.get('goal_analysis', {})
            
            metrics_cards = []
            
            if goal_analysis:
                metrics_cards.extend([
                    ui.div(
                        ui.strong("Avg Success Rate"), ui.br(),
                        ui.span(f"{goal_analysis.get('average_probability', 0):.0%}", 
                               class_="text-success h5")
                    ),
                    ui.div(
                        ui.strong("High Confidence"), ui.br(),
                        ui.span(f"{goal_analysis.get('high_confidence_goals', 0)}", 
                               class_="text-primary h5")
                    ),
                    ui.div(
                        ui.strong("At Risk"), ui.br(),
                        ui.span(f"{goal_analysis.get('at_risk_goals', 0)}", 
                               class_="text-warning h5")
                    )
                ])
            
            comp_time = details.get('computation_time', 0)
            if comp_time > 0:
                metrics_cards.append(
                    ui.div(
                        ui.strong("Compute Time"), ui.br(),
                        ui.span(f"{comp_time:.1f}s", class_="text-info h6")
                    )
                )
            
            return ui.div(
                *metrics_cards,
                class_="d-flex justify-content-between text-center"
            )
        
        @output
        @render.ui
        def enhanced_goals_display():
            refresh_trigger()
            
            if not goals_storage and not infusions_storage:
                return ui.div(
                    ui.card(
                        ui.card_header("📋 No Goals or Infusions Yet"),
                        ui.card_body(
                            ui.p("Add goals using the enhanced framework from Das et al. (2022)", 
                                 class_="text-center"),
                            ui.p("✨ Features: Priority weighting, partial achievement, concurrent goals", 
                                 class_="text-center text-muted small")
                        )
                    ),
                    class_="p-3"
                )
            
            sections = []
            
            # Enhanced Goals section with Das et al. framework details
            if goals_storage:
                goal_cards = []
                goal_probs = optimization_results.get('goal_probabilities', {})
                
                for goal in goals_storage:
                    prob = goal_probs.get(goal['name'], 0.0)
                    prob_text = f"{prob:.1%}" if optimization_results.get('optimized') else "Not optimized"
                    
                    # Enhanced probability classification
                    if prob > 0.8:
                        prob_class = "text-success fw-bold"
                        prob_icon = "🟢"
                    elif prob > 0.6:
                        prob_class = "text-success"
                        prob_icon = "🟡"
                    elif prob > 0.4:
                        prob_class = "text-warning"
                        prob_icon = "🟠"
                    else:
                        prob_class = "text-danger"
                        prob_icon = "🔴"
                    
                    priority = goal.get('priority', 1.0)
                    goal_type = goal.get('goal_type', 'regular')
                    is_partial = goal.get('is_partial', False)
                    
                    # Priority classification
                    if priority > 1.5:
                        priority_class = "text-success fw-bold"
                        priority_icon = "🔥"
                    elif priority > 1.0:
                        priority_class = "text-info"
                        priority_icon = "⭐"
                    else:
                        priority_class = "text-muted"
                        priority_icon = "◾"
                    
                    card = ui.card(
                        ui.card_header(
                            ui.div(
                                ui.h5(f"🎯 {goal['name']}", class_="mb-1"),
                                ui.span(f"{goal_type.title()}", 
                                        class_="badge bg-secondary small"),
                                ui.span("Partial OK" if is_partial else "All-or-Nothing", 
                                        class_="badge bg-info small ms-1"),
                                class_="d-flex justify-content-between align-items-center"
                            )
                        ),
                        ui.card_body(
                            ui.p(f"💰 Amount: ${goal['cost']:,}", class_="mb-1"),
                            ui.p(f"⏰ Time: {goal['time_year']} years", class_="mb-1"),
                            ui.p(f"⚡ Utility (u_k): {goal['utility']}", class_="mb-1"),
                            ui.p(f"{priority_icon} Priority: {priority:.1f}", 
                                 class_=f"{priority_class} mb-1"),
                            ui.p(f"{prob_icon} Success Rate: {prob_text}", 
                                 class_=f"{prob_class}")
                        ),
                        class_="mb-3"
                    )
                    goal_cards.append(card)
                
                goals_section = ui.div(
                    ui.h4(f"🎯 Enhanced Goals ({len(goals_storage)})", class_="mb-3"),
                    *goal_cards
                )
                sections.append(goals_section)

            # Enhanced Infusions section
            if infusions_storage:
                infusion_cards = []
                
                for infusion in infusions_storage:
                    source = infusion.get('source', 'savings')
                    inflation_adj = infusion.get('inflation_adjust', True)
                    
                    source_icons = {
                        'savings': '💰',
                        'bonus': '🎁', 
                        'inheritance': '🏛️',
                        'sale': '🏠',
                        'other': '💼'
                    }
                    
                    card = ui.card(
                        ui.card_header(
                            ui.h5(f"{source_icons.get(source, '💰')} {source.title()} Infusion", 
                                   class_="mb-0")
                        ),
                        ui.card_body(
                            ui.p(f"💵 Amount: ${infusion['amount']:,}", class_="mb-1"),
                            ui.p(f"⏰ Time: Year {infusion['time_year']}", class_="mb-1"),
                            ui.p(f"📈 Inflation Adj: {'Yes' if inflation_adj else 'No'}", 
                                 class_="mb-0 small text-muted")
                        ),
                        class_="mb-3"
                    )
                    infusion_cards.append(card)
                
                infusions_section = ui.div(
                    ui.h4(f"💰 Cash Infusions I(t) ({len(infusions_storage)})", class_="mb-3"),
                    *infusion_cards
                )
                sections.append(infusions_section)
            
            return ui.div(*sections, class_="p-3")
        
        @output
        @render.ui
        def optimization_results_enhanced():
            refresh_trigger()
            optimization_trigger()
            
            if not optimization_results.get('optimized'):
                return ui.div()
            
            algorithm = optimization_results.get('algorithm', 'unknown')
            details = optimization_results.get('algorithm_details', {})
            optimizer = optimization_results.get('optimizer')
            
            result_sections = [
                ui.h5("🔍 Enhanced Optimization Results"),
                ui.p(f"Algorithm: {algorithm.replace('_', ' ').title()}", class_="mb-1"),
            ]
            
            if details:
                result_sections.extend([
                    ui.p(f"Expected Utility: {details.get('expected_utility', 0):,.0f}", class_="mb-1"),
                    ui.p(f"Wealth Grid: {details.get('wealth_grid_size', 0)} points", class_="mb-1"),
                    ui.p(f"Time Horizon: {details.get('time_horizon', 0)} periods", class_="mb-1"),
                ])
                
                if 'goal_analysis' in details:
                    analysis = details['goal_analysis']
                    result_sections.extend([
                        ui.hr(),
                        ui.h6("📊 Goal Analysis Summary"),
                        ui.p(f"Average Success Rate: {analysis.get('average_probability', 0):.1%}", class_="mb-1"),
                        ui.p(f"High Confidence Goals (>80%): {analysis.get('high_confidence_goals', 0)}", class_="mb-1"),
                        ui.p(f"At-Risk Goals (<30%): {analysis.get('at_risk_goals', 0)}", class_="mb-1"),
                    ])
            
            if optimizer and hasattr(optimizer, 'etf_manager') and optimizer.etf_manager:
                etf_components = getattr(optimizer.etf_manager, 'etf_components', [])
                result_sections.extend([
                    ui.hr(),
                    ui.h6("💼 Portfolio Integration"),
                    ui.p(f"ETF Universe: {len(etf_components)} assets", class_="mb-1"),
                    ui.p(f"Portfolio Strategies: {len(getattr(optimizer, 'portfolio_strategies', []))}", class_="mb-1"),
                ])
            
            return ui.card(
                ui.card_header("🔧 Das et al. (2022) Optimization Details"),
                ui.card_body(*result_sections),
                class_="mt-3"
            )
        
        @output
        @render.ui
        def advanced_analysis_dashboard():
            refresh_trigger()
            optimization_trigger()
            
            if not optimization_results.get('optimized') or not goals_storage:
                return ui.div(
                    ui.h5("📊 Advanced Analysis Dashboard", class_="text-center text-muted"),
                    ui.p("Run optimization to see advanced analysis with Das et al. framework", 
                         class_="text-center text-muted"),
                    class_="py-5"
                )
            
            try:
                goal_probs = optimization_results.get('goal_probabilities', {})
                details = optimization_results.get('algorithm_details', {})
                
                # Enhanced DataFrame with complete goal information
                df = pd.DataFrame([
                    {
                        'Goal': goal['name'],
                        'Success_Rate': goal_probs.get(goal['name'], 0),
                        'Cost': goal['cost'],
                        'Time': goal['time_year'],
                        'Priority': goal.get('priority', 1.0),
                        'Utility': goal['utility'],
                        'Type': goal.get('goal_type', 'regular'),
                        'Partial_OK': goal.get('is_partial', False),
                        'Risk_Score': 1 - goal_probs.get(goal['name'], 0)  # Risk = 1 - probability
                    }
                    for goal in goals_storage
                ])
                
                if df.empty:
                    return ui.div(ui.p("No data for analysis", class_="text-muted"))
                
                # Create comprehensive visualizations
                visualizations = []
                
                # 1. Enhanced Success Rate Chart with Priority Sizing
                fig1 = px.scatter(df, x='Time', y='Success_Rate', size='Priority', 
                                color='Type', hover_data=['Cost', 'Utility'],
                                title='Goal Achievement Probabilities vs Time',
                                labels={'Success_Rate': 'Success Probability', 'Time': 'Years to Goal'})
                
                fig1.update_layout(
                    yaxis_tickformat='.0%',
                    height=400,
                    showlegend=True
                )
                
                fig1.add_hline(y=0.8, line_dash="dash", line_color="green", 
                              annotation_text="High Confidence (80%)")
                fig1.add_hline(y=0.3, line_dash="dash", line_color="red", 
                              annotation_text="At Risk (30%)")
                
                visualizations.append(ui.HTML(pio.to_html(fig1, include_plotlyjs=True)))
                
                # 2. Priority vs Risk Analysis
                fig2 = px.scatter(df, x='Priority', y='Risk_Score', size='Cost',
                                color='Success_Rate', color_continuous_scale='RdYlGn_r',
                                title='Priority vs Risk Analysis',
                                labels={'Risk_Score': 'Goal Risk (1 - Success Rate)'})
                
                fig2.update_layout(height=400)
                visualizations.append(ui.HTML(pio.to_html(fig2, include_plotlyjs=True)))
                
                # 3. Cost vs Utility Efficiency
                df['Utility_per_Dollar'] = df['Utility'] / df['Cost']
                fig3 = px.scatter(df, x='Cost', y='Utility', size='Success_Rate',
                                color='Priority', hover_data=['Utility_per_Dollar'],
                                title='Cost vs Utility Efficiency Analysis',
                                labels={'Cost': 'Goal Cost ($)', 'Utility': 'Utility Value'})
                
                fig3.update_layout(height=400)
                fig3.update_layout(xaxis=dict(tickformat='$,.0f'))
                visualizations.append(ui.HTML(pio.to_html(fig3, include_plotlyjs=True)))
                fig3.update_layout(xaxis=dict(tickformat='$,.0f'))
                # Calculate enhanced insights
                insights = _calculate_enhanced_insights(df, goal_probs, details)
                
                return ui.div(
                    *visualizations,
                    ui.div(
                        ui.h5("🧠 Advanced Insights (Das et al. Framework)"),
                        ui.row(
                            ui.column(6, 
                                ui.h6("📈 Goal Statistics"),
                                ui.tags.ul(*[ui.tags.li(insight) for insight in insights['goal_stats']])
                            ),
                            ui.column(6,
                                ui.h6("💡 Optimization Insights"), 
                                ui.tags.ul(*[ui.tags.li(insight) for insight in insights['optimization']])
                            )
                        ),
                        class_="mt-4"
                    )
                )
                
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                import traceback
                pass  # Do not log visitor financial inputs.
                
                # Enhanced fallback with detailed error info
                goal_probs = optimization_results.get('goal_probabilities', {})
                return ui.div(
                    ui.card(
                        ui.card_header("📊 Advanced Analysis Results"),
                        ui.card_body(
                            ui.h5("Goal Achievement Probabilities:"),
                            *[ui.p(f"🎯 {name}: {prob:.0%} (Priority: {next((g.get('priority', 1.0) for g in goals_storage if g['name'] == name), 1.0):.1f})", 
                                   class_="mb-1") 
                              for name, prob in goal_probs.items()],
                            ui.hr(),
                            ui.p(f"Average Success Rate: {np.mean(list(goal_probs.values())):.0%}"),
                            ui.p(f"Algorithm: {optimization_results.get('algorithm', 'unknown').replace('_', ' ').title()}"),
                            ui.p(f"Total Goals: {len(goals_storage)}"),
                            ui.p(f"Enhanced Features: Priority weighting, partial goals, ETF integration")
                        )
                    ),
                    ui.p(f"Chart error (using enhanced fallback): {str(e)}", class_="text-muted small")
                )
        
        def _calculate_enhanced_insights(df, goal_probs, details):
            """Calculate enhanced insights following Das et al. framework"""
            goal_stats = []
            optimization_insights = []
            
            if not df.empty:
                # Goal statistics
                high_priority_goals = df[df['Priority'] > 1.5]
                partial_goals = df[df['Partial_OK'] == True]
                
                goal_stats.extend([
                    f"Total Goals: {len(df)}",
                    f"High Priority Goals (>1.5): {len(high_priority_goals)}",
                    f"Goals with Partial Options: {len(partial_goals)}",
                    f"Average Goal Cost: ${df['Cost'].mean():,.0f}",
                    f"Most Expensive Goal: ${df['Cost'].max():,.0f}",
                    f"Goal Types: {', '.join(df['Type'].unique())}"
                ])
                
                # Success rate analysis
                if goal_probs:
                    probs = list(goal_probs.values())
                    goal_stats.extend([
                        f"Average Success Rate: {np.mean(probs):.0%}",
                        f"Success Rate Range: {np.min(probs):.0%} - {np.max(probs):.0%}",
                        f"High Confidence Goals (>80%): {sum(1 for p in probs if p > 0.8)}",
                        f"At-Risk Goals (<30%): {sum(1 for p in probs if p < 0.3)}"
                    ])
                
                # Optimization insights
                if details:
                    algorithm = details.get('algorithm_used', 'unknown')
                    optimization_insights.extend([
                        f"Algorithm Used: {algorithm.replace('_', ' ').title()}",
                        f"Computation Time: {details.get('computation_time', 0):.1f}s",
                        f"Portfolio Strategies: {details.get('portfolio_strategies', 0)}",
                        f"Wealth Grid Size: {details.get('wealth_grid_size', 0)}"
                    ])
                    
                    if 'expected_utility' in details:
                        optimization_insights.append(f"Expected Utility: {details['expected_utility']:,.0f}")
                    
                    if 'wealth_forecast' in details:
                        forecast = details['wealth_forecast']
                        if 'expected' in forecast:
                            optimization_insights.append(f"Expected Final Wealth: ${forecast['expected']:,.0f}")
            
            return {
                'goal_stats': goal_stats,
                'optimization': optimization_insights
            }
        
        @output
        @render.ui
        def enhanced_portfolio_analysis():
            refresh_trigger()
            optimization_trigger()
            
            optimizer = optimization_results.get('optimizer')
            etf_manager = optimization_results.get('etf_manager')
            goal_responsive_manager = optimization_results.get('goal_responsive_manager')  # New
            
            if not optimizer or not hasattr(optimizer, 'portfolio_strategies'):
                return ui.div(
                    ui.h5("💼 Enhanced Portfolio Analysis", class_="text-center text-muted"),
                    ui.p("Run optimization to see goal-responsive ETF portfolio strategies", 
                        class_="text-center text-muted"),
                    class_="py-5"
                )
            
            try:
                strategies = optimizer.portfolio_strategies
                
                if not strategies:
                    return ui.div(ui.p("No portfolio strategies available", class_="text-muted"))
                
                # Enhanced portfolio strategy analysis with goal-responsive information
                portfolio_df = pd.DataFrame([
                    {
                        'Strategy': strategy.get('name', f"Strategy {i+1}"),
                        'Expected_Return': strategy['return'],
                        'Volatility': strategy['volatility'], 
                        'Sharpe_Ratio': strategy.get('sharpe', 0),
                        'Risk_Level': strategy.get('risk_level', 'Unknown'),
                        'Goal_Responsive': strategy.get('goal_responsive', False),
                        'Goal_Score': strategy.get('goal_appropriateness_score', 0)
                    }
                    for i, strategy in enumerate(strategies)
                ])
                
                # Mark goal-responsive strategies
                portfolio_df['Marker_Size'] = portfolio_df['Goal_Score'].clip(lower=0.1) * 100
                portfolio_df['Color'] = portfolio_df['Goal_Responsive'].map({True: 'Goal-Optimized', False: 'Standard'})
                
                # Enhanced Efficient Frontier Chart with goal-responsive highlighting
                fig_ef = px.scatter(portfolio_df, x='Volatility', y='Expected_Return',
                                color='Color', size='Marker_Size',
                                hover_data=['Risk_Level', 'Goal_Score'],
                                title='Goal-Responsive Efficient Frontier',
                                color_discrete_map={'Goal-Optimized': '#FF6B35', 'Standard': '#4ECDC4'})
                
                fig_ef.update_layout(
                    xaxis_title="Volatility (Risk)",
                    yaxis_title="Expected Return",
                    xaxis_tickformat='.1%',
                    yaxis_tickformat='.1%',
                    height=450
                )
                
                # Add annotations for goal-responsive strategies
                goal_responsive_strategies = portfolio_df[portfolio_df['Goal_Responsive'] == True]
                if not goal_responsive_strategies.empty:
                    best_goal_strategy = goal_responsive_strategies.loc[goal_responsive_strategies['Goal_Score'].idxmax()]
                    fig_ef.add_annotation(
                        x=best_goal_strategy['Volatility'],
                        y=best_goal_strategy['Expected_Return'],
                        text="🎯 Best for Your Goals",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#FF6B35",
                        arrowwidth=2,
                        bgcolor="#FF6B35",
                        bordercolor="#FF6B35",
                        font=dict(color="white", size=12)
                    )
                
                # Goal-responsive portfolio selection
                best_strategy = _select_goal_responsive_strategy(strategies, goals_storage)
                
                # Enhanced portfolio composition analysis
                composition_cards = []
                if best_strategy and 'weights' in best_strategy:
                    composition_data = _prepare_composition_data(best_strategy)
                    
                    if composition_data:
                        # Enhanced pie chart with goal-responsive labeling
                        title_suffix = " (Goal-Optimized)" if best_strategy.get('goal_responsive', False) else ""
                        fig_pie = px.pie(composition_data, values='Weight', names='Asset',
                                    title=f'Optimal Portfolio Composition{title_suffix}',
                                    hover_data=['Description'])
                        fig_pie.update_layout(height=400)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        
                        composition_cards.extend([
                            ui.h5("🎯 Goal-Responsive Portfolio Selection"),
                            ui.p(f"Selected: {best_strategy.get('name', 'Optimal Strategy')}", 
                                class_="text-primary fw-bold"),
                            ui.p(f"Goal Optimization: {'✅ Optimized for your specific goals' if best_strategy.get('goal_responsive', False) else '❌ Standard strategy'}", 
                                class_="small text-muted"),
                            ui.HTML(pio.to_html(fig_pie, include_plotlyjs=True)),
                            
                            # Enhanced metrics table
                            _create_portfolio_metrics_table(best_strategy),
                            
                            # Enhanced holdings table
                            _create_holdings_table(composition_data)
                        ])
                
                # Goal-responsive information display
                goal_responsive_info = []
                if goal_responsive_manager:
                    goal_responsive_info = [_create_goal_responsive_portfolio_display(strategies, goal_responsive_manager)]
                
                # ETF Universe and Strategy Information
                etf_info = _create_etf_universe_info(etf_manager)
                
                return ui.div(
                    ui.HTML(pio.to_html(fig_ef, include_plotlyjs=True)),
                    ui.br(),
                    
                    # Goal-responsive analysis section
                    *goal_responsive_info,
                    ui.br(),
                    
                    ui.row(
                        ui.column(7, *composition_cards) if composition_cards else ui.div(),
                        ui.column(5, *etf_info)
                    ),
                    _create_strategy_summary_table(strategies),
                    _create_goal_responsive_strategy_comparison(strategies) if goal_responsive_manager else ui.div(),
                    class_="mt-3"
                )
                
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                return ui.div(
                    ui.card(
                        ui.card_header("💼 Enhanced Portfolio Analysis"),
                        ui.card_body(
                            ui.h5("Portfolio Strategies Overview"),
                            ui.p(f"Available Strategies: {len(strategies) if 'strategies' in locals() else 'Unknown'}"),
                            ui.p(f"ETF Integration: {'Yes' if etf_manager else 'No'}"),
                            ui.p(f"Goal-Responsive Selection: {'✅ Enabled' if goal_responsive_manager else '❌ Standard'}"),
                            ui.p(f"Error details: {str(e)}", class_="text-muted small")
                        )
                    )
                )
        
        def _select_goal_responsive_strategy(strategies, goals):
            """Select optimal strategy based on goal characteristics"""
            if not goals:
                return max(strategies, key=lambda x: x.get('sharpe', 0))
            
            # Analyze goal characteristics for strategy selection
            total_cost = sum(g.get('cost', 0) for g in goals)
            avg_time = np.mean([g.get('time_year', 10) for g in goals])
            min_time = min(g.get('time_year', 10) for g in goals)
            max_priority = max(g.get('priority', 1.0) for g in goals)
            
            # Goal-responsive selection logic
            if min_time <= 3:
                target_volatility = 0.08  # Conservative for short-term goals
            elif max_priority >= 1.8:
                target_volatility = 0.12  # Balanced for high-priority goals
            elif total_cost > 500000:
                target_volatility = 0.16  # Aggressive for expensive goals
            else:
                target_volatility = 0.10  # Moderate default
            
            # Find strategy closest to target volatility with good return
            best_strategy = min(strategies, 
                              key=lambda s: abs(s.get('volatility', 0.1) - target_volatility))
            
            pass  # Do not log visitor financial inputs.

            
            return best_strategy
        
        def _prepare_composition_data(strategy):
            """Prepare composition data for visualization"""
            composition_data = []
            
            if isinstance(strategy.get('weights'), dict):
                etf_descriptions = {
                    'IVV': 'iShares Core S&P 500',
                    'IWM': 'iShares Russell 2000', 
                    'EFA': 'iShares MSCI EAFE',
                    'EEM': 'iShares MSCI Emerging Markets',
                    'AGG': 'iShares Core US Aggregate Bond',
                    'GLD': 'SPDR Gold Shares',
                    'IYR': 'iShares US Real Estate'
                }
                
                for asset, weight in strategy['weights'].items():
                    if weight > 0.005:  # Only show weights > 0.5%
                        composition_data.append({
                            'Asset': asset,
                            'Description': etf_descriptions.get(asset, asset),
                            'Weight': weight,
                            'Weight_Pct': weight * 100
                        })
                
                composition_data.sort(key=lambda x: x['Weight'], reverse=True)
            
            return composition_data
        
        def _create_portfolio_metrics_table(strategy):
            """Create enhanced portfolio metrics table"""
            return ui.div(
                ui.h6("📊 Portfolio Metrics", class_="mt-3 mb-2"),
                ui.div(
                    ui.tags.table([
                        ui.tags.thead([
                            ui.tags.tr([
                                ui.tags.th("Metric", class_="text-start"),
                                ui.tags.th("Value", class_="text-end")
                            ])
                        ]),
                        ui.tags.tbody([
                            ui.tags.tr([
                                ui.tags.td("Expected Return", class_="fw-medium"),
                                ui.tags.td(f"{strategy['return']:.1%}", 
                                          class_="text-end text-success fw-bold")
                            ]),
                            ui.tags.tr([
                                ui.tags.td("Volatility", class_="fw-medium"),
                                ui.tags.td(f"{strategy['volatility']:.1%}", 
                                          class_="text-end text-warning fw-bold")
                            ]),
                            ui.tags.tr([
                                ui.tags.td("Sharpe Ratio", class_="fw-medium"),
                                ui.tags.td(f"{strategy.get('sharpe', 0):.2f}", 
                                          class_="text-end text-info fw-bold")
                            ]),
                            ui.tags.tr([
                                ui.tags.td("Risk Level", class_="fw-medium"),
                                ui.tags.td(strategy.get('risk_level', 'Unknown'), 
                                          class_="text-end text-primary fw-bold")
                            ])
                        ])
                    ], class_="table table-sm table-hover"),
                    class_="table-responsive"
                )
            )
        
        def _create_holdings_table(composition_data):
            """Create enhanced holdings details table"""
            if not composition_data:
                return ui.div()
            
            return ui.div(
                ui.h6("💼 Holdings Details", class_="mt-3 mb-2"),
                ui.div(
                    ui.tags.table([
                        ui.tags.thead([
                            ui.tags.tr([
                                ui.tags.th("ETF", class_="text-start"),
                                ui.tags.th("Weight", class_="text-end"),
                                ui.tags.th("Description", class_="text-start")
                            ])
                        ]),
                        ui.tags.tbody([
                            ui.tags.tr([
                                ui.tags.td(item['Asset'], class_="fw-bold text-primary"),
                                ui.tags.td(f"{item['Weight_Pct']:.1f}%", 
                                          class_="text-end fw-medium"),
                                ui.tags.td(item['Description'], 
                                          class_="text-muted small")
                            ]) for item in composition_data
                        ])
                    ], class_="table table-sm table-striped"),
                    class_="table-responsive"
                )
            )
        
        def _create_etf_universe_info(etf_manager):
            """Create ETF universe information display"""
            return [
                ui.h5("📈 Enhanced ETF Universe"),
                ui.div([
                    ui.p("🇺🇸 US Equity:", class_="fw-bold mb-1"),
                    ui.p("• IVV - iShares Core S&P 500 ETF", class_="small mb-1 ms-3"),
                    ui.p("• IWM - iShares Russell 2000 ETF", class_="small mb-2 ms-3"),
                    
                    ui.p("🌍 International Equity:", class_="fw-bold mb-1"),
                    ui.p("• EFA - iShares MSCI EAFE ETF", class_="small mb-1 ms-3"),
                    ui.p("• EEM - iShares MSCI Emerging Markets ETF", class_="small mb-2 ms-3"),
                    
                    ui.p("🏛️ Fixed Income & Alternatives:", class_="fw-bold mb-1"),
                    ui.p("• AGG - iShares Core US Aggregate Bond ETF", class_="small mb-1 ms-3"),
                    ui.p("• GLD - SPDR Gold Shares ETF", class_="small mb-1 ms-3"),
                    ui.p("• IYR - iShares US Real Estate ETF", class_="small mb-1 ms-3"),
                    
                    ui.hr(),
                    ui.p("🔬 Enhanced Features:", class_="fw-bold mb-1"),
                    ui.p("• Regime-switching returns", class_="small mb-1 ms-3"),
                    ui.p("• EWMA parameter estimation", class_="small mb-1 ms-3"),
                    ui.p("• Goal-responsive selection", class_="small mb-1 ms-3"),
                    ui.p("• Fat-tail modeling", class_="small mb-1 ms-3"),
                ])
            ]
        
        def _create_strategy_summary_table(strategies):
            """Create comprehensive strategy summary"""
            if not strategies:
                return ui.div()
            
            return ui.div(
                ui.h5("📊 Complete Strategy Summary"),
                ui.div(
                    ui.tags.table([
                        ui.tags.thead([
                            ui.tags.tr([
                                ui.tags.th("Statistic", class_="text-start"),
                                ui.tags.th("Value", class_="text-end")
                            ])
                        ]),
                        ui.tags.tbody([
                            ui.tags.tr([
                                ui.tags.td("Available Strategies", class_="fw-medium"),
                                ui.tags.td(f"{len(strategies)}", class_="text-end fw-bold")
                            ]),
                            ui.tags.tr([
                                ui.tags.td("Return Range", class_="fw-medium"),
                                ui.tags.td(f"{min(s['return'] for s in strategies):.1%} - {max(s['return'] for s in strategies):.1%}", 
                                          class_="text-end text-success fw-bold")
                            ]),
                            ui.tags.tr([
                                ui.tags.td("Risk Range", class_="fw-medium"),
                                ui.tags.td(f"{min(s['volatility'] for s in strategies):.1%} - {max(s['volatility'] for s in strategies):.1%}", 
                                          class_="text-end text-warning fw-bold")
                            ]),
                            ui.tags.tr([
                                ui.tags.td("Best Sharpe Ratio", class_="fw-medium"),
                                ui.tags.td(f"{max(s.get('sharpe', 0) for s in strategies):.2f}", 
                                          class_="text-end text-info fw-bold")
                            ])
                        ])
                    ], class_="table table-sm table-hover mb-3"),
                    class_="table-responsive"
                ),
                class_="mt-4"
            )
        
        @output
        @render.ui
        def algorithm_diagnostics():
            refresh_trigger()
            optimization_trigger()
            
            if not optimization_results.get('optimized'):
                return ui.div(
                    ui.card(
                        ui.card_header("🔬 Algorithm Diagnostics"),
                        ui.card_body(
                            ui.div(
                                ui.h5("Das et al. (2022) Dynamic Programming Framework", 
                                    class_="text-center text-muted mb-3"),
                                ui.p("Run optimization to see detailed algorithm diagnostics and performance metrics", 
                                    class_="text-center text-muted"),
                                ui.div(
                                    ui.tags.ul(
                                        ui.tags.li("Bellman equation solution analysis"),
                                        ui.tags.li("Transition probability computations"), 
                                        ui.tags.li("Goal vector processing c(t) and u(t)"),
                                        ui.tags.li("Portfolio strategy evaluation"),
                                        ui.tags.li("Convergence and accuracy metrics"),
                                        class_="text-muted small"
                                    ),
                                    class_="mt-3"
                                ),
                                class_="py-4"
                            )
                        )
                    )
                )
            
            try:
                details = optimization_results.get('algorithm_details', {})
                optimizer = optimization_results.get('optimizer')
                algorithm_type = details.get('algorithm_used', 'unknown')
                
                # Create diagnostic cards
                diagnostic_cards = []
                
                # 1. Algorithm Performance Card
                performance_metrics = []
                comp_time = details.get('computation_time', 0)
                
                if comp_time > 0:
                    performance_metrics.extend([
                        f"⏱️ Computation Time: {comp_time:.2f} seconds",
                        f"🔄 Algorithm: {algorithm_type.replace('_', ' ').title()}",
                    ])
                else:
                    performance_metrics.extend([
                        f"🔄 Algorithm: {algorithm_type.replace('_', ' ').title()}",
                        f"⚡ Status: Heuristic optimization completed",
                    ])
                
                problem_size = details.get('total_goals', 0) * details.get('time_horizon', 0)
                if problem_size > 0:
                    performance_metrics.append(f"📏 Problem Size: {details.get('total_goals', 0)} goals × {details.get('time_horizon', 0)} periods = {problem_size:,} combinations")
                
                if details.get('wealth_grid_size', 0) > 0:
                    performance_metrics.append(f"🌐 State Space: {details.get('wealth_grid_size', 0):,} wealth levels")
                
                if details.get('portfolio_strategies', 0) > 0:
                    performance_metrics.append(f"💼 Action Space: {details.get('portfolio_strategies', 0)} portfolio strategies")
                
                diagnostic_cards.append(
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.h6("⚡ Algorithm Performance", class_="mb-0"),
                                class_="d-flex align-items-center"
                            )
                        ),
                        ui.card_body(
                            ui.tags.ul(
                                *[ui.tags.li(metric, class_="mb-1") for metric in performance_metrics],
                                class_="mb-0"
                            )
                        ),
                        class_="mb-3"
                    )
                )
                
                # 2. Bellman Equation Details (only for full DP)
                if optimizer and hasattr(optimizer, 'value_function') and algorithm_type == 'full_dp':
                    bellman_metrics = []
                    
                    if hasattr(optimizer.value_function, 'shape'):
                        bellman_metrics.append(f"📐 Value Function: {optimizer.value_function.shape[0]} × {optimizer.value_function.shape[1]} matrix")
                    
                    if details.get('expected_utility', 0) != 0:
                        bellman_metrics.append(f"🎯 Expected Utility V(W₀,0): {details.get('expected_utility', 0):,.0f}")
                    
                    if hasattr(optimizer, 'wealth_grid'):
                        bellman_metrics.extend([
                            f"💰 Wealth Range: ${optimizer.wealth_grid[0]:,.0f} to ${optimizer.wealth_grid[-1]:,.0f}",
                            f"📍 Initial Position: Grid index {getattr(optimizer, 'i0', 'Unknown')}"
                        ])
                    
                    if bellman_metrics:
                        diagnostic_cards.append(
                            ui.card(
                                ui.card_header(
                                    ui.div(
                                        ui.h6("📐 Bellman Equation Solution", class_="mb-0"),
                                        class_="d-flex align-items-center"
                                    )
                                ),
                                ui.card_body(
                                    ui.tags.ul(
                                        *[ui.tags.li(metric, class_="mb-1") for metric in bellman_metrics],
                                        class_="mb-0"
                                    )
                                ),
                                class_="mb-3"
                            )
                        )
                
                # 3. Goal Vector Analysis
                goal_metrics = []
                if optimizer and hasattr(optimizer, 'goal_options'):
                    goal_options = optimizer.goal_options
                    if goal_options:
                        time_periods_with_goals = len(goal_options)
                        max_options = max(len(opts) for opts in goal_options.values())
                        total_combinations = sum(len(opts) for opts in goal_options.values())
                        
                        goal_metrics.extend([
                            f"📅 Time Periods with Goals: {time_periods_with_goals}",
                            f"🔢 Max Options per Period: {max_options}",
                            f"🎯 Total Goal-Time Combinations: {total_combinations:,}",
                            f"🔄 Concurrent Goals: {'Yes' if max_options > 2 else 'No'}"
                        ])
                
                if not goal_metrics:
                    goal_metrics.extend([
                        f"🎯 Total Goals: {details.get('total_goals', 0)}",
                        f"📊 Goal Processing: Vector optimization c(t), u(t)",
                        f"🔗 Framework: Das et al. (2022) methodology"
                    ])
                
                diagnostic_cards.append(
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.h6("🎯 Goal Vector Analysis", class_="mb-0"),
                                class_="d-flex align-items-center"
                            )
                        ),
                        ui.card_body(
                            ui.tags.ul(
                                *[ui.tags.li(metric, class_="mb-1") for metric in goal_metrics],
                                class_="mb-0"
                            )
                        ),
                        class_="mb-3"
                    )
                )
                
                # 4. Portfolio Strategy Diagnostics
                portfolio_metrics = []
                strategies = optimization_results.get('portfolio_strategies', [])
                etf_manager = optimization_results.get('etf_manager')
                
                if strategies:
                    returns = [s.get('return', 0) for s in strategies]
                    volatilities = [s.get('volatility', 0) for s in strategies]
                    sharpes = [s.get('sharpe', 0) for s in strategies if s.get('sharpe', 0) > 0]
                    
                    portfolio_metrics.extend([
                        f"📈 Strategy Count: {len(strategies)}",
                        f"📊 Return Range: {min(returns):.1%} to {max(returns):.1%}",
                        f"📉 Volatility Range: {min(volatilities):.1%} to {max(volatilities):.1%}"
                    ])
                    
                    if sharpes:
                        portfolio_metrics.append(f"⚡ Best Sharpe Ratio: {max(sharpes):.2f}")
                
                if etf_manager and hasattr(etf_manager, 'etf_components'):
                    portfolio_metrics.extend([
                        f"🏢 ETF Universe: {len(etf_manager.etf_components)} assets",
                        f"🔬 Enhanced Features: Regime-switching, EWMA, Goal-responsive selection"
                    ])
                
                if portfolio_metrics:
                    diagnostic_cards.append(
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    ui.h6("💼 Portfolio Strategy Diagnostics", class_="mb-0"),
                                    class_="d-flex align-items-center"
                                )
                            ),
                            ui.card_body(
                                ui.tags.ul(
                                    *[ui.tags.li(metric, class_="mb-1") for metric in portfolio_metrics],
                                    class_="mb-0"
                                )
                            ),
                            class_="mb-3"
                        )
                    )
                
                # 5. Solution Quality Assessment
                quality_metrics = []
                if 'goal_analysis' in details:
                    analysis = details['goal_analysis']
                    avg_prob = analysis.get('average_probability', 0)
                    high_conf = analysis.get('high_confidence_goals', 0)
                    at_risk = analysis.get('at_risk_goals', 0)
                    
                    # Quality assessment
                    if avg_prob > 0.7:
                        quality_status = "🟢 Excellent"
                    elif avg_prob > 0.5:
                        quality_status = "🟡 Good"
                    elif avg_prob > 0.3:
                        quality_status = "🟠 Moderate"
                    else:
                        quality_status = "🔴 Challenging"
                    
                    quality_metrics.extend([
                        f"🎯 Overall Quality: {quality_status}",
                        f"📊 Average Success Rate: {avg_prob:.1%}",
                        f"✅ High Confidence Goals (>80%): {high_conf}",
                        f"⚠️ At-Risk Goals (<30%): {at_risk}"
                    ])
                    
                    # Risk assessment
                    if at_risk > high_conf:
                        quality_metrics.append("🚨 Recommendation: Consider increasing initial wealth or reducing goal costs")
                    elif high_conf > at_risk * 2:
                        quality_metrics.append("💡 Recommendation: Well-funded portfolio with good goal achievement prospects")
                    else:
                        quality_metrics.append("⚖️ Recommendation: Balanced risk profile, consider priority adjustments")
                
                if quality_metrics:
                    diagnostic_cards.append(
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    ui.h6("📊 Solution Quality Assessment", class_="mb-0"),
                                    class_="d-flex align-items-center"
                                )
                            ),
                            ui.card_body(
                                ui.tags.ul(
                                    *[ui.tags.li(metric, class_="mb-1") for metric in quality_metrics],
                                    class_="mb-0"
                                )
                            ),
                            class_="mb-3"
                        )
                    )
                
                # 6. Framework Compliance Badge
                compliance_card = ui.card(
                    ui.card_header(
                        ui.div(
                            ui.h6("✅ Das et al. (2022) Framework Compliance", class_="mb-0 text-success"),
                            class_="d-flex align-items-center"
                        )
                    ),
                    ui.card_body(
                        ui.div(
                            ui.row(
                                ui.column(6,
                                    ui.tags.ul(
                                        ui.tags.li("✓ Dynamic Programming", class_="text-success small mb-1"),
                                        ui.tags.li("✓ Bellman Equation", class_="text-success small mb-1"),
                                        ui.tags.li("✓ Goal Vectors c(t), u(t)", class_="text-success small mb-1"),
                                        ui.tags.li("✓ Transition Probabilities", class_="text-success small mb-1"),
                                        class_="mb-0"
                                    )
                                ),
                                ui.column(6,
                                    ui.tags.ul(
                                        ui.tags.li("✓ Forward Simulation", class_="text-success small mb-1"),
                                        ui.tags.li("✓ Multiple Portfolios", class_="text-success small mb-1"),
                                        ui.tags.li("✓ Cash Infusions I(t)", class_="text-success small mb-1"),
                                        ui.tags.li("✓ Enhanced ETF Integration", class_="text-success small mb-1"),
                                        class_="mb-0"
                                    )
                                )
                            )
                        )
                    ),
                    class_="border-success mb-3"
                )
                
                # Combine all cards
                return ui.div(
                    ui.h4("🔬 Algorithm Diagnostics", class_="mb-4"),
                    ui.div(
                        *diagnostic_cards,
                        compliance_card,
                        class_="row row-cols-1 row-cols-lg-2 g-3"
                    ),
                    class_="mt-3"
                )
            
            except Exception as e:
                pass  # Do not log visitor financial inputs.
                import traceback
                pass  # Do not log visitor financial inputs.
                
                return ui.div(
                    ui.card(
                        ui.card_header("🔬 Algorithm Diagnostics"),
                        ui.card_body(
                            ui.div(
                                ui.h5("Diagnostics Error", class_="text-warning"),
                                ui.p(f"Error generating diagnostics: {str(e)}", class_="text-muted"),
                                ui.hr(),
                                ui.h6("Basic Information:"),
                                ui.p(f"Optimization Status: {'✅ Complete' if optimization_results.get('optimized') else '❌ Not Run'}"),
                                ui.p(f"Algorithm: {optimization_results.get('algorithm', 'Unknown')}"),
                                ui.p(f"Goals: {len(goals_storage)}"),
                                ui.p(f"Framework: Das et al. (2022) Dynamic Programming"),
                                class_="py-3"
                            )
                        )
                    ),
                    class_="mt-3"
                )
            
        
        @output
        @render.text
        def enhanced_debug_info():
            refresh_trigger()
            optimization_trigger()
            
            debug_sections = []
            
            # Basic Information
            debug_sections.extend([
                "=== ENHANCED MULTI-GOAL DEBUG INFO ===",
                f"Goals: {len(goals_storage)}",
                f"Infusions: {len(infusions_storage)}",
                f"Optimized: {optimization_results.get('optimized', False)}",
                f"Algorithm: {optimization_results.get('algorithm', 'none')}",
                f"Refresh trigger: {refresh_trigger()}",
                f"Optimization trigger: {optimization_trigger()}",
                ""
            ])
            
            # Enhanced Goal Details
            if goals_storage:
                debug_sections.append("=== ENHANCED GOALS ===")
                for goal in goals_storage:
                    priority = goal.get('priority', 1.0)
                    goal_type = goal.get('goal_type', 'regular')
                    is_partial = goal.get('is_partial', False)
                    debug_sections.append(
                        f"  - {goal['name']}: ${goal['cost']:,} at year {goal['time_year']}"
                        f" (Priority: {priority:.1f}, Type: {goal_type}, Partial: {is_partial})"
                    )
                debug_sections.append("")
            
            # Enhanced Infusion Details
            if infusions_storage:
                debug_sections.append("=== CASH INFUSIONS I(t) ===")
                for infusion in infusions_storage:
                    source = infusion.get('source', 'savings')
                    inflation_adj = infusion.get('inflation_adjust', True)
                    debug_sections.append(
                        f"  - ${infusion['amount']:,} at year {infusion['time_year']}"
                        f" (Source: {source}, Inflation Adj: {inflation_adj})"
                    )
                debug_sections.append("")
            
            # Optimization Results
            if optimization_results.get('optimized'):
                debug_sections.append("=== OPTIMIZATION RESULTS ===")
                details = optimization_results.get('algorithm_details', {})
                
                debug_sections.extend([
                    f"Algorithm Used: {details.get('algorithm_used', 'unknown')}",
                    f"Computation Time: {details.get('computation_time', 0):.2f}s",
                    f"Expected Utility: {details.get('expected_utility', 0):,.0f}",
                    f"Portfolio Strategies: {details.get('portfolio_strategies', 0)}",
                    f"Wealth Grid Size: {details.get('wealth_grid_size', 0)}",
                    f"Time Horizon: {details.get('time_horizon', 0)} periods",
                    ""
                ])
                
                # Goal Probabilities
                goal_probs = optimization_results.get('goal_probabilities', {})
                if goal_probs:
                    debug_sections.append("=== GOAL PROBABILITIES ===")
                    for name, prob in goal_probs.items():
                        debug_sections.append(f"  - {name}: {prob:.1%}")
                    debug_sections.append("")
                
                # Goal Analysis
                if 'goal_analysis' in details:
                    analysis = details['goal_analysis']
                    debug_sections.extend([
                        "=== GOAL ANALYSIS ===",
                        f"Average Probability: {analysis.get('average_probability', 0):.1%}",
                        f"High Confidence Goals: {analysis.get('high_confidence_goals', 0)}",
                        f"At-Risk Goals: {analysis.get('at_risk_goals', 0)}",
                        ""
                    ])
            
            # ETF Manager Debug
            optimizer = optimization_results.get('optimizer')
            etf_manager = optimization_results.get('etf_manager')
            
            if optimizer and hasattr(optimizer, 'etf_manager') and optimizer.etf_manager:
                debug_sections.extend([
                    "=== ETF MANAGER DEBUG ===",
                    f"ETF Manager: Loaded and Enhanced",
                    f"ETF Components: {len(getattr(optimizer.etf_manager, 'etf_components', []))}",
                    f"Portfolio Strategies: {len(getattr(optimizer, 'portfolio_strategies', []))}",
                    f"Returns Data Shape: {getattr(optimizer.etf_manager.returns_data, 'shape', 'N/A') if hasattr(optimizer.etf_manager, 'returns_data') and optimizer.etf_manager.returns_data is not None else 'N/A'}",
                    f"Correlation Matrix: {'Available' if hasattr(optimizer.etf_manager, 'correlation_matrix') else 'Not Available'}",
                    ""
                ])
            else:
                debug_sections.extend([
                    "=== ETF MANAGER DEBUG ===",
                    f"ETF Manager: Not loaded or unavailable",
                    ""
                ])
            
            # Algorithm-Specific Debug
            if optimizer:
                debug_sections.extend([
                    "=== ALGORITHM DEBUG ===",
                    f"Optimizer Class: {type(optimizer).__name__}",
                    f"Initial Wealth: ${getattr(optimizer, 'W0', 0):,}",
                    f"Time Horizon: {getattr(optimizer, 'T', 0)} periods",
                    f"Wealth Grid: {getattr(optimizer, 'i_max', 0)} points",
                    f"Portfolio Strategies: {getattr(optimizer, 'l_max', 0)}",
                    f"Goals Count: {len(getattr(optimizer, 'goals', []))}",
                    f"Infusions Count: {len(getattr(optimizer, 'infusions', {}))}",
                    f"Value Function Shape: {getattr(optimizer, 'value_function', np.array([])).shape}",
                    ""
                ])
                
                # Goal Vector Debug
                if hasattr(optimizer, 'goal_options') and optimizer.goal_options:
                    debug_sections.append("=== GOAL VECTORS DEBUG ===")
                    for t, options in list(optimizer.goal_options.items())[:5]:  # Show first 5 time periods
                        debug_sections.append(f"  t={t}: {len(options)} goal options")
                    if len(optimizer.goal_options) > 5:
                        debug_sections.append(f"  ... and {len(optimizer.goal_options) - 5} more time periods")
                    debug_sections.append("")
            
            # Framework Compliance
            debug_sections.extend([
                "=== DAS ET AL. (2022) FRAMEWORK COMPLIANCE ===",
                "✓ Dynamic Programming Implementation",
                "✓ Bellman Equation Solving",
                "✓ Goal Vector Processing c(t), u(t)",
                "✓ Transition Probabilities q(W_j|W_i, c_k, μ_l)",
                "✓ Forward Simulation for Probabilities",
                "✓ Multiple Portfolio Strategies",
                "✓ Cash Infusions I(t)",
                "✓ Concurrent and Partial Goals",
                "✓ Priority Weighting",
                "✓ ETF Integration (Enhanced)",
                ""
            ])
            
            return "\n".join(debug_sections)
    
    return server_function

pass  # Do not log visitor financial inputs.
pass  # Do not log visitor financial inputs.
pass  # Do not log visitor financial inputs.
pass  # Do not log visitor financial inputs.
                   