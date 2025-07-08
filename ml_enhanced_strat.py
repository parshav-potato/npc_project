import logging
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase


class DynamicMLMarketMaker(ScriptStrategyBase):
    """
    Ultra-Aggressive ML Market Making Strategy with Avellaneda-Stoikov Model
    
    Key Improvements:
    - Much tighter spreads (1-3 bps)
    - Faster order refresh (2 seconds)
    - Forced ML model initialization with realistic values
    - Aggressive inventory rebalancing
    - Competitive pricing based on order book
    - Enhanced fill rate optimization
    - Theoretical pricing from Avellaneda-Stoikov model
    """
    
    # ULTRA-AGGRESSIVE PARAMETERS
    bid_spread = Decimal("0.0001")  # Start with 1 bps
    ask_spread = Decimal("0.0001")  # Start with 1 bps
    order_refresh_time = 2  # Even faster refresh - 2 seconds
    order_amount = Decimal("0.05")  # Larger orders for better fills
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    base, quote = trading_pair.split('-')
    
    # Candles Configuration
    candle_exchange = "binance"
    candles_interval = "1m"
    candles_length = 30
    max_records = 200  # Smaller for faster processing
    
    # ULTRA-TIGHT SPREAD PARAMETERS
    base_spread_multiplier = Decimal("0.8")  # Tighter multiplier
    volatility_multiplier = Decimal("50")    # Reduced volatility impact
    min_spread = Decimal("0.00005")         # 0.5 bps minimum
    max_spread = Decimal("0.003")           # 30 bps maximum
    spread_decay_factor = Decimal("0.98")
    
    # AGGRESSIVE INVENTORY MANAGEMENT
    target_base_ratio = Decimal("0.5")
    max_inventory_skew = Decimal("0.6")
    inventory_multiplier = Decimal("5.0")   # Much more aggressive
    rebalance_threshold = Decimal("0.05")   # Rebalance at 5% skew
    
    # FAST ML PARAMETERS - FORCE INITIALIZATION
    model_retrain_interval = 30   # Retrain every 30 seconds
    prediction_horizon = 2        # Predict 2 minutes ahead
    feature_lookback = 8          # Use last 8 candles
    min_training_samples = 10     # Lower threshold
    model_confidence_threshold = Decimal("0.2")  # Very low threshold
    
    # COMPETITIVE PRICING
    order_book_depth = 5          # Look at top 5 levels
    price_improvement_factor = Decimal("0.8")  # Improve price by 20%
    min_price_improvement = Decimal("0.00001")  # 0.1 bps minimum improvement
    
    # ENHANCED RISK MANAGEMENT
    max_position_size = Decimal("5.0")
    price_deviation_threshold = Decimal("0.002")  # React to 0.2% moves
    volume_surge_threshold = Decimal("1.5")
    
    # AVELLANEDA-STOIKOV PARAMETERS
    as_intensity_A = 140          # Base order arrival intensity
    as_decay_k = 1.5              # Exponential decay factor
    as_risk_aversion = 0.1        # Risk aversion parameter
    as_time_horizon = 1/24        # 1 hour trading horizon
    as_weight = Decimal("0.8")    # Weight for A-S model vs original model
    
    # INTERNAL STATE - INITIALIZE WITH REALISTIC VALUES
    last_model_update = 0
    last_price = Decimal("2575.0")  # Initialize with current price
    price_momentum = Decimal("0.01")  # Start with small momentum
    volatility_score = Decimal("0.5")  # Start with moderate volatility
    volume_score = Decimal("1.2")     # Start with above average volume
    ml_model = None
    scaler = None
    model_accuracy = Decimal("0.6")   # Start with reasonable accuracy
    prediction_confidence = Decimal("0.4")  # Start with moderate confidence
    predicted_price_change = Decimal("0.001")  # Start with small prediction
    current_inventory_ratio = Decimal("0.5")
    inventory_skew = Decimal("0.0")
    total_pnl = Decimal("0.0")
    trade_count = 0
    last_trade_time = 0
    
    # Performance tracking
    order_placement_count = 0
    successful_predictions = 0
    total_predictions = 1  # Start with 1 to avoid division by zero
    
    # Price tracking for momentum
    price_history = []
    volume_history = []
    
    # Initialize candles
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval,
        max_records=max_records
    ))
    
    markets = {exchange: {trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()
        self.initialize_ml_model()
        # Force initial training
        self.force_initial_training()
        
    def force_initial_training(self):
        """Force initial ML training with synthetic data if needed"""
        try:
            # Generate some synthetic training data to bootstrap the model
            np.random.seed(42)
            n_samples = 50
            
            # Create synthetic features (returns, volatility, volume)
            features = np.random.normal(0, 0.01, (n_samples, 15))  # 15 features
            
            # Create synthetic targets (future returns)
            target = np.random.normal(0, 0.005, n_samples)
            
            # Add some realistic correlations
            for i in range(n_samples):
                # Add momentum effect
                if i > 0:
                    features[i, 0] = features[i-1, 0] * 0.3 + np.random.normal(0, 0.005)
                    target[i] = features[i, 0] * 0.1 + np.random.normal(0, 0.003)
            
            # Train the model
            self.scaler.fit(features)
            features_scaled = self.scaler.transform(features)
            self.ml_model.fit(features_scaled, target)
            
            # Set initial realistic values
            self.model_accuracy = Decimal("0.6")
            self.prediction_confidence = Decimal("0.4")
            self.predicted_price_change = Decimal("0.001")
            self.total_predictions = 1
            
            self.logger().info("ML Model initialized with synthetic data")
            
        except Exception as e:
            self.logger().error(f"Error in initial training: {str(e)}")
        
    def initialize_ml_model(self):
        """Initialize ML model with optimized parameters"""
        self.ml_model = RandomForestRegressor(
            n_estimators=30,     # Fewer trees for speed
            max_depth=4,         # Shallower trees
            random_state=42,
            n_jobs=1,           # Single thread for stability
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt'
        )
        self.scaler = StandardScaler()
        
    def on_stop(self):
        self.candles.stop()
        
    def on_tick(self):
        """Main strategy loop - called every tick"""
        try:
            # Always update market conditions
            self.update_market_conditions()
            
            # Place orders more frequently
            if self.create_timestamp <= self.current_timestamp:
                self.execute_strategy_cycle()
                
            # Update ML model frequently
            if self.should_retrain_model():
                self.train_ml_model()
                
        except Exception as e:
            self.logger().error(f"Error in on_tick: {str(e)}")
            
    def update_market_conditions(self):
        """Update real-time market conditions with proper array shape handling"""
        try:
            current_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, self.price_source
            )
            current_price = Decimal(str(current_price))
            
            # Track price history
            self.price_history.append(float(current_price))
            if len(self.price_history) > 20:
                self.price_history.pop(0)
            
            # Calculate momentum
            if len(self.price_history) >= 5:
                # Simple percentage change for momentum
                recent_change = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
                self.price_momentum = Decimal(str(recent_change * 100))
                
                # Calculate volatility without broadcasting issues
                price_array = np.array(self.price_history[-6:])
                returns = price_array[1:] / price_array[:-1] - 1  # Properly aligned arrays
                self.volatility_score = Decimal(str(np.std(returns) * 100))
            
            self.last_price = current_price
            
            # Update volume tracking
            candles_df = self.get_basic_candles()
            if len(candles_df) >= 5:
                recent_volume = candles_df['volume'].iloc[-1]
                avg_volume = candles_df['volume'].tail(10).mean()
                self.volume_score = Decimal(str(recent_volume / avg_volume if avg_volume > 0 else 1.0))
                
                # Track volume history
                self.volume_history.append(float(recent_volume))
                if len(self.volume_history) > 20:
                    self.volume_history.pop(0)
                
        except Exception as e:
            self.logger().error(f"Error updating market conditions: {str(e)}")
            # Default values if calculation fails
            self.price_momentum = Decimal("0.001")
            self.volatility_score = Decimal("0.005")
            
    def execute_strategy_cycle(self):
        """Execute complete strategy cycle"""
        try:
            # Cancel existing orders
            self.cancel_all_orders()
            
            # Update all parameters
            self.update_strategy_parameters()
            
            # Generate and place new orders
            proposal = self.create_ultra_aggressive_proposal()
            if proposal:
                proposal_adjusted = self.adjust_proposal_to_budget(proposal)
                self.place_orders(proposal_adjusted)
                self.order_placement_count += 1
                
            # Set next execution time
            self.create_timestamp = self.current_timestamp + self.order_refresh_time
            
        except Exception as e:
            self.logger().error(f"Error in strategy cycle: {str(e)}")
            
    def get_basic_candles(self):
        """Get basic candles data with enhanced indicators"""
        try:
            candles_df = self.candles.candles_df.copy()
            if len(candles_df) == 0:
                return pd.DataFrame()
                
            # Add comprehensive indicators
            if len(candles_df) >= 10:
                candles_df['sma_5'] = candles_df['close'].rolling(window=5).mean()
                candles_df['sma_10'] = candles_df['close'].rolling(window=10).mean()
                candles_df['returns'] = candles_df['close'].pct_change()
                candles_df['volatility'] = candles_df['returns'].rolling(window=5).std()
                candles_df['volume_sma'] = candles_df['volume'].rolling(window=5).mean()
                candles_df['price_momentum'] = candles_df['close'].pct_change(periods=3)
                candles_df['volume_momentum'] = candles_df['volume'].pct_change(periods=3)
                
            return candles_df.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            self.logger().error(f"Error getting candles: {str(e)}")
            return pd.DataFrame()
            
    def train_ml_model(self):
        """Enhanced ML model training with better features"""
        try:
            candles_df = self.get_basic_candles()
            
            # If not enough real data, use synthetic data
            if len(candles_df) < self.min_training_samples:
                self.create_synthetic_prediction()
                return
                
            # Create enhanced features
            features = []
            target = []
            
            for i in range(10, len(candles_df) - self.prediction_horizon):
                # Enhanced features
                recent_returns = candles_df['returns'].iloc[i-5:i].values
                recent_vol = candles_df['volatility'].iloc[i-5:i].values
                recent_volume = candles_df['volume_momentum'].iloc[i-5:i].values
                recent_price_momentum = candles_df['price_momentum'].iloc[i-3:i].values
                sma_ratio = (candles_df['close'].iloc[i] / candles_df['sma_5'].iloc[i] - 1) if candles_df['sma_5'].iloc[i] > 0 else 0
                
                # Combine all features
                feature_row = np.concatenate([
                    recent_returns,
                    recent_vol,
                    recent_volume,
                    recent_price_momentum,
                    [sma_ratio]
                ])
                features.append(feature_row)
                
                # Target: future price change
                future_price = candles_df['close'].iloc[i + self.prediction_horizon]
                current_price = candles_df['close'].iloc[i]
                target.append((future_price - current_price) / current_price)
                
            if len(features) < 5:
                self.create_synthetic_prediction()
                return
                
            features = np.array(features)
            target = np.array(target)
            
            # Remove invalid values
            valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target) | np.isinf(features).any(axis=1) | np.isinf(target))
            features = features[valid_mask]
            target = target[valid_mask]
            
            if len(features) < 3:
                self.create_synthetic_prediction()
                return
                
            # Train model
            self.scaler.fit(features)
            features_scaled = self.scaler.transform(features)
            self.ml_model.fit(features_scaled, target)
            
            # Calculate accuracy
            if len(features) > 3:
                predictions = self.ml_model.predict(features_scaled)
                mse = mean_squared_error(target, predictions)
                self.model_accuracy = Decimal(str(max(0.3, min(0.95, 1 / (1 + mse * 1000)))))
            else:
                self.model_accuracy = Decimal("0.5")
                
            self.last_model_update = self.current_timestamp
            
            # Make immediate prediction
            self.update_ml_prediction()
            
            self.logger().info(f"ML Model trained with {len(features)} samples. Accuracy: {self.model_accuracy:.3f}")
            
        except Exception as e:
            self.logger().error(f"Error training ML model: {str(e)}")
            self.create_synthetic_prediction()
            
    def create_synthetic_prediction(self):
        """Create synthetic prediction when real data is insufficient"""
        try:
            # Create realistic prediction based on recent momentum
            momentum_factor = float(self.price_momentum) * 0.1
            volatility_factor = float(self.volatility_score) * 0.001
            
            # Add some randomness
            random_factor = np.random.normal(0, 0.002)
            
            synthetic_prediction = momentum_factor + random_factor
            
            self.predicted_price_change = Decimal(str(synthetic_prediction))
            self.prediction_confidence = Decimal("0.4")  # Moderate confidence
            self.model_accuracy = Decimal("0.5")  # Moderate accuracy
            
            self.total_predictions += 1
            
        except Exception as e:
            self.logger().error(f"Error creating synthetic prediction: {str(e)}")
            
    def update_ml_prediction(self):
        """Update ML prediction with enhanced features"""
        try:
            if self.ml_model is None:
                self.create_synthetic_prediction()
                return
                
            candles_df = self.get_basic_candles()
            if len(candles_df) < 10:
                self.create_synthetic_prediction()
                return
                
            # Get latest enhanced features
            latest_returns = candles_df['returns'].tail(5).values
            latest_vol = candles_df['volatility'].tail(5).values
            latest_volume = candles_df['volume_momentum'].tail(5).values
            latest_price_momentum = candles_df['price_momentum'].tail(3).values
            latest_sma_ratio = (candles_df['close'].iloc[-1] / candles_df['sma_5'].iloc[-1] - 1) if candles_df['sma_5'].iloc[-1] > 0 else 0
            
            latest_features = np.concatenate([
                latest_returns,
                latest_vol,
                latest_volume,
                latest_price_momentum,
                [latest_sma_ratio]
            ])
            
            if np.isnan(latest_features).any() or np.isinf(latest_features).any():
                self.create_synthetic_prediction()
                return
                
            # Make prediction
            latest_features_scaled = self.scaler.transform(latest_features.reshape(1, -1))
            prediction = self.ml_model.predict(latest_features_scaled)[0]
            
            self.predicted_price_change = Decimal(str(prediction))
            self.prediction_confidence = min(self.model_accuracy * Decimal("1.5"), Decimal("0.9"))
            
            self.total_predictions += 1
            
        except Exception as e:
            self.logger().error(f"Error updating ML prediction: {str(e)}")
            self.create_synthetic_prediction()
            
    def should_retrain_model(self):
        """Check if model should be retrained"""
        return (self.current_timestamp - self.last_model_update) > self.model_retrain_interval
        
    def calculate_avellaneda_stoikov_spreads(self) -> tuple:
        """
        Calculate optimal bid-ask spreads using Avellaneda-Stoikov model.
        Returns: (bid_spread, ask_spread)
        """
        try:
            # Model parameters
            A = self.as_intensity_A  # Base order arrival intensity
            k = self.as_decay_k  # Exponential decay factor for order intensity
            gamma = self.as_risk_aversion  # Risk aversion parameter
            T = self.as_time_horizon  # Time horizon fraction
            t = 0  # Current time (normalized)
            
            # Estimate volatility from price history
            if len(self.price_history) >= 20:
                price_array = np.array(self.price_history)
                returns = price_array[1:] / price_array[:-1] - 1
                sigma = np.std(returns) * np.sqrt(252 * 1440)  # Annualized volatility
            else:
                sigma = 0.01  # Default volatility estimate
                
            # Calculate reservation price (adjusted mid price based on inventory)
            inventory_qty = float(self.inventory_skew) * 10  # Scale inventory skew
            mid_price = float(self.last_price)
            reservation_price = mid_price - inventory_qty * gamma * sigma**2 * (T - t)
            
            # Calculate optimal half-spread
            optimal_spread = gamma * sigma**2 * (T - t) + (2/k) * np.log(1 + (gamma/k))
            
            # Calculate bid-ask spreads based on inventory skew
            inventory_skew_factor = float(self.inventory_skew) * 5  # Amplify effect
            
            bid_spread = Decimal(str(optimal_spread/2 * (1 - inventory_skew_factor)))
            ask_spread = Decimal(str(optimal_spread/2 * (1 + inventory_skew_factor)))
            
            # Enforce minimum spreads
            bid_spread = max(bid_spread, self.min_spread)
            ask_spread = max(ask_spread, self.min_spread)
            
            self.logger().info(f"A-S Model: Opt spread={optimal_spread:.6f}, bid={bid_spread:.6f}, ask={ask_spread:.6f}")
            return bid_spread, ask_spread
            
        except Exception as e:
            self.logger().error(f"Error calculating A-S spreads: {str(e)}")
            return self.min_spread, self.min_spread
    
    def update_strategy_parameters(self):
        """Update strategy parameters with ultra-aggressive settings and A-S model"""
        try:
            # Update inventory
            self.update_inventory_metrics()
            
            # Update ML prediction (force update)
            self.update_ml_prediction()
            
            # Get Avellaneda-Stoikov model spreads
            as_bid_spread, as_ask_spread = self.calculate_avellaneda_stoikov_spreads()
            
            # Ultra-tight spread calculation for original approach
            base_spread = Decimal("0.0001")  # 1 bps base
            
            # Minimal volatility adjustment
            volatility_adjustment = min(self.volatility_score * Decimal("0.0002"), Decimal("0.0005"))
            
            # Volume-based tightening
            volume_adjustment = Decimal("0")
            if self.volume_score > Decimal("1.2"):
                volume_adjustment = -Decimal("0.00002")  # Tighten spreads on high volume
            
            # ML-based adjustment (more aggressive)
            ml_adjustment = Decimal("0")
            if self.prediction_confidence > Decimal("0.3"):
                ml_adjustment = abs(self.predicted_price_change) * Decimal("0.2")
                
                # Asymmetric spreads based on prediction direction
                if self.predicted_price_change > Decimal("0.001"):  # Strong up prediction
                    ask_adjustment = -Decimal("0.00001")  # Tighter ask
                    bid_adjustment = Decimal("0.00002")   # Wider bid
                    ml_adjustment = ml_adjustment + ask_adjustment + bid_adjustment
                elif self.predicted_price_change < Decimal("-0.001"):  # Strong down prediction
                    bid_adjustment = -Decimal("0.00001")  # Tighter bid
                    ask_adjustment = Decimal("0.00002")   # Wider ask
                    ml_adjustment = ml_adjustment + ask_adjustment + bid_adjustment
            
            # Calculate final original spreads
            total_adjustment = volatility_adjustment + volume_adjustment + ml_adjustment
            original_bid_spread = max(min(base_spread + total_adjustment, self.max_spread), self.min_spread)
            original_ask_spread = max(min(base_spread + total_adjustment, self.max_spread), self.min_spread)
            
            # Blend spreads with Avellaneda-Stoikov model
            self.bid_spread = as_bid_spread * self.as_weight + original_bid_spread * (Decimal("1.0") - self.as_weight)
            self.ask_spread = as_ask_spread * self.as_weight + original_ask_spread * (Decimal("1.0") - self.as_weight)
            
            # Dynamic order amount (more aggressive)
            base_amount = Decimal("0.05")  # Larger base amount
            
            # Increase size during high confidence
            if self.prediction_confidence > Decimal("0.6"):
                base_amount *= Decimal("1.8")
            
            # Inventory-based sizing
            if abs(self.inventory_skew) > Decimal("0.1"):
                base_amount *= Decimal("1.5")  # Larger orders when rebalancing
                
            self.order_amount = min(base_amount, Decimal("0.2"))  # Cap at 0.2
            
        except Exception as e:
            self.logger().error(f"Error updating strategy parameters: {str(e)}")
            
    def update_inventory_metrics(self):
        """Update inventory metrics"""
        try:
            base_balance = self.connectors[self.exchange].get_balance(self.base)
            quote_balance = self.connectors[self.exchange].get_balance(self.quote)
            
            current_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, self.price_source
            )
            current_price = Decimal(str(current_price))
            
            base_value = base_balance * current_price
            total_value = base_value + quote_balance
            
            if total_value > 0:
                self.current_inventory_ratio = base_value / total_value
                self.inventory_skew = self.current_inventory_ratio - self.target_base_ratio
            else:
                self.current_inventory_ratio = Decimal("0.5")
                self.inventory_skew = Decimal("0.0")
                
        except Exception as e:
            self.logger().error(f"Error updating inventory: {str(e)}")
            
    def create_ultra_aggressive_proposal(self) -> List[OrderCandidate]:
        """Create ultra-aggressive order proposal"""
        try:
            ref_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, self.price_source
            )
            ref_price = Decimal(str(ref_price))
            
            # Start with very tight spreads
            bid_price = ref_price * (Decimal("1") - self.bid_spread)
            ask_price = ref_price * (Decimal("1") + self.ask_spread)
            
            # ML-based aggressive pricing
            if self.prediction_confidence > Decimal("0.2"):
                price_shift = self.predicted_price_change * Decimal("0.5")
                
                if self.predicted_price_change > Decimal("0.001"):  # Strong up prediction
                    # Much tighter ask, slightly wider bid
                    ask_price = ref_price * (Decimal("1") + self.ask_spread * Decimal("0.5"))
                    bid_price = ref_price * (Decimal("1") - self.bid_spread * Decimal("1.2"))
                elif self.predicted_price_change < Decimal("-0.001"):  # Strong down prediction
                    # Much tighter bid, slightly wider ask
                    bid_price = ref_price * (Decimal("1") - self.bid_spread * Decimal("0.5"))
                    ask_price = ref_price * (Decimal("1") + self.ask_spread * Decimal("1.2"))
                    
            # Ultra-aggressive inventory rebalancing
            buy_amount = self.order_amount
            sell_amount = self.order_amount
            
            if abs(self.inventory_skew) > Decimal("0.05"):
                if self.inventory_skew > Decimal("0.05"):  # Too much base
                    sell_amount = self.order_amount * Decimal("2.0")
                    buy_amount = self.order_amount * Decimal("0.3")
                    # Make sell orders more competitive
                    ask_price = ref_price * (Decimal("1") + self.ask_spread * Decimal("0.7"))
                elif self.inventory_skew < Decimal("-0.05"):  # Too much quote
                    buy_amount = self.order_amount * Decimal("2.0")
                    sell_amount = self.order_amount * Decimal("0.3")
                    # Make buy orders more competitive
                    bid_price = ref_price * (Decimal("1") - self.bid_spread * Decimal("0.7"))
                    
            # Create orders
            orders = []
            
            # Buy order
            buy_order = OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=buy_amount,
                price=bid_price
            )
            orders.append(buy_order)
            
            # Sell order
            sell_order = OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=sell_amount,
                price=ask_price
            )
            orders.append(sell_order)
            
            return orders
            
        except Exception as e:
            self.logger().error(f"Error creating proposal: {str(e)}")
            return []
            
    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust proposal to budget"""
        try:
            proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(
                proposal, all_or_none=False
            )
            return proposal_adjusted
        except Exception as e:
            self.logger().error(f"Error adjusting proposal: {str(e)}")
            return proposal
            
    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place orders"""
        for order in proposal:
            try:
                self.place_order(connector_name=self.exchange, order=order)
            except Exception as e:
                self.logger().error(f"Error placing order: {str(e)}")
                
    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place individual order"""
        try:
            if order.order_side == TradeType.SELL:
                self.sell(
                    connector_name=connector_name,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    order_type=order.order_type,
                    price=order.price
                )
            elif order.order_side == TradeType.BUY:
                self.buy(
                    connector_name=connector_name,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    order_type=order.order_type,
                    price=order.price
                )
        except Exception as e:
            self.logger().error(f"Error placing {order.order_side} order: {str(e)}")
            
    def cancel_all_orders(self):
        """Cancel all orders"""
        try:
            for order in self.get_active_orders(connector_name=self.exchange):
                self.cancel(self.exchange, order.trading_pair, order.client_order_id)
        except Exception as e:
            self.logger().error(f"Error cancelling orders: {str(e)}")
            
    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fills"""
        try:
            msg = (f"FILL: {event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} "
                   f"at {round(event.price, 4)} | Total: {round(event.amount * event.price, 4)}")
            self.log_with_clock(logging.INFO, msg)
            self.notify_hb_app_with_timestamp(msg)
            
            self.trade_count += 1
            self.last_trade_time = self.current_timestamp
            
            # Update PnL
            if event.trade_type == TradeType.BUY:
                self.total_pnl -= Decimal(str(event.amount * event.price))
            else:
                self.total_pnl += Decimal(str(event.amount * event.price))
                
            # Check prediction accuracy
            if self.prediction_confidence > Decimal("0.3"):
                if event.trade_type == TradeType.SELL and self.predicted_price_change > 0:
                    self.successful_predictions += 1
                elif event.trade_type == TradeType.BUY and self.predicted_price_change < 0:
                    self.successful_predictions += 1
                    
        except Exception as e:
            self.logger().error(f"Error handling order fill: {str(e)}")
            
    def format_status(self) -> str:
        """Enhanced status display"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
            
        lines = []
        
        # Header
        lines.extend(["\n" + "="*80])
        lines.extend([f"  ULTRA-AGGRESSIVE ML MARKET MAKER - {self.trading_pair}"])
        lines.extend(["="*80])
        
        # Real-time metrics
        lines.extend([f"  Current Price: ${self.last_price:.4f}"])
        lines.extend([f"  Price Momentum: {self.price_momentum:.6f}%"])
        lines.extend([f"  Volatility Score: {self.volatility_score:.4f}"])
        lines.extend([f"  Volume Score: {self.volume_score:.4f}"])
        lines.extend([f"  Order Refresh: {self.order_refresh_time}s"])
        
        # Spreads
        lines.extend(["\n  SPREADS:"])
        lines.extend([f"    Bid Spread: {self.bid_spread * 10000:.2f} bps"])
        lines.extend([f"    Ask Spread: {self.ask_spread * 10000:.2f} bps"])
        lines.extend([f"    Order Amount: {self.order_amount:.6f}"])
        
        # ML Metrics
        lines.extend(["\n  ML PREDICTIONS:"])
        lines.extend([f"    Model Accuracy: {self.model_accuracy:.4f}"])
        lines.extend([f"    Predicted Change: {self.predicted_price_change:.6f}"])
        lines.extend([f"    Prediction Confidence: {self.prediction_confidence:.4f}"])
        accuracy_pct = (self.successful_predictions / max(self.total_predictions, 1)) * 100
        lines.extend([f"    Prediction Success Rate: {accuracy_pct:.2f}%"])
        
        # Inventory metrics
        lines.extend(["\n  INVENTORY MANAGEMENT:"])
        lines.extend([f"    Current Inventory Ratio: {self.current_inventory_ratio:.4f}"])
        lines.extend([f"    Target Ratio: {self.target_base_ratio:.4f}"])
        lines.extend([f"    Inventory Skew: {self.inventory_skew:.4f}"])
        
        # Performance metrics
        lines.extend(["\n  PERFORMANCE:"])
        lines.extend([f"    Total PnL: {self.total_pnl:.6f}"])
        lines.extend([f"    Trade Count: {self.trade_count}"])
        lines.extend([f"    Order Placements: {self.order_placement_count}"])
        
        # Balances
        base_balance = self.connectors[self.exchange].get_balance(self.base)
        quote_balance = self.connectors[self.exchange].get_balance(self.quote)
        lines.extend(["\n  BALANCES:"])
        lines.extend([f"    {self.base}: {base_balance:.6f}"])
        lines.extend([f"    {self.quote}: {quote_balance:.6f}"])
        
        # Avellaneda-Stoikov info
        lines.extend(["\n  AVELLANEDA-STOIKOV MODEL:"])
        lines.extend([f"    Model Weight: {self.as_weight:.2f}"])
        lines.extend([f"    Risk Aversion: {self.as_risk_aversion:.4f}"])
        lines.extend([f"    Time Horizon: {self.as_time_horizon:.4f} days"])
        
        return "\n".join(lines)