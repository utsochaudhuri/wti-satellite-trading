import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class FixedCloudTradingModel:
    def __init__(self, initial_capital=10000, max_position_size=0.1, stop_loss=0.05, take_profit=0.1):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trades = []
        self.portfolio_value = []
        
    def load_and_prepare_data(self):
        """Load and clean the data from output.csv"""
        df = pd.read_csv('output.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        numeric_columns = ['cloud_coverage_percent', 'wti_price']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Loaded {len(df)} rows from output.csv")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Non-null cloud coverage data points: {df['cloud_coverage_percent'].notna().sum()}")
        print(f"Non-null price data points: {df['wti_price'].notna().sum()}")
        
        return df
    
    def create_features(self, df):
        """Create features with strict time-series constraints - NO LOOK-AHEAD"""
        # Fill missing cloud coverage data with forward-looking method removed
        df['cloud_coverage_filled'] = df['cloud_coverage_percent'].fillna(method='ffill')
        
        # ALL features use only PAST data (lagged)
        df['cloud_coverage_lag1'] = df['cloud_coverage_filled'].shift(1)
        df['cloud_coverage_lag2'] = df['cloud_coverage_filled'].shift(2) 
        df['cloud_coverage_lag3'] = df['cloud_coverage_filled'].shift(3)
        
        # Moving averages - using ONLY past data
        df['cloud_coverage_ma3'] = df['cloud_coverage_filled'].shift(1).rolling(window=3, min_periods=1).mean()
        df['cloud_coverage_ma7'] = df['cloud_coverage_filled'].shift(1).rolling(window=7, min_periods=1).mean()
        df['cloud_coverage_ma14'] = df['cloud_coverage_filled'].shift(1).rolling(window=14, min_periods=1).mean()
        
        # Changes and volatility - using ONLY past data
        df['cloud_coverage_change'] = df['cloud_coverage_filled'].shift(1).diff()
        df['cloud_coverage_volatility'] = df['cloud_coverage_filled'].shift(1).rolling(window=7, min_periods=1).std()
        
        # Work only with rows that have price data
        df_with_price = df[df['wti_price'].notna()].copy().reset_index(drop=True)
        
        if len(df_with_price) > 1:
            # Price features - ALL using ONLY past data
            df_with_price['price_return_lag1'] = df_with_price['wti_price'].pct_change().shift(1)
            df_with_price['price_return_lag2'] = df_with_price['wti_price'].pct_change().shift(2)
            df_with_price['price_volatility'] = df_with_price['wti_price'].pct_change().shift(1).rolling(window=10, min_periods=1).std()
            df_with_price['price_ma5'] = df_with_price['wti_price'].shift(1).rolling(window=5, min_periods=1).mean()
            df_with_price['price_ma20'] = df_with_price['wti_price'].shift(1).rolling(window=20, min_periods=1).mean()
            
            # Price momentum using only past data
            df_with_price['price_momentum'] = (df_with_price['wti_price'].shift(1) / df_with_price['price_ma5'] - 1)
            
            # Volatility regime - using only past data
            volatility_rolling = df_with_price['wti_price'].pct_change().shift(1).rolling(window=20, min_periods=1).std()
            volatility_threshold = volatility_rolling.shift(1).rolling(window=50, min_periods=1).quantile(0.75)
            df_with_price['volatility_regime'] = np.where(volatility_rolling > volatility_threshold, 1, 0)
            
            # TARGET VARIABLES - what we want to predict (FUTURE data)
            df_with_price['next_price'] = df_with_price['wti_price'].shift(-1)
            df_with_price['future_return'] = (df_with_price['next_price'] / df_with_price['wti_price'] - 1)
            df_with_price['price_direction'] = (df_with_price['future_return'] > 0).astype(int)
            
            # Remove last row since we don't have future price
            df_with_price = df_with_price.iloc[:-1]
        
        return df, df_with_price
    
    def walk_forward_analysis(self, df_with_price):
        """Proper walk-forward analysis to prevent data leakage"""
        # Define feature columns - only lagged features
        feature_cols = [
            'cloud_coverage_lag1', 'cloud_coverage_lag2', 'cloud_coverage_lag3',
            'cloud_coverage_ma3', 'cloud_coverage_ma7', 'cloud_coverage_ma14',
            'cloud_coverage_change', 'cloud_coverage_volatility',
            'price_return_lag1', 'price_return_lag2', 'price_volatility', 'price_momentum'
        ]
        
        # Remove rows with NaN and ensure we have enough data
        clean_data = df_with_price[feature_cols + ['price_direction', 'future_return', 'date', 'wti_price']].dropna()
        
        if len(clean_data) < 50:
            print("Warning: Not enough clean data points for walk-forward analysis.")
            return None
        
        # Remove first 30 rows to ensure all moving averages are stable
        clean_data = clean_data.iloc[30:].reset_index(drop=True)
        
        # Parameters for walk-forward
        min_train_size = 30  # Minimum training window
        test_size = 1  # Test one period at a time
        results = []
        
        print(f"Starting walk-forward analysis on {len(clean_data)} data points...")
        
        for i in range(min_train_size, len(clean_data)):
            # Training data: all data up to current point
            train_data = clean_data.iloc[:i]
            # Test data: current point only
            test_data = clean_data.iloc[i:i+1]
            
            if len(train_data) < min_train_size:
                continue
                
            # Prepare training data
            X_train = train_data[feature_cols].fillna(0)
            y_train_class = train_data['price_direction']
            y_train_reg = train_data['future_return']
            
            # Prepare test data  
            X_test = test_data[feature_cols].fillna(0)
            
            try:
                # Train models on historical data only
                rf_class = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                rf_reg = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                
                rf_class.fit(X_train, y_train_class)
                rf_reg.fit(X_train, y_train_reg)
                
                # Make predictions
                direction_prob = rf_class.predict_proba(X_test)
                predicted_return = rf_reg.predict(X_test)
                
                # Store results
                result_row = test_data.iloc[0].copy()
                result_row['direction_prob_up'] = direction_prob[0, 1] if direction_prob.shape[1] > 1 else 0.5
                result_row['predicted_return'] = predicted_return[0]
                
                results.append(result_row)
                
            except Exception as e:
                print(f"Error at step {i}: {e}")
                continue
        
        if not results:
            return None
            
        results_df = pd.DataFrame(results).reset_index(drop=True)
        print(f"Walk-forward analysis completed. Generated {len(results_df)} predictions.")
        return results_df
    
    def generate_adaptive_signals(self, df_predictions):
        """Generate adaptive trading signals that adjust to market regimes"""
        df_predictions = df_predictions.copy()
        df_predictions['signal'] = 0
        
        # Calculate rolling volatility and market regime
        if 'price_volatility' in df_predictions.columns:
            vol_series = df_predictions['price_volatility'].fillna(0.02)
        else:
            vol_series = pd.Series([0.02] * len(df_predictions))
        
        # Define market regimes based on volatility - fix the quantile calculation
        rolling_vol = vol_series.rolling(window=min(250, len(vol_series)), min_periods=min(50, len(vol_series)))
        low_vol_threshold = rolling_vol.quantile(0.33)
        high_vol_threshold = rolling_vol.quantile(0.67)
        
        # Fill NaN values for thresholds
        low_vol_threshold = low_vol_threshold.fillna(vol_series.quantile(0.33))
        high_vol_threshold = high_vol_threshold.fillna(vol_series.quantile(0.67))
        
        # Adaptive thresholds based on market regime
        base_confidence_high = 0.58
        base_confidence_low = 0.42
        base_min_return = 0.005
        
        for i in range(len(df_predictions)):
            current_vol = vol_series.iloc[i] if i < len(vol_series) else 0.02
            low_thresh = low_vol_threshold.iloc[i] if i < len(low_vol_threshold) else vol_series.quantile(0.33)
            high_thresh = high_vol_threshold.iloc[i] if i < len(high_vol_threshold) else vol_series.quantile(0.67)
            
            # Adjust thresholds based on volatility regime
            if current_vol < low_thresh:  # Low volatility - be more aggressive
                confidence_high = base_confidence_high - 0.05  # 0.53
                confidence_low = base_confidence_low + 0.05   # 0.47
                min_return = base_min_return * 0.7  # 0.0035
            elif current_vol > high_thresh:  # High volatility - be more conservative
                confidence_high = base_confidence_high + 0.07  # 0.65
                confidence_low = base_confidence_low - 0.07   # 0.35
                min_return = base_min_return * 1.5  # 0.0075
            else:  # Normal volatility
                confidence_high = base_confidence_high
                confidence_low = base_confidence_low
                min_return = base_min_return
            
            # Get current row data
            row = df_predictions.iloc[i]
            prob_up = row.get('direction_prob_up', 0.5)
            pred_return = row.get('predicted_return', 0)
            momentum = row.get('price_momentum', 0)
            
            # Primary signals
            long_condition = (prob_up > confidence_high) & (pred_return > min_return)
            short_condition = (prob_up < confidence_low) & (pred_return < -min_return)
            
            # Additional momentum-based signals for low-volatility periods
            if current_vol < low_thresh * 1.2:  # More momentum signals in calmer markets
                momentum_long = (prob_up > 0.54) & (pred_return > 0.003) & (momentum > 0.008)
                momentum_short = (prob_up < 0.46) & (pred_return < -0.003) & (momentum < -0.008)
                
                long_condition = long_condition | momentum_long
                short_condition = short_condition | momentum_short
            
            # Mean reversion signals for very low volatility
            if current_vol < low_thresh * 0.8:  # Very low volatility
                # Look for oversold/overbought conditions
                if prob_up < 0.35 and pred_return > 0.002:  # Oversold bounce
                    long_condition = True
                elif prob_up > 0.65 and pred_return < -0.002:  # Overbought decline
                    short_condition = True
            
            # Set signals
            if long_condition:
                df_predictions.iloc[i, df_predictions.columns.get_loc('signal')] = 1
            elif short_condition:
                df_predictions.iloc[i, df_predictions.columns.get_loc('signal')] = -1
        
        # Enhanced signal smoothing to prevent whipsaws
        for i in range(3, len(df_predictions)):
            current_signal = df_predictions.iloc[i]['signal']
            recent_signals = df_predictions.iloc[i-3:i]['signal'].values
            
            # Reduce noise in signal generation
            if current_signal != 0:
                # Count recent opposite signals
                opposite_count = sum(recent_signals == -current_signal)
                if opposite_count >= 2:  # Too many recent opposite signals
                    current_prob = df_predictions.iloc[i]['direction_prob_up']
                    # Only keep very strong signals
                    if current_signal == 1 and current_prob < 0.65:
                        df_predictions.iloc[i, df_predictions.columns.get_loc('signal')] = 0
                    elif current_signal == -1 and current_prob > 0.35:
                        df_predictions.iloc[i, df_predictions.columns.get_loc('signal')] = 0
        
        # Add trend-following component for sustained periods
        window = min(20, len(df_predictions) // 4)
        for i in range(window, len(df_predictions)):
            recent_data = df_predictions.iloc[i-window:i]
            
            # If no signals recently, look for trend continuation
            if recent_data['signal'].abs().sum() < 3:  # Less than 3 signals in window
                current_row = df_predictions.iloc[i]
                
                # Check for strong trend
                if 'price_momentum' in current_row:
                    momentum = current_row['price_momentum']
                    prob_up = current_row['direction_prob_up']
                    
                    # Trend continuation signals
                    if momentum > 0.015 and prob_up > 0.52:  # Strong uptrend
                        df_predictions.iloc[i, df_predictions.columns.get_loc('signal')] = 1
                    elif momentum < -0.015 and prob_up < 0.48:  # Strong downtrend
                        df_predictions.iloc[i, df_predictions.columns.get_loc('signal')] = -1
        
        # Calculate and print statistics
        total_signals = (df_predictions['signal'] != 0).sum()
        long_signals = (df_predictions['signal'] == 1).sum()
        short_signals = (df_predictions['signal'] == -1).sum()
        signal_rate = (df_predictions['signal'] != 0).mean() * 100
        
        print(f"Generated adaptive signals: {long_signals} long, {short_signals} short")
        print(f"Total signal rate: {signal_rate:.1f}% of periods")
        
        # Period-by-period analysis
        n_periods = len(df_predictions)
        period_size = max(1, n_periods // 4)
        
        for period in range(4):
            start_idx = period * period_size
            end_idx = min((period + 1) * period_size, n_periods) if period < 3 else n_periods
            
            if start_idx < n_periods:
                period_data = df_predictions.iloc[start_idx:end_idx]
                period_signal_rate = (period_data['signal'] != 0).mean() * 100
                
                start_date = period_data.iloc[0]['date'] if 'date' in period_data.columns else f"Period {period+1}"
                print(f"Period {period+1} ({start_date}): {period_signal_rate:.1f}% signal rate")
        
        return df_predictions
    
    def backtest_active(self, df_signals):
        """More active backtesting with proper risk management"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        position_size = 0
        days_in_position = 0
        max_hold_days = 10  # Maximum days to hold a position
        
        results = []
        
        for i in range(len(df_signals)):
            current_row = df_signals.iloc[i]
            current_price = current_row['wti_price'] 
            current_date = current_row['date']
            signal = current_row['signal']
            
            if pd.isna(current_price):
                continue
            
            # Increment days in position
            if position != 0:
                days_in_position += 1
            
            # Handle existing position
            if position != 0:
                # Calculate current P&L
                if position == 1:  # Long position 
                    pnl_pct = (current_price - entry_price) / entry_price
                elif position == -1:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Check exit conditions (more active exit strategy)
                should_exit = (
                    pnl_pct <= -self.stop_loss or  # Stop loss
                    pnl_pct >= self.take_profit or  # Take profit  
                    signal == -position or  # Opposite signal
                    days_in_position >= max_hold_days or  # Maximum hold period
                    (signal == 0 and days_in_position >= 3)  # No signal after 3 days
                )
                
                if should_exit:
                    # Execute exit
                    capital_change = capital * position_size * pnl_pct
                    capital += capital_change
                    
                    # Record trade
                    if pnl_pct <= -self.stop_loss:
                        exit_reason = 'Stop Loss'
                    elif pnl_pct >= self.take_profit:
                        exit_reason = 'Take Profit'
                    elif signal == -position:
                        exit_reason = 'Opposite Signal'
                    elif days_in_position >= max_hold_days:
                        exit_reason = 'Max Hold Period'
                    else:
                        exit_reason = 'Signal Change'
                    
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'position': 'Long' if position == 1 else 'Short',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct * 100,
                        'capital_after': capital,
                        'position_size': position_size,
                        'days_held': days_in_position,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    position_size = 0
                    days_in_position = 0
            
            # Handle new position entry (more active)
            if signal != 0 and position == 0 and capital > 0:
                position = signal
                entry_price = current_price
                entry_date = current_date
                days_in_position = 0
                
                # More active position sizing
                base_size = self.max_position_size
                
                # Adjust for volatility but be less conservative
                volatility = current_row.get('price_volatility', 0.02)
                if pd.isna(volatility) or volatility == 0:
                    volatility = 0.02
                    
                # Less aggressive volatility adjustment
                vol_adjustment = min(1.0, 0.03 / max(volatility, 0.01))
                position_size = base_size * vol_adjustment
                position_size = min(position_size, 0.20)  # Cap at 20%
                position_size = max(position_size, 0.02)  # Minimum 2%
            
            # Record state
            results.append({
                'date': current_date,
                'price': current_price,
                'signal': signal,
                'position': position,
                'capital': capital,
                'position_size': position_size,
                'days_in_position': days_in_position,
                'direction_prob_up': current_row.get('direction_prob_up', 0.5),
                'predicted_return': current_row.get('predicted_return', 0),
                'actual_return': current_row.get('future_return', 0)
            })
        
        return pd.DataFrame(results)
    
    def analyze_performance(self, results_df):
        """Comprehensive performance analysis"""
        if len(self.trades) == 0:
            print("No trades were executed.")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        final_capital = results_df['capital'].iloc[-1]
        total_return = (final_capital / self.initial_capital - 1) * 100
        num_trades = len(trades_df)
        
        # Win/Loss analysis
        winning_trades = trades_df[trades_df['pnl_pct'] > 0]
        losing_trades = trades_df[trades_df['pnl_pct'] <= 0]
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        # Risk metrics
        capital_series = results_df['capital']
        returns = capital_series.pct_change().dropna()
        
        # Maximum drawdown calculation
        rolling_max = capital_series.expanding().max()
        drawdown = (rolling_max - capital_series) / rolling_max
        max_drawdown = drawdown.max() * 100
        
        # Sharpe ratio (annualized)
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Print results
        print("=== FIXED TRADING STRATEGY PERFORMANCE ===")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${final_capital:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Number of Trades: {num_trades}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            profit_factor = abs(winning_trades['pnl_pct'].sum() / losing_trades['pnl_pct'].sum())
            print(f"Profit Factor: {profit_factor:.2f}")
        
        # Exit reason analysis
        print("\n=== EXIT REASONS ===")
        exit_reasons = trades_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"{reason}: {count} trades ({count/num_trades*100:.1f}%)")
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': num_trades
        }
    
    def plot_results(self, results_df):
        """Plot comprehensive results"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price and signals
        axes[0].plot(results_df['date'], results_df['price'], label='WTI Price', linewidth=2, color='black')
        
        long_signals = results_df[results_df['signal'] == 1]
        short_signals = results_df[results_df['signal'] == -1]
        
        if len(long_signals) > 0:
            axes[0].scatter(long_signals['date'], long_signals['price'], 
                           color='green', marker='^', s=100, label='Long Signal', alpha=0.8)
        if len(short_signals) > 0:
            axes[0].scatter(short_signals['date'], short_signals['price'], 
                           color='red', marker='v', s=100, label='Short Signal', alpha=0.8)
        
        axes[0].set_title('WTI Oil Price and Trading Signals (Fixed Strategy)')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Portfolio value with drawdown
        axes[1].plot(results_df['date'], results_df['capital'], 
                    label='Portfolio Value', color='blue', linewidth=2)
        axes[1].axhline(y=self.initial_capital, color='gray', linestyle='--', 
                       label='Initial Capital', alpha=0.7)
        
        # Add drawdown visualization
        rolling_max = results_df['capital'].expanding().max()
        axes[1].fill_between(results_df['date'], results_df['capital'], rolling_max, 
                           where=(results_df['capital'] < rolling_max), 
                           alpha=0.3, color='red', label='Drawdown')
        
        axes[1].set_title('Portfolio Value Over Time')
        axes[1].set_ylabel('Capital ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Prediction confidence
        axes[2].plot(results_df['date'], results_df['direction_prob_up'], 
                    label='Direction Probability (Up)', color='purple', linewidth=1)
        axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[2].axhline(y=0.7, color='green', linestyle=':', alpha=0.7, label='Long Threshold')
        axes[2].axhline(y=0.3, color='red', linestyle=':', alpha=0.7, label='Short Threshold')
        
        axes[2].set_title('Model Confidence Over Time')
        axes[2].set_ylabel('Probability')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Run the more active model
print("=== RUNNING ACTIVE CLOUD TRADING MODEL ===")
model = FixedCloudTradingModel(
    initial_capital=10000,
    max_position_size=0.10,  # 10% position size (more active)
    stop_loss=0.03,  # 3% stop loss  
    take_profit=0.06  # 6% take profit
)

print("1. Loading data...")
df = model.load_and_prepare_data()

print("2. Creating features (no look-ahead)...")
df_full, df_with_price = model.create_features(df)

if len(df_with_price) >= 50:
    print("3. Running walk-forward analysis...")
    df_predictions = model.walk_forward_analysis(df_with_price)
    
    if df_predictions is not None:
        print("4. Generating adaptive signals...")
        df_signals = model.generate_adaptive_signals(df_predictions)
        
        print("5. Running active backtest...")
        results = model.backtest_active(df_signals)
        
        print("6. Analyzing performance...")
        metrics = model.analyze_performance(results)
        
        print("7. Creating plots...")
        model.plot_results(results)
        
        # Additional analysis for active strategy
        print("\n=== ADDITIONAL ACTIVE STRATEGY METRICS ===")
        avg_days_held = pd.DataFrame(model.trades)['days_held'].mean() if model.trades else 0
        print(f"Average days per trade: {avg_days_held:.1f}")
        
        signal_rate = (df_signals['signal'] != 0).mean() * 100
        print(f"Signal frequency: {signal_rate:.1f}% of trading days")
        
        if len(model.trades) > 0:
            trades_df = pd.DataFrame(model.trades)
            monthly_trades = len(trades_df) / (len(results) / 30) if len(results) > 30 else len(trades_df)
            print(f"Estimated monthly trade frequency: {monthly_trades:.1f} trades/month")
    else:
        print("Walk-forward analysis failed - insufficient data")
else:
    print("Not enough price data points for analysis.")