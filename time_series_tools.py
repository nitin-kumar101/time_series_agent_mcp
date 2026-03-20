"""
Time Series Analysis Tools
Comprehensive tools for trend analysis, seasonality detection, and forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, List, Any, Optional
from prophet import Prophet
import warnings
import os
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """Comprehensive time series analysis toolkit"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_trend(self, data: pd.Series, window: int = 30) -> Dict[str, Any]:
        """Analyze trend in time series data"""
        if len(data) < window:
            window = len(data) // 2
            
        # Moving average trend
        moving_avg = data.rolling(window=window).mean()
        
        # Linear trend
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        model = LinearRegression().fit(X, y)
        linear_trend = model.predict(X)
        
        # Trend direction
        trend_slope = model.coef_[0]
        if trend_slope > 0:
            trend_direction = "increasing"
        elif trend_slope < 0:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
            
        # Trend strength (R-squared)
        trend_strength = model.score(X, y)
        
        return {
            'trend_direction': trend_direction,
            'trend_slope': trend_slope,
            'trend_strength': trend_strength,
            'moving_average': moving_avg,
            'linear_trend': pd.Series(linear_trend, index=data.index),
            'window_size': window
        }
    
    def detect_seasonality(self, data: pd.Series, period: int = None) -> Dict[str, Any]:
        """Detect seasonality in time series data"""
        if len(data) < 24:  # Need at least 2 periods
            return {'seasonality_detected': False, 'reason': 'Insufficient data'}
            
        # Auto-detect period if not provided
        if period is None:
            period = self._auto_detect_period(data)
            
        if period is None:
            return {'seasonality_detected': False, 'reason': 'No clear period detected'}
            
        try:
            # Seasonal decomposition
            decomposition = seasonal_decompose(data, model='additive', period=period)
            
            # Calculate seasonality strength
            seasonal_strength = np.var(decomposition.seasonal) / np.var(data)
            
            # Check if seasonality is significant
            is_seasonal = seasonal_strength > 0.1  # Threshold for significance
            
            return {
                'seasonality_detected': is_seasonal,
                'period': period,
                'seasonal_strength': seasonal_strength,
                'decomposition': decomposition,
                'seasonal_component': decomposition.seasonal,
                'trend_component': decomposition.trend,
                'residual_component': decomposition.resid
            }
            
        except Exception as e:
            return {'seasonality_detected': False, 'reason': f'Decomposition failed: {e}'}
    
    def _auto_detect_period(self, data: pd.Series) -> Optional[int]:
        """Auto-detect the period of seasonality"""
        # Try common periods
        common_periods = [7, 12, 24, 30, 52, 365]  # daily, monthly, hourly, etc.
        
        best_period = None
        best_score = 0
        
        for period in common_periods:
            if len(data) >= period * 2:
                try:
                    decomposition = seasonal_decompose(data, model='additive', period=period)
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(data)
                    
                    if seasonal_strength > best_score:
                        best_score = seasonal_strength
                        best_period = period
                        
                except:
                    continue
                    
        return best_period if best_score > 0.1 else None
    
    def test_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """Test stationarity using ADF and KPSS tests"""
        # Remove NaN values
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            return {'error': 'Insufficient data for stationarity test'}
            
        # ADF Test
        adf_result = adfuller(clean_data)
        
        # KPSS Test
        kpss_result = kpss(clean_data, regression='c')
        
        # Interpretation
        adf_stationary = adf_result[1] < 0.05  # p-value < 0.05 means stationary
        kpss_stationary = kpss_result[1] > 0.05  # p-value > 0.05 means stationary
        
        # Overall stationarity
        is_stationary = adf_stationary and kpss_stationary
        
        return {
            'is_stationary': is_stationary,
            'adf_test': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'stationary': adf_stationary
            },
            'kpss_test': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'stationary': kpss_stationary
            }
        }
    
    def forecast_arima(self, data: pd.Series, periods: int = 30, order: tuple = None) -> Dict[str, Any]:
        """Forecast using ARIMA model"""
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            return {'error': 'Insufficient data for ARIMA forecasting'}
            
        try:
            # Auto-select order if not provided
            if order is None:
                order = self._auto_select_arima_order(clean_data)
                
            # Fit ARIMA model
            model = ARIMA(clean_data, order=order)
            fitted_model = model.fit()
            
            # Make forecast
            forecast = fitted_model.forecast(steps=periods)
            conf_int = fitted_model.get_forecast(steps=periods).conf_int()
            
            # Calculate forecast dates
            last_date = clean_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq='D')[1:]
            else:
                forecast_dates = range(len(clean_data), len(clean_data) + periods)
                
            return {
                'forecast': pd.Series(forecast.values, index=forecast_dates),
                'confidence_interval': conf_int,
                'model_summary': fitted_model.summary(),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'order': order,
                'fitted_values': fitted_model.fittedvalues
            }
            
        except Exception as e:
            return {'error': f'ARIMA forecasting failed: {e}'}
    
    def _auto_select_arima_order(self, data: pd.Series) -> tuple:
        """Auto-select ARIMA order using simple heuristics"""
        # Simple approach: try common orders
        orders = [(1, 1, 1), (2, 1, 2), (1, 0, 1), (0, 1, 1)]
        
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for order in orders:
            try:
                model = ARIMA(data, order=order)
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = order
            except:
                continue
                
        return best_order
    
    def forecast_exponential_smoothing(self, data: pd.Series, periods: int = 30) -> Dict[str, Any]:
        """Forecast using exponential smoothing"""
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            return {'error': 'Insufficient data for exponential smoothing'}
            
        try:
            # Fit exponential smoothing model
            model = ExponentialSmoothing(clean_data, seasonal=None)
            fitted_model = model.fit()
            
            # Make forecast
            forecast = fitted_model.forecast(steps=periods)
            
            # Calculate forecast dates
            last_date = clean_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq='D')[1:]
            else:
                forecast_dates = range(len(clean_data), len(clean_data) + periods)
                
            return {
                'forecast': pd.Series(forecast.values, index=forecast_dates),
                'fitted_values': fitted_model.fittedvalues,
                'model_params': fitted_model.params,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
        except Exception as e:
            return {'error': f'Exponential smoothing failed: {e}'}
    
    def calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        # Align series
        common_index = actual.index.intersection(predicted.index)
        if len(common_index) == 0:
            return {'error': 'No common index between actual and predicted'}
            
        actual_aligned = actual.loc[common_index]
        predicted_aligned = predicted.loc[common_index]
        
        # Calculate metrics
        mse = mean_squared_error(actual_aligned, predicted_aligned)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_aligned, predicted_aligned)
        mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def comprehensive_analysis(self, data: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive time series analysis"""
        results = {}
        
        # Basic statistics
        results['basic_stats'] = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'count': data.count(),
            'missing': data.isnull().sum()
        }
        
        # Trend analysis
        results['trend'] = self.analyze_trend(data)
        
        # Seasonality detection
        results['seasonality'] = self.detect_seasonality(data)
        
        # Stationarity test
        results['stationarity'] = self.test_stationarity(data)
        
        # Forecasting
        results['arima_forecast'] = self.forecast_arima(data)
        results['exp_smoothing_forecast'] = self.forecast_exponential_smoothing(data)
        
        # Anomaly detection via Prophet
        results['prophet_anomalies'] = self.detect_anomalies_prophet(data)
        
        return results

    def detect_anomalies_prophet(self, data: pd.Series, interval_width: float = 0.99) -> Dict[str, Any]:
        """Detect anomalies by flagging points outside Prophet prediction intervals."""
        series = data.dropna()
        if len(series) < 10:
            return {'error': 'Insufficient data for Prophet anomaly detection'}

        # Prophet requires columns ds (datetime) and y (values); drop tz for compatibility
        ds_index = series.index
        if hasattr(ds_index, 'tz') and ds_index.tz is not None:
            ds_index = ds_index.tz_convert(None)

        df_prophet = pd.DataFrame({'ds': ds_index, 'y': series.values})

        model = Prophet(interval_width=interval_width)
        model.fit(df_prophet)

        forecast = model.predict(df_prophet)

        merged = pd.DataFrame({
            'y': df_prophet['y'].values,
            'yhat': forecast['yhat'].values,
            'yhat_lower': forecast['yhat_lower'].values,
            'yhat_upper': forecast['yhat_upper'].values
        }, index=pd.DatetimeIndex(df_prophet['ds']))

        merged['is_anomaly'] = (merged['y'] < merged['yhat_lower']) | (merged['y'] > merged['yhat_upper'])
        anomalies = merged[merged['is_anomaly']]

        return {
            'anomalies': anomalies,
            'forecast': forecast,
            'model': model
        }


class TimeSeriesVisualizer:
    """Visualization tools for time series analysis"""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_time_series(self, data: pd.Series, title: str = "Time Series", 
                        figsize: tuple = (12, 6)) -> go.Figure:
        """Create interactive time series plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='Data',
            line=dict(color=self.colors[0], width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_trend_analysis(self, data: pd.Series, trend_results: Dict[str, Any]) -> go.Figure:
        """Plot trend analysis results"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Original Data with Trend', 'Trend Components'],
            vertical_spacing=0.1
        )
        
        # Original data
        fig.add_trace(
            go.Scatter(x=data.index, y=data.values, mode='lines', name='Original Data'),
            row=1, col=1
        )
        
        # Moving average
        if 'moving_average' in trend_results:
            fig.add_trace(
                go.Scatter(x=trend_results['moving_average'].index, 
                          y=trend_results['moving_average'].values, 
                          mode='lines', name='Moving Average'),
                row=1, col=1
            )
        
        # Linear trend
        if 'linear_trend' in trend_results:
            fig.add_trace(
                go.Scatter(x=trend_results['linear_trend'].index, 
                          y=trend_results['linear_trend'].values, 
                          mode='lines', name='Linear Trend'),
                row=1, col=1
            )
        
        # Trend components
        fig.add_trace(
            go.Scatter(x=data.index, y=data.values, mode='lines', name='Original'),
            row=2, col=1
        )
        
        if 'moving_average' in trend_results:
            fig.add_trace(
                go.Scatter(x=trend_results['moving_average'].index, 
                          y=trend_results['moving_average'].values, 
                          mode='lines', name='Trend'),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="Trend Analysis")
        return fig
    
    def plot_seasonal_decomposition(self, decomposition) -> go.Figure:
        """Plot seasonal decomposition"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.05
        )
        
        # Original
        fig.add_trace(
            go.Scatter(x=decomposition.observed.index, 
                      y=decomposition.observed.values, 
                      mode='lines', name='Original'),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=decomposition.trend.index, 
                      y=decomposition.trend.values, 
                      mode='lines', name='Trend'),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=decomposition.seasonal.index, 
                      y=decomposition.seasonal.values, 
                      mode='lines', name='Seasonal'),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(x=decomposition.resid.index, 
                      y=decomposition.resid.values, 
                      mode='lines', name='Residual'),
            row=4, col=1
        )
        
        fig.update_layout(height=1000, title_text="Seasonal Decomposition")
        return fig
    
    def plot_forecast(self, data: pd.Series, forecast_results: Dict[str, Any], 
                     forecast_type: str = "ARIMA") -> go.Figure:
        """Plot forecast results"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='Historical Data',
            line=dict(color=self.colors[0])
        ))
        
        # Forecast
        if 'forecast' in forecast_results:
            forecast = forecast_results['forecast']
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines',
                name=f'{forecast_type} Forecast',
                line=dict(color=self.colors[1], dash='dash')
            ))
            
            # Confidence interval
            if 'confidence_interval' in forecast_results:
                ci = forecast_results['confidence_interval']
                fig.add_trace(go.Scatter(
                    x=ci.index,
                    y=ci.iloc[:, 1],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=ci.index,
                    y=ci.iloc[:, 0],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    name='Confidence Interval'
                ))
        
        fig.update_layout(
            title=f'{forecast_type} Forecast',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig

    def plot_prophet_anomalies(self, data: pd.Series, anomalies_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
        """Plot actuals, Prophet forecast with confidence interval, and anomaly markers."""
        fig = go.Figure()

        # Actual data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='Actual',
            line=dict(color=self.colors[0], width=2)
        ))

        # Forecast mean
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='Prophet yhat',
            line=dict(color=self.colors[1], dash='dash')
        ))

        # Confidence interval band
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)',
            name='Confidence Interval'
        ))

        # Anomaly markers
        if anomalies_df is not None and len(anomalies_df) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies_df.index,
                y=anomalies_df['y'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#d62728', size=8, symbol='x')
            ))

        fig.update_layout(
            title='Prophet Anomaly Detection',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig


if __name__ == "__main__":
    # Run analysis on commit_history.csv by building a numeric time series
    analyzer = TimeSeriesAnalyzer()

    # Load commit history CSV
    csv_path = "commit_history.csv"
    df = pd.read_csv(csv_path)

    # Parse date column (timezone-aware safe), drop invalid
    if 'date' not in df.columns:
        raise ValueError("Expected a 'date' column in commit_history.csv")

    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df = df.dropna(subset=['date']).sort_values('date').set_index('date')

    # Build a numeric time series: commits per day
    commits_per_day = df.resample('D').size().astype(float)

    # Ensure there is sufficient data
    if commits_per_day.empty or commits_per_day.count() < 10:
        raise ValueError("Not enough data after resampling to perform analysis.")

    # Perform analysis
    results = analyzer.comprehensive_analysis(commits_per_day)

    print("Time Series Analysis Results (Commits per Day):")
    print(f"Total days: {len(commits_per_day)}")
    print(f"Trend Direction: {results['trend']['trend_direction']}")
    print(f"Seasonality Detected: {results['seasonality']['seasonality_detected']}")
    print(f"Stationary: {results['stationarity']['is_stationary']}")
    if isinstance(results.get('prophet_anomalies'), dict) and 'anomalies' in results['prophet_anomalies']:
        print(f"Prophet anomalies detected: {len(results['prophet_anomalies']['anomalies'])}")
        # Save interactive anomaly visualization
        out_dir = 'output'
        os.makedirs(out_dir, exist_ok=True)
        viz = TimeSeriesVisualizer()
        fig = viz.plot_prophet_anomalies(
            commits_per_day,
            results['prophet_anomalies']['anomalies'],
            results['prophet_anomalies']['forecast']
        )
        fig.write_html(os.path.join(out_dir, 'prophet_anomalies.html'))
        print("Saved: output/prophet_anomalies.html")