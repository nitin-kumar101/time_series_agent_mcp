"""
Visualization and Reporting Components
Generate comprehensive reports and visualizations for time series analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
from time_series_tools import TimeSeriesAnalyzer, TimeSeriesVisualizer
import os
import json
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ReportGenerator:
    """Generate comprehensive reports for time series analysis"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_summary_report(self, analysis_results: Dict[str, Any], 
                              data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary report of the analysis"""
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_source': 'CSV Analysis',
                'total_records': data_info.get('total_records', 0)
            },
            'data_overview': {
                'time_range': data_info.get('time_range', {}),
                'time_span_days': data_info.get('time_span_days', 0),
                'numeric_columns': data_info.get('numeric_columns', []),
                'categorical_columns': data_info.get('categorical_columns', []),
                'missing_values': data_info.get('missing_values', {})
            },
            'analysis_summary': {},
            'recommendations': []
        }
        
        # Extract key findings
        if 'trend' in analysis_results:
            trend = analysis_results['trend']
            report['analysis_summary']['trend'] = {
                'direction': trend.get('trend_direction', 'unknown'),
                'strength': round(trend.get('trend_strength', 0), 3),
                'slope': round(trend.get('trend_slope', 0), 6)
            }
            
        if 'seasonality' in analysis_results:
            seasonality = analysis_results['seasonality']
            report['analysis_summary']['seasonality'] = {
                'detected': seasonality.get('seasonality_detected', False),
                'period': seasonality.get('period', None),
                'strength': round(seasonality.get('seasonal_strength', 0), 3)
            }
            
        if 'stationarity' in analysis_results:
            stationarity = analysis_results['stationarity']
            report['analysis_summary']['stationarity'] = {
                'is_stationary': stationarity.get('is_stationary', False),
                'adf_p_value': round(stationarity.get('adf_test', {}).get('p_value', 1), 4),
                'kpss_p_value': round(stationarity.get('kpss_test', {}).get('p_value', 1), 4)
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(analysis_results)
        
        return report
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Trend recommendations
        if 'trend' in analysis_results:
            trend = analysis_results['trend']
            if trend.get('trend_strength', 0) > 0.7:
                recommendations.append("Strong trend detected. Consider detrending for better forecasting.")
            elif trend.get('trend_direction') == 'increasing':
                recommendations.append("Upward trend detected. Monitor for continued growth.")
            elif trend.get('trend_direction') == 'decreasing':
                recommendations.append("Downward trend detected. Investigate potential causes.")
        
        # Seasonality recommendations
        if 'seasonality' in analysis_results:
            seasonality = analysis_results['seasonality']
            if seasonality.get('seasonality_detected', False):
                period = seasonality.get('period', 'unknown')
                recommendations.append(f"Seasonal pattern detected with period {period}. Use seasonal models for forecasting.")
        
        # Stationarity recommendations
        if 'stationarity' in analysis_results:
            stationarity = analysis_results['stationarity']
            if not stationarity.get('is_stationary', True):
                recommendations.append("Data is non-stationary. Consider differencing or transformation.")
        
        # Forecasting recommendations
        if 'arima_forecast' in analysis_results and 'error' not in analysis_results['arima_forecast']:
            recommendations.append("ARIMA model successfully fitted. Use for short-term forecasting.")
        
        if 'exp_smoothing_forecast' in analysis_results and 'error' not in analysis_results['exp_smoothing_forecast']:
            recommendations.append("Exponential smoothing model available. Good for trend-based forecasting.")
        
        if not recommendations:
            recommendations.append("Consider collecting more data for better analysis.")
            
        return recommendations
    
    def export_report_json(self, report: Dict[str, Any], filename: str = "ts_analysis_report.json"):
        """Export report to JSON file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return filename
    
    def export_report_html(self, report: Dict[str, Any], plots: List[go.Figure], 
                          filename: str = "ts_analysis_report.html") -> str:
        """Export report to HTML file with embedded plots"""
        html_content = self._generate_html_report(report, plots)
        
        with open(filename, 'w') as f:
            f.write(html_content)
        return filename
    
    def _generate_html_report(self, report: Dict[str, Any], plots: List[go.Figure]) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Time Series Analysis Report</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4fd; border-radius: 5px; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
                .plot-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Time Series Analysis Report</h1>
                <p>Generated on: {report['metadata']['generated_at']}</p>
                <p>Total Records: {report['metadata']['total_records']}</p>
            </div>
            
            <div class="section">
                <h2>Data Overview</h2>
                <div class="metric">Time Span: {report['data_overview']['time_span_days']} days</div>
                <div class="metric">Numeric Columns: {len(report['data_overview']['numeric_columns'])}</div>
                <div class="metric">Categorical Columns: {len(report['data_overview']['categorical_columns'])}</div>
            </div>
            
            <div class="section">
                <h2>Analysis Summary</h2>
        """
        
        # Add analysis summary
        if 'trend' in report['analysis_summary']:
            trend = report['analysis_summary']['trend']
            html += f"""
                <div class="metric">
                    <strong>Trend:</strong> {trend['direction']} (strength: {trend['strength']})
                </div>
            """
        
        if 'seasonality' in report['analysis_summary']:
            seasonality = report['analysis_summary']['seasonality']
            html += f"""
                <div class="metric">
                    <strong>Seasonality:</strong> {'Detected' if seasonality['detected'] else 'Not detected'}
                    {f"(period: {seasonality['period']})" if seasonality['period'] else ''}
                </div>
            """
        
        if 'stationarity' in report['analysis_summary']:
            stationarity = report['analysis_summary']['stationarity']
            html += f"""
                <div class="metric">
                    <strong>Stationarity:</strong> {'Stationary' if stationarity['is_stationary'] else 'Non-stationary'}
                </div>
            """
        
        # Add recommendations
        html += """
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
        """
        
        for rec in report['recommendations']:
            html += f'<div class="recommendation">{rec}</div>'
        
        # Add plots
        html += """
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
        """
        
        for i, plot in enumerate(plots):
            plot_html = plot.to_html(include_plotlyjs=False, div_id=f"plot_{i}")
            plot_div = plot_html.split('<body>')[1].split('</body>')[0]
            html += f'<div class="plot-container">{plot_div}</div>'
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html


class DashboardGenerator:
    """Generate interactive dashboards for time series analysis"""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def create_analysis_dashboard(self, data: pd.Series, analysis_results: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive analysis dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Time Series Data',
                'Trend Analysis',
                'Seasonal Decomposition',
                'Stationarity Test Results',
                'Forecast Comparison',
                'Analysis Summary'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Original time series
        fig.add_trace(
            go.Scatter(x=data.index, y=data.values, mode='lines', name='Data'),
            row=1, col=1
        )
        
        # Plot 2: Trend analysis
        if 'trend' in analysis_results:
            trend = analysis_results['trend']
            fig.add_trace(
                go.Scatter(x=data.index, y=data.values, mode='lines', name='Original'),
                row=1, col=2
            )
            
            if 'moving_average' in trend:
                fig.add_trace(
                    go.Scatter(x=trend['moving_average'].index, 
                              y=trend['moving_average'].values, 
                              mode='lines', name='Moving Avg'),
                    row=1, col=2
                )
            
            if 'linear_trend' in trend:
                fig.add_trace(
                    go.Scatter(x=trend['linear_trend'].index, 
                              y=trend['linear_trend'].values, 
                              mode='lines', name='Linear Trend'),
                    row=1, col=2
                )
        
        # Plot 3: Seasonal decomposition
        if 'seasonality' in analysis_results and 'decomposition' in analysis_results['seasonality']:
            decomp = analysis_results['seasonality']['decomposition']
            fig.add_trace(
                go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values, mode='lines', name='Seasonal'),
                row=2, col=1
            )
        
        # Plot 4: Stationarity test results
        if 'stationarity' in analysis_results:
            stationarity = analysis_results['stationarity']
            test_names = ['ADF Test', 'KPSS Test']
            p_values = [
                stationarity.get('adf_test', {}).get('p_value', 1),
                stationarity.get('kpss_test', {}).get('p_value', 1)
            ]
            
            fig.add_trace(
                go.Bar(x=test_names, y=p_values, name='P-values'),
                row=2, col=2
            )
        
        # Plot 5: Forecast comparison
        if 'arima_forecast' in analysis_results and 'forecast' in analysis_results['arima_forecast']:
            arima_forecast = analysis_results['arima_forecast']['forecast']
            fig.add_trace(
                go.Scatter(x=data.index, y=data.values, mode='lines', name='Historical'),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=arima_forecast.index, y=arima_forecast.values, mode='lines', name='ARIMA Forecast'),
                row=3, col=1
            )
        
        # Plot 6: Analysis summary metrics
        metrics = []
        values = []
        
        if 'trend' in analysis_results:
            metrics.append('Trend Strength')
            values.append(analysis_results['trend'].get('trend_strength', 0))
        
        if 'seasonality' in analysis_results:
            metrics.append('Seasonal Strength')
            values.append(analysis_results['seasonality'].get('seasonal_strength', 0))
        
        if metrics:
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Metrics'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Time Series Analysis Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_forecast_dashboard(self, data: pd.Series, forecast_results: Dict[str, Any]) -> go.Figure:
        """Create a forecast-focused dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Historical Data & Forecasts',
                'Forecast Accuracy Metrics',
                'Residual Analysis',
                'Model Comparison'
            ],
            vertical_spacing=0.1
        )
        
        # Plot 1: Historical data and forecasts
        fig.add_trace(
            go.Scatter(x=data.index, y=data.values, mode='lines', name='Historical Data'),
            row=1, col=1
        )
        
        # Add forecasts
        forecast_count = 0
        for model_name, results in forecast_results.items():
            if 'forecast' in results and 'error' not in results:
                forecast = results['forecast']
                fig.add_trace(
                    go.Scatter(x=forecast.index, y=forecast.values, 
                              mode='lines', name=f'{model_name} Forecast'),
                    row=1, col=1
                )
                forecast_count += 1
        
        # Plot 2: Accuracy metrics
        if forecast_count > 0:
            model_names = []
            rmse_values = []
            
            for model_name, results in forecast_results.items():
                if 'forecast' in results and 'error' not in results:
                    # Calculate metrics (simplified)
                    model_names.append(model_name)
                    rmse_values.append(np.random.uniform(0.1, 2.0))  # Placeholder
            
            fig.add_trace(
                go.Bar(x=model_names, y=rmse_values, name='RMSE'),
                row=1, col=2
            )
        
        # Plot 3: Residual analysis (placeholder)
        residuals = np.random.normal(0, 1, len(data))
        fig.add_trace(
            go.Scatter(x=data.index, y=residuals, mode='markers', name='Residuals'),
            row=2, col=1
        )
        
        # Plot 4: Model comparison
        models = ['ARIMA', 'Exponential Smoothing']
        aic_values = [np.random.uniform(100, 200), np.random.uniform(120, 220)]
        
        fig.add_trace(
            go.Bar(x=models, y=aic_values, name='AIC'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Forecast Analysis Dashboard",
            showlegend=True
        )
        
        return fig


class ExportManager:
    """Manage export of analysis results in various formats"""
    
    def __init__(self):
        self.report_generator = ReportGenerator()
        self.dashboard_generator = DashboardGenerator()
    
    def export_complete_analysis(self, data: pd.Series, analysis_results: Dict[str, Any], 
                                data_info: Dict[str, Any], output_dir: str = "output") -> Dict[str, str]:
        """Export complete analysis in multiple formats"""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # Generate report
        report = self.report_generator.generate_summary_report(analysis_results, data_info)
        
        # Export JSON report
        json_file = os.path.join(output_dir, "analysis_report.json")
        self.report_generator.export_report_json(report, json_file)
        exported_files['json_report'] = json_file
        
        # Generate visualizations
        plots = []
        
        # Main dashboard
        dashboard = self.dashboard_generator.create_analysis_dashboard(data, analysis_results)
        plots.append(dashboard)
        
        # Export HTML report with plots
        html_file = os.path.join(output_dir, "analysis_report.html")
        self.report_generator.export_report_html(report, plots, html_file)
        exported_files['html_report'] = html_file
        
        # Export individual plots as HTML
        for i, plot in enumerate(plots):
            plot_file = os.path.join(output_dir, f"plot_{i}.html")
            plot.write_html(plot_file)
            exported_files[f'plot_{i}'] = plot_file
        
        # Export data summary
        summary_file = os.path.join(output_dir, "data_summary.csv")
        data.describe().to_csv(summary_file)
        exported_files['data_summary'] = summary_file
        
        return exported_files


if __name__ == "__main__":
    # Generate reports for commit_history.csv using real analysis
    analyzer = TimeSeriesAnalyzer()
    exp = ExportManager()

    csv_path = "commit_history.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load and prepare time index
    df = pd.read_csv(csv_path)
    if 'date' not in df.columns:
        raise ValueError("Expected a 'date' column in commit_history.csv")

    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df = df.dropna(subset=['date']).sort_values('date').set_index('date')

    # Aggregate to numeric series: commits per day
    commits_per_day = df.resample('D').size().astype(float)
    if commits_per_day.count() < 10:
        raise ValueError("Not enough data after resampling to perform analysis.")

    # Run comprehensive analysis (includes Prophet anomaly detection if available)
    analysis_results = analyzer.comprehensive_analysis(commits_per_day)

    # Build data_info for reporting
    data_info = {
        'total_records': int(len(df)),
        'time_range': {
            'start': df.index.min(),
            'end': df.index.max()
        },
        'time_span_days': int((df.index.max() - df.index.min()).days),
        'numeric_columns': ['commits_per_day'],
        'categorical_columns': [],
        'missing_values': {'commits_per_day': int(commits_per_day.isna().sum())}
    }

    # Export full report and dashboard
    outputs = exp.export_complete_analysis(commits_per_day, analysis_results, data_info, output_dir="output")
    print("Report generated successfully!")
    for k, v in outputs.items():
        print(f"{k}: {v}")