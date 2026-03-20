"""
Data Type Detection Module for Time Series Analysis
Automatically detects data types and prepares data for time series analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class DataTypeDetector:
    """Automatically detects data types and prepares data for time series analysis"""
    
    def __init__(self):
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\w{3}\s+\d{1,2},?\s+\d{4}',  # Mon DD, YYYY
            r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}',  # Mon DD HH:MM:SS YYYY
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        ]
        
    def detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the most likely time/date column"""
        time_columns = []
        
        for col in df.columns:
            if self._is_time_column(df[col]):
                time_columns.append(col)
        
        if not time_columns:
            return None
            
        # Return the column with the highest time-like score
        scores = {col: self._calculate_time_score(df[col]) for col in time_columns}
        return max(scores, key=scores.get)
    
    def _is_time_column(self, series: pd.Series) -> bool:
        """Check if a column contains time-like data"""
        # Check for common time-related column names
        time_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'modified']
        col_name = series.name.lower()
        
        if any(keyword in col_name for keyword in time_keywords):
            return True
            
        # Check if values look like dates/times
        sample_values = series.dropna().head(10)
        if len(sample_values) == 0:
            return False
            
        time_like_count = 0
        for value in sample_values:
            if self._looks_like_time(str(value)):
                time_like_count += 1
                
        return time_like_count / len(sample_values) > 0.5
    
    def _looks_like_time(self, value: str) -> bool:
        """Check if a string value looks like a time/date"""
        for pattern in self.date_patterns:
            if re.search(pattern, value):
                return True
        return False
    
    def _calculate_time_score(self, series: pd.Series) -> float:
        """Calculate a score indicating how time-like a column is"""
        score = 0.0
        sample_values = series.dropna().head(100)
        
        if len(sample_values) == 0:
            return 0.0
            
        # Check column name
        col_name = series.name.lower()
        time_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'modified']
        if any(keyword in col_name for keyword in time_keywords):
            score += 0.3
            
        # Check if values are parseable as dates
        parseable_count = 0
        for value in sample_values:
            pd.to_datetime(str(value), errors='raise', utc=True)
            parseable_count += 1
                
        score += (parseable_count / len(sample_values)) * 0.7
        
        return score
    
    def detect_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that contain numeric data"""
        numeric_columns = []
        
        for col in df.columns:
            if self._is_numeric_column(df[col]):
                numeric_columns.append(col)
                
        return numeric_columns
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column contains numeric data"""
        # Try to convert to numeric
        pd.to_numeric(series, errors='coerce')
        # Check if at least 70% of values are numeric
        numeric_count = pd.to_numeric(series, errors='coerce').notna().sum()
        return numeric_count / len(series) > 0.7
    
    def detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that contain categorical data"""
        categorical_columns = []
        
        for col in df.columns:
            if self._is_categorical_column(df[col]):
                categorical_columns.append(col)
                
        return categorical_columns
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Check if a column contains categorical data"""
        # Check if it's not numeric and not time
        if self._is_numeric_column(series) or self._is_time_column(series):
            return False
            
        # Check if it has a reasonable number of unique values
        unique_ratio = series.nunique() / len(series)
        return unique_ratio < 0.5 and series.nunique() > 1
    
    def prepare_time_series_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for time series analysis with robust date/time column checking"""

        result = {
            'original_data': df.copy(),
            'time_column': None,
            'numeric_columns': [],
            'categorical_columns': [],
            'prepared_data': None,
            'data_info': {}
        }

        # Detect column types
        result['time_column'] = self.detect_time_column(df)
        result['numeric_columns'] = self.detect_numeric_columns(df)
        result['categorical_columns'] = self.detect_categorical_columns(df)

        if result['time_column']:
            # Check and standardize time column types
            df_prepared = df.copy()
            time_col = result['time_column']

            # First, try converting the entire column to datetime, with errors='coerce' and utc=True for timezone-aware strings
            df_prepared[time_col] = pd.to_datetime(df_prepared[time_col], errors='coerce', infer_datetime_format=True, utc=True)

            # Check if all values are successfully converted to timestamps (i.e., not null)
            is_not_time = df_prepared[time_col].isna()
            if is_not_time.any():
                # There are rows that could not be converted to time
                # Option 1: Drop such rows
                df_prepared = df_prepared.loc[~is_not_time]
                reason = f"Dropped {is_not_time.sum()} rows due to invalid time formats in column '{time_col}'."
            else:
                reason = None

            # After handling, check if column is still dtype datetime64[ns] (robustness)
            if not pd.api.types.is_datetime64_any_dtype(df_prepared[time_col]):
                # Force conversion again, errors='raise' to trigger if any problematic cell remains
                df_prepared[time_col] = pd.to_datetime(df_prepared[time_col], errors='raise', infer_datetime_format=True, utc=True)

            # Sort and set index
            df_prepared = df_prepared.sort_values(time_col)
            df_prepared = df_prepared.set_index(time_col)

            result['prepared_data'] = df_prepared

            # Generate data info
            result['data_info'] = {
                'total_records': len(df_prepared),
                'time_range': {
                    'start': df_prepared.index.min(),
                    'end': df_prepared.index.max()
                },
                'time_span_days': (df_prepared.index.max() - df_prepared.index.min()).days
                    if len(df_prepared) > 0 else 0,
                'numeric_columns': result['numeric_columns'],
                'categorical_columns': result['categorical_columns'],
                'missing_values': df_prepared.isnull().sum().to_dict()
            }
            if reason:
                result['data_info']['time_column_note'] = reason
        else:
            result['prepared_data'] = df
            result['data_info'] = {
                'total_records': len(df),
                'numeric_columns': result['numeric_columns'],
                'categorical_columns': result['categorical_columns'],
                'missing_values': df.isnull().sum().to_dict()
            }
        
        return result


class DataAnalyzer:
    """Analyze data characteristics for time series analysis"""
    
    def __init__(self):
        self.detector = DataTypeDetector()
    
    def analyze_csv(self, file_path: str) -> Dict[str, Any]:
        """Analyze a CSV file and return comprehensive data information"""
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Detect data types
        data_info = self.detector.prepare_time_series_data(df)
        print("data_info", data_info['prepared_data'].index[:5])  # Show first 5 time index values
        # Add additional analysis
        data_info['analysis'] = self._perform_basic_analysis(data_info['prepared_data'])
        
        return data_info
    
    def _perform_basic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic statistical analysis"""
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'summary_stats': {}
        }
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['summary_stats'] = df[numeric_cols].describe().to_dict()
        
        # Check for time series characteristics
        if isinstance(df.index, pd.DatetimeIndex):
            analysis['time_series_info'] = {
                'frequency': self._detect_frequency(df.index),
                'is_regular': self._is_regular_frequency(df.index),
                'gaps': self._detect_gaps(df.index)
            }
        
        return analysis
    
    def _detect_frequency(self, index: pd.DatetimeIndex) -> str:
        """Detect the frequency of the time series"""
        if len(index) < 2:
            return 'unknown'
            
        # Calculate time differences
        diffs = index.to_series().diff().dropna()
        
        # Find the most common difference
        mode_diff = diffs.mode()
        if len(mode_diff) > 0:
            freq = pd.infer_freq(index)
            return freq if freq else 'irregular'
        
        return 'irregular'
    
    def _is_regular_frequency(self, index: pd.DatetimeIndex) -> bool:
        """Check if the time series has regular frequency"""
        if len(index) < 3:
            return False
            
        diffs = index.to_series().diff().dropna()
        return diffs.std() < diffs.mean() * 0.1  # Less than 10% variation
    
    def _detect_gaps(self, index: pd.DatetimeIndex) -> List[Dict]:
        """Detect gaps in the time series"""
        gaps = []
        if len(index) < 2:
            return gaps
            
        diffs = index.to_series().diff().dropna()
        expected_diff = diffs.median()
        
        # Find gaps larger than expected
        large_gaps = diffs[diffs > expected_diff * 2]
        
        for gap_start, gap_size in large_gaps.items():
            gaps.append({
                'start': gap_start,
                'size': gap_size,
                'days': gap_size.days
            })
        
        return gaps


if __name__ == "__main__":
    # Test the data type detector
    analyzer = DataAnalyzer()
    result = analyzer.analyze_csv("commit_history.csv")
    # print("result", result['prepared_data']['date'])
    # print("Data Analysis Results:", result['prepared_data']['date'])
    print(f"Time Column: {result.get('time_column', 'Not detected')}")
    print(f"Numeric Columns: {result.get('numeric_columns', [])}")
    print(f"Categorical Columns: {result.get('categorical_columns', [])}")
    print(f"Data Info: {result.get('data_info', {})}")
