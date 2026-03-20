# Time Series Analysis Agent

A comprehensive Python application that uses LangGraph and AI agents to automatically analyze CSV data for time series patterns. The agent intelligently detects data types, performs trend analysis, seasonality detection, and forecasting.

## Features

- **🤖 AI-Powered Agent**: Built with LangGraph for intelligent workflow orchestration
- **🔍 Automatic Data Detection**: Automatically identifies time columns, numeric data, and categorical variables
- **📈 Comprehensive Analysis**: 
  - Trend analysis with moving averages and linear regression
  - Seasonality detection with seasonal decomposition
  - Stationarity testing (ADF and KPSS tests)
  - Forecasting with ARIMA and Exponential Smoothing models
- **📊 Interactive Visualizations**: Plotly-based charts and dashboards
- **📋 Detailed Reports**: HTML and JSON reports with analysis summaries
- **🎯 Multiple Interfaces**: Command-line and interactive modes

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Azure OpenAI credentials (required for AI document answers):
```bash
export AZURE_OPENAI_API_KEY='your-api-key-here'
export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'
export AZURE_OPENAI_DEPLOYMENT='your-deployment-name'
```

**Note**: Azure OpenAI config is required for AI-powered document Q&A. Without it, you can still search documents but won't get generated answers.

## Usage

### Interactive Mode
```bash
python main.py
```

### Streamlit Web Interface
```bash
# Run the interactive web chatbot interface
streamlit run streamlit_app.py

# Or use the batch file
run_ui.bat
```

The web interface provides:
- **Chat interface** for natural language interaction with analysis tools
- **Dual file upload** for both CSV (time series) and PDF (documents) files
- **Real-time analysis** results and visualizations
- **Document Q&A** using RAG (Retrieval-Augmented Generation) with AI-powered answers
- **Report generation** with downloadable HTML/JSON reports
- **Interactive plots** using Plotly
- **Document management** with upload, search, and statistics

### Chat Commands Examples

The chatbot understands natural language commands:

**Time Series Analysis:**
- *"analyze my data"* → Performs comprehensive analysis
- *"forecast next 6 months"* → Generates 6-period forecast
- *"detect anomalies"* → Finds outliers in the data
- *"generate report"* → Creates HTML/JSON analysis reports

**Document Q&A (RAG):**
- *"search for machine learning"* → Searches uploaded documents
- *"what is machine learning"* → AI generates comprehensive answers
- *"explain this concept"* → Contextual explanations from documents
- *"list my documents"* → Shows all uploaded PDFs
- *"show me the stats"* → Displays RAG system statistics

**RAG Generation Features:**
- **AI-Powered Answers**: Uses Azure OpenAI to generate comprehensive responses
- **Source Citations**: Answers include references to source documents
- **Contextual Understanding**: Synthesizes information from multiple document chunks
- **Graceful Degradation**: Falls back to raw search results when Azure OpenAI config unavailable

### Command Line Mode
```bash
# Analyze a specific CSV file
python main.py data.csv

# Specify output directory
python main.py data.csv --output results

# Force interactive mode
python main.py --interactive
```

### Programmatic Usage
```python
from ts_agent import TimeSeriesAnalysisAgent

# Create agent
agent = TimeSeriesAnalysisAgent()

# Analyze CSV
results = agent.analyze_csv("your_data.csv")

if results["success"]:
    print("Analysis completed!")
    print(agent.get_analysis_summary(results["analysis_results"]))
```

## Project Structure

```
time-series-tool/
├── main.py                 # Main application entry point
├── ts_agent.py            # LangGraph agent implementation
├── data_detector.py       # Automatic data type detection
├── time_series_tools.py  # Time series analysis algorithms
├── reporting.py           # Report generation and visualization
├── requirements.txt       # Python dependencies
└── commit_history.csv     # Sample data file
```

## How It Works

1. **Data Loading**: Automatically loads and validates CSV files
2. **Data Detection**: Intelligently identifies:
   - Time/date columns using pattern matching
   - Numeric columns for analysis
   - Categorical variables
3. **Data Preparation**: Converts time columns to datetime and sets as index
4. **Analysis Pipeline**:
   - Trend analysis with multiple methods
   - Seasonality detection and decomposition
   - Stationarity testing
   - Forecasting with multiple models
5. **Visualization**: Generates interactive plots and dashboards
6. **Reporting**: Creates comprehensive HTML and JSON reports

## Supported Data Formats

The agent can handle various CSV formats with:
- Different date/time formats (ISO, US, European, etc.)
- Mixed data types
- Missing values
- Irregular time intervals

## Output Files

For each analyzed column, the agent generates:
- `analysis_report.html` - Interactive HTML report
- `analysis_report.json` - Detailed JSON results
- `plot_*.html` - Individual visualization files
- `data_summary.csv` - Statistical summary
- `analysis_summary.txt` - Human-readable summary

## Example Analysis

The included `commit_history.csv` demonstrates the agent's capabilities:
- Detects the `date` column as time series data
- Analyzes commit patterns over time
- Identifies trends and seasonality in development activity
- Generates forecasts for future activity

## Dependencies

- **LangGraph**: Agent workflow orchestration
- **LangChain**: LLM integration and tools
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Statsmodels**: Statistical models and tests
- **Scikit-learn**: Machine learning utilities
- **Prophet**: Advanced forecasting (optional)

## Troubleshooting

### Common Issues

1. **"No time column detected"**: Ensure your CSV has a column with dates/times
2. **"No numeric columns found"**: Make sure you have numeric data to analyze
3. **"Analysis failed"**: Check that your CSV file is properly formatted

### Getting Help

The interactive mode provides helpful error messages and suggestions. For command-line usage, check the generated error reports in the output directory.

## Contributing

This is a demonstration project showcasing LangGraph agent capabilities for time series analysis. Feel free to extend and modify for your specific needs.

## License

This project is for educational and demonstration purposes.
