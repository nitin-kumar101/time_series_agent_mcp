"""
Hybrid RAG + Time Series Chatbot Agent
- Answers user questions from files in ./db (PDF, CSV) using retrieval-augmented generation
- If intent detected as time series analysis for a CSV, runs analysis and responds with results
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from data_detector import DataAnalyzer
from time_series_tools import TimeSeriesAnalyzer, TimeSeriesVisualizer
from reporting import ReportGenerator, DashboardGenerator, ExportManager

# Optional Groq support (fallback when Azure isn't configured)
try:
    from langchain_groq import ChatGroq  # type: ignore
except Exception:  # pragma: no cover
    ChatGroq = None


class AgentState(TypedDict):
    """State for the time series analysis agent"""
    messages: Annotated[List[Any], "List of messages in the conversation"]
    data_path: Optional[str]
    data_info: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]
    current_task: Optional[str]
    error_message: Optional[str]
    output_files: Optional[Dict[str, str]]


class HybridChatAgent:
    """RAG + Time Series hybrid chatbot."""

    def __init__(self, model_name: str = "llama-3.1-8b-instant", db_dir: str = "db"):
        # Azure OpenAI configuration is expected via:
        # AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        if azure_key and azure_endpoint and azure_deployment:
            self.llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=azure_api_version,
                temperature=0.2,
                max_tokens=1000,
                api_key=azure_key,
            )
        elif os.getenv("GROQ_API_KEY") and ChatGroq is not None:
            self.llm = ChatGroq(model=model_name, temperature=0.2, max_tokens=1000, api_key=os.getenv("GROQ_API_KEY"))
        else:
            self.llm = None
        self.data_analyzer = DataAnalyzer()
        self.ts_analyzer = TimeSeriesAnalyzer()
        self.export_manager = ExportManager()
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)

        # Embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = Chroma(collection_name="dbc", embedding_function=self.embeddings, persist_directory=os.path.join(self.db_dir, ".chroma"))
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Initialize index from db folder
        self._index_db()
        
    def _index_db(self) -> None:
        """Load PDFs and CSVs from db/ into vector store."""
        docs = []
        if not os.path.exists(self.db_dir):
            return
        for root, _, files in os.walk(self.db_dir):
            for f in files:
                path = os.path.join(root, f)
                if f.lower().endswith('.pdf'):
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                elif f.lower().endswith('.csv'):
                    try:
                        loader = CSVLoader(path)
                        docs.extend(loader.load())
                    except Exception:
                        continue
        if not docs:
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()

    def _detect_intent(self, question: str) -> str:
        q = question.lower()
        if ("time series" in q) or ("timeseries" in q) or ("forecast" in q) or ("seasonal" in q) or ("trend" in q):
            return "timeseries"
        if ("analyz" in q or "analyse" in q or "analysis" in q) and ("csv" in q or ".csv" in q):
            return "timeseries"
        return "rag"

    def _answer_rag(self, question: str) -> str:
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content[:1200] for d in docs]) or ""
        prompt = f"You are a helpful assistant. Use the provided context to answer. If answer is not in context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        if self.llm is None:
            return "LLM not available. Configure Azure OpenAI (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT) or Groq (GROQ_API_KEY) to enable generated answers."
        resp = self.llm.invoke(prompt)
        return resp.content if hasattr(resp, 'content') else str(resp)

    def _analyze_csv_path(self, csv_path: str) -> Dict[str, Any]:
        # Build numeric series: commits per day if commit_history.csv else best-effort for first numeric col per time
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Identify time column
        from data_detector import DataTypeDetector
        det = DataTypeDetector()
        info = det.prepare_time_series_data(df)
        prepared = info.get('prepared_data', df)
        time_col = info.get('time_column')
        numeric_cols = info.get('numeric_columns', [])
        if isinstance(prepared.index, pd.DatetimeIndex) and numeric_cols:
            series = prepared[numeric_cols[0]].astype(float)
        elif 'date' in df.columns:
            d2 = df.copy()
            d2['date'] = pd.to_datetime(d2['date'], errors='coerce', utc=True)
            d2 = d2.dropna(subset=['date']).set_index('date').sort_index()
            series = d2.resample('D').size().astype(float)
        else:
            raise ValueError("Could not derive a numeric time series from the CSV")
        results = self.ts_analyzer.comprehensive_analysis(series)
        # Minimal data_info for exporting
        data_info = {
            'total_records': int(len(df)),
            'time_range': {'start': series.index.min(), 'end': series.index.max()},
            'time_span_days': int((series.index.max() - series.index.min()).days),
            'numeric_columns': [series.name or 'value'],
            'categorical_columns': [],
            'missing_values': {}
        }
        out = self.export_manager.export_complete_analysis(series, results, data_info, output_dir="output/ts")

        # Anomaly visualization using Prophet results, if available
        anomalies_path = None
        pa = results.get('prophet_anomalies')
        if isinstance(pa, dict) and 'anomalies' in pa and 'forecast' in pa:
            anomalies_df = pa['anomalies']
            if hasattr(anomalies_df, 'empty') and not anomalies_df.empty:
                try:
                    os.makedirs('output/ts', exist_ok=True)
                    viz = TimeSeriesVisualizer()
                    fig = viz.plot_prophet_anomalies(series, anomalies_df, pa['forecast'])
                    base = os.path.splitext(os.path.basename(csv_path))[0]
                    anomalies_path = os.path.join('output/ts', f'prophet_anomalies_{base}.html')
                    fig.write_html(anomalies_path)
                    out['prophet_anomalies_html'] = anomalies_path
                except Exception:
                    pass

        # Build summary including anomaly count if present
        summary = self.get_analysis_summary({(series.name or 'value'): results})
        if isinstance(pa, dict) and 'anomalies' in pa:
            try:
                count = int(len(pa['anomalies']))
                extra = f"\nAnomalies detected (Prophet): {count}"
                if anomalies_path:
                    extra += f"\nAnomaly chart: {anomalies_path}"
                summary = summary + extra
            except Exception:
                pass

        return {"summary": summary, "outputs": out}
    
    def _load_data_node(self, state: AgentState) -> AgentState:
        """Load and validate data"""
        data_path = state.get("data_path")
        if not data_path:
            state["error_message"] = "No data path provided"
            return state
            
        if not os.path.exists(data_path):
            state["error_message"] = f"Data file not found: {data_path}"
            return state
            
        # Load data using our analyzer
        data_info = self.data_analyzer.analyze_csv(data_path)
        
        if "error" in data_info:
            state["error_message"] = data_info["error"]
            return state
            
        state["data_info"] = data_info
        state["current_task"] = "Data loaded successfully"
        
        return state
    
    def _analyze_data_node(self, state: AgentState) -> AgentState:
        """Analyze data characteristics"""
        data_info = state.get("data_info")
        if not data_info:
            state["error_message"] = "No data info available"
            return state
            
        prepared_data = data_info.get("prepared_data")
        if prepared_data is None:
            state["error_message"] = "No prepared data available"
            return state
            
        # Check if we have time series data
        if not isinstance(prepared_data.index, pd.DatetimeIndex):
            state["error_message"] = "Data does not have a proper time index"
            return state
            
        # Check if we have numeric columns to analyze
        numeric_cols = data_info.get("numeric_columns", [])
        if not numeric_cols:
            state["error_message"] = "No numeric columns found for time series analysis"
            return state
            
        state["current_task"] = "Data analysis completed"
        return state
    
    def _time_series_analysis_node(self, state: AgentState) -> AgentState:
        """Perform comprehensive time series analysis"""
        data_info = state.get("data_info")
        prepared_data = data_info.get("prepared_data")
        numeric_cols = data_info.get("numeric_columns", [])
        
        analysis_results = {}
        
        # Analyze each numeric column
        for col in numeric_cols:
            if col in prepared_data.columns:
                series = prepared_data[col].dropna()
                
                if len(series) > 0:
                    # Perform comprehensive analysis
                    col_analysis = self.ts_analyzer.comprehensive_analysis(series)
                    analysis_results[col] = col_analysis
        
        state["analysis_results"] = analysis_results
        state["current_task"] = "Time series analysis completed"
        
        return state
    
    def _visualization_node(self, state: AgentState) -> AgentState:
        """Generate visualizations"""
        data_info = state.get("data_info")
        analysis_results = state.get("analysis_results")
        
        if not analysis_results:
            state["error_message"] = "No analysis results available for visualization"
            return state
            
        prepared_data = data_info.get("prepared_data")
        numeric_cols = data_info.get("numeric_columns", [])
        
        # Generate visualizations for each numeric column
        visualizations = {}
        
        for col in numeric_cols:
            if col in prepared_data.columns and col in analysis_results:
                series = prepared_data[col].dropna()
                col_analysis = analysis_results[col]
                
                # Create main dashboard
                dashboard = self.dashboard_generator.create_analysis_dashboard(
                    series, col_analysis
                )
                visualizations[f"{col}_dashboard"] = dashboard
                
                # Create individual plots
                ts_plot = self.visualizer.plot_time_series(series, f"{col} Time Series")
                visualizations[f"{col}_timeseries"] = ts_plot
                
                # Trend plot if available
                if "trend" in col_analysis:
                    trend_plot = self.visualizer.plot_trend_analysis(series, col_analysis["trend"])
                    visualizations[f"{col}_trend"] = trend_plot
                
                # Seasonal decomposition if available
                if "seasonality" in col_analysis and "decomposition" in col_analysis["seasonality"]:
                    decomp_plot = self.visualizer.plot_seasonal_decomposition(
                        col_analysis["seasonality"]["decomposition"]
                    )
                    visualizations[f"{col}_decomposition"] = decomp_plot
        
        state["visualizations"] = visualizations
        state["current_task"] = "Visualizations generated"
        
        return state
    
    def _reporting_node(self, state: AgentState) -> AgentState:
        """Generate comprehensive reports"""
        data_info = state.get("data_info")
        analysis_results = state.get("analysis_results")
        visualizations = state.get("visualizations", {})
        
        if not analysis_results:
            state["error_message"] = "No analysis results available for reporting"
            return state
            
        # Generate reports for each analyzed column
        output_files = {}
        
        for col, col_analysis in analysis_results.items():
            prepared_data = data_info.get("prepared_data")
            series = prepared_data[col].dropna()
            
            # Generate summary report
            report = self.report_generator.generate_summary_report(col_analysis, data_info)
            
            # Get visualizations for this column
            col_plots = [v for k, v in visualizations.items() if k.startswith(col)]
            
            # Export complete analysis
            col_output_files = self.export_manager.export_complete_analysis(
                series, col_analysis, data_info, f"output/{col}"
            )
            
            output_files[col] = col_output_files
        
        state["output_files"] = output_files
        state["current_task"] = "Reports generated successfully"
        
        return state
    
    def _error_handler_node(self, state: AgentState) -> AgentState:
        """Handle errors and provide user feedback"""
        error_message = state.get("error_message", "Unknown error occurred")
        
        # Generate error report
        error_report = {
            "error": error_message,
            "timestamp": pd.Timestamp.now().isoformat(),
            "suggestions": [
                "Check if the CSV file exists and is accessible",
                "Verify the CSV file has proper headers",
                "Ensure the data contains time-based information",
                "Check if there are numeric columns to analyze"
            ]
        }
        
        state["error_report"] = error_report
        return state
    
    def chat(self, question: str) -> str:
        intent = self._detect_intent(question)
        if intent == "timeseries":
            # attempt to find a CSV path mentioned; fallback to commit_history.csv
            import re
            m = re.search(r"([\w./\\-]+\.csv)", question, re.IGNORECASE)
            csv_path = m.group(1) if m else os.path.join(self.db_dir, "commit_history.csv") if os.path.exists(os.path.join(self.db_dir, "commit_history.csv")) else "commit_history.csv"
            if not os.path.exists(csv_path):
                return f"I couldn't find the CSV file '{csv_path}'. Please provide a valid path."
            analysis = self._analyze_csv_path(csv_path)
            return f"Time series analysis summary for {csv_path}:\n\n{analysis['summary']}\n\nOutputs saved to: {analysis['outputs']}"
        return self._answer_rag(question)
    
    def get_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the analysis"""
        if not analysis_results:
            return "No analysis results available."
            
        summary_parts = []
        
        for col, results in analysis_results.items():
            summary_parts.append(f"\n=== Analysis for {col} ===")
            
            # Basic stats
            if "basic_stats" in results:
                stats = results["basic_stats"]
                summary_parts.append(f"Mean: {stats.get('mean', 'N/A'):.2f}")
                summary_parts.append(f"Std Dev: {stats.get('std', 'N/A'):.2f}")
                summary_parts.append(f"Range: {stats.get('min', 'N/A'):.2f} to {stats.get('max', 'N/A'):.2f}")
            
            # Trend analysis
            if "trend" in results:
                trend = results["trend"]
                summary_parts.append(f"Trend: {trend.get('trend_direction', 'unknown')} (strength: {trend.get('trend_strength', 0):.2f})")
            
            # Seasonality
            if "seasonality" in results:
                seasonality = results["seasonality"]
                if seasonality.get("seasonality_detected", False):
                    summary_parts.append(f"Seasonality: Detected (period: {seasonality.get('period', 'unknown')})")
                else:
                    summary_parts.append("Seasonality: Not detected")
            
            # Stationarity
            if "stationarity" in results:
                stationarity = results["stationarity"]
                summary_parts.append(f"Stationarity: {'Stationary' if stationarity.get('is_stationary', False) else 'Non-stationary'}")
            
            # Forecasting availability
            forecast_available = False
            if "arima_forecast" in results and "error" not in results["arima_forecast"]:
                forecast_available = True
            if "exp_smoothing_forecast" in results and "error" not in results["exp_smoothing_forecast"]:
                forecast_available = True
                
            summary_parts.append(f"Forecasting: {'Available' if forecast_available else 'Not available'}")
            # Anomalies info
            if "prophet_anomalies" in results and isinstance(results["prophet_anomalies"], dict):
                try:
                    anomalies_df = results["prophet_anomalies"].get("anomalies")
                    count = int(len(anomalies_df)) if anomalies_df is not None else 0
                    summary_parts.append(f"Prophet anomalies: {count}")
                except Exception:
                    pass
        
        return "\n".join(summary_parts)


# Tool definitions for LangChain integration
@tool
def rag_query(question: str) -> str:
    """Answer a question using files in ./db (PDF, CSV). Automatically retrieves relevant chunks."""
    agent = HybridChatAgent()
    return agent.chat(question)


@tool
def analyze_time_series_csv(csv_path: str) -> str:
    """Force time-series analysis of a given CSV path and return a short summary."""
    agent = HybridChatAgent()
    if not os.path.exists(csv_path):
        return f"CSV not found: {csv_path}"
    res = agent._analyze_csv_path(csv_path)
    return res['summary']


if __name__ == "__main__":
    agent = HybridChatAgent()
    while True:
        question = input("Enter a question: ")
        if question.lower() == "exit":
            break
        print(agent.chat(question))