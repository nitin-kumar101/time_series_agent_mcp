import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import re

# PDF processing
import PyPDF2
import pymupdf as fitz  # PyMuPDF for better text extraction

# Vector embeddings and similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from mcp.server.fastmcp import FastMCP

# Time series analysis
import pandas as pd
import plotly.graph_objects as go
from data_detector import DataAnalyzer, DataTypeDetector
from time_series_tools import TimeSeriesAnalyzer, TimeSeriesVisualizer
from reporting import ReportGenerator, DashboardGenerator

# LLM for RAG generation
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Optional Groq support (fallback when Azure isn't configured)
try:
    from langchain_groq import ChatGroq  # type: ignore
except Exception:  # pragma: no cover
    ChatGroq = None

# Create an MCP server
mcp = FastMCP()

# Initialize components
class RAGSystem:
    def __init__(self, storage_dir: str = "rag_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Storage paths
        self.documents_dir = self.storage_dir / "documents"
        self.chunks_dir = self.storage_dir / "chunks"
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.index_file = self.storage_dir / "faiss_index.bin"
        self.metadata_file = self.storage_dir / "metadata.json"
        
        # Create directories
        for dir_path in [self.documents_dir, self.chunks_dir, self.embeddings_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        
        # Initialize FAISS index
        self.index = self._load_or_create_index()
    
    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"documents": {}, "chunks": {}, "next_chunk_id": 0}
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _load_or_create_index(self):
        if self.index_file.exists():
            return faiss.read_index(str(self.index_file))
        else:
            # Create new index with dimension 384 (all-MiniLM-L6-v2 embedding size)
            return faiss.IndexFlatIP(384)
    
    def _save_index(self):
        faiss.write_index(self.index, str(self.index_file))

# Initialize RAG system
rag_system = RAGSystem()

# Initialize time series analysis components
data_analyzer = DataAnalyzer()
data_detector = DataTypeDetector()
ts_analyzer = TimeSeriesAnalyzer()
ts_visualizer = TimeSeriesVisualizer()
report_generator = ReportGenerator()
dashboard_generator = DashboardGenerator()

# Initialize LLM for RAG generation (Azure preferred, then Groq)
llm = None
try:
    # Prefer Azure if configured
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    groq_key = os.getenv("GROQ_API_KEY")
    azure_ok = bool(azure_key and azure_endpoint and azure_deployment)
    groq_ok = bool(groq_key)

    if azure_ok:
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            api_key=azure_key,
            temperature=0.1,
            max_tokens=1000,
        )
        print("LLM provider selected: Azure OpenAI (AZURE_* env vars detected)")
    elif groq_ok:
        if ChatGroq is None:
            print("Warning: GROQ_API_KEY is set but langchain-groq is not available.")
        else:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=1000,
                api_key=groq_key,
            )
            print("LLM provider selected: Groq (GROQ_API_KEY detected)")
    else:
        print(
            "Warning: No LLM credentials found. Set either Azure OpenAI "
            "(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT) "
            "or Groq (GROQ_API_KEY). RAG generation will be disabled."
        )
except Exception as e:
    print(f"Warning: Failed to initialize LLM provider: {e}. RAG generation will be disabled.")

#### Tools ####

@mcp.tool()
def upload_pdf(file_path: str, document_name: Optional[str] = None) -> Dict[str, Any]:
    """Upload and process a PDF file for RAG system"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if not file_path.suffix.lower() == '.pdf':
            return {"error": "File must be a PDF"}
        
        # Generate document ID
        doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
        doc_name = document_name or file_path.stem
        
        # Extract text from PDF
        text_content = _extract_pdf_text(file_path)
        if not text_content.strip():
            return {"error": "No text content found in PDF"}
        
        # Save original document
        doc_file = rag_system.documents_dir / f"{doc_id}.txt"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        # Create chunks
        chunks = _create_text_chunks(text_content)
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = rag_system.metadata["next_chunk_id"]
            rag_system.metadata["next_chunk_id"] += 1
            
            # Save chunk
            chunk_file = rag_system.chunks_dir / f"{chunk_id}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # Generate embedding
            embedding = rag_system.embedding_model.encode([chunk])[0]
            
            # Add to FAISS index
            rag_system.index.add(embedding.reshape(1, -1))
            
            # Save embedding
            np.save(rag_system.embeddings_dir / f"{chunk_id}.npy", embedding)
            
            # Update metadata
            rag_system.metadata["chunks"][str(chunk_id)] = {
                "document_id": doc_id,
                "chunk_index": i,
                "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                "created_at": datetime.now().isoformat()
            }
            
            chunk_ids.append(chunk_id)
        
        # Update document metadata
        rag_system.metadata["documents"][doc_id] = {
            "name": doc_name,
            "original_path": str(file_path),
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunks),
            "created_at": datetime.now().isoformat()
        }
        
        # Save metadata and index
        rag_system._save_metadata()
        rag_system._save_index()
        
        return {
            "success": True,
            "document_id": doc_id,
            "document_name": doc_name,
            "chunks_created": len(chunks),
            "message": f"Successfully processed PDF: {doc_name}"
        }
        
    except Exception as e:
        return {"error": f"Failed to process PDF: {str(e)}"}

@mcp.tool()
def search_documents(query: str, top_k: int = 5, generate_answer: bool = False) -> Dict[str, Any]:
    """Search for relevant document chunks using semantic similarity"""
    try:
        if rag_system.index.ntotal == 0:
            return {"error": "No documents in the system. Please upload some PDFs first."}
        
        # Generate query embedding
        query_embedding = rag_system.embedding_model.encode([query])[0]
        
        # Search in FAISS index
        scores, indices = rag_system.index.search(query_embedding.reshape(1, -1), min(top_k, rag_system.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            # Load chunk text
            chunk_file = rag_system.chunks_dir / f"{idx}.txt"
            if chunk_file.exists():
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_text = f.read()
                
                # Get chunk metadata
                chunk_meta = rag_system.metadata["chunks"].get(str(idx), {})
                doc_meta = rag_system.metadata["documents"].get(chunk_meta.get("document_id", ""), {})
                
                results.append({
                    "chunk_id": int(idx),
                    "score": float(score),
                    "text": chunk_text,
                    "document_name": doc_meta.get("name", "Unknown"),
                    "document_id": chunk_meta.get("document_id"),
                    "chunk_index": chunk_meta.get("chunk_index")
                })
        
        response = {
            "success": True,
            "query": query,
            "results": results,
            "total_results": len(results)
        }

        # Optionally generate an answer using RAG
        if generate_answer and results:
            if llm is None:
                response["generation_error"] = (
                    "LLM not available. Configure either Azure OpenAI "
                    "(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT) "
                    "or Groq (GROQ_API_KEY)."
                )
            else:
                try:
                    generation_result = generate_rag_answer(query, results)
                    if "success" in generation_result and generation_result["success"]:
                        response["generated_answer"] = generation_result["answer"]
                        response["sources_used"] = generation_result["sources_used"]
                        response["generation_sources"] = generation_result["sources"]
                    else:
                        response["generation_error"] = generation_result.get("error", "Failed to generate answer")
                except Exception as e:
                    response["generation_error"] = f"Answer generation failed: {str(e)}"

        return response
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

@mcp.tool()
def list_documents() -> Dict[str, Any]:
    """List all uploaded documents"""
    try:
        documents = []
        for doc_id, doc_info in rag_system.metadata["documents"].items():
            documents.append({
                "document_id": doc_id,
                "name": doc_info["name"],
                "chunk_count": doc_info["chunk_count"],
                "created_at": doc_info["created_at"]
            })
        
        return {
            "success": True,
            "documents": documents,
            "total_documents": len(documents)
        }
        
    except Exception as e:
        return {"error": f"Failed to list documents: {str(e)}"}

@mcp.tool()
def delete_document(document_id: str) -> Dict[str, Any]:
    """Delete a document and its associated chunks"""
    try:
        if document_id not in rag_system.metadata["documents"]:
            return {"error": f"Document not found: {document_id}"}
        
        doc_info = rag_system.metadata["documents"][document_id]
        chunk_ids = doc_info["chunk_ids"]
        
        # Delete chunk files and embeddings
        for chunk_id in chunk_ids:
            chunk_file = rag_system.chunks_dir / f"{chunk_id}.txt"
            embedding_file = rag_system.embeddings_dir / f"{chunk_id}.npy"
            
            if chunk_file.exists():
                chunk_file.unlink()
            if embedding_file.exists():
                embedding_file.unlink()
            
            # Remove from metadata
            if str(chunk_id) in rag_system.metadata["chunks"]:
                del rag_system.metadata["chunks"][str(chunk_id)]
        
        # Delete document file
        doc_file = rag_system.documents_dir / f"{document_id}.txt"
        if doc_file.exists():
            doc_file.unlink()
        
        # Remove from metadata
        del rag_system.metadata["documents"][document_id]
        
        # Rebuild FAISS index (simple approach - could be optimized)
        rag_system.index = faiss.IndexFlatIP(384)
        for chunk_id in rag_system.metadata["chunks"].keys():
            embedding_file = rag_system.embeddings_dir / f"{chunk_id}.npy"
            if embedding_file.exists():
                embedding = np.load(embedding_file)
                rag_system.index.add(embedding.reshape(1, -1))
        
        # Save metadata and index
        rag_system._save_metadata()
        rag_system._save_index()
        
        return {
            "success": True,
            "message": f"Successfully deleted document: {doc_info['name']}"
        }
        
    except Exception as e:
        return {"error": f"Failed to delete document: {str(e)}"}

@mcp.tool()
def get_rag_stats() -> Dict[str, Any]:
    """Get statistics about the RAG system"""
    try:
        total_docs = len(rag_system.metadata["documents"])
        total_chunks = len(rag_system.metadata["chunks"])
        
        # Calculate storage usage
        storage_size = sum(f.stat().st_size for f in rag_system.storage_dir.rglob('*') if f.is_file())
        
        return {
            "success": True,
            "statistics": {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "storage_size_bytes": storage_size,
                "storage_size_mb": round(storage_size / (1024 * 1024), 2),
                "embedding_dimension": 384,
                "storage_directory": str(rag_system.storage_dir)
            }
        }
        
    except Exception as e:
        return {"error": f"Failed to get statistics: {str(e)}"}

@mcp.tool()
def generate_rag_answer(query: str, context_chunks: List[Dict[str, Any]], max_tokens: int = 500) -> Dict[str, Any]:
    """Generate an answer using RAG (Retrieval-Augmented Generation) with retrieved document chunks"""
    try:
        if not context_chunks:
            return {"error": "No context chunks provided for generation"}

        if llm is None:
            return {
                "error": "LLM not available. Configure Azure OpenAI or Groq (GROQ_API_KEY)."
            }

        # Prepare context from retrieved chunks
        context_text = ""
        sources = []

        for i, chunk in enumerate(context_chunks[:5]):  # Use top 5 chunks
            chunk_text = chunk.get('text', '')
            doc_name = chunk.get('document_name', 'Unknown')
            score = chunk.get('score', 0)

            context_text += f"\n[Source {i+1}] {doc_name} (relevance: {score:.3f}):\n{chunk_text}\n"

            sources.append({
                "source_id": i+1,
                "document_name": doc_name,
                "relevance_score": round(score, 3),
                "text_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            })

        # Create RAG prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided document context.
Use only the information from the provided sources to answer the question.
If the context doesn't contain enough information to fully answer the question, say so clearly.
Be comprehensive but concise, and cite your sources when relevant.
Format your answer naturally without explicitly mentioning "Source X" in the main text unless necessary."""

        user_prompt = f"""Question: {query}

Context from documents:
{context_text}

Please provide a comprehensive answer based on the above context."""

        # Generate answer using LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages)
        answer = response.content

        return {
            "success": True,
            "query": query,
            "answer": answer,
            "sources_used": len(sources),
            "sources": sources,
            "context_length": len(context_text),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        return {"error": f"Failed to generate RAG answer: {str(e)}"}

#### Time Series Analysis Tools ####

@mcp.tool()
def analyze_csv_file(csv_path: str) -> Dict[str, Any]:
    """Analyze a CSV file to detect time series data and provide basic statistics"""
    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            return {"error": f"File not found: {csv_path}"}

        if not file_path.suffix.lower() == '.csv':
            return {"error": "File must be a CSV"}

        # Analyze the CSV file
        analysis_result = data_analyzer.analyze_csv(str(file_path))

        # Convert to JSON-serializable format
        json_result = _make_json_serializable(analysis_result)

        return {
            "success": True,
            "file_path": csv_path,
            "analysis": json_result
        }

    except Exception as e:
        return {"error": f"Failed to analyze CSV: {str(e)}"}

@mcp.tool()
def perform_comprehensive_ts_analysis(csv_path: str, time_column: Optional[str] = None, value_column: Optional[str] = None) -> Dict[str, Any]:
    """Perform comprehensive time series analysis on CSV data"""
    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            return {"error": f"File not found: {csv_path}"}

        # Read CSV
        df = pd.read_csv(csv_path)

        # Detect time series data if not specified
        if time_column is None or value_column is None:
            detector = data_detector
            time_col = detector.detect_time_column(df)
            numeric_cols = detector.detect_numeric_columns(df)

            if time_col is None:
                return {"error": "No time column detected in CSV"}

            if value_column is None:
                if len(numeric_cols) == 0:
                    return {"error": "No numeric columns found for analysis"}  
                value_column = numeric_cols[0]  # Use first numeric column
        else:
            time_col = time_column

        # Prepare time series data
        prepared_data = data_detector.prepare_time_series_data(df)
        if "error" in prepared_data:
            return prepared_data

        # Get the time series data
        series_data = prepared_data["prepared_data"][value_column]
        # Small preview for UI plotting (avoid sending full series).
        preview_series = series_data.dropna()
        if len(preview_series) > 500:
            preview_series = preview_series.iloc[-500:]
        series_preview = {
            "index": [
                idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
                for idx in preview_series.index
            ],
            "values": preview_series.tolist(),
        }

        # Perform comprehensive analysis
        analysis_results = ts_analyzer.comprehensive_analysis(series_data)

        # Convert numpy types to JSON-serializable formats
        serializable_analysis = _convert_pandas_objects_to_serializable(analysis_results)
        serializable_data_info = _convert_pandas_objects_to_serializable(prepared_data["data_info"])

        return {
            "success": True,
            "file_path": csv_path,
            "time_column": time_col,
            "value_column": value_column,
            "data_info": serializable_data_info,
            "series_preview": series_preview,
            "analysis_results": serializable_analysis
        }

    except Exception as e:
        return {"error": f"Failed to perform time series analysis: {str(e)}"}

@mcp.tool()
def detect_anomalies(csv_path: str, time_column: Optional[str] = None, value_column: Optional[str] = None,
                    method: str = "prophet", interval_width: float = 0.99) -> Dict[str, Any]:
    """Detect anomalies in time series data using various methods"""
    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            return {"error": f"File not found: {csv_path}"}

        # Read CSV and prepare data (similar to comprehensive analysis)
        df = pd.read_csv(csv_path)

        if time_column is None or value_column is None:
            detector = data_detector
            time_col = detector.detect_time_column(df)
            numeric_cols = detector.detect_numeric_columns(df)

            if time_col is None:
                return {"error": "No time column detected in CSV"}
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for analysis"}

            time_col = time_col or time_column
            value_col = value_column or numeric_cols[0]
        else:
            time_col = time_column
            value_col = value_column

        prepared_data = data_detector.prepare_time_series_data(df)
        if "error" in prepared_data:
            return prepared_data

        series_data = prepared_data["prepared_data"][value_col]

        # Detect anomalies based on method
        if method.lower() == "prophet":
            anomalies_result = ts_analyzer.detect_anomalies_prophet(series_data, interval_width)
        else:
            return {"error": f"Unsupported anomaly detection method: {method}. Use 'prophet'"}

        # Convert pandas objects to JSON-serializable formats
        serializable_anomalies = _convert_pandas_objects_to_serializable(anomalies_result)

        return {
            "success": True,
            "file_path": csv_path,
            "method": method,
            "time_column": time_col,
            "value_column": value_col,
            "anomalies": serializable_anomalies
        }

    except Exception as e:
        return {"error": f"Failed to detect anomalies: {str(e)}"}

@mcp.tool()
def forecast_time_series(csv_path: str, periods: int = 30, method: str = "arima",
                        time_column: Optional[str] = None, value_column: Optional[str] = None) -> Dict[str, Any]:
    """Generate forecasts for time series data using various methods"""
    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            return {"error": f"File not found: {csv_path}"}

        # Read CSV and prepare data
        df = pd.read_csv(csv_path)

        if time_column is None or value_column is None:
            detector = data_detector
            time_col = detector.detect_time_column(df)
            numeric_cols = detector.detect_numeric_columns(df)

            if time_col is None:
                return {"error": "No time column detected in CSV"}
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for analysis"}

            time_col = time_col or time_column
            value_col = value_column or numeric_cols[0]
        else:
            time_col = time_column
            value_col = value_column

        prepared_data = data_detector.prepare_time_series_data(df)
        if "error" in prepared_data:
            return prepared_data

        series_data = prepared_data["prepared_data"][value_col]

        # Generate forecast based on method
        if method.lower() == "arima":
            forecast_result = ts_analyzer.forecast_arima(series_data, periods)
        elif method.lower() == "exponential_smoothing":
            forecast_result = ts_analyzer.forecast_exponential_smoothing(series_data, periods)
        else:
            return {"error": f"Unsupported forecasting method: {method}. Use 'arima' or 'exponential_smoothing'"}

        # Convert pandas objects to JSON-serializable formats
        serializable_forecast = _convert_pandas_objects_to_serializable(forecast_result)

        return {
            "success": True,
            "file_path": csv_path,
            "method": method,
            "periods": periods,
            "time_column": time_col,
            "value_column": value_col,
            "forecast": serializable_forecast
        }

    except Exception as e:
        return {"error": f"Failed to generate forecast: {str(e)}"}

def _convert_numpy_value(value: Any) -> Any:
    """Convert individual numpy values to Python types"""
    if hasattr(value, 'dtype') and hasattr(value, 'tolist'):
        # numpy scalar
        return value.item()
    elif hasattr(value, 'isoformat'):
        # datetime/timestamp
        return value.isoformat()
    elif isinstance(value, (pd.Timestamp, pd.Period)):
        # pandas timestamp/period
        return str(value)
    elif isinstance(value, dict):
        return _convert_pandas_objects_to_serializable(value)
    else:
        return value

def _convert_pandas_objects_to_serializable(data_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert pandas objects in data results to JSON-serializable formats"""
    if "error" in data_result:
        return data_result

    serializable = {}

    for key, value in data_result.items():
        if isinstance(value, pd.Series):
            # Convert pandas Series to dictionary with index as keys
            if hasattr(value.index, 'strftime'):
                # For datetime index, convert to ISO format strings
                serializable[key] = {
                    "values": value.tolist(),
                    "index": [idx.isoformat() if hasattr(idx, 'isoformat') else str(idx) for idx in value.index]
                }
            else:
                # For regular index
                serializable[key] = {
                    "values": value.tolist(),
                    "index": [str(idx) for idx in value.index]
                }
        elif isinstance(value, pd.DataFrame):
            # Convert DataFrame to dictionary, handling datetime objects
            df_dict = value.to_dict('records')
            # Convert any Timestamp objects to ISO strings
            for record in df_dict:
                for k, v in record.items():
                    if hasattr(v, 'isoformat'):  # Timestamp or datetime
                        record[k] = v.isoformat()
                    elif isinstance(v, (pd.Timestamp, pd.Period)):
                        record[k] = str(v)

            serializable[key] = {
                "values": df_dict,
                "columns": value.columns.tolist(),
                "index": [idx.isoformat() if hasattr(idx, 'isoformat') else str(idx) for idx in value.index]
            }
        elif hasattr(value, 'summary'):  # ARIMA model summary
            # Convert model summary to string
            serializable[key] = str(value)
        elif hasattr(value, 'dtype') and hasattr(value, 'tolist'):  # numpy array/scalar
            # Convert numpy arrays and scalars to Python types
            if hasattr(value, 'shape') and len(value.shape) > 0:  # array
                serializable[key] = value.tolist()
            else:  # scalar
                serializable[key] = value.item()
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            serializable[key] = _convert_pandas_objects_to_serializable(value)
        elif isinstance(value, list):
            # Handle lists that might contain numpy types
            serializable[key] = [_convert_numpy_value(item) for item in value]
        elif hasattr(value, 'isoformat'):  # Timestamp, datetime
            # Convert timestamp/datetime to ISO string
            serializable[key] = value.isoformat()
        elif isinstance(value, (pd.Timestamp, pd.Period)):
            # Handle pandas timestamp/period objects
            serializable[key] = str(value)
        else:
            # Keep other values as-is if they're already serializable
            try:
                json.dumps(value)  # Test if it's JSON serializable
                serializable[key] = value
            except (TypeError, ValueError):
                # Convert to string if not serializable
                serializable[key] = str(value)

    return serializable

@mcp.tool()
def generate_analysis_report(csv_path: str, output_dir: str = "output/ts",
                           time_column: Optional[str] = None, value_column: Optional[str] = None) -> Dict[str, Any]:
    """Generate a comprehensive analysis report with visualizations"""
    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            return {"error": f"File not found: {csv_path}"}

        # Read CSV and prepare data
        df = pd.read_csv(csv_path)

        if time_column is None or value_column is None:
            detector = data_detector
            time_col = detector.detect_time_column(df)
            numeric_cols = detector.detect_numeric_columns(df)

            if time_col is None:
                return {"error": "No time column detected in CSV"}
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for analysis"}

            time_col = time_col or time_column
            value_col = value_column or numeric_cols[0]
        else:
            time_col = time_column
            value_col = value_column

        prepared_data = data_detector.prepare_time_series_data(df)
        if "error" in prepared_data:
            return prepared_data

        series_data = prepared_data["prepared_data"][value_col]

        # Perform comprehensive analysis
        analysis_results = ts_analyzer.comprehensive_analysis(series_data)

        # Generate report using the ReportGenerator
        report_data = report_generator.generate_summary_report(analysis_results, prepared_data["data_info"])
        # Add additional fields
        report_data["file_name"] = file_path.name
        report_data["time_column"] = time_col
        report_data["value_column"] = value_col
        report_data["analysis"] = analysis_results

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate HTML report with plots
        plots = []

        # Add trend plot if available
        if "trend" in analysis_results:
            trend_plot = ts_visualizer.plot_trend_analysis(series_data, analysis_results["trend"])
            plots.append(trend_plot)

        # Add seasonal decomposition plot if available
        if "seasonality" in analysis_results:
            # Create a simple seasonal plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series_data.index, y=series_data.values, mode='lines', name='Original'))
            plots.append(fig)

        # Add forecast plots if available
        if "arima_forecast" in analysis_results:
            forecast_plot = ts_visualizer.plot_forecast(series_data, analysis_results["arima_forecast"], "ARIMA Forecast")
            plots.append(forecast_plot)

        if "exp_smoothing_forecast" in analysis_results:
            forecast_plot = ts_visualizer.plot_forecast(series_data, analysis_results["exp_smoothing_forecast"], "Exponential Smoothing Forecast")
            plots.append(forecast_plot)

        # Add anomaly plot if available
        if "prophet_anomalies" in analysis_results and "anomalies" in analysis_results["prophet_anomalies"]:
            anomaly_data = analysis_results["prophet_anomalies"]
            if "anomalies" in anomaly_data and not anomaly_data["anomalies"].empty:
                anomaly_plot = ts_visualizer.plot_prophet_anomalies(series_data, anomaly_data["anomalies"], anomaly_data.get("forecast", pd.DataFrame()))
                plots.append(anomaly_plot)

        # Export report
        report_generator.export_report_html(report_data, plots, str(output_path / "analysis_report.html"))
        report_generator.export_report_json(report_data, str(output_path / "analysis_report.json"))

        return {
            "success": True,
            "file_path": csv_path,
            "output_directory": str(output_path),
            "report_files": [
                str(output_path / "analysis_report.html"),
                str(output_path / "analysis_report.json")
            ],
            "plots_generated": len(plots)
        }

    except Exception as e:
        return {"error": f"Failed to generate report: {str(e)}"}

@mcp.tool()
def create_interactive_dashboard(csv_path: str, output_file: str = "dashboard.html",
                               time_column: Optional[str] = None, value_column: Optional[str] = None) -> Dict[str, Any]:
    """Create an interactive dashboard for time series analysis"""
    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            return {"error": f"File not found: {csv_path}"}

        # Read CSV and prepare data
        df = pd.read_csv(csv_path)

        if time_column is None or value_column is None:
            detector = data_detector
            time_col = detector.detect_time_column(df)
            numeric_cols = detector.detect_numeric_columns(df)

            if time_col is None:
                return {"error": "No time column detected in CSV"}
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for analysis"}

            time_col = time_col or time_column
            value_col = value_column or numeric_cols[0]
        else:
            time_col = time_column
            value_col = value_column

        prepared_data = data_detector.prepare_time_series_data(df)
        if "error" in prepared_data:
            return prepared_data

        series_data = prepared_data["prepared_data"][value_col]

        # Perform comprehensive analysis
        analysis_results = ts_analyzer.comprehensive_analysis(series_data)

        # Create dashboard
        dashboard = dashboard_generator.create_analysis_dashboard(series_data, analysis_results)

        # Save dashboard
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard.write_html(str(output_path))

        return {
            "success": True,
            "file_path": csv_path,
            "dashboard_file": str(output_path),
            "time_column": time_col,
            "value_column": value_col
        }

    except Exception as e:
        return {"error": f"Failed to create dashboard: {str(e)}"}

@mcp.tool()
def analyze_trend(csv_path: str, window: int = 30, time_column: Optional[str] = None,
                 value_column: Optional[str] = None) -> Dict[str, Any]:
    """Analyze trend in time series data"""
    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            return {"error": f"File not found: {csv_path}"}

        # Read CSV and prepare data
        df = pd.read_csv(csv_path)

        if time_column is None or value_column is None:
            detector = data_detector
            time_col = detector.detect_time_column(df)
            numeric_cols = detector.detect_numeric_columns(df)

            if time_col is None:
                return {"error": "No time column detected in CSV"}
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for analysis"}

            time_col = time_col or time_column
            value_col = value_column or numeric_cols[0]
        else:
            time_col = time_column
            value_col = value_column

        prepared_data = data_detector.prepare_time_series_data(df)
        if "error" in prepared_data:
            return prepared_data

        series_data = prepared_data["prepared_data"][value_col]

        # Analyze trend
        trend_results = ts_analyzer.analyze_trend(series_data, window)

        return {
            "success": True,
            "file_path": csv_path,
            "time_column": time_col,
            "value_column": value_col,
            "window": window,
            "trend_analysis": trend_results
        }

    except Exception as e:
        return {"error": f"Failed to analyze trend: {str(e)}"}

@mcp.tool()
def detect_seasonality(csv_path: str, period: Optional[int] = None, time_column: Optional[str] = None,
                      value_column: Optional[str] = None) -> Dict[str, Any]:
    """Detect seasonality patterns in time series data"""
    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            return {"error": f"File not found: {csv_path}"}

        # Read CSV and prepare data
        df = pd.read_csv(csv_path)

        if time_column is None or value_column is None:
            detector = data_detector
            time_col = detector.detect_time_column(df)
            numeric_cols = detector.detect_numeric_columns(df)

            if time_col is None:
                return {"error": "No time column detected in CSV"}
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for analysis"}

            time_col = time_col or time_column
            value_col = value_column or numeric_cols[0]
        else:
            time_col = time_column
            value_col = value_column

        prepared_data = data_detector.prepare_time_series_data(df)
        if "error" in prepared_data:
            return prepared_data

        series_data = prepared_data["prepared_data"][value_col]

        # Detect seasonality
        seasonality_results = ts_analyzer.detect_seasonality(series_data, period)

        return {
            "success": True,
            "file_path": csv_path,
            "time_column": time_col,
            "value_column": value_col,
            "seasonality_analysis": seasonality_results
        }

    except Exception as e:
        return {"error": f"Failed to detect seasonality: {str(e)}"}

@mcp.tool()
def test_stationarity(csv_path: str, time_column: Optional[str] = None,
                     value_column: Optional[str] = None) -> Dict[str, Any]:
    """Test for stationarity in time series data"""
    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            return {"error": f"File not found: {csv_path}"}

        # Read CSV and prepare data
        df = pd.read_csv(csv_path)

        if time_column is None or value_column is None:
            detector = data_detector
            time_col = detector.detect_time_column(df)
            numeric_cols = detector.detect_numeric_columns(df)

            if time_col is None:
                return {"error": "No time column detected in CSV"}
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for analysis"}

            time_col = time_col or time_column
            value_col = value_column or numeric_cols[0]
        else:
            time_col = time_column
            value_col = value_column

        prepared_data = data_detector.prepare_time_series_data(df)
        if "error" in prepared_data:
            return prepared_data

        series_data = prepared_data["prepared_data"][value_col]

        # Test stationarity
        stationarity_results = ts_analyzer.test_stationarity(series_data)

        return {
            "success": True,
            "file_path": csv_path,
            "time_column": time_col,
            "value_column": value_col,
            "stationarity_test": stationarity_results
        }

    except Exception as e:
        return {"error": f"Failed to test stationarity: {str(e)}"}

@mcp.tool()
def intelligent_query_router(user_query: str, available_csv_files: Optional[List[str]] = None,
                           has_uploaded_documents: bool = False) -> Dict[str, Any]:
    """Intelligently analyze user query and route to appropriate tools using LLM"""
    try:
        if not llm:
            return {
                "error": "LLM not available for intelligent routing. Configure Azure OpenAI or Groq (GROQ_API_KEY)."
            }

        # Get list of available CSV files if not provided
        if available_csv_files is None:
            available_csv_files = []
            try:
                for file_path in Path(".").glob("*.csv"):
                    available_csv_files.append(str(file_path))
            except:
                pass

        # Create system prompt for intelligent routing
        system_prompt = """You are an intelligent assistant that analyzes user queries about time series data and document analysis.
Your task is to determine which tools should be called based on the user's query.

Available tools and their purposes:

TIME SERIES TOOLS (require CSV data):
- analyze_csv_file: Basic CSV structure analysis and data types
- perform_comprehensive_ts_analysis: Full analysis (trend, seasonality, stationarity, forecasting)
- detect_anomalies: Find outliers using Prophet
- forecast_time_series: Generate forecasts (ARIMA/Exponential Smoothing)
- generate_analysis_report: Create HTML/JSON reports
- analyze_trend: Trend analysis only
- detect_seasonality: Seasonality detection only
- test_stationarity: Stationarity testing only

RAG/DOCUMENT TOOLS (require uploaded documents):
- search_documents: Search documents with optional AI answer generation
- list_documents: Show all uploaded documents
- get_rag_stats: Get RAG system statistics
- generate_rag_answer: Generate answers from document context

RESPONSE FORMAT: You must respond with a valid JSON object containing:
{
  "analysis": "brief explanation of what the user wants",
  "primary_intent": "time_series|document_search|general_help",
  "recommended_tools": [
    {
      "tool_name": "exact_tool_name_from_above",
      "reason": "why this tool is needed",
      "parameters": {
        "param_name": "param_value"
      },
      "priority": 1-5 (1=highest priority)
    }
  ],
  "needs_csv_data": true|false,
  "needs_documents": true|false,
  "confidence": 0.0-1.0,
  "clarification_needed": "optional message if query is unclear"
}

Guidelines:
- If no CSV files are available and user asks for time series analysis, suggest uploading CSV
- If no documents are uploaded and user asks for document search, suggest uploading PDFs
- For complex queries, recommend multiple tools in priority order
- Be specific with parameter values when possible
- Set confidence based on how clear the user's intent is"""

        # Prepare context information
        context_info = f"""
Available CSV files: {available_csv_files if available_csv_files else 'None'}
Has uploaded documents: {has_uploaded_documents}
User query: {user_query}"""

        # Call LLM for intelligent routing
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze this user query and recommend appropriate tools:\n\n{context_info}")
        ]

        response = llm.invoke(messages)
        raw_response = response.content.strip()

        def _sanitize_json_like(s: str) -> str:
            # Remove JS-style comments and trailing commas (LLMs sometimes emit these).
            s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
            s = re.sub(r"/\\*[\\s\\S]*?\\*/", "", s)
            s = re.sub(r",(\\s*[}\\]])", r"\\1", s)
            return s.strip()

        def _try_parse_routing_json(text: str) -> Optional[Dict[str, Any]]:
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Direct parse attempt
            try:
                return json.loads(_sanitize_json_like(cleaned))
            except json.JSONDecodeError:
                pass

            # Extract outermost JSON object
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = cleaned[start : end + 1]
                try:
                    return json.loads(_sanitize_json_like(candidate))
                except json.JSONDecodeError:
                    return None

            return None

        routing_decision = _try_parse_routing_json(raw_response)
        if routing_decision is None:
            # Heuristic fallback: keep the tool usable even if LLM output isn't valid JSON.
            q = (user_query or "").lower()
            ts_keywords = ["time series", "forecast", "forecasting", "anomaly", "trend", "season", "stationar", "arima", "prophet", "csv"]
            doc_keywords = ["document", "pdf", "rag", "search", "chunk", "embedding", "faiss"]
            is_ts = any(k in q for k in ts_keywords)
            is_doc = any(k in q for k in doc_keywords)

            recommended_tools: List[Dict[str, Any]] = []
            needs_csv_data = False
            needs_documents = False
            primary_intent = "general_help"

            if is_doc and not is_ts:
                primary_intent = "document_search"
                needs_documents = True
                recommended_tools = [
                    {
                        "tool_name": "search_documents",
                        "reason": "Query appears document/RAG related",
                        "parameters": {"query": user_query, "top_k": 5, "generate_answer": True},
                        "priority": 1,
                    }
                ]
            elif is_ts:
                primary_intent = "time_series"
                needs_csv_data = True
                csv_path = available_csv_files[0] if available_csv_files else ""
                if "forecast" in q:
                    recommended_tools = [
                        {
                            "tool_name": "forecast_time_series",
                            "reason": "Query asks for forecasting",
                            "parameters": {"csv_path": csv_path, "periods": 12, "method": "arima", "time_column": None, "value_column": None},
                            "priority": 1,
                        }
                    ]
                elif "report" in q:
                    recommended_tools = [
                        {
                            "tool_name": "generate_analysis_report",
                            "reason": "Query asks for a report",
                            "parameters": {"csv_path": csv_path, "output_dir": "output/ts", "time_column": None, "value_column": None},
                            "priority": 1,
                        }
                    ]
                else:
                    recommended_tools = [
                        {
                            "tool_name": "perform_comprehensive_ts_analysis",
                            "reason": "Query asks for time series analysis",
                            "parameters": {"csv_path": csv_path, "time_column": None, "value_column": None},
                            "priority": 1,
                        }
                    ]
            else:
                recommended_tools = []
                primary_intent = "general_help"

            routing_decision = {
                "analysis": "Heuristic routing fallback due to non-JSON LLM response.",
                "primary_intent": primary_intent,
                "recommended_tools": recommended_tools,
                "needs_csv_data": needs_csv_data,
                "needs_documents": needs_documents,
                "confidence": 0.35,
                "clarification_needed": "LLM routing returned invalid JSON; using heuristic routing.",
            }

        # Validate the response structure
        required_fields = ["analysis", "primary_intent", "recommended_tools", "needs_csv_data", "needs_documents"]
        for field in required_fields:
            if field not in routing_decision:
                routing_decision[field] = "unknown" if field in ["analysis", "primary_intent"] else []

        if "confidence" not in routing_decision:
            routing_decision["confidence"] = 0.5

        # Add metadata
        routing_decision["user_query"] = user_query
        routing_decision["routing_timestamp"] = datetime.now().isoformat()
        routing_decision["available_csv_files"] = available_csv_files or []
        routing_decision["has_documents"] = has_uploaded_documents

        # Normalize common tool parameter mistakes from LLM output (csv_file -> csv_path)
        tools = routing_decision.get("recommended_tools")
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                params = tool.get("parameters")
                if isinstance(params, dict) and "csv_path" not in params:
                    for alt in ("csv_file", "csv_filename", "file", "path"):
                        if isinstance(params.get(alt), str):
                            params["csv_path"] = params.pop(alt)
                            break

        return {
            "success": True,
            "routing_decision": routing_decision
        }

    except Exception as e:
        return {
            "error": f"Intelligent routing failed: {str(e)}",
            "user_query": user_query
        }

# Helper functions
def _make_json_serializable(data):
    """Convert data structures to JSON-serializable format"""
    if isinstance(data, dict):
        return {key: _make_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_make_json_serializable(item) for item in data]
    elif isinstance(data, pd.DataFrame):
        # Convert index to list of serializable values
        index_values = []
        for idx_val in data.index[:5]:  # Only first 5 index values
            if hasattr(idx_val, 'to_pydatetime'):
                index_values.append(idx_val.to_pydatetime().isoformat())
            elif hasattr(idx_val, 'isoformat'):
                index_values.append(idx_val.isoformat())
            else:
                index_values.append(str(idx_val))

        # Convert data to dict without index
        head_dict = {}
        for col in data.columns:
            values = data[col].head().tolist()
            # Convert any timestamps in values
            converted_values = []
            for val in values:
                if hasattr(val, 'to_pydatetime'):
                    converted_values.append(val.to_pydatetime().isoformat())
                elif hasattr(val, 'isoformat'):
                    converted_values.append(val.isoformat())
                else:
                    converted_values.append(val)
            head_dict[col] = converted_values

        return {
            "shape": list(data.shape),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "index_type": str(type(data.index)),
            "index_preview": index_values,
            "head": head_dict
        }
    elif isinstance(data, pd.Series):
        # Convert index to list of serializable values
        index_values = []
        for idx_val in data.index[:10]:  # Only first 10 index values
            if hasattr(idx_val, 'to_pydatetime'):
                index_values.append(idx_val.to_pydatetime().isoformat())
            elif hasattr(idx_val, 'isoformat'):
                index_values.append(idx_val.isoformat())
            else:
                index_values.append(str(idx_val))

        return {
            "name": data.name,
            "dtype": str(data.dtype),
            "length": len(data),
            "index_type": str(type(data.index)),
            "index_preview": index_values,
            "values": data.head(10).tolist() if len(data) > 0 else []
        }
    elif isinstance(data, pd.DatetimeIndex):
        return {
            "name": data.name,
            "length": len(data),
            "start": str(data.min()) if len(data) > 0 else None,
            "end": str(data.max()) if len(data) > 0 else None,
            "frequency": pd.infer_freq(data) if len(data) > 2 else None
        }
    elif hasattr(data, 'isoformat'):  # datetime objects
        return data.isoformat()
    elif hasattr(data, 'to_pydatetime'):  # pandas Timestamp
        return data.to_pydatetime().isoformat()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif hasattr(data, 'dtype'):  # pandas/numpy dtypes
        return str(data)
    elif hasattr(data, 'summary'):  # statsmodels summary objects
        return str(data)
    elif 'prophet' in str(type(data)).lower():  # prophet model objects
        return f"Prophet model: {type(data).__name__}"
    else:
        # Try to convert to basic types, otherwise return string representation
        try:
            json.dumps(data)
            return data
        except TypeError:
            return str(data)

def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF using PyMuPDF for better quality"""
    try:
        doc = fitz.open(file_path)
        text_content = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_content += page.get_text()
            text_content += "\n\n"  # Add page separator
        
        doc.close()
        return text_content.strip()
        
    except Exception as e:
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
                    text_content += "\n\n"
                
                return text_content.strip()
        except Exception as e2:
            raise Exception(f"Failed to extract text with both PyMuPDF and PyPDF2: {str(e2)}")

def _create_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Create overlapping text chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:  # Only if we find a good break point
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return [chunk for chunk in chunks if chunk.strip()]

#### Resources ####

@mcp.resource("rag://documents")
def get_documents_resource() -> str:
    """Get list of all documents in the RAG system"""
    try:
        docs = []
        for doc_id, doc_info in rag_system.metadata["documents"].items():
            docs.append(f"- {doc_info['name']} (ID: {doc_id}, Chunks: {doc_info['chunk_count']})")
        
        if not docs:
            return "No documents uploaded yet."
        
        return "Documents in RAG system:\n" + "\n".join(docs)
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

@mcp.resource("rag://document/{document_id}")
def get_document_content(document_id: str) -> str:
    """Get the full content of a specific document"""
    try:
        if document_id not in rag_system.metadata["documents"]:
            return f"Document not found: {document_id}"
        
        doc_file = rag_system.documents_dir / f"{document_id}.txt"
        if not doc_file.exists():
            return f"Document file not found: {document_id}"
        
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc_info = rag_system.metadata["documents"][document_id]
        header = f"Document: {doc_info['name']}\n"
        header += f"Created: {doc_info['created_at']}\n"
        header += f"Chunks: {doc_info['chunk_count']}\n"
        header += "=" * 50 + "\n\n"
        
        return header + content
    except Exception as e:
        return f"Error retrieving document: {str(e)}"

@mcp.resource("rag://stats")
def get_rag_stats_resource() -> str:
    """Get RAG system statistics"""
    try:
        stats = get_rag_stats()
        if "error" in stats:
            return stats["error"]

        s = stats["statistics"]
        return f"""RAG System Statistics:
- Total Documents: {s['total_documents']}
- Total Chunks: {s['total_chunks']}
- Storage Size: {s['storage_size_mb']} MB
- Embedding Dimension: {s['embedding_dimension']}
- Storage Directory: {s['storage_directory']}"""
    except Exception as e:
        return f"Error retrieving statistics: {str(e)}"

#### Time Series Analysis Resources ####

@mcp.resource("ts://csv-files")
def get_available_csv_files() -> str:
    """Get list of available CSV files in the current directory"""
    try:
        csv_files = list(Path(".").glob("*.csv"))
        if not csv_files:
            return "No CSV files found in the current directory."

        files_info = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                files_info.append(f"- {csv_file.name} (Rows: {len(df)}, Columns: {len(df.columns)})")
            except Exception:
                files_info.append(f"- {csv_file.name} (Could not read)")

        return "Available CSV files:\n" + "\n".join(files_info)
    except Exception as e:
        return f"Error retrieving CSV files: {str(e)}"

@mcp.resource("ts://analysis-capabilities")
def get_ts_analysis_capabilities() -> str:
    """Get information about available time series analysis capabilities"""
    return """Time Series Analysis Capabilities:

📊 Data Analysis:
- CSV file analysis and structure detection
- Automatic time column and numeric column detection
- Data quality assessment and gap detection

📈 Time Series Analysis:
- Trend analysis with moving averages and linear regression
- Seasonality detection and decomposition
- Stationarity testing (ADF, KPSS tests)
- Comprehensive analysis combining all methods

🔮 Forecasting:
- ARIMA forecasting with automatic order selection
- Exponential smoothing forecasting
- Prophet-based anomaly detection

📋 Reporting & Visualization:
- Interactive HTML dashboards
- Comprehensive analysis reports (JSON/HTML)
- Multiple visualization types (trend, seasonal, forecast, anomalies)
- Export capabilities for reports and plots

🛠️ Tools Available:
- analyze_csv_file: Basic CSV analysis
- perform_comprehensive_ts_analysis: Full time series analysis
- detect_anomalies: Anomaly detection using Prophet
- forecast_time_series: Generate forecasts
- generate_analysis_report: Create detailed reports
- create_interactive_dashboard: Build interactive dashboards
- analyze_trend: Trend analysis
- detect_seasonality: Seasonality detection
- test_stationarity: Stationarity testing"""

@mcp.resource("ts://sample-data")
def get_sample_data_info() -> str:
    """Get information about sample datasets available"""
    sample_files = ["Electric_Production.csv", "Weather_dataset.csv", "commit_history.csv"]

    info = "Sample Datasets Available:\n\n"
    for filename in sample_files:
        file_path = Path(filename)
        if file_path.exists():
            try:
                df = pd.read_csv(filename)
                info += f"📁 {filename}:\n"
                info += f"  - Rows: {len(df)}\n"
                info += f"  - Columns: {len(df.columns)}\n"
                info += f"  - Column names: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}\n\n"
            except Exception as e:
                info += f"📁 {filename}: (Error reading: {str(e)})\n\n"
        else:
            info += f"📁 {filename}: Not found\n\n"

    return info

#### Prompts ####

@mcp.prompt()
def rag_query_prompt(query: str, context_chunks: str) -> List[tuple]:
    """Generate a prompt for RAG-based question answering"""
    return [
        ("system", "You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer questions. If the context doesn't contain enough information to answer the question, say so clearly."),
        ("user", f"Context:\n{context_chunks}\n\nQuestion: {query}\n\nPlease answer the question based on the provided context.")
    ]

@mcp.prompt()
def document_summary_prompt(document_content: str) -> str:
    """Generate a prompt for document summarization"""
    return f"Please provide a comprehensive summary of the following document:\n\n{document_content}"

@mcp.prompt()
def search_suggestions_prompt(query: str, available_documents: str) -> List[tuple]:
    """Generate search suggestions based on available documents"""
    return [
        ("system", "You are a helpful assistant that suggests better search queries based on available documents."),
        ("user", f"Available documents:\n{available_documents}\n\nUser query: '{query}'\n\nSuggest 3-5 alternative or refined search queries that might yield better results from these documents.")
    ]

#### Time Series Analysis Prompts ####

@mcp.prompt()
def ts_analysis_prompt(csv_filename: str, analysis_type: str) -> str:
    """Generate a prompt for time series analysis guidance"""
    return f"""Please analyze the time series data in '{csv_filename}' focusing on {analysis_type}.

Available analysis tools:
- analyze_csv_file: Basic CSV structure analysis
- perform_comprehensive_ts_analysis: Full analysis (trend, seasonality, stationarity, forecasting)
- detect_anomalies: Find anomalies using Prophet
- forecast_time_series: Generate forecasts using ARIMA or Exponential Smoothing
- generate_analysis_report: Create detailed HTML/JSON reports
- create_interactive_dashboard: Build interactive visualizations
- analyze_trend: Trend analysis only
- detect_seasonality: Seasonality detection only
- test_stationarity: Stationarity testing only

Start with basic CSV analysis, then proceed with comprehensive analysis, and finally generate a report."""

@mcp.prompt()
def forecasting_guidance_prompt(data_characteristics: str) -> List[tuple]:
    """Generate guidance for choosing forecasting methods"""
    return [
        ("system", "You are a time series forecasting expert. Provide guidance on which forecasting methods to use based on data characteristics."),
        ("user", f"Data characteristics: {data_characteristics}\n\nRecommend the most appropriate forecasting method(s) from: ARIMA, Exponential Smoothing, or Prophet-based approaches. Explain why each method would be suitable or not suitable for this data.")
    ]

@mcp.prompt()
def anomaly_detection_prompt(data_description: str, business_context: str) -> str:
    """Generate guidance for anomaly detection analysis"""
    return f"""Analyze anomalies in this time series data: {data_description}

Business Context: {business_context}

Use the detect_anomalies tool with Prophet method to identify unusual patterns. Consider:
- What constitutes a meaningful anomaly in this business context?
- Are there expected seasonal patterns that should be considered normal?
- What threshold (interval_width) would be appropriate?
- How should detected anomalies be interpreted and acted upon?

After detection, generate a report to visualize and document the findings."""


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='sse')