"""
Time Series Analysis Chatbot - Streamlit UI
A comprehensive web interface for time series analysis using MCP server tools
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

# MCP imports
from mcp import ClientSession
from mcp.client.sse import sse_client

# Local imports
from mcp_server import (
    perform_comprehensive_ts_analysis,
    forecast_time_series,
    detect_anomalies,
    generate_analysis_report,
    analyze_csv_file,
    upload_pdf,
    search_documents,
    list_documents,
    get_rag_stats,
    generate_rag_answer,
    intelligent_query_router
)


class TimeSeriesChatbot:
    """Advanced chatbot class for time series analysis and RAG with intelligent routing"""

    def __init__(self):
        self.server_url = "http://localhost:8000/sse"
        self.conversation_history = []
        self.current_analysis_results = None
        self.uploaded_csv_file = None
        self.available_csv_files = []
        self.has_documents = False
        self.last_routing_decision = None

    def add_message(self, role: str, content: str, message_type: str = "text", metadata: Optional[Dict[str, Any]] = None):
        """Add a message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "type": message_type,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(message)

        # Keep only last 50 messages to prevent memory issues
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def get_available_tools(self) -> Dict[str, str]:
        """Get available MCP tools"""
        return {
            # Time Series Tools
            "analyze_csv_file": "Analyze basic CSV file structure and data types",
            "perform_comprehensive_ts_analysis": "Perform comprehensive time series analysis (trend, seasonality, stationarity, forecasting)",
            "detect_anomalies": "Detect anomalies in time series data using Prophet",
            "forecast_time_series": "Generate time series forecasts using ARIMA or Exponential Smoothing",
            "generate_analysis_report": "Generate detailed HTML and JSON analysis reports",
            "analyze_trend": "Analyze trend in time series data",
            "detect_seasonality": "Detect seasonality patterns in time series data",
            "test_stationarity": "Test for stationarity in time series data",
            # RAG Tools
            "upload_pdf": "Upload and process PDF documents for RAG system",
            "search_documents": "Search documents with AI-powered answer generation",
            "generate_rag_answer": "Generate comprehensive answers from document context",
            "list_documents": "List all documents available in the RAG system",
            "get_rag_stats": "Get statistics about the RAG system",
            # Intelligent Routing
            "intelligent_query_router": "Analyze user queries and route to appropriate tools"
        }

    def update_available_files(self):
        """Update list of available CSV files"""
        try:
            self.available_csv_files = [str(f) for f in Path(".").glob("*.csv")]
        except Exception as e:
            self.available_csv_files = []

    def check_documents_status(self):
        """Check if documents are available in RAG system"""
        try:
            stats = get_rag_stats()
            self.has_documents = stats.get("statistics", {}).get("total_documents", 0) > 0
        except:
            self.has_documents = False

    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server tool with enhanced error handling"""
        try:
            async with sse_client(url=self.server_url) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()

                    result = await session.call_tool(tool_name, arguments=arguments)

                    # Enhanced JSON parsing with error recovery
                    try:
                        response = json.loads(result.content[0].text)
                        return response
                    except json.JSONDecodeError as json_error:
                        # Attempt to clean and parse the response
                        raw_content = result.content[0].text
                        # Remove markdown code blocks if present
                        if raw_content.startswith('```json'):
                            raw_content = raw_content[7:]
                        if raw_content.endswith('```'):
                            raw_content = raw_content[:-3]

                        try:
                            response = json.loads(raw_content.strip())
                            return response
                        except json.JSONDecodeError:
                            return {
                                "error": f"Failed to parse tool response as JSON: {str(json_error)}",
                                "raw_response": raw_content,
                                "tool_name": tool_name,
                                "arguments": arguments
                            }

        except Exception as e:
            return {
                "error": f"Failed to call MCP tool '{tool_name}': {str(e)}",
                "tool_name": tool_name,
                "arguments": arguments
            }

    async def intelligent_process_query(self, user_query: str) -> Dict[str, Any]:
        """Use intelligent router to process user query and execute appropriate tools"""
        try:
            # Update available files and document status
            self.update_available_files()
            self.check_documents_status()

            # Call intelligent router
            routing_result = intelligent_query_router(
                user_query=user_query,
                available_csv_files=self.available_csv_files,
                has_uploaded_documents=self.has_documents
            )

            if "error" in routing_result:
                return {
                    "action": "error",
                    "message": routing_result["error"],
                    "routing_failed": True
                }

            routing_decision = routing_result.get("routing_decision", {})
            self.last_routing_decision = routing_decision

            # Log routing decision
            analysis = routing_decision.get("analysis", "Unknown analysis")
            confidence = routing_decision.get("confidence", 0.0)
            primary_intent = routing_decision.get("primary_intent", "unknown")

            self.add_message("assistant", f"🤖 Analyzing your query: {analysis} (confidence: {confidence:.2f})", "routing_info",
                           {"routing_decision": routing_decision})

            # Check if clarification is needed
            if "clarification_needed" in routing_decision and routing_decision["clarification_needed"]:
                return {
                    "action": "clarification_needed",
                    "message": routing_decision["clarification_needed"],
                    "routing_decision": routing_decision
                }

            # Check requirements
            needs_csv = routing_decision.get("needs_csv_data", False)
            needs_docs = routing_decision.get("needs_documents", False)

            if needs_csv and not self.available_csv_files:
                return {
                    "action": "needs_csv",
                    "message": "You need to upload a CSV file first for time series analysis. Please use the CSV upload section in the sidebar.",
                    "routing_decision": routing_decision
                }

            if needs_docs and not self.has_documents:
                return {
                    "action": "needs_documents",
                    "message": "You need to upload PDF documents first for document analysis. Please use the PDF upload section in the sidebar.",
                    "routing_decision": routing_decision
                }

            # Execute recommended tools in priority order
            recommended_tools = routing_decision.get("recommended_tools", [])
            if not recommended_tools:
                return {
                    "action": "no_tools_recommended",
                    "message": "I couldn't determine which tools to use for your query. Please try rephrasing your request.",
                    "routing_decision": routing_decision
                }

            # Sort by priority (lower number = higher priority)
            recommended_tools.sort(key=lambda x: x.get("priority", 5))

            results = []
            for tool_info in recommended_tools[:3]:  # Execute up to 3 tools
                tool_name = tool_info.get("tool_name")
                parameters = tool_info.get("parameters", {})
                reason = tool_info.get("reason", "")

                # Set default parameters based on context
                if tool_name in ["analyze_csv_file", "perform_comprehensive_ts_analysis", "detect_anomalies",
                               "forecast_time_series", "generate_analysis_report", "analyze_trend",
                               "detect_seasonality", "test_stationarity"]:
                    if "csv_path" not in parameters and self.uploaded_csv_file:
                        parameters["csv_path"] = str(self.uploaded_csv_file)
                    elif "csv_path" not in parameters and self.available_csv_files:
                        parameters["csv_path"] = self.available_csv_files[0]

                # Execute tool
                tool_result = await self.execute_tool(tool_name, parameters, reason)
                results.append({
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "result": tool_result,
                    "reason": reason
                })

            return {
                "action": "tools_executed",
                "results": results,
                "routing_decision": routing_decision
            }

        except Exception as e:
            return {
                "action": "error",
                "message": f"Intelligent processing failed: {str(e)}",
                "error_details": str(e)
            }

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Execute a specific tool with parameters"""
        try:
            # Map tool names to actual functions
            tool_mapping = {
                "analyze_csv_file": analyze_csv_file,
                "perform_comprehensive_ts_analysis": perform_comprehensive_ts_analysis,
                "detect_anomalies": detect_anomalies,
                "forecast_time_series": forecast_time_series,
                "generate_analysis_report": generate_analysis_report,
                "analyze_trend": analyze_trend,
                "detect_seasonality": detect_seasonality,
                "test_stationarity": test_stationarity,
                "search_documents": search_documents,
                "list_documents": list_documents,
                "get_rag_stats": get_rag_stats,
                "generate_rag_answer": generate_rag_answer,
            }

            if tool_name not in tool_mapping:
                return {"error": f"Unknown tool: {tool_name}"}

            # Execute the tool
            tool_func = tool_mapping[tool_name]

            # Handle async vs sync functions
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**parameters)
            else:
                result = tool_func(**parameters)

            return result

        except Exception as e:
            return {
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name,
                "parameters": parameters
            }

    def format_tool_result(self, tool_name: str, result: Dict[str, Any], reason: str) -> str:
        """Format tool result for display"""
        if "error" in result:
            return f"❌ **{tool_name}** failed: {result['error']}"

        if "success" in result and result["success"]:
            success_msg = f"✅ **{tool_name}** completed successfully"

            # Add specific details based on tool type
            if tool_name == "analyze_csv_file":
                analysis = result.get("analysis", {})
                if "shape" in analysis:
                    success_msg += f"\n- Shape: {analysis['shape']}"
                if "columns" in analysis:
                    success_msg += f"\n- Columns: {len(analysis['columns'])}"

            elif tool_name in ["perform_comprehensive_ts_analysis", "analyze_trend", "detect_seasonality", "test_stationarity"]:
                if "data_info" in result:
                    data_info = result["data_info"]
                    success_msg += f"\n- Time column: {data_info.get('time_column', 'N/A')}"
                    success_msg += f"\n- Value column: {data_info.get('value_column', 'N/A')}"

            elif tool_name == "forecast_time_series":
                success_msg += f"\n- Method: {result.get('method', 'N/A')}"
                success_msg += f"\n- Periods: {result.get('periods', 'N/A')}"

            elif tool_name == "detect_anomalies":
                success_msg += f"\n- Method: {result.get('method', 'N/A')}"

            elif tool_name == "generate_analysis_report":
                if "report_files" in result:
                    files = result["report_files"]
                    success_msg += f"\n- Generated {len(files)} report files"

            elif tool_name == "search_documents":
                total_results = result.get("total_results", 0)
                success_msg += f"\n- Found {total_results} results"
                if "generated_answer" in result:
                    success_msg += "\n- Generated AI answer"

            elif tool_name == "list_documents":
                total_docs = result.get("total_documents", 0)
                success_msg += f"\n- Total documents: {total_docs}"

            elif tool_name == "get_rag_stats":
                stats = result.get("statistics", {})
                success_msg += f"\n- Documents: {stats.get('total_documents', 0)}"
                success_msg += f"\n- Chunks: {stats.get('total_chunks', 0)}"

            return success_msg
        else:
            return f"⚠️ **{tool_name}** completed with warnings"


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Time Series Analysis Chatbot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 AI Analysis Chatbot")
    st.markdown("*Time Series Analysis & Document Q&A powered by MCP Server Tools*")

    # Check for LLM (Azure OpenAI preferred, then Groq) API configuration
    import os
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    groq_key = os.getenv("GROQ_API_KEY")

    azure_ok = bool(azure_key and azure_endpoint and azure_deployment)
    groq_ok = bool(groq_key)

    if not azure_ok and not groq_ok:
        st.warning(
            "⚠️ **No LLM credentials found**: AI-powered document answer generation will be disabled. "
            "Configure either Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`) "
            "or Groq (`GROQ_API_KEY`)."
        )
    elif not azure_ok and groq_ok:
        st.info("Using Groq for AI-powered document answers (Azure OpenAI not configured).")
    else:
        st.info("Using Azure OpenAI for AI-powered document answers." if azure_ok else "Using Groq for AI-powered document answers.")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = TimeSeriesChatbot()

    chatbot = st.session_state.chatbot

    # Sidebar for file upload and tools
    with st.sidebar:
        st.header("📁 File Upload")

        # CSV Upload
        st.subheader("📊 Time Series Data (CSV)")
        csv_file = st.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")

        if csv_file is not None:
            # Save uploaded file temporarily
            temp_path = Path("temp_upload.csv")
            with open(temp_path, "wb") as f:
                f.write(csv_file.getvalue())

            chatbot.uploaded_csv_file = temp_path
            chatbot.update_available_files()  # Update the available files list
            st.success(f"✅ CSV uploaded: {csv_file.name}")

            # Display basic file info
            df = pd.read_csv(temp_path)
            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write("**Column names:**", ", ".join(df.columns.tolist()[:5]))
            if len(df.columns) > 5:
                st.write(f"... and {len(df.columns) - 5} more")

        st.markdown("---")

        # PDF Upload for RAG
        st.subheader("📄 Documents (PDF)")
        pdf_file = st.file_uploader("Upload PDF for RAG", type=['pdf'], key="pdf_uploader")

        if pdf_file is not None:
            # Save uploaded PDF temporarily
            temp_pdf_path = Path("temp_upload.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())

            st.success(f"✅ PDF uploaded: {pdf_file.name}")

            # Upload to RAG system
            if st.button("Process PDF for RAG", key="process_pdf"):
                with st.spinner("Processing PDF and adding to knowledge base..."):
                    try:
                        result = upload_pdf(str(temp_pdf_path), pdf_file.name)
                        if "error" in result:
                            st.error(f"❌ Error: {result['error']}")
                        else:
                            st.success("✅ PDF processed successfully!")
                            st.write(f"**Document ID:** {result.get('document_id', 'N/A')}")
                            st.write(f"**Chunks created:** {result.get('chunks_created', 0)}")
                            # Clean up temp file
                            temp_pdf_path.unlink(missing_ok=True)
                            # Update document status
                            chatbot.check_documents_status()
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")

        # RAG System Stats
        st.markdown("---")
        st.subheader("📊 RAG System")
        if st.button("Show RAG Stats", key="rag_stats_btn"):
            with st.spinner("Getting RAG system statistics..."):
                try:
                    result = get_rag_stats()
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        stats = result.get("statistics", {})
                        st.write(f"**Documents:** {stats.get('total_documents', 0)}")
                        st.write(f"**Chunks:** {stats.get('total_chunks', 0)}")
                        st.write(f"**Storage:** {stats.get('storage_size_mb', 0):.2f} MB")
                except Exception as e:
                    st.error(f"❌ Error getting stats: {str(e)}")

        if st.button("List Documents", key="list_docs_btn"):
            with st.spinner("Getting document list..."):
                try:
                    result = list_documents()
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        docs = result.get("documents", [])
                        if not docs:
                            st.info("No documents uploaded yet.")
                        else:
                            st.write(f"**Total Documents:** {result.get('total_documents', 0)}")
                            for doc in docs[:5]:  # Show first 5
                                st.write(f"• **{doc['name']}** ({doc['chunk_count']} chunks)")
                            if len(docs) > 5:
                                st.write(f"... and {len(docs) - 5} more")
                except Exception as e:
                    st.error(f"❌ Error listing documents: {str(e)}")

        # Quick search
        st.markdown("---")
        st.subheader("🔍 Quick Search")
        search_query = st.text_input("Search documents", key="quick_search", placeholder="Ask a question about your documents...")
        generate_answer = st.checkbox(
            "Generate AI answer",
            value=False,
            key="generate_checkbox",
            help="Use AI to generate an answer based on search results (requires Azure OpenAI or Groq config)",
        )

        if st.button("Search", key="quick_search_btn") and search_query:
            with st.spinner(f"Searching for: '{search_query}'{' with AI generation' if generate_answer else ''}"):
                try:
                    result = search_documents(search_query, top_k=5, generate_answer=generate_answer)
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.success(f"Found {result.get('total_results', 0)} results")

                        # Display generated answer if available
                        if generate_answer and "generated_answer" in result:
                            st.markdown("### 🤖 AI Generated Answer")
                            st.info(result["generated_answer"])

                            with st.expander("View Sources"):
                                for source in result.get("generation_sources", []):
                                    st.write(f"**Source {source['source_id']}:** {source['document_name']} (relevance: {source['relevance_score']})")
                                    st.write(f"*{source['text_preview']}*")
                                    st.write("---")

                        # Display raw search results
                        if result.get("results"):
                            with st.expander("View Raw Search Results", expanded=not generate_answer):
                                for i, res in enumerate(result.get("results", [])[:3], 1):
                                    st.write(f"**Result {i}** - {res.get('document_name', 'Unknown')} (Score: {res.get('score', 0):.3f})")
                                    st.write(res.get('text', '')[:200] + "..." if len(res.get('text', '')) > 200 else res.get('text', ''))
                                    st.write("---")

                        # Show generation error if any
                        if generate_answer and "generation_error" in result:
                            st.warning(f"⚠️ Answer generation failed: {result['generation_error']}")

                except Exception as e:
                    st.error(f"❌ Error searching: {str(e)}")

        st.header("🛠️ Available Tools")

        # Group tools by category
        tools = chatbot.get_available_tools()

        st.subheader("📊 Time Series Analysis")
        ts_tools = {k: v for k, v in tools.items() if k in ["analyze_csv", "comprehensive_analysis", "forecast", "detect_anomalies", "generate_report"]}
        for tool_name, description in ts_tools.items():
            st.write(f"**{tool_name.replace('_', ' ').title()}:**")
            st.write(f"*{description}*")
            st.write("---")

        st.subheader("📄 Document Analysis (RAG)")
        rag_tools = {k: v for k, v in tools.items() if k in ["upload_pdf", "search_documents", "list_documents", "rag_stats"]}
        for tool_name, description in rag_tools.items():
            st.write(f"**{tool_name.replace('_', ' ').title()}:**")
            st.write(f"*{description}*")
            st.write("---")

    # Main chat interface
    st.header("💬 Intelligent Chat Interface")

    # Display conversation history with enhanced formatting
    chat_container = st.container()

    with chat_container:
        if not chatbot.conversation_history:
            st.info("👋 Welcome! I'm your AI assistant for time series analysis and document Q&A. Upload some data and ask me questions!")
        else:
            for i, message in enumerate(chatbot.conversation_history[-20:]):  # Show last 20 messages
                timestamp = message.get("timestamp", datetime.now())
                time_str = timestamp.strftime("%H:%M:%S")

                if message["role"] == "user":
                    # User message with timestamp
                    st.markdown(f"""
<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0; border-left: 4px solid #2196f3;'>
<strong>👤 You</strong> <small style='color: #666;'>({time_str})</small><br>
{message['content']}
</div>
""", unsafe_allow_html=True)

                else:
                    # Assistant message with different styling based on type
                    icon_map = {
                        "routing_info": "🤖",
                        "analysis_result": "📊",
                        "search_results": "🔍",
                        "rag_answer": "💡",
                        "document_list": "📄",
                        "rag_stats": "📈",
                        "report": "📋",
                        "error": "❌",
                        "success": "✅"
                    }

                    icon = icon_map.get(message["type"], "💬")

                    # Different colors for different message types
                    color_map = {
                        "routing_info": "#fff3e0",
                        "analysis_result": "#e8f5e8",
                        "search_results": "#fce4ec",
                        "rag_answer": "#f3e5f5",
                        "document_list": "#e0f2f1",
                        "rag_stats": "#f9fbe7",
                        "report": "#ede7f6",
                        "error": "#ffebee",
                        "success": "#e8f5e8"
                    }

                    bg_color = color_map.get(message["type"], "#f5f5f5")

                    st.markdown(f"""
<div style='background-color: {bg_color}; padding: 10px; border-radius: 10px; margin: 5px 0; border-left: 4px solid #4caf50;'>
<strong>{icon} Assistant</strong> <small style='color: #666;'>({time_str})</small><br>
{message['content']}
</div>
""", unsafe_allow_html=True)

                    # Show additional details for certain message types
                    if message["type"] in ["analysis_result", "report"] and "metadata" in message:
                        with st.expander("View Details", expanded=False):
                            metadata = message["metadata"]
                            if "routing_decision" in metadata:
                                st.json(metadata["routing_decision"])
                            else:
                                st.json(metadata)

    # Chat input section
    st.markdown("---")

    # Quick action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Analyze Data", key="quick_analyze", help="Perform comprehensive analysis"):
            if not chatbot.available_csv_files:
                st.warning("Please upload a CSV file first")
                return

            csv_path = str(getattr(chatbot, "uploaded_csv_file", None) or chatbot.available_csv_files[0])
            with st.spinner("Analyzing data..."):
                tool_output = perform_comprehensive_ts_analysis(csv_path=csv_path, time_column=None, value_column=None)

            chatbot.current_analysis_results = {
                "action": "tools_executed",
                "results": [
                    {
                        "tool_name": "perform_comprehensive_ts_analysis",
                        "reason": "Quick button",
                        "result": tool_output,
                    }
                ],
            }
            st.rerun()

    with col2:
        if st.button("🔮 Forecast", key="quick_forecast", help="Generate forecast"):
            if not chatbot.available_csv_files:
                st.warning("Please upload a CSV file first")
                return

            csv_path = str(getattr(chatbot, "uploaded_csv_file", None) or chatbot.available_csv_files[0])
            with st.spinner("Generating forecast..."):
                tool_output = forecast_time_series(csv_path=csv_path, periods=12, method="arima", time_column=None, value_column=None)

            chatbot.current_analysis_results = {
                "action": "tools_executed",
                "results": [
                    {
                        "tool_name": "forecast_time_series",
                        "reason": "Quick button",
                        "result": tool_output,
                    }
                ],
            }
            st.rerun()

    with col3:
        if st.button("🔍 Search Docs", key="quick_doc_search", help="Search documents"):
            if not chatbot.has_documents:
                st.warning("Please upload PDF documents first")
                return

            with st.spinner("Searching documents..."):
                tool_output = search_documents(query="machine learning", top_k=5, generate_answer=True)

            chatbot.current_analysis_results = {
                "action": "tools_executed",
                "results": [
                    {
                        "tool_name": "search_documents",
                        "reason": "Quick button",
                        "result": tool_output,
                    }
                ],
            }
            st.rerun()

    with col4:
        if st.button("📋 Generate Report", key="quick_report", help="Create analysis report"):
            if not chatbot.available_csv_files:
                st.warning("Please upload a CSV file first")
                return

            csv_path = str(getattr(chatbot, "uploaded_csv_file", None) or chatbot.available_csv_files[0])
            with st.spinner("Generating report..."):
                tool_output = generate_analysis_report(csv_path=csv_path, output_dir="output/ts", time_column=None, value_column=None)

            chatbot.current_analysis_results = {
                "action": "tools_executed",
                "results": [
                    {
                        "tool_name": "generate_analysis_report",
                        "reason": "Quick button",
                        "result": tool_output,
                    }
                ],
            }
            st.rerun()

    # Main chat input
    with st.form(key="chat_form"):
        user_input = st.text_input(
            "Ask me anything about your data:",
            key="user_input",
            placeholder="e.g., 'analyze trends in my data', 'forecast next month', 'search for machine learning concepts'",
            help="Try natural language queries - I'll intelligently route your request to the appropriate tools"
        )

        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            use_intelligent_routing = st.checkbox("Use Intelligent Routing", value=True,
                                                help="Let AI analyze your query and choose the best tools")
            show_routing_details = st.checkbox("Show Routing Details", value=False,
                                             help="Display how the AI analyzed your query")

        submitted = st.form_submit_button("🚀 Send Query")

    if submitted and user_input:
        # Add user message
        chatbot.add_message("user", user_input)

        if use_intelligent_routing:
            # Use intelligent routing
            with st.spinner("🤖 Analyzing your query with AI..."):
                try:
                    result = asyncio.run(chatbot.intelligent_process_query(user_input))

                    # Handle different result types
                    if result["action"] == "error":
                        chatbot.add_message("assistant", result["message"], "error")

                    elif result["action"] == "clarification_needed":
                        chatbot.add_message("assistant", f"🤔 {result['message']}", "routing_info")
                        if show_routing_details and "routing_decision" in result:
                            chatbot.add_message("assistant", f"Routing analysis: {result['routing_decision'].get('analysis', 'N/A')}",
                                              "routing_info", {"routing_decision": result["routing_decision"]})

                    elif result["action"] == "needs_csv":
                        chatbot.add_message("assistant", result["message"], "error")

                    elif result["action"] == "needs_documents":
                        chatbot.add_message("assistant", result["message"], "error")

                    elif result["action"] == "no_tools_recommended":
                        chatbot.add_message("assistant", result["message"], "error")

                    elif result["action"] == "tools_executed":
                        # Process tool results
                        for tool_result in result["results"]:
                            tool_name = tool_result["tool_name"]
                            tool_output = tool_result["result"]
                            reason = tool_result["reason"]

                            formatted_result = chatbot.format_tool_result(tool_name, tool_output, reason)
                            message_type = "error" if "error" in tool_output else "success"

                            chatbot.add_message("assistant", formatted_result, message_type,
                                              {"tool_result": tool_output, "tool_name": tool_name})

                        chatbot.current_analysis_results = result

                        # Show routing details if requested
                        if show_routing_details and "routing_decision" in result:
                            with st.expander("Routing Analysis", expanded=False):
                                st.json(result["routing_decision"])

                except Exception as e:
                    chatbot.add_message("assistant", f"❌ Processing failed: {str(e)}", "error")

        else:
            # Fallback to basic processing (keeping old logic for compatibility)
            chatbot.add_message("assistant", "⚠️ Intelligent routing disabled. Using basic command parsing.", "routing_info")

            # Basic command processing (simplified version)
            user_input_lower = user_input.lower()
            if "analyze" in user_input_lower and chatbot.available_csv_files:
                result = analyze_csv_file(chatbot.available_csv_files[0])
                if "error" not in result:
                    chatbot.add_message("assistant", "✅ Basic analysis completed", "success",
                                      {"analysis_result": result})
                else:
                    chatbot.add_message("assistant", f"❌ Error: {result['error']}", "error")
            else:
                chatbot.add_message("assistant", "Please upload data and try a more specific query, or enable intelligent routing.", "error")

        # Rerun to update chat display
        st.rerun()

    # Analysis Results Display (tool-based wrapper from intelligent_process_query)
    if chatbot.current_analysis_results:
        st.header("📊 Results")

        wrapper = chatbot.current_analysis_results

        # Normal path: intelligent_process_query returns {"action":"tools_executed","results":[...tool runs...] }
        if wrapper.get("action") == "tools_executed" and isinstance(wrapper.get("results"), list):
            for tool_run in wrapper["results"]:
                tool_name = tool_run.get("tool_name", "unknown_tool")
                tool_output = tool_run.get("result") or {}
                reason = tool_run.get("reason", "")

                with st.expander(f"{tool_name}", expanded=True):
                    if reason:
                        st.caption(f"Reason: {reason}")

                    if "error" in tool_output:
                        st.error(tool_output["error"])
                        with st.expander("Raw output", expanded=False):
                            st.json(tool_output)
                        continue

                    # ---- Time series: comprehensive analysis ----
                    if tool_name == "perform_comprehensive_ts_analysis":
                        analysis = tool_output.get("analysis_results") or {}
                        series_preview = tool_output.get("series_preview")

                        if isinstance(series_preview, dict) and "index" in series_preview and "values" in series_preview:
                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                    x=series_preview["index"],
                                    y=series_preview["values"],
                                    mode="lines",
                                    name="Series",
                                    line=dict(color="#1f77b4", width=2),
                                )
                            )
                            trend = analysis.get("trend") if isinstance(analysis, dict) else None
                            if isinstance(trend, dict):
                                ma = trend.get("moving_average")
                                lt = trend.get("linear_trend")
                                if isinstance(ma, dict) and "values" in ma:
                                    x_ma = ma.get("index") or ma.get("index_preview")
                                    if x_ma is not None:
                                        fig.add_trace(go.Scatter(x=x_ma, y=ma["values"], mode="lines", name="Moving avg", line=dict(color="#ff7f0e")))
                                if isinstance(lt, dict) and "values" in lt:
                                    x_lt = lt.get("index") or lt.get("index_preview")
                                    if x_lt is not None:
                                        fig.add_trace(go.Scatter(x=x_lt, y=lt["values"], mode="lines", name="Linear trend", line=dict(color="#2ca02c")))
                            fig.update_layout(title="Time Series Preview (with trend overlays)", xaxis_title="Time", yaxis_title="Value", hovermode="x unified")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback plot using whatever trend components are available
                            trend = analysis.get("trend") if isinstance(analysis, dict) else None
                            if isinstance(trend, dict):
                                ma = trend.get("moving_average")
                                lt = trend.get("linear_trend")
                                fig = go.Figure()
                                if isinstance(ma, dict) and "values" in ma:
                                    x_ma = ma.get("index") or ma.get("index_preview")
                                    if x_ma is not None:
                                        fig.add_trace(go.Scatter(x=x_ma, y=ma["values"], mode="lines", name="Moving avg", line=dict(color="#ff7f0e")))
                                if isinstance(lt, dict) and "values" in lt:
                                    x_lt = lt.get("index") or lt.get("index_preview")
                                    if x_lt is not None:
                                        fig.add_trace(go.Scatter(x=x_lt, y=lt["values"], mode="lines", name="Linear trend", line=dict(color="#2ca02c")))
                                if len(fig.data) > 0:
                                    fig.update_layout(title="Trend Components", xaxis_title="Time", yaxis_title="Value", hovermode="x unified")
                                    st.plotly_chart(fig, use_container_width=True)

                        # Tabs for trend/seasonality/stationarity
                        tab_labels = []
                        tab_map = {}
                        if isinstance(analysis, dict):
                            if "trend" in analysis:
                                tab_labels.append("Trend")
                                tab_map["trend"] = analysis["trend"]
                            if "seasonality" in analysis:
                                tab_labels.append("Seasonality")
                                tab_map["seasonality"] = analysis["seasonality"]
                            if "stationarity" in analysis:
                                tab_labels.append("Stationarity")
                                tab_map["stationarity"] = analysis["stationarity"]

                        if tab_labels:
                            ui_tabs = st.tabs(tab_labels)
                            for ui_tab, lbl in zip(ui_tabs, tab_labels):
                                with ui_tab:
                                    if lbl == "Trend":
                                        st.markdown(display_trend_analysis(tab_map["trend"]))
                                    elif lbl == "Seasonality":
                                        st.markdown(display_seasonality_analysis(tab_map["seasonality"]))
                                    elif lbl == "Stationarity":
                                        st.markdown(display_stationarity_analysis(tab_map["stationarity"]))
                        else:
                            st.info("No analysis_results returned.")

                    # ---- Time series: forecasting ----
                    elif tool_name == "forecast_time_series":
                        st.subheader("🔮 Forecast Results")
                        forecast_container = tool_output.get("forecast") or {}
                        forecast_series = forecast_container.get("forecast") if isinstance(forecast_container, dict) else None
                        if isinstance(forecast_series, dict) and "values" in forecast_series:
                            x_fc = forecast_series.get("index") or forecast_series.get("index_preview")
                            if x_fc is None:
                                with st.expander("Raw forecast payload", expanded=False):
                                    st.json(tool_output)
                            else:
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_fc,
                                        y=forecast_series["values"],
                                        mode="lines+markers",
                                        name="Forecast",
                                        line=dict(color="red"),
                                    )
                                )
                                fig.update_layout(title="Time Series Forecast", xaxis_title="Date", yaxis_title="Value")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            with st.expander("Raw forecast payload", expanded=False):
                                st.json(tool_output)

                    # ---- Reports ----
                    elif tool_name == "generate_analysis_report":
                        report_files = tool_output.get("report_files") or []
                        if report_files:
                            for fp in report_files:
                                st.write(f"📄 {Path(fp).name}")
                            html_file = next((f for f in report_files if str(f).endswith(".html")), None)
                            if html_file and Path(html_file).exists():
                                with open(html_file, "r", encoding="utf-8") as f:
                                    st.components.v1.html(f.read(), height=650, scrolling=True)
                        else:
                            st.info("No report files returned.")

                    # ---- Search docs / interactive answer ----
                    elif tool_name == "search_documents":
                        st.subheader("🔍 Search Documents")
                        if tool_output.get("generated_answer"):
                            st.markdown("### 🤖 AI Generated Answer")
                            st.success(tool_output["generated_answer"])
                            if tool_output.get("generation_sources"):
                                with st.expander("📚 Sources Used", expanded=False):
                                    for source in tool_output["generation_sources"]:
                                        st.write(f"**Source {source['source_id']}:** {source['document_name']}")
                                        st.write(f"*Relevance: {source['relevance_score']}*")
                                        st.write(f"Preview: {source['text_preview']}")
                                        st.write("---")
                        if tool_output.get("generation_error"):
                            st.warning(tool_output["generation_error"])

                        results = tool_output.get("results") or []
                        st.write(f"**Total Results:** {tool_output.get('total_results', len(results))}")
                        if not results:
                            st.info("No relevant documents found.")
                        else:
                            for i, res in enumerate(results[:10], 1):
                                with st.expander(f"Result {i}: {res.get('document_name','Unknown')} (Score: {res.get('score',0):.3f})", expanded=False):
                                    st.write(res.get("text", "")[:800] + ("..." if len(res.get("text","")) > 800 else ""))

                    # ---- Fallback: show raw JSON ----
                    else:
                        with st.expander("Raw output", expanded=False):
                            st.json(tool_output)

        # Backward-compatibility: older shapes
        else:
            with st.expander("Raw results (unrecognized shape)", expanded=False):
                st.json(wrapper)

    # Footer
    st.markdown("---")
    st.markdown("*Time Series Analysis Chatbot - Built with Streamlit and MCP Server*")


def display_analysis_result(content: Dict[str, Any]):
    """Display analysis results in a formatted way"""
    # For simple results, show as formatted text
    if "success" in content and len(content) <= 3:
        if content.get("success"):
            st.success("✅ Operation completed successfully!")
        else:
            st.error("❌ Operation failed")
    else:
        # For complex results, show as expandable JSON
        with st.expander("View Raw Results"):
            st.json(content)


def display_report(content: Dict[str, Any]):
    """Display report information"""
    if "report_files" in content:
        st.success("Report generated successfully!")
        for file_path in content["report_files"]:
            st.write(f"📄 {Path(file_path).name}")

        # Try to display HTML report inline
        html_file = next((f for f in content["report_files"] if f.endswith('.html')), None)
        if html_file and Path(html_file).exists():
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)


def display_trend_analysis(trend_data: Dict[str, Any]) -> str:
    """Display trend analysis results"""
    content = "### Trend Analysis\n\n"

    if "trend_direction" in trend_data:
        direction = trend_data["trend_direction"]
        strength = trend_data.get("trend_strength", 0)
        slope = trend_data.get("trend_slope", 0)

        content += f"**Direction:** {direction.title()}\n\n"
        content += f"**Strength:** {strength:.3f}\n\n"
        content += f"**Slope:** {slope:.6f}\n\n"

        # Add interpretation
        if strength > 0.7:
            content += "📈 **Strong trend detected**\n"
        elif strength > 0.3:
            content += "📊 **Moderate trend detected**\n"
        else:
            content += "📉 **Weak or no significant trend**\n"

    return content


def display_seasonality_analysis(seasonality_data: Dict[str, Any]) -> str:
    """Display seasonality analysis results"""
    content = "### Seasonality Analysis\n\n"

    detected = seasonality_data.get("seasonality_detected", False)
    period = seasonality_data.get("period")
    strength = seasonality_data.get("seasonal_strength", 0)

    content += f"**Seasonality Detected:** {'Yes' if detected else 'No'}\n\n"

    if period:
        content += f"**Period:** {period} time units\n\n"

    content += f"**Strength:** {strength:.3f}\n\n"

    if detected and strength > 0.5:
        content += "🔄 **Strong seasonal pattern detected**\n"
    elif detected:
        content += "🔄 **Seasonal pattern detected**\n"
    else:
        content += "➡️ **No significant seasonality**\n"

    return content


def display_stationarity_analysis(stationarity_data: Dict[str, Any]) -> str:
    """Display stationarity analysis results"""
    content = "### Stationarity Analysis\n\n"

    is_stationary = stationarity_data.get("is_stationary", False)

    content += f"**Stationary:** {'Yes' if is_stationary else 'No'}\n\n"

    # ADF test results
    adf_data = stationarity_data.get("adf_test", {})
    if adf_data:
        p_value = adf_data.get("p_value", 1)
        content += f"**ADF Test p-value:** {p_value:.4f}\n\n"
        content += f"**ADF Test Result:** {'Stationary' if p_value < 0.05 else 'Non-stationary'}\n\n"

    # KPSS test results
    kpss_data = stationarity_data.get("kpss_test", {})
    if kpss_data:
        p_value = kpss_data.get("p_value", 1)
        content += f"**KPSS Test p-value:** {p_value:.4f}\n\n"
        content += f"**KPSS Test Result:** {'Non-stationary' if p_value < 0.05 else 'Stationary'}\n\n"

    if is_stationary:
        content += "✅ **Time series is stationary - good for forecasting**\n"
    else:
        content += "⚠️ **Time series is non-stationary - may need differencing**\n"

    return content


if __name__ == "__main__":
    main()