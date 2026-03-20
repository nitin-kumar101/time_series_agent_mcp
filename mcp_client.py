import asyncio
import json
from pathlib import Path
from mcp import ClientSession
from mcp.client.sse import sse_client


class MCPClient:
    def __init__(self, server_url: str = "http://localhost:8000/sse"):
        self.server_url = server_url
    
    async def run_rag_demo(self):
        """Run a comprehensive demo of the RAG system"""
        async with sse_client(url=self.server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()

                print("🚀 RAG System Demo Starting...")
                print("=" * 50)

                # 1. List available tools
                await self._list_tools(session)

                # 2. Check initial system stats
                await self._check_system_stats(session)

                # 3. List documents (should be empty initially)
                await self._list_documents(session)

                # 4. Demo PDF upload (you'll need to provide a PDF path)
                await self._demo_pdf_upload(session)

                # 5. Demo document search
                await self._demo_search(session)

                # 6. Demo resources
                await self._demo_resources(session)

                # 7. Demo prompts
                await self._demo_prompts(session)

                print("\n🎉 RAG System Demo Complete!")

    async def run_ts_demo(self):
        """Run a comprehensive demo of time series analysis tools"""
        async with sse_client(url=self.server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()

                print("📊 Time Series Analysis Demo Starting...")
                print("=" * 50)

                # 1. List available CSV files
                await self._list_csv_files(session)

                # 2. Demo CSV analysis
                await self._demo_csv_analysis(session)

                # 3. Demo comprehensive time series analysis
                await self._demo_comprehensive_analysis(session)

                # 4. Demo anomaly detection
                await self._demo_anomaly_detection(session)

                # 5. Demo forecasting
                await self._demo_forecasting(session)

                # 6. Demo report generation
                await self._demo_report_generation(session)

                # 7. Demo time series resources
                await self._demo_ts_resources(session)

                # 8. Demo time series prompts
                await self._demo_ts_prompts(session)

                print("\n🎉 Time Series Analysis Demo Complete!")

    async def run_full_demo(self):
        """Run both RAG and Time Series demos"""
        await self.run_rag_demo()
        print("\n" + "=" * 50)
        await self.run_ts_demo()
    
    async def _list_tools(self, session):
        """List all available tools"""
        print("\n📋 Available Tools:")
        tools = await session.list_tools()
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")
    
    async def _check_system_stats(self, session):
        """Check RAG system statistics"""
        print("\n📊 System Statistics:")
        try:
            result = await session.call_tool("get_rag_stats")
            stats = json.loads(result.content[0].text)
            if "error" in stats:
                print(f"  Error: {stats['error']}")
            else:
                s = stats["statistics"]
                print(f"  Documents: {s['total_documents']}")
                print(f"  Chunks: {s['total_chunks']}")
                print(f"  Storage: {s['storage_size_mb']} MB")
        except Exception as e:
            print(f"  Error getting stats: {e}")
    
    async def _list_documents(self, session):
        """List all documents in the system"""
        print("\n📚 Documents in System:")
        try:
            result = await session.call_tool("list_documents")
            docs = json.loads(result.content[0].text)
            if "error" in docs:
                print(f"  Error: {docs['error']}")
            elif docs["total_documents"] == 0:
                print("  No documents uploaded yet.")
            else:
                for doc in docs["documents"]:
                    print(f"  - {doc['name']} ({doc['chunk_count']} chunks)")
        except Exception as e:
            print(f"  Error listing documents: {e}")
    
    async def _demo_pdf_upload(self, session):
        """Demo PDF upload functionality"""
        print("\n📄 PDF Upload Demo:")
        
        # You can modify this path to point to an actual PDF file
        sample_pdf_path = input("Enter path to a PDF file (or press Enter to skip): ").strip()
        
        if not sample_pdf_path:
            print("  Skipping PDF upload demo (no file provided)")
            return
        
        if not Path(sample_pdf_path).exists():
            print(f"  File not found: {sample_pdf_path}")
            return
        
        try:
            result = await session.call_tool("upload_pdf", arguments={
                "file_path": sample_pdf_path,
                "document_name": Path(sample_pdf_path).stem
            })
            
            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"  Error: {response['error']}")
            else:
                print(f"  ✅ Success: {response['message']}")
                print(f"  Document ID: {response['document_id']}")
                print(f"  Chunks created: {response['chunks_created']}")
        except Exception as e:
            print(f"  Error uploading PDF: {e}")
    
    async def _demo_search(self, session):
        """Demo search functionality"""
        print("\n🔍 Search Demo:")
        
        # First check if we have any documents
        try:
            result = await session.call_tool("list_documents")
            docs = json.loads(result.content[0].text)
            
            if docs["total_documents"] == 0:
                print("  No documents to search. Please upload a PDF first.")
                return
            
            # Demo searches
            search_queries = [
                "What is the main topic?",
                "key findings",
                "methodology",
                "conclusion"
            ]
            
            for query in search_queries:
                print(f"\n  Query: '{query}'")
                try:
                    result = await session.call_tool("search_documents", arguments={
                        "query": query,
                        "top_k": 3
                    })
                    
                    search_results = json.loads(result.content[0].text)
                    if "error" in search_results:
                        print(f"    Error: {search_results['error']}")
                    else:
                        print(f"    Found {search_results['total_results']} results:")
                        for i, res in enumerate(search_results['results'][:2], 1):
                            print(f"    {i}. Score: {res['score']:.3f}")
                            print(f"       Document: {res['document_name']}")
                            print(f"       Text: {res['text'][:100]}...")
                except Exception as e:
                    print(f"    Error searching: {e}")
        
        except Exception as e:
            print(f"  Error in search demo: {e}")
    
    async def _demo_resources(self, session):
        """Demo resource functionality"""
        print("\n📦 Resources Demo:")

        # List available resources
        resources = await session.list_resources()
        print("  Available resources:")
        for resource in resources.resources:
            print(f"    - {resource.uri}: {resource.description}")
        
        # Try to read some resources
        resource_uris = ["rag://documents", "rag://stats"]
        
        for uri in resource_uris:
            try:
                print(f"\n  Reading resource: {uri}")
                content = await session.read_resource(uri)
                print(f"    Content: {content.contents[0].text[:200]}...")
            except Exception as e:
                print(f"    Error reading {uri}: {e}")
    
    async def _demo_prompts(self, session):
        """Demo prompt functionality"""
        print("\n💬 Prompts Demo:")

        # List available prompts
        prompts = await session.list_prompts()
        print("  Available prompts:")
        for prompt in prompts.prompts:
            print(f"    - {prompt.name}: {prompt.description}")
        
        # Demo RAG query prompt
        try:
            print(f"\n  Testing RAG query prompt:")
            prompt = await session.get_prompt("rag_query_prompt", arguments={
                "query": "What are the key findings?",
                "context_chunks": "Sample context: This document discusses important research findings about AI and machine learning applications."
            })
            print(f"    Generated prompt with {len(prompt.messages)} messages")
            for i, msg in enumerate(prompt.messages):
                print(f"    {i+1}. {msg.role}: {msg.content.text[:100]}...")
        except Exception as e:
            print(f"    Error with RAG prompt: {e}")

    #### Time Series Analysis Methods ####

    async def _list_csv_files(self, session):
        """List available CSV files"""
        print("\n📁 Available CSV Files:")
        try:
            content = await session.read_resource("ts://csv-files")
            print(content.contents[0].text)
        except Exception as e:
            print(f"  Error retrieving CSV files: {e}")

    async def _demo_csv_analysis(self, session):
        """Demo CSV file analysis"""
        print("\n📊 CSV Analysis Demo:")

        # Get available CSV files
        try:
            csv_files_result = await session.read_resource("ts://csv-files")
            csv_content = csv_files_result.contents[0].text

            if "No CSV files found" in csv_content:
                print("  No CSV files available for analysis.")
                return

            # Extract first CSV file name for demo
            lines = csv_content.split('\n')
            for line in lines:
                if line.startswith('- '):
                    csv_filename = line.split(' ')[1].split(' ')[0]
                    break
            else:
                print("  Could not find a CSV file to analyze.")
                return

            print(f"  Analyzing: {csv_filename}")

            result = await session.call_tool("analyze_csv_file", arguments={
                "csv_path": csv_filename
            })

            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"  Error: {response['error']}")
            else:
                print("  ✅ Analysis successful!")
                analysis = response["analysis"]
                print(f"    - Time column detected: {analysis.get('time_column_detected', 'None')}")
                print(f"    - Numeric columns: {len(analysis.get('numeric_columns', []))}")
                print(f"    - Total rows: {analysis.get('total_rows', 'Unknown')}")

        except Exception as e:
            print(f"  Error in CSV analysis demo: {e}")

    async def _demo_comprehensive_analysis(self, session):
        """Demo comprehensive time series analysis"""
        print("\n🔬 Comprehensive Time Series Analysis Demo:")

        try:
            # Use Electric_Production.csv as example
            csv_path = "Electric_Production.csv"
            if not Path(csv_path).exists():
                print(f"  Sample file {csv_path} not found. Please ensure sample data is available.")
                return

            print(f"  Analyzing: {csv_path}")

            result = await session.call_tool("perform_comprehensive_ts_analysis", arguments={
                "csv_path": csv_path
            })

            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"  Error: {response['error']}")
            else:
                print("  ✅ Comprehensive analysis completed!")
                results = response["analysis_results"]
                print("    Analysis components:")
                for key in results.keys():
                    if key in ['trend', 'seasonality', 'stationarity', 'arima_forecast', 'exp_smoothing_forecast', 'prophet_anomalies']:
                        print(f"    - {key}: ✓")

        except Exception as e:
            print(f"  Error in comprehensive analysis demo: {e}")

    async def _demo_anomaly_detection(self, session):
        """Demo anomaly detection"""
        print("\n🔍 Anomaly Detection Demo:")

        try:
            csv_path = "Electric_Production.csv"
            if not Path(csv_path).exists():
                print(f"  Sample file {csv_path} not found.")
                return

            print(f"  Detecting anomalies in: {csv_path}")

            result = await session.call_tool("detect_anomalies", arguments={
                "csv_path": csv_path,
                "method": "prophet"
            })

            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"  Error: {response['error']}")
            else:
                print("  ✅ Anomaly detection completed!")
                anomalies = response["anomalies"]
                if "anomalies" in anomalies and not anomalies["anomalies"].empty:
                    anomaly_count = len(anomalies["anomalies"])
                    print(f"    - Detected {anomaly_count} anomalies")
                else:
                    print("    - No anomalies detected")

        except Exception as e:
            print(f"  Error in anomaly detection demo: {e}")

    async def _demo_forecasting(self, session):
        """Demo time series forecasting"""
        print("\n🔮 Forecasting Demo:")

        try:
            csv_path = "Electric_Production.csv"
            if not Path(csv_path).exists():
                print(f"  Sample file {csv_path} not found.")
                return

            print(f"  Generating forecast for: {csv_path}")

            # Demo ARIMA forecasting
            result = await session.call_tool("forecast_time_series", arguments={
                "csv_path": csv_path,
                "method": "arima",
                "periods": 12
            })

            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"  Error: {response['error']}")
            else:
                print("  ✅ ARIMA forecast completed!")
                forecast = response["forecast"]
                print(f"    - Forecast periods: {len(forecast.get('forecast', []))}")
                print(".4f")
                print(".4f")

        except Exception as e:
            print(f"  Error in forecasting demo: {e}")

    async def _demo_report_generation(self, session):
        """Demo report generation"""
        print("\n📋 Report Generation Demo:")

        try:
            csv_path = "Electric_Production.csv"
            if not Path(csv_path).exists():
                print(f"  Sample file {csv_path} not found.")
                return

            print(f"  Generating analysis report for: {csv_path}")

            result = await session.call_tool("generate_analysis_report", arguments={
                "csv_path": csv_path,
                "output_dir": "output/demo"
            })

            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"  Error: {response['error']}")
            else:
                print("  ✅ Report generation completed!")
                print(f"    - Output directory: {response['output_directory']}")
                print(f"    - Report files: {len(response['report_files'])}")
                print(f"    - Plots generated: {response['plots_generated']}")

        except Exception as e:
            print(f"  Error in report generation demo: {e}")

    async def _demo_ts_resources(self, session):
        """Demo time series resources"""
        print("\n📦 Time Series Resources Demo:")

        # List available resources
        resources = await session.list_resources()
        ts_resources = [r for r in resources.resources if str(r.uri).startswith("ts://")]
        print(f"  Available time series resources ({len(ts_resources)}):")
        for resource in ts_resources:
            print(f"    - {resource.uri}: {resource.description}")

        # Try to read time series resources
        resource_uris = ["ts://csv-files", "ts://analysis-capabilities", "ts://sample-data"]

        for uri in resource_uris:
            try:
                print(f"\n  Reading resource: {uri}")
                content = await session.read_resource(uri)
                # Show first 300 characters
                text = content.contents[0].text[:300]
                if len(content.contents[0].text) > 300:
                    text += "..."
                print(f"    Content: {text}")
            except Exception as e:
                print(f"    Error reading {uri}: {e}")

    async def _demo_ts_prompts(self, session):
        """Demo time series prompts"""
        print("\n💬 Time Series Prompts Demo:")

        # List available prompts
        prompts = await session.list_prompts()
        ts_prompts = [p for p in prompts.prompts if "ts" in p.name.lower() or "forecasting" in p.name.lower() or "anomaly" in p.name.lower()]
        print(f"  Available time series prompts ({len(ts_prompts)}):")
        for prompt in ts_prompts:
            print(f"    - {prompt.name}: {prompt.description}")

        # Demo time series analysis prompt
        try:
            print("\n  Testing time series analysis prompt:")
            prompt = await session.get_prompt("ts_analysis_prompt", arguments={
                "csv_filename": "Electric_Production.csv",
                "analysis_type": "trend and seasonality analysis"
            })
            print(f"    Generated prompt with {len(prompt.messages)} messages")
            for i, msg in enumerate(prompt.messages):
                content = msg.content.text[:150]
                if len(msg.content.text) > 150:
                    content += "..."
                print(f"    {i+1}. {msg.role}: {content}")
        except Exception as e:
            print(f"    Error with time series prompt: {e}")

    async def _list_all_resources(self, session):
        """List all available resources"""
        resources = await session.list_resources()
        print(f"\n📦 All Resources ({len(resources.resources)}):")
        for resource in resources.resources:
            print(f"  - {resource.uri}: {resource.description}")

    async def _list_all_prompts(self, session):
        """List all available prompts"""
        prompts = await session.list_prompts()
        print(f"\n💬 All Prompts ({len(prompts.prompts)}):")
        for prompt in prompts.prompts:
            print(f"  - {prompt.name}: {prompt.description}")

    async def interactive_mode(self):
        """Run interactive mode for testing both RAG and Time Series tools"""
        async with sse_client(url=self.server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()

                print("\n🔧 Interactive MCP Client")
                print("Available commands:")
                print("RAG Commands: upload, search, list-docs, rag-stats")
                print("Time Series Commands: analyze-csv, ts-analysis, anomalies, forecast, report, ts-list")
                print("General: tools, resources, prompts, quit")

                while True:
                    try:
                        command = input("\nmcp> ").strip().lower()

                        if command == "quit":
                            break
                        elif command == "tools":
                            await self._list_tools(session)
                        elif command == "resources":
                            await self._list_all_resources(session)
                        elif command == "prompts":
                            await self._list_all_prompts(session)

                        # RAG Commands
                        elif command == "upload":
                            file_path = input("PDF file path: ").strip()
                            if file_path and Path(file_path).exists():
                                result = await session.call_tool("upload_pdf", arguments={
                                    "file_path": file_path
                                })
                                response = json.loads(result.content[0].text)
                                print(json.dumps(response, indent=2))
                            else:
                                print("Invalid file path")

                        elif command == "search":
                            if query := input("Search query: ").strip():
                                result = await session.call_tool("search_documents", arguments={
                                    "query": query,
                                    "top_k": 5
                                })
                                response = json.loads(result.content[0].text)
                                print(json.dumps(response, indent=2))

                        elif command == "list-docs":
                            result = await session.call_tool("list_documents")
                            response = json.loads(result.content[0].text)
                            print(json.dumps(response, indent=2))

                        elif command == "rag-stats":
                            result = await session.call_tool("get_rag_stats")
                            response = json.loads(result.content[0].text)
                            print(json.dumps(response, indent=2))

                        # Time Series Commands
                        elif command == "analyze-csv":
                            csv_path = input("CSV file path: ").strip()
                            if csv_path and Path(csv_path).exists():
                                result = await session.call_tool("analyze_csv_file", arguments={
                                    "csv_path": csv_path
                                })
                                response = json.loads(result.content[0].text)
                                print(json.dumps(response, indent=2))
                            else:
                                print("Invalid CSV file path")

                        elif command == "ts-analysis":
                            csv_path = input("CSV file path: ").strip()
                            if csv_path and Path(csv_path).exists():
                                result = await session.call_tool("perform_comprehensive_ts_analysis", arguments={
                                    "csv_path": csv_path
                                })
                                print(result)
                                response = json.loads(result.content[0].text)
                                print(json.dumps(response, indent=2))
                            else:
                                print("Invalid CSV file path")

                        elif command == "anomalies":
                            csv_path = input("CSV file path: ").strip()
                            method = input("Detection method (prophet): ").strip() or "prophet"
                            if csv_path and Path(csv_path).exists():
                                result = await session.call_tool("detect_anomalies", arguments={
                                    "csv_path": csv_path,
                                    "method": method
                                })
                                print(result)
                                response = json.loads(result.content[0].text)
                                print(json.dumps(response, indent=2))
                            else:
                                print("Invalid CSV file path")

                        elif command == "forecast":
                            csv_path = input("CSV file path: ").strip()
                            method = input("Forecasting method (arima/exponential_smoothing): ").strip() or "arima"
                            periods = int(input("Number of periods to forecast (default 30): ").strip() or "30")
                            if csv_path and Path(csv_path).exists():
                                result = await session.call_tool("forecast_time_series", arguments={
                                    "csv_path": csv_path,
                                    "method": method,
                                    "periods": periods
                                })
                                print(result)
                                response = json.loads(result.content[0].text)
                                print(json.dumps(response, indent=2))
                            else:
                                print("Invalid CSV file path")

                        elif command == "report":
                            csv_path = input("CSV file path: ").strip()
                            output_dir = input("Output directory (default: output/ts): ").strip() or "output/ts"
                            if csv_path and Path(csv_path).exists():
                                result = await session.call_tool("generate_analysis_report", arguments={
                                    "csv_path": csv_path,
                                    "output_dir": output_dir
                                })
                                response = json.loads(result.content[0].text)
                                print(json.dumps(response, indent=2))
                            else:
                                print("Invalid CSV file path")

                        elif command == "ts-list":
                            await self._list_csv_files(session)

                        else:
                            print("Unknown command. Type 'tools' to see all available commands.")

                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"Error: {e}")


async def main():
    client = MCPClient()

    print("MCP System Client")
    print("Available modes:")
    print("1. Run RAG demo")
    print("2. Run Time Series demo")
    print("3. Run full demo (RAG + Time Series)")
    print("4. Interactive mode")

    choice = input("Choose mode (1-4): ").strip()

    if choice == "1":
        await client.run_rag_demo()
    elif choice == "2":
        await client.run_ts_demo()
    elif choice == "3":
        await client.run_full_demo()
    elif choice == "4":
        await client.interactive_mode()
    else:
        print("Invalid choice")



if __name__ == "__main__":
    asyncio.run(main())