"""
RAG Generation Demo
Demonstrates the Retrieval-Augmented Generation capabilities
"""

import os
from mcp_server import search_documents, generate_rag_answer

def demo_rag():
    """Demo RAG functionality"""
    print("RAG Generation Demo")
    print("=" * 50)

    # Check if Azure or Groq config is available
    azure_ready = (
        os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )
    groq_ready = bool(os.getenv("GROQ_API_KEY"))

    if not azure_ready and not groq_ready:
        print("Warning: No LLM config found. Demo will show search-only functionality.")
        print("Set Azure OpenAI (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT) or Groq (GROQ_API_KEY) to see AI-generated answers.\n")

    # Mock search results (since we don't have real documents uploaded)
    mock_results = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions.',
            'document_name': 'ML_Introduction.pdf',
            'score': 0.95
        },
        {
            'text': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers. These networks can learn complex patterns and representations from large datasets.',
            'document_name': 'Deep_Learning_Guide.pdf',
            'score': 0.89
        },
        {
            'text': 'Supervised learning requires labeled training data where each example has an input and corresponding output. The algorithm learns to map inputs to outputs.',
            'document_name': 'ML_Types.pdf',
            'score': 0.78
        }
    ]

    query = "What is machine learning?"

    print(f"Query: {query}")
    print("\nRetrieved Context Chunks:")
    for i, result in enumerate(mock_results, 1):
        print(f"\n{i}. From: {result['document_name']} (score: {result['score']:.2f})")
        print(f"   Text: {result['text'][:100]}...")

    print("\nGenerating AI Answer...")
    if azure_ready or groq_ready:
        try:
            answer_result = generate_rag_answer(query, mock_results)
            if "success" in answer_result and answer_result["success"]:
                print("\nAI Generated Answer:")
                print(answer_result["answer"])
                print(f"\nSources used: {answer_result['sources_used']}")
            else:
                print(f"\nGeneration failed: {answer_result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"\nError: {e}")
    else:
        print("\nTo see AI-generated answers, configure an LLM provider (Azure OpenAI or Groq).")

    print("\n" + "=" * 50)
    print("Demo completed!")

if __name__ == "__main__":
    demo_rag()