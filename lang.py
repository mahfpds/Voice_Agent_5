
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM
import ollama
import time

# Modern prompt template using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are Jane, a helpful and friendly AI assistant. You help schedule meetings for dr. James. 
    You are polite, respectful, and aim to provide concise responses of less than 20 words.
    Repeat the user's request back to them to confirm understanding.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Global LLM variable - initialize once at startup
_ollama_llm = None
_chain_with_history = None


def load_ollama_model():
    """Load and warm up Ollama model at startup"""
    global _ollama_llm, _chain_with_history

    if _ollama_llm is None:
        print("[INFO] Loading Ollama model at startup...")

        # Test Ollama service connection
        try:
            client = ollama.Client(host="http://localhost:11434")
            models = client.list()
            print(
                f"[INFO] ✅ Ollama service connected - {len(models['models'])} models available")
        except Exception as e:
            raise ConnectionError(f"❌ Ollama service not available: {e}")

        # Initialize LLM
        _ollama_llm = OllamaLLM(
            model='gemma2:27b', base_url="http://localhost:11434")

        # Create the chain
        chain = prompt_template | _ollama_llm

        # Create the runnable with message history
        _chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # Warm up the model with a test call
        print("[INFO] Warming up Gemma2:27b model...")
        try:
            start_time = time.time()
            test_response = _chain_with_history.invoke(
                {"input": "test"},
                config={"session_id": "warmup_session"}
            )
            warmup_time = time.time() - start_time
            print(
                f"[INFO] ✅ Ollama model warmed up successfully ({warmup_time:.2f}s)")
        except Exception as e:
            print(f"[WARNING] Model warmup failed: {e}")

    return _ollama_llm, _chain_with_history


# Chat history storage
chat_sessions = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]


def get_llm_response(text: str, session_id: str = "default_session") -> str:
    # Use the pre-loaded chain
    _, chain_with_history = load_ollama_model()

    # Invoke the chain with history
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )

    return response.strip()
