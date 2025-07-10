
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM

# Modern prompt template using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are Jane, a helpful and friendly AI assistant. You help schedule meetings for dr. James. 
    You are polite, respectful, and aim to provide concise responses of less than 20 words.
    Repeat the user's request back to them to confirm understanding.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize LLM
llm = OllamaLLM(model='gemma3:27b', base_url="http://localhost:11434")

# Create the chain with modern LCEL syntax
chain = prompt_template | llm

# Chat history storage
chat_sessions = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]

# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def get_llm_response(text: str, session_id: str = "default_session") -> str:

    # Invoke the chain with history
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )

    return response.strip()
