# Required Installations:
# pip install langchain langchain_community langchain_openai faiss-cpu ollama duckduckgo-search python-dotenv pydantic==1.10.11 dateparser tk requests beautifulsoup4 PyYAML

import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog, filedialog, Checkbutton, BooleanVar
import threading
import time
from datetime import datetime, timedelta
import dateparser # More robust parsing than dateutil.parser
# import json # Not explicitly used, can be removed if truly unused
# import os # Already imported
import yaml # Not explicitly used, can be removed if truly unused
import logging
import traceback
from queue import Queue # For thread-safe communication
import sqlite3 # For persistent calendar
import os # For FAISS path check

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory # Still used for direct LLM and reminders
# from langchain.agents import AgentExecutor, create_openai_tools_agent, tool # create_openai_tools_agent and AgentExecutor removed
from langchain.agents import tool # Keep @tool decorator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # ChatPromptTemplate might not be directly used by new agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool # Experimental but useful
from langchain_core.pydantic_v1 import BaseModel, Field # Use v1 for compatibility if needed

# --- Configuration ---
# User and Host (from original config)
USER_NAME = "User" # Replace with your desired user name
HOST_NAME = "Assistant" # Replace with your desired assistant name

# LLM Configuration
LOCAL_API_BASE = "http://localhost:1234/v1" # Standard Ollama API endpoint (e.g., LM Studio)
OLLAMA_LOCAL_API_BASE = "http://localhost:11434/v1" # Standard Ollama API endpoint (for embeddings if different)
LOCAL_MODEL_NAME = "qwen2.5:14b-instruct-q6_K" # CHANGE TO YOUR AVAILABLE OLLAMA MODEL
DUMMY_API_KEY = "ollama" # Placeholder API key

# Embedding Configuration
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Common Ollama embedding model
OLLAMA_BASE_URL = OLLAMA_LOCAL_API_BASE.replace("/v1", "")

# RAG / Vector Store Configuration
FAISS_INDEX_PATH = "faiss_index_assistant"

# Calendar Configuration
CALENDAR_DB_PATH = "calendar_events.db" # SQLite database for calendar
CALENDAR_REMINDER_MINUTES = 15
CALENDAR_CHECK_INTERVAL_SECONDS = 60

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Setup ---
def _get_db_connection():
    conn = sqlite3.connect(CALENDAR_DB_PATH)
    return conn

def init_calendar_db():
    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            event_datetime TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logging.info(f"Calendar database initialized at {CALENDAR_DB_PATH}")

# --- LangChain Setup ---

# 1. LLM
try:
    llm = ChatOpenAI(
        base_url=LOCAL_API_BASE,
        model=LOCAL_MODEL_NAME,
        api_key=DUMMY_API_KEY,
        # stream=False, # stream is deprecated, use streaming=False
        temperature=0.2,
        streaming=False, # Explicitly set streaming to False
    )
    logging.info(f"LLM initialized: Model={LOCAL_MODEL_NAME}, BaseURL={LOCAL_API_BASE}")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}", exc_info=True)
    messagebox.showerror("LLM Error", f"Could not connect to LLM at {LOCAL_API_BASE}. Please ensure Ollama is running and the model '{LOCAL_MODEL_NAME}' is available.\nError: {e}")
    exit()

# 2. Embeddings
try:
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL
    )
    _ = embeddings.embed_query("Test query") # Test embedding
    logging.info(f"Embeddings initialized: Model={EMBEDDING_MODEL_NAME}, BaseURL={OLLAMA_BASE_URL}")
except Exception as e:
    logging.error(f"Failed to initialize Embeddings: {e}", exc_info=True)
    messagebox.showerror("Embedding Error", f"Could not initialize Ollama embeddings with model '{EMBEDDING_MODEL_NAME}' at {OLLAMA_BASE_URL}.\nError: {e}")
    exit()

# 3. Vector Store (FAISS) and In-Memory Calendar (populated from DB)
knowledge_base_texts = []
knowledge_base_metadatas = []
calendar_events = [] # List to store {'id': int, 'content': str, 'datetime': datetime} - NOW POPULATED FROM DB
calendar_lock = threading.Lock() # Protects access to calendar_events list
vector_store = None
next_event_id = 1 # Will be set based on max ID from DB

def load_vector_store():
    global vector_store
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            logging.info(f"Loaded FAISS index from {FAISS_INDEX_PATH}")
        except Exception as e:
            logging.error(f"Failed to load FAISS index: {e}. Creating a new one.", exc_info=True)
            vector_store = FAISS.from_texts(["Initial knowledge base entry: Assistant started."], embedding=embeddings, metadatas=[{"timestamp": datetime.now().isoformat()}])
    else:
        logging.info("No FAISS index found. Creating a new one.")
        vector_store = FAISS.from_texts(["Initial knowledge base entry: Assistant started."], embedding=embeddings, metadatas=[{"timestamp": datetime.now().isoformat()}])

def save_vector_store():
    if vector_store:
        try:
            vector_store.save_local(FAISS_INDEX_PATH)
            logging.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")
        except Exception as e:
            logging.error(f"Failed to save FAISS index: {e}", exc_info=True)

# 4. Memory (For direct LLM calls and reminders, not for LangGraph agent primarily)
memory = ConversationBufferWindowMemory(
    k=5, memory_key="chat_history", return_messages=True
)

# --- Custom Tools ---

@tool
def save_note_to_vector_store(note: str) -> str:
    """Saves a piece of information or a note provided by the user to the knowledge base."""
    global vector_store
    if not note or not isinstance(note, str):
        return "Error: Invalid note content provided."
    try:
        metadata = {"timestamp": datetime.now().isoformat(), "source": "user_note"}
        vector_store.add_texts([note], metadatas=[metadata])
        save_vector_store()
        logging.info(f"Saved note to vector store: {note[:50]}...")
        return f"Note saved successfully: '{note[:50]}...'"
    except Exception as e:
        logging.error(f"Error saving note: {e}", exc_info=True)
        return f"Error: Could not save note. Details: {e}"

@tool
def retrieve_notes_from_vector_store(query: str) -> str:
    """Retrieves relevant information or notes from the knowledge base based on a user's query."""
    global vector_store
    if not query or not isinstance(query, str):
        return "Error: Invalid query provided."
    try:
        results = vector_store.similarity_search_with_score(query, k=3)
        if not results:
            return "No relevant notes found."
        formatted_results = [
            f"- Note (Saved {datetime.fromisoformat(doc.metadata.get('timestamp', 'N/A')).strftime('%Y-%m-%d %H:%M') if doc.metadata.get('timestamp') else 'N/A'}, Relevance: {score:.2f}): {doc.page_content}"
            for doc, score in results
        ]
        logging.info(f"Retrieved notes for query: {query[:50]}...")
        return "Found the following relevant notes:\n" + "\n".join(formatted_results)
    except Exception as e:
        logging.error(f"Error retrieving notes: {e}", exc_info=True)
        return f"Error: Could not retrieve notes. Details: {e}"

class CalendarEventInput(BaseModel):
    event_content: str = Field(..., description="The description or title of the calendar event.")
    event_datetime_str: str = Field(..., description="The date and time of the event (e.g., 'tomorrow at 3 pm', 'next Friday at 10am'). Should be a specific, parsable date and time.")

@tool("add_calendar_event", args_schema=CalendarEventInput)
def add_calendar_event(event_content: str, event_datetime_str: str) -> str:
    """Adds an event to the user's calendar. Stores persistently."""
    try:
        event_dt = dateparser.parse(event_datetime_str, settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False})
        if not event_dt:
            return f"Error: Could not understand the date/time '{event_datetime_str}'. Please specify a clearer date and time (e.g., 'tomorrow at 3 pm', 'July 10th 2pm')."

        conn = _get_db_connection()
        cursor = conn.cursor()
        event_dt_iso = event_dt.isoformat()
        cursor.execute("INSERT INTO events (content, event_datetime) VALUES (?, ?)", (event_content, event_dt_iso))
        new_event_id = cursor.lastrowid
        conn.commit()
        logging.info(f"DB: Added calendar event ID {new_event_id}: '{event_content}' at {event_dt_iso}")
        # Inform the GUI thread to refresh calendar by calling a method or using the queue
        # For now, the response will trigger a refresh if calendar_potentially_modified is true
        return f"OK. I've added '{event_content}' to your calendar for {event_dt.strftime('%A, %B %d, %Y at %I:%M %p')} (ID: {new_event_id}). Your calendar view should update shortly if the agent indicates a change."
    except Exception as e:
        logging.error(f"Error in add_calendar_event tool: {e}", exc_info=True)
        return f"Error: Could not add calendar event to database. Details: {e}"
    finally:
        if 'conn' in locals() and conn:
            conn.close()


class CalendarRetrieveInput(BaseModel):
    start_datetime_str: str = Field(..., description="The start date/time for retrieving events (e.g., 'today', 'next Monday').")
    end_datetime_str: str = Field(..., description="The end date/time for retrieving events (e.g., 'end of today', 'next Friday').")

@tool("retrieve_calendar_events", args_schema=CalendarRetrieveInput)
def retrieve_calendar_events(start_datetime_str: str, end_datetime_str: str) -> str:
    """Retrieves events from the user's calendar within a specified date/time range from persistent storage."""
    try:
        start_dt = dateparser.parse(start_datetime_str, settings={'PREFER_DATES_FROM': 'current_period', 'RETURN_AS_TIMEZONE_AWARE': False})
        end_dt = dateparser.parse(end_datetime_str, settings={'PREFER_DATES_FROM': 'current_period', 'RETURN_AS_TIMEZONE_AWARE': False})


        if not start_dt:
            return f"Error: Could not understand the start date/time '{start_datetime_str}'."
        if not end_dt:
             return f"Error: Could not understand the end date/time '{end_datetime_str}'."

        if start_dt > end_dt: # If range is inverted, assume user means from start_dt to end of that day
            end_dt = start_dt.replace(hour=23, minute=59, second=59)
            logging.info(f"End date was before start date for calendar retrieval. Adjusted end date to end of start day: {end_dt.isoformat()}")


        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content, event_datetime FROM events WHERE event_datetime BETWEEN ? AND ? ORDER BY event_datetime ASC",
            (start_dt.isoformat(), end_dt.isoformat())
        )
        db_events = cursor.fetchall()
        conn.close()

        found_events = [
            f"- (ID: {row[0]}) {row[1]} on {datetime.fromisoformat(row[2]).strftime('%A, %B %d at %I:%M %p')}"
            for row in db_events
        ]
        
        logging.info(f"Retrieved {len(found_events)} calendar events from DB between {start_dt.isoformat()} and {end_dt.isoformat()}")

        if not found_events:
            return f"You have no events scheduled between {start_dt.strftime('%B %d')} and {end_dt.strftime('%B %d')}."
        else:
            return f"Here are your scheduled events between {start_dt.strftime('%B %d')} and {end_dt.strftime('%B %d')}:\n" + "\n".join(found_events)
    except Exception as e:
        logging.error(f"Error retrieving calendar events from DB: {e}", exc_info=True)
        return f"Error: Could not retrieve calendar events. Details: {e}"

# --- Agent Setup ---
tools = [
    DuckDuckGoSearchRun(), PythonREPLTool(),
    save_note_to_vector_store, retrieve_notes_from_vector_store,
    add_calendar_event, retrieve_calendar_events,
]

# System prompt text for LangGraph ReAct Agent
# The original agent_prompt is not directly used in the same way.
SYSTEM_PROMPT_TEXT = (
    f"You are '{HOST_NAME}', a helpful personal assistant for '{USER_NAME}'. "
    f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
    f"You have access to the following tools: {[tool.name for tool in tools]}. "
    "Use tools when necessary to answer the user's request or perform actions. "
    "Previous conversation history will be part of the messages you receive. "
    "When asked to add a calendar event, ensure you get a specific date and time from the user if it's ambiguous. "
    "If a date/time is ambiguous for adding an event, ask for clarification before calling the add_calendar_event tool."
)

checkpointer = None
langgraph_agent_app = None
LANGGRAPH_THREAD_ID = "main_assistant_chat_session" # Define a consistent thread ID

try:
    checkpointer = MemorySaver()
    # Create the LangGraph ReAct agent
    langgraph_agent_app = create_react_agent(
        llm,
        tools=tools,
        checkpointer=checkpointer
        # create_react_agent typically infers prompt structure for ReAct.
        # The system message is usually passed as the first message in a new thread.
    )
    logging.info("LangGraph ReAct Agent created successfully.")
except Exception as e:
    logging.error(f"Failed to create LangGraph ReAct Agent: {e}", exc_info=True)
    messagebox.showerror("Agent Error", f"Could not create LangGraph agent.\nError: {e}")
    # langgraph_agent_app will remain None, handled in process_user_input


# --- Tkinter Application ---
class VirtualAssistantApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title(f"{USER_NAME}'s Personal Assistant ({HOST_NAME})")
        self.root.geometry("1024x600")

        self.chat_history = [] # For UI display primarily
        self.reminded_event_ids = set()
        self.message_queue = Queue()

        # Menu Bar
        self.menu_bar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Clear Chat", command=self.clear_chat)
        self.file_menu.add_command(label="Save Chat History", command=self.save_chat_history)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_closing)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.root.config(menu=self.menu_bar)

        self.top_frame = tk.Frame(self.root, pady=5)
        self.top_frame.pack(fill=tk.X)
        self.use_tools_var = BooleanVar(value=True)
        self.tools_checkbox = Checkbutton(self.top_frame, text="Enable Tools (Agent)", variable=self.use_tools_var)
        self.tools_checkbox.pack(side=tk.LEFT, padx=10)

        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.split_frame = tk.Frame(self.main_frame)
        self.split_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_frame = tk.Frame(self.split_frame)
        self.chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.calendar_frame = tk.Frame(self.split_frame, bd=2, relief=tk.GROOVE, width=300)
        self.calendar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.calendar_frame.pack_propagate(False)
        
        self.calendar_header = tk.Label(self.calendar_frame, text="Upcoming Events (Next 7 Days)", font=('Arial', 10, 'bold'))
        self.calendar_header.pack(pady=5)
        self.calendar_display = scrolledtext.ScrolledText(self.calendar_frame, wrap=tk.WORD, width=35, state='disabled')
        self.calendar_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.refresh_calendar_btn = tk.Button(self.calendar_frame, text="Refresh Calendar", command=self.manual_refresh_calendar)
        self.refresh_calendar_btn.pack(pady=5)

        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, state='disabled')
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_display.tag_configure('user', foreground='blue', font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure('assistant', foreground='green', font=('Arial', 10))
        self.chat_display.tag_configure('system', foreground='gray', font=('Arial', 9, 'italic'))
        self.chat_display.tag_configure('error', foreground='red', font=('Arial', 10, 'bold'))

        self.input_frame = tk.Frame(self.chat_frame)
        self.input_frame.pack(fill=tk.X)
        self.input_entry = tk.Entry(self.input_frame, font=('Arial', 11))
        self.input_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 10))
        self.input_entry.bind("<Return>", self.send_message_event)
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message_event, width=10)
        self.send_button.pack(side=tk.RIGHT)

        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.set_status("Ready.")

        # --- Initialization ---
        init_calendar_db()
        load_vector_store()
        self._load_calendar_events_from_db()

        self.add_message("System", f"Assistant initialized. Type 'clear' to reset chat. Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n", tag='system')
        
        self.calendar_thread_stop_event = threading.Event()
        self.calendar_thread = threading.Thread(target=self.background_calendar_checker, daemon=True)
        self.calendar_thread.start()
        self.refresh_calendar_display()

        self.root.after(100, self.check_message_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _load_calendar_events_from_db(self):
        global calendar_events, next_event_id
        max_id_found = 0
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, event_datetime FROM events ORDER BY event_datetime ASC")
            db_rows = cursor.fetchall()
            conn.close()

            with calendar_lock:
                calendar_events.clear()
                for row in db_rows:
                    event_id, content, dt_str = row
                    try:
                        event_datetime_obj = datetime.fromisoformat(dt_str)
                        calendar_events.append({
                            'id': event_id,
                            'content': content,
                            'datetime': event_datetime_obj
                        })
                        if event_id > max_id_found:
                            max_id_found = event_id
                    except ValueError as ve:
                        logging.error(f"Could not parse datetime string '{dt_str}' for event ID {event_id}: {ve}")

                next_event_id = max_id_found + 1
            logging.info(f"Loaded {len(calendar_events)} events from DB into memory. Next event ID calculated as {next_event_id}.")
        except Exception as e:
            logging.error(f"Error loading calendar events from DB: {e}", exc_info=True)
            messagebox.showerror("Calendar DB Error", f"Could not load calendar events from database.\nError: {e}")
        # self.refresh_calendar_display() # Refresh after loading - called in init already

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def add_message(self, sender, message, tag='assistant'):
        self.chat_display.config(state='normal')
        if self.chat_history: self.chat_display.insert(tk.END, "\n")
        self.chat_display.insert(tk.END, f"{sender}: ", ('user' if tag == 'user' else tag,))
        self.chat_display.insert(tk.END, message, (tag,))
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        if tag != 'system': self.chat_history.append(f"{sender}: {message}")

    def send_message_event(self, event=None):
        user_input = self.input_entry.get().strip()
        if not user_input: return
        self.input_entry.delete(0, tk.END)
        self.add_message(USER_NAME, user_input, tag='user')
        if user_input.lower() == 'clear':
            self.clear_chat()
            return
        self.set_status("Thinking...")
        self.send_button.config(state='disabled')
        self.input_entry.config(state='disabled')
        threading.Thread(target=self.process_user_input, args=(user_input,)).start()

    def process_user_input(self, user_input):
        response_text = ""
        error_occurred = False
        calendar_potentially_modified = False
        try:
            use_agent = self.use_tools_var.get()
            if use_agent and langgraph_agent_app and checkpointer: # Check for langgraph_agent_app and checkpointer
                logging.info(f"Invoking LangGraph ReAct Agent for input: {user_input}")
                
                thread_config = {"configurable": {"thread_id": LANGGRAPH_THREAD_ID}}
                
                # Construct messages for LangGraph agent
                # Check if this is the first message in the thread to prepend system prompt
                current_graph_state = checkpointer.get(thread_config)
                
                messages_for_graph = []
                if not current_graph_state or not current_graph_state.get("messages"):
                    # No history for this thread_id in checkpointer, so add SystemMessage
                    messages_for_graph.append(SystemMessage(content=SYSTEM_PROMPT_TEXT))
                
                messages_for_graph.append(HumanMessage(content=user_input))
                
                inputs = {"messages": messages_for_graph}
                
                # Invoke the LangGraph agent
                # The stream method can be used for intermediate steps, invoke for final answer
                full_graph_response = langgraph_agent_app.invoke(inputs, config=thread_config)
                
                # Extract the final AI response
                final_messages = full_graph_response.get('messages', [])
                if final_messages and isinstance(final_messages[-1], AIMessage):
                    response_text = final_messages[-1].content
                else:
                    response_text = "LangGraph agent did not provide a final AIMessage."
                    logging.warning(f"Unexpected LangGraph response structure: {full_graph_response}")

                # Check for tool usage that modified the calendar
                for msg in final_messages:
                    # AIMessage with tool_calls means the LLM decided to call a tool
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            # tc is a dict: {'name': 'tool_name', 'args': {...}, 'id': '...'}
                            if tc.get('name') in ["add_calendar_event", "delete_calendar_event", "update_calendar_event"]: # Add other relevant tool names
                                calendar_potentially_modified = True
                                logging.info(f"Calendar modifying tool '{tc.get('name')}' detected in agent response.")
                                break
                    if calendar_potentially_modified:
                        break
                # The ConversationBufferWindowMemory (`memory`) is not updated here, as LangGraph uses checkpointer.

            else: # Direct LLM call (if tools disabled or LangGraph agent failed to init)
                logging.info(f"Invoking LLM directly for input: {user_input} (Tools disabled: {not use_agent}, Agent not init: {not langgraph_agent_app})")
                current_history = memory.load_memory_variables({})['chat_history']
                # Use a simple system message for direct calls
                direct_system_msg = SystemMessage(content=f"You are '{HOST_NAME}', a helpful assistant for '{USER_NAME}'. The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Please respond directly to the user.")
                messages = [direct_system_msg] + current_history + [HumanMessage(content=user_input)]
                
                llm_response = llm.invoke(messages)
                response_text = llm_response.content
                # Save to the separate ConversationBufferWindowMemory
                memory.save_context({"input": user_input}, {"output": response_text})
        except Exception as e:
            logging.error(f"Error processing input: {e}", exc_info=True)
            response_text = f"Sorry, an error occurred: {e}"
            error_occurred = True
            traceback.print_exc() # For console debugging
        
        # Schedule UI update on the main thread
        self.root.after(0, self.display_response_and_refresh, response_text, error_occurred, calendar_potentially_modified)

    def display_response_and_refresh(self, response_text, error_occurred, calendar_modified):
        self.add_message(HOST_NAME, response_text, tag='error' if error_occurred else 'assistant')
        self.set_status("Ready.")
        self.send_button.config(state='normal')
        self.input_entry.config(state='normal')

        if calendar_modified:
            logging.info("Calendar potentially modified by agent. Reloading data from DB and refreshing UI.")
            self._load_calendar_events_from_db() # Reload global calendar_events from DB
            self.refresh_calendar_display()      # Update UI from (now refreshed) global calendar_events

    def clear_chat(self):
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the chat history?"):
            self.chat_display.config(state='normal')
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state='disabled')
            self.chat_history.clear() # UI history
            
            # Clear LangChain ConversationBufferWindowMemory (for direct LLM calls & reminders)
            memory.clear() 
            
            # Clear LangGraph agent's memory from the checkpointer
            if checkpointer and isinstance(checkpointer, MemorySaver):
                thread_config = {"configurable": {"thread_id": LANGGRAPH_THREAD_ID}}
                # For MemorySaver, we can try to remove the thread_id entry from its internal storage
                if LANGGRAPH_THREAD_ID in checkpointer.storage:
                    try:
                        del checkpointer.storage[LANGGRAPH_THREAD_ID]
                        logging.info(f"Cleared LangGraph agent state for thread '{LANGGRAPH_THREAD_ID}' from MemorySaver.")
                    except Exception as e:
                        logging.error(f"Could not directly delete thread '{LANGGRAPH_THREAD_ID}' from MemorySaver: {e}")
                else:
                    logging.info(f"No state found for thread '{LANGGRAPH_THREAD_ID}' in MemorySaver to clear.")
            elif checkpointer:
                 logging.warning("Checkpointer is not MemorySaver, manual clearing of thread state might be needed or not supported directly.")


            self.add_message("System", "Chat history cleared.", tag='system')
            self.set_status("Chat cleared.")
            logging.info("Chat history cleared (UI, buffer memory, and attempted LangGraph agent state).")


    def save_chat_history(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Chat History As..."
        )
        if not file_path: return
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                history_text = self.chat_display.get(1.0, tk.END)
                f.write(history_text)
            self.set_status(f"Chat history saved to {os.path.basename(file_path)}")
            logging.info(f"Chat history saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save chat history.\nError: {e}")
            logging.error(f"Failed to save chat history: {e}", exc_info=True)


    def background_calendar_checker(self):
        logging.info("Background calendar checker started.")
        time.sleep(5) # Initial delay
        try:
            reminder_llm = ChatOpenAI(
                base_url=LOCAL_API_BASE, model=LOCAL_MODEL_NAME, api_key=DUMMY_API_KEY, temperature=0.1, streaming=False
            )
        except Exception as e:
            logging.error(f"Failed to initialize reminder_llm in background thread: {e}", exc_info=True)
            # Potentially post a message to UI or stop thread
            self.message_queue.put({"sender": "System", "message": "Error: Calendar reminder LLM failed to init.", "tag": "error"})
            return

        while not self.calendar_thread_stop_event.is_set():
            try:
                now = datetime.now()
                reminder_time_limit = now + timedelta(minutes=CALENDAR_REMINDER_MINUTES)
                events_to_remind = []
                
                with calendar_lock: 
                    current_calendar_copy = list(calendar_events) 
                
                for event in current_calendar_copy:
                    if now <= event['datetime'] <= reminder_time_limit and event['id'] not in self.reminded_event_ids:
                        events_to_remind.append(event)
                        # Mark as reminded immediately to avoid race conditions if generation is slow
                        self.reminded_event_ids.add(event['id']) 

                for event in events_to_remind:
                    try:
                        logging.info(f"Generating reminder for event ID {event['id']}: {event['content']}")
                        reminder_prompt = (
                            f"You are '{HOST_NAME}', '{USER_NAME}'s assistant. Generate a brief, friendly reminder "
                            f"for the user about their upcoming event: '{event['content']}' scheduled for "
                            f"{event['datetime'].strftime('%I:%M %p on %A, %B %d')}. "
                            f"The current time is {now.strftime('%I:%M %p')}. Keep it concise and natural."
                        )
                        messages = [SystemMessage(content=f"You are {HOST_NAME}."), HumanMessage(content=reminder_prompt)]
                        response = reminder_llm.invoke(messages)
                        reminder_message = response.content.strip()
                        self.message_queue.put({"sender": HOST_NAME, "message": reminder_message, "tag": "assistant", "event_id": event['id']})
                        logging.info(f"Reminder generated and queued for event ID {event['id']}")
                    except Exception as e:
                        logging.error(f"Error generating reminder for event ID {event['id']}: {e}", exc_info=True)
                        # If reminder failed, potentially allow it to be reminded again later by removing from reminded_event_ids
                        # self.reminded_event_ids.discard(event['id']) # Or handle based on error type
            except Exception as e:
                logging.error(f"Error in background_calendar_checker loop: {e}", exc_info=True)
            
            # Wait for the specified interval or until the stop event is set
            self.calendar_thread_stop_event.wait(CALENDAR_CHECK_INTERVAL_SECONDS)
        logging.info("Background calendar checker stopped.")

    def check_message_queue(self):
        while not self.message_queue.empty():
            try:
                msg_data = self.message_queue.get_nowait()
                self.add_message(msg_data.get("sender", "System"), msg_data.get("message", ""), tag=msg_data.get("tag", "system"))
                # If the reminder is from the assistant, add it to the (direct LLM) memory for context if tools are then disabled.
                # Note: This does NOT add it to the LangGraph agent's checkpointer memory.
                if msg_data.get("tag") == 'assistant' and msg_data.get("sender") == HOST_NAME:
                    memory.save_context({"input": f"[Reminder Sent for Event ID {msg_data.get('event_id', 'N/A')}: {msg_data.get('message')}]"}, {"output": ""}) # Log reminder as user input, assistant output is implicit
            except Exception as e:
                logging.error(f"Error processing message queue: {e}", exc_info=True)
        self.root.after(100, self.check_message_queue)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.set_status("Shutting down...")
            logging.info("Shutdown requested.")
            self.calendar_thread_stop_event.set()
            # Join can be added here if essential, but daemon thread will exit with main.
            # If self.calendar_thread.is_alive():
            #    self.calendar_thread.join(timeout=2.0) 
            save_vector_store()
            self.root.destroy()
            logging.info("Application closed.")
            
    def manual_refresh_calendar(self):
        logging.info("Manual calendar refresh triggered.")
        self.set_status("Refreshing calendar...")
        self._load_calendar_events_from_db()
        self.refresh_calendar_display()
        self.set_status("Calendar refreshed.")

    def refresh_calendar_display(self):
        self.calendar_display.config(state='normal')
        self.calendar_display.delete(1.0, tk.END)
        now = datetime.now()
        # Display events for the next 7 days in the UI
        end_time_display_limit = now + timedelta(days=7) 
        
        upcoming_display_events_ui = []
        with calendar_lock: 
            # Filter and sort directly from the already sorted calendar_events list for UI
            # Ensure event['datetime'] is actually a datetime object
            for event in calendar_events:
                if isinstance(event['datetime'], datetime):
                    if now <= event['datetime'] <= end_time_display_limit:
                        upcoming_display_events_ui.append(event)
                else:
                    logging.warning(f"Event ID {event.get('id')} has non-datetime object: {event['datetime']}")
        
        # Sort again just to be absolutely sure for display, if needed (should be sorted from DB load)
        # upcoming_display_events_ui.sort(key=lambda x: x['datetime'])


        if not upcoming_display_events_ui:
            self.calendar_display.insert(tk.END, "No upcoming events in the next 7 days.")
        else:
            for event in upcoming_display_events_ui:
                event_time_str = event['datetime'].strftime('%a, %b %d at %I:%M %p')
                self.calendar_display.insert(tk.END, f"• {event['content']}\n   {event_time_str}\n\n")
        self.calendar_display.config(state='disabled')

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualAssistantApp(root)
    root.mainloop()