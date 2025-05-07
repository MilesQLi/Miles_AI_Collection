# Required Installations:
# pip install langchain langchain_community langchain_openai faiss-cpu ollama duckduckgo-search python-dotenv pydantic==1.10.11 dateparser tk requests beautifulsoup4 PyYAML

import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog, filedialog, Checkbutton, BooleanVar
import threading
import time
from datetime import datetime, timedelta
import dateparser # More robust parsing than dateutil.parser
import json
import os
import yaml
import logging
import traceback
from queue import Queue # For thread-safe communication
import sqlite3 # For persistent calendar

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
        stream=False,
        temperature=0.2,
        streaming=False,
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
    _ = embeddings.embed_query("Test query")
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

# 4. Memory
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
    event_datetime_str: str = Field(..., description="The date and time of the event (e.g., 'tomorrow at 3 pm').")

@tool("add_calendar_event", args_schema=CalendarEventInput)
def add_calendar_event(event_content: str, event_datetime_str: str) -> str:
    """Adds an event to the user's calendar. Stores persistently."""
    try:
        event_dt = dateparser.parse(event_datetime_str, settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False})
        if not event_dt:
            return f"Error: Could not understand the date/time '{event_datetime_str}'. Please specify a clearer date and time."

        conn = _get_db_connection()
        cursor = conn.cursor()
        event_dt_iso = event_dt.isoformat()
        cursor.execute("INSERT INTO events (content, event_datetime) VALUES (?, ?)", (event_content, event_dt_iso))
        new_event_id = cursor.lastrowid
        conn.commit()
        logging.info(f"DB: Added calendar event ID {new_event_id}: '{event_content}' at {event_dt_iso}")
        return f"OK. I've added '{event_content}' to your calendar for {event_dt.strftime('%A, %B %d, %Y at %I:%M %p')} (ID: {new_event_id}). Your calendar view should update shortly."
    except Exception as e:
        logging.error(f"Error in add_calendar_event tool: {e}", exc_info=True)
        return f"Error: Could not add calendar event to database. Details: {e}"
    finally:
        if 'conn' in locals() and conn:
            conn.close()


class CalendarRetrieveInput(BaseModel):
    start_datetime_str: str = Field(..., description="The start date/time for retrieving events (e.g., 'today').")
    end_datetime_str: str = Field(..., description="The end date/time for retrieving events (e.g., 'end of today').")

@tool("retrieve_calendar_events", args_schema=CalendarRetrieveInput)
def retrieve_calendar_events(start_datetime_str: str, end_datetime_str: str) -> str:
    """Retrieves events from the user's calendar within a specified date/time range from persistent storage."""
    try:
        start_dt = dateparser.parse(start_datetime_str, settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False})
        end_dt = dateparser.parse(end_datetime_str, settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False})

        if not start_dt or not end_dt:
            return "Error: Could not understand the date range."
        if start_dt > end_dt:
            end_dt = start_dt.replace(hour=23, minute=59, second=59) # Default to end of start day

        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content, event_datetime FROM events WHERE event_datetime BETWEEN ? AND ? ORDER BY event_datetime ASC",
            (start_dt.isoformat(), end_dt.isoformat())
        )
        db_events = cursor.fetchall()
        conn.close()

        found_events = [
            f"- {row[0]} on {datetime.fromisoformat(row[1]).strftime('%A, %B %d at %I:%M %p')}"
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

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", f"You are '{HOST_NAME}', a helpful personal assistant for '{USER_NAME}'. The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Use tools when necessary."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

try:
    agent = create_openai_tools_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=memory, verbose=True,
        handle_parsing_errors=True, max_iterations=10,
    )
    logging.info("Agent Executor created successfully.")
except Exception as e:
    logging.error(f"Failed to create Agent Executor: {e}", exc_info=True)
    messagebox.showerror("Agent Error", f"Could not create LangChain agent.\nError: {e}")
    agent_executor = None


# --- Tkinter Application ---
class VirtualAssistantApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title(f"{USER_NAME}'s Personal Assistant ({HOST_NAME})")
        self.root.geometry("1024x600")

        self.chat_history = []
        self.reminded_event_ids = set()
        self.message_queue = Queue()

        # Menu Bar
        self.menu_bar = tk.Menu(self.root)
        # ... (menu setup - unchanged) ...
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
        
        self.calendar_header = tk.Label(self.calendar_frame, text="Upcoming Events", font=('Arial', 10, 'bold'))
        self.calendar_header.pack(pady=5)
        self.calendar_display = scrolledtext.ScrolledText(self.calendar_frame, wrap=tk.WORD, width=35, state='disabled') # Adjusted width
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
        init_calendar_db() # Ensure DB and table exist
        load_vector_store()
        self._load_calendar_events_from_db() # Load calendar from DB into memory

        self.add_message("System", f"Assistant initialized. Type 'clear' to reset chat. Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n", tag='system')
        
        self.calendar_thread_stop_event = threading.Event()
        self.calendar_thread = threading.Thread(target=self.background_calendar_checker, daemon=True)
        self.calendar_thread.start()
        self.refresh_calendar_display() # Initial display

        self.root.after(100, self.check_message_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _load_calendar_events_from_db(self):
        global calendar_events, next_event_id # next_event_id is less critical now
        max_id_found = 0
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, event_datetime FROM events ORDER BY event_datetime ASC")
            db_rows = cursor.fetchall()
            conn.close()

            with calendar_lock:
                calendar_events.clear() # Clear existing in-memory events
                for row in db_rows:
                    event_id, content, dt_str = row
                    calendar_events.append({
                        'id': event_id,
                        'content': content,
                        'datetime': datetime.fromisoformat(dt_str)
                    })
                    if event_id > max_id_found:
                        max_id_found = event_id
                next_event_id = max_id_found + 1
            logging.info(f"Loaded {len(calendar_events)} events from DB into memory. Next event ID calculated as {next_event_id}.")
        except Exception as e:
            logging.error(f"Error loading calendar events from DB: {e}", exc_info=True)
            messagebox.showerror("Calendar DB Error", f"Could not load calendar events from database.\nError: {e}")

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
            if use_agent and agent_executor:
                logging.info(f"Invoking Agent for input: {user_input}")
                response = agent_executor.invoke({"input": user_input})
                response_text = response.get('output', "Agent did not provide an output.")
                # Check if calendar modifying tools were used
                if response.get("intermediate_steps"):
                    for step in response["intermediate_steps"]:
                        action = step[0] # AIMessageChunk or similar, depending on agent
                        if hasattr(action, 'tool') and action.tool in ["add_calendar_event", "delete_calendar_event"]: # Add other modifying tools here
                             calendar_potentially_modified = True
                             break
                        # Fallback for some agent types if tool name is in log_to_str output
                        if "tool='add_calendar_event'" in str(action) or "tool='delete_calendar_event'" in str(action):
                            calendar_potentially_modified = True
                            break


            else: # Direct LLM call
                logging.info(f"Invoking LLM directly for input: {user_input}")
                current_history = memory.load_memory_variables({})['chat_history']
                messages = [SystemMessage(content=f"You are '{HOST_NAME}', a helpful assistant.")] + current_history + [HumanMessage(content=user_input)]
                llm_response = llm.invoke(messages)
                response_text = llm_response.content
                memory.save_context({"input": user_input}, {"output": response_text})
        except Exception as e:
            logging.error(f"Error processing input: {e}", exc_info=True)
            response_text = f"Sorry, an error occurred: {e}"
            error_occurred = True
            traceback.print_exc()
        
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
            # ... (clear chat UI and memory - unchanged) ...
            self.chat_display.config(state='normal')
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state='disabled')
            self.chat_history.clear()
            memory.clear() 
            self.add_message("System", "Chat history cleared.", tag='system')
            self.set_status("Chat cleared.")
            logging.info("Chat history cleared.")


    def save_chat_history(self):
        # ... (save chat history - unchanged) ...
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
        time.sleep(5) 
        reminder_llm = ChatOpenAI(
            base_url=LOCAL_API_BASE, model=LOCAL_MODEL_NAME, api_key=DUMMY_API_KEY, temperature=0.1
        )
        while not self.calendar_thread_stop_event.is_set():
            try:
                now = datetime.now()
                reminder_time_limit = now + timedelta(minutes=CALENDAR_REMINDER_MINUTES)
                events_to_remind = []
                
                with calendar_lock: # Protects access to shared calendar_events list
                    # Iterate over a copy for safety if modifications were possible (not in this loop)
                    current_calendar_copy = list(calendar_events) 
                
                for event in current_calendar_copy:
                    if now <= event['datetime'] <= reminder_time_limit and event['id'] not in self.reminded_event_ids:
                        events_to_remind.append(event)
                        self.reminded_event_ids.add(event['id'])

                for event in events_to_remind:
                    try:
                        logging.info(f"Generating reminder for event ID {event['id']}: {event['content']}")
                        reminder_prompt = (
                            f"You are '{HOST_NAME}', '{USER_NAME}'s assistant. Generate a brief, friendly reminder "
                            f"for the user about their upcoming event: '{event['content']}' scheduled for "
                            f"{event['datetime'].strftime('%I:%M %p on %A, %B %d')}. "
                            f"Current time: {now.strftime('%I:%M %p')}. Keep it concise."
                        )
                        messages = [SystemMessage(content=f"You are {HOST_NAME}."), HumanMessage(content=reminder_prompt)]
                        response = reminder_llm.invoke(messages)
                        reminder_message = response.content.strip()
                        self.message_queue.put({"sender": HOST_NAME, "message": reminder_message, "tag": "assistant", "event_id": event['id']})
                        logging.info(f"Reminder generated and queued for event ID {event['id']}")
                    except Exception as e:
                        logging.error(f"Error generating reminder for event ID {event['id']}: {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Error in background_calendar_checker loop: {e}", exc_info=True)
            self.calendar_thread_stop_event.wait(CALENDAR_CHECK_INTERVAL_SECONDS)
        logging.info("Background calendar checker stopped.")

    def check_message_queue(self):
        while not self.message_queue.empty():
            try:
                msg_data = self.message_queue.get_nowait()
                self.add_message(msg_data.get("sender", "System"), msg_data.get("message", ""), tag=msg_data.get("tag", "system"))
                if msg_data.get("tag") == 'assistant' and msg_data.get("sender") == HOST_NAME:
                    memory.save_context({"input": f"[Reminder Sent for Event ID {msg_data.get('event_id', 'N/A')}]"}, {"output": msg_data.get("message")})
            except Exception as e:
                logging.error(f"Error processing message queue: {e}", exc_info=True)
        self.root.after(100, self.check_message_queue)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.set_status("Shutting down...")
            logging.info("Shutdown requested.")
            self.calendar_thread_stop_event.set()
            # self.calendar_thread.join(timeout=1.0) # Optional: wait for thread
            save_vector_store()
            self.root.destroy()
            logging.info("Application closed.")
            
    def manual_refresh_calendar(self):
        """Manually triggers a reload from DB and UI refresh for the calendar."""
        logging.info("Manual calendar refresh triggered.")
        self.set_status("Refreshing calendar...")
        self._load_calendar_events_from_db()
        self.refresh_calendar_display()
        self.set_status("Calendar refreshed.")

    def refresh_calendar_display(self):
        self.calendar_display.config(state='normal')
        self.calendar_display.delete(1.0, tk.END)
        now = datetime.now()
        end_time_display_limit = now + timedelta(days=7) # Show events for next 7 days in UI
        
        # The calendar_events list is now managed by _load_calendar_events_from_db
        with calendar_lock: # Ensure thread-safe access if other threads might modify it (though unlikely here)
            # Filter and sort directly from the already sorted calendar_events list
            upcoming_display_events = [
                event for event in calendar_events 
                if now <= event['datetime'] <= end_time_display_limit
            ]
            # calendar_events is already sorted by datetime from _load_calendar_events_from_db

        if not upcoming_display_events:
            self.calendar_display.insert(tk.END, "No upcoming events in the next 7 days.")
        else:
            for event in upcoming_display_events:
                event_time_str = event['datetime'].strftime('%a, %b %d at %I:%M %p') # Shortened format for UI
                self.calendar_display.insert(tk.END, f"• {event['content']}\n   {event_time_str}\n\n")
        self.calendar_display.config(state='disabled')

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualAssistantApp(root)
    root.mainloop()