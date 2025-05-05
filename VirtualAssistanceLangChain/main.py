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

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent, tool
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
LOCAL_API_BASE = "http://localhost:11434/v1" # Standard Ollama API endpoint
# LOCAL_API_BASE = "http://host.docker.internal:11434/v1" # Use if running Ollama in Docker Desktop
LOCAL_MODEL_NAME = "qwen2.5:14b-instruct-q6_K" # CHANGE TO YOUR AVAILABLE OLLAMA MODEL
DUMMY_API_KEY = "ollama" # Placeholder API key for Ollama

# Embedding Configuration
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Common Ollama embedding model
OLLAMA_BASE_URL = LOCAL_API_BASE.replace("/v1", "") # Ollama embeddings often use the base URL

# RAG / Vector Store Configuration
FAISS_INDEX_PATH = "faiss_index_assistant"

# Calendar Configuration
CALENDAR_REMINDER_MINUTES = 15
CALENDAR_CHECK_INTERVAL_SECONDS = 60 # Check every 60 seconds

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LangChain Setup ---

# 1. LLM
try:
    llm = ChatOpenAI(
        base_url=LOCAL_API_BASE,
        model=LOCAL_MODEL_NAME,
        api_key=DUMMY_API_KEY,
        temperature=0.2, # Lower temperature for more factual/tool-based answers
        streaming=False, # Streaming complicates agent execution logic in Tkinter
        # request_timeout=60, # Increase if needed
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
        base_url=OLLAMA_BASE_URL # Use the modified base URL
    )
    # Test embedding
    _ = embeddings.embed_query("Test query")
    logging.info(f"Embeddings initialized: Model={EMBEDDING_MODEL_NAME}, BaseURL={OLLAMA_BASE_URL}")
except Exception as e:
    logging.error(f"Failed to initialize Embeddings: {e}", exc_info=True)
    messagebox.showerror("Embedding Error", f"Could not initialize Ollama embeddings with model '{EMBEDDING_MODEL_NAME}' at {OLLAMA_BASE_URL}.\nError: {e}")
    exit()

# 3. Vector Store (FAISS) and In-Memory Calendar
knowledge_base_texts = []
knowledge_base_metadatas = []
calendar_events = [] # List to store {'id': int, 'content': str, 'datetime': datetime}
calendar_lock = threading.Lock()
vector_store = None
next_event_id = 1

def load_vector_store():
    global vector_store, knowledge_base_texts, knowledge_base_metadatas
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            # Note: Loading doesn't automatically repopulate the text/metadata lists.
            # This simple implementation assumes we primarily interact via the vector store interface.
            # For perfect state restoration, texts/metadata would need separate persistence.
            logging.info(f"Loaded FAISS index from {FAISS_INDEX_PATH}")
        except Exception as e:
            logging.error(f"Failed to load FAISS index: {e}. Creating a new one.", exc_info=True)
            vector_store = FAISS.from_texts(["Initial knowledge base entry: Assistant started."], embedding=embeddings, metadatas=[{"timestamp": datetime.now().isoformat()}])
    else:
        logging.info("No FAISS index found. Creating a new one.")
        # Initialize with a dummy entry to avoid issues with empty stores
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
    k=5, # Remember last 5 turns
    memory_key="chat_history",
    return_messages=True # Important for agents
)

# --- Custom Tools ---

# Tool for saving notes to FAISS
@tool
def save_note_to_vector_store(note: str) -> str:
    """
    Saves a piece of information or a note provided by the user to the knowledge base
    for later retrieval. Use this when the user provides important information to remember
    or asks to save something. Input should be the text content of the note.
    """
    global vector_store
    if not note or not isinstance(note, str):
        return "Error: Invalid note content provided. Please provide text to save."
    try:
        metadata = {"timestamp": datetime.now().isoformat(), "source": "user_note"}
        vector_store.add_texts([note], metadatas=[metadata])
        save_vector_store() # Save after adding
        logging.info(f"Saved note to vector store: {note[:50]}...")
        return f"Note saved successfully: '{note[:50]}...'"
    except Exception as e:
        logging.error(f"Error saving note to vector store: {e}", exc_info=True)
        return f"Error: Could not save note. Details: {e}"

# Tool for retrieving notes from FAISS
@tool
def retrieve_notes_from_vector_store(query: str) -> str:
    """
    Retrieves relevant information or notes from the knowledge base based on a user's query.
    Use this when the user asks about past information, saved notes, or refers to something
    that might have been discussed and saved earlier. Input should be the user's query about the information needed.
    """
    global vector_store
    if not query or not isinstance(query, str):
        return "Error: Invalid query provided. Please provide text to search for."
    try:
        results = vector_store.similarity_search_with_score(query, k=3)
        if not results:
            return "No relevant notes found in the knowledge base for your query."

        formatted_results = []
        for doc, score in results:
            timestamp = doc.metadata.get('timestamp', 'N/A')
            # Format timestamp if possible
            try:
                ts_dt = datetime.fromisoformat(timestamp)
                formatted_ts = ts_dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_ts = timestamp # Keep original if parsing fails
            formatted_results.append(f"- Note (Saved {formatted_ts}, Relevance: {score:.2f}): {doc.page_content}")

        logging.info(f"Retrieved notes for query: {query[:50]}...")
        return "Found the following relevant notes:\n" + "\n".join(formatted_results)
    except Exception as e:
        logging.error(f"Error retrieving notes from vector store: {e}", exc_info=True)
        return f"Error: Could not retrieve notes. Details: {e}"

# Pydantic model for Calendar Event Input
class CalendarEventInput(BaseModel):
    event_content: str = Field(..., description="The description or title of the calendar event.")
    event_datetime_str: str = Field(..., description="The date and time of the event in a natural language format (e.g., 'tomorrow at 3 pm', 'next Friday noon', '2024-08-15 10:00').")

# Tool for adding calendar events
@tool("add_calendar_event", args_schema=CalendarEventInput)
def add_calendar_event(event_content: str, event_datetime_str: str) -> str:
    """
    Adds an event to the user's calendar. Use this when the user wants to schedule
    something, set a reminder, or add an appointment. Requires the event description
    and the date/time.
    """
    global calendar_events, next_event_id
    try:
        # Use dateparser for robust parsing
        event_dt = dateparser.parse(event_datetime_str, settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False})
        if not event_dt:
            return f"Error: Could not understand the date/time '{event_datetime_str}'. Please specify a clearer date and time."

        with calendar_lock:
            event_id = next_event_id
            calendar_events.append({
                'id': event_id,
                'content': event_content,
                'datetime': event_dt
            })
            next_event_id += 1
            # Sort events after adding
            calendar_events.sort(key=lambda x: x['datetime'])

        logging.info(f"Added calendar event: '{event_content}' at {event_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        return f"OK. I've added '{event_content}' to your calendar for {event_dt.strftime('%A, %B %d, %Y at %I:%M %p')}."
    except Exception as e:
        logging.error(f"Error adding calendar event: {e}", exc_info=True)
        return f"Error: Could not add calendar event. Details: {e}"

# Pydantic model for Calendar Retrieval Input
class CalendarRetrieveInput(BaseModel):
    start_datetime_str: str = Field(..., description="The start date/time for retrieving events (e.g., 'today', 'next week', 'August 1st').")
    end_datetime_str: str = Field(..., description="The end date/time for retrieving events (e.g., 'today evening', 'end of next week', 'August 5th').")

# Tool for retrieving calendar events
@tool("retrieve_calendar_events", args_schema=CalendarRetrieveInput)
def retrieve_calendar_events(start_datetime_str: str, end_datetime_str: str) -> str:
    """
    Retrieves events from the user's calendar within a specified date/time range.
    Use this when the user asks about their schedule, upcoming events, or what they have planned.
    """
    global calendar_events
    try:
        start_dt = dateparser.parse(start_datetime_str, settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False})
        end_dt = dateparser.parse(end_datetime_str, settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False})

        if not start_dt or not end_dt:
            return "Error: Could not understand the date range. Please specify clearer start and end times."
        if start_dt > end_dt:
             # Sensible default: If end is before start, assume end is end of start day
             end_dt = start_dt.replace(hour=23, minute=59, second=59)
             # return "Error: Start date/time must be before the end date/time."

        found_events = []
        with calendar_lock:
            for event in calendar_events:
                if start_dt <= event['datetime'] <= end_dt:
                    found_events.append(f"- {event['content']} on {event['datetime'].strftime('%A, %B %d at %I:%M %p')}")

        logging.info(f"Retrieved calendar events between {start_dt.strftime('%Y-%m-%d %H:%M')} and {end_dt.strftime('%Y-%m-%d %H:%M')}")

        if not found_events:
            return f"You have no events scheduled between {start_dt.strftime('%B %d')} and {end_dt.strftime('%B %d')}."
        else:
            return f"Here are your scheduled events between {start_dt.strftime('%B %d')} and {end_dt.strftime('%B %d')}:\n" + "\n".join(found_events)

    except Exception as e:
        logging.error(f"Error retrieving calendar events: {e}", exc_info=True)
        return f"Error: Could not retrieve calendar events. Details: {e}"

# --- Agent Setup ---
tools = [
    DuckDuckGoSearchRun(),
    PythonREPLTool(),
    save_note_to_vector_store,
    retrieve_notes_from_vector_store,
    add_calendar_event,
    retrieve_calendar_events,
]

# System prompt defining the agent's behavior and tool usage
# Note: Adjust persona details as needed
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are '{HOST_NAME}', a helpful and friendly personal assistant for '{USER_NAME}'. "
            f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
            "You have access to the following tools. Use them when necessary to answer the user's questions or fulfill their requests accurately and efficiently. "
            "Think step-by-step if needed before responding. If you use a tool, briefly mention it in your response (e.g., 'According to a web search...', 'I've saved that note for you.', 'Looking at your calendar...'). "
            "Prioritize using the calendar and notes tools for relevant requests before resorting to web search or code execution unless specifically asked. "
            "If asked to save information, use the 'save_note_to_vector_store' tool. "
            "If asked about past information or notes, use the 'retrieve_notes_from_vector_store' tool. "
            "If asked to add an event or reminder, use the 'add_calendar_event' tool. "
            "If asked about the schedule or events, use the 'retrieve_calendar_events' tool. "
            "For calculations or coding tasks, use the Python REPL tool. "
            "For general knowledge or current events beyond your internal knowledge or saved notes, use the search engine."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's intermediate steps
    ]
)

try:
    agent = create_openai_functions_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True, # Set to False for less console output
        handle_parsing_errors=True, # Try to gracefully handle LLM output errors
        max_iterations=5, # Prevent runaway agents
    )
    logging.info("Agent Executor created successfully.")
except Exception as e:
    logging.error(f"Failed to create Agent Executor: {e}", exc_info=True)
    messagebox.showerror("Agent Error", f"Could not create the LangChain agent.\nError: {e}")
    agent_executor = None # Disable agent if creation fails


# --- Tkinter Application ---
class VirtualAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{USER_NAME}'s Personal Assistant ({HOST_NAME})")
        # Increase initial size
        self.root.geometry("800x600")

        # --- Data ---
        self.chat_history = [] # For display purposes
        self.reminded_event_ids = set()
        self.message_queue = Queue() # Thread-safe queue for messages from background thread

        # --- UI Elements ---
        # Menu Bar
        self.menu_bar = tk.Menu(root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Clear Chat", command=self.clear_chat)
        self.file_menu.add_command(label="Save Chat History", command=self.save_chat_history)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_closing)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.root.config(menu=self.menu_bar)

        # Top Frame for Controls
        self.top_frame = tk.Frame(root, pady=5)
        self.top_frame.pack(fill=tk.X)

        # Tool Usage Checkbox
        self.use_tools_var = BooleanVar(value=True) # Default to using tools
        self.tools_checkbox = Checkbutton(self.top_frame, text="Enable Tools (Agent)", variable=self.use_tools_var)
        self.tools_checkbox.pack(side=tk.LEFT, padx=10)


        # Main Frame for Chat and Input
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Chat Display Area
        self.chat_display = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, state='disabled', height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        # Configure tags for styling
        self.chat_display.tag_configure('user', foreground='blue', font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure('assistant', foreground='green', font=('Arial', 10))
        self.chat_display.tag_configure('system', foreground='gray', font=('Arial', 9, 'italic'))
        self.chat_display.tag_configure('error', foreground='red', font=('Arial', 10, 'bold'))

        # Input Area
        self.input_frame = tk.Frame(self.main_frame)
        self.input_frame.pack(fill=tk.X)

        self.input_entry = tk.Entry(self.input_frame, font=('Arial', 11))
        self.input_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 10))
        self.input_entry.bind("<Return>", self.send_message_event)

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message_event, width=10)
        self.send_button.pack(side=tk.RIGHT)

        # Status Bar (Optional)
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.set_status("Ready.")

        # --- Initialization ---
        load_vector_store() # Load existing index if present
        self.add_message("System", f"Assistant initialized. Type 'clear' to reset chat. Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}", tag='system')
        # Load initial memory (optional, could load from saved state)
        # memory.chat_memory.add_message(SystemMessage(content=agent_prompt.messages[0].prompt.template))

        # Start background calendar checker
        self.calendar_thread_stop_event = threading.Event()
        self.calendar_thread = threading.Thread(target=self.background_calendar_checker, daemon=True)
        self.calendar_thread.start()

        # Start queue checker
        self.root.after(100, self.check_message_queue)

        # Handle closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def add_message(self, sender, message, tag='assistant'):
        """Adds a message to the chat display."""
        self.chat_display.config(state='normal')
        if self.chat_history: # Add newline if not the first message
             self.chat_display.insert(tk.END, "\n")
        self.chat_display.insert(tk.END, f"{sender}: ", ('user' if tag == 'user' else tag,))
        self.chat_display.insert(tk.END, message, (tag,))
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END) # Auto-scroll
        if tag != 'system': # Don't add system status messages to history list
             self.chat_history.append(f"{sender}: {message}")

    def send_message_event(self, event=None):
        """Handles sending a message from the input entry."""
        user_input = self.input_entry.get().strip()
        if not user_input:
            return

        self.input_entry.delete(0, tk.END)
        self.add_message(USER_NAME, user_input, tag='user')

        # Special command: clear chat
        if user_input.lower() == 'clear':
            self.clear_chat()
            return

        self.set_status("Thinking...")
        self.send_button.config(state='disabled')
        self.input_entry.config(state='disabled')

        # Run LLM/Agent interaction in a separate thread to avoid blocking UI
        thread = threading.Thread(target=self.process_user_input, args=(user_input,))
        thread.start()

    def process_user_input(self, user_input):
        """Processes user input using LLM or Agent in a background thread."""
        response_text = ""
        error_occurred = False
        try:
            use_agent = self.use_tools_var.get()

            if use_agent and agent_executor:
                # Use the agent executor
                logging.info(f"Invoking Agent for input: {user_input}")
                # The agent executor automatically uses and updates the memory
                response = agent_executor.invoke({"input": user_input})
                response_text = response.get('output', "Agent did not provide an output.")
            else:
                # Direct LLM call (without tools, but with memory)
                logging.info(f"Invoking LLM directly for input: {user_input}")
                current_history = memory.load_memory_variables({})['chat_history']
                messages = [SystemMessage(content=f"You are '{HOST_NAME}', a helpful assistant for '{USER_NAME}'. Keep your answers concise and friendly.")] + current_history + [HumanMessage(content=user_input)]
                response = llm.invoke(messages)
                response_text = response.content
                # Manually update memory for direct calls
                memory.save_context({"input": user_input}, {"output": response_text})

        except Exception as e:
            logging.error(f"Error processing input: {e}", exc_info=True)
            response_text = f"Sorry, an error occurred: {e}"
            error_occurred = True
            traceback.print_exc() # Print full traceback to console

        # Schedule UI update back on the main thread
        self.root.after(0, self.display_response, response_text, error_occurred)


    def display_response(self, response_text, error_occurred):
        """Updates the UI with the assistant's response."""
        self.add_message(HOST_NAME, response_text, tag='error' if error_occurred else 'assistant')
        self.set_status("Ready.")
        self.send_button.config(state='normal')
        self.input_entry.config(state='normal')


    def clear_chat(self):
        """Clears the chat display and memory."""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the chat history?"):
            self.chat_display.config(state='normal')
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state='disabled')
            self.chat_history.clear()
            memory.clear() # Clear LangChain memory
            self.add_message("System", "Chat history cleared.", tag='system')
            self.set_status("Chat cleared.")
            logging.info("Chat history cleared.")


    def save_chat_history(self):
        """Saves the displayed chat history to a text file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Chat History As..."
        )
        if not file_path:
            return # User cancelled

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Get text directly from the widget to preserve formatting
                history_text = self.chat_display.get(1.0, tk.END)
                f.write(history_text)
            self.set_status(f"Chat history saved to {os.path.basename(file_path)}")
            logging.info(f"Chat history saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save chat history.\nError: {e}")
            logging.error(f"Failed to save chat history: {e}", exc_info=True)

    def background_calendar_checker(self):
        """Periodically checks for upcoming calendar events."""
        logging.info("Background calendar checker started.")
        # Give LLM time to initialize if needed
        time.sleep(5)

        # Create a separate LLM instance for this thread? Maybe not needed if stateless
        # Or ensure thread-safe usage if shared
        reminder_llm = ChatOpenAI(
            base_url=LOCAL_API_BASE, model=LOCAL_MODEL_NAME, api_key=DUMMY_API_KEY, temperature=0.1
        )

        while not self.calendar_thread_stop_event.is_set():
            try:
                now = datetime.now()
                reminder_time_limit = now + timedelta(minutes=CALENDAR_REMINDER_MINUTES)
                events_to_remind = []

                with calendar_lock:
                    # Iterate safely over a copy or manage indices carefully if modifying
                    current_calendar = list(calendar_events)
                    for event in current_calendar:
                        # Check if event is upcoming, within the window, and not already reminded
                        if now <= event['datetime'] <= reminder_time_limit and event['id'] not in self.reminded_event_ids:
                            events_to_remind.append(event)
                            self.reminded_event_ids.add(event['id']) # Mark as reminded immediately

                for event in events_to_remind:
                    try:
                        logging.info(f"Generating reminder for event ID {event['id']}: {event['content']}")
                        reminder_prompt = (
                            f"You are '{HOST_NAME}', '{USER_NAME}'s assistant. "
                            f"Generate a brief, friendly reminder message for the user about their upcoming event. "
                            f"Current time: {now.strftime('%I:%M %p')}. "
                            f"Event details: '{event['content']}' scheduled for {event['datetime'].strftime('%I:%M %p on %A, %B %d')}."
                             "Keep the message concise and start directly with the reminder (e.g., 'Just a reminder...')."
                        )
                        # Use the separate LLM instance for the reminder
                        messages = [SystemMessage(content=f"You are a helpful assistant named {HOST_NAME}."), HumanMessage(content=reminder_prompt)]
                        response = reminder_llm.invoke(messages)
                        reminder_message = response.content.strip()

                        # Send message to main thread via queue
                        self.message_queue.put({"sender": HOST_NAME, "message": reminder_message, "tag": "assistant"})
                        logging.info(f"Reminder generated and queued for event ID {event['id']}")

                    except Exception as e:
                        logging.error(f"Error generating reminder for event ID {event['id']}: {e}", exc_info=True)
                        # Optionally remove from reminded_event_ids if generation fails?
                        # self.reminded_event_ids.discard(event['id'])


            except Exception as e:
                logging.error(f"Error in background_calendar_checker loop: {e}", exc_info=True)

            # Wait before next check
            self.calendar_thread_stop_event.wait(CALENDAR_CHECK_INTERVAL_SECONDS)

        logging.info("Background calendar checker stopped.")


    def check_message_queue(self):
        """Checks the queue for messages from background threads and updates UI."""
        while not self.message_queue.empty():
            try:
                msg_data = self.message_queue.get_nowait()
                sender = msg_data.get("sender", "System")
                message = msg_data.get("message", "")
                tag = msg_data.get("tag", "system")
                self.add_message(sender, message, tag=tag)
                # Also add reminder to memory?
                if tag == 'assistant' and sender == HOST_NAME: # Add reminders sent by assistant to memory
                    memory.save_context({"input": f"[Reminder Sent for Event ID {msg_data.get('event_id', 'N/A')}]"}, {"output": message})
            except Exception as e:
                logging.error(f"Error processing message queue: {e}", exc_info=True)

        # Schedule next check
        self.root.after(100, self.check_message_queue)


    def on_closing(self):
        """Handles application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit? Unsaved notes might be lost if not saved via the tool."):
            self.set_status("Shutting down...")
            logging.info("Shutdown requested.")
            # Signal the calendar thread to stop
            self.calendar_thread_stop_event.set()
            # Wait briefly for the thread to potentially finish its current loop
            # self.calendar_thread.join(timeout=1.0) # Don't block UI for too long

            # Save the vector store one last time
            save_vector_store()

            self.root.destroy()
            logging.info("Application closed.")

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualAssistantApp(root)
    root.mainloop()