import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox
import threading
import time
import json
import requests
import os
from typing import Optional, Dict, Any

class AICompletionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Completion")
        self.root.geometry("900x600")

        # Configuration variables
        self.auto_completion_enabled = tk.BooleanVar(value=True)
        self.completion_delay = tk.IntVar(value=3)  # seconds
        self.num_tokens = tk.IntVar(value=20)
        self.api_type = tk.StringVar(value="openai")  # "openai" or "ollama"
        self.api_url = tk.StringVar(value="http://127.0.0.1:5000/v1/completions")  # Default for OpenAI
        self.api_key = tk.StringVar()
        self.model = tk.StringVar(value="phi4:14b-q8_0")

        # Completion state
        self.completion_timer = None
        self.completion_thread = None
        self.completion_text = None
        self.has_pending_completion = False
        self.last_keypress_time = 0
        self.is_completing = False

        self._create_ui()
        self._bind_events()
        self._load_config()

    def _create_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_command(label="Save As", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure", command=self.open_settings)

        # Create text area with both horizontal and vertical scrollbars
        self.text_frame = ttk.Frame(main_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Text widget with custom tags for styling
        self.text_area = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD, font=("Consolas", 11))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.text_area.tag_configure("completion", foreground="gray")

        # Status bar
        self.status_bar = ttk.Frame(main_frame)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))

        # Auto-completion toggle
        self.auto_completion_check = ttk.Checkbutton(
            self.status_bar, 
            text="Auto-Completion", 
            variable=self.auto_completion_enabled,
            command=self._on_auto_completion_toggled
        )
        self.auto_completion_check.pack(side=tk.LEFT, padx=(0, 10))

        # Status label
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.RIGHT)

        self.current_file = None

    def _bind_events(self):
        self.text_area.bind("<Key>", self._on_text_change)
        self.text_area.bind("<Tab>", self._on_tab_press)
        self.text_area.bind("<BackSpace>", self._on_backspace)
        self.text_area.bind("<Return>", self._on_return)
        
    def _load_config(self):
        config_file = os.path.expanduser("~/.aicompletion.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.api_type.set(config.get("api_type", "openai"))
                self.api_url.set(config.get("api_url", "http://127.0.0.1:5000/v1/completions"))
                self.api_key.set(config.get("api_key", ""))
                self.model.set(config.get("model", "llama2"))
                self.completion_delay.set(config.get("delay", 3))
                self.num_tokens.set(config.get("tokens", 20))
                self.auto_completion_enabled.set(config.get("enabled", True))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")

    def _save_config(self):
        config = {
            "api_type": self.api_type.get(),
            "api_url": self.api_url.get(),
            "api_key": self.api_key.get(),
            "model": self.model.get(),
            "delay": self.completion_delay.get(),
            "tokens": self.num_tokens.get(),
            "enabled": self.auto_completion_enabled.get()
        }
        
        config_file = os.path.expanduser("~/.aicompletion.json")
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")

    def _on_text_change(self, event):
        # Reset the timer on every keypress
        self.last_keypress_time = time.time()
        
        if self.completion_timer:
            self.root.after_cancel(self.completion_timer)
            
        # If there's a pending completion, remove it
        self._remove_completion()
        
        # If a completion is currently being generated, abort it
        if self.is_completing:
            self.is_completing = False
            self.status_label.config(text="Completion aborted")
            return None

        # Skip starting a new timer for special keys
        if event.keysym in ('Tab', 'Return', 'BackSpace', 'Delete', 'Escape'):
            return
            
        # Don't schedule if auto-completion is disabled
        if not self.auto_completion_enabled.get():
            return
            
        # Schedule a new completion request
        delay_ms = self.completion_delay.get() * 1000
        self.completion_timer = self.root.after(delay_ms, self._request_completion)
        
        # Allow the event to proceed normally
        return None

    def _on_tab_press(self, event):
        if self.has_pending_completion:
            # Accept the completion
            current_pos = self.text_area.index(tk.INSERT)
            
            # Find all text with the completion tag
            start_index = "1.0"
            completion_ranges = []
            
            while True:
                tag_range = self.text_area.tag_nextrange("completion", start_index)
                if not tag_range:
                    break
                completion_ranges.append(tag_range)
                start_index = tag_range[1]
            
            # If we found completion ranges, accept them all
            if completion_ranges:
                # Remove the completion tag from all ranges
                for start, end in completion_ranges:
                    self.text_area.tag_remove("completion", start, end)
                
                # Update cursor position to end of the last completion
                self.text_area.mark_set(tk.INSERT, completion_ranges[-1][1])
                
                self.has_pending_completion = False
                self.status_label.config(text="Completion accepted")
                
                return "break"  # Prevent default Tab behavior
            else:
                # Fallback to the original line-based approach if no tagged ranges found
                completion_start = f"{current_pos} linestart"
                completion_end = f"{current_pos} lineend"
                
                # Get the text including the completion
                text_with_completion = self.text_area.get(completion_start, completion_end)
                
                # Remove the completion tag - this will change it back to normal text color
                self.text_area.tag_remove("completion", completion_start, completion_end)
                
                self.has_pending_completion = False
                self.status_label.config(text="Completion accepted")
                
                # Update cursor position to end of accepted completion
                self.text_area.mark_set(tk.INSERT, completion_end)
                
                return "break"  # Prevent default Tab behavior
        return None

    def _on_backspace(self, event):
        # If backspace, remove any pending completions
        if self.has_pending_completion:
            self._remove_completion()
        return None  # Allow normal backspace behavior
    
    def _on_return(self, event):
        # If return, remove any pending completions
        if self.has_pending_completion:
            self._remove_completion()
        return None  # Allow normal return behavior

    def _on_auto_completion_toggled(self):
        status = "enabled" if self.auto_completion_enabled.get() else "disabled"
        self.status_label.config(text=f"Auto-completion {status}")
        if not self.auto_completion_enabled.get():
            self._remove_completion()

    def _remove_completion(self):
        if self.has_pending_completion:
            # Find all text with completion tag and remove it
            start_index = "1.0"
            while True:
                tag_range = self.text_area.tag_nextrange("completion", start_index)
                if not tag_range:
                    break
                self.text_area.delete(tag_range[0], tag_range[1])
                start_index = tag_range[0]
            
            self.has_pending_completion = False

    def _request_completion(self):
        if self.is_completing:
            return
            
        # Get the current text
        text = self.text_area.get("1.0", tk.INSERT)
        if not text.strip():
            return
            
        # Don't request completion if nothing has changed
        if self.completion_thread and self.completion_thread.is_alive():
            return
            
        self.is_completing = True
        self.status_label.config(text="Generating completion...")
        
        # Start a new thread to avoid blocking the UI
        self.completion_thread = threading.Thread(
            target=self._get_completion_from_api,
            args=(text,)
        )
        self.completion_thread.daemon = True
        self.completion_thread.start()

    def _get_completion_from_api(self, text):
        try:
            if self.api_type.get() == "openai":
                completion = self._get_openai_completion(text)
            else:  # ollama
                completion = self._get_ollama_completion(text)
                
            # Only display the completion if we're still in completing state
            # (i.e., the user hasn't typed anything to abort it)
            if completion and self.is_completing:
                # Schedule displaying the completion on the main thread
                self.root.after(0, lambda: self._display_completion(completion))
        except Exception as e:
            error_message = str(e)
            self.root.after(0, lambda: messagebox.showerror("API Error", error_message))
        finally:
            self.is_completing = False

    def _get_openai_completion(self, text):
        # OpenAI API call
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key.get()}"
            }
            
            data = {
                "model": self.model.get(),
                "prompt": text,
                "max_tokens": self.num_tokens.get(),
                "temperature": 0.7
            }
            
            response = requests.post(
                self.api_url.get() or "http://127.0.0.1:5000/v1/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if "choices" in result and len(result["choices"]) > 0:
                if "text" in result["choices"][0]:
                    # Standard OpenAI completions endpoint
                    return result["choices"][0]["text"]
                elif "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    # Chat completions endpoint
                    return result["choices"][0]["message"]["content"]
                    
            return ""
            
        except Exception as e:
            error_message = str(e)
            if "Read timed out" in error_message:
                error_message = f"Connection to OpenAI server timed out. Please check your connection and server status."
            self.root.after(0, lambda: messagebox.showerror("API Error", error_message))
            return ""

    def _get_ollama_completion(self, text):
        # Ollama API call
        try:
            headers = {"Content-Type": "application/json"}
            
            data = {
                "model": self.model.get(),
                "prompt": text,
                "stream": False,
                "raw": True,
                "options": {
                    "num_predict": self.num_tokens.get()
                }
            }
            
            response = requests.post(
                self.api_url.get(),
                headers=headers,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "response" in result:
                return result["response"]
                
            return ""
            
        except Exception as e:
            error_message = str(e)
            if "Read timed out" in error_message:
                error_message = f"Connection to Ollama server timed out. Please make sure Ollama is running at {self.api_url.get()}"
            self.root.after(0, lambda: messagebox.showerror("API Error", error_message))
            return ""

    def _display_completion(self, completion_text):
        if not completion_text or not self.auto_completion_enabled.get():
            self.status_label.config(text="Ready")
            return
            
        # Remove any existing completion
        self._remove_completion()
        
        # Insert the completion at the current cursor position
        current_pos = self.text_area.index(tk.INSERT)
        self.text_area.insert(current_pos, completion_text, "completion")
        
        # Mark that we have a pending completion
        self.has_pending_completion = True
        
        # Move cursor back to where it was before the completion
        self.text_area.mark_set(tk.INSERT, current_pos)
        
        self.status_label.config(text="Completion ready (Tab to accept)")

    def open_settings(self):
        # Create a new top-level window for settings
        settings = tk.Toplevel(self.root)
        settings.title("Settings")
        settings.geometry("400x400")
        settings.transient(self.root)
        settings.grab_set()
        
        settings_frame = ttk.Frame(settings, padding="10")
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # API Type Selection
        ttk.Label(settings_frame, text="API Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        api_type_frame = ttk.Frame(settings_frame)
        api_type_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        def update_api_url(*args):
            if self.api_type.get() == "openai":
                self.api_url.set("http://127.0.0.1:5000/v1/completions")
            else:  # ollama
                self.api_url.set("http://localhost:11434/api/generate")
        
        # Bind the update function to api_type changes
        self.api_type.trace_add("write", update_api_url)
        
        ttk.Radiobutton(api_type_frame, text="OpenAI", variable=self.api_type, value="openai").pack(side=tk.LEFT)
        ttk.Radiobutton(api_type_frame, text="Ollama", variable=self.api_type, value="ollama").pack(side=tk.LEFT, padx=(10, 0))
        
        # API URL
        ttk.Label(settings_frame, text="API URL:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(settings_frame, textvariable=self.api_url, width=30).grid(row=1, column=1, sticky=tk.EW, pady=5)
        
        # API Key
        ttk.Label(settings_frame, text="API Key:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(settings_frame, textvariable=self.api_key, width=30, show="*").grid(row=2, column=1, sticky=tk.EW, pady=5)
        
        # Model
        ttk.Label(settings_frame, text="Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(settings_frame, textvariable=self.model, width=30).grid(row=3, column=1, sticky=tk.EW, pady=5)
        
        # Completion Delay
        ttk.Label(settings_frame, text="Delay (seconds):").grid(row=4, column=0, sticky=tk.W, pady=5)
        delay_spinner = ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.completion_delay, width=5)
        delay_spinner.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Number of Tokens
        ttk.Label(settings_frame, text="Tokens to generate:").grid(row=5, column=0, sticky=tk.W, pady=5)
        tokens_spinner = ttk.Spinbox(settings_frame, from_=5, to=100, textvariable=self.num_tokens, width=5)
        tokens_spinner.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # Auto-Completion Toggle
        ttk.Checkbutton(
            settings_frame, 
            text="Enable Auto-Completion", 
            variable=self.auto_completion_enabled
        ).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=10)
        
        # Info text
        info_text = ("For OpenAI: Default URL is http://127.0.0.1:5000/v1/completions\n"
                     "For Ollama: Default URL is http://localhost:11434/api/generate")
        ttk.Label(settings_frame, text=info_text, wraplength=380).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.grid(row=8, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Save", command=lambda: self._save_settings(settings)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings.destroy).pack(side=tk.LEFT, padx=5)

    def _save_settings(self, settings_window):
        # Save settings to config file
        self._save_config()
        settings_window.destroy()
        self.status_label.config(text="Settings saved")

    def new_file(self):
        if self.text_area.get("1.0", tk.END).strip():
            if not messagebox.askyesno("Confirm", "Discard current text?"):
                return
        
        self.text_area.delete("1.0", tk.END)
        self.current_file = None
        self.status_label.config(text="New file")

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert(tk.END, content)
                self.current_file = file_path
                self.status_label.config(text=f"Opened: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {e}")

    def save_file(self):
        if not self.current_file:
            self.save_as()
            return
            
        try:
            # Get text without any completions
            self._remove_completion()
            content = self.text_area.get("1.0", tk.END)
            
            with open(self.current_file, 'w', encoding='utf-8') as file:
                file.write(content)
                
            self.status_label.config(text=f"Saved: {os.path.basename(self.current_file)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    def save_as(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_file = file_path
            self.save_file()

if __name__ == "__main__":
    root = tk.Tk()
    app = AICompletionApp(root)
    root.mainloop()