import tkinter as tk
from tkinter import ttk, messagebox
import os
import subprocess
import sys

class AppLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Miles' Open Source AI Application Collection")
        self.root.geometry("600x400")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title Label
        title_label = ttk.Label(
            main_frame, 
            text="Miles' Open Source AI Application Collection",
            font=("Helvetica", 14, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Create frame for app list
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create listbox
        self.app_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Helvetica", 11),
            selectmode=tk.SINGLE,
            height=10
        )
        self.app_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=self.app_listbox.yview)
        
        # Create description frame
        desc_frame = ttk.Frame(main_frame)
        desc_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Description label
        self.desc_label = ttk.Label(
            desc_frame,
            text="Select an application to launch",
            wraplength=550,
            justify=tk.LEFT
        )
        self.desc_label.pack(fill=tk.X)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(20, 0))
        
        # Launch button
        self.launch_button = ttk.Button(
            button_frame,
            text="Launch Application",
            command=self.launch_app
        )
        self.launch_button.pack()
        
        # Populate apps list
        self.populate_apps()
        
        # Bind selection event
        self.app_listbox.bind('<<ListboxSelect>>', self.on_select)

    def populate_apps(self):
        """Populate the listbox with available applications"""
        # Dictionary to store app descriptions
        self.app_info = {
            "auto_completion_app": {
                "name": "AI Text Completion App",
                "description": "A desktop application that provides real-time AI-powered text completion suggestions as you type, supporting both OpenAI and Ollama language models."
            },
            "Linux_intelligent_web_console": {
                "name": "Intelligent Linux Web Console",
                "description": "A web-based Linux terminal console powered by LangChain and LLMs that helps users execute commands through natural language understanding. Use 'ask' followed by your request in quotes to get AI assistance with commands."
            }
            # Add more apps here as they are developed
        }
        
        # Add apps to listbox
        for app_id, info in self.app_info.items():
            if os.path.exists(app_id) and os.path.isfile(os.path.join(app_id, "main.py")):
                self.app_listbox.insert(tk.END, info["name"])

    def on_select(self, event):
        """Handle selection event"""
        if not self.app_listbox.curselection():
            return
            
        # Get selected app name
        selected_name = self.app_listbox.get(self.app_listbox.curselection())
        
        # Find the app_id for the selected name
        selected_app_id = None
        for app_id, info in self.app_info.items():
            if info["name"] == selected_name:
                selected_app_id = app_id
                break
        
        if selected_app_id:
            # Update description
            self.desc_label.config(text=self.app_info[selected_app_id]["description"])

    def launch_app(self):
        """Launch the selected application"""
        if not self.app_listbox.curselection():
            messagebox.showwarning("No Selection", "Please select an application to launch.")
            return
            
        # Get selected app name
        selected_name = self.app_listbox.get(self.app_listbox.curselection())
        
        # Find the app_id for the selected name
        selected_app_id = None
        for app_id, info in self.app_info.items():
            if info["name"] == selected_name:
                selected_app_id = app_id
                break
        
        if selected_app_id:
            try:
                # Get the path to the Python interpreter
                python_executable = sys.executable
                
                # Construct the path to main.py
                main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         selected_app_id, "main.py")
                
                # Launch the application
                subprocess.Popen([python_executable, main_script])
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to launch application: {str(e)}")

def main():
    root = tk.Tk()
    app = AppLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()
