import os
import csv
import json
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from typing import List, Dict, Optional
import random
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

class AnswerEvaluation(BaseModel):
    """Evaluation of a user's answer to a test question."""
    is_correct: bool = Field(description="Whether the user's answer is correct or not")
    hint: str = Field(description="A hint to help the user improve their answer if incorrect")
    note: str = Field(description="A note about the mistake the user made, not shown to them directly")

class TestSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Test System")
        self.root.geometry("800x600")
        
        # Data variables
        self.questions = []
        self.current_question_idx = -1
        self.context = ""
        self.data_folder = ""
        self.error_notes = []
        
        # LLM setup
        self.llm = ChatOpenAI(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen2.5:32b-instruct-q4_K_S",
            api_key="ollama",
            temperature=0.2,
        )
        self.parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)
        
        # GUI setup
        self.setup_gui()
    
    def setup_gui(self):
        # Top frame for folder selection
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.load_btn = tk.Button(top_frame, text="Load Test Dataset", command=self.load_dataset)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.dataset_label = tk.Label(top_frame, text="No dataset loaded")
        self.dataset_label.pack(side=tk.LEFT, padx=5)
        
        # Middle frame for context display
        context_frame = tk.Frame(self.root)
        context_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(context_frame, text="Test Context:").pack(anchor='w')
        
        self.context_text = scrolledtext.ScrolledText(context_frame, height=4, wrap=tk.WORD)
        self.context_text.pack(fill=tk.X, pady=5)
        self.context_text.config(state=tk.DISABLED)
        
        # Question display
        question_frame = tk.Frame(self.root)
        question_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(question_frame, text="Question:").pack(anchor='w')
        
        self.question_text = scrolledtext.ScrolledText(question_frame, height=4, wrap=tk.WORD)
        self.question_text.pack(fill=tk.X, pady=5)
        self.question_text.config(state=tk.DISABLED)
        
        # Answer input
        answer_frame = tk.Frame(self.root)
        answer_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(answer_frame, text="Your Answer:").pack(anchor='w')
        
        self.answer_text = scrolledtext.ScrolledText(answer_frame, height=5, wrap=tk.WORD)
        self.answer_text.pack(fill=tk.X, pady=5)
        
        # Feedback display
        feedback_frame = tk.Frame(self.root)
        feedback_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(feedback_frame, text="Feedback:").pack(anchor='w')
        
        self.feedback_text = scrolledtext.ScrolledText(feedback_frame, height=5, wrap=tk.WORD)
        self.feedback_text.pack(fill=tk.X, pady=5)
        self.feedback_text.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.submit_btn = tk.Button(button_frame, text="Submit Answer", command=self.submit_answer)
        self.submit_btn.pack(side=tk.LEFT, padx=5)
        self.submit_btn.config(state=tk.DISABLED)
        
        self.show_answer_btn = tk.Button(button_frame, text="Show Answer", command=self.show_answer)
        self.show_answer_btn.pack(side=tk.LEFT, padx=5)
        self.show_answer_btn.config(state=tk.DISABLED)
        
        self.next_btn = tk.Button(button_frame, text="Next Question", command=self.next_question)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn.config(state=tk.DISABLED)
        
        self.notes_btn = tk.Button(button_frame, text="Show Error Notes", command=self.show_error_notes)
        self.notes_btn.pack(side=tk.LEFT, padx=5)
        self.notes_btn.config(state=tk.DISABLED)
    
    def load_dataset(self):
        self.data_folder = filedialog.askdirectory(title="Select Test Dataset Folder")
        if not self.data_folder:
            return
        
        # Find description file (.txt)
        description_files = [f for f in os.listdir(self.data_folder) if f.endswith('.txt')]
        if not description_files:
            messagebox.showerror("Error", "No description (.txt) file found in the selected folder.")
            return
        
        # Find questions file (.csv)
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        if not csv_files:
            messagebox.showerror("Error", "No questions (.csv) file found in the selected folder.")
            return
        
        # Load description
        with open(os.path.join(self.data_folder, description_files[0]), 'r') as f:
            self.context = f.read().strip()
        
        # Load questions
        self.questions = []
        with open(os.path.join(self.data_folder, csv_files[0]), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Question' in row and 'Answer' in row:
                    self.questions.append({
                        'question': row['Question'],
                        'answer': row['Answer']
                    })
        
        if not self.questions:
            messagebox.showerror("Error", "No valid questions found in the CSV file.")
            return
        
        # Reset error notes
        self.error_notes = []
        
        # Update UI
        self.dataset_label.config(text=f"Loaded {len(self.questions)} questions")
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete(1.0, tk.END)
        self.context_text.insert(tk.END, self.context)
        self.context_text.config(state=tk.DISABLED)
        
        # Enable buttons
        self.next_btn.config(state=tk.NORMAL)
        
        # Load first question
        self.current_question_idx = -1
        self.next_question()
    
    def next_question(self):
        # Reset feedback
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.config(state=tk.DISABLED)
        
        # Clear answer
        self.answer_text.delete(1.0, tk.END)
        
        # Get next question
        if self.current_question_idx < len(self.questions) - 1:
            self.current_question_idx += 1
        else:
            self.current_question_idx = 0  # Wrap around to the beginning
        
        # Display question
        self.question_text.config(state=tk.NORMAL)
        self.question_text.delete(1.0, tk.END)
        self.question_text.insert(tk.END, self.questions[self.current_question_idx]['question'])
        self.question_text.config(state=tk.DISABLED)
        
        # Enable/disable buttons
        self.submit_btn.config(state=tk.NORMAL)
        self.show_answer_btn.config(state=tk.NORMAL)
        self.notes_btn.config(state=tk.NORMAL if self.error_notes else tk.DISABLED)
    
    def submit_answer(self):
        user_answer = self.answer_text.get(1.0, tk.END).strip()
        if not user_answer:
            messagebox.showinfo("Info", "Please enter an answer before submitting.")
            return
        
        correct_answer = self.questions[self.current_question_idx]['answer']
        question = self.questions[self.current_question_idx]['question']
        
        # Evaluate answer with LLM
        evaluation = self.evaluate_answer(question, correct_answer, user_answer)
        
        # Display feedback
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)
        
        if evaluation.is_correct:
            self.feedback_text.insert(tk.END, "✅ Correct!\n\n")
        else:
            self.feedback_text.insert(tk.END, "❌ Not quite right.\n\nHint: ")
            self.feedback_text.insert(tk.END, evaluation.hint)
            
            # Save error note
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            note = {
                "timestamp": timestamp,
                "question": question,
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "note": evaluation.note
            }
            self.error_notes.append(note)
            
            # Write to file
            self.save_error_notes()
        
        self.feedback_text.config(state=tk.DISABLED)
    
    def evaluate_answer(self, question, correct_answer, user_answer):
        try:
            # Format prompt with parser instructions
            format_instructions = self.parser.get_format_instructions()
            
            messages = [
                SystemMessage(content=f"""You are an expert test evaluator. 
You will be given:
1. The context of the test
2. A question
3. The correct answer to the question
4. A user's answer to the question

Your job is to determine if the user's answer is correct. Focus on the underlying concepts and meaning, not just exact wording.
If the answer is incorrect, provide a helpful hint that guides them toward the correct answer without giving it away completely.
Also include a note about what mistake they made for educational purposes.

{format_instructions}"""),
                
                HumanMessage(content=f"""
Context: {self.context}

Question: {question}

Correct Answer: {correct_answer}

User's Answer: {user_answer}

Evaluate whether the user's answer is correct. Return your evaluation in the requested JSON format.
""")
            ]
            
            response = self.llm.invoke(messages)
            parsed_response = self.parser.parse(response.content)
            return parsed_response
            
        except Exception as e:
            messagebox.showerror("Error", f"Error evaluating answer: {str(e)}")
            # Return a default evaluation
            return AnswerEvaluation(
                is_correct=False,
                hint="Error evaluating your answer. Please try again.",
                note=f"System error: {str(e)}"
            )
    
    def show_answer(self):
        correct_answer = self.questions[self.current_question_idx]['answer']
        
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.insert(tk.END, f"Correct Answer:\n\n{correct_answer}")
        self.feedback_text.config(state=tk.DISABLED)
    
    def save_error_notes(self):
        if not self.error_notes or not self.data_folder:
            return
            
        notes_path = os.path.join(self.data_folder, "error_notes.json")
        
        try:
            with open(notes_path, 'w') as f:
                json.dump(self.error_notes, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save error notes: {str(e)}")
    
    def show_error_notes(self):
        if not self.error_notes:
            messagebox.showinfo("Info", "No error notes available.")
            return
            
        # Create a popup window
        notes_window = tk.Toplevel(self.root)
        notes_window.title("Error Notes")
        notes_window.geometry("700x500")
        
        # Create a scrolled text widget
        notes_text = scrolledtext.ScrolledText(notes_window, wrap=tk.WORD)
        notes_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Format and display notes
        for i, note in enumerate(self.error_notes):
            notes_text.insert(tk.END, f"Note #{i+1} - {note['timestamp']}\n")
            notes_text.insert(tk.END, f"Question: {note['question']}\n")
            notes_text.insert(tk.END, f"User Answer: {note['user_answer']}\n")
            notes_text.insert(tk.END, f"Correct Answer: {note['correct_answer']}\n")
            notes_text.insert(tk.END, f"Note: {note['note']}\n")
            notes_text.insert(tk.END, "\n" + "-"*50 + "\n\n")
        
        notes_text.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = TestSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()
