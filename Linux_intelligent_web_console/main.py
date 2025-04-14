# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import os
import json
import asyncio
from typing import List, Dict, Any
import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import uvicorn
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# LLM Configuration
LOCAL_API_BASE = "http://host.docker.internal:11434/v1"
LOCAL_MODEL_NAME = "qwen2.5:14b-instruct-q6_K" 
DUMMY_API_KEY = "ollama"

# Add these Pydantic models after the imports and before the app initialization
class LLMResponse(BaseModel):
    explanation: str = Field(..., description="Explanation of how to solve the task")
    command: str = Field(..., description="The exact command to run")

class WebSocketResponse(BaseModel):
    status: str = Field(..., pattern="^(success|error|info|confirm)$")
    output: str
    type: str = Field(..., pattern="^(output|error|ai_response|ai_processing|confirm)$")
    command: str | None = None
    current_dir: str | None = None

class CommandResponse(BaseModel):
    explanation: str = Field(description="A brief explanation of how to solve the task")
    command: str = Field(description="The exact command to run")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "explanation": "List all files in the current directory",
                    "command": "ls -la"
                }
            ]
        }

# Initialize LLM
try:
    llm = ChatOpenAI(
        base_url=LOCAL_API_BASE,
        model=LOCAL_MODEL_NAME,
        api_key=DUMMY_API_KEY,
        temperature=0.2,
    )
    logger.info(f"LLM initialized with model: {LOCAL_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    llm = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# Track current working directory
current_working_dir = "/workspace/"
# Create the directory if it doesn't exist
os.makedirs(current_working_dir, exist_ok=True)
logger.info(f"Initial working directory set to: {current_working_dir}")

# Execute shell command
async def execute_command(command: str) -> Dict[str, Any]:
    global current_working_dir
    
    try:
        # Handle cd command specially to update the current working directory
        if command.startswith("cd "):
            target_dir = command[3:].strip()
            
            # Handle relative paths
            if target_dir == "..":
                new_dir = os.path.dirname(current_working_dir)
            elif target_dir == ".":
                new_dir = current_working_dir
            elif target_dir.startswith("~"):
                # Expand home directory
                new_dir = os.path.expanduser(target_dir)
            elif os.path.isabs(target_dir):
                # Absolute path
                new_dir = target_dir
            else:
                # Relative path
                new_dir = os.path.join(current_working_dir, target_dir)
            
            # Check if directory exists
            if os.path.isdir(new_dir):
                current_working_dir = new_dir
                return {
                    "status": "success",
                    "output": f"Changed directory to: {current_working_dir}",
                    "type": "output",
                    "current_dir": current_working_dir
                }
            else:
                return {
                    "status": "error",
                    "output": f"Directory not found: {target_dir}",
                    "type": "error",
                    "current_dir": current_working_dir
                }
        
        # For all other commands, use the current working directory
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            shell=True,
            cwd=current_working_dir  # Set the working directory for the command
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return {
                "status": "success",
                "output": stdout.decode('utf-8', errors='replace'),
                "type": "output",
                "current_dir": current_working_dir
            }
        else:
            return {
                "status": "error",
                "output": stderr.decode('utf-8', errors='replace'),
                "type": "error",
                "current_dir": current_working_dir
            }
    except Exception as e:
        logger.error(f"Command execution error: {str(e)}")
        return {
            "status": "error",
            "output": f"Error executing command: {str(e)}",
            "type": "error",
            "current_dir": current_working_dir
        }

# Ask LLM for command suggestions
async def ask_llm(query: str) -> Dict[str, Any]:
    if not llm:
        return WebSocketResponse(
            status="error",
            output="LLM is not available. Please check server logs.",
            type="error",
            current_dir=current_working_dir
        ).model_dump()
    
    try:
        # Initialize the parser with our Pydantic model
        parser = PydanticOutputParser(pydantic_object=CommandResponse)
        
        # Create a prompt template that includes format instructions
        prompt = PromptTemplate(
            template="""You are a Linux command expert. Generate the exact one line command to accomplish this task: 

{query}

{format_instructions}

Keep it concise. Focus on the most efficient solution.""",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Format the prompt with our query
        formatted_prompt = prompt.format(query=query)
        
        # Get response from LLM
        messages = [HumanMessage(content=formatted_prompt)]
        response = llm.invoke(messages)
        
        try:
            # Parse the response into our Pydantic model
            parsed_response = parser.parse(response.content)
            
            # Create WebSocket response
            return WebSocketResponse(
                status="success",
                output=parsed_response.explanation,
                command=parsed_response.command,
                type="ai_response",
                current_dir=current_working_dir
            ).model_dump()
            
        except Exception as parse_error:
            logger.error(f"Failed to parse LLM response: {str(parse_error)}")
            return WebSocketResponse(
                status="error",
                output=f"Failed to parse LLM response into the required format. Please try again.",
                type="error",
                current_dir=current_working_dir
            ).model_dump()
            
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        return WebSocketResponse(
            status="error",
            output=f"Error processing query with LLM: {str(e)}",
            type="error",
            current_dir=current_working_dir
        ).model_dump()

# WebSocket endpoint for terminal
@app.websocket("/ws/terminal")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial welcome message
        welcome_msg = json.dumps({
            "status": "success",
            "output": f"Linux Web Terminal with AI Command Assistant\nCurrent directory: {current_working_dir}\nType commands or use 'ask' followed by your question in quotes.\nType 'help' to see available commands.",
            "type": "output",
            "current_dir": current_working_dir
        })
        await manager.send_personal_message(welcome_msg, websocket)
        
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                command = message.get("command", "").strip()
                
                if not command:
                    continue
                
                logger.info(f"Received command: {command}")
                
                # Handle AI assistance requests
                if command.startswith("ask "):
                    query = command[4:].strip()
                    # Remove quotes if present
                    if query.startswith('"') and query.endswith('"') or \
                       query.startswith("'") and query.endswith("'"):
                        query = query[1:-1]
                    
                    # Process query with LLM
                    processing_msg = json.dumps({
                        "status": "info",
                        "output": f"Processing query: {query}",
                        "type": "ai_processing",
                        "current_dir": current_working_dir
                    })
                    await manager.send_personal_message(processing_msg, websocket)
                    
                    llm_response = await ask_llm(query)
                    llm_response["current_dir"] = current_working_dir
                    await manager.send_personal_message(json.dumps(llm_response), websocket)
                    
                    # Ask for confirmation if a command was suggested
                    if llm_response.get("command"):
                        confirm_msg = json.dumps({
                            "status": "confirm",
                            "output": f"Execute this command? (Y/N):\n{llm_response.get('command')}",
                            "command": llm_response.get("command"),
                            "type": "confirm",
                            "current_dir": current_working_dir
                        })
                        await manager.send_personal_message(confirm_msg, websocket)
                
                # Handle confirmation responses
                elif message.get("type") == "confirmation":
                    if message.get("confirm", "").lower() in ["y", "yes"]:
                        confirmed_command = message.get("command", "")
                        result = await execute_command(confirmed_command)
                        await manager.send_personal_message(json.dumps(result), websocket)
                    else:
                        cancel_msg = json.dumps({
                            "status": "info",
                            "output": "Command execution cancelled.",
                            "type": "output",
                            "current_dir": current_working_dir
                        })
                        await manager.send_personal_message(cancel_msg, websocket)
                
                # Execute normal command
                else:
                    result = await execute_command(command)
                    await manager.send_personal_message(json.dumps(result), websocket)
                    
                    # After command execution, send current directory
                    if command.startswith("cd "):
                        dir_msg = json.dumps({
                            "status": "info",
                            "output": f"Current directory: {current_working_dir}",
                            "type": "output",
                            "current_dir": current_working_dir
                        })
                        await manager.send_personal_message(dir_msg, websocket)
                    
            except json.JSONDecodeError:
                error_msg = json.dumps({
                    "status": "error",
                    "output": "Invalid message format",
                    "type": "error"
                })
                await manager.send_personal_message(error_msg, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            error_msg = json.dumps({
                "status": "error",
                "output": f"Server error: {str(e)}",
                "type": "error"
            })
            await manager.send_personal_message(error_msg, websocket)
        except:
            pass

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def get():
    with open("static/index.html", "r") as f:
        return f.read()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)