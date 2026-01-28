import requests
import json
import time
from typing import List, Dict, Optional, Union

class NCloudLLM:
    """
    NCloud HCX-007 Wrapper with Thinking Support
    """
    def __init__(self, 
                 api_key: str, 
                 api_url: str, 
                 thinking_effort: str = "none",
                 temperature: float = 0.5,
                 max_tokens: int = 4096):
        self.api_key = api_key
        self.api_url = api_url
        self.thinking_effort = thinking_effort
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: List[Dict[str, str]], thinking_effort: str = None) -> str:
        """
        Generate response from HCX-007
        """
        # Ensure api_key starts with Bearer if not already
        auth_header = self.api_key if self.api_key.startswith("Bearer ") else f"Bearer {self.api_key}"
        
        headers = {
            "Authorization": auth_header,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": "pageindex-comparison-llm",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }

        # Override thinking effort if provided
        effort = thinking_effort if thinking_effort else self.thinking_effort
        
        # Adjust parameters for thinking mode
        if effort != "none":
            # Thinking mode requires lower temperature and sufficient token budget
            temp = 0.5 
            max_tok = max(self.max_tokens, 2048) # Ensure enough tokens for thinking
        else:
            temp = self.temperature
            max_tok = self.max_tokens

        data = {
            "messages": messages,
            "topP": 0.8,
            "topK": 0,
            "maxCompletionTokens": max_tok,
            "temperature": temp,
            "repetitionPenalty": 1.1,
            "stopBefore": [],
            "includeAiFilters": True,
            "seed": 0
        }

        # Add thinking parameter if supported/needed 
        # Note: Actual API parameter for thinking might vary. 
        # Based on KG-RAG guide, we assume standard chat completion, 
        # but if specific parameter is needed for thinking, add it here.
        # For now, we rely on system prompt or model's inherent capability 
        # unless explicit parameter is documented in the code we copied.
        # (KG-RAG guide mentions thinking tokens in response, suggesting it's native)
        
        # If thinking control is via parameter (hypothetical, need to check API docs or legacy code):
        if effort != "none":
             data["thinking"] = {"effort": effort}


        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data, stream=True)
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if not response.ok:
                    print(f"API Error: {response.status_code} - {response.text}")
                    
                response.raise_for_status()
                
                # Handle streaming response
                full_content = ""
                thinking_content = ""
                current_event = ""
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('event:'):
                            current_event = decoded_line.split(':', 1)[1].strip()
                        elif decoded_line.startswith('data:'):
                            data_str = decoded_line.split(':', 1)[1]
                            try:
                                data_json = json.loads(data_str)
                                if "message" in data_json:
                                    content = data_json["message"].get("content", "")
                                    
                                    # 'result' event contains complete response - use it directly
                                    if current_event == "result":
                                        full_content = content
                                    elif content:
                                        # For other events, only add if content is not empty
                                        # and not already in full_content (prevent duplicates)
                                        if not full_content.endswith(content):
                                            full_content += content
                                    
                                    # Capture thinking content if available
                                    if "thinkingContent" in data_json["message"]:
                                        thinking_content += data_json["message"]["thinkingContent"]
                                        
                            except json.JSONDecodeError:
                                pass
                                
                return full_content.strip()

            except Exception as e:
                print(f"Error generating response: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise e
        
        return ""
