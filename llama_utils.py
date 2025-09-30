import requests
import json

class LlamaAnalyzer:
    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        """
        Initialize Llama analyzer with Ollama.
        
        Args:
            model: Model name (default: llama3.2)
                   Options: llama3.2, llama3.2:1b, llama3.3
            base_url: Ollama API endpoint (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        print(f"🤖 Using Ollama model: {model}")
    
    def analyze_captions(self, prompt):
        """
        Sends frame captions to Llama for analysis.
        
        Args:
           prompts
        
        Returns:
            Analysis text from Llama
        """

        print("Sending to Llama... (this may take a minute)")
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure Ollama is running (ollama serve)")
    
    def analyze_captions_streaming(self, prompt):
        """
        Streams response from Llama for real-time output.
        Shows the response as it's being generated.
        """
        
        print("Streaming response from Llama...\n")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=300
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    chunk = json_response.get("response", "")
                    full_response += chunk
                    print(chunk, end="", flush=True)
            
            print("\n")  # New line at end
            return full_response
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure Ollama is running (ollama serve)")
