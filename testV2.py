from openai import OpenAI
from openai import AuthenticationError as OpenAIAuthError
import json
from typing import List, Dict
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
xai_api_key = os.getenv("XAI_API_KEY")



deepseek_url = "https://api.deepseek.com"
xai_url = "https://api.x.ai/v1"



deepseek_model = "deepseek-chat"
xai_model = "grok-3"

max_tokens = 3000

def generate_response(input_text: str, chat_history: List[Dict] = None) -> Dict:
    if chat_history is None:
        chat_history = []
    
    responses = {}

    # DeepSeek
    try:
        deepseek_messages = []
        # Add conversation history
        for chat in chat_history:
            if chat.get("user", "").strip():
                deepseek_messages.append({"role": "user", "content": chat["user"]})
            if chat.get("deepseek", "").strip():
                deepseek_messages.append({"role": "assistant", "content": chat["deepseek"]})
        
        # Add current input
        deepseek_messages.append({"role": "user", "content": input_text})

        client = OpenAI(api_key=deepseek_api_key,
                        base_url=xai_url
                        )

        response_obj = client.chat.completions.create(
            model=deepseek_model,
            max_tokens=max_tokens,
            messages=deepseek_messages,
            stream=False
        )



        deepseek_response = response_obj['choices'][0]['message']['content'] if 'choices' in response_obj else "No response generated."
        
    except OpenAIAuthError:
        deepseek_response = "Invalid DeepSeek API key."
    except Exception as e:
        deepseek_response = f"DeepSeek error: {str(e)}"
    
    responses["deepseek"] = deepseek_response





    # XAI
    try:
        xai_messages = []
        # Add conversation history
        for chat in chat_history:
            if chat.get("user", "").strip():
                xai_messages.append({"role": "user", "content": chat["user"]})
            if chat.get("xai", "").strip():
                xai_messages.append({"role": "assistant", "content": chat["xai"]})
        
        # Add current input
        xai_messages.append({"role": "user", "content": input_text})


        client = OpenAI(api_key=xai_api_key,
                        base_url=xai_url,
                        )

        response_obj = client.chat.completions.create(
        model=xai_model,
        max_tokens=max_tokens,
        messages=xai_messages
        )

        xai_response = response_obj['choices'][0]['message']['content'] if 'choices' in response_obj else "No response generated."
        
    except OpenAIAuthError:
        xai_response = "Invalid XAI API key."
    except Exception as e:
        xai_response = f"XAI error: {str(e)}"
    
    responses["xai"] = xai_response

    return responses




if __name__ == "__main__":
    input_text = "hello"

    # Simulate backend-provided chat history
    chat_history = [
        {
            "user": "Hello how are you?",
            "chatgpt": "I'm doing well, thank you for asking!",
            "claude": "I'm doing great, thanks for asking!",
            "gemini": "I'm fine, thank you!",
            "deepseek": "I'm doing well, thanks for asking!",
            "xai": "I'm doing great, thanks for asking!"
        },
    ] 

    result = generate_response(input_text, chat_history)
    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    print(json_output)