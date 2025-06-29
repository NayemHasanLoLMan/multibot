from typing import List, Dict
from anthropic import Anthropic
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import os
from anthropic._exceptions import AuthenticationError as ClaudeAuthError
from openai import AuthenticationError as OpenAIAuthError
import json

# Load environment variables from .env file
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
xai_api_key = os.getenv("XAI_API_KEY")



genai_model = genai.GenerativeModel('gemini-2.0-flash')
claude_model = "claude-3-5-sonnet-20241022"
chatgpt_model = "gpt-4-turbo"
deepseek_model = "deepseek-chat"
xai_model = "x-ai/grok-3"



deepseek_url = os.environ.get("DEEPSEEK_URL")
xai_url = os.environ.get("XAI_URL")


max_tokens = 1600


def generate_response(input_text: str, chat_history: List[Dict] = None) -> Dict:
    if chat_history is None:
        chat_history = []
    
    responses = {}
    
    # Claude
    try:
        claude_messages = []
        # Add conversation history
        for chat in chat_history:
            if chat.get("user", "").strip():
                claude_messages.append({"role": "user", "content": chat["user"]})
            if chat.get("claude", "").strip():
                claude_messages.append({"role": "assistant", "content": chat["claude"]})
        
        # Add current input
        claude_messages.append({"role": "user", "content": input_text})
        
        response_obj = anthropic_client.messages.create(
            model= claude_model,
            max_tokens=max_tokens,
            messages=claude_messages
        )
        claude_response = response_obj.content[0].text if response_obj and response_obj.content else "No response generated."
        
    except ClaudeAuthError:
        claude_response = "Invalid Claude API key."
    except Exception as e:
        claude_response = f"Claude error: {str(e)}"
    
    responses["claude"] = claude_response
    
    # ChatGPT
    try:
        chatgpt_messages = []
        # Add conversation history
        for chat in chat_history:
            if chat.get("user", "").strip():
                chatgpt_messages.append({"role": "user", "content": chat["user"]})
            if chat.get("chatgpt", "").strip():
                chatgpt_messages.append({"role": "assistant", "content": chat["chatgpt"]})
        
        # Add current input
        chatgpt_messages.append({"role": "user", "content": input_text})

        client = OpenAI(api_key=openai_api_key)
        
        response_obj = client.chat.completions.create(
            model=chatgpt_model,
            max_tokens=max_tokens,
            messages=chatgpt_messages
        )
        chatgpt_response = response_obj.choices[0].message.content if response_obj and response_obj.choices else "No response generated."
        
    except OpenAIAuthError:
        chatgpt_response = "Invalid OpenAI API key."
    except Exception as e:
        chatgpt_response = f"ChatGPT error: {str(e)}"
    
    responses["chatgpt"] = chatgpt_response

    # Gemini
    try:
        conversation_context = ""
        # Add conversation history
        for chat in chat_history:
            if chat.get("user", "").strip():
                conversation_context += f"User: {chat['user']}\n"
            if chat.get("gemini", "").strip():
                conversation_context += f"Assistant: {chat['gemini']}\n"
        
        # Add current input
        conversation_context += f"User: {input_text}\nAssistant:"
        
        response_obj = genai_model.generate_content(
            conversation_context,
            generation_config={"max_output_tokens": max_tokens}
        )
        gemini_response = response_obj.text.strip() if response_obj else "No response generated."
        
    except Exception as e:
        gemini_response = f"Gemini error: {str(e)}"
    
    responses["gemini"] = gemini_response
      



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
                        base_url=deepseek_url
                        )

        response_obj = client.chat.completions.create(
            model=deepseek_model,
            max_tokens=max_tokens,
            messages=deepseek_messages,
            stream=False
        )



        deepseek_response = response_obj.choices[0].message.content if response_obj.choices else "No response generated."
  
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

        xai_response = response_obj.choices[0].message.content if response_obj.choices else "No response generated."
        
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