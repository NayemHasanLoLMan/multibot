from typing import List, Dict
from anthropic import Anthropic
import google.generativeai as genai
import openai
from dotenv import load_dotenv
import os
from anthropic._exceptions import AuthenticationError as ClaudeAuthError
from openai import AuthenticationError as OpenAIAuthError
import time

import atexit

# Load environment variables from .env file
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
genai_model = genai.GenerativeModel('gemini-2.0-flash')
anthropic_client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

# Initialize OpenAI API key (older version)
openai.api_key = os.getenv("OPENAI_API_KEY")



# System prompts
MAIN_PROMPTS = {
    "claude": """You are Claude, a helpful AI assistant created by Anthropic. Provide thoughtful, nuanced responses while maintaining conversation context.

    RESPONSE STRUCTURE

    - Use consistent markdown formatting with proper spacing and paragraph breaks.
    - Organize complex responses using clear headers and logical sections.
    - Scale response length to match query complexity.
    - Include relevant code blocks, examples, or reference materials when appropriate.

    
    """,

    "gemini": """You are a helpful AI assistant powered by Google. Provide thoughtful, nuanced responses while maintaining conversation context.

    RESPONSE STRUCTURE

    - Use consistent markdown formatting with proper spacing and paragraph breaks.
    - Organize complex responses using clear headers and logical sections.
    - Scale response length to match query complexity.
    - Include relevant code blocks, examples, or reference materials when appropriate.
    
    """,

    "chatgpt": """You are a helpful and professional assistant. Provide thoughtful, nuanced responses while maintaining conversation context.

    RESPONSE STRUCTURE

    - Use consistent markdown formatting with proper spacing and paragraph breaks.
    - Organize complex responses using clear headers and logical sections.
    - Scale response length to match query complexity.
    - Include relevant code blocks, examples, or reference materials when appropriate.

    """
}

RULES_PROMPT = """  
    **IMPORTANT CONVERSATION HISTORY RULE**:

    - Use the conversation history for context and continuity.
    - Always provide your own natural, unique response without copying previous answers.
    - Even if similar questions have been asked before, give fresh insights and your own perspective.
    - Respond naturally as yourself without referencing other AI models or previous responses.
    - Focus on being helpful and providing value with your own expertise and approach.
    """

SYSTEM_PROMPTS_CLAUDE = MAIN_PROMPTS["claude"] + RULES_PROMPT
SYSTEM_PROMPTS_CHATGPT = MAIN_PROMPTS["chatgpt"] + RULES_PROMPT
SYSTEM_PROMPTS_GEMINI = MAIN_PROMPTS["gemini"] + RULES_PROMPT


def generate_title_from_message(message: str, model: str) -> str:
    """
    Generate a concise title from the first message using the specified model.
    """
    try:
        # Simplified and more specific prompt
        title_prompt = f"Create a short and proper only the title (3-5 words maximum) for this message: \"{message}\" (without any explanation or additional text)."     
        if model == "claude":
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=25,
                temperature=0.3,
                messages=[{"role": "user", "content": title_prompt}]
            )
            title = response.content[0].text if response.content else "New Conversation"
            
        elif model == "chatgpt":
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": title_prompt}],
                max_tokens=25,
                temperature=0.3
            )
            title = response['choices'][0]['message']['content'] if 'choices' in response else "New Conversation"
            
        elif model == "gemini":
            # More specific prompt for Gemini with clear instructions
            gemini_prompt = title_prompt + f"Title: {message}" 
            response = genai_model.generate_content(
                gemini_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 25,
                    "candidate_count": 1
                }
            )
            title = response.text.strip() if response and response.text else "New Conversation"         
        else:
            title = "New Conversation"          
        # Clean up the title more aggressively
        title = title.strip().strip('"').strip("'").strip()  
        # Remove common prefixes that models might add
        prefixes_to_remove = ["Title:", "title:", "TITLE:", "Chat:", "Conversation:", "Message:"]
        for prefix in prefixes_to_remove:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()  
        # Remove markdown formatting and extra characters
        title = title.replace('*', '').replace('-', '').replace('#', '').replace('_', '')   
        # Remove extra whitespace and newlines
        title = ' '.join(title.split())
        # Limit length and return
        return title[:30] if title else "New Conversation"
    except Exception as e:
        print(f"[Title Generation Error] {str(e)}")
        return "New Conversation"


def filter_conversation_history_for_model(chat_history: List[Dict], current_model: str, current_question: str) -> List[Dict]:
    """
    Filter conversation history to provide context while avoiding duplicate responses from the same model.
    
    Strategy:
    1. Include conversations from OTHER models to provide context
    2. Include the current model's DIFFERENT conversations (different questions)
    3. Exclude the current model's responses to the SAME question to avoid repetition
    """
    if not chat_history:
        return []
    
    filtered_history = []
    current_question_lower = current_question.strip().lower()
    
    for chat in chat_history:
        user_message = chat.get("user", "").strip()
        user_message_lower = user_message.lower()
        chat_model = chat.get("model", "").lower()
        
        # Skip empty messages
        if not user_message or not chat.get("assistant", "").strip():
            continue
            
        # Include if it's from a different model (provides context from other AI perspectives)
        if chat_model and chat_model != current_model.lower():
            filtered_history.append(chat)
        # Include if it's from the same model but for a DIFFERENT question
        elif chat_model == current_model.lower() and user_message_lower != current_question_lower:
            filtered_history.append(chat)
        # Skip if it's from the same model answering the same question (avoid repetition)
        
    return filtered_history


def build_conversation_messages(chat_history: List[Dict], current_input: str, current_model: str) -> List[Dict]:
    """
    Build messages array for API calls with smart filtering to avoid repetitive responses.
    """
    messages = []
    
    # Filter chat history to provide context while avoiding repetition
    filtered_history = filter_conversation_history_for_model(chat_history, current_model, current_input)
    
    # Add filtered conversations to messages
    for chat in filtered_history:
        user_message = chat.get("user", "").strip()
        assistant_response = chat.get("assistant", "").strip()
        model_used = chat.get("model", "unknown")
        
        if user_message and assistant_response:
            messages.append({"role": "user", "content": user_message})
            # Add response without model attribution to keep it natural
            messages.append({"role": "assistant", "content": assistant_response})
    
    # Add current user input
    messages.append({"role": "user", "content": current_input})
    
    return messages


def generate_response(model: str, input_text: str, chat_history: List[Dict] = None) -> Dict:
    if chat_history is None:
        chat_history = []
    try:
        # Title generation - only generate new title if no history exists
        if not chat_history:
            conversation_title = generate_title_from_message(input_text, model)
        else:
            # Use existing title if available, otherwise generate new one
            conversation_title = getattr(chat_history, 'title', None) or generate_title_from_message(chat_history[0].get("user", input_text), model)
        
        # Build messages with smart filtering to avoid repetitive responses
        messages = build_conversation_messages(chat_history, input_text, model)

        # Handle responses for different models
        if model == "claude":
            try:
                retries = 3
                for attempt in range(retries):
                    try:
                        response_obj = anthropic_client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=2500,
                            temperature=0.7,
                            system=SYSTEM_PROMPTS_CLAUDE,
                            messages=messages
                        )
                        if response_obj and response_obj.content:
                            response = response_obj.content[0].text
                            break
                        else:
                            response = "No response generated."
                            break
                    except Exception as retry_e:
                        if attempt == retries - 1:  # Last attempt
                            raise retry_e
                        time.sleep(1)  # Wait before retry
                else:
                    response = "Claude is temporarily unavailable. Please try again later."
            except ClaudeAuthError:
                return {"content": "Invalid Claude API key.", "title": "Error"}
            except Exception as e:
                if 'overloaded_error' in str(e).lower():
                    response = "Claude is temporarily overloaded. Please try again shortly."
                else:
                    response = f"Claude error: {str(e)}"
            
        # ChatGPT
        elif model == "chatgpt":
            try:
                # Add system message at the beginning
                chatgpt_messages = [{"role": "system", "content": SYSTEM_PROMPTS_CHATGPT}] + messages
                response_obj = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    max_tokens=2500,
                    temperature=0.7,
                    messages=chatgpt_messages
                )
                response = response_obj['choices'][0]['message']['content'] if 'choices' in response_obj else "No response generated."
            except OpenAIAuthError:
                return {"content": "Invalid OpenAI API key.", "title": "Error"}
            except Exception as e:
                return {"content": f"ChatGPT error: {str(e)}", "title": "Error"}

        # Gemini
        elif model == "gemini":
            try:
                # Build conversation context for Gemini
                conversation_context = SYSTEM_PROMPTS_GEMINI + "\n\n"
                
                # Add filtered chat history (excluding system messages)
                user_assistant_messages = [msg for msg in messages if msg["role"] != "system"]
                for i in range(0, len(user_assistant_messages) - 1, 2):  # Process in pairs
                    if i + 1 < len(user_assistant_messages):
                        user_msg = user_assistant_messages[i]
                        assistant_msg = user_assistant_messages[i + 1]
                        if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                            conversation_context += f"User: {user_msg['content']}\nAssistant: {assistant_msg['content']}\n\n"
                
                # Add current input (last message should be user input)
                last_message = user_assistant_messages[-1]
                if last_message["role"] == "user":
                    conversation_context += f"User: {last_message['content']}\nAssistant:"
                
                response_obj = genai_model.generate_content(
                    conversation_context,
                    generation_config={"temperature": 0.7, "max_output_tokens": 2500}
                )
                response = response_obj.text if response_obj else "No response generated."
            except Exception as e:
                return {"content": f"Gemini error: {str(e)}", "title": "Error"}
        else:
            return {"content": f"Unsupported model: {model}", "title": "Error"}     
        
        return {"content": response, "title": conversation_title}

    except Exception as e:
        print(f"[generate_response Error] {str(e)}")
        return {"content": f"Unexpected error: {str(e)}", "title": "Error"}


if __name__ == "__main__":
    model = "gemini"  # Change to "claude", "chatgpt", or "gemini" as needed
    input_text = "what did we talk in our conversation?"

    # Simulate backend-provided chat history
    chat_history = [
  {
    "user": "How will be a public speaker?",
    "assistant": "To become an effective public speaker, you can follow several steps to develop your skills and confidence. Here's a structured guide:\n\n### 1. Understand Your Audience\n- **Research**: Learn about the interests, age, and background of your audience.\n- **Tailor Content**: Adapt your speech to be relevant and engaging for your audience.\n\n### 2. Master the Subject\n- **In-depth Knowledge**: Ensure you have a thorough understanding of the topic.\n- **Continuous Learning**: Stay updated with new information and perspectives.\n\n### 3. Practice Regularly\n- **Rehearse**: Practice your speech multiple times.\n- **Simulate Conditions**: Try to practice in conditions similar to those you will face during the actual speech.\n\n### 4. Develop Public Speaking Skills\n- **Clarity and Pace**: Work on speaking clearly and at a pace that is comfortable for the audience to follow.\n- **Body Language**: Use gestures and facial expressions to convey your message more effectively.\n- **Voice Modulation**: Vary your pitch and tone to keep the audience engaged.\n\n### 5. Manage Anxiety\n- **Preparation**: Being well-prepared can reduce nervousness.\n- **Breathing Techniques**: Use deep breathing exercises to calm nerves before and during your speech.\n- **Positive Visualization**: Imagine a successful speaking event to boost confidence.\n\n### 6. Gather and Use Feedback\n- **Ask for Feedback**: Get opinions from peers or mentors and use it to improve.\n- **Record Your Speeches**: Watching recordings can help you see areas for improvement.\n\n### 7. Engage with Your Audience\n- **Interactivity**: Include questions or prompts to make the speech interactive.\n- **Eye Contact**: Maintain eye contact with various members of the audience to create a connection.\n\n### 8. Use Visual Aids Wisely\n- **Supportive Visuals**: Use slides, videos, or physical objects to support and enhance your message.\n- **Don't Overdo It**: Ensure that visuals complement the speech rather than distract from it.\n\n### 9. Be Adaptable\n- **Adapt on the Fly**: Be ready to adjust your speech based on the audience's reactions and feedback during the event.\n- **Handling Unexpected Issues**: Prepare for potential issues like technical difficulties or interruptions.\n\n### 10. Reflect and Improve\n- **Self-Review**: After each speech, reflect on what went well and what could be improved.\n- **Seek Opportunities**: Regularly look for new opportunities to speak publicly to continue honing your skills.\n\nBy systematically working through these steps, you can develop into a knowledgeable and engaging public speaker over time.",
    "model": "chatgpt"
  },
  {
    "user": "How will be a public speaker?",
    "assistant": "To become an effective public speaker, you can follow several steps to develop your skills and confidence. Here's a structured guide:\n\n### 1. Understand Your Audience\n- **Research**: Learn about the interests, age, and background of your audience.\n- **Tailor Content**: Adapt your speech to be relevant and engaging for your audience.\n\n### 2. Master the Subject\n- **In-depth Knowledge**: Ensure you have a thorough understanding of the topic.\n- **Continuous Learning**: Stay updated with new information and perspectives.\n\n### 3. Practice Regularly\n- **Rehearse**: Practice your speech multiple times.\n- **Simulate Conditions**: Try to practice in conditions similar to those you will face during the actual speech.\n\n### 4. Develop Public Speaking Skills\n- **Clarity and Pace**: Work on speaking clearly and at a pace that is comfortable for the audience to follow.\n- **Body Language**: Use gestures and facial expressions to convey your message more effectively.\n- **Voice Modulation**: Vary your pitch and tone to keep the audience engaged.\n\n### 5. Manage Anxiety\n- **Preparation**: Being well-prepared can reduce nervousness.\n- **Breathing Techniques**: Use deep breathing exercises to calm nerves before and during your speech.\n- **Positive Visualization**: Imagine a successful speaking event to boost confidence.\n\n### 6. Gather and Use Feedback\n- **Ask for Feedback**: Get opinions from peers or mentors and use it to improve.\n- **Record Your Speeches**: Watching recordings can help you see areas for improvement.\n\n### 7. Engage with Your Audience\n- **Interactivity**: Include questions or prompts to make the speech interactive.\n- **Eye Contact**: Maintain eye contact with various members of the audience to create a connection.\n\n### 8. Use Visual Aids Wisely\n- **Supportive Visuals**: Use slides, videos, or physical objects to support and enhance your message.\n- **Don't Overdo It**: Ensure that visuals complement the speech rather than distract from it.\n\n### 9. Be Adaptable\n- **Adapt on the Fly**: Be ready to adjust your speech based on the audience's reactions and feedback during the event.\n- **Handling Unexpected Issues**: Prepare for potential issues like technical difficulties or interruptions.\n\n### 10. Reflect and Improve\n- **Self-Review**: After each speech, reflect on what went well and what could be improved.\n- **Seek Opportunities**: Regularly look for new opportunities to speak publicly to continue honing your skills.\n\nBy systematically working through these steps, you can develop into a knowledgeable and engaging public speaker over time.",
    "model": "claude"
  }
] 
    result = generate_response(model, input_text, chat_history)
    print(f"Response: {result}")