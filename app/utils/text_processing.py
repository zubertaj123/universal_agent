"""
Text processing utilities
"""
import re
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    logger.warning("Could not download NLTK data")

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separator: str = " "
) -> List[str]:
    """Split text into chunks with overlap"""
    words = text.split(separator)
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = separator.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
            
    return chunks

def extract_phone_number(text: str) -> List[str]:
    """Extract phone numbers from text"""
    # Simple regex for phone numbers
    pattern = r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3,5}[-\s\.]?[0-9]{3,5}'
    return re.findall(pattern, text)

def extract_email(text: str) -> List[str]:
    """Extract email addresses from text"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)

def extract_dates(text: str) -> List[str]:
    """Extract dates from text"""
    # Simple date patterns
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}\b',
    ]
    
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return dates

def clean_transcript(text: str) -> str:
    """Clean transcript text"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common ASR errors
    corrections = {
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "lemme": "let me",
    }
    
    for wrong, correct in corrections.items():
        text = re.sub(r'\b' + wrong + r'\b', correct, text, flags=re.IGNORECASE)
        
    return text.strip()

def extract_intent(text: str) -> Dict[str, Any]:
    """Extract intent from text (simplified)"""
    text_lower = text.lower()
    
    # Define intent patterns
    intents = {
        "create_claim": ["file a claim", "report", "accident", "damage", "incident"],
        "check_status": ["status", "update", "where is", "track", "progress"],
        "schedule": ["appointment", "schedule", "book", "arrange", "meeting"],
        "complaint": ["complaint", "unhappy", "dissatisfied", "problem", "issue"],
        "information": ["information", "how to", "what is", "explain", "tell me about"],
        "transfer": ["agent", "human", "person", "representative", "someone else"],
    }
    
    detected_intents = []
    confidence_scores = {}
    
    for intent, keywords in intents.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            detected_intents.append(intent)
            confidence_scores[intent] = score / len(keywords)
            
    return {
        "intents": detected_intents,
        "scores": confidence_scores,
        "primary_intent": detected_intents[0] if detected_intents else "unknown"
    }

def summarize_conversation(messages: List[Dict[str, str]]) -> str:
    """Generate conversation summary"""
    if not messages:
        return "No conversation to summarize."
        
    # Extract key points
    key_points = []
    
    for msg in messages:
        if msg.get("role") == "user":
            # Extract important information from user messages
            phones = extract_phone_number(msg["content"])
            emails = extract_email(msg["content"])
            dates = extract_dates(msg["content"])
            
            if phones or emails or dates:
                key_points.append({
                    "phones": phones,
                    "emails": emails,
                    "dates": dates,
                    "message": msg["content"][:100]
                })
                
    # Build summary
    summary = f"Conversation with {len(messages)} messages. "
    
    if key_points:
        summary += "Key information extracted: "
        for point in key_points:
            if point["phones"]:
                summary += f"Phone: {', '.join(point['phones'])}. "
            if point["emails"]:
                summary += f"Email: {', '.join(point['emails'])}. "
            if point["dates"]:
                summary += f"Date: {', '.join(point['dates'])}. "
                
    return summary