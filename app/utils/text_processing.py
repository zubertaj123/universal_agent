"""
Text processing utilities
"""
import re
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from collections import Counter
from datetime import datetime, timedelta
import json
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")

class TextProcessor:
    """Advanced text processing and analysis"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            logger.warning("NLTK resources not available, using basic processing")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'])
            self.lemmatizer = None
        
        # Insurance and call center specific terms
        self.domain_keywords = {
            'insurance': ['insurance', 'policy', 'coverage', 'premium', 'deductible', 'claim', 'liability', 'comprehensive'],
            'auto': ['car', 'vehicle', 'auto', 'automobile', 'accident', 'collision', 'damage', 'repair'],
            'home': ['home', 'house', 'property', 'dwelling', 'fire', 'flood', 'theft', 'burglary'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'treatment', 'medication', 'illness'],
            'life': ['life', 'death', 'beneficiary', 'term', 'whole', 'universal'],
            'service': ['help', 'assist', 'support', 'service', 'agent', 'representative', 'customer']
        }
        
        # Emotion keywords
        self.emotion_keywords = {
            'angry': ['angry', 'mad', 'furious', 'enraged', 'livid', 'irate', 'outraged', 'pissed'],
            'frustrated': ['frustrated', 'annoyed', 'irritated', 'aggravated', 'fed up', 'sick of'],
            'worried': ['worried', 'concerned', 'anxious', 'nervous', 'scared', 'afraid', 'panicked'],
            'confused': ['confused', 'lost', 'unclear', 'puzzled', 'bewildered', "don't understand"],
            'satisfied': ['satisfied', 'happy', 'pleased', 'content', 'glad', 'delighted'],
            'urgent': ['urgent', 'emergency', 'asap', 'immediately', 'right now', 'critical']
        }
        
        # Intent patterns
        self.intent_patterns = {
            'create_claim': [
                r'file\s+a?\s*claim', r'report\s+an?\s*(accident|incident)', r'make\s+a?\s*claim',
                r'submit\s+a?\s*claim', r'new\s+claim', r'start\s+a?\s*claim'
            ],
            'check_status': [
                r'check\s+(status|progress)', r'where\s+is\s+my', r'status\s+of', r'update\s+on',
                r'what.*happening\s+with', r'track\s+my', r'follow\s+up'
            ],
            'schedule': [
                r'schedule\s+an?\s*appointment', r'book\s+an?\s*appointment', r'set\s+up\s+a?\s*meeting',
                r'arrange\s+a?\s*time', r'when\s+can\s+I', r'available\s+times'
            ],
            'complaint': [
                r'complain', r'not\s+happy', r'dissatisfied', r'poor\s+service',
                r'terrible', r'awful', r'worst', r'unacceptable'
            ],
            'information': [
                r'what\s+is', r'how\s+do\s+I', r'can\s+you\s+explain', r'tell\s+me\s+about',
                r'information\s+about', r'details\s+on', r'help\s+me\s+understand'
            ],
            'transfer': [
                r'speak\s+to\s+a?\s*(human|person|agent)', r'transfer\s+me', r'talk\s+to\s+someone',
                r'human\s+agent', r'real\s+person', r'manager'
            ],
            'cancel': [
                r'cancel\s+my', r'discontinue', r'stop\s+my', r'end\s+my', r'terminate'
            ]
        }

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separator: str = " "
) -> List[str]:
    """Split text into overlapping chunks"""
    if not text or chunk_size <= 0:
        return []
    
    words = text.split(separator)
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = separator.join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Stop if we've reached the end
        if i + chunk_size >= len(words):
            break
            
    return chunks

def extract_phone_number(text: str) -> List[str]:
    """Extract phone numbers from text using comprehensive patterns"""
    patterns = [
        # US phone numbers
        r'\+?1?[-.\s]?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
        # International format
        r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        # Various separators
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        # With parentheses
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
        # Extensions
        r'\d{10,}\s*(?:ext?\.?\s*\d+)?'
    ]
    
    phone_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1] if len(match) > 1 else ""
            
            # Remove extra characters and normalize
            clean_number = re.sub(r'[^\d+]', '', match)
            if len(clean_number) >= 10:
                phone_numbers.append(match.strip())
    
    return list(set(phone_numbers))  # Remove duplicates

def extract_email(text: str) -> List[str]:
    """Extract email addresses from text"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return list(set(emails))  # Remove duplicates

def extract_dates(text: str) -> List[str]:
    """Extract dates from text using multiple patterns"""
    patterns = [
        # MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY
        r'\b(?:0?[1-9]|1[0-2])[/-.](?:0?[1-9]|[12]\d|3[01])[/-.](?:19|20)?\d{2}\b',
        # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        r'\b(?:0?[1-9]|[12]\d|3[01])[/-.](?:0?[1-9]|1[0-2])[/-.](?:19|20)?\d{2}\b',
        # YYYY-MM-DD, YYYY/MM/DD
        r'\b(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])\b',
        # Month DD, YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:0?[1-9]|[12]\d|3[01]),?\s+(?:19|20)?\d{2}\b',
        # DD Month YYYY
        r'\b(?:0?[1-9]|[12]\d|3[01])\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)?\d{2}\b',
        # Today, yesterday, tomorrow
        r'\b(?:today|yesterday|tomorrow)\b',
        # Last/next week/month/year
        r'\b(?:last|next)\s+(?:week|month|year)\b',
        # Days of week
        r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        # Relative dates
        r'\b(?:\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b',
        r'\bin\s+\d+\s+(?:days?|weeks?|months?|years?)\b'
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return list(set(dates))  # Remove duplicates

def extract_policy_numbers(text: str) -> List[str]:
    """Extract insurance policy numbers"""
    patterns = [
        r'\b[A-Z]{2,4}\d{6,12}\b',  # Standard format
        r'\b\d{8,15}\b',  # Numeric only
        r'\b[A-Z]\d{7,11}\b',  # Letter + numbers
        r'\bpolicy\s*#?\s*([A-Z0-9-]{6,15})\b'  # With policy keyword
    ]
    
    policy_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        policy_numbers.extend(matches)
    
    return list(set(policy_numbers))

def extract_claim_numbers(text: str) -> List[str]:
    """Extract claim numbers"""
    patterns = [
        r'\bCLM[-_]?\d{6,12}\b',
        r'\bclaim\s*#?\s*([A-Z0-9-]{6,15})\b',
        r'\b[A-Z]{3}\d{8,12}\b'
    ]
    
    claim_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if isinstance(matches[0], tuple) if matches else False:
            matches = [m[0] for m in matches]
        claim_numbers.extend(matches)
    
    return list(set(claim_numbers))

def extract_monetary_amounts(text: str) -> List[Dict[str, Any]]:
    """Extract monetary amounts with context"""
    patterns = [
        r'\$\s*(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?',  # $1,000.00
        r'(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?\s*dollars?',  # 1000 dollars
        r'(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?\s*USD',  # 1000 USD
    ]
    
    amounts = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            amount_text = match.group()
            # Extract numeric value
            numeric_match = re.search(r'[\d,]+\.?\d*', amount_text)
            if numeric_match:
                try:
                    value = float(numeric_match.group().replace(',', ''))
                    amounts.append({
                        'text': amount_text,
                        'value': value,
                        'position': match.span(),
                        'context': text[max(0, match.start()-20):match.end()+20]
                    })
                except ValueError:
                    continue
    
    return amounts

def clean_transcript(text: str) -> str:
    """Clean and normalize transcript text"""
    if not text:
        return ""
    
    # Remove multiple spaces and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common ASR errors and colloquialisms
    corrections = {
        r'\bgonna\b': 'going to',
        r'\bwanna\b': 'want to',
        r'\bgotta\b': 'got to',
        r'\blemme\b': 'let me',
        r'\byeah\b': 'yes',
        r'\bnah\b': 'no',
        r'\bokay\b': 'OK',
        r'\buh\b': '',
        r'\bum\b': '',
        r'\ber\b': '',
        r'\buh-huh\b': 'yes',
        r'\buh-uh\b': 'no'
    }
    
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Remove filler words at sentence boundaries
    text = re.sub(r'\b(?:well|so|like|you know)\s*,?\s*', '', text, flags=re.IGNORECASE)
    
    # Fix punctuation
    text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after punctuation
    
    # Capitalize first letter of sentences
    sentences = re.split(r'([.!?])', text)
    cleaned_sentences = []
    for i, sentence in enumerate(sentences):
        if i % 2 == 0:  # Actual sentence content
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
        cleaned_sentences.append(sentence)
    
    text = ''.join(cleaned_sentences)
    
    return text.strip()

def extract_intent(text: str) -> Dict[str, Any]:
    """Extract intent from text using pattern matching and keywords"""
    processor = TextProcessor()
    text_lower = text.lower()
    
    detected_intents = []
    confidence_scores = {}
    
    # Pattern-based intent detection
    for intent, patterns in processor.intent_patterns.items():
        pattern_matches = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                pattern_matches += 1
        
        if pattern_matches > 0:
            confidence = pattern_matches / len(patterns)
            detected_intents.append(intent)
            confidence_scores[intent] = confidence
    
    # Keyword-based enhancement
    for category, keywords in processor.domain_keywords.items():
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        if keyword_matches > 0:
            # Boost related intents
            if category == 'insurance' and 'create_claim' in confidence_scores:
                confidence_scores['create_claim'] += 0.1
            elif category == 'service' and 'information' in confidence_scores:
                confidence_scores['information'] += 0.1
    
    # Sort by confidence
    sorted_intents = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "intents": detected_intents,
        "scores": confidence_scores,
        "primary_intent": sorted_intents[0][0] if sorted_intents else "unknown",
        "confidence": sorted_intents[0][1] if sorted_intents else 0.0,
        "all_ranked": sorted_intents
    }

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment and emotional content"""
    processor = TextProcessor()
    text_lower = text.lower()
    
    emotion_scores = {}
    
    # Count emotion keywords
    for emotion, keywords in processor.emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score / len(keywords)
    
    # Analyze linguistic indicators
    negative_indicators = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor']
    positive_indicators = ['yes', 'good', 'great', 'excellent', 'perfect', 'amazing', 'wonderful']
    
    negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
    positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
    
    # Determine overall sentiment
    if emotion_scores:
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        primary_emotion = dominant_emotion[0]
        emotion_confidence = dominant_emotion[1]
    else:
        primary_emotion = "neutral"
        emotion_confidence = 0.0
    
    # Overall sentiment based on emotions and indicators
    if primary_emotion in ['angry', 'frustrated', 'worried'] or negative_count > positive_count:
        overall_sentiment = "negative"
    elif primary_emotion in ['satisfied'] or positive_count > negative_count:
        overall_sentiment = "positive"
    else:
        overall_sentiment = "neutral"
    
    return {
        "overall_sentiment": overall_sentiment,
        "primary_emotion": primary_emotion,
        "emotion_confidence": emotion_confidence,
        "emotion_scores": emotion_scores,
        "negative_indicators": negative_count,
        "positive_indicators": positive_count,
        "urgency_level": "high" if primary_emotion == "urgent" else "normal"
    }

def extract_entities(text: str) -> Dict[str, List[Any]]:
    """Extract various entities from text"""
    entities = {
        "phone_numbers": extract_phone_number(text),
        "email_addresses": extract_email(text),
        "dates": extract_dates(text),
        "policy_numbers": extract_policy_numbers(text),
        "claim_numbers": extract_claim_numbers(text),
        "monetary_amounts": extract_monetary_amounts(text),
        "names": extract_names(text),
        "addresses": extract_addresses(text)
    }
    
    return entities

def extract_names(text: str) -> List[str]:
    """Extract potential person names (basic implementation)"""
    # This is a simplified approach - in production, you'd use NER
    patterns = [
        r'\bMr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\bMrs\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\bMs\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\bDr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r"\bI'm\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    ]
    
    names = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        names.extend(matches)
    
    return list(set(names))

def extract_addresses(text: str) -> List[str]:
    """Extract potential addresses (basic implementation)"""
    patterns = [
        r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Circle|Cir|Place|Pl)',
        r'\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?'
    ]
    
    addresses = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        addresses.extend(matches)
    
    return list(set(addresses))

def summarize_conversation(messages: List[Dict[str, str]]) -> str:
    """Generate conversation summary with key insights"""
    if not messages:
        return "No conversation to summarize."
    
    # Separate user and agent messages
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    agent_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    
    # Extract key information from all messages
    all_text = " ".join([msg.get("content", "") for msg in messages])
    
    # Extract entities
    entities = extract_entities(all_text)
    
    # Analyze sentiment progression
    user_sentiments = []
    for msg in user_messages:
        sentiment = analyze_sentiment(msg.get("content", ""))
        user_sentiments.append(sentiment["overall_sentiment"])
    
    # Extract intents
    user_intents = []
    for msg in user_messages:
        intent_info = extract_intent(msg.get("content", ""))
        if intent_info["primary_intent"] != "unknown":
            user_intents.append(intent_info["primary_intent"])
    
    # Build summary
    summary_parts = []
    
    # Basic stats
    summary_parts.append(f"Conversation with {len(messages)} total messages ({len(user_messages)} from customer, {len(agent_messages)} from agent)")
    
    # Duration estimate
    if len(messages) >= 2:
        try:
            start_time = datetime.fromisoformat(messages[0].get("timestamp", ""))
            end_time = datetime.fromisoformat(messages[-1].get("timestamp", ""))
            duration = end_time - start_time
            summary_parts.append(f"Duration: approximately {duration.total_seconds() / 60:.1f} minutes")
        except:
            pass
    
    # Key intents
    if user_intents:
        unique_intents = list(set(user_intents))
        summary_parts.append(f"Primary customer intents: {', '.join(unique_intents)}")
    
    # Sentiment progression
    if user_sentiments:
        initial_sentiment = user_sentiments[0] if user_sentiments else "neutral"
        final_sentiment = user_sentiments[-1] if user_sentiments else "neutral"
        if initial_sentiment != final_sentiment:
            summary_parts.append(f"Customer sentiment changed from {initial_sentiment} to {final_sentiment}")
        else:
            summary_parts.append(f"Customer sentiment remained {final_sentiment}")
    
    # Key information extracted
    info_extracted = []
    if entities["phone_numbers"]:
        info_extracted.append(f"Phone: {entities['phone_numbers'][0]}")
    if entities["email_addresses"]:
        info_extracted.append(f"Email: {entities['email_addresses'][0]}")
    if entities["policy_numbers"]:
        info_extracted.append(f"Policy: {entities['policy_numbers'][0]}")
    if entities["claim_numbers"]:
        info_extracted.append(f"Claim: {entities['claim_numbers'][0]}")
    
    if info_extracted:
        summary_parts.append(f"Key information: {', '.join(info_extracted)}")
    
    # Issues or concerns
    concerns = []
    for msg in user_messages:
        sentiment = analyze_sentiment(msg.get("content", ""))
        if sentiment["primary_emotion"] in ["angry", "frustrated", "worried"]:
            concerns.append(sentiment["primary_emotion"])
    
    if concerns:
        unique_concerns = list(set(concerns))
        summary_parts.append(f"Customer emotions detected: {', '.join(unique_concerns)}")
    
    return ". ".join(summary_parts) + "."

def extract_keywords(text: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Extract important keywords with TF-IDF-like scoring"""
    processor = TextProcessor()
    
    if not text:
        return []
    
    # Tokenize and clean
    try:
        words = word_tokenize(text.lower())
    except:
        words = text.lower().split()
    
    # Remove punctuation and stop words
    words = [word for word in words if word not in string.punctuation and word not in processor.stop_words]
    
    # Apply lemmatization if available
    if processor.lemmatizer:
        try:
            words = [processor.lemmatizer.lemmatize(word) for word in words]
        except:
            pass
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Boost domain-specific terms
    for category, keywords in processor.domain_keywords.items():
        for keyword in keywords:
            if keyword in word_freq:
                word_freq[keyword] *= 2
    
    # Get top keywords
    top_keywords = word_freq.most_common(top_k)
    
    # Normalize scores
    max_freq = max(word_freq.values()) if word_freq else 1
    normalized_keywords = [(word, freq / max_freq) for word, freq in top_keywords]
    
    return normalized_keywords

def detect_urgency(text: str) -> Dict[str, Any]:
    """Detect urgency indicators in text"""
    urgency_indicators = {
        'emergency': ['emergency', 'urgent', 'critical', 'asap', 'immediately'],
        'time_sensitive': ['today', 'right now', 'as soon as possible', 'deadline', 'expires'],
        'distress': ['help', 'desperate', 'crisis', 'can\'t wait', 'need now'],
        'escalation': ['manager', 'supervisor', 'complaint', 'unacceptable', 'escalate']
    }
    
    text_lower = text.lower()
    detected_indicators = {}
    urgency_score = 0
    
    for category, indicators in urgency_indicators.items():
        matches = [indicator for indicator in indicators if indicator in text_lower]
        if matches:
            detected_indicators[category] = matches
            urgency_score += len(matches)
    
    # Determine urgency level
    if urgency_score == 0:
        urgency_level = "low"
    elif urgency_score <= 2:
        urgency_level = "medium"
    elif urgency_score <= 4:
        urgency_level = "high"
    else:
        urgency_level = "critical"
    
    return {
        "urgency_level": urgency_level,
        "urgency_score": urgency_score,
        "indicators": detected_indicators,
        "requires_immediate_attention": urgency_level in ["high", "critical"]
    }

def analyze_conversation_quality(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze conversation quality metrics"""
    if not messages:
        return {"error": "No messages to analyze"}
    
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    agent_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    
    # Calculate metrics
    metrics = {
        "total_messages": len(messages),
        "user_messages": len(user_messages),
        "agent_messages": len(agent_messages),
        "conversation_balance": len(user_messages) / len(messages) if messages else 0,
        "avg_user_message_length": 0,
        "avg_agent_message_length": 0,
        "politeness_score": 0,
        "clarity_score": 0,
        "resolution_indicators": 0
    }
    
    # Message length analysis
    if user_messages:
        user_lengths = [len(msg.get("content", "").split()) for msg in user_messages]
        metrics["avg_user_message_length"] = sum(user_lengths) / len(user_lengths)
    
    if agent_messages:
        agent_lengths = [len(msg.get("content", "").split()) for msg in agent_messages]
        metrics["avg_agent_message_length"] = sum(agent_lengths) / len(agent_lengths)
    
    # Politeness indicators
    polite_words = ['please', 'thank you', 'thanks', 'sorry', 'excuse me', 'appreciate']
    all_agent_text = " ".join([msg.get("content", "").lower() for msg in agent_messages])
    
    politeness_count = sum(1 for word in polite_words if word in all_agent_text)
    metrics["politeness_score"] = min(1.0, politeness_count / len(agent_messages)) if agent_messages else 0
    
    # Resolution indicators
    resolution_words = ['resolved', 'completed', 'done', 'finished', 'help', 'assist', 'solve']
    resolution_count = sum(1 for word in resolution_words if word in all_agent_text)
    metrics["resolution_indicators"] = resolution_count
    
    # Overall quality score
    balance_score = 1.0 - abs(0.4 - metrics["conversation_balance"])  # Ideal is ~40% user
    length_score = 1.0 if 10 <= metrics["avg_agent_message_length"] <= 30 else 0.5
    
    metrics["overall_quality_score"] = (
        balance_score * 0.3 +
        metrics["politeness_score"] * 0.3 +
        length_score * 0.2 +
        min(1.0, metrics["resolution_indicators"] / 5) * 0.2
    )
    
    return metrics