"""
User model for customers
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Boolean
from app.core.database import Base

class User(Base):
    """Customer/User model"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    phone_number = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    
    # Personal information
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    account_number = Column(String, unique=True, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Preferences
    preferred_language = Column(String, default="en")
    preferences = Column(JSON, default=dict)
    
    # History
    call_count = Column(Integer, default=0)
    last_call_date = Column(DateTime, nullable=True)