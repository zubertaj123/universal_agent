"""
User model for customers
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer
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
    
    def __repr__(self):
        return f"<User(id='{self.id}', name='{self.first_name} {self.last_name}', phone='{self.phone_number}')>"
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            "id": self.id,
            "phone_number": self.phone_number,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "account_number": self.account_number,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "preferred_language": self.preferred_language,
            "preferences": self.preferences,
            "call_count": self.call_count,
            "last_call_date": self.last_call_date.isoformat() if self.last_call_date else None
        }