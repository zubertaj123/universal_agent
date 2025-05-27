#!/usr/bin/env python3
"""
Populate database with sample data
"""
import asyncio
import sys
from datetime import datetime, timedelta
import random
sys.path.append('.')

from app.core.database import init_db, async_session, Base
from app.models.user import User
from app.models.conversation import CallRecord, Claim, CallStatus
from app.utils.logger import setup_logger
import uuid

logger = setup_logger(__name__)

async def create_sample_users():
    """Create sample users"""
    users = [
        {
            "id": str(uuid.uuid4()),
            "phone_number": "+1234567890",
            "email": "john.doe@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "account_number": "ACC001",
        },
        {
            "id": str(uuid.uuid4()),
            "phone_number": "+0987654321",
            "email": "jane.smith@example.com",
            "first_name": "Jane",
            "last_name": "Smith",
            "account_number": "ACC002",
        },
        {
            "id": str(uuid.uuid4()),
            "phone_number": "+1122334455",
            "email": "bob.wilson@example.com",
            "first_name": "Bob",
            "last_name": "Wilson",
            "account_number": "ACC003",
        },
    ]
    
    async with async_session() as session:
        for user_data in users:
            user = User(**user_data)
            session.add(user)
        
        await session.commit()
        logger.info(f"Created {len(users)} sample users")
        
    return users

async def create_sample_calls(users):
    """Create sample call records"""
    call_records = []
    
    async with async_session() as session:
        for i, user in enumerate(users):
            # Create 2-3 calls per user
            for j in range(random.randint(2, 3)):
                call = CallRecord(
                    id=str(uuid.uuid4()),
                    session_id=f"session-{i}-{j}",
                    status=random.choice(list(CallStatus)),
                    started_at=datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                    duration=random.randint(60, 600),
                    customer_phone=user["phone_number"],
                    customer_email=user["email"],
                    customer_id=user["id"],
                    transcript=[
                        {"role": "agent", "content": "Hello, how can I help you today?"},
                        {"role": "user", "content": "I need to file a claim"},
                        {"role": "agent", "content": "I'd be happy to help you with that."},
                    ],
                    summary={
                        "main_issue": "Insurance claim",
                        "resolution": "Claim filed successfully",
                        "follow_up_needed": False
                    },
                    sentiment="positive" if random.random() > 0.3 else "neutral"
                )
                
                session.add(call)
                call_records.append(call)
                
        await session.commit()
        logger.info(f"Created {len(call_records)} sample call records")
        
    return call_records

async def create_sample_claims(call_records):
    """Create sample claims"""
    claim_types = ["auto", "home", "health", "life"]
    
    async with async_session() as session:
        claims_created = 0
        
        for call in call_records:
            # 70% chance of having a claim
            if random.random() < 0.7:
                claim = Claim(
                    id=f"CLM-{datetime.utcnow().strftime('%Y%m%d')}-{claims_created:04d}",
                    call_id=call.id,
                    claim_type=random.choice(claim_types),
                    status=random.choice(["open", "in_progress", "closed"]),
                    description=f"Sample {random.choice(claim_types)} claim",
                    incident_date=datetime.utcnow() - timedelta(days=random.randint(1, 10)),
                    data={
                        "amount": random.randint(100, 10000),
                        "location": "Sample location",
                        "details": "Sample claim details"
                    }
                )
                
                session.add(claim)
                claims_created += 1
                
        await session.commit()
        logger.info(f"Created {claims_created} sample claims")

async def main():
    """Populate database with sample data"""
    logger.info("Starting database population...")
    
    # Initialize database
    await init_db()
    
    # Create sample data
    users = await create_sample_users()
    call_records = await create_sample_calls(users)
    await create_sample_claims(call_records)
    
    logger.info("Database population completed!")

if __name__ == "__main__":
    asyncio.run(main())