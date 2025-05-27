"""
Cache service for managing application caching
"""
from typing import Optional, Any, Union
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
import diskcache
import redis.asyncio as redis
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class CacheService:
    """Unified cache service supporting Redis and disk cache"""
    
    def __init__(self):
        self.backend = None
        self.cache_type = "memory"
        self._initialize_cache()
        
    def _initialize_cache(self):
        """Initialize cache backend"""
        if settings.REDIS_URL:
            try:
                self.backend = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True
                )
                self.cache_type = "redis"
                logger.info("Using Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._fallback_to_disk()
        else:
            self._fallback_to_disk()
            
    def _fallback_to_disk(self):
        """Fallback to disk cache"""
        cache_dir = settings.CACHE_DIR / "disk_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = diskcache.Cache(str(cache_dir))
        self.cache_type = "disk"
        logger.info("Using disk cache")
        
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from prefix and arguments"""
        key_data = f"{prefix}:"
        
        # Add positional arguments
        for arg in args:
            key_data += f"{str(arg)}:"
            
        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            key_data += f"{k}={str(v)}:"
            
        # Hash if too long
        if len(key_data) > 200:
            return f"{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"
            
        return key_data.rstrip(":")
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.cache_type == "redis":
                value = await self.backend.get(key)
                if value:
                    return json.loads(value)
            else:
                return self.backend.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
            
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL (seconds)"""
        try:
            if self.cache_type == "redis":
                serialized = json.dumps(value, default=str)
                if ttl:
                    await self.backend.setex(key, ttl, serialized)
                else:
                    await self.backend.set(key, serialized)
            else:
                self.backend.set(key, value, expire=ttl)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.cache_type == "redis":
                await self.backend.delete(key)
            else:
                if key in self.backend:
                    del self.backend[key]
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
            
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        count = 0
        try:
            if self.cache_type == "redis":
                cursor = 0
                while True:
                    cursor, keys = await self.backend.scan(
                        cursor, match=pattern, count=100
                    )
                    if keys:
                        await self.backend.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            else:
                # Disk cache doesn't support pattern matching
                # Clear all keys starting with pattern
                for key in list(self.backend.keys()):
                    if key.startswith(pattern.replace('*', '')):
                        del self.backend[key]
                        count += 1
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            
        return count
        
    async def get_or_set(
        self,
        key: str,
        factory_fn,
        ttl: Optional[int] = None
    ) -> Any:
        """Get from cache or compute and set"""
        value = await self.get(key)
        if value is not None:
            return value
            
        # Compute value
        if asyncio.iscoroutinefunction(factory_fn):
            value = await factory_fn()
        else:
            value = factory_fn()
            
        # Cache it
        await self.set(key, value, ttl)
        return value
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if self.cache_type == "redis":
                return await self.backend.exists(key) > 0
            else:
                return key in self.backend
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in cache"""
        try:
            if self.cache_type == "redis":
                return await self.backend.incrby(key, amount)
            else:
                current = self.backend.get(key, 0)
                new_value = current + amount
                self.backend.set(key, new_value)
                return new_value
        except Exception as e:
            logger.error(f"Cache increment error: {e}")
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for existing key"""
        try:
            if self.cache_type == "redis":
                return await self.backend.expire(key, ttl)
            else:
                # For disk cache, we need to get and reset with expiration
                value = self.backend.get(key)
                if value is not None:
                    self.backend.set(key, value, expire=ttl)
                    return True
                return False
        except Exception as e:
            logger.error(f"Cache expire error: {e}")
            return False
    
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        try:
            if self.cache_type == "redis":
                info = await self.backend.info()
                return {
                    "type": "redis",
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "0B"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
            else:
                return {
                    "type": "disk",
                    "size": len(self.backend),
                    "volume": self.backend.volume(),
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"type": self.cache_type, "error": str(e)}

# Singleton instance
cache_service = CacheService()