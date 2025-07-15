# Universal API Gateway Architecture

## Overview

The Universal API Gateway serves as the central entry point and orchestration layer between the Musical Spork frontend application and various backend services, including internal microservices and external internet-available services.

## Architecture Principles

### Design Goals

- **Single Entry Point**: Unified interface for all client-server communication
- **Service Abstraction**: Hide backend complexity from frontend clients
- **Protocol Translation**: Support multiple communication protocols (REST, GraphQL, WebSocket)
- **Security Enforcement**: Centralized authentication, authorization, and rate limiting
- **Observability**: Comprehensive logging, monitoring, and tracing
- **Scalability**: Horizontal scaling with load balancing
- **Resilience**: Circuit breakers, retries, and failover mechanisms

### Core Responsibilities

1. **Request Routing**: Intelligent routing based on path, headers, and content
2. **Authentication & Authorization**: JWT validation, RBAC, API key management
3. **Rate Limiting**: Per-user, per-service, and global rate controls
4. **Request/Response Transformation**: Data format conversion and enrichment
5. **Caching**: Response caching with TTL and invalidation strategies
6. **Monitoring**: Request metrics, error tracking, and performance analytics
7. **Security**: Input validation, CORS handling, and threat protection

## System Architecture

```text
┌─────────────────┐    ┌─────────────────────────────────┐    ┌─────────────────┐
│   Frontend      │    │        API Gateway              │    │   Backend       │
│   (React/TS)    │    │                                 │    │   Services      │
├─────────────────┤    ├─────────────────────────────────┤    ├─────────────────┤
│ • Web App       │◄──►│ ┌─────────────────────────────┐ │◄──►│ • User Service  │
│ • Mobile App    │    │ │      Load Balancer          │ │    │ • Auth Service  │
│ • Admin Panel   │    │ │   (NGINX/Kong/Traefik)      │ │    │ • Content API   │
└─────────────────┘    │ └─────────────────────────────┘ │    │ • Payment API   │
                       │ ┌─────────────────────────────┐ │    │ • Notification  │
┌─────────────────┐    │ │     Gateway Core            │ │    │ • Analytics     │
│   External      │    │ │  (Node.js/Python/Go)        │ │    └─────────────────┘
│   Services      │◄──►│ │                             │ │
├─────────────────┤    │ │ • Authentication            │ │    ┌─────────────────┐
│ • Payment APIs  │    │ │ • Authorization             │ │    │   External      │
│ • Social Auth   │    │ │ • Rate Limiting             │ │    │   Services      │
│ • Email Service │    │ │ • Request Routing           │ │    ├─────────────────┤
│ • SMS Provider  │    │ │ • Response Caching          │ │    │ • Stripe/PayPal │
│ • Analytics     │    │ │ • Protocol Translation      │ │    │ • SendGrid      │
└─────────────────┘    │ │ • Error Handling            │ │    │ • Google OAuth  │
                       │ └─────────────────────────────┘ │    │ • Twilio        │
┌─────────────────┐    │ ┌─────────────────────────────┐ │    │ • Analytics     │
│   Data Stores   │◄──►│ │     Supporting Services     │ │    └─────────────────┘
├─────────────────┤    │ │                             │ │
│ • Redis Cache   │    │ │ • Configuration Store       │ │    ┌─────────────────┐
│ • Rate Limits   │    │ │ • Service Discovery         │ │    │   Monitoring    │
│ • Session Store │    │ │ • Health Checks             │ │    │   & Logging     │
│ • API Keys      │    │ │ • Metrics Collection        │ │    ├─────────────────┤
└─────────────────┘    │ └─────────────────────────────┘ │    │ • Prometheus    │
                       └─────────────────────────────────┘    │ • Grafana       │
                                                              │ • ELK Stack     │
                                                              │ • Jaeger        │
                                                              └─────────────────┘
```

## Technology Stack

### Primary Implementation: Python/FastAPI

The API Gateway is built using **Python/FastAPI** as the core technology, providing high performance, excellent developer experience, and seamless integration with the Python ecosystem.

```python
from fastapi import FastAPI, Depends, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import redis.asyncio as redis
from typing import Optional, Dict, Any
import asyncio
from contextlib import asynccontextmanager

# Initialize FastAPI app with lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize connections and services
    await initialize_redis()
    await initialize_service_registry()
    yield
    # Shutdown: Clean up resources
    await cleanup_connections()

app = FastAPI(
    title="Musical Spork API Gateway",
    description="Universal API Gateway for Musical Spork microservices",
    version="1.0.0",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*.musical-spork.com", "localhost"])

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://app.musical-spork.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"]
)

# Service discovery and intelligent routing
@app.api_route("/api/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def gateway_proxy(
    service_name: str,
    path: str,
    request: Request,
    authenticated_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Intelligent routing to backend services with authentication,
    rate limiting, caching, and error handling.
    """
    return await route_request(service_name, path, request, authenticated_user)
```

### Key Technology Benefits

- **High Performance**: ASGI-based async framework with excellent throughput
- **Type Safety**: Full Pydantic integration for request/response validation
- **Auto Documentation**: OpenAPI/Swagger docs generated automatically
- **Python Ecosystem**: Easy integration with ML libraries, data processing tools
- **Developer Experience**: Excellent IDE support, debugging, and testing
- **Production Ready**: Built-in security features, monitoring, and deployment support

## Core Components

### 1. Authentication & Authorization Module

```python
# auth/middleware.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel
from typing import Optional, List
import redis.asyncio as redis
from datetime import datetime, timedelta
import os

# Pydantic models for user data
class User(BaseModel):
    id: str
    email: str
    roles: List[str]
    is_active: bool = True

class TokenData(BaseModel):
    user_id: Optional[str] = None
    scopes: List[str] = []

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Verify JWT token and return authenticated user.
    Includes token blacklist checking via Redis.
    """
    try:
        # Check if token is blacklisted
        is_blacklisted = await redis_client.get(f"blacklist:{credentials.credentials}")
        if is_blacklisted:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )

        # Decode and verify token
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

        # Get user from cache or database
        user = await get_user_by_id(user_id)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )

        return user

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def require_roles(required_roles: List[str]):
    """
    Dependency factory for role-based access control.
    Usage: Depends(require_roles(["admin", "user"]))
    """
    def role_checker(current_user: User = Depends(verify_token)) -> User:
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user

    return role_checker

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[User]:
    """
    Optional authentication - returns None if not authenticated.
    Used for endpoints that work with or without authentication.
    """
    try:
        return await verify_token(credentials)
    except HTTPException:
        return None

async def get_user_by_id(user_id: str) -> Optional[User]:
    """
    Get user from cache or database with Redis caching.
    """
    # Try cache first
    cached_user = await redis_client.get(f"user:{user_id}")
    if cached_user:
        return User.parse_raw(cached_user)

    # Fallback to database (implement based on your DB choice)
    # user = await database.fetch_user(user_id)
    # if user:
    #     await redis_client.setex(f"user:{user_id}", 300, user.json())
    #     return user

    return None
```

### 2. Service Discovery & Routing

```python
# routing/service_registry.py
from pydantic import BaseModel, HttpUrl
from typing import Dict, Optional, List
import httpx
import asyncio
from enum import Enum
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class ServiceConfig(BaseModel):
    name: str
    url: HttpUrl
    health_endpoint: str = "/health"
    timeout: float = 30.0
    retries: int = 3
    circuit_breaker: Dict = {
        "failure_threshold": 5,
        "recovery_timeout": 60,
        "success_threshold": 3
    }
    weight: int = 1  # For load balancing

class ServiceInstance(BaseModel):
    config: ServiceConfig
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    failure_count: int = 0
    circuit_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    last_failure: Optional[datetime] = None

class ServiceRegistry:
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None

    async def register_service(self, config: ServiceConfig):
        """Register a new service instance."""
        instance = ServiceInstance(config=config)

        if config.name not in self.services:
            self.services[config.name] = []

        self.services[config.name].append(instance)
        logger.info(f"Registered service: {config.name} at {config.url}")

        # Start health checking if not already running
        if self.health_check_task is None:
            self.health_check_task = asyncio.create_task(self._health_check_loop())

    async def get_healthy_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Get a healthy service instance using round-robin load balancing."""
        if service_name not in self.services:
            return None

        instances = self.services[service_name]
        healthy_instances = [
            instance for instance in instances
            if instance.status == ServiceStatus.HEALTHY and
               instance.circuit_state != CircuitBreakerState.OPEN
        ]

        if not healthy_instances:
            # Try half-open instances for recovery
            half_open_instances = [
                instance for instance in instances
                if instance.circuit_state == CircuitBreakerState.HALF_OPEN
            ]
            if half_open_instances:
                return half_open_instances[0]
            return None

        # Simple round-robin (in production, use weighted round-robin)
        return min(healthy_instances, key=lambda x: x.failure_count)

    async def _health_check_loop(self):
        """Continuous health checking of all registered services."""
        while True:
            tasks = []
            for service_name, instances in self.services.items():
                for instance in instances:
                    tasks.append(self._check_instance_health(instance))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(self.health_check_interval)

    async def _check_instance_health(self, instance: ServiceInstance):
        """Check health of a single service instance."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{instance.config.url}{instance.config.health_endpoint}"
                )

                if response.status_code == 200:
                    instance.status = ServiceStatus.HEALTHY
                    instance.failure_count = 0
                    instance.last_health_check = datetime.utcnow()

                    # Reset circuit breaker if in half-open state
                    if instance.circuit_state == CircuitBreakerState.HALF_OPEN:
                        instance.circuit_state = CircuitBreakerState.CLOSED
                else:
                    await self._handle_health_check_failure(instance)

        except Exception as e:
            logger.warning(f"Health check failed for {instance.config.name}: {e}")
            await self._handle_health_check_failure(instance)

    async def _handle_health_check_failure(self, instance: ServiceInstance):
        """Handle health check failure and circuit breaker logic."""
        instance.status = ServiceStatus.UNHEALTHY
        instance.failure_count += 1
        instance.last_failure = datetime.utcnow()

        # Circuit breaker logic
        threshold = instance.config.circuit_breaker["failure_threshold"]
        if instance.failure_count >= threshold:
            instance.circuit_state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened for {instance.config.name}")

        # Check for circuit breaker recovery
        if instance.circuit_state == CircuitBreakerState.OPEN:
            recovery_timeout = instance.config.circuit_breaker["recovery_timeout"]
            if (datetime.utcnow() - instance.last_failure).seconds > recovery_timeout:
                instance.circuit_state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker half-open for {instance.config.name}")

# Global service registry instance
service_registry = ServiceRegistry()

# Request routing with intelligent load balancing
async def route_request(
    service_name: str,
    path: str,
    request: Request,
    user: Optional[User] = None
) -> Response:
    """
    Route request to appropriate service with load balancing,
    circuit breaking, and retry logic.
    """
    instance = await service_registry.get_healthy_instance(service_name)

    if not instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service {service_name} is currently unavailable"
        )

    # Prepare request headers
    headers = dict(request.headers)
    if user:
        headers["X-User-ID"] = user.id
        headers["X-User-Roles"] = ",".join(user.roles)

    # Make request with retry logic
    for attempt in range(instance.config.retries + 1):
        try:
            async with httpx.AsyncClient(timeout=instance.config.timeout) as client:
                response = await client.request(
                    method=request.method,
                    url=f"{instance.config.url}/{path}",
                    params=request.query_params,
                    headers=headers,
                    content=await request.body()
                )

                # Success - reset failure count
                instance.failure_count = 0

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )

        except Exception as e:
            logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
            if attempt == instance.config.retries:
                # Final attempt failed
                await service_registry._handle_health_check_failure(instance)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to reach {service_name} service"
                )

            # Exponential backoff for retries
            await asyncio.sleep(2 ** attempt)
```

### 3. Rate Limiting & Throttling

```python
# middleware/rate_limiting.py
from fastapi import Request, HTTPException, status, Depends
import redis.asyncio as redis
from typing import Callable, Optional
import time
import json
from pydantic import BaseModel
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RateLimitStrategy(str, Enum):
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"

class RateLimitConfig(BaseModel):
    key_prefix: str
    window_seconds: int
    max_requests: int
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_limit: Optional[int] = None  # For token bucket

class RateLimitResult(BaseModel):
    allowed: bool
    current_requests: int
    max_requests: int
    reset_time: int
    retry_after: Optional[int] = None

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_rate_limit(
        self,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """
        Check rate limit using specified strategy.
        """
        if config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_check(key, config)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket_check(key, config)
        else:  # FIXED_WINDOW
            return await self._fixed_window_check(key, config)

    async def _sliding_window_check(
        self,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """
        Sliding window rate limiting using Redis sorted sets.
        """
        now = time.time()
        window_start = now - config.window_seconds

        pipe = self.redis.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(now): now})

        # Set expiration
        pipe.expire(key, config.window_seconds)

        results = await pipe.execute()
        current_count = results[1]

        allowed = current_count < config.max_requests
        reset_time = int(now + config.window_seconds)

        return RateLimitResult(
            allowed=allowed,
            current_requests=current_count,
            max_requests=config.max_requests,
            reset_time=reset_time,
            retry_after=1 if not allowed else None
        )

    async def _fixed_window_check(
        self,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """
        Fixed window rate limiting using Redis counters.
        """
        now = time.time()
        window = int(now // config.window_seconds)
        window_key = f"{key}:{window}"

        pipe = self.redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, config.window_seconds)
        results = await pipe.execute()

        current_count = results[0]
        allowed = current_count <= config.max_requests
        reset_time = int((window + 1) * config.window_seconds)

        return RateLimitResult(
            allowed=allowed,
            current_requests=current_count,
            max_requests=config.max_requests,
            reset_time=reset_time,
            retry_after=reset_time - int(now) if not allowed else None
        )

    async def _token_bucket_check(
        self,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """
        Token bucket rate limiting for burst handling.
        """
        now = time.time()
        bucket_data = await self.redis.get(key)

        if bucket_data:
            bucket = json.loads(bucket_data)
            last_refill = bucket.get("last_refill", now)
            tokens = bucket.get("tokens", config.max_requests)
        else:
            last_refill = now
            tokens = config.max_requests

        # Calculate tokens to add based on time elapsed
        time_passed = now - last_refill
        tokens_to_add = time_passed * (config.max_requests / config.window_seconds)
        tokens = min(config.max_requests, tokens + tokens_to_add)

        allowed = tokens >= 1
        if allowed:
            tokens -= 1

        # Save bucket state
        bucket_data = {
            "tokens": tokens,
            "last_refill": now
        }
        await self.redis.setex(key, config.window_seconds * 2, json.dumps(bucket_data))

        return RateLimitResult(
            allowed=allowed,
            current_requests=int(config.max_requests - tokens),
            max_requests=config.max_requests,
            reset_time=int(now + config.window_seconds),
            retry_after=1 if not allowed else None
        )

# Global rate limiter instance
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
rate_limiter = RateLimiter(redis_client)

# Rate limiting dependency factories
def create_rate_limit(config: RateLimitConfig):
    """
    Create a rate limiting dependency for FastAPI endpoints.
    """
    async def rate_limit_dependency(request: Request, user: Optional[User] = None):
        # Generate rate limit key based on user or IP
        if user:
            key = f"{config.key_prefix}:user:{user.id}"
        else:
            client_ip = request.client.host
            key = f"{config.key_prefix}:ip:{client_ip}"

        result = await rate_limiter.check_rate_limit(key, config)

        if not result.allowed:
            logger.warning(f"Rate limit exceeded for key: {key}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(result.max_requests),
                    "X-RateLimit-Remaining": str(max(0, result.max_requests - result.current_requests)),
                    "X-RateLimit-Reset": str(result.reset_time),
                    "Retry-After": str(result.retry_after) if result.retry_after else "1"
                }
            )

        return result

    return rate_limit_dependency

# Predefined rate limiters for different use cases
user_rate_limit = create_rate_limit(RateLimitConfig(
    key_prefix="api",
    window_seconds=900,  # 15 minutes
    max_requests=1000,
    strategy=RateLimitStrategy.SLIDING_WINDOW
))

public_rate_limit = create_rate_limit(RateLimitConfig(
    key_prefix="public",
    window_seconds=900,  # 15 minutes
    max_requests=100,
    strategy=RateLimitStrategy.SLIDING_WINDOW
))

burst_rate_limit = create_rate_limit(RateLimitConfig(
    key_prefix="burst",
    window_seconds=60,  # 1 minute
    max_requests=50,
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    burst_limit=100
))

# Usage in endpoints
@app.get("/api/users", dependencies=[Depends(user_rate_limit)])
async def get_users(user: User = Depends(verify_token)):
    """Example endpoint with user-based rate limiting."""
    pass

@app.get("/public/health", dependencies=[Depends(public_rate_limit)])
async def health_check():
    """Example public endpoint with IP-based rate limiting."""
    return {"status": "healthy"}
```

### 4. Response Caching

```python
# middleware/caching.py
from fastapi import Request, Response, Depends
import redis.asyncio as redis
import json
import hashlib
from typing import Optional, Callable, Any, Dict
from pydantic import BaseModel
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class CacheStrategy(str, Enum):
    SIMPLE = "simple"
    LRU = "lru"
    TTL_WITH_REFRESH = "ttl_with_refresh"

class CacheConfig(BaseModel):
    ttl_seconds: int = 300  # 5 minutes default
    strategy: CacheStrategy = CacheStrategy.SIMPLE
    key_prefix: str = "cache"
    include_user_context: bool = False
    include_query_params: bool = True
    cache_headers: bool = True
    invalidation_patterns: Optional[list] = None

class CacheManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get_cache_key(
        self,
        request: Request,
        config: CacheConfig,
        user: Optional[User] = None
    ) -> str:
        """
        Generate cache key based on request and configuration.
        """
        key_components = [
            config.key_prefix,
            request.method,
            str(request.url.path)
        ]

        if config.include_query_params and request.query_params:
            # Sort query params for consistent caching
            sorted_params = sorted(request.query_params.items())
            query_hash = hashlib.md5(str(sorted_params).encode()).hexdigest()
            key_components.append(f"query:{query_hash}")

        if config.include_user_context and user:
            key_components.append(f"user:{user.id}")

        return ":".join(key_components)

    async def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """
        Retrieve cached response from Redis.
        """
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval error for key {cache_key}: {e}")

        return None

    async def set_cached_response(
        self,
        cache_key: str,
        response_data: Dict,
        ttl: int
    ):
        """
        Store response in cache with TTL.
        """
        try:
            await self.redis.setex(
                cache_key,
                ttl,
                json.dumps(response_data)
            )
            logger.debug(f"Cached response for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache storage error for key {cache_key}: {e}")

    async def invalidate_pattern(self, pattern: str):
        """
        Invalidate cache entries matching pattern.
        """
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries matching: {pattern}")
        except Exception as e:
            logger.warning(f"Cache invalidation error for pattern {pattern}: {e}")

    async def should_cache_response(
        self,
        request: Request,
        response: Response,
        config: CacheConfig
    ) -> bool:
        """
        Determine if response should be cached based on various criteria.
        """
        # Only cache GET requests
        if request.method != "GET":
            return False

        # Only cache successful responses
        if response.status_code != 200:
            return False

        # Check for cache-control headers
        if "no-cache" in response.headers.get("cache-control", "").lower():
            return False

        # Don't cache if response has Set-Cookie headers
        if "set-cookie" in response.headers:
            return False

        return True

# Global cache manager
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
cache_manager = CacheManager(redis_client)

def create_cache_middleware(config: CacheConfig):
    """
    Create caching middleware for FastAPI endpoints.
    """
    async def cache_dependency(
        request: Request,
        user: Optional[User] = None
    ):
        # Only process GET requests
        if request.method != "GET":
            return None

        cache_key = await cache_manager.get_cache_key(request, config, user)
        cached_response = await cache_manager.get_cached_response(cache_key)

        if cached_response:
            logger.debug(f"Cache HIT for key: {cache_key}")
            return {
                "cached": True,
                "data": cached_response,
                "cache_key": cache_key
            }

        logger.debug(f"Cache MISS for key: {cache_key}")
        return {
            "cached": False,
            "cache_key": cache_key,
            "config": config
        }

    return cache_dependency

# Response caching decorator for endpoints
def cached_response(config: CacheConfig):
    """
    Decorator for caching endpoint responses.
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Extract request and user from kwargs
            request = kwargs.get("request")
            user = kwargs.get("user")
            cache_info = kwargs.get("cache_info")

            # If we have cached data, return it
            if cache_info and cache_info.get("cached"):
                cached_data = cache_info["data"]
                response = Response(
                    content=cached_data["content"],
                    status_code=cached_data["status_code"],
                    headers=cached_data.get("headers", {}),
                    media_type=cached_data.get("media_type", "application/json")
                )
                response.headers["X-Cache-Status"] = "HIT"
                return response

            # Execute the original function
            response = await func(*args, **kwargs)

            # Cache the response if conditions are met
            if (cache_info and
                await cache_manager.should_cache_response(request, response, config)):

                response_data = {
                    "content": response.body.decode() if hasattr(response, 'body') else "",
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "media_type": response.media_type,
                    "timestamp": time.time()
                }

                await cache_manager.set_cached_response(
                    cache_info["cache_key"],
                    response_data,
                    config.ttl_seconds
                )

                response.headers["X-Cache-Status"] = "MISS"

            return response

        return wrapper
    return decorator

# Predefined cache configurations
short_cache = CacheConfig(
    ttl_seconds=60,  # 1 minute
    key_prefix="short",
    include_user_context=False
)

medium_cache = CacheConfig(
    ttl_seconds=300,  # 5 minutes
    key_prefix="medium",
    include_user_context=True
)

long_cache = CacheConfig(
    ttl_seconds=3600,  # 1 hour
    key_prefix="long",
    include_user_context=False
)

# Usage examples
@app.get("/api/public/config")
@cached_response(long_cache)
async def get_public_config(
    request: Request,
    cache_info = Depends(create_cache_middleware(long_cache))
):
    """Example of long-term cached public configuration."""
    return {"version": "1.0", "features": ["auth", "payments"]}

@app.get("/api/users/{user_id}/profile")
@cached_response(medium_cache)
async def get_user_profile(
    user_id: str,
    request: Request,
    user: User = Depends(verify_token),
    cache_info = Depends(create_cache_middleware(medium_cache))
):
    """Example of user-specific cached data."""
    # Implementation here
    pass

# Cache invalidation utilities
async def invalidate_user_cache(user_id: str):
    """Invalidate all cache entries for a specific user."""
    await cache_manager.invalidate_pattern(f"*:user:{user_id}*")

async def invalidate_service_cache(service_name: str):
    """Invalidate cache entries for a specific service."""
    await cache_manager.invalidate_pattern(f"*:{service_name}:*")
```

## Deployment Architecture

### Docker Configuration

#### Dockerfile for API Gateway

```dockerfile
# Multi-stage build for Python FastAPI application
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml* ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' appuser

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /opt/venv/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /opt/venv/bin/

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Docker Compose Configuration

```yaml
# docker-compose.yml
version: "3.8"

services:
  api-gateway:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    image: musical-spork/api-gateway:latest
    container_name: musical-spork-gateway
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/musical_spork
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - USER_SERVICE_URL=http://user-service:8001
      - AUTH_SERVICE_URL=http://auth-service:8002
    depends_on:
      - postgres
      - redis
      - user-service
      - auth-service
    networks:
      - musical-spork-network
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:15-alpine
    container_name: musical-spork-db
    restart: unless-stopped
    environment:
      - POSTGRES_DB=musical_spork
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    networks:
      - musical-spork-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: musical-spork-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf
    networks:
      - musical-spork-network
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  user-service:
    build:
      context: ../user-service
      dockerfile: Dockerfile
    image: musical-spork/user-service:latest
    container_name: musical-spork-user-service
    restart: unless-stopped
    expose:
      - "8001"
    environment:
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/musical_spork
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - musical-spork-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  auth-service:
    build:
      context: ../auth-service
      dockerfile: Dockerfile
    image: musical-spork/auth-service:latest
    container_name: musical-spork-auth-service
    restart: unless-stopped
    expose:
      - "8002"
    environment:
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/musical_spork
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - musical-spork-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: musical-spork-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - musical-spork-network
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/etc/prometheus/console_libraries"
      - "--web.console.templates=/etc/prometheus/consoles"

  grafana:
    image: grafana/grafana:latest
    container_name: musical-spork-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - musical-spork-network

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: musical-spork-jaeger
    restart: unless-stopped
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - musical-spork-network

networks:
  musical-spork-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Gateway Dockerfile

```dockerfile
# gateway/Dockerfile
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim as runtime

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r gateway && useradd -r -g gateway gateway

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/gateway/.local

# Copy application code
COPY --chown=gateway:gateway . .

# Set PATH to include local Python packages
ENV PATH=/home/gateway/.local/bin:$PATH

# Switch to non-root user
USER gateway

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Requirements File

```txt
# gateway/requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
redis[hiredis]==5.0.1
httpx==0.25.2
python-jose[cryptography]==3.3.0
python-multipart==0.0.6
prometheus-client==0.19.0
structlog==23.2.0
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23
alembic==1.13.1
bcrypt==4.1.2
```

## Configuration Management

### Environment-Based Configuration

```python
# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Musical Spork API Gateway"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    workers: int = Field(default=1, description="Number of worker processes")

    # Security settings
    jwt_secret_key: str = Field(..., min_length=32, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration time in minutes")
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "*.musical-spork.com"],
        description="Allowed hosts for CORS"
    )

    # Database settings
    database_url: str = Field(..., description="Database connection URL")
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    database_max_overflow: int = Field(default=20, description="Database max overflow connections")

    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_max_connections: int = Field(default=10, description="Redis max connections")

    # Service URLs
    user_service_url: str = Field(..., description="User service URL")
    auth_service_url: str = Field(..., description="Auth service URL")
    payment_service_url: Optional[str] = Field(None, description="Payment service URL")
    notification_service_url: Optional[str] = Field(None, description="Notification service URL")

    # External service API keys
    stripe_api_key: Optional[str] = Field(None, description="Stripe API key")
    sendgrid_api_key: Optional[str] = Field(None, description="SendGrid API key")
    google_oauth_client_id: Optional[str] = Field(None, description="Google OAuth client ID")
    google_oauth_client_secret: Optional[str] = Field(None, description="Google OAuth client secret")

    # Rate limiting settings
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(default=100, description="Requests per minute per user")
    rate_limit_burst_size: int = Field(default=200, description="Burst size for rate limiting")

    # Caching settings
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_default_ttl: int = Field(default=300, description="Default cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache entries")

    # Monitoring and logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    jaeger_endpoint: Optional[str] = Field(None, description="Jaeger tracing endpoint")

    # Health check settings
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    health_check_timeout: int = Field(default=10, description="Health check timeout in seconds")

    @validator("environment")
    def validate_environment(cls, v):
        if v == Environment.PRODUCTION:
            # Additional validation for production
            pass
        return v

    @validator("jwt_secret_key")
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v

    @validator("workers")
    def validate_workers(cls, v, values):
        if values.get("environment") == Environment.PRODUCTION and v < 2:
            raise ValueError("Production environment should have at least 2 workers")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

        # Environment variable mappings
        fields = {
            "jwt_secret_key": {"env": "JWT_SECRET_KEY"},
            "database_url": {"env": "DATABASE_URL"},
            "redis_url": {"env": "REDIS_URL"},
            "user_service_url": {"env": "USER_SERVICE_URL"},
            "auth_service_url": {"env": "AUTH_SERVICE_URL"},
            "stripe_api_key": {"env": "STRIPE_API_KEY"},
            "sendgrid_api_key": {"env": "SENDGRID_API_KEY"},
        }

# Global settings instance
settings = Settings()

# Configuration validation on startup
def validate_configuration():
    """
    Validate configuration settings on application startup.
    """
    required_services = ["user_service_url", "auth_service_url"]
    missing_services = [
        service for service in required_services
        if not getattr(settings, service)
    ]

    if missing_services:
        raise ValueError(f"Missing required service URLs: {missing_services}")

    if settings.environment == Environment.PRODUCTION:
        # Additional production validations
        if settings.debug:
            raise ValueError("Debug mode should be disabled in production")

        if not settings.stripe_api_key and settings.payment_service_url:
            logger.warning("Payment service configured but Stripe API key missing")

    logger.info(f"Configuration validated for {settings.environment} environment")

# Environment-specific configurations
class DevelopmentConfig(Settings):
    debug: bool = True
    log_level: LogLevel = LogLevel.DEBUG
    workers: int = 1

class ProductionConfig(Settings):
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    workers: int = 4
    rate_limit_requests_per_minute: int = 1000

class TestingConfig(Settings):
    environment: Environment = Environment.DEVELOPMENT
    database_url: str = "sqlite:///test.db"
    redis_url: str = "redis://localhost:6379/1"  # Use different Redis DB for tests
    jwt_expire_minutes: int = 5  # Short expiration for tests

# Configuration factory
def get_settings() -> Settings:
    """
    Factory function to get appropriate settings based on environment.
    """
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()
```

userServiceUrl: z.string().url(),
authServiceUrl: z.string().url(),
paymentServiceUrl: z.string().url().optional(),

// External services
stripeApiKey: z.string().optional(),
sendgridApiKey: z.string().optional(),

// Cache & Storage
redisUrl: z.string().url(),

// Rate limiting
rateLimitWindow: z.number().default(15 _ 60 _ 1000), // 15 minutes
rateLimitMax: z.number().default(100),

// Monitoring
enableMetrics: z.boolean().default(true),
logLevel: z.enum(["error", "warn", "info", "debug"]).default("info"),
});

export const config = configSchema.parse({
port: parseInt(process.env.PORT || "8080"),
nodeEnv: process.env.NODE_ENV,
jwtSecret: process.env.JWT_SECRET,
jwtExpiresIn: process.env.JWT_EXPIRES_IN,
userServiceUrl: process.env.USER_SERVICE_URL,
authServiceUrl: process.env.AUTH_SERVICE_URL,
paymentServiceUrl: process.env.PAYMENT_SERVICE_URL,
stripeApiKey: process.env.STRIPE_API_KEY,
sendgridApiKey: process.env.SENDGRID_API_KEY,
redisUrl: process.env.REDIS_URL,
rateLimitWindow: parseInt(process.env.RATE_LIMIT_WINDOW || "900000"),
rateLimitMax: parseInt(process.env.RATE_LIMIT_MAX || "100"),
enableMetrics: process.env.ENABLE_METRICS === "true",
logLevel: process.env.LOG_LEVEL as any,
});

````

## Monitoring & Observability

### Metrics Collection

```typescript
// monitoring/metrics.ts
import { Request, Response, NextFunction } from "express";
import prometheus from "prom-client";

// Metrics definitions
const httpRequestDuration = new prometheus.Histogram({
  name: "http_request_duration_seconds",
  help: "Duration of HTTP requests in seconds",
  labelNames: ["method", "route", "status_code", "service"],
  buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10],
});

const httpRequestTotal = new prometheus.Counter({
  name: "http_requests_total",
  help: "Total number of HTTP requests",
  labelNames: ["method", "route", "status_code", "service"],
});

const activeConnections = new prometheus.Gauge({
  name: "gateway_active_connections",
  help: "Number of active connections",
});

export const metricsMiddleware = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const start = Date.now();

  activeConnections.inc();

  res.on("finish", () => {
    const duration = (Date.now() - start) / 1000;
    const route = req.route?.path || req.path;

    httpRequestDuration
      .labels(req.method, route, res.statusCode.toString(), "gateway")
      .observe(duration);

    httpRequestTotal
      .labels(req.method, route, res.statusCode.toString(), "gateway")
      .inc();

    activeConnections.dec();
  });

  next();
};

// Metrics endpoint
export const metricsHandler = async (req: Request, res: Response) => {
  res.set("Content-Type", prometheus.register.contentType);
  res.end(await prometheus.register.metrics());
};
````

### Structured Logging

```typescript
// monitoring/logger.ts
import winston from "winston";

const logger = winston.createLogger({
  level: config.logLevel,
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: "api-gateway" },
  transports: [
    new winston.transports.File({ filename: "logs/error.log", level: "error" }),
    new winston.transports.File({ filename: "logs/combined.log" }),
  ],
});

if (config.nodeEnv !== "production") {
  logger.add(
    new winston.transports.Console({
      format: winston.format.simple(),
    })
  );
}

export const requestLogger = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const start = Date.now();

  res.on("finish", () => {
    const duration = Date.now() - start;

    logger.info("HTTP Request", {
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration,
      userAgent: req.get("User-Agent"),
      ip: req.ip,
      userId: req.user?.id,
    });
  });

  next();
};

export default logger;
```

## Security Considerations

### Input Validation & Sanitization

```typescript
// security/validation.ts
import { body, param, query, validationResult } from "express-validator";
import DOMPurify from "dompurify";
import { JSDOM } from "jsdom";

const window = new JSDOM("").window;
const purify = DOMPurify(window);

export const sanitizeInput = (input: any): any => {
  if (typeof input === "string") {
    return purify.sanitize(input);
  }
  if (Array.isArray(input)) {
    return input.map(sanitizeInput);
  }
  if (typeof input === "object" && input !== null) {
    const sanitized: any = {};
    for (const [key, value] of Object.entries(input)) {
      sanitized[key] = sanitizeInput(value);
    }
    return sanitized;
  }
  return input;
};

export const validateRequest = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      error: "Validation failed",
      details: errors.array(),
    });
  }

  // Sanitize request body
  if (req.body) {
    req.body = sanitizeInput(req.body);
  }

  next();
};
```

### CORS Configuration

```typescript
// security/cors.ts
import cors from "cors";

const corsOptions: cors.CorsOptions = {
  origin: (origin, callback) => {
    const allowedOrigins = [
      "http://localhost:3000", // Development
      "https://app.musical-spork.com", // Production
      "https://admin.musical-spork.com", // Admin panel
    ];

    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  },
  credentials: true,
  optionsSuccessStatus: 200,
  methods: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With"],
  exposedHeaders: ["X-RateLimit-Limit", "X-RateLimit-Remaining"],
};

export default cors(corsOptions);
```

## Error Handling & Circuit Breakers

```typescript
// resilience/circuit-breaker.ts
interface CircuitBreakerConfig {
  failureThreshold: number;
  resetTimeout: number;
  monitoringPeriod: number;
}

enum CircuitState {
  CLOSED = "CLOSED",
  OPEN = "OPEN",
  HALF_OPEN = "HALF_OPEN",
}

export class CircuitBreaker {
  private state = CircuitState.CLOSED;
  private failureCount = 0;
  private lastFailureTime = 0;
  private successCount = 0;

  constructor(private config: CircuitBreakerConfig) {}

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === CircuitState.OPEN) {
      if (Date.now() - this.lastFailureTime > this.config.resetTimeout) {
        this.state = CircuitState.HALF_OPEN;
        this.successCount = 0;
      } else {
        throw new Error("Circuit breaker is OPEN");
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess() {
    this.failureCount = 0;
    if (this.state === CircuitState.HALF_OPEN) {
      this.successCount++;
      if (this.successCount >= 3) {
        // Require 3 successes to close
        this.state = CircuitState.CLOSED;
      }
    }
  }

  private onFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.failureCount >= this.config.failureThreshold) {
      this.state = CircuitState.OPEN;
    }
  }
}
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

1. Set up basic Express.js/FastAPI server structure
2. Implement authentication middleware
3. Create basic routing and proxy functionality
4. Add health check endpoints
5. Set up Redis for session and cache storage

### Phase 2: Core Features (Week 3-4)

1. Implement rate limiting and throttling
2. Add service discovery mechanism
3. Create response caching layer
4. Implement request/response transformation
5. Add basic monitoring and logging

### Phase 3: Resilience (Week 5-6)

1. Implement circuit breakers
2. Add retry mechanisms with exponential backoff
3. Create failover strategies
4. Implement request timeout handling
5. Add comprehensive error handling

### Phase 4: Observability (Week 7-8)

1. Set up Prometheus metrics collection
2. Implement distributed tracing with Jaeger
3. Create comprehensive logging strategy
4. Add performance monitoring dashboards
5. Implement alerting mechanisms

### Phase 5: Security & Optimization (Week 9-10)

1. Implement advanced security measures
2. Add input validation and sanitization
3. Optimize performance and caching
4. Implement API versioning strategy
5. Add comprehensive testing suite

## Testing Strategy

### Unit Tests

```typescript
// tests/middleware/auth.test.ts
import { authenticateJWT } from "../../src/auth/middleware";
import jwt from "jsonwebtoken";

describe("Authentication Middleware", () => {
  const mockRequest = (headers: any) => ({
    headers,
  });

  const mockResponse = () => {
    const res: any = {};
    res.status = jest.fn().mockReturnValue(res);
    res.json = jest.fn().mockReturnValue(res);
    return res;
  };

  const mockNext = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should authenticate valid JWT token", async () => {
    const token = jwt.sign(
      { id: "123", email: "test@example.com" },
      process.env.JWT_SECRET!
    );

    const req = mockRequest({ authorization: `Bearer ${token}` });
    const res = mockResponse();

    authenticateJWT(req as any, res, mockNext);

    expect(mockNext).toHaveBeenCalled();
    expect(req.user).toBeDefined();
  });

  it("should reject invalid token", async () => {
    const req = mockRequest({ authorization: "Bearer invalid-token" });
    const res = mockResponse();

    authenticateJWT(req as any, res, mockNext);

    expect(res.status).toHaveBeenCalledWith(403);
    expect(mockNext).not.toHaveBeenCalled();
  });
});
```

### Integration Tests

```typescript
// tests/integration/gateway.test.ts
import request from "supertest";
import app from "../../src/app";

describe("API Gateway Integration", () => {
  it("should proxy requests to user service", async () => {
    const response = await request(app)
      .get("/api/users/123")
      .set("Authorization", `Bearer ${validToken}`)
      .expect(200);

    expect(response.body).toHaveProperty("id", "123");
  });

  it("should apply rate limiting", async () => {
    // Make requests up to the limit
    for (let i = 0; i < 100; i++) {
      await request(app).get("/api/users").expect(200);
    }

    // Next request should be rate limited
    await request(app).get("/api/users").expect(429);
  });
});
```

#### Requirements.txt

```txt
# Core FastAPI dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.13.0

# Redis
redis==5.0.1
hiredis==2.2.3

# HTTP client
httpx==0.25.2
aiohttp==3.9.1

# Authentication & Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Monitoring & Logging
prometheus-client==0.19.0
structlog==23.2.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
opentelemetry-exporter-jaeger==1.21.0

# Utilities
python-dotenv==1.0.0
click==8.1.7
typer==0.9.0

# Development dependencies (optional)
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
```

## Conclusion

This Universal API Gateway architecture provides a robust, scalable, and secure foundation for Musical Spork's microservices ecosystem. The modular design allows for incremental implementation while maintaining flexibility for future requirements.

Key benefits:

- **Centralized Security**: Single point for authentication and authorization
- **Service Abstraction**: Frontend remains decoupled from backend changes
- **Observability**: Comprehensive monitoring and logging
- **Scalability**: Horizontal scaling and load balancing support
- **Resilience**: Circuit breakers and failover mechanisms
- **Developer Experience**: Clear APIs and extensive documentation

The implementation can start with basic functionality and gradually add advanced features as the application scales and requirements evolve.
