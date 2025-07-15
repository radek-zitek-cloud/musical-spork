# Musical Spork API Gateway

## Overview

This directory contains the Universal API Gateway implementation based on Python/FastAPI. The gateway serves as the central entry point and orchestration layer between the Musical Spork frontend application and various backend services.

## Folder Structure

```
server/api-gateway/
├── README.md                           # This file - project overview and setup
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project configuration and build settings
├── Dockerfile                          # Docker container configuration
├── docker-compose.yml                 # Local development orchestration
├── .env.example                        # Environment variables template
├── main.py                             # FastAPI application entry point
│
├── app/                                # Main application package
│   ├── __init__.py
│   ├── api/                            # API layer and route definitions
│   │   ├── __init__.py
│   │   ├── v1/                         # API version 1 routes
│   │   │   ├── __init__.py
│   │   │   ├── auth.py                 # Authentication endpoints
│   │   │   ├── users.py                # User management proxy routes
│   │   │   ├── health.py               # Health check endpoints
│   │   │   └── admin.py                # Admin panel endpoints
│   │   └── dependencies.py             # Shared API dependencies
│   │
│   ├── auth/                           # Authentication & authorization module
│   │   ├── __init__.py
│   │   ├── middleware.py               # JWT verification and user extraction
│   │   ├── models.py                   # User, Role, Token Pydantic models
│   │   ├── dependencies.py             # Auth-related FastAPI dependencies
│   │   ├── permissions.py              # Role-based access control (RBAC)
│   │   └── utils.py                    # JWT helpers, password hashing
│   │
│   ├── routing/                        # Service discovery & request routing
│   │   ├── __init__.py
│   │   ├── registry.py                 # Service registry management
│   │   ├── router.py                   # Request routing logic
│   │   ├── load_balancer.py            # Load balancing strategies
│   │   ├── health_checker.py           # Service health monitoring
│   │   └── models.py                   # Service definition models
│   │
│   ├── rate_limiting/                  # Rate limiting & throttling
│   │   ├── __init__.py
│   │   ├── middleware.py               # Rate limiting middleware
│   │   ├── strategies.py               # Different rate limiting algorithms
│   │   ├── storage.py                  # Redis-based rate limit storage
│   │   └── models.py                   # Rate limit configuration models
│   │
│   ├── caching/                        # Response caching system
│   │   ├── __init__.py
│   │   ├── middleware.py               # Cache middleware
│   │   ├── manager.py                  # Cache management and invalidation
│   │   ├── strategies.py               # Cache key generation and TTL logic
│   │   └── models.py                   # Cache configuration models
│   │
│   ├── monitoring/                     # Observability and monitoring
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Prometheus metrics collection
│   │   ├── logging.py                  # Structured logging setup
│   │   ├── tracing.py                  # Distributed tracing with Jaeger
│   │   ├── health.py                   # Health check implementations
│   │   └── middleware.py               # Request/response monitoring
│   │
│   ├── security/                       # Security and validation
│   │   ├── __init__.py
│   │   ├── cors.py                     # CORS configuration
│   │   ├── validation.py               # Input validation and sanitization
│   │   ├── headers.py                  # Security headers middleware
│   │   └── threat_protection.py        # Basic threat detection
│   │
│   ├── circuit_breaker/                # Circuit breaker and resilience
│   │   ├── __init__.py
│   │   ├── breaker.py                  # Circuit breaker implementation
│   │   ├── retry.py                    # Retry logic with backoff
│   │   └── fallback.py                 # Fallback response handling
│   │
│   ├── clients/                        # HTTP clients for backend services
│   │   ├── __init__.py
│   │   ├── base.py                     # Base HTTP client with common features
│   │   ├── user_service.py             # User service client
│   │   ├── auth_service.py             # Auth service client
│   │   ├── payment_service.py          # Payment service client
│   │   └── external_apis.py            # External API clients (Stripe, etc.)
│   │
│   ├── models/                         # Shared Pydantic models
│   │   ├── __init__.py
│   │   ├── common.py                   # Common base models and types
│   │   ├── requests.py                 # Request/response models
│   │   ├── config.py                   # Configuration models
│   │   └── errors.py                   # Error response models
│   │
│   ├── core/                           # Core utilities and configuration
│   │   ├── __init__.py
│   │   ├── config.py                   # Application configuration
│   │   ├── database.py                 # Database connection (if needed)
│   │   ├── redis.py                    # Redis connection management
│   │   ├── exceptions.py               # Custom exception classes
│   │   └── events.py                   # Application lifecycle events
│   │
│   └── utils/                          # General utilities
│       ├── __init__.py
│       ├── helpers.py                  # General helper functions
│       ├── decorators.py               # Custom decorators
│       ├── validators.py               # Custom validation functions
│       └── constants.py                # Application constants
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                     # Pytest configuration and fixtures
│   ├── unit/                           # Unit tests
│   │   ├── __init__.py
│   │   ├── test_auth/
│   │   ├── test_routing/
│   │   ├── test_rate_limiting/
│   │   ├── test_caching/
│   │   └── test_utils/
│   ├── integration/                    # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py
│   │   ├── test_service_proxying.py
│   │   └── test_end_to_end.py
│   └── load/                           # Load testing
│       ├── __init__.py
│       └── test_performance.py
│
├── scripts/                            # Utility scripts
│   ├── setup_dev.py                    # Development environment setup
│   ├── generate_keys.py                # JWT key generation
│   ├── health_check.py                 # Health check script
│   └── benchmark.py                    # Performance benchmarking
│
├── config/                             # Configuration files
│   ├── development.yaml                # Development configuration
│   ├── staging.yaml                    # Staging configuration
│   ├── production.yaml                 # Production configuration
│   └── logging.yaml                    # Logging configuration
│
├── monitoring/                         # Monitoring configuration
│   ├── prometheus.yml                  # Prometheus configuration
│   ├── grafana/
│   │   ├── dashboards/                 # Grafana dashboard definitions
│   │   └── datasources/                # Grafana datasource configurations
│   └── jaeger/                         # Jaeger tracing configuration
│
└── docs/                               # Additional documentation
    ├── api_reference.md                # API endpoint documentation
    ├── deployment.md                   # Deployment instructions
    ├── configuration.md                # Configuration guide
    └── troubleshooting.md              # Common issues and solutions
```

## Key Design Principles

### 1. **Modular Architecture**
- Each core functionality (auth, routing, caching, etc.) is in its own module
- Clear separation of concerns with well-defined interfaces
- Easy to test, maintain, and extend individual components

### 2. **FastAPI Best Practices**
- Pydantic models for all data validation and serialization
- Dependency injection for shared resources (Redis, HTTP clients)
- Async/await throughout for high performance
- Automatic OpenAPI documentation generation

### 3. **Production Ready**
- Comprehensive monitoring and observability
- Circuit breakers and error handling
- Security middleware and validation
- Docker containerization and orchestration

### 4. **Developer Experience**
- Clear folder structure and naming conventions
- Comprehensive test coverage
- Development scripts and utilities
- Extensive documentation

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run Development Server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access Documentation**
   - Interactive API docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Architecture Alignment

This folder structure directly implements the components described in the API Gateway Architecture document:

- **Core Components**: Each major component (auth, routing, rate limiting, caching) has its own module
- **Security**: Dedicated security module with CORS, validation, and threat protection
- **Monitoring**: Comprehensive observability with metrics, logging, and tracing
- **Resilience**: Circuit breakers and retry mechanisms for fault tolerance
- **Configuration**: Environment-based configuration management
- **Testing**: Full test coverage with unit, integration, and load testing

The structure supports the complete implementation roadmap outlined in the architecture while maintaining flexibility for future enhancements and scaling requirements.
