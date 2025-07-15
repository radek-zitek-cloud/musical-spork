# Copilot Instructions

## Project Architecture

Musical Spork is a full-stack web application with the following structure:

```
musical-spork/
├── client/          # Frontend (TypeScript/React) - currently empty, ready for setup
├── server/          # Backend API (TypeScript/Node.js or Python/FastAPI) - currently empty
├── shared/          # Shared types and utilities between client/server
├── database/        # Database schemas, migrations, and seed data
├── docker/          # Container configuration for development and deployment
├── docs/            # Technical documentation and API specs
└── .github/         # CI/CD workflows and contribution guidelines
```

**Current Status**: Early development phase - core directories established, comprehensive contributing guidelines in place, ready for initial application scaffolding.

## Essential Project Context

- **Multi-language stack**: TypeScript for frontend/backend APIs, Python for data processing
- **Documentation-driven**: Extensive CONTRIBUTING.md (564 lines) defines development workflows
- **Quality-focused**: Strict coding standards with type safety, testing, and security requirements
- **Container-ready**: Docker setup planned for consistent development environments

## Development Workflows

### Initial Project Setup (Current Priority)
When setting up new components, follow the patterns established in CONTRIBUTING.md:

```bash
# Frontend setup (client/)
npm create react-app@latest . --template typescript
npm install @types/node @types/react @types/react-dom

# Backend setup (server/) - choose one:
npm init -y && npm install express @types/express typescript ts-node
# OR
python -m venv venv && pip install fastapi uvicorn pydantic

# Shared types (shared/)
# Create TypeScript interfaces that both client and server can import
```

### Project-Specific Patterns

- **Layered Architecture**: Follow service → repository → database pattern shown in existing docs
- **Type Safety First**: All functions must include TypeScript interfaces or Python type hints
- **Structured Logging**: Use JSON format with context fields (see copilot-instructions.md examples)
- **Error Handling**: Custom error classes with specific codes (ValidationError, BusinessLogicError, etc.)

### Key Files to Reference

- `CONTRIBUTING.md` - Complete development setup and coding standards (lines 28-87 for project structure)
- `docs/api-gateway-architecture.md` - Universal API Gateway design and implementation guide
- `.gitignore` - Comprehensive ignore patterns for Python, Node.js, Docker, and IDEs
- This file - Architectural patterns and language-specific examples

## Critical Next Steps

1. **API Gateway Implementation**: Follow `docs/api-gateway-architecture.md` for universal gateway layer
2. **Choose Backend Technology**: Implement either TypeScript/Express or Python/FastAPI in `server/`
3. **Setup Build System**: Add package.json/pyproject.toml with scripts matching CONTRIBUTING.md workflows
4. **Database Layer**: Implement schema in `database/` with migrations
5. **Shared Types**: Create common interfaces in `shared/` for API contracts
6. **Docker Configuration**: Add Dockerfile and docker-compose.yml in `docker/`

When implementing any component, reference the extensive examples in the full copilot-instructions.md below for language-specific patterns, security practices, and testing approaches.

## Architecture Patterns

### Service Layer Pattern

#### TypeScript
```typescript
// Services should handle business logic and coordinate between layers
export class UserService {
    constructor(
        private userRepository: UserRepository,
        private emailService: EmailService,
        private logger: Logger
    ) {}

    async createUser(userData: CreateUserRequest): Promise<User> {
        // Validate input, coordinate operations, handle errors
    }
}
```

#### Python
```python
# Services should handle business logic and coordinate between layers
from typing import Protocol
from dataclasses import dataclass

class UserRepositoryProtocol(Protocol):
    async def create(self, user_data: CreateUserRequest) -> User:
        ...

@dataclass
class UserService:
    user_repository: UserRepositoryProtocol
    email_service: EmailService
    logger: Logger

    async def create_user(self, user_data: CreateUserRequest) -> User:
        """Validate input, coordinate operations, handle errors"""
        # Implementation here
```

### Repository Pattern

#### TypeScript
```typescript
// Repositories should handle data access and persistence
export class UserRepository {
    async findById(id: string): Promise<User | null> {
        // Database operations with proper error handling
    }

    async create(user: CreateUserData): Promise<User> {
        // Insert operations with validation
    }
}
```

#### Python
```python
# Repositories should handle data access and persistence
from abc import ABC, abstractmethod
from typing import Optional

class UserRepository(ABC):
    @abstractmethod
    async def find_by_id(self, user_id: str) -> Optional[User]:
        """Database operations with proper error handling"""
        pass

    @abstractmethod
    async def create(self, user: CreateUserData) -> User:
        """Insert operations with validation"""
        pass

class SQLUserRepository(UserRepository):
    def __init__(self, database: Database):
        self.database = database

    async def find_by_id(self, user_id: str) -> Optional[User]:
        # Implementation with proper error handling
        pass

    async def create(self, user: CreateUserData) -> User:
        # Implementation with validation
        pass
```

### Error Handling Strategy

#### TypeScript
```typescript
// Use custom error classes for different error types
export class ValidationError extends Error {
    constructor(
        message: string,
        public field: string,
        public code: string = 'VALIDATION_ERROR'
    ) {
        super(message);
        this.name = 'ValidationError';
    }
}

// Implement centralized error handling
export class ErrorHandler {
    static handle(error: Error, context: string): void {
        // Log error with context and determine response strategy
    }
}
```

#### Python
```python
# Use custom exception classes for different error types
class ValidationError(Exception):
    """Raised when data validation fails"""
    def __init__(self, message: str, field: str, code: str = 'VALIDATION_ERROR'):
        super().__init__(message)
        self.field = field
        self.code = code

class BusinessLogicError(Exception):
    """Raised when business rule is violated"""
    def __init__(self, message: str, code: str = 'BUSINESS_LOGIC_ERROR'):
        super().__init__(message)
        self.code = code

# Implement centralized error handling
class ErrorHandler:
    @staticmethod
    def handle(error: Exception, context: str) -> None:
        """Log error with context and determine response strategy"""
        # Implementation here
```

## Type Safety and Validation

### TypeScript Usage
- Use strict TypeScript configuration
- Define interfaces for all data structures
- Avoid `any` type; use unknown or proper types
- Implement proper null checking and optional chaining
- Use discriminated unions for complex state management

### Python Type Hints
- Use type hints for all function signatures and class attributes
- Import from `typing` module for complex types
- Use `Optional` for nullable values
- Implement proper type checking with mypy
- Use dataclasses or Pydantic models for structured data

### Input Validation

#### TypeScript
```typescript
// Use validation libraries (Zod, Joi, or class-validator)
import { z } from 'zod';

const CreateUserSchema = z.object({
    email: z.string().email(),
    name: z.string().min(2).max(50),
    age: z.number().int().min(18).max(120)
});

export const validateCreateUser = (data: unknown): CreateUserRequest => {
    return CreateUserSchema.parse(data);
};
```

#### Python
```python
# Use Pydantic for validation and serialization
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional

class CreateUserRequest(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., ge=18, le=120)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

# Alternative with dataclasses and manual validation
from dataclasses import dataclass
import re

@dataclass
class CreateUserData:
    email: str
    name: str
    age: int
    
    def __post_init__(self):
        if not re.match(r'^[^@]+@[^@]+\.[^@]+

## Security Guidelines

### Authentication and Authorization

#### TypeScript
```typescript
// Implement proper JWT handling
export class AuthService {
    async validateToken(token: string): Promise<UserPayload> {
        // Verify signature, check expiration, validate claims
    }

    async generateToken(user: User): Promise<string> {
        // Create secure JWT with appropriate expiration
    }
}
```

#### Python
```python
# Implement proper JWT handling with PyJWT
from typing import Dict, Any
import jwt
from datetime import datetime, timedelta, timezone

class AuthService:
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Verify signature, check expiration, validate claims"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

    async def generate_token(self, user: User) -> str:
        """Create secure JWT with appropriate expiration"""
        payload = {
            'user_id': user.id,
            'email': user.email,
            'exp': datetime.now(timezone.utc) + timedelta(hours=24),
            'iat': datetime.now(timezone.utc)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
```

### Data Sanitization

#### TypeScript
```typescript
// Sanitize all user inputs
import DOMPurify from 'dompurify';

export const sanitizeHtml = (input: string): string => {
    return DOMPurify.sanitize(input);
};

// Validate and escape SQL inputs when using raw queries
export const escapeSQL = (input: string): string => {
    // Proper SQL escaping implementation
};
```

#### Python
```python
# Sanitize all user inputs
import bleach
import html
from typing import List

def sanitize_html(input_text: str, allowed_tags: List[str] = None) -> str:
    """Sanitize HTML input to prevent XSS attacks"""
    if allowed_tags is None:
        allowed_tags = ['p', 'br', 'strong', 'em']
    
    return bleach.clean(input_text, tags=allowed_tags, strip=True)

def escape_html(input_text: str) -> str:
    """Escape HTML entities"""
    return html.escape(input_text)

# Use parameterized queries instead of string formatting for SQL
# Good: cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
# Bad: cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

### Environment Configuration

#### TypeScript
```typescript
// Use environment variables for sensitive configuration
export const config = {
    port: Number(process.env.PORT) || 3000,
    dbUrl: process.env.DATABASE_URL || '',
    jwtSecret: process.env.JWT_SECRET || '',
    apiKey: process.env.API_KEY || ''
};

// Validate required environment variables on startup
export const validateEnvironment = (): void => {
    const required = ['DATABASE_URL', 'JWT_SECRET'];
    const missing = required.filter(key => !process.env[key]);
    
    if (missing.length > 0) {
        throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }
};
```

#### Python
```python
# Use environment variables for sensitive configuration
import os
from typing import List
from dataclasses import dataclass

@dataclass
class Config:
    port: int = int(os.getenv('PORT', '8000'))
    database_url: str = os.getenv('DATABASE_URL', '')
    jwt_secret: str = os.getenv('JWT_SECRET', '')
    api_key: str = os.getenv('API_KEY', '')
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'

def validate_environment() -> None:
    """Validate required environment variables on startup"""
    required_vars = ['DATABASE_URL', 'JWT_SECRET']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Alternative using pydantic for configuration management
from pydantic import BaseSettings

class Settings(BaseSettings):
    port: int = 8000
    database_url: str
    jwt_secret: str
    api_key: str
    debug: bool = False
    
    class Config:
        env_file = '.env'
```

## Testing Standards

### Unit Testing

#### TypeScript
```typescript
// Use descriptive test names and comprehensive coverage
describe('UserService', () => {
    describe('createUser', () => {
        it('should create user with valid data and return user object', async () => {
            // Arrange: Set up test data and mocks
            // Act: Execute the function under test
            // Assert: Verify expected outcomes
        });

        it('should throw ValidationError when email is invalid', async () => {
            // Test error scenarios with specific error types
        });
    });
});
```

#### Python
```python
# Use pytest for testing with descriptive test names
import pytest
from unittest.mock import Mock, AsyncMock
from services.user_service import UserService
from exceptions import ValidationError

class TestUserService:
    @pytest.fixture
    def user_service(self):
        """Set up UserService with mocked dependencies"""
        mock_repository = Mock()
        mock_email_service = Mock()
        mock_logger = Mock()
        return UserService(mock_repository, mock_email_service, mock_logger)

    @pytest.mark.asyncio
    async def test_create_user_with_valid_data_returns_user_object(self, user_service):
        """Should create user with valid data and return user object"""
        # Arrange: Set up test data and mocks
        user_data = CreateUserRequest(email="test@example.com", name="Test User", age=25)
        expected_user = User(id="123", email="test@example.com", name="Test User")
        user_service.user_repository.create = AsyncMock(return_value=expected_user)
        
        # Act: Execute the function under test
        result = await user_service.create_user(user_data)
        
        # Assert: Verify expected outcomes
        assert result == expected_user
        user_service.user_repository.create.assert_called_once_with(user_data)

    @pytest.mark.asyncio
    async def test_create_user_with_invalid_email_raises_validation_error(self, user_service):
        """Should raise ValidationError when email is invalid"""
        # Arrange
        invalid_data = CreateUserRequest(email="invalid-email", name="Test", age=25)
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            await user_service.create_user(invalid_data)
        
        assert exc_info.value.field == "email"
```

### Integration Testing

#### TypeScript
```typescript
// Test API endpoints with proper setup and teardown
describe('POST /api/users', () => {
    beforeEach(async () => {
        // Set up test database and clean state
    });

    afterEach(async () => {
        // Clean up test data
    });

    it('should create user and return 201 status with user data', async () => {
        // Full integration test with actual HTTP requests
    });
});
```

#### Python
```python
# Test API endpoints with FastAPI TestClient or Django TestCase
import pytest
from fastapi.testclient import TestClient
from main import app

class TestUserAPI:
    @pytest.fixture
    def client(self):
        """Set up test client"""
        return TestClient(app)

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Set up test database and clean state"""
        # Setup
        await setup_test_database()
        yield
        # Teardown
        await cleanup_test_database()

    def test_create_user_returns_201_with_user_data(self, client):
        """Should create user and return 201 status with user data"""
        # Arrange
        user_data = {
            "email": "test@example.com",
            "name": "Test User",
            "age": 25
        }
        
        # Act
        response = client.post("/api/users", json=user_data)
        
        # Assert
        assert response.status_code == 201
        assert response.json()["email"] == user_data["email"]
        assert response.json()["name"] == user_data["name"]

    def test_create_user_with_invalid_data_returns_422(self, client):
        """Should return 422 for invalid user data"""
        # Arrange
        invalid_data = {"email": "invalid-email", "name": "", "age": 15}
        
        # Act
        response = client.post("/api/users", json=invalid_data)
        
        # Assert
        assert response.status_code == 422
```

## Documentation Standards

### Function Documentation

#### TypeScript
```typescript
/**
 * Calculates compound interest based on principal, rate, and time period
 * 
 * @param principal - Initial investment amount in currency units
 * @param annualRate - Annual interest rate as decimal (0.05 for 5%)
 * @param years - Investment period in years
 * @param compoundingFrequency - Number of times interest compounds per year
 * @returns Total amount after compound interest calculation
 * 
 * @example
 * ```typescript
 * const result = calculateCompoundInterest(1000, 0.05, 10, 12);
 * console.log(result); // 1643.62
 * ```
 */
export function calculateCompoundInterest(
    principal: number,
    annualRate: number,
    years: number,
    compoundingFrequency: number
): number {
    // Implementation with clear variable names and comments
}
```

#### Python
```python
def calculate_compound_interest(
    principal: float, 
    annual_rate: float, 
    years: int, 
    compounding_frequency: int
) -> float:
    """
    Calculate compound interest based on principal, rate, and time period.
    
    Args:
        principal: Initial investment amount in currency units
        annual_rate: Annual interest rate as decimal (0.05 for 5%)
        years: Investment period in years
        compounding_frequency: Number of times interest compounds per year
        
    Returns:
        Total amount after compound interest calculation
        
    Raises:
        ValueError: If any parameter is negative or compounding_frequency is zero
        
    Examples:
        >>> calculate_compound_interest(1000, 0.05, 10, 12)
        1643.62
        
        >>> calculate_compound_interest(5000, 0.03, 5, 4)
        5808.08
    """
    if principal < 0 or annual_rate < 0 or years < 0:
        raise ValueError("Principal, rate, and years must be non-negative")
    
    if compounding_frequency <= 0:
        raise ValueError("Compounding frequency must be positive")
    
    # Implementation with clear variable names and comments
```

### API Documentation

#### TypeScript
```typescript
/**
 * @route POST /api/users
 * @description Creates a new user account
 * @access Public
 * @param {CreateUserRequest} req.body - User creation data
 * @returns {Promise<ApiResponse<User>>} Created user data
 * @throws {ValidationError} When input data is invalid
 * @throws {ConflictError} When user already exists
 */
export const createUser = async (req: Request, res: Response): Promise<void> => {
    // Controller implementation
};
```

#### Python
```python
from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated

router = APIRouter()

@router.post("/users", response_model=ApiResponse[User], status_code=201)
async def create_user(
    user_data: CreateUserRequest,
    user_service: Annotated[UserService, Depends(get_user_service)]
) -> ApiResponse[User]:
    """
    Create a new user account.
    
    Args:
        user_data: User creation data including email, name, and age
        user_service: Injected user service dependency
        
    Returns:
        ApiResponse containing the created user data
        
    Raises:
        HTTPException: 422 when input data is invalid
        HTTPException: 409 when user already exists
        HTTPException: 500 for internal server errors
    """
    try:
        user = await user_service.create_user(user_data)
        return ApiResponse(success=True, data=user)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.message)
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=e.message)
```

## Performance Guidelines

### Database Operations

#### TypeScript
```typescript
// Use efficient queries and proper indexing
export class UserRepository {
    async findUsersWithPagination(
        limit: number,
        offset: number,
        filters: UserFilters
    ): Promise<PaginatedResult<User>> {
        // Implement efficient pagination with proper indexes
        // Use query builders or ORM optimizations
    }

    async findUsersByIds(ids: string[]): Promise<User[]> {
        // Batch operations instead of N+1 queries
    }
}
```

#### Python
```python
# Use efficient queries and proper indexing
from typing import List, Optional
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_users_with_pagination(
        self, 
        limit: int, 
        offset: int, 
        filters: UserFilters
    ) -> PaginatedResult[User]:
        """Implement efficient pagination with proper indexes"""
        # Use SQLAlchemy for optimized queries
        query = select(User)
        
        if filters.name:
            query = query.where(User.name.ilike(f"%{filters.name}%"))
        if filters.email:
            query = query.where(User.email == filters.email)
            
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        # Execute with proper error handling
        result = await self.session.execute(query)
        users = result.scalars().all()
        
        return PaginatedResult(data=users, total=len(users), offset=offset, limit=limit)

    async def find_users_by_ids(self, user_ids: List[str]) -> List[User]:
        """Batch operations instead of N+1 queries"""
        if not user_ids:
            return []
            
        query = select(User).where(User.id.in_(user_ids))
        result = await self.session.execute(query)
        return result.scalars().all()
```

### Caching Strategy

#### TypeScript
```typescript
// Implement appropriate caching layers
export class CacheService {
    async get<T>(key: string): Promise<T | null> {
        // Redis or in-memory cache implementation
    }

    async set<T>(key: string, value: T, ttl: number): Promise<void> {
        // Cache with appropriate expiration
    }
}
```

#### Python
```python
# Implement appropriate caching layers
import redis.asyncio as redis
import json
from typing import Optional, TypeVar, Generic
from datetime import timedelta

T = TypeVar('T')

class CacheService(Generic[T]):
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache with JSON deserialization"""
        try:
            cached_value = await self.redis.get(key)
            if cached_value:
                return json.loads(cached_value)
            return None
        except (redis.RedisError, json.JSONDecodeError) as e:
            # Log error and return None to fallback to data source
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: T, ttl: int) -> None:
        """Set value in cache with JSON serialization and TTL"""
        try:
            serialized_value = json.dumps(value)
            await self.redis.setex(key, ttl, serialized_value)
        except (redis.RedisError, json.JSONEncodeError) as e:
            # Log error but don't raise to avoid breaking the main flow
            logger.warning(f"Cache set error for key {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except redis.RedisError as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

# Decorator for caching function results
from functools import wraps

def cache_result(cache_service: CacheService, ttl: int = 300):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_service.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
```

## API Design Patterns

### RESTful Endpoints

#### TypeScript (Express)
```typescript
// Follow REST conventions and HTTP status codes
export const userRoutes = Router();

userRoutes.get('/', getUserList);           // 200 OK
userRoutes.get('/:id', getUserById);        // 200 OK, 404 Not Found
userRoutes.post('/', createUser);           // 201 Created, 400 Bad Request
userRoutes.put('/:id', updateUser);         // 200 OK, 404 Not Found
userRoutes.delete('/:id', deleteUser);      // 204 No Content, 404 Not Found
```

#### Python (FastAPI)
```python
# Follow REST conventions and HTTP status codes
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List

router = APIRouter(prefix="/api/users", tags=["users"])

@router.get("/", response_model=List[User])
async def get_user_list(
    skip: int = 0, 
    limit: int = 100,
    user_service: UserService = Depends(get_user_service)
) -> List[User]:
    """Get list of users with pagination"""  # 200 OK
    return await user_service.get_users(skip=skip, limit=limit)

@router.get("/{user_id}", response_model=User)
async def get_user_by_id(
    user_id: str,
    user_service: UserService = Depends(get_user_service)
) -> User:
    """Get user by ID"""  # 200 OK, 404 Not Found
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: CreateUserRequest,
    user_service: UserService = Depends(get_user_service)
) -> User:
    """Create new user"""  # 201 Created, 400 Bad Request
    return await user_service.create_user(user_data)

@router.put("/{user_id}", response_model=User)
async def update_user(
    user_id: str,
    user_data: UpdateUserRequest,
    user_service: UserService = Depends(get_user_service)
) -> User:
    """Update existing user"""  # 200 OK, 404 Not Found
    return await user_service.update_user(user_id, user_data)

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    user_service: UserService = Depends(get_user_service)
) -> None:
    """Delete user"""  # 204 No Content, 404 Not Found
    await user_service.delete_user(user_id)
```

### Response Format

#### TypeScript
```typescript
// Consistent API response structure
export interface ApiResponse<T> {
    success: boolean;
    data?: T;
    error?: {
        code: string;
        message: string;
        details?: Record<string, any>;
    };
    meta?: {
        timestamp: string;
        requestId: string;
        pagination?: PaginationMeta;
    };
}
```

#### Python
```python
# Consistent API response structure
from typing import Generic, TypeVar, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

T = TypeVar('T')

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class PaginationMeta(BaseModel):
    total: int
    page: int
    per_page: int
    total_pages: int

class ResponseMeta(BaseModel):
    timestamp: datetime = datetime.utcnow()
    request_id: str
    pagination: Optional[PaginationMeta] = None

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[ErrorDetail] = None
    meta: Optional[ResponseMeta] = None

# Success response helper
def create_success_response(
    data: T, 
    request_id: str,
    pagination: Optional[PaginationMeta] = None
) -> ApiResponse[T]:
    return ApiResponse(
        success=True,
        data=data,
        meta=ResponseMeta(
            request_id=request_id,
            pagination=pagination
        )
    )

# Error response helper
def create_error_response(
    code: str,
    message: str,
    request_id: str,
    details: Optional[Dict[str, Any]] = None
) -> ApiResponse[None]:
    return ApiResponse(
        success=False,
        error=ErrorDetail(code=code, message=message, details=details),
        meta=ResponseMeta(request_id=request_id)
    )
```

## Logging and Monitoring

### Structured Logging

#### TypeScript
```typescript
// Use structured logging for better observability
export class Logger {
    info(message: string, context: Record<string, any> = {}): void {
        console.log(JSON.stringify({
            level: 'info',
            message,
            timestamp: new Date().toISOString(),
            ...context
        }));
    }

    error(message: string, error: Error, context: Record<string, any> = {}): void {
        console.error(JSON.stringify({
            level: 'error',
            message,
            timestamp: new Date().toISOString(),
            error: {
                name: error.name,
                message: error.message,
                stack: error.stack
            },
            ...context
        }));
    }

    warn(message: string, context: Record<string, any> = {}): void {
        console.warn(JSON.stringify({
            level: 'warn',
            message,
            timestamp: new Date().toISOString(),
            ...context
        }));
    }
}
```

#### Python
```python
# Use structured logging for better observability
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler with JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname.lower(),
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            # Add extra fields
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)
                
            return json.dumps(log_entry)

    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log info message with optional context"""
        extra = {'extra_fields': context} if context else {}
        self.logger.info(message, extra=extra)

    def error(self, message: str, exc_info: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        """Log error message with exception info and context"""
        extra = {'extra_fields': context} if context else {}
        self.logger.error(message, exc_info=exc_info, extra=extra)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message with optional context"""
        extra = {'extra_fields': context} if context else {}
        self.logger.warning(message, extra=extra)

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log debug message with optional context"""
        extra = {'extra_fields': context} if context else {}
        self.logger.debug(message, extra=extra)

# Usage example
logger = StructuredLogger(__name__)

# Log with context
logger.info("User created successfully", {
    "user_id": "123",
    "email": "user@example.com",
    "request_id": "req-456"
})

# Log error with exception
try:
    # Some operation
    pass
except Exception as e:
    logger.error("Failed to create user", exc_info=e, context={
        "user_data": {"email": "user@example.com"},
        "request_id": "req-456"
    })
```

## Dependencies and Libraries

### Preferred Libraries

#### TypeScript/Node.js
- **Validation**: Zod or class-validator
- **HTTP Client**: Axios or fetch with proper error handling
- **Testing**: Jest, Supertest for integration testing
- **Database**: Prisma, TypeORM, or Drizzle for type-safe database access
- **Logging**: Winston or Pino for structured logging
- **Environment**: dotenv with validation
- **Web Framework**: Express.js, Fastify, or NestJS
- **Authentication**: jsonwebtoken, passport
- **Caching**: node-redis, ioredis

#### Python
- **Validation**: Pydantic, marshmallow
- **HTTP Client**: httpx (async), requests (sync)
- **Testing**: pytest, pytest-asyncio
- **Database**: SQLAlchemy (ORM), asyncpg (PostgreSQL), motor (MongoDB)
- **Logging**: structlog, loguru
- **Environment**: python-dotenv, pydantic-settings
- **Web Framework**: FastAPI, Django, Flask
- **Authentication**: python-jose, PyJWT
- **Caching**: redis-py, aiocache
- **Task Queue**: Celery, arq
- **Data Processing**: pandas, numpy

### Import Organization

#### TypeScript
```typescript
// Group imports logically
import { readFileSync } from 'fs';
import { join } from 'path';

import express from 'express';
import cors from 'cors';

import { UserService } from './services/user-service';
import { DatabaseConnection } from './database/connection';
import { Logger } from './utils/logger';

import type { User, CreateUserRequest } from './types/user';
```

#### Python
```python
# Group imports logically according to PEP 8
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import fastapi
import sqlalchemy
from pydantic import BaseModel

from services.user_service import UserService
from database.connection import DatabaseConnection
from utils.logger import Logger

from models.user import User, CreateUserRequest
```

## Error Recovery and Resilience

### Retry Logic

#### TypeScript
```typescript
// Implement exponential backoff for external API calls
export async function withRetry<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await operation();
        } catch (error) {
            lastError = error as Error;
            
            if (attempt === maxRetries) {
                throw lastError;
            }
            
            // Exponential backoff with jitter
            const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    
    throw lastError!;
}
```

#### Python
```python
# Implement exponential backoff for external API calls
import asyncio
import random
from typing import TypeVar, Callable, Awaitable
from functools import wraps

T = TypeVar('T')

async def with_retry(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> T:
    """
    Execute an async operation with exponential backoff retry logic.
    
    Args:
        operation: Async function to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        
    Returns:
        Result of the operation
        
    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                raise last_exception
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
            total_delay = delay + jitter
            
            logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {total_delay:.2f}s", {
                "exception": str(e),
                "attempt": attempt + 1,
                "max_retries": max_retries
            })
            
            await asyncio.sleep(total_delay)
    
    raise last_exception

# Decorator version for easier use
def retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for adding retry logic to async functions"""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async def operation():
                return await func(*args, **kwargs)
            return await with_retry(operation, max_retries, base_delay)
        return wrapper
    return decorator

# Usage example
@retry(max_retries=3, base_delay=1.0)
async def call_external_api(url: str) -> dict:
    """Call external API with automatic retry"""
    # Implementation here
    pass
```

### Circuit Breaker Pattern

#### TypeScript
```typescript
// Implement circuit breaker for external dependencies
export class CircuitBreaker {
    private failures: number = 0;
    private lastFailTime: number = 0;
    private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';

    constructor(
        private failureThreshold: number = 5,
        private recoveryTimeout: number = 60000,
        private monitoringPeriod: number = 120000
    ) {}

    async call<T>(operation: () => Promise<T>): Promise<T> {
        if (this.state === 'OPEN') {
            if (Date.now() - this.lastFailTime > this.recoveryTimeout) {
                this.state = 'HALF_OPEN';
            } else {
                throw new Error('Circuit breaker is OPEN');
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

    private onSuccess(): void {
        this.failures = 0;
        this.state = 'CLOSED';
    }

    private onFailure(): void {
        this.failures++;
        this.lastFailTime = Date.now();

        if (this.failures >= this.failureThreshold) {
            this.state = 'OPEN';
        }
    }
}
```

#### Python
```python
# Implement circuit breaker for external dependencies
import asyncio
import time
from enum import Enum
from typing import TypeVar, Callable, Awaitable, Optional
from dataclasses import dataclass

T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    success_threshold: int = 2  # successes needed to close from half-open
    timeout: float = 30.0  # operation timeout

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        
    async def call(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute operation through circuit breaker"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                operation(), 
                timeout=self.config.timeout
            )
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._reset()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _reset(self) -> None:
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker reset to CLOSED")

# Usage example
circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    success_threshold=2
))

async def call_external_service():
    """Call external service with circuit breaker protection"""
    async def operation():
        # External service call implementation
        pass
    
    return await circuit_breaker.call(operation)
```

## Notes for Copilot

When generating code for this project:

### Universal Guidelines
1. **Always include proper type annotations and interfaces (TypeScript) or type hints (Python)**
2. **Implement comprehensive error handling with custom error/exception classes**
3. **Add documentation comments (JSDoc for TypeScript, docstrings for Python)**
4. **Use async/await pattern consistently instead of Promise chains or callback-style code**
5. **Include input validation for all user-facing functions**
6. **Follow the established patterns for logging and monitoring**
7. **Consider security implications and implement proper sanitization**
8. **Write testable code with dependency injection where appropriate**
9. **Use environment variables for configuration, never hardcode secrets**
10. **Implement proper resource cleanup and error recovery**

### TypeScript-Specific Guidelines
- Use strict TypeScript configuration with proper null checking
- Prefer interfaces over types for object shapes
- Use discriminated unions for complex state management
- Implement proper Promise handling with error boundaries
- Use Express.js or FastAPI patterns for API development
- Follow Node.js best practices for server-side development

### Python-Specific Guidelines
- Use type hints consistently with `typing` module imports
- Follow PEP 8 style guidelines for naming and formatting
- Use dataclasses or Pydantic models for structured data
- Implement proper exception hierarchies with custom exception classes
- Use async/await for I/O operations when possible
- Follow FastAPI or Django patterns for web development
- Use context managers for resource management

### Language Selection Guidance
- **TypeScript**: Use for frontend applications, Node.js backends, real-time features, and when seamless JavaScript ecosystem integration is needed
- **Python**: Use for data processing, machine learning, scientific computing, backend APIs requiring rapid development, and when leveraging Python's rich ecosystem

### Code Quality Standards
- All functions should have comprehensive error handling
- Include unit tests for business logic functions
- Use dependency injection for better testability
- Implement proper logging with structured data
- Follow single responsibility principle
- Use consistent naming conventions within each language

### Security Considerations
- Validate and sanitize all user inputs
- Use parameterized queries for database operations
- Implement proper authentication and authorization
- Never log sensitive information
- Use secure random generators for tokens and secrets
- Implement rate limiting for public APIs

Remember: Code should be production-ready, secure, well-documented, and maintainable. Choose the appropriate language based on the specific requirements and context of the task., self.email):
            raise ValidationError('Invalid email format', 'email')
        if not (2 <= len(self.name.strip()) <= 50):
            raise ValidationError('Name must be between 2 and 50 characters', 'name')
        if not (18 <= self.age <= 120):
            raise ValidationError('Age must be between 18 and 120', 'age')
```

## Security Guidelines

### Authentication and Authorization
```typescript
// Implement proper JWT handling
export class AuthService {
    async validateToken(token: string): Promise<UserPayload> {
        // Verify signature, check expiration, validate claims
    }

    async generateToken(user: User): Promise<string> {
        // Create secure JWT with appropriate expiration
    }
}
```

### Data Sanitization
```typescript
// Sanitize all user inputs
import DOMPurify from 'dompurify';

export const sanitizeHtml = (input: string): string => {
    return DOMPurify.sanitize(input);
};

// Validate and escape SQL inputs when using raw queries
export const escapeSQL = (input: string): string => {
    // Proper SQL escaping implementation
};
```

### Environment Configuration
```typescript
// Use environment variables for sensitive configuration
export const config = {
    port: Number(process.env.PORT) || 3000,
    dbUrl: process.env.DATABASE_URL || '',
    jwtSecret: process.env.JWT_SECRET || '',
    apiKey: process.env.API_KEY || ''
};

// Validate required environment variables on startup
export const validateEnvironment = (): void => {
    const required = ['DATABASE_URL', 'JWT_SECRET'];
    const missing = required.filter(key => !process.env[key]);
    
    if (missing.length > 0) {
        throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }
};
```

## Testing Standards

### Unit Testing
```typescript
// Use descriptive test names and comprehensive coverage
describe('UserService', () => {
    describe('createUser', () => {
        it('should create user with valid data and return user object', async () => {
            // Arrange: Set up test data and mocks
            // Act: Execute the function under test
            // Assert: Verify expected outcomes
        });

        it('should throw ValidationError when email is invalid', async () => {
            // Test error scenarios with specific error types
        });
    });
});
```

### Integration Testing
```typescript
// Test API endpoints with proper setup and teardown
describe('POST /api/users', () => {
    beforeEach(async () => {
        // Set up test database and clean state
    });

    afterEach(async () => {
        // Clean up test data
    });

    it('should create user and return 201 status with user data', async () => {
        // Full integration test with actual HTTP requests
    });
});
```

## Documentation Standards

### Function Documentation
```typescript
/**
 * Calculates compound interest based on principal, rate, and time period
 * 
 * @param principal - Initial investment amount in currency units
 * @param annualRate - Annual interest rate as decimal (0.05 for 5%)
 * @param years - Investment period in years
 * @param compoundingFrequency - Number of times interest compounds per year
 * @returns Total amount after compound interest calculation
 * 
 * @example
 * ```typescript
 * const result = calculateCompoundInterest(1000, 0.05, 10, 12);
 * console.log(result); // 1643.62
 * ```
 */
export function calculateCompoundInterest(
    principal: number,
    annualRate: number,
    years: number,
    compoundingFrequency: number
): number {
    // Implementation with clear variable names and comments
}
```

### API Documentation
```typescript
/**
 * @route POST /api/users
 * @description Creates a new user account
 * @access Public
 * @param {CreateUserRequest} req.body - User creation data
 * @returns {Promise<ApiResponse<User>>} Created user data
 * @throws {ValidationError} When input data is invalid
 * @throws {ConflictError} When user already exists
 */
export const createUser = async (req: Request, res: Response): Promise<void> => {
    // Controller implementation
};
```

## Performance Guidelines

### Database Operations
```typescript
// Use efficient queries and proper indexing
export class UserRepository {
    async findUsersWithPagination(
        limit: number,
        offset: number,
        filters: UserFilters
    ): Promise<PaginatedResult<User>> {
        // Implement efficient pagination with proper indexes
        // Use query builders or ORM optimizations
    }

    async findUsersByIds(ids: string[]): Promise<User[]> {
        // Batch operations instead of N+1 queries
    }
}
```

### Caching Strategy
```typescript
// Implement appropriate caching layers
export class CacheService {
    async get<T>(key: string): Promise<T | null> {
        // Redis or in-memory cache implementation
    }

    async set<T>(key: string, value: T, ttl: number): Promise<void> {
        // Cache with appropriate expiration
    }
}
```

## API Design Patterns

### RESTful Endpoints
```typescript
// Follow REST conventions and HTTP status codes
export const userRoutes = Router();

userRoutes.get('/', getUserList);           // 200 OK
userRoutes.get('/:id', getUserById);        // 200 OK, 404 Not Found
userRoutes.post('/', createUser);           // 201 Created, 400 Bad Request
userRoutes.put('/:id', updateUser);         // 200 OK, 404 Not Found
userRoutes.delete('/:id', deleteUser);      // 204 No Content, 404 Not Found
```

### Response Format
```typescript
// Consistent API response structure
export interface ApiResponse<T> {
    success: boolean;
    data?: T;
    error?: {
        code: string;
        message: string;
        details?: Record<string, any>;
    };
    meta?: {
        timestamp: string;
        requestId: string;
        pagination?: PaginationMeta;
    };
}
```

## Logging and Monitoring

### Structured Logging
```typescript
// Use structured logging for better observability
export class Logger {
    info(message: string, context: Record<string, any> = {}): void {
        console.log(JSON.stringify({
            level: 'info',
            message,
            timestamp: new Date().toISOString(),
            ...context
        }));
    }

    error(message: string, error: Error, context: Record<string, any> = {}): void {
        // Comprehensive error logging with stack traces
    }
}
```

## Dependencies and Libraries

### Preferred Libraries
- **Validation**: Zod or class-validator
- **HTTP Client**: Axios or fetch with proper error handling
- **Testing**: Jest, Supertest for integration testing
- **Database**: Prisma, TypeORM, or Drizzle for type-safe database access
- **Logging**: Winston or Pino for structured logging
- **Environment**: dotenv with validation

### Import Organization
```typescript
// Group imports logically
import { readFileSync } from 'fs';
import { join } from 'path';

import express from 'express';
import cors from 'cors';

import { UserService } from './services/user-service';
import { DatabaseConnection } from './database/connection';
import { Logger } from './utils/logger';

import type { User, CreateUserRequest } from './types/user';
```

## Error Recovery and Resilience

### Retry Logic
```typescript
// Implement exponential backoff for external API calls
export async function withRetry<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
): Promise<T> {
    // Exponential backoff implementation with jitter
}
```

### Circuit Breaker Pattern
```typescript
// Implement circuit breaker for external dependencies
export class CircuitBreaker {
    private failures: number = 0;
    private lastFailTime: number = 0;
    private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';

    async call<T>(operation: () => Promise<T>): Promise<T> {
        // Circuit breaker logic implementation
    }
}
```

## Notes for Copilot

When generating code for this project:

1. **Always include proper TypeScript types and interfaces**
2. **Implement comprehensive error handling with custom error classes**
3. **Add JSDoc comments for public functions and classes**
4. **Use async/await pattern consistently instead of Promise chains**
5. **Include input validation for all user-facing functions**
6. **Follow the established patterns for logging and monitoring**
7. **Consider security implications and implement proper sanitization**
8. **Write testable code with dependency injection where appropriate**
9. **Use environment variables for configuration, never hardcode secrets**
10. **Implement proper resource cleanup and error recovery**

Remember: Code should be production-ready, secure, well-documented, and maintainable.