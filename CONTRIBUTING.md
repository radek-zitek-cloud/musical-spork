# Contributing to Musical Spork

Welcome to Musical Spork! We're excited that you're interested in contributing to our project. This document provides guidelines and information for contributors to help maintain code quality and consistency across the codebase.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Guidelines](#documentation-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Security Vulnerabilities](#security-vulnerabilities)

## Code of Conduct

By participating in this project, you agree to maintain a respectful, inclusive, and collaborative environment. We expect all contributors to:

- Be respectful and constructive in all interactions
- Welcome newcomers and help them get started
- Focus on what's best for the community and project
- Show empathy towards other community members
- Accept constructive criticism gracefully

## Getting Started

### Prerequisites

Before contributing, ensure you have the following installed:

- **Node.js** (v18.0.0 or higher)
- **Python** (v3.9 or higher)
- **Git** (v2.30 or higher)
- **Docker** (for containerized development)
- **VS Code** (recommended IDE with relevant extensions)

### Project Structure

```text
musical-spork/
├── client/          # Frontend application (TypeScript/React)
├── server/          # Backend API server (TypeScript/Node.js or Python)
├── shared/          # Shared utilities and types
├── database/        # Database schemas and migrations
├── docker/          # Docker configuration files
├── docs/            # Project documentation
├── .github/         # GitHub workflows and templates
└── README.md        # Project overview
```

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/musical-spork.git
cd musical-spork

# Add upstream remote
git remote add upstream https://github.com/radek-zitek-cloud/musical-spork.git
```

### 2. Environment Configuration

```bash
# Copy environment templates
cp .env.example .env
cp client/.env.example client/.env
cp server/.env.example server/.env

# Edit environment files with your local configuration
```

### 3. Install Dependencies

```bash
# Install root dependencies
npm install

# Install client dependencies
cd client && npm install && cd ..

# Install server dependencies (if Node.js)
cd server && npm install && cd ..

# Install server dependencies (if Python)
cd server && pip install -r requirements.txt && cd ..
```

### 4. Database Setup

```bash
# Start database services
docker-compose up -d database

# Run migrations
npm run db:migrate

# Seed development data (optional)
npm run db:seed
```

### 5. Start Development Environment

```bash
# Start all services in development mode
npm run dev

# Or start services individually
npm run dev:client    # Frontend development server
npm run dev:server    # Backend API server
npm run dev:database  # Database services
```

## Contributing Process

### 1. Create a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
# or
git checkout -b docs/documentation-update
```

### 2. Make Your Changes

- Follow the [Code Standards](#code-standards) outlined below
- Write or update tests for your changes
- Update documentation as needed
- Ensure your changes don't break existing functionality

### 3. Commit Your Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
# Commit format: type(scope): description
git commit -m "feat(auth): add JWT token validation"
git commit -m "fix(api): resolve user creation error handling"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(user): add unit tests for user service"
```

**Commit Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without feature changes
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

### 4. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request through GitHub UI
```

## Code Standards

Our project follows strict coding standards to ensure maintainability and consistency. All code must adhere to the guidelines specified in our [Copilot Instructions](/.github/copilot-instructions.md).

### TypeScript Standards

```typescript
// ✅ Good: Proper type annotations and error handling
export class UserService {
  constructor(
    private userRepository: UserRepository,
    private logger: Logger
  ) {}

  async createUser(userData: CreateUserRequest): Promise<ApiResponse<User>> {
    try {
      const validatedData = validateCreateUser(userData);
      const user = await this.userRepository.create(validatedData);
      
      this.logger.info('User created successfully', {
        userId: user.id,
        email: user.email
      });
      
      return createSuccessResponse(user);
    } catch (error) {
      this.logger.error('Failed to create user', error, { userData });
      throw error;
    }
  }
}
```

### Python Standards

```python
# ✅ Good: Proper type hints and error handling
from typing import Optional
from pydantic import BaseModel

class UserService:
    def __init__(
        self,
        user_repository: UserRepository,
        logger: Logger
    ) -> None:
        self.user_repository = user_repository
        self.logger = logger

    async def create_user(self, user_data: CreateUserRequest) -> ApiResponse[User]:
        """Create a new user with proper validation and error handling."""
        try:
            user = await self.user_repository.create(user_data)
            
            self.logger.info('User created successfully', {
                'user_id': user.id,
                'email': user.email
            })
            
            return create_success_response(user)
        except Exception as error:
            self.logger.error('Failed to create user', exc_info=error, context={
                'user_data': user_data.dict()
            })
            raise
```

### Code Quality Requirements

- **Type Safety**: All code must include proper type annotations (TypeScript interfaces, Python type hints)
- **Error Handling**: Implement comprehensive error handling with custom error classes
- **Documentation**: Add JSDoc/docstring comments for all public functions
- **Validation**: Include input validation for all user-facing functions
- **Logging**: Use structured logging for all significant operations
- **Security**: Sanitize inputs and follow security best practices

## Testing Requirements

### Test Coverage

- **Unit Tests**: Minimum 80% code coverage for business logic
- **Integration Tests**: Cover all API endpoints and database operations
- **E2E Tests**: Critical user flows must have end-to-end test coverage

### TypeScript Testing

```typescript
// Example unit test structure
describe('UserService', () => {
  describe('createUser', () => {
    it('should create user with valid data and return user object', async () => {
      // Arrange: Set up test data and mocks
      const userData = { email: 'test@example.com', name: 'Test User' };
      const mockUser = { id: '123', ...userData };
      
      // Act: Execute the function under test
      const result = await userService.createUser(userData);
      
      // Assert: Verify expected outcomes
      expect(result.success).toBe(true);
      expect(result.data).toEqual(mockUser);
    });

    it('should throw ValidationError when email is invalid', async () => {
      // Test error scenarios
      const invalidData = { email: 'invalid', name: 'Test' };
      
      await expect(userService.createUser(invalidData))
        .rejects.toThrow(ValidationError);
    });
  });
});
```

### Python Testing

```python
# Example unit test structure
import pytest
from unittest.mock import Mock, AsyncMock

class TestUserService:
    @pytest.fixture
    def user_service(self):
        mock_repository = Mock()
        mock_logger = Mock()
        return UserService(mock_repository, mock_logger)

    @pytest.mark.asyncio
    async def test_create_user_with_valid_data_returns_user_object(self, user_service):
        """Should create user with valid data and return user object."""
        # Arrange
        user_data = CreateUserRequest(email="test@example.com", name="Test User")
        expected_user = User(id="123", email="test@example.com", name="Test User")
        user_service.user_repository.create = AsyncMock(return_value=expected_user)

        # Act
        result = await user_service.create_user(user_data)

        # Assert
        assert result.success is True
        assert result.data == expected_user

    @pytest.mark.asyncio
    async def test_create_user_with_invalid_email_raises_validation_error(self, user_service):
        """Should raise ValidationError when email is invalid."""
        # Arrange
        invalid_data = CreateUserRequest(email="invalid", name="Test")

        # Act & Assert
        with pytest.raises(ValidationError):
            await user_service.create_user(invalid_data)
```

### Running Tests

```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Run specific test suites
npm run test:unit
npm run test:integration
npm run test:e2e

# Run tests in watch mode during development
npm run test:watch
```

## Documentation Guidelines

### Code Documentation

- **Functions**: Document all public functions with comprehensive JSDoc/docstrings
- **Classes**: Include class-level documentation explaining purpose and usage
- **APIs**: Document all endpoints with request/response examples
- **Complex Logic**: Add inline comments for complex algorithms or business rules

### Documentation Updates

When making changes, ensure you update:

- API documentation for endpoint changes
- README.md for setup or usage changes
- CHANGELOG.md for notable changes
- Inline code comments for logic changes

## Pull Request Process

### Before Submitting

1. **Sync with upstream**: Ensure your branch is up-to-date
2. **Run tests**: All tests must pass locally
3. **Check code quality**: Run linting and formatting tools
4. **Update documentation**: Include relevant documentation updates
5. **Self-review**: Review your own changes thoroughly

### Pull Request Template

When creating a pull request, include:

```markdown
## Description
Brief description of changes and motivation

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Added new tests for changes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes without version bump
```

### Review Process

1. **Automated Checks**: All CI/CD checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Reviewer should test changes locally if significant
4. **Documentation**: Verify documentation is complete and accurate

### Merge Requirements

- ✅ All automated checks passing
- ✅ At least one approving review from maintainer
- ✅ No unresolved review comments
- ✅ Branch is up-to-date with main
- ✅ Conventional commit format followed

## Issue Reporting

### Bug Reports

When reporting bugs, include:

```markdown
**Bug Description**
Clear and concise description of the bug

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., macOS 12.0]
- Browser: [e.g., Chrome 91.0]
- Node.js version: [e.g., 18.15.0]
- Project version: [e.g., 1.2.3]

**Additional Context**
Screenshots, logs, or other relevant information
```

### Feature Requests

When requesting features, include:

```markdown
**Feature Description**
Clear and concise description of the desired feature

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
Describe your proposed solution

**Alternatives Considered**
Other solutions you've considered

**Additional Context**
Mockups, examples, or other relevant information
```

## Security Vulnerabilities

If you discover a security vulnerability, please:

1. **DO NOT** create a public issue
2. **Email** the maintainers directly at [security@zitek.cloud](mailto:security@zitek.cloud)
3. **Include** detailed information about the vulnerability
4. **Wait** for a response before disclosing publicly

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Development Tools and Scripts

### Useful Commands

```bash
# Code quality and formatting
npm run lint              # Run ESLint/Pylint
npm run format            # Format code with Prettier/Black
npm run type-check        # TypeScript type checking

# Database operations
npm run db:reset          # Reset database
npm run db:migrate        # Run migrations
npm run db:seed           # Seed development data

# Build and deployment
npm run build             # Build for production
npm run build:client      # Build client only
npm run build:server      # Build server only

# Docker development
npm run docker:build      # Build Docker images
npm run docker:up         # Start all services
npm run docker:down       # Stop all services
```

### IDE Configuration

#### VS Code Extensions (Recommended)

```json
{
  "recommendations": [
    "ms-vscode.vscode-typescript-next",
    "esbenp.prettier-vscode",
    "ms-python.python",
    "ms-python.black-formatter",
    "bradlc.vscode-tailwindcss",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-vscode.vscode-eslint"
  ]
}
```

#### Settings

```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true,
    "source.organizeImports": true
  },
  "typescript.preferences.importModuleSpecifier": "relative",
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true
}
```

## Getting Help

- **GitHub Discussions**: For general questions and community support
- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Check the `/docs` directory for detailed guides
- **Code Examples**: Look at existing code for patterns and best practices

## Recognition

Contributors who make significant improvements to the project will be:

- Added to the CONTRIBUTORS.md file
- Mentioned in release notes
- Invited to join the core team (for consistent, high-quality contributions)

## License

By contributing to Musical Spork, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to Musical Spork! Your efforts help make this project better for everyone. 🎵
