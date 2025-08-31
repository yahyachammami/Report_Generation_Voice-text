# AI Meeting Summarizer API

A production-ready FastAPI service that ingests audio files for analysis, built on a template with authentication, async database operations, and Docker support.

## Features

### Application Features
-   **AI-Powered Summarization**: Uses Groq and Llama 3 to generate summaries, topics, decisions, and action items.
-   **Speaker Diarization**: Leverages Pyannote.audio to distinguish between speakers.
-   **Accurate Transcription**: Utilizes OpenAI's Whisper for high-quality speech-to-text.
-   **PDF & Markdown Reports**: Generates and serves reports in multiple formats via the API.

### Platform Features
-   **Modern Python**: Type hints, async/await syntax, and the latest FastAPI features.
-   **JWT Authentication**: Complete authentication system with access and refresh tokens.
-   **SQLAlchemy with Async**: Fully async database operations using SQLAlchemy 2.0+.
-   **Alembic Migrations**: Database schema migrations with Alembic.
-   **Role-based Access Control**: User roles with different permission levels (active, staff, superuser).
-   **Docker Support**: Ready-to-use Docker and Docker Compose configurations.
-   **Developer-friendly**: Auto-reload, debugging, and development tools.
-   **Production-ready**: Configuration for deployment in production environments.

## Project Structure

```
.
├── alembic/                 # Database migrations
├── app/                     # Main application package
│   ├── api/                 # API endpoints
│   ├── core/                # Core functionality (config, security)
│   ├── db/                  # Database session and base
│   ├── models/              # SQLAlchemy models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic (analyzer, summarizer)
│   └── utils/               # Utility functions
├── docker-compose.yml       # Docker Compose for production
├── docker-compose.dev.yml   # Docker Compose for development
├── Dockerfile               # Docker configuration
├── alembic.ini              # Alembic configuration
├── main.py                  # Application entry point
├── pyproject.toml           # Project dependencies and metadata
├── start.sh                 # Production startup script
└── start-dev.sh             # Development startup script
```

## Requirements

- Python 3.11+
- Docker (optional)

## Installation

### Using Docker (recommended)

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd fastapi-template
   ```
2. Create a `.env` file from the example in the "Local Development" section below and fill in your details, especially the required API keys.

3. Start the application with Docker Compose:
   ```bash
   # For development
   docker-compose -f docker-compose.dev.yml up --build

   # For production
   docker-compose up --build
   ```

4. The API will be available at http://localhost:8000

### Local Development

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd fastapi-template
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies. **Ensure your `pyproject.toml` or `requirements.txt` includes the new AI/ML packages.**
   ```bash
   # Key new dependencies to add:
   # openai-whisper, pyannote.audio==2.1.1, groq, reportlab, torch,
   # huggingface_hub<0.23,>=0.22.2

   pip install -e ".[dev]"
   ```

4. Set up environment variables (create a `.env` file):
   ```
   DEBUG=true
   SECRET_KEY=your-secret-key
   DB_ENGINE=sqlite  # or postgresql
   # For PostgreSQL, add these:
   # DB_USER=postgres
   # DB_PASSWORD=password
   # DB_HOST=localhost
   # DB_PORT=5432
   # DB_NAME=app

   # --- Required API Keys for AI Features ---
   GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxx"
   HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```
   **IMPORTANT**: For speaker diarization to work, you must accept the user conditions for the [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) models on Hugging Face.

5. Run migrations:
   ```bash
   alembic upgrade head
   ```

6. Start the application:
   ```bash
   uvicorn main:app --reload
   ```

7. The API will be available at http://localhost:8000

## API Documentation

Once the application is running, you can access:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication

- `POST /auth/signup` - Register a new user
- `POST /auth/login` - Authenticate and get tokens
- `POST /auth/token/refresh` - Refresh access token
- `POST /auth/logout` - Logout user
- `GET /auth/me` - Get current user information

### Reports & Analysis

-   `POST /reports/upload` - Upload an audio file to start analysis.
-   `GET /reports` - Retrieve a list of all reports for the user.
-   `GET /reports/{report_id}` - Fetch a specific report by its ID.
-   `GET /reports/{report_id}/pdf` - Download the PDF version of a report.
-   `GET /reports/{report_id}/md` - Download the Markdown version of a report.
-   `DELETE /reports/{report_id}` - Delete a specific report.

### System

- `GET /health` - Health check endpoint

## Configuration

The application is configured through environment variables which can be set in a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `true` |
| `SECRET_KEY` | JWT secret key | `supersecretkey` |
| `ALGORITHM` | JWT algorithm | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token expiration time | `60` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token expiration time | `7` |
| `CORS_ORIGINS` | CORS allowed origins | `["*"]` |
| `DB_ENGINE` | Database engine | `sqlite` |
| `DB_USER` | Database user | `""` |
| `DB_PASSWORD` | Database password | `""` |
| `DB_HOST` | Database host | `""` |
| `DB_PORT` | Database port | `""` |
| `DB_NAME` | Database name | `app.db` |
| `GROQ_API_KEY` | API Key for Groq Cloud for summarization | `None` |
| `HUGGINGFACE_TOKEN` | Token for Hugging Face for diarization | `None` |

## Development

### Running Tests

```bash
pytest
```

### Code Quality Tools

The project uses several tools to ensure code quality:

- **Black**: Code formatter
- **isort**: Import sorter
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality checks

To set up pre-commit hooks:

```bash
pre-commit install
```

## Database

The template supports SQLite for development and PostgreSQL for production. The default is SQLite.

### Migrations

To create a new migration after changing models:

```bash
alembic revision --autogenerate -m "Description of changes"
```

To apply migrations:

```bash
alembic upgrade head
```

## Docker

The project includes Docker configurations for both development and production:

- `docker-compose.yml`: Production setup
- `docker-compose.dev.yml`: Development setup with hot-reload

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b ft/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin ft/my-feature`