---
title: LLM Quiz Solver
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# LLM Quiz Solver

An automated system that solves data analysis quizzes using GPT-5-nano via AIPipe.

## Features

- **AI-Powered**: Uses GPT-5-nano for intelligent quiz solving.
- **Web Scraping**: Selenium-based browser automation
- **Data Analysis**: Handles PDFs, CSVs, Excel files
- **Dockerized**: Fully containerized for easy deployment
- **Secure**: Secret-based authentication

## API Endpoints

### Health Check
```bash
GET /health
```

Returns service status and configuration.

### Quiz Solver
```bash
POST /quiz
Content-Type: application/json

{
  "email": "your.email@example.com",
  "secret": "secret",
  "url": "https://example.com/quiz-url"
}
```


## Local Development
```bash
# Build and run
docker-compose up --build

# Test
curl http://localhost:7860/health
```

## License

MIT License