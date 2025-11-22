# Use Python 3.11 slim image
FROM python:3.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Install system dependencies for Chromium and Selenium
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utilities
    wget \
    curl \
    gnupg \
    unzip \
    ca-certificates \
    # Chromium and driver
    chromium \
    chromium-driver \
    # Font support
    fonts-liberation \
    fonts-noto-color-emoji \
    # Audio/Video libraries
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    # System libraries
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libexpat1 \
    libgbm1 \
    libglib2.0-0 \
    # GTK and X11
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    # Wayland support
    libwayland-client0 \
    # Additional utilities
    xdg-utils \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Verify Chrome/ChromeDriver installation
RUN chromium --version && \
    chromedriver --version && \
    echo "✅ Chrome and ChromeDriver installed successfully"

# Create app directory
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip list && \
    echo "✅ Python dependencies installed"

# Copy application files
COPY . .

# Create non-root user for security (Hugging Face/Render compatible)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    # Create logs directory with proper permissions
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app/logs && \
    echo "✅ Non-root user created"

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Health check with longer start period for cold starts
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run with gunicorn - optimized for quiz solver
# - 2 workers for parallel processing
# - 4 threads per worker
# - 300s timeout for long-running quiz chains (max 3 minutes per quiz)
# - Logging to stdout/stderr for container logs
CMD ["gunicorn", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "300", \
     "--worker-class", "sync", \
     "--worker-tmp-dir", "/dev/shm", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--capture-output", \
     "app:app"]