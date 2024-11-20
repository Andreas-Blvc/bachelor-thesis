# Use the official Python image as the base
FROM python:3.11-slim

# Install build tools and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        pkg-config \
        tk-dev \
        ffmpeg \
        libncurses5-dev \
        libncursesw5-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        cmake && \
    rm -rf /var/lib/apt/lists/*

# Install Python Requirements
RUN pip install -r requirements.txt

# Create a non-root user
RUN useradd -m nonrootuser

# Set the working directory for the non-root user
WORKDIR /app

# Ensure /app is owned by nonrootuser
RUN chown -R nonrootuser:nonrootuser /app

# Switch to the non-root user
USER nonrootuser

# Copy the entire project into the container
COPY --chown=nonrootuser:nonrootuser . .

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=$(cat /app/jupyter_token.txt)"]

