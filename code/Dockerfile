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

# Install Python build tools
RUN pip install --no-cache-dir meson ipywidgets

# Create a non-root user
RUN useradd -m nonrootuser

# Set the working directory for the non-root user
WORKDIR /app

# Switch to the non-root user
USER nonrootuser

# Add local user bin to PATH
ENV PATH="/home/nonrootuser/.local/bin:$PATH"

# Copy the entire project into the container
COPY --chown=nonrootuser:nonrootuser . .

# Install Python dependencies as the non-root user
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token='41b8b8ec553624e31baa17b394e6d907'"]

