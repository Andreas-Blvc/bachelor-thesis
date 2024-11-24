#!/bin/bash

# Ensure the script exits on any command failure
set -e

# Variables
PROJECT_DIR="/var/www/html/bachelor-thesis/code"  # Path to your project
CONTAINER_ID="$1"                                # Container ID passed as the first argument
DEST_DIR="/app"                                  # Target directory in the container
DOCKERIGNORE_FILE="$PROJECT_DIR/.dockerignore"

# Check if container ID is provided
if [[ -z "$CONTAINER_ID" ]]; then
    echo "Usage: $0 <container_id>"
    exit 1
fi

# Check if .dockerignore exists
if [[ ! -f "$DOCKERIGNORE_FILE" ]]; then
    echo ".dockerignore file not found in $PROJECT_DIR"
    exit 1
fi

# Create a temporary tarball excluding .dockerignore entries
TMP_TAR="/tmp/project_sync.tar"

# Create tar archive excluding paths from .dockerignore
tar -cf "$TMP_TAR" --exclude-vcs --exclude-backups \
    $(awk '!/^#|^$/ { print "--exclude=" $0 }' "$DOCKERIGNORE_FILE") \
    -C "$PROJECT_DIR" .

# Copy the tarball to the container
docker cp "$TMP_TAR" "$CONTAINER_ID:/tmp/project_sync.tar"

# Extract the tarball inside the container to the target directory
docker exec "$CONTAINER_ID" sh -c "
    tar -xf /tmp/project_sync.tar -C $DEST_DIR &&
    rm /tmp/project_sync.tar
"

# Clean up
rm "$TMP_TAR"

echo "Project synced successfully to container $CONTAINER_ID:$DEST_DIR"

