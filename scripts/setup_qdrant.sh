#!/bin/bash

# ============================================================================
# Qdrant Docker Setup Script for EvenementsRAG
# ============================================================================
# This script manages the Qdrant vector database Docker container
#
# Usage:
#   ./scripts/setup_qdrant.sh [start|stop|restart|status|logs|clean]
#
# Commands:
#   start   - Start Qdrant container (creates if doesn't exist)
#   stop    - Stop Qdrant container
#   restart - Restart Qdrant container
#   status  - Check Qdrant container status
#   logs    - Show Qdrant container logs
#   clean   - Stop and remove container + data (WARNING: deletes all data)
# ============================================================================

set -e  # Exit on error

# Configuration
CONTAINER_NAME="qdrant-evenementsrag"
QDRANT_IMAGE="qdrant/qdrant:latest"
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
STORAGE_PATH="$(pwd)/data/vector_database/qdrant"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "ℹ $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker is installed"
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# ============================================================================
# Main Functions
# ============================================================================

start_qdrant() {
    echo "Starting Qdrant vector database..."

    check_docker

    # Create storage directory if it doesn't exist
    if [ ! -d "$STORAGE_PATH" ]; then
        print_info "Creating storage directory: $STORAGE_PATH"
        mkdir -p "$STORAGE_PATH"
    fi

    if container_running; then
        print_warning "Qdrant container is already running"
        show_connection_info
        return 0
    fi

    if container_exists; then
        print_info "Starting existing Qdrant container..."
        docker start "$CONTAINER_NAME"
    else
        print_info "Creating new Qdrant container..."
        docker run -d \
            --name "$CONTAINER_NAME" \
            -p "$QDRANT_PORT:6333" \
            -p "$QDRANT_GRPC_PORT:6334" \
            -v "$STORAGE_PATH:/qdrant/storage" \
            "$QDRANT_IMAGE"
    fi

    # Wait for Qdrant to be ready
    print_info "Waiting for Qdrant to be ready..."
    sleep 3

    # Check if Qdrant is responding
    MAX_RETRIES=10
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s -f "http://localhost:$QDRANT_PORT/healthz" > /dev/null 2>&1; then
            print_success "Qdrant is ready!"
            show_connection_info
            return 0
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -n "."
        sleep 1
    done

    print_error "Qdrant failed to start properly"
    print_info "Check logs with: ./scripts/setup_qdrant.sh logs"
    exit 1
}

stop_qdrant() {
    echo "Stopping Qdrant vector database..."

    if ! container_running; then
        print_warning "Qdrant container is not running"
        return 0
    fi

    docker stop "$CONTAINER_NAME"
    print_success "Qdrant stopped"
}

restart_qdrant() {
    echo "Restarting Qdrant vector database..."
    stop_qdrant
    sleep 2
    start_qdrant
}

show_status() {
    echo "Qdrant Status"
    echo "============================================"

    if container_exists; then
        if container_running; then
            print_success "Container is running"
            echo ""
            docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            echo ""

            # Check health endpoint
            if curl -s -f "http://localhost:$QDRANT_PORT/healthz" > /dev/null 2>&1; then
                print_success "Health check passed"

                # Show version info
                VERSION=$(curl -s "http://localhost:$QDRANT_PORT/" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
                if [ -n "$VERSION" ]; then
                    echo "Version: $VERSION"
                fi
            else
                print_warning "Health check failed - service may not be ready"
            fi

            echo ""
            show_connection_info
        else
            print_warning "Container exists but is not running"
            echo "Start it with: ./scripts/setup_qdrant.sh start"
        fi
    else
        print_info "Container does not exist"
        echo "Create it with: ./scripts/setup_qdrant.sh start"
    fi

    echo ""
    echo "Storage location: $STORAGE_PATH"
    if [ -d "$STORAGE_PATH" ]; then
        STORAGE_SIZE=$(du -sh "$STORAGE_PATH" 2>/dev/null | cut -f1)
        echo "Storage size: $STORAGE_SIZE"
    fi
}

show_logs() {
    if ! container_exists; then
        print_error "Qdrant container does not exist"
        exit 1
    fi

    echo "Showing Qdrant logs (Ctrl+C to exit)..."
    docker logs -f "$CONTAINER_NAME"
}

clean_qdrant() {
    echo "Cleaning Qdrant (removing container and data)..."
    print_warning "This will delete all Qdrant data!"

    read -p "Are you sure you want to continue? (yes/no): " -r
    echo

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_info "Clean operation cancelled"
        exit 0
    fi

    # Stop container if running
    if container_running; then
        print_info "Stopping container..."
        docker stop "$CONTAINER_NAME"
    fi

    # Remove container if exists
    if container_exists; then
        print_info "Removing container..."
        docker rm "$CONTAINER_NAME"
    fi

    # Remove storage directory
    if [ -d "$STORAGE_PATH" ]; then
        print_info "Removing storage directory..."
        rm -rf "$STORAGE_PATH"
    fi

    print_success "Qdrant cleaned successfully"
}

show_connection_info() {
    echo ""
    echo "Connection Information:"
    echo "============================================"
    echo "REST API:  http://localhost:$QDRANT_PORT"
    echo "gRPC API:  localhost:$QDRANT_GRPC_PORT"
    echo "Dashboard: http://localhost:$QDRANT_PORT/dashboard"
    echo ""
    echo "Python connection:"
    echo "  from qdrant_client import QdrantClient"
    echo "  client = QdrantClient(host='localhost', port=$QDRANT_PORT)"
}

show_help() {
    echo "Qdrant Docker Setup Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start   - Start Qdrant container (creates if doesn't exist)"
    echo "  stop    - Stop Qdrant container"
    echo "  restart - Restart Qdrant container"
    echo "  status  - Check Qdrant container status"
    echo "  logs    - Show Qdrant container logs (follow mode)"
    echo "  clean   - Stop and remove container + data (WARNING: deletes all data)"
    echo "  help    - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start          # Start Qdrant"
    echo "  $0 status         # Check if Qdrant is running"
    echo "  $0 logs           # View logs"
}

# ============================================================================
# Main Script
# ============================================================================

COMMAND=${1:-help}

case "$COMMAND" in
    start)
        start_qdrant
        ;;
    stop)
        stop_qdrant
        ;;
    restart)
        restart_qdrant
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    clean)
        clean_qdrant
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        show_help
        exit 1
        ;;
esac

exit 0
