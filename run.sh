#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Gaussian Splat Application...${NC}"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: .venv directory not found.${NC}"
    echo -e "${YELLOW}Please run ./install.sh first to set up the project.${NC}"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${RED}Error: node_modules directory not found.${NC}"
    echo -e "${YELLOW}Please run ./install.sh first to install dependencies.${NC}"
    exit 1
fi

# Start the application
echo -e "${GREEN}Launching development server...${NC}"
npm run dev
