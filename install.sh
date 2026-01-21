#!/bin/bash
set -e

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting installation...${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check for Python
echo "Checking for Python..."
if command_exists python3; then
    PYTHON_CMD=python3
elif command_exists python; then
    PYTHON_CMD=python
else
    echo -e "${RED}Python is not installed.${NC}"
    echo "Please download and install Python from https://www.python.org/downloads/"
    exit 1
fi
echo -e "${GREEN}Found Python: $($PYTHON_CMD --version)${NC}"

# 2. Check for Node.js
echo "Checking for Node.js..."
if ! command_exists node; then
    echo -e "${RED}Node.js is not installed.${NC}"
    echo "Please download and install Node.js from https://nodejs.org/"
    exit 1
fi
echo -e "${GREEN}Found Node.js: $(node --version)${NC}"

# 3. Check for npm
if ! command_exists npm; then
    echo -e "${RED}npm is not installed.${NC}"
    exit 1
fi

# 4. Set up Python Virtual Environment
echo "Setting up Python virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment directory (.venv) already exists."
else
    $PYTHON_CMD -m venv .venv
    echo -e "${GREEN}Created virtual environment.${NC}"
fi

# 5. Install Python dependencies
echo "Installing Python dependencies..."
# We use the python executable inside the venv to ensure we install there
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Python dependencies installed successfully.${NC}"
else
    echo -e "${RED}Failed to install Python dependencies.${NC}"
    exit 1
fi

# 6. Install Root Node dependencies
echo "Installing root Node dependencies..."
npm install

# 7. Install Frontend Node dependencies
echo "Installing frontend Node dependencies..."
cd app/frontend
npm install
cd ../..

echo -e "${GREEN}=========================================="
echo -e "Installation Complete!"
echo -e "==========================================${NC}"
echo ""
echo "To start the application:"
echo "1. Run the following command in the root directory:"
echo -e "${GREEN}   npm run dev${NC}"
echo ""
