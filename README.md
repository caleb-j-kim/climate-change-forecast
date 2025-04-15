# Climate Change Forecast

## Overview
The **Climate Change Forecast** tool provides an interactive graphical interface for analyzing climate change trends in a selected region. It enables users to visualize historical climate data and forecast potential changes influenced by global warming. This tool is designed to support informed decision-making and enhance climate awareness through predictive analytics.

---

## Installation & Setup
### Prerequisites
Ensure the following are installed before proceeding:
- **Enable Virtualization** in your system's BIOS
- **Docker Desktop** (Download and sign in)
- **VS Code** with the following extensions:
  - Docker
  - Dev Containers

### Setup Instructions
1. Clone this repository:
   ```sh
   git clone https://github.com/caleb-j-kim/climate-change-forecast.git
   cd climate-change-forecast
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Open VS Code and install the necessary extensions.
4. Reopen the project in a **Dev Container**:
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type: `Remote-Containers: Reopen in Container`
5. Set up a terminal connection with Docker:
   - Install **Ubuntu** and configure remote connections in VS Code.
   - Run the following command in the terminal (Windows only):
     ```sh
     wsl --install --no-distribution
     ```

---

## Running the Application
### Build the Docker Image
Ensure **Docker Desktop** is running before executing the following command:
```sh
docker build -t climate-change-forecast .
```
To force a fresh build (if dependencies or the Dockerfile have been updated):
```sh
docker build --no-cache -t climate-change-forecast .
```

### Run the Application
Run the application using one of the following methods:
1. Standard execution:
   ```sh
   docker run -p 5000:5000 climate-change-forecast
   ```
2. Interactive mode:
   ```sh
   docker run -it -p 5000:5000 climate-change-forecast bash
   ```
   Then, manually start Flask:
   ```sh
   python -m flask run --host=0.0.0.0
   ```
3. Testing changes while developing:
   ```sh
   docker run -p 5000:5000 -v "$(pwd)":/app climate-change-forecast
   ```

---

## Git Workflow
### Committing and Pushing Changes
1. Stage changes:
   ```sh
   git add .
   ```
2. Commit changes:
   ```sh
   git commit -m "Describe changes here"
   ```
3. Push to the remote repository:
   ```sh
   git push
   ```

---

## Backend Testing
1. Train models:
   ``` sh
   Invoke-RestMethod -Method GET -Uri "http://localhost:5000/train"

2. Test models:
   ``` sh
   Invoke-RestMethod -Method GET -Uri "http://localhost:5000/test"

3. Predict climate:
   ``` sh
Invoke-RestMethod -Method POST -Uri "http://localhost:5000/predict" `
    -Body '{"dataset": "country", "year": 2020, "month": 07, "location": "United States"}' `
    -ContentType "application/json"

   ``` sh
Invoke-RestMethod -Method POST -Uri "http://localhost:5000/predict" `
    -Body '{"dataset": "city", "year": 2000, "month": 7, "location": {"city": "New York", "country": "United States"}}' `
    -ContentType "application/json"

Invoke-RestMethod -Method POST -Uri "http://localhost:5000/predict" `
    -Body '{"dataset": "state", "year": 2000, "month": 7, "location": {"state": "Virginia", "country": "United States"}}' `
    -ContentType "application/json"
