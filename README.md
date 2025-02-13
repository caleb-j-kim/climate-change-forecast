# climate-change-forecast
Developing a customized climate-change analysis and forecasting tool that's equipped with a graphical interface that allows users to analyze climate-change trends in a selected region. This tool may also create predictions or forecast changes that could occur due to increasing trends of global warming.

# pip install -r requirements.txt
Enable Virtualization inside of BIOS
Install Docker Desktop from online -> sign in
Install Docker and Dev Containers Extensions in VS Code (Ctrl+Shift+P or Cmd+Shift+P and type 'Remote-Containers: Reopen in Container')
Connect your VS Code to a terminal connection with Docker (you will need to install Ubuntu and set up terminal / remote connections in VS Code)
In terminal, 'wsl --install --no-distribution' (Windows)
In terminal, 'docker build -t climate-change-forecast .' (Keep Docker Desktop open when doing this)
(if changes have been made to dependencies or Dockerfile, force a fresh build: docker build --no-cache -t climate-change-forecast .)
In terminal, 'docker run -p 5000:5000 climate-change-forecast' OR 'docker run -it -p 5000:5000 climate-change-forecast bash' -> 'python -m flask run --host=0.0.0.0'
Test changes: 'docker run -p 5000:5000 -v "$(pwd)":/app climate-change-forecast'

# GitHub commands
Stage changes: 'git add .'
Commit changes: 'git commit -m "Describe changes here'
Push to remote repo: 'git push'