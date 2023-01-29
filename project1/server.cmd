SET FOLDER_ROOT=%~dp0
SET PYTH=%FOLDER_ROOT%\cmia2023\python.exe

%PYTH% server.py --tickers AAPL --port 8888 --sampling 5
pause