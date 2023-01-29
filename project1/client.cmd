SET FOLDER_ROOT=%~dp0
SET PYTH=%FOLDER_ROOT%\cmia2023\python.exe

%PYTH% client.py --host 127.0.0.1 --port 8888
pause