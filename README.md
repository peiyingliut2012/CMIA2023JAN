# CMIA2023
# Project 1
--------------------
#### Launch Server
- (1) open server.cmd and modify input params\
![image](https://user-images.githubusercontent.com/123887929/215348237-9a234ab2-7e71-413e-b9a4-d5d59eedb784.png)
- (2) click server.cmd


#### Launch Client
- (1) open server.cmd and modify input params\
![image](https://user-images.githubusercontent.com/123887929/215348248-3a7c0842-a554-43ef-8db2-2f35fc350858.png)
- (1) click client.cmd

#### Project Structure
- conda env folder: CMIA2023/project1/cmia2023/
- log folder: CMIA2023/project1/log/
- main python lib: CMIA2023/project1/projectone_lib.py
- server cmd: CMIA2023/project1/server.cmd
- client cmd: CMIA2023/project1/client.cmd
- server python: CMIA2023/project1/server.py
- client python: CMIA2023/project1/client.py


#### Log folder
- logs will be saved to log folder 
![image](https://user-images.githubusercontent.com/123887929/215348361-df8da4ef-3099-40a4-bccd-c081ae8dc161.png)


#### Conda env
- This project uses conda env "cmia2023" in CMIA2023\project1\cmia2023
- CMIA2023\project1\cmia2023\python.exe is used to run server and client as specified in server.cmd and client.cmd

#### Features
- data request
  - client\
    ![image](https://user-images.githubusercontent.com/123887929/215348723-9ea28bd8-c9fb-4a1f-9d9b-a72a4a50689d.png)
  - server\
    ![image](https://user-images.githubusercontent.com/123887929/215348755-1e309d86-2a97-4439-88d6-346700ef252b.png)
- add ticker
  - client example 1\
    ![image](https://user-images.githubusercontent.com/123887929/215348978-ef07f8da-dba8-4cd0-b7b3-5cb51b7030ce.png)
  - client example 2: the new ticker won't be in data after "add". only "report" will refresh the data (based on the project doc).
    ![image](https://user-images.githubusercontent.com/123887929/215349199-8a991ba0-4357-46cc-9e17-30943657c99c.png)
- report
  - client\
    ![image](https://user-images.githubusercontent.com/123887929/215349044-7a6cfc7d-5331-431c-bf05-2bb99752c7d8.png)
- delete ticker
  - client\
  ![image](https://user-images.githubusercontent.com/123887929/215349080-e5314daf-7898-40a4-9a65-814d80a1d0e5.png)


#### Server side threading
- one thread to wait for ctrl + c
- one thread per connection\
![image](https://user-images.githubusercontent.com/123887929/215349370-c0081db2-670b-4c53-9f06-05ee1a24a585.png)

#### Flags in projectone_lib.py
- ALLOW_SHORT_SELL: when ALLOW_SHORT_SELL = True, pos can be negative.
- DEBUG: when DEBUG = True, print additional msg on client's screen.
- USE_LOCAL_DATA: when USE_LOCAL_DATA = True, use local csv file to avoid reaching the api limit

#### Error handles
- server: when port is being used\
![image](https://user-images.githubusercontent.com/123887929/215348542-79fc4735-69e2-472e-aeb7-6cb907ee2231.png)
- client: when server is down, enter any key in client to try to connect to server again\
![image](https://user-images.githubusercontent.com/123887929/215348899-ba6d4be7-e7c1-48db-9ae2-e23c7cb75dc6.png)
- client: ctrl + c\
![image](https://user-images.githubusercontent.com/123887929/215349231-dedce750-74aa-4b3f-9e26-25911d9d0a25.png)
- server: ctrl + c\
![image](https://user-images.githubusercontent.com/123887929/215349255-6a20ee04-e3fe-46ed-8064-38a75c7bc065.png)





