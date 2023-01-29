import argparse
import sys
from projectone_lib import ClientCMIA, validate_host, validate_port

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="connect to server running on the IP address", default="127.0.0.1")
    parser.add_argument("--port", help="network port for the server", default="8000")
    args = vars(parser.parse_args())
    
    l_port = validate_port(args["port"])
    str_host = validate_host(args["host"])
    
    if l_port is None or str_host is None:
        print("Error: Exist invalid input. Please try again.")
        sys.exit()

    myClient = ClientCMIA(host=str_host, port=l_port)
    myClient.run_client()