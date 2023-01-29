import argparse
import sys
from projectone_lib import ServerCMIA, validate_port, validate_sampling

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="comma-separated tickers to download", default="AAPL,MSFT,TOST")
    parser.add_argument("--port", help="network port for the server", default="8000")
    parser.add_argument("--sampling", help="sampling period for the price data in minutes. It only accepts (5,15,30,60) as inputs." , default="5")
    args = vars(parser.parse_args())
    
    l_port = validate_port(args["port"])
    set_tickers = set(args["tickers"].split(","))
    l_sampling = validate_sampling(args["sampling"])
    
    if l_port is None or l_sampling is None:
        print("Error: Exist invalid input. Please try again.")
        sys.exit()
 
    myServer = ServerCMIA(tickers=set_tickers, port=l_port, sampling=l_sampling)
    myServer.run_server()