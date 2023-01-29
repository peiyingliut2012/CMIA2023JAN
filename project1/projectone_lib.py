import os
import re
import sys
import socket
import logging
import _thread
import requests
import traceback
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

# Flags for debug
DEBUG = False
USE_LOCAL_DATA = False

# Config params
ALLOW_SHORT_SELL = True  # when ALLOW_SHORT_SELL = True, pos can be negative
lst_valid_sampling = [5, 15, 30, 60]
API_KEY1 = "O61HVUYSUTY793CZ"
API_KEY2 = "cfb9h49r01qqlprlim10cfb9h49r01qqlprlim1g"
LOG_FOLDER = os.path.join(os.getcwd(), "log")


def log_it(str_msg, force_print=False):
    """ write to log """
    logging.info(str_msg)
    if DEBUG or force_print:
        print(str_msg)

# ------------------------------
# for ctrl + c
# ------------------------------


def exit_by_ctrl_c(str_from="Server"):
    """ exit """
    log_it("{0} says Bye".format(str_from), True)
    os._exit(0)


def exit_by_ctrl_c_helper():
    """ exit helper for server """
    try:
        while True:
            command = input()
    except Exception as e:
        exit_by_ctrl_c()

# ------------------------------
# input validations
# ------------------------------


def validate_port(port):
    """valid port: 1024 - 65535"""
    l_port = None
    try:
        l_port = int(port)
        if l_port < 1024 or l_port > 65535:
            l_port = None
    except Exception as e:
        pass

    if l_port is None:
        log_it(
            "Invalid port: -- port {0}\nport must be an integer between 1024 and 65535.\n".format(
                port), True
        )
    return l_port


def validate_host(host):
    """valid host: XXX.XXX.XXX.XXX"""
    str_host = None
    regex = "^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])$"
    if(re.search(regex, host)):
        str_host = host
    if str_host is None:
        log_it(
            "Invalid host: -- host {0}\n".format(
                host), True
        )
    return str_host


def validate_sampling(sampling):
    """valid sampling: 5, 15, 30, 60"""
    l_sampling = None

    try:
        l_sampling = int(sampling)
        if l_sampling not in lst_valid_sampling:
            l_sampling = None
    except Exception as e:
        pass

    if l_sampling is None:
        log_it(
            "Invalid input: -- sampling {0}\nValid sampl41ings are {1}\n".format(
                sampling,
                ", ".join([str(e) for e in lst_valid_sampling])), True
        )

    return l_sampling

# ------------------------------
# repot utils
# ------------------------------


def get_signal(l_avg, l_std, l_price):
    """
    get action signal for buy 1, sell 1 or no action
    """
    l_ret = 0
    if l_price > l_avg + l_std:
        l_ret = 1
    elif l_price < l_avg - l_std:
        l_ret = -1
    return l_ret


def get_pnl(pos, df_rolling):
    """
    get pnl: PnL(t) = Pos(t-1) x [S(t) - S(t-1)]
    """
    np_price = df_rolling.loc[pos.index, "price"]
    return pos[0] * (np_price[1] - np_price[0])


def add_pos(df_rolling):
    """
    get position based on past signals.
    """
    np_pos = df_rolling.signal.values.copy()
    cur_pos = 0

    for l_idx, l_action in np.ndenumerate(np_pos):
        if ALLOW_SHORT_SELL:
            cur_pos = cur_pos + l_action
        else:
            cur_pos = max(0, cur_pos + l_action)
        np_pos[l_idx[0]] = cur_pos
    df_rolling["pos"] = np_pos

# ------------------------------
# request utils
# ------------------------------


def send_msg(conn, str_msg):
    """
    send a len msg first
    then full msg
    """
    byte_msg = str_msg.encode()
    byte_msg_len = str(len(str_msg)).encode()
    conn.send(byte_msg_len)
    conn.send(byte_msg)


def get_request_clean(str_request):
    """clean user's input"""
    return str_request.lower().strip()


def get_request_type(str_request):
    """get request type"""
    str_request_type = ""
    str_request_clean = get_request_clean(str_request)
    if str_request_clean.startswith("data"):
        str_request_type = "data"
    elif str_request_clean.startswith("delete"):
        str_request_type = "delete_ticker"
    elif str_request_clean.startswith("add"):
        str_request_type = "add_ticker"
    elif str_request_clean == "report":
        str_request_type = "report"
    else:
        str_request_type = "invalid_request"
    return str_request_type

# ------------------------------
# other utils
# ------------------------------


def roundTime(dt, roundTo=60):
    """ 
    roundTo is in seconds 
    roundTime(dt, 5 * 60): round to the nearest 5min
    """
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + timedelta(0, rounding-seconds, -dt.microsecond)


class ServerCMIA:

    def __init__(self, tickers, port, sampling):
        self.tickers = set()  # to avoid dups
        self.tickers_candidate = tickers
        self.port = port
        self.sampling = sampling
        self.report = pd.DataFrame()
        self.l_timeout = 0.5
        self.l_listen_num = 5
        # store temp data. self.report_data = {"AAPL": {"price_data": df1, "report_data": df2}}
        self.report_data = {}
        self.df_report = pd.DataFrame()  # store final report data
        self.url_s1 = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={0}&interval={1}min&apikey={2}&outputsize=full"
        self.url_s2 = "https://finnhub.io/api/v1/quote?symbol={0}&token={1}"

    # ------------------------------
    # download: get data from source 1 and source 2
    # ------------------------------
    def get_data_from_source1(self, str_ticker):
        """ 
        get data from source 1 
        return None if ticker is not valid
        """

        df_price = None
        url = self.url_s1.format(str_ticker, self.sampling, API_KEY1)
        r = requests.get(url)
        data = r.json()
        if "Error Message" in data:
            log_it("Invalid ticker: {0}".format(str_ticker), True)
        else:
            df_price = pd.DataFrame.from_dict(data["Time Series ({0}min)".format(self.sampling)], orient="index").reset_index()[["index", "4. close"]].rename(columns={
                "index": "datetime",
                "4. close": "price",
            })
            df_price["datetime"] = pd.to_datetime(df_price["datetime"])
            df_price.sort_values(by="datetime", ascending=True, inplace=True)
        return df_price

    def get_data_from_source2(self, str_ticker, df_price):
        """ 
        get data from srouce2
        check timestamp of source2:
        if < the last price in source1, do not add
        else: add to df_price
        """
        url = self.url_s2.format(str_ticker, API_KEY2)
        r = requests.get(url)
        res = r.json()
        dt_current_price_time = datetime.fromtimestamp(res["t"]).replace(tzinfo=None)
        dt_last_price_time = pd.to_datetime(df_price.datetime.values[-1]).to_pydatetime().replace(tzinfo=None)

        log_it("last price time in source1: {0}".format(dt_last_price_time) , True)
        log_it("current price time in source2: {0}".format(dt_current_price_time) , True)
        l_sec_diff = (dt_current_price_time - dt_last_price_time).total_seconds()

        if l_sec_diff <= 0:
            log_it("data already available in source 1. ignore source 2.", True)
        else:
            if l_sec_diff <= self.sampling * 60:
                dt_current_price_time = dt_current_price_time + \
                    timedelta(minutes=self.sampling)
            else:
                dt_current_price_time = roundTime(
                    dt_current_price_time, self.sampling * 60)
            df = pd.DataFrame(
                [{"price": res["c"], "datetime": dt_current_price_time}])
            df.set_index("datetime")
            log_it("add source2: {0}".format(df), True)
            df_price = pd.concat([df_price, df])
        return df_price

    def get_price_data(self, str_ticker):
        """ 
        get price data 
        return none if ticker is not valid
        """
        log_it("---- get price data: start", True)
        df_price = self.get_data_from_source1(str_ticker)
        if df_price is not None:
            df_price = self.get_data_from_source2(str_ticker, df_price)
            df_price["datetime"] = pd.to_datetime(df_price["datetime"])
        log_it(df_price)
        log_it("---- get price data: complete", True)
        return df_price

    # ------------------------------
    # report: preprcoess, postprocess, gen, refresh
    # ------------------------------

    def preprocess_new_ticker(self, str_ticker, refresh=False):
        """ 
        Validate ticker, get price data and save to report_data.
        return True if the ticker is valid.

        note: if USE_LOCAL_DATA (testing), use local file to avoid pulling.
        """
        b_valid = True
        if str_ticker not in self.report_data or refresh:
            df_price = pd.DataFrame()
            if USE_LOCAL_DATA:
                str_filename = "{0}.csv".format(str_ticker)
                if os.path.isfile(str_filename):
                    df_price = pd.read_csv(str_filename)
                    df_price["datetime"] = pd.to_datetime(df_price["datetime"])
                else:
                    log_it("USE_LOCAL_DATA: {0} does not exist".format(
                        str_filename), True)
                    b_valid = False
            else:
                df_price = self.get_price_data(str_ticker)
                if df_price is None:
                    b_valid = False
                else:
                    df_price.to_csv("{0}.csv".format(str_ticker), index=False)
            if b_valid:
                self.report_data[str_ticker] = {
                    "price_data": df_price, "report_data": pd.DataFrame()}
                self.tickers.add(str_ticker)
        else:
            log_it("ticker is already available. do nothing.", True)
        return b_valid

    def postprocess_new_ticker(self, str_ticker):
        """ postprocess price data and save to report_data """
        df_rolling = self.report_data[str_ticker]["price_data"].copy()

        # get avg and std
        df_rolling["priceavg"] = df_rolling["price"]
        df_rolling["pricestd"] = df_rolling["price"]
        df_rolling.set_index("datetime", inplace=True)
        df_rolling = df_rolling.rolling("1D").agg(
            {"priceavg": "mean", "pricestd": "std", "price": lambda rows: rows[-1]})

        # clean up null
        df_rolling = df_rolling[~df_rolling.priceavg.isnull()]

        # add signal
        df_rolling["signal"] = list(map(
            lambda l_avg, l_std, l_price: get_signal(l_avg, l_std, l_price),
            df_rolling.priceavg,
            df_rolling.pricestd,
            df_rolling.price))

        # add pos
        add_pos(df_rolling)

        # add pnl
        df_rolling["pnl"] = df_rolling.rolling(2)["pos"].apply(
            get_pnl, args=(df_rolling,), raw=False)

        # format report
        df_report = df_rolling[["price", "signal", "pnl"]].reset_index()
        df_report = df_report[~df_report.pnl.isnull()]
        df_report.insert(1, "ticker", str_ticker, True)
        df_report["datetime"] = pd.to_datetime(
            df_report["datetime"]).dt.strftime("%Y-%m-%d-%H:%M")
        df_report["signal"] = df_report["signal"].astype(int)
        self.report_data[str_ticker] = {
            "price_data": pd.DataFrame(), "report_data": df_report}

    def gen_report(self):
        """
        gen a new csv report using the data in report_data
        clean up report_data after done
        """
        lst_df = []
        for str_ticker in self.report_data:
            lst_df.append(self.report_data[str_ticker]["report_data"])
            self.report_data[str_ticker] = {}
        if lst_df:
            df_report = pd.concat(lst_df)
            df_report.sort_values(by=["datetime", "ticker"],
                                inplace=True, ascending=True)
            self.df_report = df_report
            df_report.to_csv("report.csv", index=False)
        else:
            log_it("Failed to gen report. No valid ticker.", True)

    def process_report_refresh(self):
        """
        refresh report: preprocess, postprocess and gen report
        """
        ret = ""
        try:
            for str_ticker in self.tickers:
                self.preprocess_new_ticker(str_ticker, refresh=True)
                self.postprocess_new_ticker(str_ticker)
            if self.tickers:
                self.gen_report()
                ret = "Successfully updated report for {0}".format(", ".join(self.tickers))
            else:
                ret = "No valid ticker. report is empty."
        except Exception as e:
            ret = "Failed to update report."
        return ret

    # ------------------------------
    # process user reuqests
    # ------------------------------

    def process_ticker_delete(self, str_ticker):
        """ Returns 0=success, 1=server error, 2=ticker not found """
        ret = "2"
        if str_ticker in self.tickers:
            self.tickers.remove(str_ticker)
            del self.report_data[str_ticker]
            self.df_report = self.df_report[self.df_report.ticker != str_ticker]
            ret = "0"
        return ret

    def process_ticker_add(self, str_ticker):
        """ Returns 0=success, 1=server error, 2=invalid ticker """
        ret = "0"
        if str_ticker not in self.tickers:
            # note: do not postprocess or gen report b/c only update when call refresh report
            b_valid = self.preprocess_new_ticker(str_ticker)
            if not b_valid:
                ret = "2"
        return ret

    def process_data_request(self, str_from_client):
        """ 
        process data 
        if no datetime, use the last time stamp
        """
        str_ret = ""
        if self.df_report.shape[0]:
            str_datetime = str_from_client.strip()
            if len(str_datetime) == 4:
                str_datetime = self.df_report.datetime.values[-1]
            elif " " in str_datetime:
                str_datetime = str_datetime.split(" ")[1]

            df_ret = self.df_report[self.df_report.datetime ==
                                    str_datetime][["ticker", "price", "signal"]]

            if df_ret.shape[0]:
                df_ret["ret_str"] = df_ret["ticker"] + "\t" + \
                    df_ret["price"].map("{:,.2f}".format) + \
                    "," + df_ret["signal"].astype(str)
                str_ret = "\n".join(df_ret["ret_str"].values)
            else:
                str_ret = "Server has no data"
        else:
            str_ret = "Server has no data"

        return str_ret

    def check_server(self):
        """ admin function: check server status"""
        log_it("tickers: {0}".format(self.tickers), True)
        log_it("port: {0}".format(self.port), True)
        log_it("sampling: {0}".format(self.sampling), True)
        log_it("report: {0}".format(self.df_report.shape), True)
        log_it(self.df_report.head(), True)

    def exec_client_request(self, conn, addr):
        try:
            while True:
                str_from_client = conn.recv(1024)

                if not str_from_client:
                    log_it("client {0} | no data continue".format(addr), True)
                    conn.close()
                    break

                str_from_client = str_from_client.decode("utf-8")
                log_it("client {1} | msg from client: {0}".format(
                    str_from_client, addr), True)

                # process request
                str_request_type = get_request_type(str_from_client)
                log_it("client {1} | request type: {0}".format(
                    str_request_type, addr), True)

                str_request_clean = get_request_clean(str_from_client)

                str_to_client = ""
                if str_request_type == "data":
                    str_to_client = self.process_data_request(
                        str_request_clean)
                elif str_request_type == "add_ticker":
                    str_to_client = self.process_ticker_add(
                        str_request_clean.split(" ")[1].strip().upper())
                elif str_request_type == "delete_ticker":
                    str_to_client = self.process_ticker_delete(
                        str_request_clean.split(" ")[1].strip().upper())
                elif str_request_type == "report":
                    str_to_client = self.process_report_refresh()

                if not str_to_client:
                    str_to_client = "Something is wrong. Please try again."

                # respond
                send_msg(conn, str_to_client)
                log_it("client {0} | sent msg: {1}".format(addr, str_to_client), True)

                conn.close()
                log_it("client {0} | closed".format(addr), True)
                break
        except ConnectionResetError:
            log_it("client forcibly closed", True)

    def run_server(self):
        """ run server """

        # server log file
        logging.basicConfig(
            filename="{2}\\server_{0}_{1}.log".format(
                datetime.now().strftime("%Y%m%d_%H%M%S"), os.getpid(), LOG_FOLDER),
            filemode="w", format="%(asctime)s %(levelname)s - %(message)s", level=logging.DEBUG)

        log_it("server init: start", True)
        serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # start
        try:
            serv.bind(("localhost", self.port))
        except Exception as e:
            log_it("Error: Failed to init port {0}. Please try again or use a different port.".format(
                self.port), True)
            sys.exit()

        for str_ticker in self.tickers_candidate:
            self.preprocess_new_ticker(str_ticker)
        for str_ticker in self.tickers:
            self.postprocess_new_ticker(str_ticker)
            log_it("processed ticker {0}".format(str_ticker), True)
        self.gen_report()
        log_it("gen_report done", True)
        log_it("server init: complete", True)


        # listen
        serv.listen(self.l_listen_num)
        serv.settimeout(self.l_timeout)

        # wait
        try:
            while True:
                try:
                    conn, addr = serv.accept()
                    log_it("-" * 10, True)
                    log_it("init client connection : {0}".format(addr), True)

                    # breakable by ctrl+c
                    _thread.start_new_thread(exit_by_ctrl_c_helper, ())
                    _thread.start_new_thread(
                        self.exec_client_request, (conn, addr, ))

                except socket.timeout:
                    pass
                except KeyboardInterrupt:
                    exit_by_ctrl_c()
                except ConnectionResetError:
                    log_it("client forcibly closed", True)
                except Exception as e:
                    log_it("Error: {0}".format(traceback.format_exc()), True)
        except KeyboardInterrupt:
            exit_by_ctrl_c()


class ClientCMIA:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def get_response(self, client, str_server_error_msg="1"):
        # get response length
        str_from_server = ""
        try:
            byte_msg_len = client.recv(1024)
            if byte_msg_len:
                l_msg_len = int(byte_msg_len.decode("utf-8"))
                log_it("response from server: len={0}".format(l_msg_len))

                # get response string
                while len(str_from_server) < l_msg_len:
                    byte_data = client.recv(1024)
                    if not byte_data:
                        break
                    else:
                        str_from_server += byte_data.decode("utf-8")
                log_it("response from server:\n{0}".format(str_from_server))

            else:
                log_it("no response from server")
        except Exception as e:
            str_from_server = str_server_error_msg
        log_it(str_from_server, True)

    def run_client(self):
        """ run client """

        # log file for client
        logging.basicConfig(
            filename="{2}\\client_{0}_{1}.log".format(
                datetime.now().strftime("%Y%m%d_%H%M%S"), os.getpid(), LOG_FOLDER),
            filemode="w", format="%(asctime)s %(levelname)s - %(message)s", level=logging.DEBUG)

        # init client
        try:
            while True:

                try:
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.connect((self.host, self.port))
                except Exception:
                    log_it(
                        "Failed to connect to server. Press enter to try again.", True)
                    str_msg = str(input(">"))
                    continue

                str_msg = str(input(">"))
                log_it("-" * 10)
                log_it("user input: {0}".format(str_msg))

                str_request_type = get_request_type(str_msg)
                if str_request_type == "invalid_request":
                    log_it(
                        "This request is not supported. Supported requests are data, delete, add, report.", True)
                else:
                    # send request
                    try:
                        byte_msg = str_msg.encode("utf-8")
                        client.send(byte_msg)
                        log_it("sent to server")
                        if str_request_type in ["add_ticker", "delete_ticker"]:
                            self.get_response(client)
                        elif str_request_type == "report":
                            self.get_response(
                                client, "Failed to update report. Please try again.")
                        elif str_request_type == "data":
                            self.get_response(
                                client, "Failed to get data. Please try again.")
                    except ConnectionResetError:
                        log_it("Server is not responding. Please try again.", True)
                    except Exception:
                        log_it("Something is wrong. Please try again.", True)

                # close client
                client.close()
                log_it("client closed")

        except KeyboardInterrupt:
            exit_by_ctrl_c("Client")
