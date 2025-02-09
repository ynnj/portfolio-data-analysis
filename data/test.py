from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.execution import ExecutionFilter
import time

class TradingApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def execDetails(self, reqId, contract, execution):
        super().execDetails(reqId, contract, execution)
        print("ExecDetails. ReqId:", reqId, "Symbol:", contract.symbol, "SecType:", contract.secType, "Currency:", contract.currency,
              "ExecId:", execution.execId, "OrderId:", execution.orderId, "Shares:", execution.shares, "Price:", execution.price,
              "AvgPrice:", execution.avgPrice, "Side:", execution.side, "CumQty:", execution.cumQty, "Time:", execution.time)

    def execDetailsEnd(self, reqId):
        super().execDetailsEnd(reqId)
        print("ExecDetailsEnd. ReqId:", reqId)

def main():
    app = TradingApp()
    app.connect("127.0.0.1", 7497, clientId=0)  # Replace with your TWS/Gateway IP and port

    # Wait for the connection to be established
    time.sleep(2)

    # Create an execution filter (optional)
    exec_filter = ExecutionFilter()

    # Request executions (trades)
    app.reqExecutions(1001, exec_filter)  # 1001 is a unique request ID

    # Run the app
    app.run()

if __name__ == "__main__":
    main()