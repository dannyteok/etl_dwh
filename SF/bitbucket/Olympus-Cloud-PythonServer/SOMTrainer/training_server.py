#! /usr/bin/env python
from server.http_server import start_server
import sys
if __name__ == '__main__':
    if len(sys.argv)>1:
        start_server(sys.argv[1])
    else:
        start_server()
