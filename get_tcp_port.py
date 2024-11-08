#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import socket


def get_tcp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            port = random.randint(50000, 59999)
            sock.bind(("127.0.0.1", port))
            sock.close()
            return port
        except:
            pass


def main():
    print("Distributed TCP PORT | {}".format(get_tcp_port()))


if __name__ == '__main__':
    main()