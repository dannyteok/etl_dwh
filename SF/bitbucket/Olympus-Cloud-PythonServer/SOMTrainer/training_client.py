import os
import argparse
import json
import sys
import time



if __name__ == '__main__':
    import socket
    import threading

    parser = argparse.ArgumentParser(description='heartbeat SOM Trainer')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    group.add_argument('-c', nargs='?', dest='configuration_json', type=str, help="the filepath to the configuration file")
    parser.add_argument('--multiprocessing', action='store_false', help="Use multiprocessing for training")
    parser.add_argument('-w', '--workers', dest='workers', default=8, type=int, help='the number of multiprocessing workers')

    parser.add_argument('--nios', dest='ios', action='store_false', help="ios training")
    parser.add_argument('--nandroid', dest='android', action='store_false', help="ios training")

    args = parser.parse_args()
    if args.configuration_json:
        with open(args.configuration_json, 'r') as configuration_file:
            configuration_json = json.load(configuration_file)
    else:
        configuration_json = json.loads(args.infile.readlines()[0])

    # Connect to the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 8007))
    wait = True
    # Send the data
    message = json.dumps(configuration_json)
    print ('Sending : "%s"' % message)
    len_sent = s.send(message)

    # Receive a response
    response = s.recv(1024)
    job_id = response
    print ('Received: "%s"' % response)
    while wait:
        print ('sending state')
        s.send(job_id)
        response = s.recv(1024)
        print ('{} Received: "{}"'.format(job_id, response))
        if response == 'completed':
            wait = False
        time.sleep(5)


    # Clean up
    s.close()
