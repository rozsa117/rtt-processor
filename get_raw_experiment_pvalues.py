#! /usr/bin/python3

import configparser
from common.rtt_db_conn import *
import sys


# Not used, just an example
def get_test_statistics(conn, test_id):
    with conn.cursor() as c:
        c.execute("""
            SELECT statistics.value FROM statistics
            JOIN subtests ON statistics.subtest_id=subtests.id
            JOIN variants ON subtests.variant_id=variants.id
            WHERE variants.test_id=%s
        """, (test_id, ))

        return [float(x[0]) for x in c.fetchall()]


def main():
    if len(sys.argv) != 2:
        print("USAGE: {} <experiment-id>".format(sys.argv[0]))
        sys.exit(1)

    exp_id = int(sys.argv[1])

    cfg = configparser.ConfigParser()
    cfg.read("config.ini")
    conn = create_mysql_db_conn(cfg)
    
    #########################
    # Write your query here #
    #########################
    with conn.cursor() as c: 
        c.execute("""
            SELECT name FROM experiments WHERE id=%s
        """, (exp_id, ))
        for result in c.fetchall():
            print(result[0])

    conn.close()


if __name__ == "__main__":
    main()
