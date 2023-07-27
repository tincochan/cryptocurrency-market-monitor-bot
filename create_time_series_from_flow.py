#!/usr/bin/python3
"""
Crete time series for future analysis from CSV flow file.

author:Tinc0 CHAN
e-mail: tincochan@foxmail.com, koumar@cesnet.cz

Copyright (C) 2022 CESNET

LICENSE TERMS

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the Company nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

ALTERNATIVELY, provided that this notice is retained in full, this product may be distributed under the terms of the GNU General Public License (GPL) version 2 or later, in which case the provisions of the GPL apply INSTEAD OF those given above.

This software is provided as is'', and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the company or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
"""
# Local Application Imports
import sys
import os
import csv
import json
import ipaddress
import argparse
from argparse import RawTextHelpFormatter
import time
import datetime
from enum import IntEnum, unique

csv.field_size_limit(sys.maxsize)

from create_datapoint import proces_flow


@unique
class CSVFields(IntEnum):
    ID_DEPENDENCY = 0
    SUM_FLOWS = 1
    SUM_PACKETS = 2
    SUM_BYTES = 3
    PACKETS = 4
    BYTES = 5
    T1 = 6
    T2 = 7


def make_output_csv(time_series: dict, arg: argparse.Namespace):
    """Create output for time series in CSV file format.

    Args:
        time_series (dict): Time series in dictionary format.
        arg (argparse.Namespace): Arguments of module.
        cnt (int): Count of file.
    """
    if arg.time_series == "":
        file = f"{arg.flowfile.split('_dependencies.csv')[0]}_time_series.csv"
    else:
        file = arg.time_series
    if os.path.isfile(file):
        pass
    else:
        cnt_ts = 0
        print(f"     Number of time series: {cnt_ts}", end="")
        with open(file, "w") as f:
            writer = csv.writer(f, delimiter=";")
            for key in time_series:
                cnt_ts += 1
                print(f"\r     Number of time series: {cnt_ts}", end="", flush=True)
                if arg.bytes is None:
                    sum_bytes = 0
                    arr_bytes = []
                else:
                    sum_bytes = time_series[key]["sum_bytes"]
                    arr_bytes = time_series[key]["bytes"]
                if (
                    arg.qheuristic != 0
                    and len(time_series[key]["packets"]) > arg.qheuristic
                ):
                    writer.writerow(
                        [
                            key,
                            time_series[key]["sum_flows"],
                            time_series[key]["sum_packets"],
                            sum_bytes,
                            time_series[key]["packets"][: arg.qheuristic],
                            arr_bytes,
                            time_series[key]["start_time"][: arg.qheuristic],
                            time_series[key]["end_time"][: arg.qheuristic],
                            time_series[key]["labels"][: arg.qheuristic],
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            key,
                            time_series[key]["sum_flows"],
                            time_series[key]["sum_packets"],
                            sum_bytes,
                            time_series[key]["packets"],
                            arr_bytes,
                            time_series[key]["start_time"],
                            time_series[key]["end_time"],
                            time_series[key]["labels"],
                        ]
                    )
        print("")
        return
    if os.path.isfile(f"{file.split('.csv')[0]}_tmp.csv"):
        pass
    else:
        f = open(f"{file.split('.csv')[0]}_tmp.csv", "x")

    cnt_ts = 0
    print(f"     Number of ts: {cnt_ts}", end="")
    with open(f"{file.split('.csv')[0]}_tmp.csv", "a") as wf:
        writer = csv.writer(wf, delimiter=";")
        with open(file, "r") as rf:
            reader = csv.reader(rf, delimiter=";")
            for row in reader:
                cnt_ts += 1
                print(f"\r     Number of flows: {cnt_ts}", end="", flush=True)
                key = row[CSVFields.ID_DEPENDENCY]
                if key in time_series:
                    row[CSVFields.SUM_FLOWS] = (
                        int(row[CSVFields.SUM_FLOWS]) + time_series[key]["sum_flows"]
                    )
                    row[CSVFields.SUM_PACKETS] = (
                        int(row[CSVFields.SUM_PACKETS])
                        + time_series[key]["sum_packets"]
                    )
                    if arg.bytes is not None:
                        row[CSVFields.SUM_BYTES] = (
                            int(row[CSVFields.SUM_BYTES])
                            + time_series[key]["sum_bytes"]
                        )
                    packets = json.loads(row[CSVFields.PACKETS])
                    if arg.qheuristic != 0 and len(packets) >= arg.qheuristic:
                        pass
                    else:
                        packets.extend(time_series[key]["packets"])
                        if arg.qheuristic != 0 and len(packets) > arg.qheuristic:
                            row[CSVFields.PACKETS] = packets[: arg.qheuristic]
                        else:
                            row[CSVFields.PACKETS] = packets
                        if arg.bytes is not None:
                            bytes = json.loads(row[CSVFields.BYTES])
                            bytes.extend(time_series[key]["bytes"])
                            if arg.qheuristic != 0 and len(bytes) > arg.qheuristic:
                                row[CSVFields.BYTES] = bytes[: arg.qheuristic]
                            else:
                                row[CSVFields.BYTES] = bytes
                        t1 = json.loads(row[CSVFields.T1])
                        t1.extend(time_series[key]["start_time"])
                        if arg.qheuristic != 0 and len(t1) > arg.qheuristic:
                            row[CSVFields.T1] = t1[: arg.qheuristic]
                        else:
                            row[CSVFields.T1] = t1
                        t2 = json.loads(row[CSVFields.T2])
                        t2.extend(time_series[key]["end_time"])
                        if arg.qheuristic != 0 and len(t2) > arg.qheuristic:
                            row[CSVFields.T2] = t2[: arg.qheuristic]
                        else:
                            row[CSVFields.T2] = t2
                writer.writerow(row)
    print("")
    os.remove(file)
    os.rename(f"{file.split('.csv')[0]}_tmp.csv", file)


def get_indexes_from_header(header, arg):
    i_packets = header.index(arg.packets)
    i_start_time = header.index(arg.start_time)
    if arg.end_time is not None:
        i_end_time = header.index(arg.end_time)
    else:
        i_end_time = None
    if arg.duration is not None:
        i_duration = header.index(arg.duration)
    else:
        i_duration = None
    if arg.packets_rev is not None:
        i_packets_rev = header.index(arg.packets_rev)
    else:
        i_packets_rev = None
    if arg.bytes is not None:
        i_bytes = header.index(arg.bytes)
        if arg.packets_rev is not None:
            i_bytes_rev = header.index(arg.bytes_rev)
        else:
            i_bytes_rev = None
    else:
        i_bytes = None
        i_bytes_rev = None
    if arg.label != "":
        i_label = header.index(arg.label)
    else:
        i_label = None
    return {
        "i_packets": i_packets,
        "i_packets_rev": i_packets_rev,
        "i_bytes": i_bytes,
        "i_bytes_rev": i_bytes_rev,
        "i_start_time": i_start_time,
        "i_end_time": i_end_time,
        "i_duration": i_duration,
        "i_label": i_label,
    }


def get_info_from_row(row, arg, indexes):
    id_dependency = row[CSVFields.ID_DEPENDENCY]
    packets = int(row[indexes["i_packets"]])
    if arg.packets_rev is not None:
        packets += int(row[indexes["i_packets_rev"]])
    if arg.bytes is not None:
        bytes = int(row[indexes["i_bytes"]])
        if arg.packets_rev is not None:
            bytes += int(row[indexes["i_bytes_rev"]])
    else:
        bytes = 0
    try:
        start_time = time.mktime(
            datetime.datetime.strptime(
                row[indexes["i_start_time"]], arg.time_format
            ).timetuple()
       )
    except:
        start_time = time.mktime(
            datetime.datetime.strptime(
                row[indexes["i_start_time"]], arg.time_format[:-3]
            ).timetuple()
       )
    if arg.end_time is not None:
        try:
            end_time = time.mktime(
                datetime.datetime.strptime(
                    row[indexes["i_end_time"]], arg.time_format
                ).timetuple()
            )
        except:
            end_time = time.mktime(
                datetime.datetime.strptime(
                    row[indexes["i_end_time"]], arg.time_format[:-3]
                ).timetuple()
            )
    elif arg.duration is not None:
        end_time = start_time + float(row[indexes["i_duration"]])
    else:
        end_time = start_time
        
    if indexes["i_label"] is not None:
        label = row[indexes["i_label"]]
    else:
        label = ""

    return {
        "id_dependency": id_dependency,
        "packets": packets,
        "bytes": bytes,
        "start_time": start_time,
        "end_time": end_time,
        "label": label,
    }


def parse_arguments():
    """Function for set arguments of module.

    Returns:
        argparse: Return setted argument of module.
    """
    parser = argparse.ArgumentParser(
        description="""

    Usage:""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--flowfile",
        help="CSV file with flows containting column with ID_DEPENDENCY.",
        type=str,
        metavar="NAME.SUFFIX",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--time_series",
        help="CSV file as output of time series. Default is setted automaticly.",
        type=str,
        metavar="NAME.SUFFIX",
        default="",
    )
    parser.add_argument(
        "--size",
        help="Size in bytes that is maximum memory usage of time series before save to csv. (default is set to 100 000 000)",
        type=int,
        metavar="bytes",
        default=100000000,
    )
    parser.add_argument(
        "-q",
        "--qheuristic",
        help="Enable heuristic for long time series. When set to some number, all time series that are longer then this number will be cut, and periodics will be searched in this shorter time series.",
        type=int,
        metavar="NUMBER",
        default=0,
    )
    parser.add_argument(
        "-N",
        "--networks",
        help='IP networks (in CIDR format) to monitor. Only data of IPs from these networks will be included. Multiple networks can be specified as "-N 192.168.1.0/24 10.0.0.0/8". If not set, all IPs are included.',
        type=str,
        nargs="+",
        metavar="IPs",
        default=None,
    )
    parser.add_argument(
        "--delimiter",
        help="Specification of delimiter in flow CSV file. Default set to ','.",
        type=str,
        metavar="STRING",
        default=",",
    )
    parser.add_argument(
        "--packets",
        help="Number of packets in flow field in flow csv file. Default set to packets.",
        type=str,
        metavar="STRING",
        default="packets",
    )
    parser.add_argument(
        "--packets_rev",
        help="Number of packets in flow field in flow csv file. Default set to packets.",
        type=str,
        metavar="STRING",
        default=None,
    )
    parser.add_argument(
        "--bytes",
        help="Number of bytes in flow field in flow csv file. Default set to bytes.",
        type=str,
        metavar="STRING",
        default=None,
    )
    parser.add_argument(
        "--bytes_rev",
        help="Number of bytes in flow field in flow csv file. Default set to bytes.",
        type=str,
        metavar="STRING",
        default=None,
    )
    parser.add_argument(
        "--start_time",
        help="First time of flow field in flow csv file. Default set to t1.",
        type=str,
        metavar="STRING",
        default="t1",
    )
    parser.add_argument(
        "--end_time",
        help="End time of flow field in flow csv file. Default set to None.",
        type=str,
        metavar="STRING",
        default=None,
    )
    parser.add_argument(
        "--duration",
        help="Duration of flow field in flow csv file. Default set to None.",
        type=str,
        metavar="STRING",
        default=None,
    )
    parser.add_argument(
        "--label",
        help="Label field in flow csv file. Default set to None.",
        type=str,
        metavar="STRING",
        default="",
    )
    parser.add_argument(
        "--time_format",
        help="Set time format for time fields.",
        type=str,
        metavar="STRING",
        default="%d/%m/%Y",
    )
    parser.add_argument(
        "--cutting_timeout",
        help="Set cutting format for TS in s.",
        type=int,
        metavar="NUMBER",
        default=21600,
    )
    arg = parser.parse_args()
    if arg.flowfile is None:
        print("CSV flow file not set using -f or --flowfile argument.")
        sys.exit(1)
    return arg


def main():
    """Main function of the module."""
    arg = parse_arguments()
    if arg.networks is not None:
        networks = []
        for net in arg.networks:
            networks.append(ipaddress.ip_network(net))
    time_series = {}
    q_dependencies = []
    # cycle through flows from CSV file
    cnt_flows = 0
    print(f"     Number of flows: {cnt_flows}", end="")
    with open(arg.flowfile, "r") as f:
        reader = csv.reader(f, delimiter=arg.delimiter)
        indexes = []
        for row in reader:
            if row[0] == "ID_DEPENDENCY":
                indexes = get_indexes_from_header(row, arg)
                continue
            # print numbers of flows
            cnt_flows += 1
            print(f"\r     Number of flows: {cnt_flows}", end="", flush=True)
            # get information from row
            inf = get_info_from_row(row, arg, indexes)
            # proces information to time series
            if arg.bytes is None:
                enable_bytes = False
            else:
                enable_bytes = True
            proces_flow(
                time_series,
                q_dependencies,
                arg.qheuristic,
                inf["id_dependency"],
                inf["packets"],
                inf["bytes"],
                inf["start_time"],
                inf["end_time"],
                label=inf["label"],
                enable_bytes=enable_bytes,
                cutting_timeout=arg.cutting_timeout,
            )
    print("")
    print(f"Make output:")
    if time_series != {}:
        make_output_csv(time_series, arg)
    print("Make output - OK")


if __name__ == "__main__":
    main()
