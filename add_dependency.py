#!/usr/bin/python3
"""
Analyze IP flows to get information about use of dependencies between devices in (private) networks.
Add dependency info to flows and send flows to output.

author:Tinc0 CHAN
e-mail: tincochan@foxmail.com

Copyright (C) 2022 CESNET

LICENSE TERMS

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the Company nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

ALTERNATIVELY, provided that this notice is retained in full, this product may be distributed under the terms of the GNU General Public License (GPL) version 2 or later, in which case the provisions of the GPL apply INSTEAD OF those given above.

This software is provided as is'', and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the company or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
"""
# Standard libraries imports
import sys
import os
import csv
import argparse
from argparse import RawTextHelpFormatter

import ipaddress

# NEMEA system library
# import pytrap

import math
import faulthandler

from pylibpcap.pcap import rpcap
from packet import *


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def check_port(port: int, ports_tb: dict):
    """Check if port used in dependency is service or registered.

    Args:
        port (int): Integer of used port by device.
        ports_tb (dic): Dictionary contains registered port defined by IANA and ICAN.

    Returns:
        bool: True if port is service or registered.
    """
    p = ports_tb.get(port)
    if p is not None:
        if p == "":
            return False
        return True
    return False


def recognize_dependency(ports_tb: dict, src_ip, src_port, dst_ip, dst_port):
    """Recognize dependency and return it.

    Args:
        rec (UNIREC): IP flow record.
        ports_tb (dict): Dictionary of protocol numbers and names registered by IANA and ICAN.

    Returns:
        str: Dependency.
    """
    if check_port(src_port, ports_tb) and check_port(dst_port, ports_tb):
        if dst_port < 1024:
            return f"{dst_ip}({dst_port})-{src_ip}"
        else:
            return f"{src_ip}({src_port})-{dst_ip}"
    elif check_port(dst_port, ports_tb):
        return f"{dst_ip}({dst_port})-{src_ip}"
    elif check_port(src_port, ports_tb):
        return f"{src_ip}({src_port})-{dst_ip}"
    else:
        return f"{src_ip}({src_port}-{dst_port})-{dst_ip}"


def load_pytrap(argv: list):
    """Init nemea libraries and set format of IP flows.

    Returns:
        tuple: Return tuple of rec and trap. Where rec is template of IP flows and trap is initialized pytrap NEMEA library.
    """
    trap = pytrap.TrapCtx()
    trap.init(argv, 1, 1)  # argv, ifcin - 1 input IFC, ifcout - 1 output IFC
    # Set the list of required fields in received messages.
    # This list is an output of e.g. flow_meter - basic flow.
    inputspec = "ipaddr DST_IP,ipaddr SRC_IP,time TIME_FIRST,time TIME_LAST,uint32 PACKETS,uint32 PACKETS_REV,uint64 BYTES,uint64 BYTES_REV,uint16 DST_PORT,uint16 SRC_PORT,uint8 PROTOCOL"
    trap.setRequiredFmt(0, pytrap.FMT_UNIREC, inputspec)
    rec = pytrap.UnirecTemplate(inputspec)

    alertspec = "string ID_DEPENDENCY,ipaddr DST_IP,ipaddr SRC_IP,time TIME_FIRST,time TIME_LAST,uint32 PACKETS,uint32 PACKETS_REV,uint64 BYTES,uint64 BYTES_REV,uint16 DST_PORT,uint16 SRC_PORT,uint8 PROTOCOL"
    alert = pytrap.UnirecTemplate(alertspec)
    trap.setDataFmt(0, pytrap.FMT_UNIREC, alertspec)
    alert.createMessage()
    return rec, trap, alert


def parse_arguments():
    """Parse program arguments using the argparse module.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="""Module for add information about Network Dependency for each flow. Supported inputs of flows are NEMEA IFC, flow CSV file.

    Usage:""",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--inputtype",
        help="Select the input type of the module. Supported types are: nemea, flow-csv, pcap. Default set to nemea.",
        type=str,
        metavar="NAME",
        default="nemea",
    )
    parser.add_argument(
        "-P",
        "--ports",
        help="Set the name with suffix of file, where are safe registered ports (default: Ports.csv). File must be .csv",
        type=str,
        metavar="NAME.SUFFIX",
        default="Ports.csv",
    )
    parser.add_argument(
        "-i",
        help='Specification of interfaces types and their parameters, see "-h trap".',
        type=str,
        metavar="IFC_SPEC",
    )
    parser.add_argument(
        "-f",
        "--flowfile",
        help="Specification of flow CSV file for flow-csv type of input.",
        type=str,
        metavar="NAME.csv",
    )
    parser.add_argument(
        "-p",
        "--pcap",
        help="Specification of pcap file for pcap type of input.",
        type=str,
        metavar="NAME.pcap",
    )
    parser.add_argument(
        "-d",
        "--dependencies",
        help="Specification of CSF file for output of pcap parser with ID dependencies.",
        type=str,
        metavar="NAME.pcap",
        default="",
    )
    parser.add_argument(
        "--delimiter",
        help="Specification of delimiter in flow CSV file. Default set to ','.",
        type=str,
        metavar="STRING",
        default=",",
    )
    parser.add_argument(
        "--src_ip",
        help="Source IP address field in flow csv file. Default set to src_ip.",
        type=str,
        metavar="STRING",
        default="src_ip",
    )
    parser.add_argument(
        "--dst_ip",
        help="Destination IP address field in flow csv file. Default set to dst_ip.",
        type=str,
        metavar="STRING",
        default="dst_ip",
    )
    parser.add_argument(
        "--src_port",
        help="Source transport layer port field in flow csv file. Default set to src_port.",
        type=str,
        metavar="STRING",
        default="src_port",
    )
    parser.add_argument(
        "--dst_port",
        help="Destination transport layer port field in flow csv file. Default set to dst_port.",
        type=str,
        metavar="STRING",
        default="dst_port",
    )
    parser.add_argument(
        "-x",
        help="Module isn't break when eof is send.",
        action="store_true",
    )
    arg = parser.parse_args()
    return arg


def ports_convert_to_int(port: str):
    try:
        return int(port)
    except:
        return port


def load_table_ports(filename: str):
    """Load ports table, that contain ports registered by IANA and ICANN, from csv and return it as dictionary.

    Returns:
        dictionary: Loaded ports table as a dictionary (port->service_name).
    """
    if filename.endswith(".csv") is False:
        print("The filename of table contains services haven't suffix or isn't .csv")
        sys.exit(1)
    if os.path.isfile(filename) is False:
        print(f"The file with name {filename} doesn't exists.")
        sys.exit(1)
    try:
        with open(filename, mode="r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            reg_ports = dict(
                (ports_convert_to_int(rows[1]), rows[0]) for rows in reader
            )
        return reg_ports
    except Exception as e:
        print(f"Error in loading file {filename}: {e}")
        sys.exit(1)


def work_with_nemea(filename, x):
    rec, trap, alert = load_pytrap(sys.argv)
    ports_tb = load_table_ports(filename)
    while True:  # main loop for load ip-flows from interfaces
        try:  # load IP flow from IFC interface
            data = trap.recv()
        except pytrap.FormatChanged as e:
            fmttype, inputspec = trap.getDataFmt(0)
            rec = pytrap.UnirecTemplate(inputspec)
            data = e.data
        if len(data) <= 1:
            trap.send(data)
            if x is True:
                continue
            else:
                break
        rec.setData(data)
        dependency = recognize_dependency(
            ports_tb, rec.SRC_IP, rec.SRC_PORT, rec.DST_IP, rec.DST_PORT
        )
        alert.ID_DEPENDENCY = dependency
        alert.SRC_IP = rec.SRC_IP
        alert.DST_IP = rec.DST_IP
        alert.TIME_FIRST = rec.TIME_FIRST
        alert.TIME_LAST = rec.TIME_LAST
        alert.PACKETS = rec.PACKETS
        alert.PACKETS_REV = rec.PACKETS_REV
        alert.BYTES = rec.BYTES
        alert.BYTES_REV = rec.BYTES_REV
        alert.SRC_PORT = rec.SRC_PORT
        alert.DST_PORT = rec.DST_PORT
        alert.PROTOCOL = rec.PROTOCOL
        trap.send(alert.getData(), 0)
    trap.finalize()  # Free allocated TRAP IFCs


def work_with_flow_csv(arg):
    ports_tb = load_table_ports(arg.ports)
    if arg.dependencies == "":
        file = f"{arg.flowfile.split('.csv')[0]}_dependencies.csv"
    else:
        file = arg.dependencies
    cnt = 0
    print(f"\r      Number of exported flows: {cnt}",end="",)
    with open(file, "w") as w:
        writer = csv.writer(w, delimiter=arg.delimiter)
        with open(arg.flowfile, "r") as r:
            reader = csv.reader(r, delimiter=arg.delimiter)
            header = False
            for row in reader:
                if header is False:
                    i_src_ip = row.index(arg.src_ip)
                    i_src_port = row.index(arg.src_port)
                    i_dst_ip = row.index(arg.dst_ip)
                    i_dst_port = row.index(arg.dst_port)
                    header = True
                    row.insert(0, "ID_DEPENDENCY")
                    writer.writerow(row)
                    continue
                dependency = recognize_dependency(
                    ports_tb,
                    row[i_src_ip],
                    int(row[i_src_port]),
                    row[i_dst_ip],
                    int(row[i_dst_port]),
                )
                row.insert(0, dependency)
                writer.writerow(row)
                cnt += 1
                print(f"\r      Number of exported flows: {cnt}",end="",flush=False,)
    print("")
    


def work_with_pcap(portsfilename, pcapfilename: str, csvfilename: str):
    ports_tb = load_table_ports(portsfilename)
    with open(csvfilename, "w") as w:
        writer = csv.writer(w)
        writer.writerow(
            [
                "ID_DEPENDENCY",
                "SRC_IP",
                "SRC_PORT",
                "DST_IP",
                "DST_PORT",
                "LENGTH",
                "DATA_LENGTH",
                "TIME",
            ]
        )
        cnt = 0
        size = 0
        print(
            f"      Number of packets: {cnt} Size: {0}------------------------------",
            end="",
        )
        for length, time, pktbuf in rpcap(pcapfilename):
            cnt += 1
            size += length
            print(
                f"\r      Number of packets: {cnt} Size: {convert_size(size)}------------------------------",
                end="",
                flush=True,
            )
            try:
                ethh = ethheader.read(pktbuf, 0)
            except:
                continue
            if ethh.ethtype == 0x0800:
                try:
                    iph = ip4header.read(pktbuf, ETHHDRLEN)
                except:
                    continue
            else:
                try:
                    iph = ip6header.read(pktbuf, ETHHDRLEN)
                except:
                    continue
            if not iph:
                continue
            if iph.proto == UDP_PROTO:
                transport = udpheader.read(pktbuf, ETHHDRLEN + iph.iphdrlen)
                protocol = 17
                if not transport:
                    continue
                # can't use len(pktbuf) because of tcpdump-applied trailers
                if transport.udphdrlen is None:
                    continue
                datalen = iph.length - iph.iphdrlen - transport.udphdrlen
            elif iph.proto == TCP_PROTO:
                transport = tcpheader.read(pktbuf, ETHHDRLEN + iph.iphdrlen)
                protocol = 6
                if not transport:
                    continue
                # can't use len(pktbuf) because of tcpdump-applied trailers
                datalen = iph.length - iph.iphdrlen - transport.tcphdrlen
            else:
                # ignore
                continue
            src_ip = ipaddress.ip_address(iph.srcaddrb)
            dst_ip = ipaddress.ip_address(iph.dstaddrb)
            src_port = int(transport.srcport)
            dst_port = int(transport.dstport)
            dependency = recognize_dependency(
                ports_tb, src_ip, src_port, dst_ip, dst_port
            )
            writer.writerow(
                [
                    dependency,
                    src_ip,
                    src_port,
                    dst_ip,
                    dst_port,
                    length,
                    datalen,
                    time,
                ]
            )
    print("")


def main():
    """Main function of the module."""
    arg = parse_arguments()
    if arg.inputtype == "nemea":
        work_with_nemea(arg.ports, arg.x)
    elif arg.inputtype == "flow-csv":
        work_with_flow_csv(arg)
    elif arg.inputtype == "pcap":
        work_with_pcap(arg.ports, arg.pcap, arg.dependencies)
    else:
        print(
            f"Input type {arg.functional} is not supported. Supported are: nemea, flow-csv"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
