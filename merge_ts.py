#!/usr/bin/python3
"""

Author:Tinc0 CHAN
e-mail: tincochan@foxmail.com, koumar@cesnet.cz

Copyright (C) 2023 CESNET

LICENSE TERMS

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the Company nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

ALTERNATIVELY, provided that this notice is retained in full, this product may be distributed under the terms of the GNU General Public License (GPL) version 2 or later, in which case the provisions of the GPL apply INSTEAD OF those given above.

This software is provided as is'', and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the company or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
"""
import pandas as pd
import sys
import csv
csv.field_size_limit(sys.maxsize)
import json

for TIME_WINDOW in [86400, 43200, 21600, 14400, 7200, 3600, 1800, 900, 600]:
    # TIME_WINDOW = 14400
    all_ports = {}
    with open(f"time_series/design.time_series.{TIME_WINDOW}.csv", "r") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            id_dep = row[0]
            ports = id_dep.split(")-")[0].split("(")[1]
            if "-" in ports:
                tmp = ports.split("-")
                if tmp[0] not in all_ports:
                    all_ports[tmp[0]] = 0
                all_ports[tmp[0]] += 1
                if tmp[1] not in all_ports:
                    all_ports[tmp[1]] = 0
                all_ports[tmp[1]] += 1
    
    NOVEL_PORTS = []
    for key in all_ports.keys():
        if all_ports[key] >= 100 and int(key) < 49152:
            NOVEL_PORTS.append(key)
    
    dependencies = {}
    with open(f"time_series/design.time_series.{TIME_WINDOW}.merged.csv", "w") as w:
        writer = csv.writer(w, delimiter=";")
        with open(f"time_series/design.time_series.{TIME_WINDOW}.csv", "r") as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                id_dep = row[0]
                tmp = id_dep.split(")-")
                ip_1 = tmp[1]
                tmp = tmp[0].split("(")
                ip_2 = tmp[0]
                ports = tmp[1]
                if "-" in ports:
                    tmp = ports.split("-")
                    if tmp[0] in NOVEL_PORTS:
                        id_dep = f"{ip_1}({tmp[0]})-{ip_2}"
                        if id_dep not in dependencies:
                            dependencies[id_dep] = {
                                                        "N_FLOWS": 0,	"N_PACKETS": 0,	"N_BYTES": 0,	"PACKETS": [],	"BYTES": [],	"START_TIMES": [],	"END_TIMES": [],	"LABELS": [],
                                                    }
                        dependencies[id_dep]["N_FLOWS"] += int(row[1])
                        dependencies[id_dep]["N_PACKETS"] += int(row[2])
                        dependencies[id_dep]["N_BYTES"] += int(row[3])
                        dependencies[id_dep]["PACKETS"] += json.loads(row[4])
                        dependencies[id_dep]["BYTES"] += json.loads(row[5])
                        dependencies[id_dep]["START_TIMES"] += json.loads(row[6])
                        dependencies[id_dep]["END_TIMES"] += json.loads(row[7])
                        if 'Miner' in row[8]:
                            dependencies[id_dep]["LABELS"].append('Miner')
                        else:
                            dependencies[id_dep]["LABELS"].append('Other')    
                    elif tmp[1] in NOVEL_PORTS:
                        id_dep = f"{ip_2}({tmp[1]})-{ip_1}"
                        if id_dep not in dependencies:
                            dependencies[id_dep] = {
                                                        "N_FLOWS": 0,	"N_PACKETS": 0,	"N_BYTES": 0,	"PACKETS": [],	"BYTES": [],	"START_TIMES": [],	"END_TIMES": [],	"LABELS": [],
                                                    }
                        dependencies[id_dep]["N_FLOWS"] += int(row[1])
                        dependencies[id_dep]["N_PACKETS"] += int(row[2])
                        dependencies[id_dep]["N_BYTES"] += int(row[3])
                        dependencies[id_dep]["PACKETS"] += json.loads(row[4])
                        dependencies[id_dep]["BYTES"] += json.loads(row[5])
                        dependencies[id_dep]["START_TIMES"] += json.loads(row[6])
                        dependencies[id_dep]["END_TIMES"] += json.loads(row[7])
                        if 'Miner' in row[8]:
                            dependencies[id_dep]["LABELS"].append('Miner')
                        else:
                            dependencies[id_dep]["LABELS"].append('Other')
                    else:
                        writer.writerow(row)
                else:
                    writer.writerow(row)
                
    with open(f"time_series/design.time_series.{TIME_WINDOW}.merged.csv", "a") as a:
        writer = csv.writer(a, delimiter=";")
        for key in dependencies:
            writer.writerow([
                key,
                len(dependencies[key]["PACKETS"]),
                sum(dependencies[key]["PACKETS"]),
                sum(dependencies[key]["BYTES"]),
                dependencies[key]["PACKETS"],
                dependencies[key]["BYTES"],
                dependencies[key]["START_TIMES"],
                dependencies[key]["END_TIMES"],
                dependencies[key]["LABELS"],
            ])
            


for TIME_WINDOW in [86400, 43200, 21600, 14400, 7200, 3600, 1800, 900, 600]:
    # TIME_WINDOW = 14400
    all_ports = {}
    with open(f"time_series/evaluation.time_series.{TIME_WINDOW}.csv", "r") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            id_dep = row[0]
            ports = id_dep.split(")-")[0].split("(")[1]
            if "-" in ports:
                tmp = ports.split("-")
                if tmp[0] not in all_ports:
                    all_ports[tmp[0]] = 0
                all_ports[tmp[0]] += 1
                if tmp[1] not in all_ports:
                    all_ports[tmp[1]] = 0
                all_ports[tmp[1]] += 1
    
    NOVEL_PORTS = []
    for key in all_ports.keys():
        if all_ports[key] >= 100 and int(key) < 49152:
            NOVEL_PORTS.append(key)
    
    dependencies = {}
    with open(f"time_series/evaluation.time_series.{TIME_WINDOW}.merged.csv", "w") as w:
        writer = csv.writer(w, delimiter=";")
        with open(f"time_series/evaluation.time_series.{TIME_WINDOW}.csv", "r") as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                id_dep = row[0]
                tmp = id_dep.split(")-")
                ip_1 = tmp[1]
                tmp = tmp[0].split("(")
                ip_2 = tmp[0]
                ports = tmp[1]
                if "-" in ports:
                    tmp = ports.split("-")
                    if tmp[0] in NOVEL_PORTS:
                        id_dep = f"{ip_1}({tmp[0]})-{ip_2}"
                        if id_dep not in dependencies:
                            dependencies[id_dep] = {
                                                        "N_FLOWS": 0,	"N_PACKETS": 0,	"N_BYTES": 0,	"PACKETS": [],	"BYTES": [],	"START_TIMES": [],	"END_TIMES": [],	"LABELS": [],
                                                    }
                        dependencies[id_dep]["N_FLOWS"] += int(row[1])
                        dependencies[id_dep]["N_PACKETS"] += int(row[2])
                        dependencies[id_dep]["N_BYTES"] += int(row[3])
                        dependencies[id_dep]["PACKETS"] += json.loads(row[4])
                        dependencies[id_dep]["BYTES"] += json.loads(row[5])
                        dependencies[id_dep]["START_TIMES"] += json.loads(row[6])
                        dependencies[id_dep]["END_TIMES"] += json.loads(row[7])
                        if 'Miner' in row[8]:
                            dependencies[id_dep]["LABELS"].append('Miner')
                        else:
                            dependencies[id_dep]["LABELS"].append('Other')    
                    elif tmp[1] in NOVEL_PORTS:
                        id_dep = f"{ip_2}({tmp[1]})-{ip_1}"
                        if id_dep not in dependencies:
                            dependencies[id_dep] = {
                                                        "N_FLOWS": 0,	"N_PACKETS": 0,	"N_BYTES": 0,	"PACKETS": [],	"BYTES": [],	"START_TIMES": [],	"END_TIMES": [],	"LABELS": [],
                                                    }
                        dependencies[id_dep]["N_FLOWS"] += int(row[1])
                        dependencies[id_dep]["N_PACKETS"] += int(row[2])
                        dependencies[id_dep]["N_BYTES"] += int(row[3])
                        dependencies[id_dep]["PACKETS"] += json.loads(row[4])
                        dependencies[id_dep]["BYTES"] += json.loads(row[5])
                        dependencies[id_dep]["START_TIMES"] += json.loads(row[6])
                        dependencies[id_dep]["END_TIMES"] += json.loads(row[7])
                        if 'Miner' in row[8]:
                            dependencies[id_dep]["LABELS"].append('Miner')
                        else:
                            dependencies[id_dep]["LABELS"].append('Other')
                    else:
                        writer.writerow(row)
                else:
                    writer.writerow(row)
    
    with open(f"time_series/evaluation.time_series.{TIME_WINDOW}.merged.csv", "a") as a:
        writer = csv.writer(a, delimiter=";")
        for key in dependencies:
            writer.writerow([
                key,
                len(dependencies[key]["PACKETS"]),
                sum(dependencies[key]["PACKETS"]),
                sum(dependencies[key]["BYTES"]),
                dependencies[key]["PACKETS"],
                dependencies[key]["BYTES"],
                dependencies[key]["START_TIMES"],
                dependencies[key]["END_TIMES"],
                dependencies[key]["LABELS"],
            ])