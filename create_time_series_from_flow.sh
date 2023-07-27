#!/bin/bash
# Crete time series from pcap.
#  - use python modules: add_dependency and create_time_series_from_pcap 
#
# author:Tinc0 CHAN
# e-mail: tincochan@foxmail.com, koumar@cesnet.cz
#
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Usage:"
   echo "  Add description of the script functions here."
   echo
   echo "  Syntax: scriptTemplate [-h | -r | -f ]"
   echo "  options:"
   echo "  -h     Print this Help."
   echo "  -f     FLOW_FILE, csv file with IP flows"
   echo "  -w     TIME_SERIES, csv file where store the Time Series"
   echo "  -s     SRC_IP"
   echo "  -d     DST_IP"
   echo "  -S     SRC_PORT"
   echo "  -D     DST_PORT"
   echo "  -p     PACKETS"
   echo "  -P     PACKETS_REV"
   echo "  -b     BYTES"
   echo "  -B     BYTES_REV"
   echo "  -t     TIME_FIRST"
   echo "  -T     TIME_LAST"
   echo "  -F     TIME_FORMAT. Default value: %Y-%m-%dT%H:%M:%S.%f"
   echo "  -l     LABEL, anotation label"
   echo "  -c     PATH"
   echo
}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

############################################################
############################################################
# Main program                                             #
############################################################
############################################################
############################################################
# Process the input options. Add options as needed.        #
############################################################
# Set variables
FLOW_FILE="NaN"
TIME_SERIES="NaN"
SRC_IP="SRC_IP"
DST_IP="DST_IP"
SRC_PORT="SRC_PORT"
DST_PORT="DST_PORT"
PACKETS="PACKETS"
PACKETS_REV="PACKETS_REV"
BYTES="BYTES"
BYTES_REV="BYTES_REV"
TIME_FIRST="TIME_FIRST"
TIME_LAST="TIME_LAST"
TIME_FORMAT="%Y-%m-%dT%H:%M:%S.%f"
LABEL=""
CPATH="."
CUTTING_TIMEOUT=21600
# Get the options
while getopts ":hf:w:s:d:S:D:p:P:b:B:t:T:F:l:c:C:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      f) # enter path to pcap file
         FLOW_FILE=$OPTARG;;
      w) # enter path to pcap file
         TIME_SERIES=$OPTARG;;
      s)
         SRC_IP=$OPTARG;;
      d)
         DST_IP=$OPTARG;;
      S)
         SRC_PORT=$OPTARG;;
      D)
         DST_PORT=$OPTARG;;
      p)
         PACKETS=$OPTARG;;
      P)
         PACKETS_REV=$OPTARG;;
      b)
         BYTES=$OPTARG;;
      B)
         BYTES_REV=$OPTARG;;
      t)
         TIME_FIRST=$OPTARG;;
      T)
         TIME_LAST=$OPTARG;;
      F)
         TIME_FORMAT=$OPTARG;;
      l)
         LABEL=$OPTARG;;
      c)
         CPATH=$OPTARG;;
      C)
         CUTTING_TIMEOUT=$OPTARG;;
      \?) # Invalid option
         echo "Load arguments"
         echo "  Error: Invalid argument"
         echo " "
         Help
         exit;;
   esac
done
echo -e "${GREEN}Load arguments"
echo "  OK"

echo "Test settings:"

if [[ $FLOW_FILE == "NaN" ]]
then
  echo -e "  ${RED}ERROR: Enter path to FLOW file in argument -f${NC}"
  exit
fi

if [[ $FLOW_FILE != *.csv ]]
then
   echo -e "  ${RED}ERROR: FLOW file without suffix .csv in -f${NC}"
   exit
fi

if [[ -f "$FLOW_FILE" ]]; then
    echo -e "  ${GREEN}OK: $FLOW_FILE exists."
else 
    echo -e "  ${RED}ERROR: $FLOW_FILE does not exist${NC}"
    exit
fi

DEPENDENCY="$FLOW_FILE.dependencies.csv"
if [[ $TIME_SERIES == "NaN" ]]
then
   TIME_SERIES="$FLOW_FILE.flow_timeseries.csv"
fi

# ./add_dependency.py --delimiter=',' --src_ip=SRC_IP --dst_ip=DST_IP --src_port=SRC_PORT --dst_port=DST_PORT
# ./create_time_series_from_flow.py --packets=PACKETS --packets_rev=PACKETS_REV --bytes=BYTES --bytes_rev=BYTES_REV --start_time=TIME_FIRST --end_time=TIME_LAST --label=APP

echo -e "${GREEN}Create flow time series to CSV file from FLOW csv file."
echo "  Run commands:"
if [[ -f "$DEPENDENCY" ]]; then
   echo -e "${GREEN}    $DEPENDENCY file already exists... skipping"
else
   echo "    $CPATH/add_dependency.py -t flow-csv -f $FLOW_FILE -d $DEPENDENCY --delimiter=',' --src_ip=$SRC_IP --dst_ip=$DST_IP --src_port=$SRC_PORT --dst_port=$DST_PORT -P $CPATH/Ports.csv"
   echo -e "${NC}" 
   $CPATH/add_dependency.py -t flow-csv -f "$FLOW_FILE" -d "$DEPENDENCY" --delimiter=',' --src_ip="$SRC_IP" --dst_ip="$DST_IP" --src_port="$SRC_PORT" --dst_port="$DST_PORT" -P "$CPATH/Ports.csv"
fi
echo -e "${GREEN}    $CPATH/create_time_series_from_flow.py -f $DEPENDENCY -t $TIME_SERIES --packets=$PACKETS --packets_rev=$PACKETS_REV --bytes=$BYTES --bytes_rev=$BYTES_REV --start_time=$TIME_FIRST --end_time=$TIME_LAST --label=$LABEL --time_format=$TIME_FORMAT --cutting_timeout=$CUTTING_TIMEOUT"
echo -e "${NC}" 
$CPATH/create_time_series_from_flow.py -f "$DEPENDENCY" -t "$TIME_SERIES" --packets="$PACKETS" --packets_rev="$PACKETS_REV" --bytes="$BYTES" --bytes_rev="$BYTES_REV" --start_time="$TIME_FIRST" --end_time="$TIME_LAST" --label="$LABEL" --time_format="$TIME_FORMAT" --cutting_timeout=$CUTTING_TIMEOUT
echo -e "${GREEN}Time series added to $TIME_SERIES"
