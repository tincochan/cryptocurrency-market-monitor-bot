#!/bin/bash
# Test of time series plugin for ipfixprobe.
#  - use python modules: add_dependency and ipfixprobe_plugin_test 
#
# author:Tinc0 CHAN
# e-mail: tincochan@foxmail.com
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
    echo "  Syntax: scriptTemplate [-h | -r ]"
    echo "  options:"
    echo "  -h     Print this Help."
    echo "  -r     Remove exists time series files."
    echo
}
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

############################################################
# Process the input options. Add options as needed.        #
############################################################
# Set variables
REMOVE="False"
# Get the options
while getopts ":hr" option; do
    case $option in
        h) # display Help
            Help
            exit;;
        r)
            REMOVE="True";;
    esac
done
echo -e "${GREEN}Load arguments"
echo "  OK"


for SEC in  86400 43200 21600 14400 7200 3600 1800 900 600
do
    echo -e "${YELLOW}$SEC "
    echo -e "${NC}" 
    if [[ $REMOVE == "True" ]]
    then
        if [[ -f "time_series/design.time_series.$SEC.csv" ]]
        then
            rm "time_series/design.time_series.$SEC.csv"
        fi
        if [[ -f "time_series/evaluation.time_series.$SEC.csv" ]]
        then
        
            rm "time_series/evaluation.time_series.$SEC.csv"
        fi
    fi
    
    if [[ -f "time_series/evaluation.time_series.$SEC.csv" ]]
    then
        echo -e "${GREEN} time_series/design.time_series.$SEC.csv etxts, skipp creating time series....."
        echo -e "${NC}" 
    else
        echo -e "${GREEN}  ./create_time_series_from_flow.sh  -C $SEC -w time_series/design.time_series.$SEC.csv -c . -f original_dataset/design.csv -s ....."
        echo -e "${NC}" 
        time ./create_time_series_from_flow.sh  -C $SEC -w "time_series/design.time_series.$SEC.csv" -c "." -f "original_dataset/design.csv" -s "ipaddr SRC_IP" -d "ipaddr DST_IP" -S "uint16 SRC_PORT" -D "uint16 DST_PORT" -p "uint32 PACKETS" -P "uint32 PACKETS_REV" -b "uint64 BYTES" -B "uint64 BYTES_REV" -t "time TIME_FIRST" -T "time TIME_LAST" -l "string LABEL" -F "%Y-%m-%dT%H:%M:%S.%f"  
    fi
    
    if [[ -f "time_series/evaluation.time_series.$SEC.csv" ]]
    then
        echo -e "${GREEN}  time_series/evaluation.time_series.$SEC.csv etxts, skipp creating time series....."
        echo -e "${NC}" 
    else
        echo -e "${GREEN}  ./create_time_series_from_flow.sh  -C $SEC -w time_series/evaluation.time_series.$SEC.csv -c . -f original_dataset/evaluation.csv -s ....."
        echo -e "${NC}" 
        time ./create_time_series_from_flow.sh  -C $SEC -w "time_series/evaluation.time_series.$SEC.csv" -c "." -f "original_dataset/evaluation.csv" -s "ipaddr SRC_IP" -d "ipaddr DST_IP" -S "uint16 SRC_PORT" -D "uint16 DST_PORT" -p "uint32 PACKETS" -P "uint32 PACKETS_REV" -b "uint64 BYTES" -B "uint64 BYTES_REV" -t "time TIME_FIRST" -T "time TIME_LAST" -l "string LABEL" -F "%Y-%m-%dT%H:%M:%S.%f"  
    fi
done

time ./merge_ts.py

for per_level in "0.9"  "0.99" "0.999"
do
    for sig_level in "0.1" "0.01" "0.001"
    do
        for SEC in  86400 43200 21600 14400 7200 3600 1800 900 600
        do
            echo -e "${YELLOW}$SEC "
            echo -e "${NC}" 
            echo -e "${GREEN}      time ../new_model.py --peridoicity_file=="time_series/design.periodicity_features.$SEC.$sig_level.$per_level.csv" --time_series="time_series/design.time_series.$SEC.merged.csv" --sig_level=$sig_level --per_level=$per_level"
            echo -e "${NC}" 
            time ./new_model.py --peridoicity_file="design.periodicity_features.$SEC.$sig_level.$per_level.csv" --time_series="time_series/design.time_series.$SEC.merged.csv" --sig_level=$sig_level --per_level=$per_level
            
            echo -e "${GREEN}      time ../new_model.py --peridoicity_file=="evaluation.periodicity_features.$SEC.$sig_level.$per_level.csv" --time_series="time_series/evaluation.time_series.$SEC.merged.csv" --sig_level=$sig_level --per_level=$per_level"
            echo -e "${NC}" 
            time ./new_model.py --peridoicity_file="evaluation.periodicity_features.$SEC.$sig_level.$per_level.csv" --time_series="time_series/evaluation.time_series.$SEC.merged.csv" --sig_level=$sig_level --per_level=$per_level
        done
    done
done
