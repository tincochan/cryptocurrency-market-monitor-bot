#!/bin/bash

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
    echo "  -i     Input file changable part."
    echo
}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

REMOVE="False"
# Get the options
while getopts ":hri:" option; do
    case $option in
        h) # display Help
            Help
            exit;;
        r)
            REMOVE="True";;
        i)
            INPUT=$OPTARG;;
    esac
done
echo -e "${GREEN}Load arguments"
echo "  OK"

for per_level in "0.9"  "0.99" "0.999"
do
    for sig_level in "0.1" "0.01" "0.001"
    do
        for SPLIT in "0.2"
        do
                echo -e "${YELLOW}Settings:  $SPLIT $sig_level $per_level"
                echo -e "${NC}" 
                if [[ $REMOVE == "True" ]]
                then
                    if [[ -f "cryptomining_$SPLIT-$sig_level.$per_level.csv" ]]
                    then
                        echo -e "${YELLOW}   rm cryptomining_$SPLIT-$sig_level.$per_level.csv"
                        echo -e "${NC}" 
                        rm "cryptomining_$SPLIT-$sig_level.$per_level.csv"
                    fi
                fi
                if [[ -f "cryptomining_$SPLIT-$sig_level.$per_level.csv" ]]
                then
                    echo -e "${GREEN}   cryptomining_$SPLIT-$sig_level.$per_level.csv with flow time series extists, skipping ..."
                    echo -e "${NC}" 
                else
                    echo -e "${GREEN}   time ./ml_periodicity.py --output_file=cryptomining_$SPLIT-$sig_level.$per_level.csv --validation=$SPLIT -k 100 --input_file=$sig_level.$per_level"
                    echo -e "${NC}" 
                    time ./ml_periodicity.py --output_file="cryptomining_$SPLIT-$sig_level.$per_level.csv" --validation="$SPLIT" -k 100 --input_file="$sig_level.$per_level"
                fi
        done
    done
done