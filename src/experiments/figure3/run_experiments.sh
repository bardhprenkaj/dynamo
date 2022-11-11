#!/bin/bash

while getopts d:u:l:o:s:c: flag
do
    case "${flag}" in
        d) data_folder=${OPTARG};;
        u) upper_limit_offset=${OPTARG};;
        l) lower_limit_offset=${OPTARG};;
        o) output_directory=${OPTARG};;
        s) step=${OPTARG};;
        c) config_file=${OPTARG};;
    esac
done
echo "Reading data from: $data_folder";

curr_dir=`pwd`
if [ -z "$output_directory" ]
then
      output_directory="${curr_dir}/start_offset_dump"
      echo $output_directory
fi
if [ -z "$config_file" ]
then
    echo "Please provide a JSON configuration file. Use the option -c"
    exit 42
fi
if [ -z "$data_folder" ]
then
      echo "Please provide a folder where to read the data. Use the option -d"
      exit -1
fi
if [ -z "$upper_limit_offset" ]
then
      upper_limit_offset=100
fi
if [ -z "$lower_limit_offset" ]
then
      lower_limit_offset=10
fi
if [ -z "$step" ]
then
      step=10
fi

if [ ! -d $data_folder ] 
then
    echo "Directory $data_folder DOES NOT exists." 
    exit -2
fi

if [ ! -d $output_directory ]
then
    # create output directory
    mkdir $output_directory
fi

files=`ls $data_folder/*.csv`
for input_signal in $files
do
    filename=$(basename -- "$input_signal")
    filename="${filename%.*}"
    makedir=""
    if [ ! -d "$filename" ]
    then
        makedir="${output_directory}/${filename}"
        mkdir $makedir
    fi
    for ((i=lower_limit_offset;i<=upper_limit_offset;i+=step)); do
        python run.py $config_file $input_signal "${makedir}/${i}.csv" $i
    done
done

for input_signal in $files
do
    filename=$(basename -- "$input_signal")
    filename="${filename%.*}"

    cd "${output_directory}/${filename}"

    f1s=()
    dump=$(find * -type f -iname "*.csv" -execdir bash -c 'printf "%s\n" "${@%.*}"' bash {} + | sort -n)
    for v in ${dump[@]}
    do
        scores=$(cat "${v}.csv" | p.df 'df.F1Score.values')
        f1s+=($(echo $scores | awk '{print $2}'))
    done
    cd ..

    for score in ${f1s[@]}; do 
        printf '%s\n' $score
    done > "${filename}_f1_scores.csv"

    rm -rf "${output_directory}/${filename}"
done