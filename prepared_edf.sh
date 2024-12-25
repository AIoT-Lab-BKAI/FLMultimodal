# wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/ -O ./benchmark/RAW_DATA/SLEEP_EDF/
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/ -P ./benchmark/RAW_DATA/SLEEP_EDF/
cp -r benchmark/RAW_DATA/SLEEP_EDF/physionet.org/files/sleep-edfx benchmark/RAW_DATA/SLEEP_EDF