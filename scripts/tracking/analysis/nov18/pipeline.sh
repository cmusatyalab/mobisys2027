#!/bin/bash
# cut -> copy to lambda -> RF-DETR (81-class) -> fetch detections
set -e
cd /home/teiszler/Videos/ddil_tracking/nov18
awk '{print $1, $2, $3}' ../analysis/nov18/cuts.txt | xargs -P 6 -n 3 sh -c 'ffmpeg -nostdin -v error -y -ss "$2" -i "$0.MP4" -t 60 -map 0:v:0 -c:v libx264 -crf 18 -preset fast -an "$1.mp4" && echo "cut $1"'
for f in $(awk '{print $2}' ../analysis/nov18/cuts.txt); do n=$(ffprobe -v error -count_packets -select_streams v -show_entries stream=nb_read_packets -of csv=p=0 $f.mp4); echo "$f frames=$n"; done
H=root@lambda.elijah.cs.cmu.edu
ssh -o BatchMode=yes $H 'mkdir -p /root/ddil_tracking/nov18'
rsync -a --partial $(awk '{print $2".mp4"}' ../analysis/nov18/cuts.txt) $H:/root/ddil_tracking/nov18/
echo "copied"
ssh -o BatchMode=yes $H 'cd /root/ddil_tracking/nov18; PY=/root/.local/share/steeleagle/pkg/https___github_com_cmusatyalab_steeleagle_plugins__engines_detection@v4_backend_rework/.pixi/envs/cuda126/bin/python; G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1); echo gpu $G; CUDA_VISIBLE_DEVICES=$G $PY ../detr_run.py *.mp4 --thr 0.1 2>&1 | grep -E "frames|Error|Trace"'
scp -q $H:/root/ddil_tracking/nov18/*.detr.csv ../analysis/nov18/
echo "fetched $(ls ../analysis/nov18/*.detr.csv | wc -l)"
