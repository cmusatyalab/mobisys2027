#!/bin/bash
# runs ON lambda: cut 27 clips from src/, then RF-DETR 81-class on all of them
cd /root/ddil_tracking/nov18
echo "start $(date +%T)"
awk '{print $1, $2, $3}' cuts.txt | xargs -P 14 -n 3 sh -c 'ffmpeg -nostdin -v error -y -ss "$2" -i "src/$0.MP4" -t 60 -map 0:v:0 -c:v libx264 -crf 18 -preset fast -threads 3 -an "$1.mp4" && echo "cut $1"'
echo "cuts done $(date +%T)"
for f in $(awk '{print $2}' cuts.txt); do echo "$f $(ffprobe -v error -count_packets -select_streams v -show_entries stream=nb_read_packets -of csv=p=0 $f.mp4)"; done
PY=/root/.local/share/steeleagle/pkg/https___github_com_cmusatyalab_steeleagle_plugins__engines_detection@v4_backend_rework/.pixi/envs/cuda126/bin/python
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1); echo "gpu $G"
CUDA_VISIBLE_DEVICES=$G $PY ../detr_run.py $(awk '{print $2".mp4"}' cuts.txt) --thr 0.1 2>&1 | grep -E "frames|Error|Trace"
echo "ALL DONE $(date +%T)"
