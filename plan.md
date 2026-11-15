# Mobisys 2027 Experiments
I want you to add the following experiments to the lab/ folder, removing everything except swiftmap. You may need
to make changes to some plugins in order to achieve these, and that is fine, just okay it with me first. For each
top level header (1.2 or 2.1 e.g.), I want you to add them as a tab in a csv so that they can be easily looked at
and added to a Google Sheet. If there are sub headers or sub-sub headers, just space them out within that tab.
I'd also like you to create figures for the following metrics that will be saved under a figs/ directory in results/.
Make sure that they are Latex friendly; in fact, you could also add a fig.tex that a tikz chart in addition to the
figure you render.

You can add these CSVs, each with a tab, to this sheet[https://docs.google.com/spreadsheets/d/1-so_lnaRN9hGMJF5UQSGeSVf-BhsxlIhcifPYa9PpII/edit?ouid=107054348052487549902&usp=sheets_home&ths=true].

For you to accurately measure precision and recall, please draft a script called label that can be run in the mobisys2027
directory on a target video. This allows me to draw bounding boxes around the mannequins in the three test videos which
are:
- parkinglot
- street
- traintrack
You can take a first crack at labeling so that I can just add a few boxes here and there to improve it. When people other
than the mannequin are in frame, please label them as people and not mannequin, as this will be important later.

I would also like you to support reading images from a directory. There are two directories with differently structured
labels in Downloads:
- sard
- wisard
These are samples from two popular datasets for search and rescue. I want you to support both their label formats.
Any jpeg/webp files, unless otherwise specified will be found in the media/ folder.

Lastly, there is a hugging face dataset I want you to use as well:
- mmla[https://huggingface.co/datasets/imageomics/mmla_wilds]

Each of the following headers should be added to a folder of that name. All experiments will be run on lambda2 and
will use root@fermi (fermi) as the child daemon. Edges marked as R-> cross over to lambda2 from fermi.

UoT means Utility of Transmission; what fraction of bytes sent actually were relevant?

## 1.1-quality
This tab already exists so use the numbers already in there!

### 1.1.1-quant
_Graphs_:
line graph, x axis QP, y axis mAP 50-95, RF-DETR2XL vs YOLO26XL
line graph, x axis resolution (high to low) , y axis mAP 50-95, RF-DETR2XL vs YOLO26XL

## 1.2-microbenchmarks

### 1.2.1-local
_Video_: parkinglot
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage
_Trials_: 3
_Flow_:
video -> detector qnn

### 1.2.2-remote
_Video_: parkinglot
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage
_Trials_: 3
_Flow_:
video -> H264 encoder R-> H264 decoder -> detector rf-detr2xl

### 1.2.3-fusion
_Video_: parkinglot
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage
_Trials_: 3
_Flow_:
video -> H264 encoder R-> H264 decoder -> detector rf-detr2xl
      -> detector qnn

## 2.1-search

### 2.1.1-yolo

#### 2.1.1.1-yolo-ed
_Notes_: run with only early discard enabled, class person, conf 0.5
_Videos_: parkinglot, traintrack, street, wisard, sard
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: normal, limited, degraded, intermittent
_Trials_: 3
_Flow_:
video -> detector qnn -> H264 encoder R-> H264 decoder

#### 2.1.1.2-yolo-recall-curve
_Notes_: run with only early discard enabled, class person, conf variable
_Videos_: parkinglot, traintrack, street, wisard, sard
_Metrics_: fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Trials_: vary conf from 0.01 -> 1.0 in 0.1 increments
_Flow_:
video -> detector qnn -> H264 encoder R-> H264 decoder
_Graphs_:
replicate ~/Downloads/example.gif for each video with the same type of figure

#### 2.1.1.3-yolo-crop
_Notes_: run with early discard enabled, crop enabled, class person, conf 0.5
_Videos_: parkinglot, traintrack, street, wisard, sard
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: normal, limited, degraded, intermittent
_Trials_: 3
_Flow_:
video -> detector qnn -> detector-crop -> H264 encoder R-> H264 decoder

#### 2.1.1.4-yolo-reintegration
_Notes_: run with early discard enabled, crop enabled, reintegration, class person, conf 0.5
_Videos_: parkinglot, traintrack, street, wisard, sard; grace 10% of video runtime, pad with black frames (only intermittent and malicious)
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: intermittent, malicious for first three videos (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> detector qnn -> detector-crop
detector-crop.reintegration -> H264 encoder R-> H264 decoder

#### 2.1.1.5-yolo-verified
_Notes_: run with early discard enabled, crop enabled, reintegration, class person, conf 0.5, rf-detr2xl conf 0.85
_Videos_: parkinglot, traintrack, street, wisard, sard; grace 10% of video runtime, pad with black frames (only intermittent and malicious)
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: normal, limited, degraded, intermittent, malicious for first three videos (see above)
_Trials_: 3
_Flow_:
video -> detector qnn -> detector-crop
detector-crop.reintegration -> H264 encoder R-> H264 decoder -> detector rf-detr2xl

### 2.1.2-change-detection
_Notes_: need to modify patchcore score to have a reintegration producer like detection-crop
_Videos_: field; grace 10% of video runtime, pad with black frames (only intermittent and malicious)
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: normal, limited, degraded, intermittent, malicious (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> patchcore-embed -> patchcore-score
patchcore-score.reintegration -> H264 encoder R-> H264 decoder

### 2.1.3-cliped

#### 2.1.3.1-cliped-text
_Notes_: prompt is "person", no learning enabled, webp 65q
_Videos_: parkinglot, traintrack, street, wisard, sard; grace 10% of video runtime, pad with black frames (intermittent and malicious only)
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: normal, limited, degraded, intermittent, malicious for first three videos (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> cliped
cliped.reintegration -> webp encoder R-> webp decoder

#### 2.1.3.2-cliped-text-learning
_Notes_: prompt is "person", learning enabled, webp 65q
_Videos_: parkinglot, traintrack, street, wisard, sard; grace 10% of video runtime, pad with black frames (intermittent and malicious only)
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: normal, limited, degraded, intermittent, malicious for first three videos (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> cliped
cliped.reintegration -> webp encoder R-> webp decoder

#### 2.1.3.3-cliped-image-mannequin
_Notes_: prompt is mannequin.jpeg, no learning enabled, webp 65q
_Videos_: parkinglot, traintrack, street; grace 10% of video runtime, pad with black frames (intermittent and malicious only)
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: normal, limited, degraded, intermittent, malicious (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> cliped
cliped.reintegration -> webp encoder R-> webp decoder

#### 2.1.3.4-cliped-image-mannequin-learning
_Notes_: prompt is mannequin.jpeg, learning enabled, webp 65q
_Videos_: parkinglot, traintrack, street; grace 10% of video runtime, pad with black frames (intermittent and malicious only)
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: normal, limited, degraded, intermittent, malicious (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> cliped
cliped.reintegration -> webp encoder R-> webp decoder

#### 2.1.3.5-cliped-wildlife-onager
_Notes_: prompt is onager.webp, learning enabled, webp 65q
_Videos_: mmla; grace 10% of video runtime, pad with black frames (intermittent and malicious only)
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall
_Profiles_: normal, limited, degraded, intermittent
_Trials_: 3
_Flow_:
video -> cliped
cliped.reintegration -> webp encoder R-> webp decoder
