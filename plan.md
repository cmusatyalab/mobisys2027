# Mobisys 2027 Experiments
I want you to add the following experiments to the lab/ folder, removing everything except swiftmap. You may need
to make changes to some plugins in order to achieve these, and that is fine, just okay it with me first. For each
top level header (1.2 or 2.1 e.g.), I want you to add them as a tab in a csv so that they can be easily looked at
and added to a Google Sheet. If there are sub headers or sub-sub headers, just space them out within that tab.
I'd also like you to create figures for the following metrics that will be saved under a figs/ directory in results/.
Make sure that they are Latex friendly; in fact, you could also add a fig.tex that a tikz chart in addition to the
figure you render.

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

MMLA is spliced (scripts/prepare_datasets.py): four 50-frame African wild dog clips placed in zebra and giraffe
footage, so 200 of its 4000 frames (5%) hold a dog. The dog was chosen over the onager because no verifier (Qwen,
Opus, Sonnet, Haiku) could tell onagers from zebras at 720p, while Qwen picks out the dogs (precision 1.0, recall
0.9-0.97 on test tiles).

Each of the following headers should be added to a folder of that name. All experiments will be run on lambda2 and
will use root@fermi (fermi) as the child daemon. Edges marked as R-> cross over to lambda2 from fermi.

UoT means Utility of Transmission; what fraction of bytes sent actually were relevant?
Effective recall is recall over only the frames the pipeline actually processed (it can't find what it never saw).
Occurrences found (YOLO and cliped, on parkinglot, traintrack, street, wisard and mmla): targets appear in runs of
consecutive frames; each run (gaps up to 1 s merged) is one occurrence, found if a frame, crop or tile holding a
target reached lambda2 from any frame of it. Reported as found/total (parkinglot 2, traintrack 2, street 2, wisard 1,
mmla 4 dog clips; the mannequin alone is 1 in parkinglot).

Lossy experiments (intermittent, malicious) end with a 60 s grace period, padded with black frames, so queued
items can still be delivered. Every reintegration queue holds up to 512 items and resends after 30 s.

## 1.1-quality
This tab already exists so use the numbers already in there!

### 1.1.1-quant
_Graphs_:
line graph, x axis QP, y axis mAP 50-95, RF-DETR2XL vs YOLO26XL, add a horizontal line showing YOLO S-Quantized mAP 50-95
line graph, x axis resolution (high to low) , y axis mAP 50-95, RF-DETR2XL vs YOLO26XL, add a horizontal line showing YOLO S-Quantized mAP 50-95

### 1.1.2-latency
_Notes_: These can be found in the 1.1 tab in columns K-P
_Graphs_:
CDF showing fermi -> cloudlet, p95 marked
CDF showing drone -> fermi, p95 marked
CDF showing drone -> cloudlet, p95 marked

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
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found
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
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found
_Profiles_: normal, limited, degraded, intermittent
_Trials_: 3
_Flow_:
video -> detector qnn -> detector-crop -> H264 encoder R-> H264 decoder

#### 2.1.1.4-yolo-reintegration
_Notes_: run with early discard enabled, crop enabled, reintegration, class person, conf 0.5. Replaces 2.1.1.5:
no verifier in any YOLO test (YOLO's crops are already ~1.0 precision, so a verifier can only cost recall)
_Videos_: parkinglot, traintrack, street, wisard, sard
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found, effective recall
_Profiles_: normal, limited, degraded, intermittent, malicious for first three videos (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> detector qnn -> detector-crop
detector-crop.reintegration -> H264 encoder R-> H264 decoder

### 2.1.2-change-detection
_Notes_: patchcore-score has a reintegration producer like detection-crop; runs on CPU-only torch
_Videos_: field
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, effective recall
_Profiles_: normal, limited, degraded, intermittent, malicious (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> patchcore-embed -> patchcore-score
patchcore-score.reintegration -> H264 encoder R-> H264 decoder

### 2.1.3-cliped
Every cliped experiment rescales to 720p first (cliped's tiling is tuned for it) and sends WebP at quality 65.
Without learning there is no verifier: what cliped sends is the result. With learning, a VLM verifier (Qwen3-VL 8B)
on lambda2 answers each tile, and its "no"s become cliped's background prompts. The image question says the tile
is low resolution and never looks exactly like the reference, and asks for "this target" (it used to demand the
same appearance, not merely the same category). On the tiles cliped sent, Qwen keeps 98% of true tiles for both
prompts: text "person" at verified precision 0.98, mannequin image at 0.77 (was 82% kept at 1.00).

#### 2.1.3.1-cliped-text
_Notes_: prompt is "person", no learning
_Videos_: parkinglot, traintrack, street, wisard, sard
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found, effective recall
_Profiles_: normal, limited, degraded, intermittent, malicious for first three videos (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> rescale -> cliped
cliped.reintegration -> webp encoder R-> webp decoder

#### 2.1.3.2-cliped-text-learning
_Notes_: prompt is "person", learning enabled, Qwen verifier
_Videos_: parkinglot, traintrack, street, wisard, sard
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found, effective recall (before and after the verifier)
_Profiles_: normal, limited, degraded, intermittent, malicious for first three videos (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> rescale -> cliped
cliped.reintegration -> webp encoder R-> webp decoder -> verifier (Qwen)

#### 2.1.3.3-cliped-image-mannequin
_Notes_: prompt is mannequin.jpeg, no learning; only the mannequin counts as found (people are what it should not send)
_Videos_: parkinglot, traintrack, street
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found, effective recall
_Profiles_: normal, limited, degraded, intermittent, malicious (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> rescale -> cliped
cliped.reintegration -> webp encoder R-> webp decoder

#### 2.1.3.4-cliped-image-mannequin-learning
_Notes_: prompt is mannequin.jpeg, learning enabled, Qwen verifier
_Videos_: parkinglot, traintrack, street
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found, effective recall (before and after the verifier)
_Profiles_: normal, limited, degraded, intermittent, malicious (only blackouts when seeing mannequins)
_Trials_: 3
_Flow_:
video -> rescale -> cliped
cliped.reintegration -> webp encoder R-> webp decoder -> verifier (Qwen)

#### 2.1.3.5-cliped-wildlife-dog
_Notes_: prompt is dog.jpg, no learning; only dogs count as found
_Videos_: mmla
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found, effective recall
_Profiles_: normal, limited, degraded, intermittent
_Trials_: 3
_Flow_:
video -> rescale -> cliped
cliped.reintegration -> webp encoder R-> webp decoder

#### 2.1.3.6-cliped-wildlife-dog-text
_Notes_: prompt is "African wild dog", no learning; only dogs count as found
_Videos_: mmla
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found, effective recall
_Profiles_: normal, limited, degraded, intermittent
_Trials_: 3
_Flow_:
video -> rescale -> cliped
cliped.reintegration -> webp encoder R-> webp decoder

#### 2.1.3.7-cliped-wildlife-dog-learning
_Notes_: prompt is dog.jpg, learning enabled, Qwen verifier; only dogs count as found. On dog test tiles the verifier's
image question keeps 30/30 dogs but also passes 13/30 zebras, 18/30 onagers, 2/30 giraffes
_Videos_: mmla
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found, effective recall (before and after the verifier)
_Profiles_: normal, limited, degraded, intermittent
_Trials_: 3
_Flow_:
video -> rescale -> cliped
cliped.reintegration -> webp encoder R-> webp decoder -> verifier (Qwen)

#### 2.1.3.8-cliped-wildlife-dog-text-learning
_Notes_: prompt is "African wild dog", learning enabled, Qwen verifier; only dogs count as found. On dog test tiles
the verifier's text question keeps 28/30 dogs and passes no other animal or empty tile
_Videos_: mmla
_Metrics_: throughput, end-to-end latency, mean/median/p95 power usage, total bytes transmitted, UoT,
fraction of bytes sent vs. equivalent 1Mbps stream, precision, recall, occurrences found, effective recall (before and after the verifier)
_Profiles_: normal, limited, degraded, intermittent
_Trials_: 3
_Flow_:
video -> rescale -> cliped
cliped.reintegration -> webp encoder R-> webp decoder -> verifier (Qwen)

## Next run
Trial 1 only, profiles normal and intermittent. Tick [x] what to run and untick [ ] what to skip; times are
estimated from the quick run's wall-clock time per trial (setup included).

- [x] 2.1.1.4-yolo-reintegration, intermittent only (quick run used the old reintegration settings): 5 trials, ~14 min
- [x] 2.1.2-change-detection (PatchCore was broken in the quick run): 2 trials, ~31 min
- [x] 2.1.3.1-cliped-text (verifier removed, scored log added): 10 trials, ~21 min
- [x] 2.1.3.3-cliped-image-mannequin (same): 6 trials, ~14 min
- [x] 2.1.3.5-cliped-wildlife-dog (new): 2 trials, ~15 min
- [x] 2.1.3.6-cliped-wildlife-dog-text (new): 2 trials, ~15 min
- [x] 2.1.3.2-cliped-text-learning (Qwen verifier): 10 trials, ~21 min
- [x] 2.1.3.4-cliped-image-mannequin-learning (Qwen verifier, new image question): 6 trials, ~14 min
- [x] 2.1.3.7-cliped-wildlife-dog-learning (new): 2 trials, ~16 min
- [x] 2.1.3.8-cliped-wildlife-dog-text-learning (new): 2 trials, ~16 min

Ticked as above: 27 trials, ~1 h 50 min; with both learning experiments, ~2 h 25 min.

Already done in the quick run, unchanged since: 1.2.1, 1.2.2, 2.1.1.1 normal, 2.1.1.2, 2.1.1.3, 2.1.1.4 normal.

Held back until the H.264 encoder is benchmarked and fixed (they sent blank ~100-byte frames or almost nothing):
- 1.2.3-fusion
- 2.1.1.1-yolo-ed intermittent: parkinglot got 1 frame, traintrack and sard 0, street 71 blank frames. Full
  H.264 frames (15-90 KB) are also too big for the intermittent link (~35 Kbps on average)
- some 2.1.1.2-yolo-recall-curve trials
