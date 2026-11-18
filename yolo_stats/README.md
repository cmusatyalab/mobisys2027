# YOLO26 QNN model stats (fermi, 2026-11-17/18)

- `yolo_stats.csv`: one row per model: accuracy, HTP inference time, and pipeline throughput, latency and power.
- `yolo_stats_per_class.csv`: AP, recall and precision per labeled class and confidence.
- `build.py`: rebuilds both from `logs/` and `pipeline_1.2.4.csv` (`python3 build.py`). Its docstring describes each source.

Models: the stock COCO and coco81 libraries from
`yolo26-nsmlx-coco-robodog-coco81-qnn2.45-htp-dsp-640-640x384-2026-11-18` (the n-l files are byte-identical to the
earlier `yolo26-coco-qnn` and `yolo26-dog-qnn` copies), and `libs640_v26_quant.so`, the older build the lab uses now.

| Columns | Measured with |
|---|---|
| `acc_*` | `accuracy_prec.py` (copy of vartether-exps `accuracy.py` with precision added) on summer_training val: 235 images, 846 boxes (person, bicycle, car, truck, bus), IoU 0.5 |
| `inference_*` | `latency.py` on okutama_frames_300 (300 frames): the server's inference stage only, without JPEG decode, preprocessing or postprocessing |
| `pipeline_*` | lab `1.2-microbenchmarks/1.2.4-local-yolo26-sizes`: parkinglot, YOLO on fermi, nothing sent, 3 trials; boxes at confidence 0.5, all classes |

The coco81 models have pipeline numbers only: the gRPC inference server only reads 80-class output.
