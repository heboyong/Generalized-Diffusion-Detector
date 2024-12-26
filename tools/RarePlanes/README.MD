参考[RarePlanes](https://github.com/VisionSystemsInc/RarePlanes) 处理检测标签的代码

用法：
```
python tools/create_coco_real.py --image_dir datasets/real/train/PS-RGB_tiled
                            --geojson_dir datasets/real/train/geojson_aircraft_tiled
                             --output_path ./aircraft_real_coco.json
```       

```
python tools/create_coco_synthetic.py --data_dir datasets/synthetic/
                            --segmentation simple
                            --output_path ./aircraft_synthetic_coco.json
```