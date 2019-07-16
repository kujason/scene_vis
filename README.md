# Scene Visualizer
Some 3D scene visualization demos.

If you find this code useful, please consider citing:

`Improving 3D Object Detection for Pedestrians with Virtual Multi-View Synthesis Orientation Estimation`

```
@inproceedings{ku2019vmvs,
  title={Improving 3D Object Detection for Pedestrians with Virtual Multi-View Synthesis Orientation Estimation},
  author={Ku, Jason and Pon, Alex D., Walsh, Sean, and Waslander, Steven L},
  journal={IROS},
  year={2019}
}
```
[In Defense of Classical Image Processing: Fast Depth Completion on the CPU](https://arxiv.org/abs/1802.00036)

```
@inproceedings{ku2018defense,
  title={In Defense of Classical Image Processing: Fast Depth Completion on the CPU},
  author={Ku, Jason and Harakeh, Ali and Waslander, Steven L},
  journal={CRV},
  year={2018}
}
```

Other works that also use this visualization:
- [AVOD](https://arxiv.org/abs/1712.02294)
- [MonoPSR](https://arxiv.org/abs/1904.01690)

---

### Setup
Tested with Python 3.5.2


```bash
workon [virtualenvname]
pip install -r requirements.txt
add2virtualenv src
```

#### Setup Scripts
- Run depth completion
    - `scripts/depth_completion/save_depth_maps_obj.py`
    - `scripts/depth_completion/save_depth_maps_raw.py`
- Save the outputs into their corresponding Kitti folders. Ex.
    - obj: `~/Kitti/object/training/depth_2_multiscale`
    - raw: `~/Kitti/raw/2011_09_26/2011_09_26_drive_0039_sync/depth_02_multiscale`

### Demos:
- kitti_obj
    - `view_sample.py` - view point cloud and boxes
- kitti_raw
    - `overlay_xxx.py` - overlay point clouds from multiple frames
