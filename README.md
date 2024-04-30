# ES-EOT-real-nuscenes

If you use the code, please cite our paper:

```text
@article{deng20243d,
  title={3D Extended Object Tracking by Fusing Roadside Sparse Radar Point Clouds and Pixel Keypoints},
  author={Jiayin Deng and Zhiqun Hu and Yuxuan Xia and Zhaoming Lu and Xiangming Wen},
  journal={arXiv preprint arXiv:2404.17903},
  year={2024},
  doi={2404.17903}
}
```

## Installation

* Clone the repository and cd to it.

    ```bash
    git clone https://github.com/RadarCameraFusionTeam-BUPT/ES-EOT-real-nuscenes.git
    cd ES-EOT
    ```

* Install dependencies according to the installation part of [github page](https://github.com/RadarCameraFusionTeam-BUPT/ES-EOT).

## Usage

**Note**: The detections of CRN and HVDetFusion are saved in json files, while the realtime detection is not released here.

* Run the main file. (The result is written in a .npy file in the same folder as main.py)

    ```bash
    cd ES-EOT
    python main.py
    ```

* Calculate the ATE and ASE values.

    ```bash
    python calculate_matrics.py
    ```

* Show the tracking results in an animation.

    ```bash
    python ShowRes.py
    ```

## Show

* Show BEV detections in a .jpg picture

    ```bash
    cd show
    python show_BEV_track.py
    ```

## Keypoints detection using pre-trained model

Note: Keypoints detections are stored in the `data/turn_left/kps_json/`. However, if you wish to reuse the pre-trained model, follow these steps:

### 1. Download the pre-trained model

* Obtain the pre-trained model weights from [model link](https://drive.google.com/file/d/1vnJbfMzvKxIPGX49Lkmc9Tlr9XrGiv-I/view?usp=drive_link), and move it into the `assets` folder.

### 2. Run Keypoints Detection

* Once you have the pre-trained model weights and dependencies installed, you can run the key points detection script

    ```bash
    python predict.py data/turn_left/images --model assets/best.pt --render
    ```

    The results are written into the `output` folder.
