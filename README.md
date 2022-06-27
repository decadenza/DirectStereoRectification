# Direct Stereo Rectification
Here you can find the algorithm for stereo rectification. This algorithm computes the rectifying homographies that minimize perspective distortion.

Our method does not use optimisation libraries and provides a closed-form solution.
It is an improvement of the approach originally introduced by Charles Loop and Zhengyou Zhang in _“Computing rectifying homographies for stereo vision”_ (1999), DOI: 10.1109/CVPR.1999.786928.

Full details in our **paper** (soon to be published):

**Lafiosca Pasquale and Ceccaroni Marta, *"Rectifying homographies for stereo vision: analytical solution for minimal distortion"*, Lecture Notes in Networks and Systems, 2022.**

Pre-print available [here](https://arxiv.org/abs/2203.00123).

Please, if you find this useful, **cite** as:
```
@article{DirectStereoRectification,
    author  = {Lafiosca, Pasquale and Ceccaroni, Marta},
    title   = {Rectifying homographies for stereo vision: analytical solution for minimal distortion},
    year    = {2022},
    journal = {Lecture Notes in Networks and Systems},
    volume  = {507},
    doi     = {}
}
```
DOI and official publication are **coming soon**.

## Dependencies
- Python 3 (tested with version 3.8.2)
- NumPy (tested with version 1.18.2)

Install as:
```
pip3 install numpy
```

OpenCV is required for the example only. You can install it with:
```
pip3 install opencv-contrib-python
```

## Usage
Try it with:
```
python3 example.py
```
Refer to comments in [example.py](example.py) and [rectification.py](rectification.py).

## Disclamer
The code is provided "as is" wihout any warranty. For details see [LICENSE](LICENSE) file.
