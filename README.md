# Direct Stereo Rectification
Here you can find the algorithm for stereo rectification. This algorithm computes the rectifying homographies that minimize perspective distortion.

Our method does not use optimisation libraries and provides a closed-form solution.
It is an improvement of the approach originally introduced by Charles Loop and Zhengyou Zhang in _“Computing rectifying homographies for stereo vision”_ (1999), DOI: 10.1109/CVPR.1999.786928.

Video presentation [YouTube link](https://youtu.be/oTkYWsB3KTk).

Full details in our **paper**:

**Lafiosca, P., Ceccaroni, M. (2022). Rectifying Homographies for Stereo Vision: Analytical Solution for Minimal Distortion. In: Arai, K. (eds) Intelligent Computing. SAI 2022. Lecture Notes in Networks and Systems, vol 507. Springer, Cham. https://doi.org/10.1007/978-3-031-10464-0_33**

Pre-print available [here](https://arxiv.org/abs/2203.00123).

Please, if you find this useful, **cite** as:
```
@inproceedings{LafioscaDirectStereoRectification,
    author  = {Lafiosca, Pasquale and Ceccaroni, Marta},
    title   = {Rectifying Homographies for Stereo Vision: Analytical Solution for Minimal Distortion},
    year    = {2022},
    journal = {Lecture Notes in Networks and Systems},
    booktitle = {Intelligent Computing},
    isbn    = {978-3-031-10464-0},
    volume  = {507},
    pages   = {484--503},
    doi     = {10.1007/978-3-031-10464-0_33},
    url     = {https://doi.org/10.1007/978-3-031-10464-0_33},
    publisher = {Springer International Publishing}
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
