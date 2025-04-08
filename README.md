This repository is used to practice various image processing techniques.
The used image was taken during the printing process of a 3D-printer and shows the printing error "Layershift". 

## Original Image
<img src="./img/3d_printer.png" alt="3D printed dice with layershift" width="400"/>

## After preprocessing

Steps include cropping image to region of interest, trasfering RGB color to grayscale and rotation.

![Image preprocessed](./img/image_preprocessed.png)

## Point operators

### Gamma operator
```python
PointOperators.apply_gamma_operator(gamma: float)
```
gamma = 0.5:

![Gamma operator gamma=0.5](./img/image_gamma_05.png)

gamma = 2:

![Gamma operator gamma=2](./img/image_gamma_2.png)


### Histogram equalization

```python
PointOperators.apply_histogram_equalization()
```

![Histogram equalization](./img/image_histogram_equaliuation.png)

## Linear operator
### Gaussian average filter

```python
LinearOperators.apply_gaussian_avg_filter(kernel_size: list, sigma: int)
```

kernel_site = (9, 9), sigma = 3:

![Gaussian blurr](./img/image_gaussian_filter.png)

### Sobel Derivatives

```python
LinearOperators.apply_derivative(orientation: str)
```

Horizontal kernel 
```math
\begin{bmatrix}
    -1 & 0 & 1\\
    -2 & 0 & 2\\
    -1 & 0 & 2
\end{bmatrix}
```
![Sobel horizontal](./img/image_derivative_horizontal.png)

Vertical kernel 
```math
\begin{bmatrix}
    -1 & -2 & 1\\
    0 & 0 & 0\\
    1 & 2 & 1
\end{bmatrix}
```
![Sobel vertival](./img/image_derivative_vertical.png)