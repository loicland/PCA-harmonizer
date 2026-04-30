# Color Harmonizer

A small tool to harmonize the colors of one image (target) to match another (reference), assuming both images are spatially aligned.

The method learns a mapping from RGB values in the target image to RGB values in the reference image.

Installation
```
pip install numpy torch pillow matplotlib tqdm scikit-image
```

## Usage

### Basic usage:
```
python harmonize.py -ref image1.png -tar image2.png
```

### Options
**Mapping type**
```-mapping {ortho, unconst, MLP}```
- *ortho (default)* Orthogonal RGB transform (rotation in color space). Most stable and recommended.
- *unconst* Unconstrained linear RGB transform. More flexible but can distort colors.
- *MLP* Small nonlinear residual neural network. Most expressive, but may overfit or hallucinate.

**Number of iterations**
```-ite 1000```

Controls optimization steps.

**Histogram matching**
```-hist {0, 1 }```
Applies histogram matching after the learned mapping to further align global color distributions. Enabled by default.

## Notes
Images must be spatially aligned  
White/black background regions are ignored during optimization  

## License

MIT
