# Local Look-Up Table Upsampling for Accelerating Image Processing

This paper provide the code, binary, subjective assessment results and distortion dataset of local LUT upsampling of the following paper.

# Paper
Teppei Tsubokawa, Hiroshi Tajima, Yoshihiro Maeda, and Norishige Fukushima
"Local Look-Up Table Upsampling for Accelerating Image Processing", Multimedia Tools and Applications, 2023.
[[Springer]](https://link.springer.com/article/10.1007/s11042-023-16405-7)

```
@article{tsubokawa2023local,
    author  = {Tsubokawa, Teppei and Tajima, Hiroshi and Yoshihiro Maeda and Norishige Fukushima},
    title   = {Local look-up table upsampling for accelerating image processing},
    journal = {Multimedia Tools and Applications},
    year    = {2023},
    doi     = {doi.org/10.1007/s11042-023-16405-7},
    organization = {Springer}
}
```
![LLU](./fig1.webp "Fig1")


# Abstract
The resolution of cameras is increasing, and speedup of various image processing is required to accompany this increase. A simple way of acceleration is processing the image at low resolution and then upsampling the result. Moreover, when we can use an additional high-resolution image as guidance formation for upsampling, we can upsample the image processing results more accurately. We propose an approach to accelerate various image processing by downsampling and joint upsampling. This paper utilizes per-pixel look-up tables (LUTs), named local LUT, which are given a low-resolution input image and output pair. Subsequently, we upsample the local LUT. We can then generate a high-resolution image only by referring to its local LUT. In our experimental results, we evaluated the proposed method on several image processing filters and applications: iterative bilateral filtering, ℓ0
 smoothing, local Laplacian filtering, inpainting, and haze removing. The proposed method accelerates image processing with sufficient approximation accuracy, and the proposed outperforms the conventional approaches in the trade-off between accuracy and efficiency. Our code is available at https://fukushimalab.github.io/LLF/
 
# Code
The code will be updated and integrated to OpenCP, which uses [OpenCV](https://opencv.org/).

* OpenCP repository will contain the optimized code (AVX vectorization and OpenMP parallelization), which is used in our experiments.
    * [OpenCP](https://github.com/norishigefukushima/OpenCP)
    * [localLUTUpsample header](https://github.com/norishigefukushima/OpenCP/blob/master/include/localLUTUpsample.hpp)

## Usage

```cpp
void testSimple()
{
	cp::LocalLUTUpsample llu;
	//images
	cv::Mat srchigh, dstlow, srclow, dsthigh;
	srchigh = cv::imread("source.png");//high resoluton source image
	dstlow = cv::imread("processed.png");//low resolution processed image
	//parameters
	int r = 2;//kernel radius for building LUT from correspondence of low resolution source and processed image
	int lut_num = 256;// the size of LUT per pixel 
	int R = 2;// filtering radius for a LUT (LUT smoothing)
	cp::LocalLUTUpsample::BUILD_LUT buildLUT = cp::LocalLUTUpsample::BUILD_LUT::L2_MIN;//building LUT method
	cp::LocalLUTUpsample::UPTENSOR upsampleLUT = cp::LocalLUTUpsample::UPTENSOR::GAUSS64;//tensor upsampling method
	cp::LocalLUTUpsample::BOUNDARY boundaryLUT = cp::LocalLUTUpsample::BOUNDARY::LINEAR;//boundary condition of LUT
	bool useOffset = true;//with/without offset map
	//run
	cv::resize(srchigh, srclow, dstlow.size());//down sample high resolution source image
	llu.upsample(srclow, dstlow, srchigh, dsthigh, r, lut_num, R, buildLUT, upsampleLUT, boundaryLUT, useOffset);//body
	//show
	cv::imshow("out", dsthigh);
	cv::waitKey();
}
```

# Binary
The pre-compiled binary can be downloaded.
The zip files contains OpenCV and OpenCP DLLs.
* [download LLU binary (under construction)](LLU.zip)

The binary is build by VisualStudio2022 with OpenCV4.8 and OpenCP on Windows.
If the redistribution package for VisualStudio2022 is installed on your PC, please install vc_redist.x64.exe file from the direct link.

* [VisualStudio2022 redistribution package for x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)

## Requirement
* AVX2 supported computer, e.g, Intel (4th generation Core-i or later) or AMD (Zen or later).
    * for Intel's low TDP CPU, such as ATOM, only support Alderlake-N or later.

## Usage
### Simplest option
```
LLU source.png processed.png out.png
```
* source.png: input high resolution image 
* processed.png: input low resolution processed image
* out.png: upsampling result


### Longest option
```
LLU source.png processed.png out.png -r=2 -n=256 -R=2 -L=0 -U=6 -B=3 -o -d
```
* -r=2: kernel radius of building LUT is 2
* -n=256: size of LUT per pixel is 256
* -R=2: filtering radius for range domain of LUT is 2 (LUT value smoothing)
* -L=0: building LUT method is L2 distance minimization
* -U=6: LUT tensor upsampling method is Gauss64 (8x8 kernel).
* -B=3: boundary condition of LUT is linear method
* -o: use offset map, no meaning in 256 LUT case. 
* -d: use GUI debug for local LUT (call `guiLUT` method).

### Help
The following options show help message: `-h`, `-?`, `--help `.
```
>LLU -h

local LUT upsampling
Usage: LLU [params] src_img proclow_img dest

        -?, -h, --help (value:true)
                show help command
        -B, --boundary_lut (value:3)
                method of boundary of LUT (0: replicate, 1: minmax, 2: 0-255, 3: linear, 4: no interpolation)
        -L, --build_lut (value:0)
                method of building LUT (0:L2, 1:L1, 2: Linf, 3: WTA, 4: DP)
        -R, --lut_radius (value:2)
                radius of LUT filtering in range domain
        -U, --upsample_lut (value:6)
                method of upsampling LUT (0:NN, 1:Box4, 2: Box16, 3: Box64, 4: Gauss4, 5: Gauss16, 6: Gauss64, 7: Linear, 8: Cubic)
        -d, --debug
                GUI of viewing local LUT ('q' is quit key for GUI)
        -n, --numlut (value:256)
                size of LUT in range domain
        -o, --offset
                use offset map
        -r, --radius (value:2)
                radius of upsampling

        src_img
                source low image
        proclow_img
                processed low image
        dest (value:out.png)
                dest image
Example:
LLU source.png processed.png out.png -r=2 -n=256 -R=2 -L=0 -U=6 -B=3 -o -d
```

# Subjective Assessment Results / Distortion Image Dataset
Subjective assessment results for each upsampling method can be downloaded.
We upload distorted images and its scores for the assessment.

(under construction)
* [Images (1024x1024 side-by-side)](https://github.com/fukushimalab/LLF/tree/main/subjective/)
* [Assessment Results](./subjective/jnd.xlsx)

The "images" directory contains 20 zip files (1.14 GB in total).
```
bf_img00.zip
l0_img00.zip
ll_img00.zip
hz_img00.zip
bf_img01.zip
...
hz_img04.zip
```
The files 5 source images(00-04) with 21 distortions with 4 filtering types: bf (bilateral filtering), l0 (L0 smoothing), ll (local Laplacian filtering), hz (haze removing).
Each image contains 2048x1024 image, which are connected no distorted image and distorted image side-by-side.


The assessment results of "jnd.xlsx" contains the following information.

* index
    * indeces for each score (0-419)
* distortion
    * 0: bf, 1:l0, 2: ll, 3: hz
* image
    * image index 0-4
* upsample
    * 0: CU, 1: JBU, 2: GIU, 3: BGU, 4: LLU (Proposed)
* level (downsampling ratio)
    * -1: no downsampling, 0: 1/4, 1: 1/16, 2: 1/64, 3: 1/256
* ID1-11, ave
    * JND results: 0: not same, 1: same

Example: local Laplacian filtering accelerated by cubic upsampling (1/64).
![sidebyside](./sidebyside.webp "sidebyside")

# Link
 This paper uses two external dataset and the followings show the download links.
* [HRHP: high-resolution high-precision images dataset](http://imagecompression.info/test_images/)
* [T-Recipes: Transform Recipes dataset](https://groups.csail.mit.edu/graphics/xform_recipes/dataset.html) [^1]

 [^1] M. Gharbi , Y. Shih, G. Chaurasia, J. Ragan-Kelley, S. Paris, F. Durand, "Transform recipes for efficient cloud photo enhancement," ACM Trans Graph 34(6), 2015.