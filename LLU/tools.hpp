#pragma once

#include<opencv2/opencv.hpp>
#include<opencp.hpp>
#include <iostream>
#include <sstream>
#define _USE_MATH_DEFINES

using namespace cv;
using namespace std;

void iterative_BF(cv::Mat& src, cv::Mat& dest, const int r, const double sigma_c, const double sigma_s, const int n, bool isShowCount = true);

class GaussianPyramid {
public:
	// Construct a Gaussian pyramid of the given image. The number of levels does
	// not count the base, which is just the given image. So, the pyramid will
	// end up having num_levels + 1 levels. The image is converted to 64-bit
	// floating point for calculations.
	GaussianPyramid(const cv::Mat& image, int num_levels);

	// Indicates that this is a subimage. If the start index is odd, this is
	// necessary to make the higher levels the correct size.
	GaussianPyramid(const cv::Mat& image, int num_levels,
		const std::vector<int>& subwindow);

	// Move constructor for having STL containers of GaussianPyramids.
	GaussianPyramid(GaussianPyramid&& other);

	// No copying or assigning.
	GaussianPyramid(const GaussianPyramid&) = delete;
	GaussianPyramid& operator=(const GaussianPyramid&) = delete;

	const cv::Mat& operator[](int level) const { return pyramid_[level]; }

	// Expand the given level a set number of times. The argument times must be
	// less than or equal to level, since the pyramid is used to determine the
	// size of the output. Having level equal to times will upsample the image to
	// the initial pixel dimensions.
	cv::Mat Expand(int level, int times) const;

	template<typename T>
	static void Expand(const cv::Mat& input,
		int row_offset,
		int col_offset,
		cv::Mat& output);

	// Output operator, prints level sizes.
	friend std::ostream& operator<<(std::ostream& output,
		const GaussianPyramid& pyramid);

	static void GetLevelSize(const std::vector<int> base_subwindow,
		int level,
		std::vector<int>* subwindow);
private:
	template<typename T>
	void PopulateTopLevel(int row_offset, int col_offset);

	// i = -2, -1, 0, 1, 2
	// a = 0.3 - Broad blurring Kernel
	// s = 0.4   Gaussian-like kernel
	// a = 0.5 - Triangle
	// a = 0.6 - Trimodal (Negative lobes)
	static double WeightingFunction(int i, double a)
	{
		switch (i) {
		case 0: return a;
		case -1: case 1: return 0.25;
		case -2: case 2: return 0.25 - 0.5 * a;
		}
		return 0;
	}


	void GetLevelSize(int level, std::vector<int>* subwindow) const;

	constexpr static const double kA = 0.4;

private:
	std::vector<cv::Mat> pyramid_;
	std::vector<int> subwindow_;
};

template<typename T>
void GaussianPyramid::PopulateTopLevel(int row_offset, int col_offset) {
	cv::Mat& previous = pyramid_[pyramid_.size() - 2];
	cv::Mat& top = pyramid_.back();

	// Calculate the end indices, based on where (0,0) is centered on the
	// previous level.
	const int kEndRow = row_offset + 2 * top.rows;
	const int kEndCol = col_offset + 2 * top.cols;
	for (int y = row_offset; y < kEndRow; y += 2) {
		for (int x = col_offset; x < kEndCol; x += 2) {
			T value = 0;
			double total_weight = 0;

			int row_start = std::max(0, y - 2);
			int row_end = std::min(previous.rows - 1, y + 2);
			for (int n = row_start; n <= row_end; n++) {
				double row_weight = WeightingFunction(n - y, kA);

				int col_start = std::max(0, x - 2);
				int col_end = std::min(previous.cols - 1, x + 2);
				for (int m = col_start; m <= col_end; m++) {
					double weight = row_weight * WeightingFunction(m - x, kA);
					total_weight += weight;
					value += weight * previous.at<T>(n, m);
				}
			}
			top.at<T>(y >> 1, x >> 1) = value / total_weight;
		}
	}
}

template<typename T>
void GaussianPyramid::Expand(const cv::Mat& input,
	int row_offset,
	int col_offset,
	cv::Mat& output)
{
	cv::Mat upsamp = cv::Mat::zeros(output.rows, output.cols, input.type());
	cv::Mat norm = cv::Mat::zeros(output.rows, output.cols, CV_64F);

	for (int i = row_offset; i < output.rows; i += 2)
	{
		for (int j = col_offset; j < output.cols; j += 2)
		{
			upsamp.at<T>(i, j) = input.at<T>(i >> 1, j >> 1);
			norm.at<double>(i, j) = 1;
		}
	}

	cv::Mat filter(5, 5, CV_64F);
	for (int i = -2; i <= 2; i++)
	{
		for (int j = -2; j <= 2; j++)
		{
			filter.at<double>(i + 2, j + 2) =
				WeightingFunction(i, kA) * WeightingFunction(j, kA);
		}
	}

	for (int i = 0; i < output.rows; i++) {
		int row_start = std::max(0, i - 2);
		int row_end = std::min(output.rows - 1, i + 2);
		for (int j = 0; j < output.cols; j++) {
			int col_start = std::max(0, j - 2);
			int col_end = std::min(output.cols - 1, j + 2);

			T value = 0;
			double total_weight = 0;
			for (int n = row_start; n <= row_end; n++) {
				for (int m = col_start; m <= col_end; m++) {
					double weight = filter.at<double>(n - i + 2, m - j + 2);
					value += weight * upsamp.at<T>(n, m);
					total_weight += weight * norm.at<double>(n, m);
				}
			}
			output.at<T>(i, j) = value / total_weight;
		}
	}
}



// Local Laplacian Filter
class LaplacianPyramid {
public:
	// Construct a blank Laplacian pyramid to be filled in by the user.
	//
	// Arguments:
	//  rows        The number of rows in the base level.
	//  cols        The number of columns of the base level.
	//  channels    The number of channels in the represented image.
	//  num_levels  The number of levels of the pyramid (excluding the top, which
	//              is the residual, or top of the Gaussian pyramid)
	LaplacianPyramid(int rows, int cols, int num_levels);
	LaplacianPyramid(int rows, int cols, int channels, int num_levels);

	// Construct the Laplacian pyramid of an image.
	//
	// Arguments:
	//  image      The input image. Can be any data type, but will be converted
	//             to double. Can be either 1 or 3 channels.
	//  num_levels The number of levels for the pyramid (excluding the top, which
	//             is the residual, or top of the Gaussian pyramid)
	//  subwindow  If this is a subimage [start_row, end_row, start_col, end_col]
	//             Both ends are inclusive.
	LaplacianPyramid(const cv::Mat& image, int num_levels);
	LaplacianPyramid(const cv::Mat& image, int num_levels,
		const std::vector<int>& subwindow);

	// Move constructor if you want STL containers using emplace_back().
	LaplacianPyramid(LaplacianPyramid&& other);

	// No copying or assigning (too much memory footprint).
	LaplacianPyramid(const LaplacianPyramid&) = delete;
	LaplacianPyramid& operator=(const LaplacianPyramid&) = delete;

	// Get a level of the pyramid.
	const cv::Mat& operator[](int level) const { return pyramid_[level]; }
	cv::Mat& operator[](int level) { return pyramid_[level]; }

	// Element access.
	template<typename T>
	T& at(int level, int row, int col) {
		return pyramid_[level].at<T>(row, col);
	}

	// Reconstruct the image from the pyramid.
	cv::Mat Reconstruct() const;

	// Get the recommended number of levels given the input size and the desired
	// size of the residual image.
	static int GetLevelCount(int rows, int cols, int desired_base_size);

	// Output operator. Outputs level sizes.
	friend std::ostream& operator<<(std::ostream& output,
		const LaplacianPyramid& pyramid);

private:
	std::vector<cv::Mat> pyramid_;
	std::vector<int> subwindow_;
};

class RemappingFunction
{
public:
	RemappingFunction(double alpha, double beta);
	~RemappingFunction();

	double alpha() const { return alpha_; }
	void set_alpha(double alpha) { alpha_ = alpha; }

	double beta() const { return beta_; }
	void set_beta(double beta) { beta_ = beta; }

	void Evaluate(double value, double reference, double sigma_r, double& output);
	void Evaluate(const Vec3d& value, const Vec3d& reference, double sigma_r, Vec3d& output);
	void Evaluate(const Vec3f& value, const Vec3f& reference, float sigma_r, Vec3f& output);

	template<typename T>
	void Evaluate(const Mat& input, Mat& output, const T& reference, double sigma_r);
	template<typename T>
	void Evaluate(const Mat& input, Mat& output, const T& reference, float sigma_r);

private:
	double DetailRemap(const double delta, const double sigma_r);
	double EdgeRemap(double delta);

	double SmoothStep(double x_min, double x_max, double x);

private:
	double alpha_, beta_;
};

inline double RemappingFunction::DetailRemap(const double delta, const double sigma_r)
{
	double fraction = delta / sigma_r;
	double polynomial = pow(fraction, alpha_);
	if (alpha_ < 1)
	{
		const double kNoiseLevel = 0.01;
		double blend = SmoothStep(kNoiseLevel,
			2 * kNoiseLevel, fraction * sigma_r);
		polynomial = blend * polynomial + (1 - blend) * fraction;
	}

	return polynomial;
}

inline double RemappingFunction::EdgeRemap(double delta)
{
	return beta_ * delta;
}

template<typename T>
void RemappingFunction::Evaluate(const cv::Mat& input, cv::Mat& output, const T& reference, double sigma_r)
{
	output.create(input.rows, input.cols, input.type());
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			Evaluate(input.at<T>(i, j), reference, sigma_r, output.at<T>(i, j));
		}
	}
}

template<typename T>
void RemappingFunction::Evaluate(const cv::Mat& input, cv::Mat& output, const T& reference, float sigma_r)
{
	output.create(input.rows, input.cols, input.type());
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			Evaluate(input.at<T>(i, j), reference, sigma_r, output.at<T>(i, j));
		}
	}
}

template<typename T, typename S>
void LocalLaplacianFilter_(const Mat &input, Mat &dst, S alpha, S beta, S sigma_r)
{
	RemappingFunction r(alpha, beta);

	int num_levels = LaplacianPyramid::GetLevelCount(input.rows, input.cols, 30);
	//cout << "Number of levels: " << num_levels << endl;

	const int kRows = input.rows;
	const int kCols = input.cols;

	GaussianPyramid gauss_input(input, num_levels);

	// Construct the unfilled Laplacian pyramid of the output. Copy the residual over from the top of the Gaussian pyramid.
	LaplacianPyramid output(kRows, kCols, input.channels(), num_levels);
	gauss_input[num_levels].copyTo(output[num_levels]);

	// Calculate each level of the output Laplacian pyramid.
	for (int l = 0; l < num_levels; l++)
	{
		int subregion_size = 3 * ((1 << (l + 2)) - 1);
		int subregion_r = subregion_size / 2;

#pragma omp parallel for
		for (int y = 0; y < output[l].rows; y++)
		{
			// Calculate the y-bounds of the region in the full-res image.
			int full_res_y = (1 << l) * y;
			int roi_y0 = full_res_y - subregion_r;
			int roi_y1 = full_res_y + subregion_r + 1;
			Range row_range(max(0, roi_y0), min(roi_y1, kRows));
			int full_res_roi_y = full_res_y - row_range.start;

			for (int x = 0; x < output[l].cols; x++)
			{
				// Calculate the x-bounds of the region in the full-res image.
				int full_res_x = (1 << l) * x;
				int roi_x0 = full_res_x - subregion_r;
				int roi_x1 = full_res_x + subregion_r + 1;
				Range col_range(max(0, roi_x0), min(roi_x1, kCols));
				int full_res_roi_x = full_res_x - col_range.start;

				// Remap the region around the current pixel.
				Mat r0 = input(row_range, col_range);
				Mat remapped;
				r.Evaluate<T>(r0, remapped, gauss_input[l].at<T>(y, x), sigma_r);

				//Construct the Laplacian pyramid for the remapped region and copy the coefficient over to the ouptut Laplacian pyramid.
				LaplacianPyramid tmp_pyr(remapped, l + 1, { row_range.start, row_range.end - 1, col_range.start, col_range.end - 1 });
				output.at<T>(l, y, x) = tmp_pyr.at<T>(l, full_res_roi_y >> l, full_res_roi_x >> l);
			}

			//cout << "Level " << (l + 1) << " (" << output[l].rows << " x "
			//	<< output[l].cols << "), footprint: " << subregion_size << "x"
			//	<< subregion_size << " ... " << round(100.0 * y / output[l].rows)
			//	<< "%\r";
			//cout.flush();
		}

		//stringstream ss;
		//ss << "level" << l << ".png";
		//imwrite(ss.str(), ByteScale(cv::abs(output[l])));
		//cout << endl;
	}

	dst = output.Reconstruct();
}

inline void LocalLaplacianFilter(const Mat &input, Mat &dst, double alpha, double beta, double sigma_r)
{
	Mat src64f, dst64f;
	input.convertTo(src64f, CV_64F, 1 / 255.f);

	LocalLaplacianFilter_<Vec3d, double>(src64f, dst64f, alpha, beta, sigma_r);

	dst64f.convertTo(dst, input.depth(), 255);
}
