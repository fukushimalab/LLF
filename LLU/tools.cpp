#include <opencv2/opencv.hpp>
#include <multiscalefilter/MultiScaleFilter.hpp>
#include "tools.hpp"

using namespace std;
using namespace cv;

void iterative_BF(Mat& src, Mat& dest, const int r, const double sigma_c, const double sigma_s, const int n, bool isShowCount)
{
	vector<Mat> s;
	vector<Mat> d(src.channels());

	split(src, s);
	for (int i = 0; i < src.channels(); i++)
		d[i] = s[i].clone();

	if (isShowCount)cout << "IBF: ";
	for (int i = 0; i < n; i++)
	{
		if (isShowCount)cout << i + 1 << " ";
		for (int c = 0; c < src.channels(); c++)
		{
			Mat temp = d[c].clone();
			cv::bilateralFilter(temp, d[c], r * 2 + 1, sigma_c, sigma_s);
			//d[c] = d[c] + 0.1*(temp-d[c]);
			//imshow("iter", src_copy); waitKey();
		}
	}
	if (isShowCount)cout << endl;
	merge(d, dest);
}

#pragma region LLF
using cv::Mat;
using cv::Vec3d;

GaussianPyramid::GaussianPyramid(const Mat& image, int num_levels)
    : GaussianPyramid(image, num_levels, { 0, image.rows - 1,
                                          0, image.cols - 1 }) {}

GaussianPyramid::GaussianPyramid(GaussianPyramid&& other)
    : pyramid_(move(other.pyramid_)) {}

GaussianPyramid::GaussianPyramid(const Mat& image, int num_levels,
    const vector<int>& subwindow)
    : pyramid_(), subwindow_(subwindow) {
    pyramid_.reserve(num_levels + 1);
    pyramid_.emplace_back();
    image.convertTo(pyramid_.back(), CV_64F);

    // This test verifies that the image is large enough to support the requested
    // number of levels.
    if (image.cols >> num_levels == 0 || image.rows >> num_levels == 0) {
        cerr << "Warning: Too many levels requested. Image size "
            << image.cols << " x " << image.rows << " and  " << num_levels
            << " levels wer requested." << endl;
    }

    for (int l = 0; l < num_levels; l++) {
        const Mat& previous = pyramid_.back();

        // Get the subwindows of the previous level and the current one.
        vector<int> prev_subwindow, current_subwindow;
        GetLevelSize(pyramid_.size() - 1, &prev_subwindow);
        GetLevelSize(pyramid_.size(), &current_subwindow);

        const int kRows = current_subwindow[1] - current_subwindow[0] + 1;
        const int kCols = current_subwindow[3] - current_subwindow[2] + 1;

        // If the subwindow starts on even indices, then (0,0) of the new level is
        // centered on (0,0) of the previous level. Otherwise, it's centered on
        // (1,1).
        int row_offset = ((prev_subwindow[0] % 2) == 0) ? 0 : 1;
        int col_offset = ((prev_subwindow[2] % 2) == 0) ? 0 : 1;

        // Push a new level onto the top of the pyramid.
        pyramid_.emplace_back(kRows, kCols, previous.type());
        Mat& next = pyramid_.back();

        // Populate the next level.
        if (next.channels() == 1) {
            PopulateTopLevel<double>(row_offset, col_offset);
        }
        else if (next.channels() == 3) {
            PopulateTopLevel<Vec3d>(row_offset, col_offset);
        }
    }
}


Mat GaussianPyramid::Expand(int level, int times) const {
    if (times < 1) return pyramid_.at(level);
    times = min(times, level);

    Mat base = pyramid_[level], expanded;

    for (int i = 0; i < times; i++) {
        vector<int> subwindow;
        GetLevelSize(level - i - 1, &subwindow);

        int out_rows = pyramid_[level - i - 1].rows;
        int out_cols = pyramid_[level - i - 1].cols;
        expanded.create(out_rows, out_cols, base.type());

        int row_offset = ((subwindow[0] % 2) == 0) ? 0 : 1;
        int col_offset = ((subwindow[2] % 2) == 0) ? 0 : 1;
        if (base.channels() == 1) {
            Expand<double>(base, row_offset, col_offset, expanded);
        }
        else {
            Expand<Vec3d>(base, row_offset, col_offset, expanded);
        }

        base = expanded;
    }

    return expanded;
}


ostream& operator<<(ostream& output, const GaussianPyramid& pyramid) {
    output << "Gaussian Pyramid:" << endl;
    for (size_t i = 0; i < pyramid.pyramid_.size(); i++) {
        output << "Level " << i << ": " << pyramid.pyramid_[i].cols << " x "
            << pyramid.pyramid_[i].rows;
        if (i != pyramid.pyramid_.size() - 1) output << endl;
    }
    return output;
}

void GaussianPyramid::GetLevelSize(int level, vector<int>* subwindow) const {
    GetLevelSize(subwindow_, level, subwindow);
}

void GaussianPyramid::GetLevelSize(const vector<int> base_subwindow,
    int level,
    vector<int>* subwindow) {
    subwindow->clear();
    subwindow->insert(begin(*subwindow),
        begin(base_subwindow), end(base_subwindow));

    for (int i = 0; i < level; i++) {
        (*subwindow)[0] = ((*subwindow)[0] >> 1) + (*subwindow)[0] % 2;
        (*subwindow)[1] = (*subwindow)[1] >> 1;
        (*subwindow)[2] = ((*subwindow)[2] >> 1) + (*subwindow)[2] % 2;
        (*subwindow)[3] = (*subwindow)[3] >> 1;
    }
}



LaplacianPyramid::LaplacianPyramid(int rows, int cols, int num_levels)
    : LaplacianPyramid(rows, cols, 1, num_levels) {}

LaplacianPyramid::LaplacianPyramid(int rows,
    int cols,
    int channels,
    int num_levels)
    : pyramid_(), subwindow_({ 0, rows - 1, 0, cols - 1 }) {
    pyramid_.reserve(num_levels + 1);
    for (int i = 0; i < num_levels + 1; i++) {
        pyramid_.emplace_back(ceil(rows / (double)(1 << i)),
            ceil(cols / (double)(1 << i)), CV_64FC(channels));
    }
}

LaplacianPyramid::LaplacianPyramid(const Mat& image, int num_levels)
    : LaplacianPyramid(image, num_levels, { 0, image.rows - 1,
                                           0, image.cols - 1 }) {}

LaplacianPyramid::LaplacianPyramid(const Mat& image, int num_levels,
    const std::vector<int>& subwindow)
    : pyramid_(), subwindow_(subwindow) {
    pyramid_.reserve(num_levels + 1);

    Mat input;
    image.convertTo(input, CV_64F);

    GaussianPyramid gauss_pyramid(input, num_levels, subwindow_);
    for (int i = 0; i < num_levels; i++) {
        pyramid_.emplace_back(gauss_pyramid[i] - gauss_pyramid.Expand(i + 1, 1));
    }
    pyramid_.emplace_back(gauss_pyramid[num_levels]);
}

LaplacianPyramid::LaplacianPyramid(LaplacianPyramid&& other)
    : pyramid_(std::move(other.pyramid_)) {}

Mat LaplacianPyramid::Reconstruct() const {
    Mat base = pyramid_.back();
    Mat expanded;

    for (int i = pyramid_.size() - 2; i >= 0; i--) {
        vector<int> subwindow;
        GaussianPyramid::GetLevelSize(subwindow_, i, &subwindow);
        int row_offset = ((subwindow[0] % 2) == 0) ? 0 : 1;
        int col_offset = ((subwindow[2] % 2) == 0) ? 0 : 1;

        expanded.create(pyramid_[i].rows, pyramid_[i].cols, base.type());

        if (base.channels() == 1) {
            GaussianPyramid::Expand<double>(base, row_offset, col_offset, expanded);
        }
        else if (base.channels() == 3) {
            GaussianPyramid::Expand<Vec3d>(base, row_offset, col_offset, expanded);
        }
        base = expanded + pyramid_[i];
    }

    return base;
}

int LaplacianPyramid::GetLevelCount(int rows, int cols, int desired_base_size) {
    int min_dim = std::min(rows, cols);

    double log2_dim = std::log2(min_dim);
    double log2_des = std::log2(desired_base_size);

    return static_cast<int>(std::ceil(std::abs(log2_dim - log2_des)));
}

std::ostream& operator<<(std::ostream& output,
    const LaplacianPyramid& pyramid) {
    output << "Laplacian Pyramid:" << std::endl;
    for (size_t i = 0; i < pyramid.pyramid_.size(); i++) {
        output << "Level " << i << ": " << pyramid.pyramid_[i].cols << " x "
            << pyramid.pyramid_[i].rows;
        if (i != pyramid.pyramid_.size() - 1) output << std::endl;
    }
    return output;
}

RemappingFunction::RemappingFunction(double alpha, double beta)
    : alpha_(alpha), beta_(beta) {}

RemappingFunction::~RemappingFunction() {}

double RemappingFunction::SmoothStep(double x_min, double x_max, double x) {
    double y = (x - x_min) / (x_max - x_min);
    y = max(0.0, min(1.0, y));
    return pow(y, 2) * pow(y - 2, 2);
}

void RemappingFunction::Evaluate(double value,
    double reference,
    double sigma_r,
    double& output) {
    double delta = std::abs(value - reference);
    int sign = value < reference ? -1 : 1;

    if (delta < sigma_r) {
        output = reference + sign * sigma_r * DetailRemap(delta, sigma_r);
    }
    else {
        output = reference + sign * (EdgeRemap(delta - sigma_r) + sigma_r);
    }
}

void RemappingFunction::Evaluate(const cv::Vec3d& value,
    const cv::Vec3d& reference,
    double sigma_r,
    cv::Vec3d& output)
{
    cv::Vec3d delta = value - reference;
    cv::Vec3d mag = Vec3d(abs(delta.val[0]), abs(delta.val[1]), abs(delta.val[2]));

    for (int c = 0; c < 3; c++)
    {
        if (mag[c] > 1e-10) delta[c] /= mag[c];

        if (mag[c] < sigma_r)
        {
            output[c] = reference[c] + delta[c] * sigma_r * DetailRemap(mag[c], sigma_r);
        }
        else
        {
            output[c] = reference[c] + delta[c] * (EdgeRemap(mag[c] - sigma_r) + sigma_r);
        }
    }
}
#pragma endregion

void generateDataFromTransformRecipes()
{
	const string dir = "data/";
	vector<int> imgend = { 20,20,12,20,12,10,20,10,19 };

	// IBF parameters
	//const int iterationBF = 10;
	const int iterationBF = 3;
	const double sigma_c = 15.0;
	const double sigma_s = 10.0;
	const int r_BF = cvRound(3 * sigma_s);
#define BF_PARAMETERS "c20s10"
	// L0 parameters
	float l0lambda = 0.005f;
	float l0kapper = 1.5f;
	//#define L0_PARAMETERS "l001k15"
#define L0_PARAMETERS "l0005k15"

	// LocalLaplacian parameters
	// Exponent for the detail remapping function.
	//(< 1 for detail enhancement, > 1 for detail suppression)
	const double alpha_llf = 0.5;
	// Slope for edge remapping function 
	//(< 1 for tone mapping, > 1 for inverse tone mapping)
	const double beta_llf = 0.5;
	// Edge threshold (in image range space)
	const double sigma_r_llf = 0.9;
	// haze remove parameters
	int hz_r_dark = 4;
	double hz_rate = 0.1;
	int hz_r_joint = 15;
	double hz_e_joint = 0.6;

#pragma omp parallel for schedule (dynamic)
	for (int distidx = 0; distidx < 9; distidx++)
	{
		for (int imgidx = 0; imgidx < imgend[distidx]; imgidx++)
		{
			cp::Timer t;
#pragma omp critical
			cout << distidx << "," << imgidx << endl;
			string dist = "";
			switch (distidx)
			{
			case 0: dist = "Dehazing"; break;//20
			case 1: dist = "DetailManipulation"; break;//20
			case 2: dist = "L0"; break;//12
			case 3: dist = "LocalLaplacian"; break;//20
			case 4: dist = "Matting"; break;//12
			case 5: dist = "Photoshop"; break;//10
			case 6: dist = "PortraitTransfer"; break;//20
			case 7: dist = "Recoloring"; break;//10
			case 8: dist = "StyleTransfer"; break;//19
			default:
				break;
			}

			Mat src = imread("data/" + dist + format("_%04d_src.png", imgidx + 1));
			Mat answer;
			iterative_BF(src, answer, r_BF, sigma_c, sigma_s, iterationBF, false);
			imwrite("data/" + dist + format("_%04d_bf.png", imgidx + 1), answer);

			cp::L0Smoothing(src, answer, l0lambda, l0kapper);
			imwrite("data/" + dist + format("_%04d_lz.png", imgidx + 1), answer);

			LocalLaplacianFilter(src, answer, alpha_llf, beta_llf, sigma_r_llf);
			imwrite("data/" + dist + format("_%04d_ll.png", imgidx + 1), answer);
			cp::HazeRemove hr;
			hr(src, answer, hz_r_dark, hz_rate, hz_r_joint, hz_e_joint);
			imwrite("data/" + dist + format("_%04d_hz.png", imgidx + 1), answer);
		}
	}
}

void sortTransformRecipes(const string dir = "E:/Transform Recipes/")
{
	vector<int> imgend = { 20,20,12,20,12,10,20,10,19 };
	for (int distidx = 0; distidx < 9; distidx++)
	{
		for (int imgidx = 0; imgidx < imgend[distidx]; imgidx++)
		{
			string dist = "";
			switch (distidx)
			{
			case 0: dist = "Dehazing"; break;//20
			case 1: dist = "DetailManipulation"; break;//20
			case 2: dist = "L0"; break;//12
			case 3: dist = "LocalLaplacian"; break;//20
			case 4: dist = "Matting"; break;//12
			case 5: dist = "Photoshop"; break;//10
			case 6: dist = "PortraitTransfer"; break;//20
			case 7: dist = "Recoloring"; break;//10
			case 8: dist = "StyleTransfer"; break;//19
			default:
				break;
			}
			string image = format("%04d/", imgidx + 1);
			string pass = dir + dist + "/" + image;
			Mat un = imread(pass + "unprocessed.png");
			Mat pr = imread(pass + "processed.png");
			imwrite("data/" + dist + format("_%04d_src.png", imgidx + 1), un);
			imwrite("data/" + dist + format("_%04d_dst.png", imgidx + 1), pr);
			//cp::guiAlphaBlend(un, pr);
		}
	}
}