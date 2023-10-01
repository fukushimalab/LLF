#include <jointBilateralUpsample.hpp>
#include <bilateralGuidedUpsample.hpp>
#include <localLUTUpsample.hpp>
#include <nedi.hpp>
#include <spatialfilter/SpatialFilter.hpp>
#include "tools.hpp"

using namespace std;
using namespace cv;
using namespace cp;

//enable upsampling
#define CU
//#define NEDI
#define JBU
#define GIU
#define BGU
#define LLU
//#define ILGU64
//#define LLU_ALL

//const bool gtomit = true;//true: skip GT computing (for processingType=0 or1)
const bool gtomit = false;//false: compute GT image (for processingType=0 or1)
//constexpr int processingType = 0;//HRHPRecompute (download: http://imagecompression.info/test_images/)
//constexpr int processingType = 1;//TRecipesRecompute (download: https://groups.csail.mit.edu/graphics/xform_recipes/dataset.html)
constexpr int processingType = 2;//MTASubjective (download: https://fukushimalab.github.io/LLF/)
//constexpr int processingType = 3;//TRecipesMTARecompute (download T-Recipes and call generateDataFromTransformRecipes function )
//constexpr int processingType = 4;//TRecipesDirect (download T-Recipes and call sortTransformRecipes)

#pragma region parameters
// IBF parameters
const int iterationBF = 3;
const double sigma_c = 15.0;
const double sigma_s = 10.0;
const int r_BF = cvRound(3 * sigma_s);
// L0 parameters
const float l0lambda = 0.005f;
const float l0kapper = 1.5f;

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
const int hz_r_dark = 4;
const double hz_rate = 0.1;
const int hz_r_joint = 15;
const double hz_e_joint = 0.6;
#pragma endregion

enum
{
	ITERLATIVE_BF,
	L0_SMOOTH,
	LOCAL_LAPLACIAN_FILTER,
	HAZE_REMOVE,
	COLOR_CONTRAST,
	INPAINT,
	STYLIZATION
};
string getProcessingDataTestName(const int emethod)
{
	string ret = "";
	switch (emethod)
	{
	case ITERLATIVE_BF: ret = "Iterative Bilateral Filter"; break;
	case L0_SMOOTH: ret = "L0smooth"; break;
	case LOCAL_LAPLACIAN_FILTER: ret = "Local Laplacian Filter"; break;
	case HAZE_REMOVE: ret = "Haze Remove"; break;
	case COLOR_CONTRAST: ret = "Color contrast"; break;
	case INPAINT: ret = "Inpaint"; break;
	case STYLIZATION: ret = "Stylization"; break;
	default: ret = "undefined"; break;
	}
	return ret;

}
int getProcessingTRecipesMod(const int distIdx)
{
	int ret = 0;
	switch (distIdx)
	{
	case 0: ret = 20; break;
	case 1: ret = 20; break;
	case 2: ret = 12; break;
	case 3: ret = 20; break;
	case 4: ret = 12; break;
	case 5: ret = 10; break;
	case 6: ret = 20; break;
	case 7: ret = 10; break;
	case 8: ret = 19; break;
	default: break;
	}
	return ret;
}
string getProcessingTRecipesName(const int distIdx)
{
	string ret = "";
	switch (distIdx)
	{
	case 0: ret = "Dehazing"; break;//20
	case 1: ret = "DetailManipulation"; break;//20
	case 2: ret = "L0"; break;//12
	case 3: ret = "LocalLaplacian"; break;//20
	case 4: ret = "Matting"; break;//12
	case 5: ret = "Photoshop"; break;//10
	case 6: ret = "PortraitTransfer"; break;//20
	case 7: ret = "Recoloring"; break;//10
	case 8: ret = "StyleTransfer"; break;//19
	default: break;
	}
	return ret;
}

void processingData(Mat& src, Mat& dst, const int upsample_size, const int dist2Idx, string img_name)
{
	//image processing in downsampled domain
	const double sub_rate = 1.0 / upsample_size;
#pragma region ip_in_ds
	if (dist2Idx == ITERLATIVE_BF)
	{
		const int r = (int)ceil(3.0 * sigma_s * sub_rate);
		iterative_BF(src, dst, r, sigma_c, sigma_s * sub_rate, iterationBF, false);
		if (img_name != "")imwrite("img/answer/bf/" + img_name + ".png", dst);
	}
	else if (dist2Idx == L0_SMOOTH)
	{
		cp::L0Smoothing(src, dst, l0lambda, l0kapper);
		if (img_name != "")imwrite("img/answer/l0/" + img_name + ".png", dst);
	}
	else if (dist2Idx == LOCAL_LAPLACIAN_FILTER)
	{
		LocalLaplacianFilter(src, dst, alpha_llf, beta_llf, sigma_r_llf);
		if (img_name != "")imwrite("img/answer/ll/" + img_name + ".png", dst);
	}
	else if (dist2Idx == HAZE_REMOVE)
	{
		cp::HazeRemove hr;
		const int r = int(hz_r_joint * sub_rate);
		const int dr = int(hz_r_dark * sub_rate);
		hr(src, dst, dr, hz_rate, r, hz_e_joint);
		//hr.removeFastGlobalSmootherFilter(srclow, low, 1, 0.1, 20000, 215, 0.25, 3);
		if (img_name != "")imwrite("img/answer/hr/" + img_name + ".png", dst);
	}
	else if (dist2Idx == COLOR_CONTRAST)
	{
		cvtColor(src, dst, COLOR_BGR2YUV);
		vector<Mat> sp;
		split(dst, sp);
		cp::contrastGamma(sp[0], sp[0], 2.0);
		merge(sp, dst);
		cvtColor(dst, dst, COLOR_YUV2BGR);
		if (img_name != "")imwrite("img/answer/cc/" + img_name + ".png", dst);
	}
	else if (dist2Idx == INPAINT)
	{
		Mat mask_ = imread("img/kodim01_mask.png", 0);
		//Mat mask_ = imread("img/kodim02_mask.png", 0);
		Mat mask;
		resize(mask_, mask, src.size(), 0, 0, INTER_NEAREST);
		xphoto::inpaint(src, mask, dst, 0);
		if (img_name != "")imwrite("img/answer/ip/" + img_name + ".png", dst);
	}
	else if (dist2Idx == STYLIZATION)
	{
		stylization(src, dst);
		if (img_name != "")imwrite("img/answer/st/" + img_name + ".png", dst);
	}
	else
	{
		cout << "undefined at processingData" << endl;
	}
#pragma endregion
}

Mat loadData(const int dist2Idx, string img_name)
{
	Mat answer;
	if (dist2Idx == ITERLATIVE_BF)
	{
		answer = imread("img/answer/bf/" + img_name + ".png");
	}
	else if (dist2Idx == L0_SMOOTH)
	{
		answer = imread("img/answer/l0/" + img_name + ".png");
	}
	else if (dist2Idx == LOCAL_LAPLACIAN_FILTER)
	{
		answer = imread("img/answer/ll/" + img_name + ".png");
	}
	else if (dist2Idx == HAZE_REMOVE)
	{
		answer = imread("img/answer/hz/" + img_name + ".png");
	}
	else if (dist2Idx == COLOR_CONTRAST)
	{
		answer = imread("img/answer/cc/" + img_name + ".png");
	}
	else if (dist2Idx == INPAINT)
	{
		answer = imread("img/answer/ip/" + img_name + ".png");
	}
	else if (dist2Idx == STYLIZATION)
	{
		answer = imread("img/answer/st/" + img_name + ".png");
	}
	else
	{
		cout << "undefined at loadData" << endl;
	}
	return answer;
}

//int processingType = 0;//HRHPRecompute
void loadHRHPAndProcessingDataHigh(Mat& src, Mat& answer, const int dist2Idx, const int imageIdx, const int color, Size size = Size(1024, 1024))
{
#pragma region readImage
	const int iidx = imageIdx % 16;
	string dir = "img/HRHP/";
	string img_name = "";

	switch (iidx)
	{
	default:
	case 0: img_name = "artificial"; break;
	case 1: img_name = "big_building"; break;
	case 2: img_name = "big_tree"; break;
	case 3: img_name = "bridge"; break;
	case 4: img_name = "cathedral"; break;
	case 5:	img_name = "deer"; break;
	case 6: img_name = "fireworks"; break;
	case 7: img_name = "flower_foveon"; break;
	case 8: img_name = "hdr"; break;
	case 9: img_name = "leaves_iso_200"; break;
	case 10: img_name = "leaves_iso_200"; break;
	case 11: img_name = "leaves_iso_1600"; break;
	case 12: img_name = "nightshot_iso_100"; break;
	case 13: img_name = "nightshot_iso_1600"; break;
	case 14: img_name = "spider_web"; break;
	case 15: img_name = "zone_plate"; break;
	}

	//string fname = "img/leaf.png";//overwrite
	src = cropMultipleFloor(imread(dir + img_name + ".png", color), 128);
	if (src.empty())cout << "file open error" << dir + img_name << endl;

	if (color == 0)
	{
		cvtColor(src, src, COLOR_GRAY2BGR);
	}
	resize(src, src, size);

#pragma endregion

	answer.create(src.rows, src.cols, src.type());
	if (gtomit == true) //accelerate image processing for large size image
	{
		answer = loadData(dist2Idx, img_name);
	}
	else //compute
	{
		cp::Timer t("compute GT", cp::TIME_MSEC);
		processingData(src, answer, 1, dist2Idx, img_name);
		//cp::guiAlphaBlend(answer, src);
	}
}
//int processingType = 1;//TRecipesRecompute
void loadTRecipesAndProcessingDataHigh(Mat& src, Mat& answer, const int distIdx, const int dist2Idx, const int imageIdx, const int color, Size size = Size(1024, 1024))
{
#pragma region readImage
	const string dist = getProcessingTRecipesName(distIdx) + "_";
	const int iidx = imageIdx % getProcessingTRecipesMod(distIdx);
	const string img_name = dist + format("%04d", iidx + 1);
	string fname = "img/TRecipesRecomp/" + img_name + "_src.png";
	src = cropMultipleFloor(imread(fname, color), 128);
	if (src.empty())cout << "file open error" << fname << endl;

	if (color == 0)
	{
		cvtColor(src, src, COLOR_GRAY2BGR);
	}
	resize(src, src, size);

#pragma endregion

	answer.create(src.rows, src.cols, src.type());
	if (gtomit == true) //accelerate image processing for large size image
	{
		answer = loadData(dist2Idx, img_name);
	}
	else //compute
	{
		cp::Timer t("compute GT", cp::TIME_MSEC);
		processingData(src, answer, 1, dist2Idx, img_name);
		//cp::guiAlphaBlend(answer, src);
	}
}

//constexpr int processingType = 2;//MTASubjective
void loadMTASubjective(Mat& src, Mat& answer, const int dist2Idx, const int imageIdx, int color, Size size = Size(1024, 1024))
{
	string dir_original = "./img/MTA_subjective_assessment/original/";
	string dir_answer = "./img/MTA_subjective_assessment/processed_source/";

	string dist = "";
	const int didx = dist2Idx % 4;
	switch (didx)
	{
	case 0: dist = "bf"; break;
	case 1: dist = "l0"; break;
	case 2: dist = "ll"; break;
	case 3: dist = "hz"; break;
	default: break;
	}

	const int iidx = imageIdx % 5;
	src = imread(dir_original + cv::format("img%02d.png", (didx == 3 && iidx > 1) ? iidx + 10 : iidx));

	string img = cv::format("img%02d", iidx);
	string mes = dist + "_cpy_" + img + "_res01.png";
	answer = imread(dir_answer + mes);

	if (color == 0)
	{
		cvtColor(src, src, COLOR_GRAY2BGR);
		cvtColor(answer, answer, COLOR_GRAY2BGR);
	}
	resize(src, src, size);
	resize(answer, answer, size);
}

//int processingType = 3;//TRecipesMTARecompute
void loadTRecipesMTARecompute(Mat& src, Mat& answer, const int distIdx, const int dist2Idx, const int imageIdx, const int color, Size size = Size(1024, 1024))
{
	const string dist = getProcessingTRecipesName(distIdx) + "_";
	const int iidx = imageIdx % getProcessingTRecipesMod(distIdx);
	src = cropMultipleFloor(imread("img/TRecipesRecomp/" + dist + format("%04d_src.png", iidx + 1), 1), 128);
	if (src.empty())cout << "img/TRecipesRecomp/" + dist + format("%04d_src.png", iidx + 1) << endl;

	if (dist2Idx == 0) answer = cropMultipleFloor(imread("img/TRecipesRecomp/" + dist + format("%04d_bf.png", iidx + 1), color), 128);
	if (dist2Idx == 1) answer = cropMultipleFloor(imread("img/TRecipesRecomp/" + dist + format("%04d_lz.png", iidx + 1), color), 128);
	if (dist2Idx == 2) answer = cropMultipleFloor(imread("img/TRecipesRecomp/" + dist + format("%04d_ll.png", iidx + 1), color), 128);
	if (dist2Idx == 3) answer = cropMultipleFloor(imread("img/TRecipesRecomp/" + dist + format("%04d_hz.png", iidx + 1), color), 128);
	if (color == 0)
	{
		cvtColor(src, src, COLOR_GRAY2BGR);
		cvtColor(answer, answer, COLOR_GRAY2BGR);
	}
	resize(src, src, size);
	resize(answer, answer, size);
}

//constexpr int processingType = 4;//TRecipesDirect
void loadTRecipesDirect(Mat& src, Mat& answer, const int distIdx, const int imageIdx, int color, Size size = Size(1024, 1024))
{
	const string dist = getProcessingTRecipesName(distIdx) + "/";
	src = cropMultipleFloor(imread("img/Transform Recipes/" + dist + format("src%02d.png", imageIdx + 1), color), 128);
	answer = cropMultipleFloor(imread("img/Transform Recipes/" + dist + format("dst%02d.png", imageIdx + 1), color), 128);
	if (color == 0)
	{
		cvtColor(src, src, COLOR_GRAY2BGR);
		cvtColor(answer, answer, COLOR_GRAY2BGR);
	}
	resize(src, src, size);
	resize(answer, answer, size);
}

void testGUI()
{
	//parameters
	string wname = "uptest";
	namedWindow(wname);
	moveWindow(wname, 50, 50);

	int color = 1; createTrackbar("color", "", &color, 1);
	int imageIdx = 0; createTrackbar("imgIdx", "", &imageIdx, 10);
	int distRecipeIndex = 0; createTrackbar("distIdx", "", &distRecipeIndex, 8);
	int distRecompIndex = 0; createTrackbar("dist2Idx", "", &distRecompIndex, 3);
	int index = 8; createTrackbar("showIndex", "", &index, 14);
	int alpha = 0; createTrackbar("showAlpha", "", &alpha, 100);
	int metrics = 0; createTrackbar("metrics", "", &metrics, 2);
	int res = 1; createTrackbar("upresolution", "", &res, 5); setTrackbarMin("res", "", 1);

	int iteration = 1; createTrackbar("iteration", "", &iteration, 100);

	int inter_d = 11; createTrackbar("downsample_method", "", &inter_d, 11);
	int sigma_clip = 3; createTrackbar("down gauss clip", "", &sigma_clip, 20);

	int rad = 2; createTrackbar("r", "", &rad, 20);

	int ss = 10; createTrackbar("jbu:ss", "", &ss, 120);
	int sr = 15; createTrackbar("jbu:sr", "", &sr, 255);

	int guided_eps = 5; createTrackbar("giu:eps*0.001", "", &guided_eps, 1000);

	int num_bgu_bin = 8;  createTrackbar("bgu:bin", "", &num_bgu_bin, 200);
	int num_bgu_div_space = 3;  createTrackbar("bgu:div_s", "", &num_bgu_div_space, 50);

	//LLU
	int num_lut_bit = 8; createTrackbar("llu:numLUTbit", "", &num_lut_bit, 8);
	int llut_lutfilter_rad = 3; createTrackbar("llu:lut_r", "", &llut_lutfilter_rad, 20);
	int llut_build_method = 0; createTrackbar("llu:buildlut", "", &llut_build_method, (int)LocalLUTUpsample::BUILD_LUT::SIZE - 1);
	int llut_boundary_method = (int)LocalLUTUpsample::BOUNDARY::LINEAR;
	//int llut_boundary_method = LocalLUT::BOUNDARY_REPLICATE;
	//int llut_boundary_method = LocalLUT::BOUNDARY_0_255;
	createTrackbar("llu:bound", "", &llut_boundary_method, (int)LocalLUTUpsample::BOUNDARY::SIZE - 1);
	int llut_tensorup_method = (int)LocalLUTUpsample::UPTENSOR::BOX4; //createTrackbar("llu:tensorup", "", &llut_tensorup_method, LocalLUT::LOCAL_LUT_UPTENSOR_SIZE - 1);

	int tsigmas = 15; createTrackbar("llu:tup space", "", &tsigmas, 100);
	int tsigmar = 30; createTrackbar("llu:tup range", "", &tsigmar, 1000);
	int tkernelr = 2; createTrackbar("llu:tup r*2", "", &tkernelr, 10);

	int llut_use_offsetmap = 1; createTrackbar("llu:offsetmap", "", &llut_use_offsetmap, 1);
	int llut_rep_offset = 0; createTrackbar("llu:rep_offset", "", &llut_rep_offset, 50);
	setTrackbarMin("llu:rep_offset", "", -50);

	int p = 10; createTrackbar("p", "", &p, 100);
	int ca = 50; createTrackbar("ca", "", &ca, 300);
	int sx = 0; createTrackbar("sx", "", &sx, 80);

	cp::Timer totalTime;
	cp::Timer time[30];
	for (auto& e : time)
	{
		e.init("", cp::TIME_MSEC, false);
	}

	cp::ConsoleImage ci(Size(900, 600), "console");
	moveWindow("console", 600, 100);

	cp::UpdateCheck ucheck(inter_d, sigma_clip, p);
	cp::UpdateCheck llucheck(num_lut_bit, llut_lutfilter_rad, llut_build_method, llut_boundary_method, llut_tensorup_method, llut_use_offsetmap, llut_rep_offset);
	cp::UpdateCheck ucDataIndex(color, imageIdx, distRecipeIndex, res, distRecompIndex);

	Mat srclow, srchigh;
	Mat dstlow, dsthigh;
	Mat answer;
	Mat show;

	cp::NewEdgeDirectedInterpolation nedi;
	cp::JointBilateralUpsample jbu;
	cp::GuidedImageFilter gfcp;
	cp::BilateralGuidedUpsample bgu;
	LocalLUTUpsample locallut;

	int key = 0;
	int upsample_size = 0;
	while (key != 'q')
	{
		totalTime.start();
#pragma region update
		const bool isDataUpdate = ucDataIndex.isUpdate(color, imageIdx, distRecipeIndex, res, distRecompIndex);
		if (ucheck.isUpdate(inter_d, sigma_clip, p) || isDataUpdate)
		{
			upsample_size = (int)pow(2, res);
			double parameter = sigma_clip;
			if (inter_d == 7)parameter *= 0.5;

			if (processingType <= 1)
			{
				if (processingType == 0) loadHRHPAndProcessingDataHigh(srchigh, answer, distRecompIndex, imageIdx, color);
				if (processingType == 1) loadTRecipesAndProcessingDataHigh(srchigh, answer, distRecipeIndex, distRecompIndex, imageIdx, color);
				cp::downsample(srchigh, srclow, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				processingData(srclow, dstlow, upsample_size, distRecompIndex, "");
			}
			else
			{
				if (processingType == 2) loadMTASubjective(srchigh, answer, distRecompIndex, imageIdx, color);
				if (processingType == 3) loadTRecipesMTARecompute(srchigh, answer, distRecipeIndex, distRecompIndex, imageIdx, color);
				if (processingType == 4) loadTRecipesDirect(srchigh, answer, distRecipeIndex, imageIdx, color);
				cp::downsample(srchigh, srclow, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer, dstlow, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
			}
		}
#pragma endregion

#pragma region console_header
		ci(COLOR_YELLOW, "press ctrl+p for showing parameter trackbar");
		ci("H: Size %d x %d", srchigh.cols, srchigh.rows);
		ci("L: Size %d x %d", srclow.cols, srclow.rows);

		if (processingType == 0) ci(getProcessingDataTestName(distRecompIndex));
		if (processingType == 1) ci(getProcessingDataTestName(distRecompIndex));
		if (processingType == 2) ci(getProcessingTRecipesName(distRecipeIndex) + "+" + getProcessingDataTestName(distRecompIndex));
		if (processingType == 3) ci(getProcessingDataTestName(distRecompIndex));
		if (processingType == 4) ci(getProcessingTRecipesName(distRecipeIndex));

		//ci("Processing time %5.3f %5.3f x%5.3f:", processing_fulltime, processing_subtime, processing_fulltime / processing_subtime);
		ci("Down sampling method: " + cp::getDowsamplingMethod(cp::Downsample(inter_d)));
		ci("Local LUT build method:" + locallut.getBuildingLUTMethod(LocalLUTUpsample::BUILD_LUT(llut_build_method)));
		ci("Local LUT boundary method:" + locallut.getBoundaryMethod((LocalLUTUpsample::BOUNDARY)llut_boundary_method));
		ci("Local LUT TensorUp method:" + locallut.getTensorUpsamplingMethod(LocalLUTUpsample::UPTENSOR(llut_tensorup_method)));
		if (llut_use_offsetmap == 1)ci(COLOR_GREEN, "Local LUT isOffset: true");
		if (llut_use_offsetmap == 0)ci(COLOR_RED, "Local LUT isOffset: false");
		ci("Time samples:" + to_string(time[0].getStatSize()));
		ci("Time Total: %f", totalTime.getLapTimeMedian());
#pragma endregion

#pragma region upsampling
		int lindex = 0;
		auto console = [&](string label, Mat& answer, Mat& dsthigh, int metrics)
			{
				if (metrics == 0)
				{
					if (index == lindex)
					{
						ci(COLOR_GREEN, format("%02d: %s PSNR: %5.2f, time: %6.2f", lindex, label, PSNR(answer, dsthigh), time[lindex].getLapTimeMedian()));
						dsthigh.copyTo(show);
					}
					else
					{
						ci(format("%02d: %s PSNR: %5.2f, time: %6.2f", lindex, label, PSNR(answer, dsthigh), time[lindex].getLapTimeMedian()));
					}
				}
				if (metrics == 1)
				{
					if (index == lindex)
					{
						ci(COLOR_GREEN, format("%02d: %s PSNR: %5.2f, time: %6.2f", lindex, label, PSNR(answer, dsthigh), time[lindex].getLapTimeMedian()));
						dsthigh.copyTo(show);
					}
					else
					{
						ci(format("%02d: %s PSNR: %5.2f, time: %6.2f", lindex, label, PSNR(answer, dsthigh), time[lindex].getLapTimeMedian()));
					}
				}
				if (metrics == 2)
				{
					if (index == lindex)
					{
						ci(COLOR_GREEN, format("%02d: %s SSIM: %5.3f, time: %6.2f", lindex, label, cp::getSSIM(answer, dsthigh), time[lindex].getLapTimeMedian()));
						dsthigh.copyTo(show);
					}
					else
					{
						ci(format("%02d: %s SSIM: %5.3f, time: %6.2f", lindex, label, cp::getSSIM(answer, dsthigh), time[lindex].getLapTimeMedian()));
					}
				}
				lindex++;
			};
#pragma region cpy
		for (int i = 0; i < iteration; i++)
		{
			time[lindex].start();
			srchigh.copyTo(dsthigh);
			time[lindex].getpushLapTime();
		}
		console("COPY  ", answer, dsthigh, metrics);

#pragma endregion

#ifdef CU
		//#define CUBIC_CV
		//dst.setTo(0);
		for (int i = 0; i < iteration; i++)
		{
			time[lindex].start();
			//upSampleCV(low, dst, up_size, INTER_CUBIC, -sx * 0.1, -sx * 0.1);
			cp::upsampleCubic(dstlow, dsthigh, upsample_size, -ca * 0.01);
			//upsampleCubic_nonparallel(low, dst, upsample_size, -ca * 0.01);
			//upsampleCubic_parallel(low, dst, upsample_size, -ca * 0.01);
			time[lindex].getpushLapTime();
		}
		console("CUBIC ", answer, dsthigh, metrics);

#ifdef CUBIC_CV
		time[lindex].start();
		//cv::setNumThreads(1);
		cv::ipp::setUseIPP(true);
		resize(low, dst, src.size(), 0, 0, INTER_CUBIC);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%d: CV1 PSNR: %5.2f, time: %6.2f", lindex, cp::PSNRBB(answer, dst, bb), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%d: CV1 PSNR: %5.2f, time: %6.2f", lindex, cp::PSNRBB(answer, dst, bb), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		time[lindex].start();
		cp::resizeShift(low, dst, dst.cols / low.cols, INTER_CUBIC, -sx * 0.1, -sx * 0.1);
		//upsampleCubic_nonparallel(low, dst, upsample_size, -ca * 0.01);
		//upsampleCubic_parallel(low, dst, upsample_size, -ca * 0.01);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			//ci(COLOR_GREEN, format("%d: MC2 PSNR: %5.2f, time: %6.2f", lindex, cp::PSNRBB(answer, dst, bb), time[lindex].getLapTimeMedian()));
			ci(COLOR_GREEN, format("%d: CV2 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			//ci(format("%d: MC2 PSNR: %5.2f, time: %6.2f", lindex, cp::PSNRBB(answer, dst, bb), time[lindex].getLapTimeMedian()));
			ci(format("%d: CV2 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;
#endif

#endif
#ifdef NEDI
		//dst.setTo(0);
		time[lindex].start();
		nedi.upsample(dstlow, dsthigh, upsample_size, 0, rad * 2, 0);
		time[lindex].getpushLapTime();
		console("NEDI  ", answer, dsthigh, metrics);
#endif

#ifdef JBU
		//dst.setTo(0);
		for (int i = 0; i < iteration; i++)
		{
			time[lindex].start();
			cp::jointBilateralUpsample(dstlow, srchigh, dsthigh, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
			time[lindex].getpushLapTime();
		}
		console("JBU   ", answer, dsthigh, metrics);
#endif

#ifdef GIU
		const float giueps = (guided_eps * 0.0001f * 255.f) * (guided_eps * 0.0001f * 255.f);
		gfcp.setDownsampleMethod(INTER_NEAREST);
		gfcp.setUpsampleMethod(INTER_CUBIC);
		for (int i = 0; i < iteration; i++)
		{
			time[lindex].start();
			gfcp.upsample(dstlow, srclow, srchigh, dsthigh, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
			//giu.upsample64f(low, src, dst, rad, giueps);
			time[lindex].getpushLapTime();
		}
		console("GIU   ", answer, dsthigh, metrics);
#endif

#ifdef BGU
		//bilateralGuidedUpsample(srclow, BGUlow, src, BGUdst, ss, 1.f / sr);
		for (int i = 0; i < iteration; i++)
		{
			time[lindex].start();
			bgu.upsample(srclow, dstlow, srchigh, dsthigh, num_bgu_div_space, num_bgu_bin);
			time[lindex].getpushLapTime();
		}
		console("BGU   ", answer, dsthigh, metrics);
#endif
#ifdef LLU
		locallut.setBoundaryReplicateOffset(llut_rep_offset);
		locallut.setTensorUpCubic(ca * 0.01f - 1.f);
		//dst.setTo(0);
		locallut.setTensorUpSigmaSpace(tsigmas * 0.1f);

		dsthigh.setTo(0);
		time[lindex].start();
		locallut.upsample(srclow, dstlow, srchigh, dsthigh, rad, (int)pow(2, num_lut_bit), llut_lutfilter_rad, LocalLUTUpsample::BUILD_LUT(llut_build_method),
			LocalLUTUpsample::UPTENSOR::NEAREST, (LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		console("LLUNN ", answer, dsthigh, metrics);

		for (int i = 0; i < iteration; i++)
		{
			time[lindex].start();
			locallut.upsample(srclow, dstlow, srchigh, dsthigh, rad, (int)pow(2, num_lut_bit), llut_lutfilter_rad, LocalLUTUpsample::BUILD_LUT(llut_build_method),
				LocalLUTUpsample::UPTENSOR::GAUSS4, (LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
			time[lindex].getpushLapTime();
		}
		console("LLUG4 ", answer, dsthigh, metrics);

		for (int i = 0; i < iteration; i++)
		{
			time[lindex].start();
			/*for (int n = 0; n < 7; n++)
			{
				locallut.upsample(dwnHRHP[n], dwnHRHP[n], srcHRHP[n], dstHRHP[n], rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUTUpsample::BUILD_LUT(llut_build_method), LocalLUTUpsample::UPTENSOR::GAUSS16, (LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
			}*/
			locallut.upsample(srclow, dstlow, srchigh, dsthigh, rad, (int)pow(2, num_lut_bit), llut_lutfilter_rad, LocalLUTUpsample::BUILD_LUT(llut_build_method),
				LocalLUTUpsample::UPTENSOR::GAUSS16, (LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
			time[lindex].getpushLapTime();

		}
		console("LLUG16", answer, dsthigh, metrics);

		for (int i = 0; i < iteration; i++)
		{
			time[lindex].start();
			locallut.upsample(srclow, dstlow, srchigh, dsthigh, rad, (int)pow(2, num_lut_bit), llut_lutfilter_rad, LocalLUTUpsample::BUILD_LUT(llut_build_method),
				LocalLUTUpsample::UPTENSOR::GAUSS64, LocalLUTUpsample::BOUNDARY(llut_boundary_method), llut_use_offsetmap == 1);
			time[lindex].getpushLapTime();
		}
		console("LLUG64", answer, dsthigh, metrics);

#ifdef ILGU64
		for (int i = 0; i < iteration; i++)
		{
			time[lindex].start();
			locallut.upsample(srclow, dstlow, srchigh, dsthigh, rad, (int)pow(2, num_lut_bit), llut_lutfilter_rad, LocalLUTUpsample::BUILD_LUT(llut_build_method), LocalLUTUpsample::UPTENSOR::GAUSS64, (LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
			const int up_iter = 1;
			for (int n = 0; n < up_iter; n++)
			{
				double parameter = sigma_clip;
				if (inter_d == 7) parameter *= 0.5;
				Mat dstlowiter;  cp::downsample(dsthigh, dstlowiter, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				locallut.upsample(dstlowiter, dstlow, dsthigh, dsthigh, rad, (int)pow(2, num_lut_bit), llut_lutfilter_rad, LocalLUTUpsample::BUILD_LUT(llut_build_method), LocalLUTUpsample::UPTENSOR::GAUSS64, (LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
			}
			time[lindex].getpushLapTime();
		}
		console("ILGU64", answer, dsthigh, metrics);
#endif
#endif

#ifdef LLU_ALL
		locallut.setBoundaryReplicateOffset(llut_rep_offset);
		locallut.setTensorUpCubic(ca * 0.01 - 1.0);

		dst.setTo(0);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::NEAREST, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LNN1  PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LNN1  PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::LINEAR, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LLI4  PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LLI4  PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::BOX4, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LBX4  PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LBX4  PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;


		dst.setTo(0);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::GAUSS4, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LGU4  PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LGU4  PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::BOX16, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LBX16 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LBX16 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		locallut.setTensorUpSigmaSpace(tsigmas * 0.1);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::GAUSS16, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LGU16 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LGU16 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::CUBIC, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LCU16 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LCU16 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		locallut.setTensorUpSigmaRange(tsigmar);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::BILATERAL16, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LBL16 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{

			ci(format("%02d: LBL16 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::BOX64, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LBX64 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LBX64 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		locallut.setTensorUpSigmaSpace(tsigmas * 0.1);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::GAUSS64, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LGU64 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LGU64 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		locallut.setTensorUpSigmaRange(tsigmar);
		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::BILATERAL64, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LBL64 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{

			ci(format("%02d: LBL64 PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		dst.setTo(0);
		locallut.setTensorUpKernelSize(2 * tkernelr);

		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::BoxNxN, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LBNxN PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{
			ci(format("%02d: LBNxN PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::GaussNxN, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LGNxN PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{

			ci(format("%02d: LGNxN PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;

		time[lindex].start();
		locallut.upsample(srclow, low, src, dst, rad, (int)pow(2, lnum), llut_lutfilter_rad, LocalLUT::BUILD_LUT(llut_build_method), LocalLUT::UPTENSOR::LaplaceNxN, (LocalLUT::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
		time[lindex].getpushLapTime();
		if (index == lindex)
		{
			dst.copyTo(show);
			ci(COLOR_GREEN, format("%02d: LLNxN PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		else
		{

			ci(format("%02d: LLNxN PSNR: %5.2f, time: %6.2f", lindex, PSNR(answer, dst), time[lindex].getLapTimeMedian()));
		}
		lindex++;
#endif
#pragma endregion

#pragma region show_and_key
		addWeighted(show, 1.0 - alpha * 0.01, answer, alpha * 0.01, 0.0, show);
		imshow(wname, show);
		//Mat viz; hconcat(show, answer, viz); imshow(wname, viz);	
		ci.show();

		key = waitKey(1);
		if (key == 's')
		{
			imwrite("img/source.png", srchigh);
			imwrite("img/processed", dstlow);
		}
		if (key == 'g') cp::guiAlphaBlend(show, answer);
		if (key == 'd') cp::guiDiff(show, answer);
		if (key == 'o')
		{
			llut_use_offsetmap = (llut_use_offsetmap == 1) ? 0 : 1;
			setTrackbarPos("llu:offsetmap", "", llut_use_offsetmap);
		}
		if (key == 'f')
		{
			alpha = (alpha == 0) ? 100 : 0;
			setTrackbarPos("alpha", "", alpha);
		}
		if (key == 'l')
		{
			locallut.guiLUT(srclow, dstlow, srchigh, answer, true, "guiLUT");
			destroyWindow("guiLUT");
		}
		if (key == 'r' || isDataUpdate || llucheck.isUpdate(num_lut_bit, llut_lutfilter_rad, llut_build_method, llut_boundary_method, llut_tensorup_method, llut_use_offsetmap, llut_rep_offset))
		{
			for (auto& e : time)
			{
				e.clearStat();
			}
		}
#pragma endregion
		totalTime.pushLapTime();
	}
}
