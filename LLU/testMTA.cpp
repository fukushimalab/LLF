#include <jointBilateralUpsample.hpp>
#include <bilateralGuidedUpsample.hpp>
#include <localLUTUpsample.hpp>
#include <nedi.hpp>
#include <spatialfilter/SpatialFilter.hpp>
#include "tools.hpp"

using namespace std;
using namespace cv;
using namespace cp;

#define getIQA getGMSD
//#define getIQA getSSIM
void mtaExperiment()
{
	int ca = 50;
	int sr = 15;
	int ss = 10;
	int rad = 1;
	int guided_eps = 5;

	int num_bin = 8;
	int num_div_space_bgu = 3;

	int tsigmas = 15;
	int tsigmar = 30;
	int tkernelr = 2;
	int llut_use_offsetmap = 1;
	int llut_rep_offset = 0;

	int lnum = 8;
	int llut_lutfilter_rad = 3;
	int llut_build_method = 0; //BUILD_LUT::L2_MIN
	int llut_boundary_method = (int)cp::LocalLUTUpsample::BOUNDARY::LINEAR;
	cp::LocalLUTUpsample::UPTENSOR uptensor = cp::LocalLUTUpsample::UPTENSOR::GAUSS16;
	vector<int> imgend = { 20,20,12,20,12,10,20,10,19 };
	int idx = 1;
	cp::CSV bf2("bf2.csv");
	cp::CSV bf4("bf4.csv");
	cp::CSV bf8("bf8.csv");
	cp::CSV bf16("bf16.csv");
	cp::CSV lz2("lz2.csv");
	cp::CSV lz4("lz4.csv");
	cp::CSV lz8("lz8.csv");
	cp::CSV lz16("lz16.csv");
	cp::CSV ll2("ll2.csv");
	cp::CSV ll4("ll4.csv");
	cp::CSV ll8("ll8.csv");
	cp::CSV ll16("ll16.csv");
	cp::CSV hz2("hz2.csv");
	cp::CSV hz4("hz4.csv");
	cp::CSV hz8("hz8.csv");
	cp::CSV hz16("hz16.csv");
	for (int distIdx = 0; distIdx < 9; distIdx++)
	{
		for (int imageIdx = 0; imageIdx < imgend[distIdx]; imageIdx++)
		{
			cp::Timer t;
			cp::JointBilateralUpsample jbu;
			//GuidedImageUpsample giu;
			///GuidedImageUpsample giu32f;
			cp::BilateralGuidedUpsample bgu;
			cp::GuidedImageFilter gfcp;
			cp::LocalLUTUpsample locallut;
			string dist = "";
			switch (distIdx)
			{
			case 0: dist = "Dehazing_"; break;//20
			case 1: dist = "DetailManipulation_"; break;//20
			case 2: dist = "L0_"; break;//12
			case 3: dist = "LocalLaplacian_"; break;//20
			case 4: dist = "Matting_"; break;//12
			case 5: dist = "Photoshop_"; break;//10
			case 6: dist = "PortraitTransfer_"; break;//20
			case 7: dist = "Recoloring_"; break;//10
			case 8: dist = "StyleTransfer_"; break;//19
			default: break;
			}
			//src = cropMultipleFloor(imread("Transform Recipes/" + dist + format("src%02d.png", imageIdx + 1), 1), 32);
			//answer = cropMultipleFloor(imread("Transform Recipes/" + dist + format("dst%02d.png", imageIdx + 1), 1), 32);
			Mat src = cropMultipleFloor(imread("./data/" + dist + format("%04d_src.png", imageIdx + 1), 1), 128);
			cout << idx++ << "/143| " << src.size() << ": ";
#if 1
			Mat answer_bf = cropMultipleFloor(imread("./data/" + dist + format("%04d_bf.png", imageIdx + 1), 1), 128);
			Mat answer_lz = cropMultipleFloor(imread("./data/" + dist + format("%04d_lz.png", imageIdx + 1), 1), 128);
			Mat answer_ll = cropMultipleFloor(imread("./data/" + dist + format("%04d_ll.png", imageIdx + 1), 1), 128);
			Mat answer_hz = cropMultipleFloor(imread("./data/" + dist + format("%04d_hz.png", imageIdx + 1), 1), 128);

			Mat srclow;
			Mat low_bf;
			Mat low_lz;
			Mat low_ll;
			Mat low_hz;
			double sigma_clip = 1.0;
			double parameter = sigma_clip;
			int p = 0;//10;
			int inter_d = 11;
			//if (inter_d == 7)parameter *= 0.5;

			Mat dst(answer_bf.size(), src.type());
			{
				int upsample_size = 2;
				cp::downsample(src, srclow, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_bf, low_bf, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_lz, low_lz, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_ll, low_ll, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_hz, low_hz, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);

				//Cubic
				cp::upsampleCubic(low_bf, dst, upsample_size, -ca * 0.01);
				bf2.write(getIQA(answer_bf, dst));
				cp::upsampleCubic(low_lz, dst, upsample_size, -ca * 0.01);
				lz2.write(getIQA(answer_lz, dst));
				cp::upsampleCubic(low_ll, dst, upsample_size, -ca * 0.01);
				ll2.write(getIQA(answer_ll, dst));
				cp::upsampleCubic(low_hz, dst, upsample_size, -ca * 0.01);
				hz2.write(getIQA(answer_hz, dst));

				//JBU
				cp::jointBilateralUpsample(low_bf, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				bf2.write(getIQA(answer_bf, dst));
				cp::jointBilateralUpsample(low_lz, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				lz2.write(getIQA(answer_lz, dst));
				cp::jointBilateralUpsample(low_ll, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				ll2.write(getIQA(answer_ll, dst));
				cp::jointBilateralUpsample(low_hz, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				hz2.write(getIQA(answer_hz, dst));

				//GIU
				const float giueps = (guided_eps * 0.0001 * 255.0) * (guided_eps * 0.0001 * 255.0);
				gfcp.setDownsampleMethod(INTER_NEAREST);
				gfcp.setUpsampleMethod(INTER_CUBIC);
				//
				gfcp.upsample(low_bf, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				bf2.write(getIQA(answer_bf, dst));
				gfcp.upsample(low_lz, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				lz2.write(getIQA(answer_lz, dst));
				gfcp.upsample(low_ll, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				ll2.write(getIQA(answer_ll, dst));
				gfcp.upsample(low_hz, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				hz2.write(getIQA(answer_hz, dst));

				//BGU
				bgu.upsample(srclow, low_bf, src, dst, num_div_space_bgu, num_bin);
				bf2.write(getIQA(answer_bf, dst));
				bgu.upsample(srclow, low_lz, src, dst, num_div_space_bgu, num_bin);
				lz2.write(getIQA(answer_lz, dst));
				bgu.upsample(srclow, low_ll, src, dst, num_div_space_bgu, num_bin);
				ll2.write(getIQA(answer_ll, dst));
				bgu.upsample(srclow, low_hz, src, dst, num_div_space_bgu, num_bin);
				hz2.write(getIQA(answer_hz, dst));

				//LLU
				locallut.setBoundaryReplicateOffset(llut_rep_offset);
				locallut.setTensorUpCubic(ca * 0.01 - 1.0);
				locallut.setTensorUpSigmaSpace(tsigmas * 0.1);
				locallut.upsample(srclow, low_bf, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				bf2.write(getIQA(answer_bf, dst));
				locallut.upsample(srclow, low_lz, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				lz2.write(getIQA(answer_lz, dst));
				locallut.upsample(srclow, low_ll, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				ll2.write(getIQA(answer_ll, dst));
				locallut.upsample(srclow, low_hz, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				hz2.write(getIQA(answer_hz, dst));

				bf2.end();
				lz2.end();
				ll2.end();
				hz2.end();
			}
			{
				int upsample_size = 4;
				cp::downsample(src, srclow, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_bf, low_bf, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_lz, low_lz, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_ll, low_ll, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_hz, low_hz, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);

				cp::upsampleCubic(low_bf, dst, upsample_size, -ca * 0.01);
				bf4.write(getIQA(answer_bf, dst));
				cp::upsampleCubic(low_lz, dst, upsample_size, -ca * 0.01);
				lz4.write(getIQA(answer_lz, dst));
				cp::upsampleCubic(low_ll, dst, upsample_size, -ca * 0.01);
				ll4.write(getIQA(answer_ll, dst));
				cp::upsampleCubic(low_hz, dst, upsample_size, -ca * 0.01);
				hz4.write(getIQA(answer_hz, dst));

				cp::jointBilateralUpsample(low_bf, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				bf4.write(getIQA(answer_bf, dst));
				cp::jointBilateralUpsample(low_lz, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				lz4.write(getIQA(answer_lz, dst));
				cp::jointBilateralUpsample(low_ll, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				ll4.write(getIQA(answer_ll, dst));
				cp::jointBilateralUpsample(low_hz, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				hz4.write(getIQA(answer_hz, dst));

				const float giueps = (guided_eps * 0.0001 * 255.0) * (guided_eps * 0.0001 * 255.0);
				gfcp.setDownsampleMethod(INTER_NEAREST);
				gfcp.setUpsampleMethod(INTER_CUBIC);
				//
				gfcp.upsample(low_bf, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				bf4.write(getIQA(answer_bf, dst));
				gfcp.upsample(low_lz, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				lz4.write(getIQA(answer_lz, dst));
				gfcp.upsample(low_ll, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				ll4.write(getIQA(answer_ll, dst));
				gfcp.upsample(low_hz, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				hz4.write(getIQA(answer_hz, dst));

				bgu.upsample(srclow, low_bf, src, dst, num_div_space_bgu, num_bin);
				bf4.write(getIQA(answer_bf, dst));
				bgu.upsample(srclow, low_lz, src, dst, num_div_space_bgu, num_bin);
				lz4.write(getIQA(answer_lz, dst));
				bgu.upsample(srclow, low_ll, src, dst, num_div_space_bgu, num_bin);
				ll4.write(getIQA(answer_ll, dst));
				bgu.upsample(srclow, low_hz, src, dst, num_div_space_bgu, num_bin);
				hz4.write(getIQA(answer_hz, dst));

				locallut.setBoundaryReplicateOffset(llut_rep_offset);
				locallut.setTensorUpCubic(ca * 0.01 - 1.0);
				locallut.setTensorUpSigmaSpace(tsigmas * 0.1);
				locallut.upsample(srclow, low_bf, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				bf4.write(getIQA(answer_bf, dst));
				locallut.upsample(srclow, low_lz, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				lz4.write(getIQA(answer_lz, dst));
				locallut.upsample(srclow, low_ll, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				ll4.write(getIQA(answer_ll, dst));
				locallut.upsample(srclow, low_hz, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				hz4.write(getIQA(answer_hz, dst));

				bf4.end();
				lz4.end();
				ll4.end();
				hz4.end();
			}

			{
				int upsample_size = 8;
				cp::downsample(src, srclow, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_bf, low_bf, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_lz, low_lz, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_ll, low_ll, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_hz, low_hz, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);

				cp::upsampleCubic(low_bf, dst, upsample_size, -ca * 0.01);
				bf8.write(getIQA(answer_bf, dst));
				cp::upsampleCubic(low_lz, dst, upsample_size, -ca * 0.01);
				lz8.write(getIQA(answer_lz, dst));
				cp::upsampleCubic(low_ll, dst, upsample_size, -ca * 0.01);
				ll8.write(getIQA(answer_ll, dst));
				cp::upsampleCubic(low_hz, dst, upsample_size, -ca * 0.01);
				hz8.write(getIQA(answer_hz, dst));

				cp::jointBilateralUpsample(low_bf, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				bf8.write(getIQA(answer_bf, dst));
				cp::jointBilateralUpsample(low_lz, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				lz8.write(getIQA(answer_lz, dst));
				cp::jointBilateralUpsample(low_ll, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				ll8.write(getIQA(answer_ll, dst));
				cp::jointBilateralUpsample(low_hz, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				hz8.write(getIQA(answer_hz, dst));

				const float giueps = (guided_eps * 0.0001 * 255.0) * (guided_eps * 0.0001 * 255.0);
				gfcp.setDownsampleMethod(INTER_NEAREST);
				gfcp.setUpsampleMethod(INTER_CUBIC);
				//
				gfcp.upsample(low_bf, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				bf8.write(getIQA(answer_bf, dst));
				gfcp.upsample(low_lz, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				lz8.write(getIQA(answer_lz, dst));
				gfcp.upsample(low_ll, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				ll8.write(getIQA(answer_ll, dst));
				gfcp.upsample(low_hz, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				hz8.write(getIQA(answer_hz, dst));

				bgu.upsample(srclow, low_bf, src, dst, num_div_space_bgu, num_bin);
				bf8.write(getIQA(answer_bf, dst));
				bgu.upsample(srclow, low_lz, src, dst, num_div_space_bgu, num_bin);
				lz8.write(getIQA(answer_lz, dst));
				bgu.upsample(srclow, low_ll, src, dst, num_div_space_bgu, num_bin);
				ll8.write(getIQA(answer_ll, dst));
				bgu.upsample(srclow, low_hz, src, dst, num_div_space_bgu, num_bin);
				hz8.write(getIQA(answer_hz, dst));

				locallut.setBoundaryReplicateOffset(llut_rep_offset);
				locallut.setTensorUpCubic(ca * 0.01 - 1.0);
				locallut.setTensorUpSigmaSpace(tsigmas * 0.1);
				locallut.upsample(srclow, low_bf, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				bf8.write(getIQA(answer_bf, dst));
				locallut.upsample(srclow, low_lz, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				lz8.write(getIQA(answer_lz, dst));
				locallut.upsample(srclow, low_ll, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				ll8.write(getIQA(answer_ll, dst));
				locallut.upsample(srclow, low_hz, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				hz8.write(getIQA(answer_hz, dst));

				bf8.end();
				lz8.end();
				ll8.end();
				hz8.end();
			}

			{
				int upsample_size = 16;
				cp::downsample(src, srclow, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_bf, low_bf, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_lz, low_lz, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_ll, low_ll, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);
				cp::downsample(answer_hz, low_hz, upsample_size, cp::Downsample(inter_d), parameter, p * 0.1);

				cp::upsampleCubic(low_bf, dst, upsample_size, -ca * 0.01);
				bf16.write(getIQA(answer_bf, dst));

				cp::upsampleCubic(low_lz, dst, upsample_size, -ca * 0.01);
				lz16.write(getIQA(answer_lz, dst));
				cp::upsampleCubic(low_ll, dst, upsample_size, -ca * 0.01);
				ll16.write(getIQA(answer_ll, dst));
				cp::upsampleCubic(low_hz, dst, upsample_size, -ca * 0.01);
				hz16.write(getIQA(answer_hz, dst));

				cp::jointBilateralUpsample(low_bf, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				bf16.write(getIQA(answer_bf, dst));
				cp::jointBilateralUpsample(low_lz, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				lz16.write(getIQA(answer_lz, dst));
				cp::jointBilateralUpsample(low_ll, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				ll16.write(getIQA(answer_ll, dst));
				cp::jointBilateralUpsample(low_hz, src, dst, rad, sr, ss, cp::JBUSchedule::ALLOC_BORDER_OMP);
				hz16.write(getIQA(answer_hz, dst));

				const float giueps = (guided_eps * 0.0001 * 255.0) * (guided_eps * 0.0001 * 255.0);
				gfcp.setDownsampleMethod(INTER_NEAREST);
				gfcp.setUpsampleMethod(INTER_CUBIC);
				//
				gfcp.upsample(low_bf, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				bf16.write(getIQA(answer_bf, dst));
				gfcp.upsample(low_lz, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				lz16.write(getIQA(answer_lz, dst));
				gfcp.upsample(low_ll, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				ll16.write(getIQA(answer_ll, dst));
				gfcp.upsample(low_hz, srclow, src, dst, rad, giueps, 5, 1);////ガイドのダウンサンプルは入力する
				hz16.write(getIQA(answer_hz, dst));

				bgu.upsample(srclow, low_bf, src, dst, num_div_space_bgu, num_bin);
				bf16.write(getIQA(answer_bf, dst));
				bgu.upsample(srclow, low_lz, src, dst, num_div_space_bgu, num_bin);
				lz16.write(getIQA(answer_lz, dst));
				bgu.upsample(srclow, low_ll, src, dst, num_div_space_bgu, num_bin);
				ll16.write(getIQA(answer_ll, dst));
				bgu.upsample(srclow, low_hz, src, dst, num_div_space_bgu, num_bin);
				hz16.write(getIQA(answer_hz, dst));

				locallut.setBoundaryReplicateOffset(llut_rep_offset);
				locallut.setTensorUpCubic(ca * 0.01 - 1.0);
				locallut.setTensorUpSigmaSpace(tsigmas * 0.1);
				locallut.upsample(srclow, low_bf, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				bf16.write(getIQA(answer_bf, dst));
				locallut.upsample(srclow, low_lz, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				lz16.write(getIQA(answer_lz, dst));
				locallut.upsample(srclow, low_ll, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				ll16.write(getIQA(answer_ll, dst));
				locallut.upsample(srclow, low_hz, src, dst, rad, pow(2, lnum), llut_lutfilter_rad, cp::LocalLUTUpsample::BUILD_LUT(llut_build_method), uptensor, (cp::LocalLUTUpsample::BOUNDARY)llut_boundary_method, llut_use_offsetmap == 1);
				hz16.write(getIQA(answer_hz, dst));

				bf16.end();
				lz16.end();
				ll16.end();
				hz16.end();
			}
#endif
		}
	}
}