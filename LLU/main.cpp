#include <opencp.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencpd.lib")
#else
#pragma comment(lib, "opencp.lib")
#endif

void testGUI();
void mtaExperiment();
int command(int argc, const char* const argv[]);
void read_subjective_assessment(bool isIndivisual = false, int waitTime = 1, bool isResizeShow = true, bool isWriteWebP = false);

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
	cp::LocalLUTUpsample::BUILD_LUT buildLUT =   cp::LocalLUTUpsample::BUILD_LUT::L2_MIN;//building LUT method
	cp::LocalLUTUpsample::UPTENSOR upsampleLUT = cp::LocalLUTUpsample::UPTENSOR::GAUSS64;//tensor upsampling method
	cp::LocalLUTUpsample::BOUNDARY boundaryLUT = cp::LocalLUTUpsample::BOUNDARY::LINEAR;//boundary condition of LUT
	bool useOffset = true;//with/without offset map
	//run
	cv::resize(srchigh, srclow, dstlow.size());//down sample high resolution source image
	llu.upsample(srclow, dstlow, srchigh, dsthigh, r, lut_num, R, buildLUT, upsampleLUT, boundaryLUT, useOffset);//body
	//show
	cv::imshow("out", dsthigh);
	llu.guiLUT(srclow, dstlow, srchigh, dsthigh);
	cv::waitKey();
}

int main(int argc, char** argv)
{
	//testSimple();//simple test
	//return command(argc, argv);//command line tool
	testGUI(); return 0; //interactive test
	//read_subjective_assessment(); return 0;
	//mtaExperiment(); //test of MTA paper
}