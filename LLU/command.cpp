#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

int command(int argc, const char* const argv[])
{
	std::string key =
		"{h help ?      |         | show help command}"
		"{@src_img      |         | source low image}"
		"{@proclow_img  |         | processed low image}"
		"{@dest         | out.png | dest image}"
		"{r radius      | 2       | radius of upsampling}"
		"{n numlut      | 256     | size of LUT in range domain }"
		"{R lut_radius  | 2       | radius of LUT filtering in range domain }"
		"{L build_lut   | 0       | method of building LUT (0:L2, 1:L1, 2: Linf, 3: WTA, 4: DP) }"
		"{U upsample_lut| 6       | method of upsampling LUT (0:NN, 1:Box4, 2: Box16, 3: Box64, 4: Gauss4, 5: Gauss16, 6: Gauss64, 7: Linear, 8: Cubic) }"
		"{B boundary_lut| 3       | method of boundary of LUT (0: replicate, 1: minmax, 2: 0-255, 3: linear, 4: no interpolation) }"
		"{o offset      |         | use offset map }"
		"{d debug       |         | GUI of viewing local LUT ('q' is quit key for GUI) }"
		;
	cv::CommandLineParser parser(argc, argv, key);

	if (parser.has("h"))
	{
		parser.about("local LUT upsampling");

		parser.printMessage();
		cout << "Example: " << endl;
		cout << "LLU source.png processed.png out.png -r=2 -n=256 -R=2 -L=0 -U=6 -B=3 -o -d" << endl;
		return 0;
	}

	LocalLUTUpsample llu;
	Mat srchigh, dstlow, srclow, dsthigh;
	if (parser.has("@src_img"))
	{
		srchigh = imread(parser.get<string>(0));
		if (srchigh.empty())
		{
			cerr << "source file open error: " << parser.get<string>(0) << endl;
		}
	}
	else
	{
		cout << "src_img is not set" << endl;
		return -1;
	}

	if (parser.has("@proclow_img"))
	{
		dstlow = imread(parser.get<string>(1));
		if (dstlow.empty())
		{
			cerr << "destlow file open error: " << parser.get<string>(1) << endl;
		}
	}
	else
	{
		cout << "proclow_img is not set" << endl;
		return -1;
	}

	const int r = parser.get<int>("r");
	const int lut_num = parser.get<int>("n");
	const int R = parser.get<int>("R");
	const int buildLUT = parser.get<int>("L");
	const int boundaryLUT = parser.get<int>("B");
	const int upsampleLUT = parser.get<int>("U");
	const bool useOffset = parser.has("o");

	resize(srchigh, srclow, dstlow.size());
	llu.upsample(srclow, dstlow, srchigh, dsthigh, r, lut_num, R, LocalLUTUpsample::BUILD_LUT(buildLUT), LocalLUTUpsample::UPTENSOR(upsampleLUT), LocalLUTUpsample::BOUNDARY(boundaryLUT), useOffset);

	if (parser.has("d")) llu.guiLUT(srclow, dstlow, srchigh, dsthigh);
	imwrite(parser.get<string>(2), dsthigh);

	return 0;
}