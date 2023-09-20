#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void read_subjective_assessment(bool isIndivisual = false, int waitTime = 1, bool isResizeShow = true, bool isWriteWebP = false)
{
	cp::CSV csv("./img/MTA_subjective_assessment/JND.csv", false);
	csv.readData();
	using namespace std;
	using namespace cv;
	string dir_original = "./img/MTA_subjective_assessment/original/";
	string dir_source = "./img/MTA_subjective_assessment/processed_source/";
	string dir_upsample = "./img/MTA_subjective_assessment/processed_upsample/";

	int count = 0;
	int index = 0;

	vector<Mat> sidebyside;
	auto console = [&](Mat& original, Mat& source, Mat& upsample, double jnd, int index, int d, int u, int l)
		{
			string dist = "";
			if (d == 0)dist = "Bilateral filter";
			if (d == 1)dist = "L0 smooth";
			if (d == 2)dist = "Local Laplacian";
			if (d == 3)dist = "Haze Remove";

			string up = "";
			if (u == -1)up = "CPY";
			if (u == 0)up = "CBU";
			if (u == 1)up = "JBU";
			if (u == 2)up = "GIU";
			if (u == 3)up = "BGU";
			if (u == 4)up = "LLU";
			Mat show;
			const int xst = 20;
			const int yst = 100;
			const int ystep = 85;
			original.copyTo(show);
			Mat mask = Mat::zeros(show.size(), CV_8U);
			Mat back = Mat::zeros(show.size(), CV_8U);
			rectangle(mask, Rect(xst, 20, 750, 700), Scalar(255 / 2, 0, 0, 0), cv::FILLED);
			rectangle(back, Rect(xst, 20, 750, 700), Scalar(0, 0, 0, 200), cv::FILLED);
			cp::alphaBlend(back, show, mask, show);
			int line = 0;
			cv::putText(show, format("index %03d", index), Point(xst, yst + ystep * line), cv::FONT_HERSHEY_SIMPLEX, 3, COLOR_WHITE, 2); line++;
			cv::putText(show, dist, Point(xst, yst + ystep * line), cv::FONT_HERSHEY_SIMPLEX, 3, COLOR_WHITE, 2); line++;
			cv::putText(show, up + format(" x%3d", (int)pow(4, l)), Point(xst, yst + ystep * line), cv::FONT_HERSHEY_SIMPLEX, 3, COLOR_WHITE, 2); line++;
			cv::putText(show, format("JND %5.2f", jnd * 100.0), Point(xst, yst + ystep * line), cv::FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2); line++;
			cv::putText(show, format("different 0 - 100 same"), Point(xst, yst + ystep * line), cv::FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WHITE, 2); line++;
			cv::putText(show, format("PSNR %5.2f dB", cp::getPSNR(source, upsample)), Point(xst, yst + ystep * line), cv::FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2); line++;
			cv::putText(show, format("SSIM %5.3f", cp::getSSIM(source, upsample)), Point(xst, yst + ystep * line), cv::FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2); line++;
			cv::putText(show, format("GMSD %5.3f", cp::getGMSD(source, upsample)), Point(xst, yst + ystep * line), cv::FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2); line++;

			if (isIndivisual)
			{
				imshow("original", original);
				imshow("source", source);
				imshow("upsample", upsample);
				cv::putText(show, format("JND %5.2f", jnd * 100.0), Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 3, COLOR_WHITE, 2);
				cv::putText(show, format("differenct 0 - 100 same"), Point(100, 200), cv::FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2);
				cv::putText(show, format("PSNR %5.2f dB", cp::getPSNR(source, upsample)), Point(100, 300), cv::FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2);
				imshow("show", show);
			}
			else
			{
				hconcat(show, source, show);
				hconcat(show, upsample, show);
				cv::putText(show, "original", Point(1024 * 0 + 350, 950), cv::FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2);
				cv::putText(show, "source", Point(1024 * 1 + 350, 950), cv::FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2);
				cv::putText(show, "upsample", Point(1024 * 2 + 350, 950), cv::FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2);

				if (isResizeShow) resize(show, show, Size(), 0.5, 0.5);
				imshow("show", show);//show with half size
				sidebyside.push_back(show);
			}
			int key = waitKey(waitTime);
			//if (key == 'q') exit(0);
		};

	for (int d = 0; d < 4; d++)
	{
		string dist;
		if (d == 0)dist = "bf";
		if (d == 1)dist = "l0";
		if (d == 2)dist = "ll";
		if (d == 3)dist = "hz";
		for (int i = 0; i < 5; i++)
		{
			string img = cv::format("img%02d", i);
			string mes = dist + "_cpy_" + img + "_res01.png";
			Mat source = imread(dir_source + mes);
			Mat original;
			if (d == 3) original = imread(dir_original + cv::format("img%02d.png", (i < 2) ? i : i + 10));
			else  original = imread(dir_original + cv::format("img%02d.png", i));
			for (int u = 0; u < 5; u++)
			{
				string up;
				if (u == 0)up = "cbu";
				if (u == 1)up = "jbu";
				if (u == 2)up = "giu";
				if (u == 3)up = "bgu";
				if (u == 4)up = "llu";

				for (int l = 0; l < 4; l++)
				{
					string res = cv::format("res%02d.png", int(pow(2, l + 1)));
					string mes = dist + "_" + up + "_" + img + "_" + res;
					Mat upsample = imread(dir_upsample + mes);

					const double jnd = csv.data[index][16];
					console(original, source, upsample, jnd, index, d, u, l + 1);
					index++;
				}
			}
		}
	}
	for (int d = 0; d < 4; d++)
	{
		string dist;
		if (d == 0)dist = "bf";
		if (d == 1)dist = "l0";
		if (d == 2)dist = "ll";
		if (d == 3)dist = "hz";
		for (int i = 0; i < 5; i++)
		{
			string img = cv::format("img%02d", i);
			string mes = dist + "_cpy_" + img + "_res01.png";
			Mat source = imread(dir_source + mes);
			Mat original;
			if (d == 3) original = imread(dir_original + cv::format("img%02d.png", (i < 2) ? i : i + 10));
			else  original = imread(dir_original + cv::format("img%02d.png", i));

			const double jnd = csv.data[index][16];
			console(original, source, source, jnd, index, d, -1, 0);
			index++;
		}
	}

	if (isWriteWebP)
	{
		vector<Mat> webpout;
		for (int i = 0; i < sidebyside.size(); i += 10)
		{
			webpout.push_back(sidebyside[i]);
		}
		vector<int> param(4);
		param[0] = cv::IMWRITE_WEBP_QUALITY;
		param[1] = 45;
		param[2] = IMWRITE_WEBP_TIMEMSPERFRAME;
		param[3] = 600;
		cp::imwriteAnimationWebp("out.webp", webpout, param);
	}
}