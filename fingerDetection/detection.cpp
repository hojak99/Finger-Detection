#include <iostream>
#include <opencv2\opencv.hpp>

cv::Point getHandCenter(cv::Mat &mask, double &radius);
float fingerEnd(float p1, float p2, float t1, float t2, float q1, float q2);

std::vector<cv::Point> saveLine;

int erosion_value = 0;
int const max_erosion = 2;
int erosion_size = 0;
int const ersion_max_size = 21;
int dilation_value = 0;
int dilation_size = 0;
int erosion_type = 0;
int drawX, drawY;

int main()
{
	cv::Mat element;
	cv::Mat frame, tmpImg;
	cv::Mat handImg, mask, mask1;
	cv::Point dst;

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	
	double radius = 5;
	cv::VideoCapture video(0);

	if (!video.isOpened()) return 0;

	cv::namedWindow("change_image", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("original_image", CV_WINDOW_AUTOSIZE);
	cv::createTrackbar("ele_erosion", "original_image", &erosion_value, max_erosion);
	cv::createTrackbar("erosion_size", "original_image", &erosion_size, ersion_max_size);
	cv::createTrackbar("ele_dilation", "original_image", &dilation_value, max_erosion);
	cv::createTrackbar("dilation_size", "original_image", &dilation_size, ersion_max_size);

	while (true)
	{
		video >> tmpImg;

		if (tmpImg.empty()) break;

		if (erosion_value == 0) erosion_type = cv::MORPH_RECT;
		else if (erosion_value == 1) erosion_type = cv::MORPH_CROSS;
		else if (erosion_value == 2) erosion_type = cv::MORPH_ELLIPSE;

		element = cv::getStructuringElement(erosion_type, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));

		//손 색깔을 검출하기 위해서 YCbCr 컬러 모델 사용. 잡음에 강하다고..
		//Y는 휘도 성분, Cb, Cr은 색차 성분
		//피부색이 가지는 Cb,Cr의 영역은 
		//Cb: 77 ~ 127
		//Cr: 133 ~ 173

		cv::cvtColor(tmpImg, handImg, CV_BGR2YCrCb);
		cv::inRange(handImg, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), handImg);

		cv::erode(handImg, handImg, element);
		cv::dilate(handImg, handImg, element);

		mask = handImg.clone();


		cv::findContours(mask, contours, hierarchy, CV_RETR_LIST, CV_CLOCKWISE, cv::Point(0,0));
		
		

		int largestContour = 0;
		
		for (int i = 0; i < contours.size(); i++) {
			if (cv::contourArea(contours[i]) > cv::contourArea(contours[largestContour])) {
				largestContour = i;
			}
		}

		cv::drawContours(tmpImg, contours, largestContour, cv::Scalar(0, 255, 255), 1, 8, std::vector < cv::Vec4i>(), 0, cv::Point());	//contour 그리기

		if (!contours.empty()) {
			std::vector<std::vector<cv::Point>>hull(1);

			cv::convexHull(cv::Mat(contours[largestContour]), hull[0], false);
			cv::drawContours(tmpImg, hull, 0, cv::Scalar(0, 255, 0), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());		//점 잇기

			if (hull[0].size() > 2) {
				
				cv::Rect boundingBox = cv::boundingRect(hull[0]);
				cv::Point center = cv::Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
				
				std::vector<cv::Point> validPoints;

				std::vector<int> hullIndex;
				cv::convexHull(cv::Mat(contours[largestContour]), hullIndex);
				std::vector<cv::Vec4i> convexityDefects;
				
				cv::convexityDefects(cv::Mat(contours[largestContour]), hullIndex, convexityDefects);

				for (int i = 0; i < convexityDefects.size(); i++) {
					cv::Point p1 = contours[largestContour][convexityDefects[i][0]];
					cv::Point p2 = contours[largestContour][convexityDefects[i][1]];
					cv::Point p3 = contours[largestContour][convexityDefects[i][2]];
					//cv::line(tmpImg, p1, p3, cv::Scalar(255, 0, 0), 2);
					//cv::line(tmpImg, p3, p2, cv::Scalar(255, 0, 0), 2);

					double angle = std::atan2(center.y - p1.y, center.x - p1.x) * 180 / CV_PI;
					double inA = fingerEnd(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
					double length = std::sqrt(std::pow(p1.x - p3.x, 2) + std::pow(p1.y - p3.y, 2));
				
					if (angle > -30 && angle < 160 && std::abs(inA) > 20 && std::abs(inA) < 120 && length > 0.1 * boundingBox.height)
					{
						validPoints.push_back(p1);
					}
				}

				int maxY = 0, maxX=0;
				
				if (validPoints.size() != 0) {
					maxY = validPoints[0].y, maxX = validPoints[0].x;

					for (int i = 0; i < validPoints.size(); i++)
					{
						if (validPoints[i].y < maxY) {
							maxY = validPoints[i].y;
							maxX = validPoints[i].x;
						}
						cv::circle(tmpImg, cv::Point(maxX, maxY), 9, cv::Scalar(255, 0, 0), 2);
						
						saveLine.push_back(cv::Point(maxX, maxY));						

						for (std::vector<int>::size_type i = 0; i < saveLine.size(); i++) {
							cv::line(tmpImg, saveLine[i], saveLine[i], cv::Scalar(255, 5, 222), 5);
						}

						if (cv::waitKey(10) == 32) {
							saveLine.clear();
						}
						maxY = 0; maxX = 0;
					}
				}
			}
		}
		
		cv::line(tmpImg, cv::Point(drawX, drawY), cv::Point(drawX, drawY), cv::Scalar(0, 0, 0), 3, 8, 0);

		cv::flip(handImg, handImg, 1);	//영상 반전
		cv::flip(tmpImg, tmpImg, 1);	//영상 반전
		
		cv::imshow("change_image", handImg);
		cv::imshow("original_image", tmpImg);

		if (cv::waitKey(10) == 27) {
			break;
		}
	}


	video.release();
	tmpImg.release();
	frame.release();

	cv::destroyAllWindows();

	return 0;
}

cv::Point getHandCenter(cv::Mat &mask, double &radius)
{
	int maxDst[2];

	cv::Mat dst;	//행렬 저장
	cv::distanceTransform(mask, dst, CV_DIST_L2, 5);

	//최솟값 필요 없음
	cv::minMaxIdx(dst, NULL, &radius, NULL, maxDst, mask);

	return cv::Point(maxDst[1], maxDst[0]);
}

float fingerEnd(float p1, float p2, float t1, float t2, float q1, float q2)
{
	//두점 사이 거리 계산하기 위해. 제곱근 계산하는 sqrt
	float dist1 = std::sqrt((p1 - q1)*(p1 - q1) + (p2 - q2)*(p2 - q2));
	float dist2 = std::sqrt((t1 - q1)*(t1 - q1) + (t2 - q2)*(t2 - q2));

	float aX, aY;
	float bX, bY;
	float cX, cY;

	cX = q1;
	cY = q2;

	if (dist1 < dist2) {
		bX = p1;
		bY = p2;
		aX = t1;
		aY = t2;
	}
	else {
		bX = t1;
		bY = t2;
		aX = p1;
		aY = p2;
	}

	float r1 = cX - aX;
	float r2 = cY - aY;
	float z1 = bX - aX;
	float z2 = bY - aY;

	float result = std::acos((z1*r1 + z1*r2) / (std::sqrt(z1*z1 + z2*z2) * std::sqrt(r1*r1 + r2*r2)));

	result = result * 180 / CV_PI;

	return result;
}