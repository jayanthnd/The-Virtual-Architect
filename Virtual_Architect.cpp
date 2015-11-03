//3-28-15
//Integration of hsv image conversion, hough line transform to detect a single wall
// Image overlay with scaling and placing.

//jayanth_hough.cpp has the hough line transform
//tanuj_overlay.cpp has the image overlay

//The virtual architect v1.1 working Fine!
//v1.1 features:
// Automatically detects wall edges
// Freeze frame
// Color selection from color chart.
// Only colors the wall which is selected.
// Place objects on the wall. Move and zoom in/out of the objects.

// 4th April 2015.
//v1.2 try.
// Features to be added:
// Multi wall color.
// Multi object placement
// Grab cut feature.


#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"

#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define HUEMAX 179
#define SATMAX 255
#define VALMAX 255

#define erosion_type1	MORPH_RECT
#define erosion_type2	MORPH_CROSS
#define erosion_type3	MORPH_ELLIPSE
#define erosion_size	0

Mat HSV;
int H =170;
int S=200;
int V =200;
int R=0;
int G=0;
int B=0;

int MAX_H=179;
int MAX_S=255;
int MAX_V=255;
int mouse_x=0;
int mouse_y=0;
char window_name[20] = "HSV Color Plot";

//Global variable for hsv color wheel plot
int max_hue_range=179;
int max_step=3; //nuber of pixel for each hue color
int wheel_width=max_hue_range*max_step;
int wheel_hight=50;
int wheel_x=50; //x-position of wheel
int wheel_y=5;//y-position of wheel

//Global variable plot for satuarion-value plot
int S_V_Width=MAX_S;
int S_V_Height=MAX_S;
int S_V_x=10;
int S_V_y=wheel_y+wheel_hight+20;

//Global variable for HSV ploat
int HSV_Width=150;
int HSV_Height=150;
int HSV_x=S_V_x+S_V_Width+30;
int HSV_y=S_V_y+50;

Mat hsvImage, bgrImage, dupBGR;
Mat image;
Mat hsvInRange, hsvTemp;
Vec3b bgr, hsv;
Mat element;
int upper = 5, lower = 5, offset_max = 180;

int newH, newS, newV;

void onTrackbar_changed(int, void*);
static void onMouse( int event, int x, int y, int, void* );
void drawPointers(void);
void splitNColor(void);
void floodFillImage(int x, int y);
//--------------------------------------------------------------------------------

vector<int> hueVector;
vector<Mat> hsvSplit;

void on_trackbar(int thresh_slider, void *)
{
	upper = thresh_slider;
	printf("max = %d\n", upper);
}

void on_trackbar2(int thresh_slider, void *)
{
	lower = thresh_slider;
	printf("min = %d\n", lower);
}

void onTrackbar_changed(int, void*){

//Plot color wheel.
int hue_range=0;
int step=1;
for(int i=wheel_y;i<wheel_hight+wheel_y;i++){
    hue_range=0;
    for(int j=wheel_x;j<wheel_width+wheel_x;j++){
   if(hue_range>=max_hue_range) hue_range=0;
       if(step++==max_step){
           hue_range++;
           step=1;
       }
        Vec3b pix;
        pix.val[0]=hue_range;
        pix.val[1]=255;
        pix.val[2]=255;


     HSV.at<Vec3b>(i,j)=pix;

    }

}


//Plot for saturation and value
int sat_range=0;
int value_range=255;
for(int i=S_V_y;i<S_V_Height+S_V_y;i++){
    value_range--;
    sat_range=0;
for(int j=S_V_x;j<S_V_Width+S_V_x;j++){
        Vec3b pix;
        pix.val[0]=H;
        pix.val[1]=sat_range++;
        pix.val[2]=value_range;
     HSV.at<Vec3b>(i,j)=pix;

    }

}

//Ploat for HSV
Mat roi1(HSV,Rect(HSV_x,HSV_y,HSV_Width,HSV_Height));
roi1=Scalar(H,S,V);
drawPointers();

Mat RGB;
cvtColor(HSV, RGB,CV_HSV2BGR);

imshow(window_name,RGB);
imwrite("hsv.jpg",RGB);

}

static void onMouse( int event, int x, int y, int f, void* ){
if(f&CV_EVENT_FLAG_LBUTTON){
        mouse_x=x;
        mouse_y=y;
    if(((wheel_x<=x)&&(x<=wheel_x+wheel_width))&&((wheel_y<=y)&&(y<=wheel_y+wheel_hight))){
        H=(x-wheel_x)/ max_step;
        newH = H;
        cvSetTrackbarPos("Hue", window_name, H);
        }
    else if(((S_V_x<=x)&&(x<=S_V_x+S_V_Width))&&((S_V_y<=y)&&(y<=S_V_y+S_V_Height))){

        S=x-S_V_x;
        y=y-S_V_y;
        V=255-y;
        
        newS = S;
        newV = V;

        cvSetTrackbarPos("Saturation", window_name, S);
        cvSetTrackbarPos("Value", window_name, V);
		}

}

}

void drawPointers(){
   // Point p(S_V_x+S,S_V_y+(255-V));
    Point p(S,255-V);



    int index=10;
    Point p1,p2;
    p1.x=p.x-index;
    p1.y=p.y;
    p2.x=p.x+index;
    p2.y=p.y;

    Mat roi1(HSV,Rect(S_V_x,S_V_y,S_V_Width,S_V_Height));
    line(roi1, p1, p2,Scalar(255,255,255),1,CV_AA,0);
    p1.x=p.x;
    p1.y=p.y-index;
    p2.x=p.x;
    p2.y=p.y+index;
    line(roi1, p1, p2,Scalar(255,255,255),1,CV_AA,0);

    int x_index=wheel_x+H*max_step;
    if(x_index>=wheel_x+wheel_width) x_index=wheel_x+wheel_width-2;
    if(x_index<=wheel_x) x_index=wheel_x+2;

    p1.x=x_index;
    p1.y=wheel_y+1;
    p2.x=x_index;
    p2.y=wheel_y+20;
    line(HSV, p1, p2,Scalar(255,255,255),2,CV_AA,0);

    Mat RGB(1,1,CV_8UC3);
    Mat temp;
    RGB=Scalar(H,S,V);
    cvtColor(RGB, temp,CV_HSV2BGR);
    Vec3b rgb=temp.at<Vec3b>(0,0);
    B=rgb.val[0];
    G=rgb.val[1];
    R=rgb.val[2];

    Mat roi2(HSV,Rect(450,130,175,175));
    roi2=Scalar(200,0,200);

    char name[30];
    sprintf(name,"R=%d",R);
    putText(HSV,name, Point(460,155) , FONT_HERSHEY_SIMPLEX, .7, Scalar(5,255,255), 2,8,false );

    sprintf(name,"G=%d",G);
    putText(HSV,name, Point(460,180) , FONT_HERSHEY_SIMPLEX, .7, Scalar(5,255,255), 2,8,false );

    sprintf(name,"B=%d",B);
    putText(HSV,name, Point(460,205) , FONT_HERSHEY_SIMPLEX, .7, Scalar(5,255,255), 2,8,false );


    sprintf(name,"H=%d",H);
    putText(HSV,name, Point(545,155) , FONT_HERSHEY_SIMPLEX, .7, Scalar(5,255,255), 2,8,false );

    sprintf(name,"S=%d",S);
    putText(HSV,name, Point(545,180) , FONT_HERSHEY_SIMPLEX, .7, Scalar(5,255,255), 2,8,false );

    sprintf(name,"V=%d",V);
    putText(HSV,name, Point(545,205) , FONT_HERSHEY_SIMPLEX, .7, Scalar(5,255,255), 2,8,false );


}

//----------------------------------Jayanth Hough Begins --------------------------------------------------------
int threshold_value = 141;
int theta_value = 70;

int canny_threshold_value_1 = 34;
int canny_threshold_value_2 = 33;
const int rho_value = 1;
int min_line_length = 50;
int max_line_gap = 10;
static bool calibrated = false;
static int x_current;
static int y_current;

void on_trackbar_hough_thresh(int thresh_slider, void *)
{
    threshold_value = thresh_slider;
//    printf("threshold = %d\n", threshold_value);
}

void on_trackbar_min_length(int thresh_slider, void *)
{
    min_line_length = thresh_slider;
//    printf("Min Length = %d\n", min_line_length);
}

void on_trackbar_max_gap(int thresh_slider, void *)
{
    max_line_gap = thresh_slider;
//    printf("Max Line Gap = %d\n", max_line_gap);
}

/*
void on_trackbar_hough_rho(int rho_slider, void *)
{
    rho_value = rho_slider;
//    printf("rho = %d\n", rho_slider);
}
*/
void on_trackbar_canny_thresh_1(int thresh_slider_1, void *)
{
    canny_threshold_value_1 = thresh_slider_1;
//    printf("threshold 1 = %d\n", thresh_slider_1);
}

void on_trackbar_canny_thresh_2(int thresh_slider_2, void *)
{
    canny_threshold_value_2 = thresh_slider_2;
//    printf("threshold 2 = %d\n", thresh_slider_2);
}

void on_trackbar_hough_theta(int theta_slider, void *)
{
    theta_value = theta_slider;
//    printf("theta = %d\n", theta_value);
}

Mat src_copy;
//This function should be called when the frame is frozen. Call from main()
void jayanth_hough(Mat src)
{
	Mat dst, cdst, ccdst, andImage;
    Mat src_copy_2;
    Mat hsv;
    vector<Mat> channels;
    
    Mat gray_image, gray_image_2;
    
        Mat src_copy_3;
        src.copyTo(src_copy_2);
        src.copyTo(src_copy);
        src.copyTo(src_copy_3);
        
        cvtColor(src, gray_image, CV_RGB2GRAY);
        blur(gray_image, gray_image_2, Size(3,3));
        Canny(gray_image, dst, canny_threshold_value_1, canny_threshold_value_2);

        vector<Vec2f> lines;
        vector<Vec4i> lines_p;
        vector<Point> rectangle_tl;
        vector<int> distances;
        
        HoughLines(dst, lines, rho_value, CV_PI/180, threshold_value, 0, 0);
        HoughLinesP(dst, lines_p, rho_value, CV_PI/180, threshold_value, min_line_length, max_line_gap );     //50,10
        
        // For contours
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        RNG rng(12345);
        
        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( src_copy, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
            line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
        }
 
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines_p[i];
            line( src_copy_2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
         }
 
        
        
        //cvtColor(src_copy, src_copy, CV_RGB2GRAY);
        blur(src_copy, src_copy, Size(3,3));
        Canny(src_copy, andImage, canny_threshold_value_1, canny_threshold_value_2, 3);
        
        // Find contours
        findContours( andImage, contours, hierarchy, CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        
        /// Approximate contours to polygons + get bounding rects and circles
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        vector<Point2f>center( contours.size() );
        vector<float>radius( contours.size() );
        
        for( int i = 0; i < contours.size(); i++ )
        { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        }
        
        /// Draw polygonal contour + bonding rects + circles
        Mat drawing = Mat::zeros( andImage.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
            rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
            //Point point = boundRect[i].tl();
            //cout<<boundRect[i].tl()<<endl;

        }
		//imshow("Hough Source", src_copy);	
		imshow("NEW image", src_copy);
		src_copy_2.release();
      
#if 0        
        if(calibrated)
        {
            cout<<"Contours: "<<contours.size()<<endl;
            for (int i=0; i<contours.size(); i++) {
                rectangle_tl.push_back(boundRect[i].tl());
            }
            cout<<"Rectangle: "<<rectangle_tl.size()<<endl;
            for (int i=0; i<rectangle_tl.size(); i++) {
                Point point_tl = boundRect[i].tl();
                Point point_br = boundRect[i].br();
                int x_ = point_tl.x;
                int y_ = point_tl.y;
                int x_br = point_br.x;
                int y_br = point_br.y;
                
                if(x_current < x_br && x_current > x_ && y_current < y_br && x_current > y_)
                {
                    int distance = sqrt((y_-y_current)*(y_-y_current) - (x_-x_current)*(x_-x_current));
                    
                    cout<<"Distance = "<<distance<<endl;
                    cout<<"TL = "<<point_tl<<"----- BR = "<<point_br<<endl;
                    
                    distances.push_back(distance);
                    
                    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                    
                    rectangle(src_copy_3 , point_tl, point_br, color, 2, 8, 0 );
                
                }

            }
            imshow("Result", src_copy_3);
            src_copy_3.release();
          }
#endif        
}
//----------------------------------Jayanth hough ends ----------------------------------------------------------

//----------------------------------Tanuj overlay begins---------------------------------------------------------
int xCor=0,yCor=0;
int xScale=100,yScale=100;
Mat foreground_old;
Mat foreground_new, result;
int key=0;

void overlayImage(const cv::Mat &background, const cv::Mat &foreground,
                  cv::Mat &output, cv::Point2i location)
{
    background.copyTo(output);
    
    // start at the row indicated by location, or at row 0 if location.y is negative.
    for(int y = std::max(location.y , 0); y < background.rows; ++y)
    {
        int fY = y - location.y; // because of the translation
        
        // we are done of we have processed all rows of the foreground image.
        if(fY >= foreground.rows)
            break;
        
        // start at the column indicated by location,
        
        // or at column 0 if location.x is negative.
        for(int x = std::max(location.x, 0); x < background.cols; ++x)
        {
            int fX = x - location.x; // because of the translation.
            
            // we are done with this row if the column is outside of the foreground image.
            if(fX >= foreground.cols)
                break;
            
            // determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
            double opacity =
            ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])
            
            / 255.;
            
            
            // and now combine the background and foreground pixel, using the opacity,
            
            // but only if opacity > 0.
            for(int c = 0; opacity > 0 && c < output.channels(); ++c)
            {
                unsigned char foregroundPx =
                foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx =
                background.data[y * background.step + x * background.channels() + c];
                output.data[y*output.step + output.channels()*x + c] =
                backgroundPx * (1.-opacity) + foregroundPx * opacity;
            }
        }
    }
}
string filePath = "/home/ubuntu/opencv-2.4.9/samples/cpp/dhaval_programs/demo_images/";
void overlayObject(int k)
{
	switch((char)k)
	{
		case '1':	//gf_clock
			foreground_old = imread(filePath+"ac.png", -1);
			key=0;
			break;
			
		case '2':
			foreground_old = imread(filePath+"lamp2.png", -1);
			key=0;
			break;
			
		case '3':
			foreground_old = imread(filePath+"img_couch.png", -1);
			key=0;
			break;
			
		case '4':
			foreground_old = imread(filePath+"table_lamp.png", -1);
			key=0;
			break;
			
		case '5':
			foreground_old = imread(filePath+"coffee_table.png", -1);
			key=0;
			break;
			
		case '6':
			foreground_old = imread(filePath+"img_chair.png", -1);
			key=0;
			break;
			
		case '7':
			foreground_old = imread(filePath+"table_2.png", -1);
			key=0;
			break;
			
		case '8':
			system("cd /home/ubuntu/NVIDIA_CUDA-6.0_Samples/7_CUDALibraries/grabcutNPP/ && ./grabcutNPP input /home/ubuntu/opencv-2.4.9/samples/cpp/dhaval_programs/demo_images/VA_chair_1.jpg");
			cout<<"Grab cut complete"<<endl;
			foreground_old = imread("Result.png", -1);
			key=0;
			break;
			
		case '9':
			system("cd /home/ubuntu/NVIDIA_CUDA-6.0_Samples/7_CUDALibraries/grabcutNPP/ && ./grabcutNPP input /home/ubuntu/opencv-2.4.9/samples/cpp/dhaval_programs/demo_images/grabcut_1.jpg");
			cout<<"Grab cut complete"<<endl;
			foreground_old = imread("Result.png", -1);
			key=0;
			break;
			
		case '0':
			system("cd /home/ubuntu/NVIDIA_CUDA-6.0_Samples/7_CUDALibraries/grabcutNPP/ && ./grabcutNPP input /home/ubuntu/opencv-2.4.9/samples/cpp/dhaval_programs/demo_images/grabcut_2.jpg");
			cout<<"Grab cut complete"<<endl;
			foreground_old = imread("Result.png", -1);
			key=0;
			break;
	}
	
	resize(foreground_old, foreground_new ,Size(xScale, yScale));
	overlayImage(dupBGR, foreground_new, result, cv::Point(xCor,yCor));
	imshow("NEW image", result);
}

//----------------------------------Tanuj overlay ends-----------------------------------------------------------

//----------------------------------Dhaval HSV begins-------------------------------------------------------
Mat hsvOutput;
Mat mask;
int h, s ,v;
int loDiff = 20;
int upDiff = 20;
int connectivity = 8;
int newMaskVal = 255;
bool enableHough = false;
bool videoMode = true;
bool onlyOnStart = true;
void onClick(int event, int x, int y, int flags, void* param)
{
	
	if( event == CV_EVENT_LBUTTONDOWN )
    {
		 x_current = x;
		 y_current = y;
		 
//		 cout<<" Y = "<<y<<", X = "<<x<<endl;
		 hsv = hsvOutput.at<Vec3b>(y,x);
//		 cout<<"H = "<<(int)hsv.val[0]<<", S = "<<(int)hsv.val[1]<<", V = "<<(int)hsv.val[2]<<endl;  
		 
		 h = (int)hsv.val[0];
		 s = (int)hsv.val[1];
		 v = (int)hsv.val[2];
		 
		 if(enableHough)
		 {
			enableHough = false;
			jayanth_hough(dupBGR);
		 }
		 else if(videoMode)
			splitNColor();
		 else
			floodFillImage(x,y);
	}
	else if( event == CV_EVENT_MOUSEMOVE)
    {
        Point pt = Point(x,y);
        xCor=pt.x;
        yCor=pt.y;
    }
    else if(event == CV_EVENT_MBUTTONDOWN)
    {
        if(xScale<=500 && yScale<=500)
        {
            xScale+=50;
            yScale+=50;

        }
    }
    else if(event== CV_EVENT_RBUTTONDOWN)
    {
        if(xScale>=50 && yScale>=50)
        {
            xScale-=50;
            yScale-=50;
            
        }
    }
}

void floodFillImage(int x, int y)
{
	Point seed = Point(x,y);
	int lo = loDiff;
	int up = upDiff;
	int flags = connectivity + (newMaskVal<<8) + CV_FLOODFILL_FIXED_RANGE;
	int b = 150;
	int g = 150;
	int r = 150;
	Rect ccomp;
	
	//Mat dst = bgrImage;
	Mat dst = src_copy;
	
	Scalar newVal = Scalar(b,g,r);
	//mask.create(bgrImage.rows + 2, bgrImage.cols + 2, CV_8UC1);
	mask.setTo(Scalar(0,0,0));
	
	threshold(mask, mask, 1, 128, CV_THRESH_BINARY);
	floodFill(dst, mask, seed, newVal, &ccomp, Scalar(lo, lo, lo),
                  Scalar(up, up, up), flags);
    
    GaussianBlur(mask, mask, Size(5,5), 0, 0, BORDER_DEFAULT);
    Mat mask_new(mask.size(), mask.type());
    mask.copyTo(mask_new);
    resize(mask, mask_new, Size(dst.rows, dst.cols), 0, 0, INTER_LINEAR);
    
//    imshow("Mask image", mask_new);
    
    split(hsvImage, hsvSplit);
    hsvSplit[0].setTo(Scalar(newH,0,0), mask_new);	//change only the H channel
	hsvSplit[1].setTo(Scalar(newS,0,0), mask_new);	//change only the S channel
        
    merge(hsvSplit, hsvTemp);
    cvtColor(hsvTemp, dupBGR, CV_HSV2BGR);
    imshow("NEW image", dupBGR);
}

void splitNColor(void)
{
	split(hsvImage, hsvSplit);
	
    Scalar sc = Scalar(h,s,v);
    Scalar from = Scalar(h-lower,0,0);		//lower = 1
    Scalar to = Scalar(h+upper,255,255);	//+5 for green cushions, +29 for walls
    
    inRange(hsvImage, from, to,  hsvInRange);
    
    hsvSplit[0].setTo(Scalar(newH,0,0), hsvInRange);	//change only the H channel
	hsvSplit[1].setTo(Scalar(newS,0,0), hsvInRange);	//change only the S channel
        
    merge(hsvSplit, hsvTemp);
    cvtColor(hsvTemp, dupBGR, CV_HSV2BGR);
    imshow("NEW image", dupBGR);
	
}

void help()
{
	cout<<"*******************The Virtual Architect v1.2***************"<<endl;
	cout<<"This is the demo of the VA. In this demo you can frame freeze the photo."<<endl;
	cout<<"The wall edges will be detected and highlighted"<<endl;
	cout<<"We can then select any color from the color chart and color any walls"<<endl;
	cout<<"We can also place an object (a wall clock in this demo) any where on the wall"<<endl;
	cout<<"For demo purposes we have used a big 3 folded card board sheet which acts as a wall"<<endl;
	cout<<"************************** KEYS ***************************"<<endl<<endl;
	cout<<"Press 'f' : Freeze frame"<<endl;
	cout<<"Press 't' : To work on test image"<<endl;
	cout<<"Press 'h' and mouse click : To find hough line transform of the image"<<endl;
	cout<<"Press 'g' : grabcut an object from image"<<endl;
	cout<<"Press '1-9' : to place an object on the image"<<endl;
	cout<<"Press 'p' : to stop the object to move"<<endl;
	cout<<"Press 'r' : to reset everything"<<endl;
	cout<<"*****************************************************************"<<endl;
}

//----------------------------------------Dhaval HSV Part Ends ------------------------------------
int main( int argc, char** argv )
{
	help();
	
    //char* filename = argc >= 2 ? argv[1] : (char*)"home_hough.jpg";
    string filename = "/home/ubuntu/opencv-2.4.9/samples/cpp/dhaval_programs/demo_images/clock2.png";
    foreground_old = imread(filename, -1);
    
    VideoCapture cap(0);
	
	if(!cap.isOpened())
	{
		cout<< "ERROR: cannot open webcam"<<endl;
		return -1;
	}
	
	if(!cap.read(bgrImage))
	{
			cout<< " ERROR: frame not read "<<endl;
			return 0;
	}
	
    if( bgrImage.empty() )
    {
        cout << "Image empty"<<endl;
        return 0;
    }

    if( bgrImage.empty() )
    {
        cout << "Image empty"<<endl;
        return 0;
    }

    //mask.create(bgrImage.rows + 2, bgrImage.cols + 2, CV_8UC1);
    
//    namedWindow( "BGR image", 0);	//CV_WINDOW_AUTOSIZE
//    namedWindow( "HSV image", 0);
    namedWindow( "NEW image", 0);		//one window to display everything
//    namedWindow( "Mask image", 0);
    setMouseCallback("NEW image", onClick, 0);	//BGR image
    
/*    
    createTrackbar("max", "HSV image", &upper, offset_max, on_trackbar);
    createTrackbar("min", "HSV image", &lower, offset_max, on_trackbar2);
    createTrackbar( "lo_diff", "HSV image", &loDiff, 255, 0 );
    createTrackbar( "up_diff", "HSV image", &upDiff, 255, 0 );
*/
 
//------------------------------ For HSV color select    
    HSV.create(390,640,CV_8UC3); //Mat to store clock image
	HSV.setTo(Scalar(200,0,200));

	namedWindow(window_name);
	createTrackbar( "Hue",window_name, &H, HUEMAX, onTrackbar_changed );
	createTrackbar( "Saturation",window_name, &S, SATMAX,onTrackbar_changed );
	createTrackbar( "Value",window_name, &V, VALMAX,onTrackbar_changed);
	onTrackbar_changed(0,0); //initialize window

	setMouseCallback( window_name, onMouse, 0 );
//----------------------------------------------------
//------------------- Jayanth inits-------------------------
//    namedWindow("Hough Source",0);
    
    createTrackbar("Threshold", "NEW image", &threshold_value, 400, on_trackbar_hough_thresh);
    createTrackbar("Canny Threshold 1", "NEW image", &canny_threshold_value_1, 300, on_trackbar_canny_thresh_1);
    createTrackbar("Canny Threshold 2", "NEW image", &canny_threshold_value_2, 300, on_trackbar_canny_thresh_2);
//    createTrackbar("Rho", "Hough Source", &rho_value, 5, on_trackbar_hough_rho);
//-------------------------------------------------------------
    bgrImage.copyTo(dupBGR);	//dupBGR - final output
    cvtColor(dupBGR, hsvImage, CV_BGR2HSV);	//hsvImage - original hsv image
    hsvImage.copyTo(hsvOutput);	// copy of hsvImage
    hsvImage.copyTo(hsvTemp);	//copy of hsvImage
    
//    imshow("HSV image", hsvImage);
    imshow("NEW image", bgrImage);	//BGR image
    
    bool takeSnap = false;
    bool enableOverlay = false;
	int k = 0;

    while(1)
    {
		
		k = waitKey(10);
		if(k == 27)
		{
			cout<< "exiting program..."<<endl;
			destroyAllWindows();
			break;
		}
		if(((char)k == 'f' || (char)k == 'F') || ((char)k == 't' || (char)k == 'T') && !takeSnap)
		{
			
			takeSnap = true;
			videoMode = false;
		
			if((char)k == 't' || (char)k == 'T')
			{
				string filename = filePath+"home_wall.JPG";
				bgrImage = imread(filename, 1);
				if( bgrImage.empty() )
				{
					cout << "Image empty"<<endl;
					return 0;
				}
			}
			else
			{
				if(!cap.read(bgrImage))
				{
					cout<< " ERROR: frame not read "<<endl;
					break;
				}
			}
			resize(bgrImage, bgrImage, Size(700, 700), 0, 0, INTER_LINEAR);
			mask.create(bgrImage.rows + 2, bgrImage.cols + 2, CV_8UC1);

			bgrImage.copyTo(dupBGR);	//dupBGR - final output
			cvtColor(dupBGR, hsvImage, CV_BGR2HSV);	//hsvImage - original hsv image
			hsvImage.copyTo(hsvOutput);	// copy of hsvImage
			hsvImage.copyTo(hsvTemp);	//copy of hsvImage
	
			imshow("NEW image", bgrImage);	//BGR image
			cout<<"Frame Freeze!"<<endl<<endl;
		}
		else if(((char)k == 'h' || (char)k == 'H'))
		{
			enableHough = true;
			cout<<"Taking Hough line transform.";
			cout<<" Red lines are the wall edges"<<endl<<endl;
		}
		else if(((char)k == '1' || (char)k == '2' || (char)k == '3' || (char)k == '4' || (char)k == '5' || (char)k == '6' || (char)k == '7' || (char)k == '8' || (char)k == '9' || (char)k == '0') && takeSnap)
		{
			enableOverlay = true;
			key = k;
			cout<<"Hover mouse on the screen to place object"<<endl;
			cout<<"Press Right click to zoom out the object"<<endl;
			cout<<"Press mouse wheel to zoom in the object"<<endl;
		}
		else if((char)k == 'p' || (char)k == 'P')
		{
			enableOverlay = false;
			result.copyTo(dupBGR);
			cout<<"Object placed"<<endl;
		}
/*
		else if((char)k == 'g' || (char)k == 'G')
		{
			system("cd /home/ubuntu/NVIDIA_CUDA-6.0_Samples/7_CUDALibraries/grabcutNPP/ && ./grabcutNPP");
			cout<<"Grab cut complete"<<endl;
			foreground_old = imread("/home/ubuntu/opencv-2.4.9/samples/cpp/dhaval_programs/Result.png", -1);
		}
*/
		else if((char)k == 'r' || (char)k == 'R')
		{
			takeSnap = false;
			enableHough = false;
			onlyOnStart = true;
			k=0;	
			cout<<"Video mode"<<endl;
		}
		if(!takeSnap)		//if nothing pressed just continue reading frames from camera
		{	
			videoMode = true;
			if(!cap.read(bgrImage))
			{
				cout<< " ERROR: frame not read "<<endl;
				return 0;
			}	
#if 1	
			bgrImage.copyTo(dupBGR);	//dupBGR - final output
			cvtColor(dupBGR, hsvImage, CV_BGR2HSV);	//hsvImage - original hsv image
			hsvImage.copyTo(hsvOutput);	// copy of hsvImage
			hsvImage.copyTo(hsvTemp);	//copy of hsvImage
#endif
			imshow("NEW image", bgrImage);	//BGR image

			splitNColor();
		}

		if(enableOverlay && takeSnap)
		{
			overlayObject(key);
		}
		   
	}
	return 0;
}

