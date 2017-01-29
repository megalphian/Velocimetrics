#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <math.h>
#include <string>

using namespace std;
using namespace cv;

static vector<Point> centers_of_roi;  // saves two roi centers
Point roi_pt1, roi_pt2;
bool roi_captured = false;
Mat cap_img;
float input_size, meters_per_pixel; // in meters
float meters_per_sec;
bool show_speed = false;

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"./data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"./data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, bool tryflip);

void selectALine(int event, int x, int y, int flags, void *param);

string cascadeName;
string nestedCascadeName;

int main( int argc, const char** argv )
{
    VideoCapture capture;
    Mat frame, image;
    string inputName;
    bool tryflip;
    CascadeClassifier cascade;
    double scale;

    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|../data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{scale|1|}{try-flip||}{@filename||}"
    );
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    cascadeName = parser.get<string>("cascade");
    scale = parser.get<double>("scale");
    if (scale < 1)
        scale = 1;
    tryflip = parser.has("try-flip");
    inputName = parser.get<string>("@filename");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }
    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
    {
        int c = inputName.empty() ? 0 : inputName[0] - '0';
        if(!capture.open(1))
            cout << "Capture from camera #" <<  c << " didn't work" << endl;
    }
    else
    {
        cout << "Could not open camera..." << endl;
    }

    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;

        int frame_count = 0;
        double t = 0;
        for(;;)
        {
          capture >> frame;

          if( frame.empty() )
              break;
            t = (double)getTickCount();
            Mat frame1 = frame.clone();
            detectAndDraw( frame1, cascade, scale, tryflip);
            frame_count += 1;

            if (frame_count == 10){
              //run the calculator every 10 frames
              t = (double)getTickCount() - t;
              t = (t /getTickFrequency());
              frame_count = 0;
              meters_per_sec = meters_per_pixel * (sqrt(pow(centers_of_roi[9].y - centers_of_roi[0].y, 2) + pow(centers_of_roi[9].x - centers_of_roi[0].x, 2) )/ t  );
              cout << "meters per s ===== " << meters_per_sec << endl;
              centers_of_roi.clear();
            }

            int c = waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
            else{
              if(c == 'g'){
                capture.grab();
                cap_img.release();

                if(capture.retrieve(cap_img))
                {
                    imshow("My_Win", cap_img);
                    cvSetMouseCallback("My_Win", selectALine, 0);
                    waitKey(0);
                }
              }

          }
        }

    }
    else
    {
        return 0;
    }
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, bool tryflip)
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    //printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        if(i > 1){
          break;
        }
        show_speed = true;
        Rect r = faces[i];
        Mat smallImgROI;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
        centers_of_roi.push_back(center);

        if (show_speed){
          ostringstream temp_;
          temp_ << meters_per_sec;
          putText(img, temp_.str(), Point(cvRound(r.x*scale), cvRound(r.y*scale)), FONT_HERSHEY_SIMPLEX, 1, Scalar(225,225 ,225));
        }
      }
      show_speed = false;
      ostringstream str_;
      str_ << meters_per_pixel;
      putText(img, str_.str(), Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(225,225 ,225));
      imshow( "result", img );
}


//references:http://stackoverflow.com/questions/16184267/selecting-a-region-opencv
void selectALine(int event, int x, int y, int flags, void *param){
  /*read in an image and select a line*/
  switch(event)
  {
    case CV_EVENT_LBUTTONDOWN:
    {
        std::cout<<"Mouse Pressed"<<std::endl;

        if(!roi_captured)
        {
            roi_pt1.x = x;
            roi_pt1.y = y;
        }
        else
        {
            std::cout<<"ROI Already Acquired"<<std::endl;
        }
    break;
    }
    case CV_EVENT_LBUTTONUP:
    {
      if(!roi_captured)
    {
        Mat cl;
        std::cout<<"Mouse LBUTTON Released"<<std::endl;

        roi_pt2.x = x;
        roi_pt2.y = y;
        line(cap_img, roi_pt1, roi_pt2, Scalar(110, 220, 0), 2);
        imshow("My_Win",cap_img);
        cout << roi_pt1.x << roi_pt1.y << endl;

        roi_captured = true;

        cout << "input real life size(in meters):"<<endl;
        cin >> input_size;
        meters_per_pixel = input_size / sqrt(pow(roi_pt1.x - roi_pt2.x, 2) +pow(roi_pt1.y - roi_pt2.y, 2) );
        cout << meters_per_pixel << endl;
    }
    else
    {
        std::cout<<"ROI Already Acquired"<<std::endl;
    }
    break;
    }

  }
}
