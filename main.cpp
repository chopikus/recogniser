#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <bits/stdc++.h>
#define shit pair<pair<int, int>, pair<int, int> >
using namespace cv;
using namespace std;
int thresh = 50, N = 11;
int a = 0;
vector<Mat> _all;
string FILE_NAME="";
double countSomeShit(Mat image);
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void training()
{
   _all.clear();
    cv::String path_arrows("images/arrows/*.jpg"); //select only jpg
    cv::String path_bulbs("images/bulbs/*.jpg"); //select only jpg
    cv::String path_ampers("images/ampers/*.jpg"); //select only jpg
    cv::String path_volts("images/volts/*.jpg"); //select only jpg
    vector<cv::String> fn_arrows, fn_bulbs,fn_ampers,fn_volts;
    cv::glob(path_arrows,fn_arrows,true); // recurse
    cv::glob(path_bulbs,fn_bulbs,true); // recurse
    cv::glob(path_ampers,fn_ampers,true); // recurse
    cv::glob(path_volts,fn_volts,true); // recurse
    //cout <<"Sizes "<< fn_arrows.size()<<" "<<fn_bulbs.size()<<" "<<fn_ampers.size()<<" "<<fn_volts.size()<<endl;
    for (size_t k=0; k<fn_arrows.size(); ++k)
    {
        cv::Mat im = cv::imread(fn_arrows[k]);
        if (im.empty()) continue;
        _all.push_back(im);
    }
    for (size_t k=0; k<fn_bulbs.size(); ++k)
    {
        cv::Mat im = cv::imread(fn_bulbs[k]);
        if (im.empty()) continue;
        _all.push_back(im);
    }
    for (size_t k=0; k<fn_ampers.size(); ++k)
    {
        cv::Mat im = cv::imread(fn_ampers[k]);
        if (im.empty()) continue;
        _all.push_back(im);
    }
    for (size_t k=0; k<fn_volts.size(); ++k)
    {
        cv::Mat im = cv::imread(fn_volts[k]);
        if (im.empty()) continue;
        _all.push_back(im);
    }

}
int countCorners(Mat src)
{
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( src.size(), CV_32FC1 );


    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat src_gray;
    cvtColor(src, src_gray, CV_BGR2GRAY);

    cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );


    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    int count = 0;

    for( int j = 0; j < dst_norm.rows ; j++ )
    {
        for( int i = 0; i < dst_norm.cols; i++ )
        {
            if( (int) dst_norm.at<float>(j,i) > 200)
            {
                count++;
            }
        }
    }
    return count;
}
void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        for( int l = 0; l < N; l++ )
        {
            if( l == 0 )
            {
                Canny(gray0, gray, 0, thresh, 5);
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                gray = gray0 >= (l+1)*255/N;
            }
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
            vector<Point> approx;

            for( int i = 0; i < contours.size(); i++ )
            {
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                if( approx.size() == 4 &&
                        fabs(contourArea(Mat(approx))) > 1000 &&
                        isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    if( maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
}

static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( int i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }


}
int countSh;
double countSomeShit(Mat image)
{

    resize(image, image, Size(100,100));
    imshow("shiiiiiiiiiiiiiit", image);
    waitKey(0);
    Rect r = Rect(15, 15, 70, 70);
    image = Mat(image, r);
    Mat gray = image.clone();
    cvtColor(gray, gray, CV_BGR2GRAY);
    GaussianBlur(gray, gray, Size(7,7), 1.1);
    Canny(gray, gray, 0, 30);
    image = gray.clone();
    //imshow("abcd", image);
    //waitKey();
    double res=0,w=0,b=0;
    for(int y=0; y<image.rows; y++)
    {
        for(int x=0; x<image.cols; x++)
        {

            Vec3b color = image.at<Vec3b>(Point(x,y));
            long long cnt = color[0]*65536+color[1]*256+color[2];
            if (abs(cnt-16777216)>cnt)
            {
                b++;
            }
            else
                w++;

        }
    }
    return (w/b)*100;
}
void forPaint()
{
    Mat src = imread(FILE_NAME);
    Mat image;
    cvtColor(src, image, CV_BGR2GRAY);
    //freopen(output_name.c_str(), "w",stdout);
    // Convert to grayscale
    Mat gray;
    cvtColor(src, gray, CV_BGR2GRAY);
    // Use Canny instead of threshold to catch squares with gradient shading
    Mat bw;
    Canny(gray, bw, 0, 50, 5);

    // Find contours

    std::vector<std::vector<Point> > contours;
    findContours(bw.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    std::vector<Point> approx;
    set<shit>  SetCircles, SetRectangles;
    Mat dst = src.clone();
    map<pair<int, int> , bool> Map;

    for (int i = 0; i < contours.size(); i++)
    {
        // Approximate contour with accuracy proportional
        // to the contour perimeter

        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
        //cout << approx.size()<<endl;
        if (approx.size() >= 4 && approx.size() <= 6)
        {
            // Number of vertices of polygonal curve
            int vtc = approx.size();

            // Get the cosines of all corners
            std::vector<double> cos;
            for (int j = 2; j < vtc+1; j++)
                cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));

            // Sort ascending the cosine values
            std::sort(cos.begin(), cos.end());

            // Get the lowest and the highest cosine
            double mincos = cos.front();
            double maxcos = cos.back();

            // Use the degrees obtained above and the number of vertices
            // to determine the shape of the contour
            if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
            {
                vector<Point> v=contours[i];
                bool boo = 0;
                for (int j=0; j<contours[i].size(); j++)
                {
                    if (!Map[{contours[i][j].x, contours[i][j].y}]){
                    Map[{contours[i][j].x, contours[i][j].y}] = 1;
                    }
                    else{
                        boo = 1;
                        break;
                    }
                }
                if (!boo)
                {
                    Rect r = boundingRect(contours[i]);
                    shit sh = make_pair(make_pair(r.x, r.y), make_pair(r.width, r.height));
                        SetRectangles.insert(sh);
                }

            }
        }
        else
        {
            // Detect and label circles
            double area = contourArea(contours[i]);
            Rect r = boundingRect(contours[i]);
            int radius = r.width / 2;
            if (radius>=30 && std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
                    std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
            {
                vector<Point> v=contours[i];
                bool boo = 0;

                shit sh = make_pair(make_pair(r.x, r.y), make_pair(r.width, r.height));
                SetCircles.insert(sh);
            }
        }

    }
    //cout << SetCircles.size()<<endl;
    vector<Rect> Volt, Ampr, Arrw, Bulb;
    for (set<shit>::iterator it = SetRectangles.begin(); it!=SetRectangles.end(); it++)
    {
        shit sh = (*it);
        Rect r = Rect(sh.first.first, sh.first.second, sh.second.first, sh.second.second);
        if (r.x>20)
        r.x-=20;
        if (r.y>20)
        r.y-=20;
        if (r.height+r.y<src.size().height-20)
        r.height+=20;
        if (r.width+r.x<src.size().width-20)
        r.width+=20;

        rectangle(src, r, Scalar(128, 0, 128), 5);

    }
    for (set<shit>::iterator it = SetCircles.begin(); it!=SetCircles.end(); it++)
    {
        shit sh = (*it);
        Rect r = Rect(sh.first.first, sh.first.second, sh.second.first, sh.second.second);
        double SHHIT = countSomeShit(Mat(src, r));
        cout << SHHIT<<endl;
      if (SHHIT>8.3 && SHHIT<10.1)
        {
            rectangle(src, r, Scalar(0, 0, 255), 5);
            //cout << "arrw"<<endl;
            //arrw
        }
        if (SHHIT<7.15)
        {
            //cout << "volt"<<endl;
            rectangle(src, r, Scalar(0, 255, 255), 5);
            //volt
        }
        if (SHHIT>=7.15 && SHHIT<8.3)
        {
           // cout << "ampr"<<endl;
            rectangle(src, r, Scalar(0, 255, 0), 5);
        }
        if (SHHIT>=10.1)
        {
            //cout << "bulb"<<endl;
            rectangle(src, r, Scalar(255, 0, 0), 5);
        }
    }
    for (int i=0; i<Arrw.size(); i++)
    {
        rectangle(src, Arrw[i], Scalar(0, 0, 255), 5);
    }
    for (int i=0; i<Ampr.size(); i++)
    {
        rectangle(src, Ampr[i], Scalar(0, 255, 0), 5);
    }
    for (int i=0; i<Bulb.size(); i++)
    {
        rectangle(src, Bulb[i], Scalar(255, 0, 0), 5);
    }
    for (int i=0; i<Volt.size(); i++)
    {
        rectangle(src, Volt[i], Scalar(255, 255, 0), 5);
    }
    imshow("window", src);
    waitKey(0);

}
void forPrint()
{
      vector<Point> res_circles;
    //namedWindow( wndname, 1 );
    vector<vector<Point> > squares;
    Mat image = imread(FILE_NAME);

    findSquares(image, squares);
    Mat src = image.clone();
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    medianBlur(gray, gray, 5);
    vector<Vec3f> circles;
    HoughCircles( gray, circles, CV_HOUGH_GRADIENT,
                  1, 200, 100, 55, 0, 0 );
    //cout <<"CIRCLES "<< circles.size()<<endl;
    for( int i = 0;  i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];
        res_circles.push_back(center);
        int minX = max(0, c[0]-radius);
        int minY = max(0, c[1]-radius);
        int maxX = minX+radius*2;
        int maxY = minY+radius*2;
        maxX = min(maxX, src.size().width-1);
        maxY = min(maxY, src.size().height-1);
        Rect r = Rect(minX, minY, maxX-minX, maxY-minY);
        //rectangle(src, r, Scalar(255, 0, 0), 5);
        Mat circleInRect = Mat(src, r);
        double SHHIT = countSomeShit(circleInRect);
        cout << SHHIT<<endl;
        if (SHHIT>=8.6 && SHHIT<=10)
        {
            rectangle(src, r, Scalar(0, 0, 255), 5);
          //  cout << "arrw"<<endl;
            //arrw
        }
        if (SHHIT<7.12)
        {
            //cout << "volt"<<endl;
            rectangle(src, r, Scalar(0, 255, 255), 5);
            //volt
        }
        if (SHHIT>=7.12 && SHHIT<=8.6)
        {
            //cout << "ampr"<<endl;
            rectangle(src, r, Scalar(0, 255, 0), 5);
        }
        if (SHHIT>10)
        {
            //cout << "bulb"<<endl;
            rectangle(src, r, Scalar(255, 0, 0), 5);
        }

    }
    for (int i=0; i<circles.size(); i++)
    {
         Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];

    }

    set<shit> Set;
    for (int i=0; i<squares.size(); i++)
    {
        int minX=INT_MAX,maxX=INT_MIN, minY=INT_MAX,maxY=INT_MIN;
        for (int j=0; j<squares[i].size(); j++)
        {
            minX = min(minX, squares[i][j].x);
            maxX = max(maxX, squares[i][j].x);
            minY = min(minY, squares[i][j].y);
            maxY = max(maxY, squares[i][j].y);
        }
        rectangle(src, Rect(minX, minY, maxX-minX, maxY-minY), Scalar(128,0,128), 5);
        Set.insert(make_pair(make_pair(minX, minY), make_pair(maxX, maxY)));

    }
    imshow("abcd", src);
    waitKey();

}
int main(int argc, char** argv)
{
    freopen("Data.txt", "r", stdin);
    bool isAPhoto=0,isPrint=0,isPaint=0;
    cin >> FILE_NAME;
    cin >> isAPhoto;
    cin >> isPrint;
    cin >> isPaint;
    if (isPrint)
    {
        forPrint();
        return 0;
    }
    if (isPaint)
    {
        forPaint();
        return 0;
    }
    vector<Point> res_circles;
    //namedWindow( wndname, 1 );
    vector<vector<Point> > squares;
    Mat image = imread(FILE_NAME);

    findSquares(image, squares);
    Mat src = image.clone();
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    medianBlur(gray, gray, 5);
    vector<Vec3f> circles;
    HoughCircles( gray, circles, CV_HOUGH_GRADIENT,
                  1, 200, 100, 55, 0, 0 );
    //cout <<"CIRCLES "<< circles.size()<<endl;
    vector<shit> resVolt, resArrw, resAmpr, resBulb, resRest;
    for( int i = 0;  i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];
        res_circles.push_back(center);
        int minX = max(0, c[0]-radius);
        int minY = max(0, c[1]-radius);
        int maxX = minX+radius*2;
        int maxY = minY+radius*2;
        maxX = min(maxX, src.size().width-1);
        maxY = min(maxY, src.size().height-1);
        Rect r = Rect(minX, minY, maxX-minX, maxY-minY);
        //rectangle(src, r, Scalar(255, 0, 0), 5);
        Mat circleInRect = Mat(src, r);
        double SHHIT = countSomeShit(circleInRect);
        cout << SHHIT << endl;
        int x1 = r.x;
            int y1 = r.y;
            int x2 = (r.x+r.width);
            int y2 = (r.y+r.height);
            shit sh;
            sh.first.first = x1;
            sh.first.second = (y1+y2)/2;
            sh.second.first = x2;
            sh.second.second = (y1+y2)/2;

        if (SHHIT>=8.58)
        {
            rectangle(src, r, Scalar(0, 0, 255), 5);
            resArrw.push_back(sh);
          //  cout << "arrw"<<endl;
            //arrw
        }
        if (SHHIT>5 && SHHIT<8)
        {
            //scout << "volt"<<endl;
            resVolt.push_back(sh);
            rectangle(src, r, Scalar(0, 255, 255), 5);
            //volt
        }
        if (SHHIT<5)
        {
            resAmpr.push_back(sh);
            //cout << "ampr"<<endl;
            rectangle(src, r, Scalar(0, 255, 0), 5);
        }
        if (SHHIT>=8 && SHHIT<8.58)
        {
            resBulb.push_back(sh);
            //cout << "bulb"<<endl;
            rectangle(src, r, Scalar(255, 0, 0), 5);
        }

    }
    for (int i=0; i<circles.size(); i++)
    {
         Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];

    }

    set<shit> Set;
    for (int i=0; i<squares.size(); i++)
    {
        int minX=INT_MAX,maxX=INT_MIN, minY=INT_MAX,maxY=INT_MIN;
        for (int j=0; j<squares[i].size(); j++)
        {
            minX = min(minX, squares[i][j].x);
            maxX = max(maxX, squares[i][j].x);
            minY = min(minY, squares[i][j].y);
            maxY = max(maxY, squares[i][j].y);
        }

        Set.insert(make_pair(make_pair(minX, minY), make_pair(maxX, maxY)));
        rectangle(src, Rect(minX, minY, maxX-minX, maxY-minY), Scalar(128, 0, 128), 5);
                    shit sh;
            sh.first.first = minX;
            sh.first.second = (minY+maxY)/2;
            sh.second.first = maxX;
            sh.second.second = (minY+maxY)/2;
        resRest.push_back(sh);
    }
   /* cout << "{"<<endl;
    cout << " \"data\": [ "<<endl;
    int cccc=1;
    for (int i=0; i<resArrw.size(); i++)
    {
        cout << "{"<<endl;
        cout << "\"type\": \"e\" ,"<<endl;
        cout << "\"name\": \"\","<<endl;
        cout << "\"value\": 0,"<<endl;
        cout << "\"nodes\": {"<<endl;
        cout << "\"from\": [1,2],"<<endl;
        cout << "\"to\": [1,2]"<<endl;
        cout << "}"<<endl;
        cout << "}"<<endl;
        if (cccc!=resArrw.size()+resAmpr.size()+resBulb.size()+resVolt.size()+resRest.size())
            cout << ",";
        cccc++;
    }
    for (int i=0; i<resAmpr.size(); i++)
    {
        cout << "{"<<endl;
        cout << "\"type\": \"am\" ,"<<endl;
        cout << "\"name\": \"\","<<endl;
        cout << "\"value\": 0,"<<endl;
        cout << "\"nodes\": {"<<endl;
        cout << "\"from\": [1,2],"<<endl;
        cout << "\"to\": [1,2]"<<endl;
        cout << "}"<<endl;
        cout << "}"<<endl;
        if (cccc!=resArrw.size()+resAmpr.size()+resBulb.size()+resVolt.size()+resRest.size())
            cout << ",";
        cccc++;
    }
    for (int i=0; i<resVolt.size(); i++)
    {
        cout << "{"<<endl;
        cout << "\"type\": \"v\" ,"<<endl;
        cout << "\"name\": \"\","<<endl;
        cout << "\"value\": 0,"<<endl;
        cout << "\"nodes\": {"<<endl;
        cout << "\"from\": [1,2],"<<endl;
        cout << "\"to\": [1,2]"<<endl;
        cout << "}"<<endl;
        cout << "}"<<endl;
        if (cccc!=resArrw.size()+resAmpr.size()+resBulb.size()+resVolt.size()+resRest.size())
            cout << ",";
        cccc++;
    }

    for (int i=0; i<resRest.size(); i++)
    {
        cout << "{"<<endl;
        cout << "\"type\": \"r\" ,"<<endl;
        cout << "\"name\": \"\","<<endl;
        cout << "\"value\": 0,"<<endl;
        cout << "\"nodes\": {"<<endl;
        cout << "\"from\": [1,2],"<<endl;
        cout << "\"to\": [1,2]"<<endl;
        cout << "}"<<endl;
        cout << "}"<<endl;
        if (cccc!=resArrw.size()+resAmpr.size()+resBulb.size()+resVolt.size()+resRest.size())
            cout << ",";
        cccc++;
    }

    cout << "]"<<endl;
    cout << "}";*/
    imshow("shiit", src);
    waitKey();
}
