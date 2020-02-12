// Computer Vision project hand gasture recognization
// Programmed by: Jerry Wang
// Massey University, Auckland, NZ
// May, 2018
// g++ gasture.cpp - o gasture `pkg-config --cflags --libs opencv`
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
//#include "opencv2/ximgproc.hpp"
#include <cstdio>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <dirent.h>
#include <unistd.h>
using namespace cv;
using namespace std;
using namespace chrono;
using namespace cv::ml;
//using namespace cv::ximgproc;
char classarray[37]={' ','0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
float PI=3.1415926;     //PI
float r;                //predict result
int r1;                 //similarity gesture with prediction
int upright=0;          //contour shape upright
Rect handposition;      //return contour rect area
float fps=0.0;          //fps of video
bool camera=0;          //parameter for the function with camera or without
int classnumber=0;      //default is 0->' ' nothing to show in the image
int savefile=0;         //parameter to create data file of images
int FDFeatureNumber=19; //change this for different FD, 19 in here (20-1)
Ptr<ANN_MLP> model;     //ANN classifier
const string& filename_to_load="jerrywang_19_96.9_10020.xml";  //my xml, ceated with 19 fourier descriptors, trained with int max_iter=10020, and gained 96.9% accuracy

//test if a file exist
static inline bool file_exist(const std::string& filename) {
    if (FILE *file = fopen(filename.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

//load a classifier name listed as above
template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{   Ptr<T> modeltoload;
    // load classifier from the specified file
    if (file_exist(filename_to_load)) {
        modeltoload = StatModel::load<T>( filename_to_load );
        if( modeltoload.empty() )
            cout << "Error: Could not read the classifier " << filename_to_load << endl;
        else
            cout << "The classifier " << filename_to_load << " is loaded.\n";
    }
    else
        cout <<"Error: The classifier " <<filename_to_load <<" is not exist.\n";
    return modeltoload;
}

//Fourier descriptors, n=FDFeatureNumber+1
static vector< float> EllipticFourierDescriptors(vector<Point>& contour)
{   vector<float> ax,ay,bx,by;
    vector< float> CE;
    int m=contour.size() ;
    int n=FDFeatureNumber+1;//number of CEs we are computing, +1 as the first number is always 2.0 and do not use it
    float t=(2*PI)/m;
    for(int k=0;k<n;k++) {
        ax.push_back(0.0);
        ay.push_back(0.0);
        bx.push_back(0.0);
        by.push_back(0.0);
        for (int i=0;i<m;i++) {
            ax[k]=ax[k]+contour[i].x*cos((k+1)*t*(i));
            bx[k]=bx[k]+contour[i].x*sin((k+1)*t*(i));
            ay[k]=ay[k]+contour[i].y*cos((k+1)*t*(i));
            by[k]=by[k]+contour[i].y*sin((k+1)*t*(i));
        }
        ax[k]=(ax[k])/m;
        bx[k]=(bx[k])/m;
        ay[k]=(ay[k])/m;
        by[k]=(by[k])/m;
    }
    for(int k=0;k<n;k++) {
        CE.push_back(sqrt((ax[k]*ax[k]+ay[k]*ay[k])/(ax[0]*ax[0]+ay[0]*ay[0]))+sqrt((bx[k]*bx[k]+by[k]*by[k])/(bx[0]*bx[0]+by[0]*by[0])));
    }
    return CE;
}

//calc similar gesture lists to r1
static void calcRandR1()
{   r1=0;
    if ((upright==0)&&(r==2)) r=17;  //1 and g
    if ((upright==1)&&(r==20)) r=19; //i and j
    if (r==2) r1=14;  //1->d
    if (r==14) r1=2;  //d->1
    if (r==1) r1=25;  //0->o
    if (r==25) r1=1;  //o ->0
    if (r==3) r1=32;  //2->v
    if (r==32) r1=3;  //v->2
    if (r==7) r1=33;  //6->w
    if (r==33) r1=7;  //w->6
    if (r==23) r1=24; //m->n
    if (r==24) r1=23; //n->m
    if (r==29) r1=30; //s->t
    if (r==30) r1=29; //t->s
    if (r==19) r1=20; //i->j
    if (r==20) r1=19; //j->i
}

//video color segmentation HSV color space
static void detectHandsHSV(Mat img)
{   Mat img_hsv;
    vector<vector<Point> > contours;
    vector<vector<Point> > contours0;
    vector<Vec4i> hierarchy;
    long int largestcontoursize=0;
    int largestcontour=0;
    upright=0;
    Mat mask(img.rows, img.cols, CV_8UC1);
    Mat temp1(img.rows,img.cols,CV_8UC1);
    Mat temp2(img.rows,img.cols,CV_8UC1);
    Mat dstImg = Mat::zeros(img.rows, img.cols, CV_8UC3);
    cvtColor(img,img_hsv,CV_BGR2HSV);
    //HSV image processing, to outline hands only
    inRange(img_hsv, Scalar(0,30,30), Scalar(40,170,256), temp1);
    inRange(img_hsv, Scalar(156,30,30), Scalar(180,170,256), temp2);
    bitwise_or(temp1, temp2, mask);
    //remove noise, edges processing
    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    erode(mask, mask, element);
    morphologyEx(mask, mask, MORPH_OPEN, element);
    dilate(mask, mask, element);
    morphologyEx(mask, mask, MORPH_CLOSE, element);
    if (savefile==0) imshow("Binary", mask);
    findContours( mask, contours0, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
//cout << contours0.size() <<endl;
    for(size_t k=0;k<contours0.size();k++) {
        if (largestcontoursize<contours0[k].size()) {
            largestcontoursize=contours0[k].size();
            largestcontour=k;
        }
    }
    contours.push_back(contours0[largestcontour]);
    approxPolyDP(Mat(contours0[largestcontour]), contours[0], 3, true);
    drawContours(dstImg, contours, 0, Scalar(0, 255, 255), 1, 8);  //draw contour
    if (savefile==0) imshow("contour",dstImg);
    handposition=boundingRect(contours0[0]);
//cout <<handposition<<endl;
    if ((handposition.height-handposition.width)>100) upright=1;
//cout <<"upright=" << upright <<endl;
    vector<float> CE;
    //get CE from an image
    CE=EllipticFourierDescriptors(contours0[largestcontour]);
    //for(int count=0;count<TEST;count++) printf("%d CE %f\n",count,CE[count]);
    Mat hand1 = Mat(Size(CE.size()-1,1),CV_32FC1,(void*)&CE[1]).clone();
//cout << hand1 <<endl;
    //get index value r
    r = model->predict(hand1);
cout << "Prediction: " << r << " Gesture: " << classarray[(int)r] << endl;
    r=(int)r;
    //adjust index value according to the hand position
    calcRandR1();
//cout << "Actually: " << r << " Gesture: " << classarray[(int)r] << endl;
if (r1!=0) cout << "Might be: " << classarray[r1] <<endl;
    char text[80];
    if (r1==0) sprintf(text,"ASL:%c",classarray[(int)r]);
    else sprintf(text,"ASL:%c/%c",classarray[(int)r],classarray[r1]);
    putText(img,text,cvPoint(10,30),FONT_HERSHEY_PLAIN,2,cvScalar(0,0,255),2,8);
    if (camera)
    {   sprintf(text,"%2.1f",fps);
        putText(img,text,cvPoint(10,80),FONT_HERSHEY_PLAIN,2,cvScalar(0,0,255),2,8);
    }
    if (savefile==0) imshow("Original",img);
}

//Static image file
static void detectHands(Mat img)
{   Mat img_gray;
    upright=0;
    vector<vector<Point> > contours0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    long int largestcontoursize=0;
    int largestcontour=0;
    Mat dstImg = Mat::zeros(img.rows, img.cols, CV_8UC3);
    cvtColor(img,img_gray,CV_BGR2GRAY);
    threshold(img_gray, img_gray, 5, 255, CV_THRESH_BINARY);
    if (savefile==0) imshow("Binary image",img_gray);
    findContours( img_gray, contours0, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE,Point());
    for(size_t k=0;k<contours0.size();k++) {
        if (largestcontoursize<contours0[k].size()) {
            largestcontoursize=contours0[k].size();
            largestcontour=k;
        }
    }
    contours.push_back(contours0[largestcontour]);
    approxPolyDP(Mat(contours0[largestcontour]), contours[0], 3, true);
    drawContours(dstImg, contours0, 0, Scalar(0, 255, 255), 1, 8);  //draw contour
    if (savefile==0) imshow("contour",dstImg);
    //detect contour position(width, height) for some special gesture
    handposition=boundingRect(contours0[0]);
//cout <<handposition<<endl;
    if ((handposition.height-handposition.width)>100) upright=1;
//cout <<"upright=" << upright <<endl;
    vector<float> CE;
    //calc CE from an image
    CE=EllipticFourierDescriptors(contours0[largestcontour]);
    Mat hand1 = Mat(Size(CE.size()-1,1),CV_32FC1,(void*)&CE[1]).clone();
//cout << hand1 <<endl;
    if (savefile==1) { //save CE to data file,Write to file!
cout << "class: " << classnumber <<" data: " << hand1 <<endl;
        ofstream datafile;// declaration of file pointer named datafile
        datafile.open("jerrywang.data", ios::app); // opens file named "xxxx" for output,add at the end
        int cs=classnumber+48;
        datafile << (char)cs;
        for (int i=1;i<FDFeatureNumber;i++) {
            datafile << "," <<CE[i];
        }
        datafile <<"\n";
        datafile.close();
    }
    else {
        //get index
        r = model->predict(hand1);
cout << "Prediction: " << r << " Gesture: " <<classarray[(int)r] << endl;
        r=(int)r;
        //adjust index value according to the hand position
        char text[80];
        if (r1==0) sprintf(text,"ASL:%c",classarray[(int)r]);
        else sprintf(text,"ASL:%c/%c",classarray[(int)r],classarray[r1]);
        putText(img,text,cvPoint(10,30),FONT_HERSHEY_PLAIN,2,cvScalar(0,0,255),2,8);
        if (camera)
        {   sprintf(text,"%2.1f",fps);
            putText(img,text,cvPoint(10,80),FONT_HERSHEY_PLAIN,2,cvScalar(0,0,255),2,8);
        }
        imshow("Original",img);
    }
}

//the function to build training data from images
static void saveimgFD2data(const char *imgdir,const char *extension)
{   DIR *dir;
    struct dirent *ent;
    string image_filename;
    int i=0;
    if ((dir = opendir (imgdir)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            if (strstr(ent->d_name,extension)!=NULL) {
                image_filename=ent->d_name;
                //find class from image_filename
                char imgclass[2];
                strcpy(imgclass,image_filename.substr(6,1).c_str());
                if (imgclass[0]>=97) {
                    classnumber=(int)imgclass[0]-96+10;
                }
                else {
                    classnumber=(int)imgclass[0]-47;
                }
                printf ("filename: %s\n", image_filename.c_str());
                cout << "char: " <<imgclass <<" class: " <<classnumber <<endl;
                //process image_filename, calc CE and save CE to file
                char imagefile[80];
                sprintf(imagefile,"%s/%s",imgdir,image_filename.c_str());
                Mat img = imread(imagefile, 1);
                if( img.empty()) {
                    cout << "Couldn't load " <<imagefile << endl;
                }
                else {
                    //call my function to save to data
                    savefile=1;
                    detectHands(img);
                    i++;
                }
            }
        }
        free(ent);
        closedir(dir);
    } else {
        /* could not open directory */
        perror ("");
    }
    cout <<"Total image files processed for training:" << i <<endl;
}

//help
static void Help(char* arg)
{   cout << "Usage 1: " << arg << " imgfile.ext" <<endl;
    cout << "Usage 2: " <<arg <<endl;
    cout << "Usage 3: " << arg << " -DATA imgfolder x " <<endl <<endl ;
    cout << "1: predict gesture form an image 2: predict gesture from video 3: create training dataset" <<endl <<endl;
    cout << "'imgfile.ext' is a hand gesture image" <<endl;
    cout << "'imgfolder' is the folder name that contain all training images" <<endl;
    cout << "'x' is the number of Fourier Descriptor, suggest between 10-30" <<endl;
    cout << "Use this function to create a training dataset 'jerrywang.data' using 'x' Fourier Descriptors with all the images in folder 'imgfolder'" <<endl;
}

//main function
int main( int argc, char** argv)
{   VideoCapture capture;
    Mat img;
    if (argc>2) //eg, usage "./execfile -DATA ./image 19" to build data file from ./image folder with 19 fourier descriptors
    {   // process all image -> FD -> save to a data file as training dataset
        if (strcmp(argv[1],"-DATA")==0) {
            savefile=1;
            if (argc==4) FDFeatureNumber=strtol(argv[3],NULL,10);
            saveimgFD2data(argv[2],".png");
            cout <<"Done!" <<endl;
        }
        else Help(argv[0]);
        return 0;
    }
    model = load_classifier<ANN_MLP>(filename_to_load);
    if(model.empty()) { cout <<"Error: XML read error" <<endl; return 0; }
    if (argc==2) {  //Image
        camera=0;
        img = imread(argv[1], 1);
        if( img.empty()) {
            cout << "Error: Couldn't load " << argv[1] << endl;
            Help(argv[0]);
            exit(0);
        }
        //my function
        detectHands(img);
        waitKey(0);
        exit(0);
    }
    else {   //Video
        camera=1;
        capture.open(0);
        Mat flipimg;
        if( capture.isOpened()) {
            capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
            capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
            do {
                system_clock::time_point start = system_clock::now();
                capture.read(flipimg);
                if( !flipimg.empty() ) {
                    //flip image
                    flip(flipimg,img,1);
                    //my function for video
                    detectHandsHSV(img);
                    system_clock::time_point end = system_clock::now();
                    float seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                    fps = 1000000 / seconds;
                }
                else {
                    printf("Warning -- No frame -- Break!\n");
                    break;
                }
            } while (waitKey(30)<0);
        }
    }
    return 0;
}
