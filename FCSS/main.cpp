#include <stdio.h>
#ifdef _WIN32
#include <direct.h>
#include <opencv2\contrib\contrib.hpp>
#include <io.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#endif

#include "CSS.h"
#include "SC5.h"

using namespace cv;

#define DEBUG 0
#define LABEL 0
#define DENSITYIMG 0
#define DENSITYTXT 0

const int choice = 2;//choice = 2 -> connect - 10, choice = 1 -> connect - 6, choice = 4 -> connect 26

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("%s [input_folder] [output_folder] [sv_num]\n", argv[0]);
#ifdef _WIN32
        printf(" input_folder --> input folder of png video frames\n");
#else
		printf(" input_folder --> input folder of ppm video frames\n");
#endif
        printf("output_folder --> output folder of segmentation results\n");
        printf("       sv_num --> desired number of supervoxels (50 - 15000)\n");
		printf("\n");
		return 1;
    }
    bool pickmode = false;
    double timedist = 1.0;
	int farthest = 1;
	int c_merge = 3;
    int c_split = 2;
	double c_sdcoef = 2.0;
	int withcontrol = 2;

    // Initialize Parameters
    int svcount = 300;
    double compactness = 10.0;
    int speed = 2;
    bool ifdraw = true;
    int iteration = 21;//actually 20 iter, the last one only compute rvt and enforce connectivity

    // Read Parameters
    char* input_path = argv[1];
    char* output_path = argv[2];
    svcount = atoi(argv[3]);
	compactness = 10.0;
	double densityexponent = 0.8;
    
    printf("%s\n",input_path);
	printf("densityexpoent:%f\n", densityexponent);
    printf("farinit:%d\n", farthest);
	printf("c_sdcoef:%f\n", c_sdcoef);
	printf("c_merge:%d\n", c_merge);
	printf("c_split:%d\n", c_split);
	printf("control:%d\n", withcontrol);
	printf("speed:%d\n", speed);
	printf("=====================\n");
    
    if (svcount < 50 || svcount > 20000 || compactness < 1 || compactness > 40) {
        fprintf(stderr, "Unable to use the input parameters.");
        return 1;
    }

	if (densityexponent < -100 || densityexponent > 1) {
		fprintf(stderr, "Unable to use the input parameters.");
		return 1;
	}

    //Count files in the input directory
#ifdef _WIN32
    Directory cvdir;
    vector<string> filenames = cvdir.GetListFiles(input_path, "*", false);
    int frame_num = filenames.size();
    if (frame_num == 0) {
        fprintf(stderr, "Unable to find video frames at %s", input_path);
        return 1;
    }
#else
    int frame_num = 0;
    struct dirent *pDirent;
    DIR *dirp = opendir(input_path);
    if (dirp != NULL) {
        while ((pDirent = readdir(dirp)) != NULL){
            size_t len = strlen(pDirent->d_name);
            if (len >= 4) {
                if (strcmp(".ppm", &(pDirent->d_name[len - 4])) == 0)
                    frame_num++;
            }
        }
    }
    closedir(dirp);
    if (frame_num == 0) {
        fprintf(stderr, "Unable to find video frames at %s", input_path);
        return 1;
    }
#endif
    
#ifdef linux
    struct timeval start, stop;
    gettimeofday(&start, NULL);
#endif
#ifdef _WIN32
	clock_t start, finish;
	start = clock();
#endif
    
    //Time Recorder
    clock_t start1, finish1;
    start1 = clock();
    
    // Read Frames
    Mat** images = new Mat*[frame_num];
    char file_path[255];
    for (int k = 0; k < frame_num; k++)
    {
#ifdef _WIN32
        sprintf(file_path,"%s\\%05d.png", input_path, k + 1);
#else
        sprintf(file_path,"%s/%05d.ppm", input_path, k + 1);
#endif
        images[k] = new Mat(imread(file_path, CV_LOAD_IMAGE_COLOR));
        if(DEBUG) printf("load --> %s\n", file_path);
    }
    finish1 = clock();
    
    //Time Recorder
    clock_t start2, finish2;
    start2 = clock();
    
    //Segmentation
    int width = images[0]->cols;
    int height = images[0]->rows;
    int sz = width*height*frame_num;
    //---------------------------------------------------------
    if (svcount < 4 || svcount > sz / 4) svcount = sz / 200;
    if (compactness < 1.0 || compactness > 80.0) compactness = 20.0;
    //---------------------------------------------------------
    int* labels = new int[sz];
    int numlabels;
    
    CSS* sc = new SC5(pickmode, timedist, farthest, iteration, ifdraw, choice);
    sc->m_exponent = densityexponent;
    
    //extract supervoxel
    sc->DoSupervoxelSegmentation_ForKSupervoxels(images, frame_num, labels, numlabels, svcount, 
		compactness, speed, c_merge, c_split, c_sdcoef, withcontrol);
    finish2 = clock();
    cout << "NumLabels: " << numlabels << endl;
    
    //Make the output directory
#ifdef _WIN32
    if (_access(output_path, 0) != 0) {//dir not exists
        if (_mkdir(output_path) == -1) {
            fprintf(stderr, "Unable to create the output directory at %s", output_path);
            return 1;
        }
    }
	if (DENSITYIMG){
		char densityimg_path[255];
		sprintf(densityimg_path, "%s\\densityImgs", output_path);
		if (_access(densityimg_path, 0) != 0) {//dir not exists
			if (_mkdir(densityimg_path) == -1) {
				fprintf(stderr, "Unable to create the output directory at %s", densityimg_path);
				return 1;
			}
		}
	}
#else
    struct stat st;
    if (stat(output_path, &st) != 0) {
        if (mkdir(output_path, S_IRWXU) != 0) {
            fprintf(stderr, "Unable to create the output directory at %s", output_path);
            return 1;
        }
    }
	if (DENSITYIMG){
		char densityimg_path[255];
		sprintf(densityimg_path, "%s/densityImgs", output_path);
		if (stat(densityimg_path, &st) != 0) {
			if (mkdir(densityimg_path, S_IRWXU) != 0) {
				fprintf(stderr, "Unable to create the output directory at %s", densityimg_path);
				return 1;
			}
		}
	}
#endif

    //Time Recorder
    clock_t start3, finish3;
    start3 = clock();
    
    //Output results
    sc->GenerateOutput(output_path, labels, numlabels, frame_num, DEBUG);
    if(LABEL) sc->SaveSupervoxelLabels(labels, width, height, frame_num, output_path);
	if(DENSITYIMG) sc->drawDensityPic(output_path);
    if(DENSITYTXT) sc->outputDensity(output_path);
    finish3 = clock();

#ifdef linux
    gettimeofday(&stop, NULL);
#endif
#ifdef _WIN32
	finish = clock();
#endif
    
    //Time Recorder
    printf("Load Frames: %f seconds\n", (double)(finish1 - start1) / CLOCKS_PER_SEC);
    printf("Segement: %f seconds\n", (double)(finish2 - start2) / CLOCKS_PER_SEC);
    printf("Save Frames: %f seconds\n", (double)(finish3 - start3) / CLOCKS_PER_SEC);

#ifdef linux
    double diff = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("ToTal Time: %f seconds\n", diff);

    ofstream myfile;
    char timefile[1024];
    snprintf(timefile, 1023, "%s%s", output_path, ".txt");
    myfile.open(timefile);
    myfile << diff << endl;
    myfile.close();
#endif
#ifdef _WIN32
	double diff = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("ToTal Time: %f seconds\n", diff);

	ofstream myfile;
	char timefile[1024];
	sprintf(timefile, "%s%s", output_path, ".txt");
	myfile.open(timefile);
	myfile << diff << endl;
	myfile.close();
#endif
    
    if (labels) delete[] labels;
    for(int i = 0; i < frame_num; i++) delete images[i];
    if (images) delete[] images;

    return 0;
}