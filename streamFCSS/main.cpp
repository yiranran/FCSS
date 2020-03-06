#include <stdio.h>
#include <time.h>

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
#include "SCST.h"

using namespace cv;

#define DEBUG 1

const int choice = 2;//choice = 2 -> connect - 10, choice = 1 -> connect - 6, choice = 4 -> connect 26

Mat** readFrames(char* input_path, int range, int frame_id_start);
void saveFrames(Mat** images, char* output_path, int range, int frame_id_start);
int countFiles(char* input_path);
bool makeDirectory(char* output_path);

int main(int argc, char **argv) {
	//if (argc < 5) {
	//if (argc < 6) {
	if (argc < 4) {
		//printf("%s [input_folder] [output_folder] [sv_num] [compactness]\n", argv[0]);
		//printf("%s [input_folder] [output_folder] [sv_num] [compactness] [densityExponent]\n", argv[0]);
		printf("%s [input_folder] [output_folder] [sv_num]\n", argv[0]);
#ifdef _WIN32
		printf(" input_folder --> input folder of png video frames\n");
#else
		printf(" input_folder --> input folder of ppm video frames\n");
#endif
		printf("output_folder --> output folder of segmentation results\n");
		printf("       sv_num --> desired number of supervoxels (50 - 10000)\n");
		return 1;
	}
	int range = 20;
	bool pickmode = false;
	double timedist = 1.0;
	bool farthest = true;
	int c_merge = 3;
	int withcontrol = 2;

	// Initialize Parameters
	int svcount = 300;
	double compactness = 10.0;
	int speed = 2;
	bool ifdraw = true;
	int iteration = 10;

	// Read Parameters
	char* input_path = argv[1];
	char* output_path = argv[2];
	svcount = atoi(argv[3]);
	compactness = 10.0;
	double densityexponent = 0.8;

	printf("%s\n", input_path);
	printf("densityexpoent:%f\n", densityexponent);
	printf("control:%d\n", withcontrol);
	printf("=====================\n");

	if (svcount < 50 || svcount > 20000 || compactness < 1 || compactness > 40) {
		fprintf(stderr, "Unable to use the input parameters.");
		return 1;
	}

	if (densityexponent < -100 || densityexponent > 1) {
		fprintf(stderr, "Unable to use the input parameters.");
		return 1;
	}

	// Start Segmentation
	SCST* scst = new SCST(pickmode, timedist, farthest, iteration, ifdraw, choice);
	scst->m_exponent = densityexponent;

	int frame_num = countFiles(input_path);
	int last_clip = frame_num % range;
	int num_clip = frame_num / range;
	if (last_clip < 5){
		range = ceil(double(frame_num) / num_clip);
		last_clip = frame_num % range;
		num_clip = frame_num / range;
	}
	int a_first = 5; // each clip a * K seeds
	int width = 0, height = 0;

#ifdef linux
	struct timeval start, stop;
	gettimeofday(&start, NULL);
#endif
#ifdef _WIN32
	clock_t start, finish;
	start = clock();
#endif

	unsigned seed = (unsigned)time(NULL);
	srand(seed);
	printf("seed in main: %d\n", seed);

	//First layer segmentation
	clock_t start1, finish1;
	start1 = clock();
	bool lastClip = last_clip > 0 ? true : false;
	int interval = range;
	int frame_id_start = 0, frame_id_end = -1;
	for (int i = 0; i < num_clip + 1; i++){
		if (i == num_clip){
			if (lastClip) interval = last_clip;
			else break;
		}
		frame_id_start = frame_id_end + 1;
		frame_id_end = frame_id_start + interval - 1;
		Mat** images = readFrames(input_path, interval, frame_id_start);
		if (width == 0 || height == 0){
			width = images[0]->cols;
			height = images[0]->rows;
		}
		int sz = width * height * interval;
		int* labels = new int[sz];
		int numlabels = 0;
		scst->DoSupervoxelSegmentation_ForClip(images, i, frame_id_start, frame_id_end, labels, 
			numlabels, a_first * svcount, compactness, speed, withcontrol);
		if (labels) delete[] labels;
		for (int i = 0; i < interval; i++) delete images[i];
		if (images) delete[] images;
	}
	finish1 = clock();

	//Second layer segmentation
	clock_t start2, finish2;
	start2 = clock();

	int numlabels = 0;
	scst->DoSupervoxelSegmentation_All(width, height, frame_num, numlabels, svcount, compactness, speed, withcontrol);

	finish2 = clock();

	//Assign labels And Output
	clock_t start3, finish3;
	start3 = clock();

	if (!makeDirectory(output_path)) return -1;

	frame_id_start = 0, frame_id_end = -1;
	interval = range;
	for (int i = 0; i < num_clip + 1; i++) {
		if (i == num_clip){
			if (lastClip) interval = last_clip;
			else break;
		}
		frame_id_start = frame_id_end + 1;
		frame_id_end = frame_id_start + interval - 1;
		Mat** images = readFrames(input_path, interval, frame_id_start);
		// Assignment
		scst->DoSupervoxelAssignmentAndOutput_ForClip(images, i, frame_id_start, frame_id_end, frame_num, svcount, 
			compactness, speed,/* choice,*/ c_merge);
		saveFrames(images, output_path, interval, frame_id_start);
		// Restore
		for (int i = 0; i < interval; i++) delete images[i];
		if (images) delete[] images;
	}

	finish3 = clock();

#ifdef linux
	gettimeofday(&stop, NULL);
#endif
#ifdef _WIN32
	finish = clock();
#endif

	printf("First layer segmentation: %f seconds\n", (double)(finish1 - start1) / CLOCKS_PER_SEC);
	printf("Second layer segmentation: %f seconds\n", (double)(finish2 - start2) / CLOCKS_PER_SEC);
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

	return 0;
}

Mat** readFrames(char* input_path, int range, int frame_id_start){
    Mat** images = new Mat*[range];
    char file_path[255];
    for (int k = 0; k < range; k++)
    {
#ifdef _WIN32
        sprintf(file_path,"%s\\%05d.png", input_path, frame_id_start + k + 1);
#else
        sprintf(file_path,"%s/%05d.ppm", input_path, frame_id_start + k + 1);
#endif
        images[k] = new Mat(imread(file_path, CV_LOAD_IMAGE_COLOR));
        if(DEBUG) printf("load --> %s\n", file_path);
    }
    return images;
}

void saveFrames(Mat** images, char* output_path, int range, int frame_id_start){
	char file_path[255];
	for (int k = 0; k < range; k++)
	{
#ifdef _WIN32
		sprintf(file_path, "%s\\%05d.png", output_path, frame_id_start + k + 1);
#else
		sprintf(file_path, "%s/%05d.png", output_path, frame_id_start + k + 1);
#endif
		imwrite(file_path, *images[k]);
		if (DEBUG) printf("save --> %s\n", file_path);
	}
	return;
}

int countFiles(char* input_path){
#ifdef _WIN32
    Directory cvdir;
    vector<string> filenames = cvdir.GetListFiles(input_path, "*", false);
    int frame_num = filenames.size();
    if (frame_num == 0) {
        fprintf(stderr, "Unable to find video frames at %s", input_path);
        return -1;
    }
    return frame_num;
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
        return -1;
    }
    return frame_num;
#endif
}

bool makeDirectory(char* output_path){
#ifdef _WIN32
    if(_access(output_path, 0) != 0){
        if (_mkdir(output_path) == -1) {
            fprintf(stderr, "Unable to create the output directory at %s", output_path);
            return false;
        }
    }
#else
    struct stat st;
    if (stat(output_path, &st) != 0) {
        if (mkdir(output_path, S_IRWXU) != 0) {
            fprintf(stderr, "Unable to create the output directory at %s", output_path);
            return false;
        }
    }
#endif
    return true;
}

