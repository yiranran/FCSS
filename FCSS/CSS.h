#pragma once

#ifndef _CSS_H_
#define _CSS_H_

#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <time.h>
#include <cfloat>
#include <cmath>
#include <map>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "matrix.h"

using namespace std;
using namespace cv;

class CSS  
{
public:
	CSS();
	CSS(bool pickmode, double timedist, int farthestInit, int iteration, bool ifdraw);
	virtual ~CSS();
	//============================================================================
	// Superpixel segmentation for a given step size (superpixel size ~= step*step)
	//============================================================================
    void DoSuperpixelSegmentation_ForGivenSuperpixelSize(
        Mat*&						img,//Each 32 bit unsigned int contains ARGB pixel values.
		int*&						klabels,
		int&						numlabels,
        const int&					superpixelsize,
        const double&               compactness,
        const double&               merge,
        const double&               split,
        const int&					speed);
	//============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
    void DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(
		Mat*&						img,
		int*&						klabels,
		int&						numlabels,
        const int&					K,//required number of superpixels
        const double&               compactness,
		const double&               merge,
		const double&               split,
		const int&					speed);//10-20 is a good value for CIELAB space
	//============================================================================
	// Save superpixel labels in a text file in raster scan order
	//============================================================================
	void SaveSuperpixelLabels(
		const int*&					labels,
		const int&					width,
		const int&					height,
		const string&				filename,
		const string&				path);
	//============================================================================
	// Function to draw boundaries around superpixels of a given 'color'.
	//============================================================================
	void DrawContoursAroundSegments(
		Mat*&						segmentedImage,
		int*&						labels,
		const int&					width,
		const int&					height,
		const unsigned int&			color );
    //============================================================================
    // Supervoxel segmentation for a given number of supervoxels(entry)
    //============================================================================
    void DoSupervoxelSegmentation_ForGivenNumberOfSupervoxels(
        Mat**&						imgs,
        const int&					frame_num,
        int*&						klabels,
        int&						numlabels,
        const int&					K,//required number of supervoxels
        const double&               compactness,
        const int&					speed);//10-20 is a good value for CIELAB space
	//============================================================================
	// Save superoxel labels in several text files in raster scan order
	//============================================================================
	void SaveSupervoxelLabels(
		int*&						labels,
		const int&					width,
		const int&					height,
		const int&					frame,
		char*&						output_path);
    //============================================================================
    // Function to draw supervoxel slices each frame.
    //============================================================================
    void GenerateOutput(
        char*&                      output_path,
        int*&                       labels,
        const int&                  numlabels,
        const int&                  total_frames,
        const bool&                 debug_info);
	//============================================================================
	// Draw density images
	//============================================================================
	void drawDensityPic(
		char*&						output_path);
	//============================================================================
	// Output density exponent txt
	//============================================================================
	void outputDensity(
		char*&                      output_path);
	//============================================================================
	// Supervoxel segmentation for a given number of supervoxels(offline entry)
	//============================================================================
	virtual void DoSupervoxelSegmentation_ForKSupervoxels(
		Mat**&						imgs,
		const int&					frame_num,
		int*&						klabels,
		int&						numlabels,
		const int&					K,//required number of supervoxels
		const double&               compactness,
		const int&					speed,
		const int&					c_merge,
		const int&					c_split,
		const double&				c_sdcoef,
		const int&					withcontrol)
	{}
	//============================================================================
	// Supervoxel segmentation for a given number of supervoxels(offline entry)
	//============================================================================
	virtual void DoSupervoxelSegmentation_ForKSupervoxels_steps(
		Mat**&						imgs,
		const int&					frame_num,
		int*&						klabels,
		int&						numlabels,
		const int&					K,//required number of superpixels
		const double&               compactness,//10-20 is a good value for CIELAB space
		const int&					speed,
		const int&					merge,
		const int&					split,
		const int&					step)
	{}
    
public:
	int								m_width;
	int								m_height;
	int								m_frame;

    bool                            pickmode;
    int                             farthestInit;
    double                          timedist;
	bool							adoptArea;
    bool                            ifdraw;
    int                             iteration;
    
    int                             m_K;
    int                             m_step;
    
    double                          ratio;
    double                          split;
    
    double                          merge_diflab;
	double							merge_coeff;
	int								merge_supsz;

	string							density_outpath;
	double							m_exponent;
    
protected:
	vector<double>                  kseedsl;
	vector<double>                  kseedsa;
	vector<double>                  kseedsb;
	vector<double>                  kseedsx;
	vector<double>                  kseedsy;
    vector<double>                  kseedsz;

	vector<vector<int> >			rvt;
	double							total_vol;

	double*							m_lvec;
	double*							m_avec;
	double*							m_bvec;
	double*							m_volvec;

	double*							m_densityvec;
	double							maxdensity;
	double							mindensity;

	double                          invwt;
	double							energy;
    
protected:
    //============================================================================
    // The main CSS algorithm for generating supervoxels (offline)
    //============================================================================
    void AssignLabels(
        vector<double>&             kseedsl,
        vector<double>&             kseedsa,
        vector<double>&             kseedsb,
        vector<double>&             kseedsx,
        vector<double>&             kseedsy,
        vector<double>&             kseedsz,
        int*&                       klabels,
        const int                   STEP,
        vector<double>&             adaptk,
        const double&               M = 10.0,
        const int&                  speed = 1);
    //============================================================================
    void AssignLabelsAndLloyd(
        vector<double>&             kseedsl,
        vector<double>&             kseedsa,
        vector<double>&             kseedsb,
        vector<double>&             kseedsx,
        vector<double>&             kseedsy,
        vector<double>&             kseedsz,
        int*&                       klabels,
        const int                   STEP,
        vector<double>&             adaptk,
        const double&               M = 10.0,
        const int&                  speed = 1);
	//============================================================================
	void NaiveInit(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		const int&                  STEP,
		const double&               M);
	//============================================================================
	void Lloyd(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*&                       klabels,
		const int&                  STEP,
		const double&               M = 10.0);
	//============================================================================
	void LloydWithDensity(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*&                       klabels,
		const int&                  STEP,
		const double&               M = 10.0,
		const int&					withcontrol = 0);
    //============================================================================
	void GetLABXYSeeds_ForGivenStepSize(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<double>&		edgemag);
    //============================================================================
    void GetLABXYZSeeds_ForGivenStepSize(
        vector<double>&				kseedsl,
        vector<double>&				kseedsa,
        vector<double>&				kseedsb,
        vector<double>&				kseedsx,
        vector<double>&				kseedsy,
        vector<double>&				kseedsz,
        const int&					STEP);
    //============================================================================
    void KMeansPP(
        vector<double>&				kseedsl,
        vector<double>&				kseedsa,
        vector<double>&				kseedsb,
        vector<double>&				kseedsx,
        vector<double>&				kseedsy,
        vector<double>&				kseedsz,
        const int&					STEP,
        const double&               M = 10.0);
	//============================================================================
	void KMeansPPBasedOnVol(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		vector<double>&				kseedsz,
		const int&					STEP,
		const double&               M = 10.0);
	//============================================================================
	void TransformImages(
		Mat**&						imgs,
		const int&					frame_num);

private:
	//============================================================================
	// The main MSLIC algorithm for generating superpixels
	//============================================================================
	void printlabel(int*& klabels, int width, int height, int itr);
	void PerformSuperpixelMSLIC(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		int*&						klabels,
		const int					STEP,
		const vector<double>&		edgemag,
		vector<double>&             adaptk,
		const double&				M = 10.0,
		const int&                  speed = 1
		);
	//============================================================================
	void SuperpixelSplit(
		int							itr,
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		int*&						klabels,
		const int					STEP,
		const vector<double>&		edgemag,
		vector<double>&             adaptk,
		int							labelnum,
		const double&				M = 10.0,
		const double&               split = 4.0
		);
	//============================================================================
	void EnforceLabelConnectivity(
		int                         itr,
		vector<double>&             adaptk,
		vector<double>&             newadaptk,
		const int*					labels,//input labels that need to be corrected to remove stray labels
		const int					width,
		const int					height,
		int*&						nlabels,//new labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K,//the number of superpixels desired by the user
		const double&               merge);

private:
	//============================================================================
	// Move the superpixel seeds to low gradient positions to avoid putting seeds
	// at region boundaries.
	//============================================================================
	void PerturbSeeds(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		const vector<double>&		edges);
    //============================================================================
    // Move the supervoxel seeds to low gradient positions to avoid putting seeds
    // at region boundaries.
    //============================================================================
    void PerturbSeeds(
        vector<double>&				kseedsl,
        vector<double>&				kseedsa,
        vector<double>&				kseedsb,
        vector<double>&				kseedsx,
        vector<double>&				kseedsy,
        vector<double>&				kseedsz,
        const vector<double>&		edges);
	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void DetectLabEdges(
		const double*				lvec,
		const double*				avec,
		const double*				bvec,
		const int&					width,
		const int&					height,
		vector<double>&				edges);
    //============================================================================
    // Detect color edges, to help PerturbSeeds() (Supervoxel version)
    //============================================================================
    void DetectLabEdges(
        const double*				lvec,
        const double*				avec,
        const double*				bvec,
        const int&					width,
        const int&					height,
        const int&					frame,
        vector<double>&				edges);
	//============================================================================
	// sRGB to XYZ conversion; helper for RGB2LAB()
	//============================================================================
	void RGB2XYZ(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						X,
		double&						Y,
		double&						Z);
	//============================================================================
	// sRGB to CIELAB conversion (uses RGB2XYZ function)
	//============================================================================
	void RGB2LAB(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						lval,
		double&						aval,
		double&						bval);
	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	void DoRGBtoLABConversion(
		Mat*&						img,
		double*&					lvec,
		double*&					avec,
		double*&					bvec);
    //============================================================================
    // sRGB to CIELAB conversion for 3-D videos (supervoxel)
    //============================================================================
    void DoRGBtoLABConversion(
        Mat**&						imgs,
        double*&					lvec,
        double*&					avec,
        double*&					bvec);
    //============================================================================
    // Randomly permute the values 0 to n-1 and store the permuted list in a vector
    //============================================================================
    void juRandomPermuteRange(
        int                         n,
        vector<int>&				v,
        unsigned int*				seed);
	//============================================================================
	// Compute Volume of Each Curved Cube
	//============================================================================
	void computeUnitVolume(
		double*&					volvec);
protected:
	//============================================================================
	// Calculate density from CIELAB information for video
	//============================================================================
	void DoDensityCalculation();
	
};

#endif
