#pragma once
#include "SCMS2.h"

#define STRICT
//#define RVT_AFTER_S_M

class SC5 : public SCMS2{
public:
	SC5(bool pickmode, double timedist, int farthestInit, int iteration, bool ifdraw, int choice)
		: SCMS2(pickmode, timedist, farthestInit, iteration, ifdraw, choice){}

	void DoSupervoxelSegmentation_ForKSupervoxels(
		Mat**&						imgs,
		const int&					frame_num,
		int*&						klabels,
		int&						numlabels,
		const int&					K,//required number of supervoxels
		const double&              	compactness,
		const int&                 	speed,
		const int&					c_merge,
		const int&					c_split,
		const double&				c_sdcoeff,
		const int&					withcontrol = 0)
	{
		const int supervoxelsize = 0.5 + double(imgs[0]->rows * imgs[0]->cols * frame_num) / double(K);
		m_K = K;
		//------------------------------------------------
		const int STEP = pow(double(supervoxelsize), 1.0 / 3.0) + 0.5;
		cout << "Step: " << STEP << endl;
		m_step = STEP;
		//------------------------------------------------
		m_width = imgs[0]->cols;
		m_height = imgs[0]->rows;
		m_frame = frame_num;
		int sz = m_width * m_height * m_frame;
		//------------------------------------------------
		if (!klabels) klabels = new int[sz];
		for (int s = 0; s < sz; s++) klabels[s] = -1;

		TransformImages(imgs, frame_num);

		//STEP 1 initialize k seeds
		cout << "-------STEP 1------" << endl;
		if (!farthestInit) {
			//farthestInit = False
			//uniformly select initial kseeds
			GetLABXYZSeeds_ForGivenStepSize(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, STEP);
		}
		else if (farthestInit == 1){
			//farthestInit = True
			//conduct k-means++-like seeding, get clusters(kseeds)
			KMeansPPBasedOnVol(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, STEP, compactness);
		}
		else if (farthestInit == 2){
			NaiveInit(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, STEP, compactness);
		}
		else if (farthestInit == 3){
			KMeansPP(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, STEP, compactness);
		}
		cout << "Done" << endl;

		DoDensityCalculation();

		int numk = (int)kseedsl.size();
		cout << "numk: " << numk << " m_K: " << m_K << endl;

		vector<double> adaptk(numk, 1);

		int successTimes = 0;

		for (int itr = 0; itr < iteration; itr++)
		{
			//STEP 2 use kseeds to assign labels
			cout << "------STEP 2------" << endl;
			clock_t t1 = clock();
			numk = kseedsl.size();
			adaptk.assign(numk, 1);
			AssignLabels(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, adaptk, compactness, speed);
			numlabels= kseedsl.size();
			if (itr < iteration - 1) {
				//STEP 3 conduct split-and-merging, get new seeds
				cout << "------STEP 3------" << endl;
				clock_t t2 = clock();
#ifdef STRICT
				int done = 0, fail = 0;
				int lastfail = 0;
				int toSplit, toMerge1, toMerge2;
				if (!pickmode) ComputeSparseDense(c_sdcoeff);
				for (int i = 0; i <20; i++){
					if (!pickmode){
						if (!HeuristicPick(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz,
							toSplit, toMerge1, toMerge2))
							continue;
					}
					else{
						if (!RandPick(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz,
							toSplit, toMerge1, toMerge2))
							continue;
					}
					//cout << "picked: " << toSplit << " " << toMerge1 << " " << toMerge2; //<< endl;
					if (CheckFeasibility(toSplit, toMerge1, toMerge2, false)){
						SplitAndMerge(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels,
							toSplit, toMerge1, toMerge2);//update klabels at same time
						done++;
						//if(!pickmode){
						//	if(r1 != -1) dense.erase(dense.begin() + r1);
						//	if(r2 != -1) sparse.erase(sparse.begin() + r2);
						//}
						cout << "progress: " << done << " -- " << fail - lastfail << endl;
						lastfail = fail;
					}
					else{
						fail++;
					}
				}
				cout << "succeed " << done << " in " << done + fail << " -- " << fail - lastfail << endl;
				successTimes += done;

				clock_t t3 = clock();

#ifdef RVT_AFTER_S_M
				//since we update klabels (besides kseeds) in SplitAndMerge, 
				//this step can be omitted in an approximate manner
				AssignLabels(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, adaptk, compactness, speed);
#endif
				
				//Lloyd(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, compactness);
				LloydWithDensity(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, compactness, withcontrol);
				
				clock_t t4 = clock();

				cout << (double)(t2 - t1) / CLOCKS_PER_SEC << ' ' << (double)(t3 - t2) / CLOCKS_PER_SEC << ' ' << (double)(t4 - t3) / CLOCKS_PER_SEC << endl;

#else
				int* newlabels = new int[sz];
				map<int, int> hashtable;
				Merge2(itr, klabels, newlabels, hashtable, numlabels, c_merge);
				clock_t t3 = clock();
				//Split2(itr, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, newlabels, hashtable, numlabels, split);
				Split2(itr, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, newlabels, hashtable, numlabels, c_split);
				clock_t t4 = clock();

				cout << (double)(t2 - t1) / CLOCKS_PER_SEC << ' ' << (double)(t3 - t2) / CLOCKS_PER_SEC << ' ' << (double)(t4 - t3) / CLOCKS_PER_SEC << endl;

				for (int i = 0; i < sz; i++) klabels[i] = newlabels[i];

				if (newlabels) delete[] newlabels;
#endif
			}
			else{
				cout << "------STEP 3------" << endl;
				clock_t t2 = clock();
				int* newlabels = new int[sz];
				map<int, int> hashtable;
				Merge8(itr, klabels, newlabels, hashtable, numlabels, c_merge);
				clock_t t3 = clock();

				cout << (double)(t2 - t1) / CLOCKS_PER_SEC << ' ' << (double)(t3 - t2) / CLOCKS_PER_SEC << endl;

				for (int i = 0; i < sz; i++) klabels[i] = newlabels[i];

				if (newlabels) delete[] newlabels;
			}
		}
		cout << "Iteration Done" << endl;
#ifdef STRICT
		cout << "success times: " << successTimes << endl << endl;
#endif
	}
};