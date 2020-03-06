#pragma once
#include "SCMS2.h"

#define WithSplitMerge

class SCST : public SCMS2{
private:

	// Prepare vector of random/unique values for access by counter
	vector<int>						randomColors;
	unsigned int					rInd;
	vector<int>						rvec;
	vector<int>						gvec;
	vector<int>						bvec;
	int*							lastFrameLabel;
	bool							addcolor;

	int								iteration_firstlayer;

	void addColor(){
		int i = rvec.size();
		int r, g, b;
#ifdef _WIN32
		r = ((randomColors[rInd + i] >> 16) % 256);
		g = ((randomColors[rInd + i] >> 8) % 256);
		b = (randomColors[rInd + i] % 256);
#else
		r = ((randomColors[i] >> 16) % 256);
		g = ((randomColors[i] >> 8) % 256);
		b = (randomColors[i] % 256);
#endif
		rvec.push_back(r);
		gvec.push_back(g);
		bvec.push_back(b);
		return;
	}

public:
	SCST(bool wsm, double timedist, bool farthestInit, int iteration, bool ifdraw, int choice, int iteration_firstlayer = 10)
		: SCMS2(wsm, timedist, farthestInit, iteration, ifdraw, choice)
	{
		this->iteration_firstlayer = iteration_firstlayer;
		lastFrameLabel = NULL;
		addcolor = false;
	}

	~SCST(){
		if (lastFrameLabel) delete lastFrameLabel;
	}

	//===========================================================================
	/// Step1: 1st Layer Seg
	//===========================================================================
	void DoSupervoxelSegmentation_ForClip(
		Mat**&                      imgs,
		const int&                  part_id,
		const int&                  frame_id_start,
		const int&                  frame_id_end,
		int*&                       klabels,
		int&                        numlabels,
		const int&                  K,//required number of supervoxels
		const double&               compactness,
		const int&                  speed,
		const int&					withcontrol = 0)
	{
		const int frame_clip = frame_id_end - frame_id_start + 1;
		const int supervoxelsize = 0.5 + double(imgs[0]->rows * imgs[0]->cols * frame_clip) / double(K);
		m_K = K;
		//------------------------------------------------
		const int STEP = pow(double(supervoxelsize), 1.0 / 3.0) + 0.5;
		cout << "Step: " << STEP << endl;
		m_step = STEP;
		//------------------------------------------------
		m_width = imgs[0]->cols;
		m_height = imgs[0]->rows;
		m_frame = frame_clip;
		int sz_clip = m_width * m_height * m_frame;
		//--------------------------------------------------
		if (!klabels) klabels = new int[sz_clip];
		for (int s = 0; s < sz_clip; s++) klabels[s] = -1;

		TransformImages(imgs, frame_clip);

		cout << "DoSupervoxelSegmentation_ForClip" << endl;
		cout << "id_start: " << frame_id_start << " id_end: " << frame_id_end << endl;
		cout << "m_frame: " << m_frame << endl;

		//STEP 1 initialize k seeds, conduct k-means++-like seeding, get clusters(kseeds)
		KMeansPP(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, STEP, compactness);
		cout << "Init Done" << endl;

		if (m_volvec) delete[] m_volvec; m_volvec = NULL;
		DoDensityCalculation();

		int numk = (int)kseedsl.size();
		cout << "numk: " << numk << " m_K: " << m_K << endl;

		vector<double> adaptk(numk, 1);
		stage = 1;

		clock_t start, end;
		start = clock();
		for (int itr = 0; itr < iteration_firstlayer; itr++)
		{
#ifndef WithSplitMerge
			AssignLabelsAndLloyd(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, adaptk, compactness, speed);
#else
			//STEP 2 use kseeds to assign labels
			cout << "------STEP 2------" << endl;
			clock_t t1 = clock();
			AssignLabels(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, adaptk, compactness, speed);
			//STEP 3 conduct split-and-merging, get new seeds
			cout << "------STEP 3------" << endl;
			clock_t t2 = clock();
			int done = 0, fail = 0;
			int lastfail = 0;
			int toSplit, toMerge1, toMerge2;
			if (!pickmode) ComputeSparseDense();
			for (int i = 0; i < 20; i++){
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
				if (CheckFeasibility(toSplit, toMerge1, toMerge2, false)){
					SplitAndMerge(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels,
						toSplit, toMerge1, toMerge2);//update klabels at same time
					done++;
					cout << "progress: " << done << " -- " << fail - lastfail << endl;
					lastfail = fail;
				}
				else{
					fail++;
				}
			}
			cout << "succeed " << done << " in " << done + fail << " -- " << fail - lastfail << endl;
			clock_t t3 = clock();
			//Lloyd(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, compactness);
			LloydWithDensity(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, compactness, withcontrol);
			clock_t t4 = clock();
			cout << (double)(t2 - t1) / CLOCKS_PER_SEC << ' ' << (double)(t3 - t2) / CLOCKS_PER_SEC << ' ' << (double)(t4 - t3) / CLOCKS_PER_SEC << endl;
#endif
		}
		end = clock();
		cout << "Iteration Done" << endl;
		printf("Lloyd time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

		if (part_id == 0){
			pointsl.clear();
			pointsa.clear();
			pointsb.clear();
			pointsx.clear();
			pointsy.clear();
			pointsz.clear();
			pointsvox.clear();
			pointsvol.clear();
			pointsdene.clear();
		}

		// add kseeds to points, for later clustering
		cout << "already " << pointsl.size() << endl;
		int startind = pointsl.size();
		pointsl.insert(pointsl.end(), kseedsl.begin(), kseedsl.end());
		pointsa.insert(pointsa.end(), kseedsa.begin(), kseedsa.end());
		pointsb.insert(pointsb.end(), kseedsb.begin(), kseedsb.end());
		pointsx.insert(pointsx.end(), kseedsx.begin(), kseedsx.end());
		pointsy.insert(pointsy.end(), kseedsy.begin(), kseedsy.end());
		for (int i = 0; i < kseedsz.size(); i++){
			pointsz.push_back(kseedsz[i] + frame_id_start);
		}
		AssignLabels(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, adaptk, compactness, speed);
		for (int i = 0; i < kseedsl.size(); i++){
			pointsvox.push_back(rvt[i].size());
		}
		pointsvol.insert(pointsvol.end(), kseedsl.size(), 0);
		pointsdene.insert(pointsdene.end(), kseedsl.size(), 0);
		cout << "kseedsl size " << kseedsl.size() << endl;
		for (int i = 0; i < kseedsl.size(); i++){
			for (int j = 0; j < rvt[i].size(); j++){
				pointsvol[startind+i] += m_volvec[rvt[i][j]];
				pointsdene[startind+i] += m_densityvec[rvt[i][j]];
			}
		}
		cout << "after " << pointsl.size() << " " << pointsdene.size() << endl;


		numlabels = (int)kseedsl.size();
		if (m_lvec) delete[] m_lvec;
		if (m_avec) delete[] m_avec;
		if (m_bvec) delete[] m_bvec;
		m_lvec = NULL;
		m_avec = NULL;
		m_bvec = NULL;
	}

	//===========================================================================
	/// Step2: 2nd Layer Seg
	//===========================================================================
	void DoSupervoxelSegmentation_All(
		const int&                  width,
		const int&                  height,
		const int&                  total_frames,
		int&                        numlabels,
		const int&                  K,//required number of supervoxels
		const double&               compactness,
		const int&                  speed,
		const int&					withcontrol = 0)
	{
		const int supervoxelsize = 0.5 + double(width * height * total_frames) / double(K);
		const int STEP = pow(double(supervoxelsize), 1.0 / 3.0) + 0.5;
		cout << "Step: " << STEP << endl;
		m_K = K; //update K
		m_step = STEP;
		m_width = width;
		m_height = height;
		m_frame = total_frames;
		cout << "DoSupervoxelSegmentation_All" << endl;
		cout << "# Points: " << pointsl.size() << ' ' << pointsa.size() << ' ' << pointsb.size() << ' ' << pointsx.size() << ' ' << pointsy.size() << ' ' << pointsz.size() << endl;

		//STEP 1 For points(2nd layer clustering), initialize k seeds, conduct k-means++-like seeding
		KMeansPP_Points(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, STEP, compactness);
		cout << "Init Done" << endl;
		int numk = (int)kseedsl.size();
		cout << "numk: " << numk << " m_K: " << m_K << endl;
		int* plabels = new int[pointsl.size()];
		for (int s = 0; s < pointsl.size(); s++) plabels[s] = -1;
		vector<double> adaptk(numk, 1);
		stage = 2;
		for (int itr = 0; itr < iteration; itr++)
		{
#ifndef WithSplitMerge
			AssignLabelsAndLloyd_Points(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, plabels, STEP, adaptk, compactness, speed);
#else
			//STEP 2 use kseeds to assign labels
			cout << "------STEP 2------" << endl;
			clock_t t1 = clock();
			AssignLabels_Points(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, plabels, STEP, adaptk, compactness, speed);
			//STEP 3 conduct split-and-merging, get new seeds
			cout << "------STEP 3------" << endl;
			clock_t t2 = clock();
			int done = 0, fail = 0;
			int lastfail = 0;
			int toSplit, toMerge1, toMerge2;
			if (!pickmode) ComputeSparseDense();
			for (int i = 0; i < 20; i++){
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
				if (CheckFeasibility(toSplit, toMerge1, toMerge2, false)){
					SplitAndMerge(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, plabels,
						toSplit, toMerge1, toMerge2);//update klabels at same time
					done++;
					cout << "progress: " << done << " -- " << fail - lastfail << endl;
					lastfail = fail;
				}
				else{
					fail++;
				}
			}
			cout << "succeed " << done << " in " << done + fail << " -- " << fail - lastfail << endl;
			clock_t t3 = clock();
			//Lloyd_Points(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, plabels, STEP, compactness);
			Lloyd_Points_Density(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, plabels, STEP, compactness, withcontrol);
			clock_t t4 = clock();
			cout << (double)(t2 - t1) / CLOCKS_PER_SEC << ' ' << (double)(t3 - t2) / CLOCKS_PER_SEC << ' ' << (double)(t4 - t3) / CLOCKS_PER_SEC << endl;
#endif
		}
		cout << "Iteration Done" << endl;

		/*for(int i = 0; i < kseedsz.size(); i++)
		cout << kseedsx[i] << ' ' << kseedsy[i] << ' ' << kseedsz[i] << ' ' << kseedsl[i] << ' ' << kseedsa[i] << ' ' << kseedsb[i] << endl;
		cout << endl;*/

		numlabels = (int)kseedsl.size();
		cout << numlabels << " seeds" << endl;

		rInd = rand() % 10000000; //seed for rand_r, start index for randomNumbers...
		juRandomPermuteRange(16777215, randomColors, &rInd);
		//choose color from random permuted numbers
		int r, g, b;
		rvec.clear(); gvec.clear(); bvec.clear();
		for (int i = 0; i < numlabels; i++){
#ifdef _WIN32
			r = ((randomColors[rInd + i] >> 16) % 256);
			g = ((randomColors[rInd + i] >> 8) % 256);
			b = (randomColors[rInd + i] % 256);
#else
			r = ((randomColors[i] >> 16) % 256);
			g = ((randomColors[i] >> 8) % 256);
			b = (randomColors[i] % 256);
#endif
			rvec.push_back(r);
			gvec.push_back(g);
			bvec.push_back(b);
		}

		lastFrameLabel = NULL;
		if (plabels) delete[] plabels;
	}

	//===========================================================================
	/// Step3: Assign and Output 1 (Not Used)
	//===========================================================================
	void DoSupervoxelAssignment_ForClip(
		Mat**&                      imgs,
		const int&                  part_id,
		const int&                  frame_id_start,
		const int&                  frame_id_end,
		const int&                  total_frames,
		const int&                  K,
		int*&                       klabels,
		const double&               compactness,
		const int&                  speed)
	{
		const int frame_clip = frame_id_end - frame_id_start + 1;
		const int supervoxelsize = 0.5 + double(imgs[0]->rows * imgs[0]->cols * total_frames) / double(K);
		const int STEP = pow(double(supervoxelsize), 1.0 / 3.0) + 0.5;
		m_step = STEP;
		m_width = imgs[0]->cols;
		m_height = imgs[0]->rows;
		m_frame = frame_clip;

		TransformImages(imgs, frame_clip);

		vector<double> adaptk(kseedsl.size(), 1);
		AssignLabels_ForClip(frame_id_start, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, adaptk, compactness, speed);

	}

	//===========================================================================
	/// Step3: Assign and Output 1 (Not Used)
	//===========================================================================
	void DoSupervoxelMerge_All(
		int*&						klabels,
		const int&					total_frames,
		//const int&					choice,
		const int&					merge,
		int&						numlabels)
	{
		int sz_all = m_width * m_height * total_frames;
		m_frame = total_frames;

		int* newlabels = new int[sz_all];
		map<int, int> hashtable;
		Merge2(iteration - 1, klabels, newlabels, hashtable, numlabels,/* choice,*/ merge);

		for (int i = 0; i < sz_all; i++) klabels[i] = newlabels[i];

		if (newlabels) delete[] newlabels;
	}

	//===========================================================================
	/// Step3: Assign and Output 2
	//===========================================================================
	void DoSupervoxelAssignmentAndOutput_ForClip(
		Mat**&                      imgs,
		const int&                  part_id,
		const int&                  frame_id_start,
		const int&                  frame_id_end,
		const int&                  total_frames,
		const int&                  K,
		const double&               compactness,
		const int&                  speed,
		//const int&					choice,
		const int&					merge)
	{
		const int frame_clip = frame_id_end - frame_id_start + 1;
		const int supervoxelsize = 0.5 + double(imgs[0]->rows * imgs[0]->cols * total_frames) / double(K);
		const int STEP = pow(double(supervoxelsize), 1.0 / 3.0) + 0.5;
		m_step = STEP;
		m_width = imgs[0]->cols;
		m_height = imgs[0]->rows;
		m_frame = frame_clip;

		TransformImages(imgs, frame_clip);

		int* klabels = new int[m_width * m_height * frame_clip];
		vector<double> adaptk(kseedsl.size(), 1);
		AssignLabels_ForClip2(frame_id_start, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsz, klabels, STEP, adaptk, compactness, speed);
		map<int, int> hashtable;
		int* newlabels = new int[m_width * m_height * frame_clip];
		int numlabels = 0;
		cout << "Do Merge2InPlace" << endl;
		Merge2InPlace(iteration - 1, klabels, newlabels, hashtable, numlabels, choice, merge);
		GenerateOutput_ForClip2(imgs, newlabels, numlabels, frame_id_start, frame_id_end);

		if (klabels) delete[] klabels;
		if (newlabels) delete[] newlabels;
	}

private:
	void GenerateOutput_ForClip2(
		Mat**						output,
		int*&                       labels,
		const int&                  numlabels,
		const int&                  frame_id_start,
		const int&                  frame_id_end)
	{

		const int frame_clip = frame_id_end - frame_id_start + 1;
		
		uchar* p;
		for (int i = 0; i < frame_clip; i++) {
			/*for (int j = 0; j < m_width * m_height; j++) {
				int label = labels[j + i * (m_width * m_height)];
				unsigned int r = rvec[label];
				unsigned int g = gvec[label];
				unsigned int b = bvec[label];
				ubuff[i][j] = (r << 16) | (g << 8) | b;
			}*/
			for (int y = 0; y < m_height; y++) {
				p = output[i]->ptr<uchar>(y);
				for (int x = 0; x < m_width; x++) {
					int label = labels[y * m_width + x + i * (m_width * m_height)];
					p[3 * x] = bvec[label];
					p[3 * x + 1] = gvec[label];
					p[3 * x + 2] = rvec[label];
				}
			}
		}

	}

	void GenerateOutput_ForClip(
		Mat**                       output,
		int*&                       labels,
		const int&                  numlabels,
		const int&					frame_id_start,
		const int&                  frame_id_end)
	{
		// Prepare vector of random/unique values for access by counter
		vector<int> randomNumbers;

		unsigned int r = rand() % 10000000; //seed for rand_r, start index for randomNumbers...
		juRandomPermuteRange(16777215, randomNumbers, &r);

		int* m_rvec = new int[numlabels];
		int* m_gvec = new int[numlabels];
		int* m_bvec = new int[numlabels];

		//choose color from random permuted numbers
		for (int i = 0; i < numlabels; i++) {
#ifdef _WIN32
			m_rvec[i] = ((randomNumbers[r + i] >> 16) % 256);
			m_gvec[i] = ((randomNumbers[r + i] >> 8) % 256);
			m_bvec[i] = (randomNumbers[r + i] % 256);
#else
			m_rvec[i] = ((randomNumbers[i] >> 16) % 256);
			m_gvec[i] = ((randomNumbers[i] >> 8) % 256);
			m_bvec[i] = (randomNumbers[i] % 256);
#endif
		}

		const int frame_clip = frame_id_end - frame_id_start + 1;
		uchar* p;
		for (int i = 0; i < frame_clip; i++) {
			//output[i] = new Mat(m_height, m_width, CV_8UC3, Scalar(0, 0, 0));
			for (int y = 0; y < m_height; y++) {
				p = output[i]->ptr<uchar>(y);
				for (int x = 0; x < m_width; x++) {
					int label = labels[y * m_width + x + i * (m_width * m_height)];
					p[3 * x] = m_bvec[label];
					p[3 * x + 1] = m_gvec[label];
					p[3 * x + 2] = m_rvec[label];
				}
			}
		}

		printf("Num of supervoxels: %d\n", numlabels);

		delete[] m_rvec;
		delete[] m_gvec;
		delete[] m_bvec;
	}

private:
	//===========================================================================
	/// KMeansPP_Points
	//===========================================================================
	void KMeansPP_Points(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		const int&                  STEP,
		const double&               M)
	{
		clock_t start1, finish1;
		start1 = clock();

		vector<double> distvec(pointsl.size(), DBL_MAX);

		invwt = 1.0 / ((STEP / M)*(STEP / M));

		kseedsl.resize(m_K);
		kseedsa.resize(m_K);
		kseedsb.resize(m_K);
		kseedsx.resize(m_K);
		kseedsy.resize(m_K);
		kseedsz.resize(m_K);

		int k = (int)((pointsl.size()-1) * ((double)rand() / RAND_MAX));//[0,pointsl.size()-1]
		kseedsl[0] = pointsl[k];
		kseedsa[0] = pointsa[k];
		kseedsb[0] = pointsb[k];
		kseedsx[0] = pointsx[k];
		kseedsy[0] = pointsy[k];
		kseedsz[0] = pointsz[k];

		double l, a, b, x, y, z;
		double dist, distxyz;
		double sum, cursum, r;
		for (int n = 1; n < m_K; n++){
			//update disvec
			for (int i = 0; i < pointsl.size(); i++)
			{
				l = pointsl[i];
				a = pointsa[i];
				b = pointsb[i];
				x = pointsx[i];
				y = pointsy[i];
				z = pointsz[i];

				dist = (l - kseedsl[n - 1])*(l - kseedsl[n - 1]) +
					(a - kseedsa[n - 1])*(a - kseedsa[n - 1]) +
					(b - kseedsb[n - 1])*(b - kseedsb[n - 1]);

				distxyz = (x - kseedsx[n - 1])*(x - kseedsx[n - 1]) +
					(y - kseedsy[n - 1])*(y - kseedsy[n - 1]) +
					(z - kseedsz[n - 1])*(z - kseedsz[n - 1]) * timedist * timedist;

				dist += distxyz * invwt;

				if (dist < distvec[i]){
					distvec[i] = dist;
				}
			}
			//distvec[i] = D(x)^2
			//Pr(choosing x) = D(x)^2/sum

			//calculate sum
			sum = 0;
			for (int i = 0; i < pointsl.size(); i++)
				sum += distvec[i];

			//get random number -- select voxel with largest probability
			r = sum * ((double)rand() / RAND_MAX);
			cursum = 0;
			k = 0;
			while (cursum <= r && k < pointsl.size()){
				cursum += distvec[k];
				k++;
			}
			//cout << k << endl;
			k = min(k, pointsl.size() - 1);
			//assign new seed
			kseedsl[n] = pointsl[k];
			kseedsa[n] = pointsa[k];
			kseedsb[n] = pointsb[k];
			kseedsx[n] = pointsx[k];
			kseedsy[n] = pointsy[k];
			kseedsz[n] = pointsz[k];
		}
		finish1 = clock();
		printf("Kmeans++ Points Init time: %f seconds\n", (double)(finish1 - start1) / CLOCKS_PER_SEC);
	}

	//===========================================================================
	/// AssignLabelsAndLlyod_Points (streaming)
	//===========================================================================
	void AssignLabelsAndLloyd_Points(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*&                       klabels,
		const int                   STEP,
		vector<double>&             adaptk,
		const double&               M,//compactness
		const int&                  speed)
	{
		size_t numk = kseedsl.size();
		adaptk.resize(numk, 1);
		invwt = 1.0 / ((STEP / M)*(STEP / M));//D^2 = dc^2 + (Nc/Ns)^2 * ds^2. here invwt = Nc/Ns = M / STEP.
		//M large -> more compact, M small -> better boundary recall

		int sz_all = m_width * m_height * m_frame;
		int step2 = pow(double(sz_all) / double(numk) + 0.5, 1.0 / 3.0) + 0.5;

		vector<double> distvec(pointsl.size(), DBL_MAX);
		for (int i = 0; i < pointsl.size(); i++) klabels[i] = -1;

		int x1, y1, x2, y2, z1, z2;
		double l, a, b, x, y, z;
		double dist;
		double distxyz;
		for (int itr = 0; itr < 1; itr++)
		{
			distvec.assign(pointsl.size(), DBL_MAX);

			//-----------------------------------------------------------------
			// Assign labels -> klabels
			//-----------------------------------------------------------------
			for (int n = 0; n < numk; n++)
			{
				int offset = (int)speed * adaptk[n] * step2;

				//2offset*2offset*2offset search window
				x1 = max(0.0, kseedsx[n] - offset);
				x2 = min((double)m_width, kseedsx[n] + offset);
				y1 = max(0.0, kseedsy[n] - offset);
				y2 = min((double)m_height, kseedsy[n] + offset);
				z1 = max(0.0, kseedsz[n] - offset);
				z2 = min((double)m_frame, kseedsz[n] + offset);

				for (int i = 0; i < pointsl.size(); i++){
					if (pointsx[i] >= x1 && pointsx[i] < x2 && pointsy[i] >= y1 && pointsy[i] < y2 && pointsz[i] >= z1 && pointsz[i] < z2){
						l = pointsl[i];
						a = pointsa[i];
						b = pointsb[i];
						x = pointsx[i];
						y = pointsy[i];
						z = pointsz[i];

						dist = (l - kseedsl[n])*(l - kseedsl[n]) +
							(a - kseedsa[n])*(a - kseedsa[n]) +
							(b - kseedsb[n])*(b - kseedsb[n]);

						distxyz = (x - kseedsx[n])*(x - kseedsx[n]) +
							(y - kseedsy[n])*(y - kseedsy[n]) +
							(z - kseedsz[n])*(z - kseedsz[n]) * timedist * timedist;

						dist += distxyz * invwt;

						if (dist < distvec[i]){
							distvec[i] = dist;
							klabels[i] = n; //label of voxel i is the id of current seed
						}
					}
				}
			}

			/*int invalid = 0;
			for (int i = 0; i < pointsl.size(); i++) {
			if (klabels[i] == -1) {
			invalid += 1;
			}
			}
			cout << "Invalid: " << invalid << " in " << pointsl.size() << endl;*/
			int invalid = 0;
			for (int i = 0; i < pointsl.size(); i++) {
				if (klabels[i] == -1) {
					invalid += 1;
					for (int n = 0; n < numk; n++)
					{
						l = pointsl[i];
						a = pointsa[i];
						b = pointsb[i];
						x = pointsx[i];
						y = pointsy[i];
						z = pointsz[i];

						dist = (l - kseedsl[n])*(l - kseedsl[n]) +
							(a - kseedsa[n])*(a - kseedsa[n]) +
							(b - kseedsb[n])*(b - kseedsb[n]);
						distxyz = (x - kseedsx[n])*(x - kseedsx[n]) +
							(y - kseedsy[n])*(y - kseedsy[n]) +
							(z - kseedsz[n])*(z - kseedsz[n]) * timedist * timedist;

						dist += distxyz * invwt;

						if (dist < distvec[i]){
							distvec[i] = dist;
							klabels[i] = n; //label of voxel i is the id of current seed
						}
					}
				}
			}
			cout << "Invalid: " << invalid << " in " << pointsl.size() << endl;


			//-----------------------------------------------------------------
			// Lloyd
			// Recalculate the centroid and store in the seed values
			//-----------------------------------------------------------------
			vector<double> sigmal(numk, 0);
			vector<double> sigmaa(numk, 0);
			vector<double> sigmab(numk, 0);
			vector<double> sigmax(numk, 0);
			vector<double> sigmay(numk, 0);
			vector<double> sigmaz(numk, 0);
			vector<double> clustersize(numk, 0);
			vector<double> inv(numk, 0);

			sigmal.assign(numk, 0);
			sigmaa.assign(numk, 0);
			sigmab.assign(numk, 0);
			sigmax.assign(numk, 0);
			sigmay.assign(numk, 0);
			sigmaz.assign(numk, 0);
			clustersize.assign(numk, 0);
			inv.assign(numk, 0);

			for (int i = 0; i < pointsl.size(); i++)
			{
				if (klabels[i] == -1)
					continue;
				sigmal[klabels[i]] += pointsl[i];
				sigmaa[klabels[i]] += pointsa[i];
				sigmab[klabels[i]] += pointsb[i];
				sigmax[klabels[i]] += pointsx[i];
				sigmay[klabels[i]] += pointsy[i];
				sigmaz[klabels[i]] += pointsz[i];
				clustersize[klabels[i]] += 1.0;
			}

			for (int k = 0; k < numk; k++)
			{
				if (clustersize[k] <= 0) clustersize[k] = 1;
				inv[k] = 1.0 / clustersize[k];//computing inverse now to multiply, than divide later
			}

			for (int k = 0; k < numk; k++)
			{
				kseedsl[k] = sigmal[k] * inv[k];
				kseedsa[k] = sigmaa[k] * inv[k];
				kseedsb[k] = sigmab[k] * inv[k];
				kseedsx[k] = sigmax[k] * inv[k];
				kseedsy[k] = sigmay[k] * inv[k];
				kseedsz[k] = sigmaz[k] * inv[k];
			}
		}
		/*cout << "# Kseeds: " << kseedsl.size() << endl;*/
	}

	//===========================================================================
	/// AssignLabels_Points (streaming)
	//===========================================================================
	void AssignLabels_Points(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*&                       klabels,
		const int                   STEP,
		vector<double>&             adaptk,
		const double&               M,//compactness
		const int&                  speed)
	{
		size_t numk = kseedsl.size();
		adaptk.resize(numk, 1);
		invwt = 1.0 / ((STEP / M)*(STEP / M));//D^2 = dc^2 + (Nc/Ns)^2 * ds^2. here invwt = Nc/Ns = M / STEP.
		//M large -> more compact, M small -> better boundary recall

		int sz_all = m_width * m_height * m_frame;
		int step2 = pow(double(sz_all) / double(numk) + 0.5, 1.0 / 3.0) + 0.5;

		vector<double> distvec(pointsl.size(), DBL_MAX);
		for (int i = 0; i < pointsl.size(); i++) klabels[i] = -1;

		for (int i = 0; i < rvt.size(); i++){
			rvt[i].clear();
			rvt[i].resize(0);
		}
		rvt.clear();
		rvt.resize(kseedsl.size());

		int x1, y1, x2, y2, z1, z2;
		double l, a, b, x, y, z;
		double dist;
		double distxyz;
		for (int itr = 0; itr < 1; itr++)
		{
			distvec.assign(pointsl.size(), DBL_MAX);

			//-----------------------------------------------------------------
			// Assign labels -> klabels
			//-----------------------------------------------------------------
			for (int n = 0; n < numk; n++)
			{
				int offset = (int)speed * adaptk[n] * step2;

				//2offset*2offset*2offset search window
				x1 = max(0.0, kseedsx[n] - offset);
				x2 = min((double)m_width, kseedsx[n] + offset);
				y1 = max(0.0, kseedsy[n] - offset);
				y2 = min((double)m_height, kseedsy[n] + offset);
				z1 = max(0.0, kseedsz[n] - offset);
				z2 = min((double)m_frame, kseedsz[n] + offset);

				for (int i = 0; i < pointsl.size(); i++){
					if (pointsx[i] >= x1 && pointsx[i] < x2 && pointsy[i] >= y1 && pointsy[i] < y2 && pointsz[i] >= z1 && pointsz[i] < z2){
						l = pointsl[i];
						a = pointsa[i];
						b = pointsb[i];
						x = pointsx[i];
						y = pointsy[i];
						z = pointsz[i];

						dist = (l - kseedsl[n])*(l - kseedsl[n]) +
							(a - kseedsa[n])*(a - kseedsa[n]) +
							(b - kseedsb[n])*(b - kseedsb[n]);

						distxyz = (x - kseedsx[n])*(x - kseedsx[n]) +
							(y - kseedsy[n])*(y - kseedsy[n]) +
							(z - kseedsz[n])*(z - kseedsz[n]) * timedist * timedist;

						dist += distxyz * invwt;

						if (dist < distvec[i]){
							distvec[i] = dist;
							klabels[i] = n; //label of voxel i is the id of current seed
						}
					}
				}
			}

			/*int invalid = 0;
			for (int i = 0; i < pointsl.size(); i++) {
			if (klabels[i] == -1) {
			invalid += 1;
			}
			}
			cout << "Invalid: " << invalid << " in " << pointsl.size() << endl;*/
			int invalid = 0;
			for (int i = 0; i < pointsl.size(); i++) {
				if (klabels[i] == -1) {
					invalid += 1;
					l = pointsl[i];
					a = pointsa[i];
					b = pointsb[i];
					x = pointsx[i];
					y = pointsy[i];
					z = pointsz[i];
					for (int n = 0; n < numk; n++)
					{
						dist = (l - kseedsl[n])*(l - kseedsl[n]) +
							(a - kseedsa[n])*(a - kseedsa[n]) +
							(b - kseedsb[n])*(b - kseedsb[n]);
						distxyz = (x - kseedsx[n])*(x - kseedsx[n]) +
							(y - kseedsy[n])*(y - kseedsy[n]) +
							(z - kseedsz[n])*(z - kseedsz[n]) * timedist * timedist;

						dist += distxyz * invwt;

						if (dist < distvec[i]){
							distvec[i] = dist;
							klabels[i] = n; //label of voxel i is the id of current seed
						}
					}
					if (klabels[i] == -1){
#ifdef _WIN32
						cout << i << " " << pointsl.size() << " " << numk << " labxyz "
							<< l << " " << a << " " << b << " " << x << " " << y << " " << z << " "
							<< _isnan(l) << " " << _isnan(a) << " " << _isnan(b) << " " << _isnan(x) << " " << _isnan(y) << " " << _isnan(z) << endl;
#else
						cout << i << " " << pointsl.size() << " " << numk << " labxyz "
							<< l << " " << a << " " << b << " " << x << " " << y << " " << z << " "
							<< isnan(l) << " " << isnan(a) << " " << isnan(b) << " " << isnan(x) << " " << isnan(y) << " " << isnan(z) << endl;
#endif
					}
				}
			}
			cout << "Invalid: " << invalid << " in " << pointsl.size() << endl;
		}

		for (int i = 0; i < pointsl.size(); i++){
			rvt[klabels[i]].push_back(i);// push_back points' ids
		}
	}

	//===========================================================================
	/// Llyod_Points (streaming)
	//===========================================================================
	void Lloyd_Points(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*&                       klabels,
		const int&                  STEP,
		const double&               M)
	{
		size_t numk = kseedsl.size();

		//-----------------------------------------------------------------
		// Lloyd
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		vector<double> sigmal(numk, 0);
		vector<double> sigmaa(numk, 0);
		vector<double> sigmab(numk, 0);
		vector<double> sigmax(numk, 0);
		vector<double> sigmay(numk, 0);
		vector<double> sigmaz(numk, 0);
		vector<double> clustersize(numk, 0);
		vector<double> inv(numk, 0);

		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		sigmaz.assign(numk, 0);
		clustersize.assign(numk, 0);
		inv.assign(numk, 0);

		for (int i = 0; i < pointsl.size(); i++)
		{
			if (klabels[i] == -1)
				continue;
			sigmal[klabels[i]] += pointsl[i];
			sigmaa[klabels[i]] += pointsa[i];
			sigmab[klabels[i]] += pointsb[i];
			sigmax[klabels[i]] += pointsx[i];
			sigmay[klabels[i]] += pointsy[i];
			sigmaz[klabels[i]] += pointsz[i];
			clustersize[klabels[i]] += 1.0;
		}

		for (int k = 0; k < numk; k++)
		{
			if (clustersize[k] <= 0) clustersize[k] = 1;
			inv[k] = 1.0 / clustersize[k];//computing inverse now to multiply, than divide later
		}

		for (int k = 0; k < numk; k++)
		{
			kseedsl[k] = sigmal[k] * inv[k];
			kseedsa[k] = sigmaa[k] * inv[k];
			kseedsb[k] = sigmab[k] * inv[k];
			kseedsx[k] = sigmax[k] * inv[k];
			kseedsy[k] = sigmay[k] * inv[k];
			kseedsz[k] = sigmaz[k] * inv[k];
		}
		cout << "# Kseeds: " << kseedsl.size() << endl;
	}

	//===========================================================================
	/// Llyod_Points (streaming)
	//===========================================================================
	void Lloyd_Points_Density(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*&                       klabels,
		const int&                  STEP,
		const double&               M,
		const int&					withcontrol = 0)
	{
		size_t numk = kseedsl.size();

		//-----------------------------------------------------------------
		// Lloyd
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		vector<double> sigmal(numk, 0);
		vector<double> sigmaa(numk, 0);
		vector<double> sigmab(numk, 0);
		vector<double> sigmax(numk, 0);
		vector<double> sigmay(numk, 0);
		vector<double> sigmaz(numk, 0);
		vector<double> clustersize(numk, 0);
		vector<double> inv(numk, 0);

		vector<double> clusteruni(numk, 0);
		vector<double> invuni(numk, 0);
		vector<double> sigmaluni(numk, 0);
		vector<double> sigmaauni(numk, 0);
		vector<double> sigmabuni(numk, 0);
		vector<double> sigmaxuni(numk, 0);
		vector<double> sigmayuni(numk, 0);
		vector<double> sigmazuni(numk, 0);

		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		sigmaz.assign(numk, 0);
		clustersize.assign(numk, 0);
		inv.assign(numk, 0);

		for (int i = 0; i < pointsl.size(); i++)
		{
			if (klabels[i] == -1)
				continue;
			sigmal[klabels[i]] += pointsl[i] * pointsdene[i];
			sigmaa[klabels[i]] += pointsa[i] * pointsdene[i];
			sigmab[klabels[i]] += pointsb[i] * pointsdene[i];
			sigmax[klabels[i]] += pointsx[i] * pointsdene[i];
			sigmay[klabels[i]] += pointsy[i] * pointsdene[i];
			sigmaz[klabels[i]] += pointsz[i] * pointsdene[i];
			clustersize[klabels[i]] += pointsdene[i]/*1.0*/;
			if (withcontrol){
				sigmaluni[klabels[i]] += pointsl[i];
				sigmaauni[klabels[i]] += pointsa[i];
				sigmabuni[klabels[i]] += pointsb[i];
				sigmaxuni[klabels[i]] += pointsx[i];
				sigmayuni[klabels[i]] += pointsy[i];
				sigmazuni[klabels[i]] += pointsz[i];
				clusteruni[klabels[i]] += 1.0;
			}
		}

		for (int k = 0; k < numk; k++)
		{
			if (clustersize[k] <= 0) clustersize[k] = 1;
			inv[k] = 1.0 / clustersize[k];//computing inverse now to multiply, than divide later
			if (withcontrol){
				if (clusteruni[k] <= 0) clusteruni[k] = 1;
				invuni[k] = 1.0 / clusteruni[k];
			}
		}

		if (!withcontrol){
			for (int k = 0; k < numk; k++)
			{
				kseedsl[k] = sigmal[k] * inv[k];
				kseedsa[k] = sigmaa[k] * inv[k];
				kseedsb[k] = sigmab[k] * inv[k];
				kseedsx[k] = sigmax[k] * inv[k];
				kseedsy[k] = sigmay[k] * inv[k];
				kseedsz[k] = sigmaz[k] * inv[k];
			}
		}
		else if (withcontrol == 1){//withcontrol == 1
			for (int k = 0; k < numk; k++)
			{
				sigmal[k] = sigmal[k] * inv[k];
				sigmaa[k] = sigmaa[k] * inv[k];
				sigmab[k] = sigmab[k] * inv[k];
				sigmax[k] = sigmax[k] * inv[k];
				sigmay[k] = sigmay[k] * inv[k];
				sigmaz[k] = sigmaz[k] * inv[k];
				sigmaluni[k] = sigmaluni[k] * invuni[k];
				sigmaauni[k] = sigmaauni[k] * invuni[k];
				sigmabuni[k] = sigmabuni[k] * invuni[k];
				sigmaxuni[k] = sigmaxuni[k] * invuni[k];
				sigmayuni[k] = sigmayuni[k] * invuni[k];
				sigmazuni[k] = sigmazuni[k] * invuni[k];
			}
			// new center without density: sigmaXuni
			// new center with density: sigmaX
			// old center: kseedsX
			int cnt1 = 0, cnt2 = 0;
			for (int k = 0; k < numk; k++)
			{
				double dist1 = (sigmaluni[k] - sigmal[k])*(sigmaluni[k] - sigmal[k]) +
					(sigmaauni[k] - sigmaa[k])*(sigmaauni[k] - sigmaa[k]) +
					(sigmabuni[k] - sigmab[k])*(sigmabuni[k] - sigmab[k]) +
					((sigmaxuni[k] - sigmax[k])*(sigmaxuni[k] - sigmax[k]) +
					(sigmayuni[k] - sigmay[k])*(sigmayuni[k] - sigmay[k]) +
					(sigmazuni[k] - sigmaz[k])*(sigmazuni[k] - sigmaz[k])*timedist*timedist) * invwt;
				double dist2 = (sigmaluni[k] - kseedsl[k])*(sigmaluni[k] - kseedsl[k]) +
					(sigmaauni[k] - kseedsa[k])*(sigmaauni[k] - kseedsa[k]) +
					(sigmabuni[k] - kseedsb[k])*(sigmabuni[k] - kseedsb[k]) +
					((sigmaxuni[k] - kseedsx[k])*(sigmaxuni[k] - kseedsx[k]) +
					(sigmayuni[k] - kseedsy[k])*(sigmayuni[k] - kseedsy[k]) +
					(sigmazuni[k] - kseedsz[k])*(sigmazuni[k] - kseedsz[k])*timedist*timedist) * invwt;
				if (dist1 <= dist2){
					kseedsl[k] = sigmal[k];
					kseedsa[k] = sigmaa[k];
					kseedsb[k] = sigmab[k];
					kseedsx[k] = sigmax[k];
					kseedsy[k] = sigmay[k];
					kseedsz[k] = sigmaz[k];
					cnt1++;
				}
				else{//dist1 > dist2
					double r = sqrt(dist2 / dist1);
					kseedsl[k] = r*sigmal[k] + (1 - r)*sigmaluni[k];
					kseedsa[k] = r*sigmaa[k] + (1 - r)*sigmaauni[k];
					kseedsb[k] = r*sigmab[k] + (1 - r)*sigmabuni[k];
					kseedsx[k] = r*sigmax[k] + (1 - r)*sigmaxuni[k];
					kseedsy[k] = r*sigmay[k] + (1 - r)*sigmayuni[k];
					kseedsz[k] = r*sigmaz[k] + (1 - r)*sigmazuni[k];
					cnt2++;
				}
			}
			cout << "# already in circle: " << cnt1 << ", # clamp to circle boundary: " << cnt2 << endl;
		}
		else if (withcontrol == 2){//withcontrol == 2
			// find the least likely to be boundary in circle
			for (int k = 0; k < numk; k++)
			{
				sigmal[k] = sigmal[k] * inv[k];
				sigmaa[k] = sigmaa[k] * inv[k];
				sigmab[k] = sigmab[k] * inv[k];
				sigmax[k] = sigmax[k] * inv[k];
				sigmay[k] = sigmay[k] * inv[k];
				sigmaz[k] = sigmaz[k] * inv[k];
				sigmaluni[k] = sigmaluni[k] * invuni[k];
				sigmaauni[k] = sigmaauni[k] * invuni[k];
				sigmabuni[k] = sigmabuni[k] * invuni[k];
				sigmaxuni[k] = sigmaxuni[k] * invuni[k];
				sigmayuni[k] = sigmayuni[k] * invuni[k];
				sigmazuni[k] = sigmazuni[k] * invuni[k];
			}
			// new center without density: sigmaXuni
			// new center with density: sigmaX
			// old center: kseedsX
			int cnt1 = 0, cnt2 = 0, cnt3 = 0;
			for (int k = 0; k < numk; k++)
			{
				double dist1 = (sigmaluni[k] - sigmal[k])*(sigmaluni[k] - sigmal[k]) +
					(sigmaauni[k] - sigmaa[k])*(sigmaauni[k] - sigmaa[k]) +
					(sigmabuni[k] - sigmab[k])*(sigmabuni[k] - sigmab[k]) +
					((sigmaxuni[k] - sigmax[k])*(sigmaxuni[k] - sigmax[k]) +
					(sigmayuni[k] - sigmay[k])*(sigmayuni[k] - sigmay[k]) +
					(sigmazuni[k] - sigmaz[k])*(sigmazuni[k] - sigmaz[k])*timedist*timedist) * invwt;
				double dist2 = (sigmaluni[k] - kseedsl[k])*(sigmaluni[k] - kseedsl[k]) +
					(sigmaauni[k] - kseedsa[k])*(sigmaauni[k] - kseedsa[k]) +
					(sigmabuni[k] - kseedsb[k])*(sigmabuni[k] - kseedsb[k]) +
					((sigmaxuni[k] - kseedsx[k])*(sigmaxuni[k] - kseedsx[k]) +
					(sigmayuni[k] - kseedsy[k])*(sigmayuni[k] - kseedsy[k]) +
					(sigmazuni[k] - kseedsz[k])*(sigmazuni[k] - kseedsz[k])*timedist*timedist) * invwt;
				if (dist1 <= dist2){
					kseedsl[k] = sigmal[k];
					kseedsa[k] = sigmaa[k];
					kseedsb[k] = sigmab[k];
					kseedsx[k] = sigmax[k];
					kseedsy[k] = sigmay[k];
					kseedsz[k] = sigmaz[k];
					cnt1++;
				}
				else{//dist1 > dist2
					//double minvol = 1e30;//boundary, volume big; not boundary, volume small
					double maxvol = -1e30;
					int mindex = -1;
					double mdist = -1;
					for (int i = 0; i < rvt[k].size(); i++)
					{
						double dist = (sigmaluni[k] - pointsl[rvt[k][i]])*(sigmaluni[k] - pointsl[rvt[k][i]]) +
							(sigmaauni[k] - pointsa[rvt[k][i]])*(sigmaauni[k] - pointsa[rvt[k][i]]) +
							(sigmabuni[k] - pointsb[rvt[k][i]])*(sigmabuni[k] - pointsb[rvt[k][i]]) +
							((sigmaxuni[k] - pointsx[rvt[k][i]])*(sigmaxuni[k] - pointsx[rvt[k][i]]) +
							(sigmayuni[k] - pointsy[rvt[k][i]])*(sigmayuni[k] - pointsy[rvt[k][i]]) +
							(sigmazuni[k] - pointsz[rvt[k][i]])*(sigmazuni[k] - pointsz[rvt[k][i]])*timedist*timedist) * invwt;
						//cout << "dist: " << dist << ", dist2: " << dist2 << endl;
						if (dist > dist2) continue;
						if (pointsdene[rvt[k][i]] > maxvol){
							maxvol = pointsdene[rvt[k][i]];
							mindex = rvt[k][i];
							mdist = dist;
						}
					}
					//cout << "rvt[k]: " << rvt[k].size() << ", clusteruni[k]: " << clusteruni[k] << ", mindex: " << mindex << ", dist: " << mdist << ", dist1: " << dist1 << ", dist2: " << dist2 << endl;
					if (mindex != -1){
						kseedsl[k] = pointsl[mindex];
						kseedsa[k] = pointsa[mindex];
						kseedsb[k] = pointsb[mindex];
						kseedsx[k] = pointsx[mindex];
						kseedsy[k] = pointsy[mindex];
						kseedsz[k] = pointsz[mindex];
						cnt2++;
					}
					else{//can't find one
						double r = sqrt(dist2 / dist1);
						kseedsl[k] = r*sigmal[k] + (1 - r)*sigmaluni[k];
						kseedsa[k] = r*sigmaa[k] + (1 - r)*sigmaauni[k];
						kseedsb[k] = r*sigmab[k] + (1 - r)*sigmabuni[k];
						kseedsx[k] = r*sigmax[k] + (1 - r)*sigmaxuni[k];
						kseedsy[k] = r*sigmay[k] + (1 - r)*sigmayuni[k];
						kseedsz[k] = r*sigmaz[k] + (1 - r)*sigmazuni[k];
						cnt3++;
					}
				}
			}
			cout << "# already in circle: " << cnt1
				<< ", # find least volume in circle: " << cnt2
				<< ", # clamp to circle boundary: " << cnt3 << endl;
		}
		cout << "# Kseeds: " << kseedsl.size() << endl;
	}

	//===========================================================================
	/// AssignLabels_ForClip (Not Used)
	//===========================================================================
	void AssignLabels_ForClip(
		const int                   frame_id_start,
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*&                       klabels,
		const int                   STEP,
		vector<double>&             adaptk,
		const double&               M,
		const int&                  speed)
	{
		size_t numk = kseedsl.size();
		invwt = 1.0 / ((STEP / M)*(STEP / M));//D^2 = dc^2 + (Nc/Ns)^2 * ds^2. here invwt = Nc/Ns = M / STEP.
		//M large -> more compact, M small -> better boundary recall

		int sz = m_width * m_height * m_frame;
		vector<double> distvec(sz, DBL_MAX);

		int x1, y1, x2, y2, z1, z2;
		double l, a, b;
		double dist;
		double distxyz;
		for (int itr = 0; itr < 1; itr++)
		{
			distvec.assign(sz, DBL_MAX);

			//-----------------------------------------------------------------
			// Assign labels -> klabels
			//-----------------------------------------------------------------
			for (int n = 0; n < numk; n++)
			{
				int offset = (int)speed*adaptk[n] * STEP;

				//2offset*2offset*2offset search window
				x1 = max(0.0, kseedsx[n] - offset);
				x2 = min((double)m_width, kseedsx[n] + offset);
				y1 = max(0.0, kseedsy[n] - offset);
				y2 = min((double)m_height, kseedsy[n] + offset);
				z1 = max(0.0, kseedsz[n] - frame_id_start - offset);
				z2 = min((double)m_frame, kseedsz[n] - frame_id_start + offset);

				for (int z = z1; z < z2; z++)
				{
					for (int y = y1; y < y2; y++)
					{
						for (int x = x1; x < x2; x++)
						{
							int i = y*m_width + x + z*(m_width*m_height);

							l = m_lvec[i];
							a = m_avec[i];
							b = m_bvec[i];

							dist = (l - kseedsl[n])*(l - kseedsl[n]) +
								(a - kseedsa[n])*(a - kseedsa[n]) +
								(b - kseedsb[n])*(b - kseedsb[n]);

							distxyz = (x - kseedsx[n])*(x - kseedsx[n]) +
								(y - kseedsy[n])*(y - kseedsy[n]) +
								(z - kseedsz[n] + frame_id_start)*(z - kseedsz[n] + frame_id_start) * timedist * timedist;

							dist += distxyz * invwt;

							if (dist < distvec[i])
							{
								distvec[i] = dist;
								klabels[i + frame_id_start*m_width*m_height] = n; //label of voxel i is the id of current seed
							}
						}
					}
				}
			}
		}

		/*int invalid = 0;
		for (int i = frame_id_start*m_width*m_height; i < frame_id_start*m_width*m_height+sz; i++) {
		if (klabels[i] == -1) {
		invalid += 1;
		}
		}
		cout << "Invalid: " << invalid << " in " << sz << endl;*/
	}

	//===========================================================================
	/// AssignLabels_ForClip2
	//===========================================================================
	void AssignLabels_ForClip2(
		const int                   frame_id_start,
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*&                       klabels,
		const int                   STEP,
		vector<double>&             adaptk,
		const double&               M,
		const int&                  speed)
	{
		size_t numk = kseedsl.size();
		invwt = 1.0 / ((STEP / M)*(STEP / M));//D^2 = dc^2 + (Nc/Ns)^2 * ds^2. here invwt = Nc/Ns = M / STEP.
		//M large -> more compact, M small -> better boundary recall

		int sz = m_width * m_height * m_frame;
		vector<double> distvec(sz, DBL_MAX);
		for (int i = 0; i < sz; i++) klabels[i] = -1;

		int x1, y1, x2, y2, z1, z2;
		double l, a, b;
		double dist;
		double distxyz;
		for (int itr = 0; itr < 1; itr++)
		{
			distvec.assign(sz, DBL_MAX);

			//-----------------------------------------------------------------
			// Assign labels -> klabels
			//-----------------------------------------------------------------
			for (int n = 0; n < numk; n++)
			{
				int offset = (int)speed*adaptk[n] * STEP;

				//2offset*2offset*2offset search window
				x1 = max(0.0, kseedsx[n] - offset);
				x2 = min((double)m_width, kseedsx[n] + offset);
				y1 = max(0.0, kseedsy[n] - offset);
				y2 = min((double)m_height, kseedsy[n] + offset);
				z1 = max(0.0, kseedsz[n] - frame_id_start - offset);
				z2 = min((double)m_frame, kseedsz[n] - frame_id_start + offset);
				//here m_frame = frame_clip, z + frame_id_start in range [kseedz[n]-offset, kseedz[n]+offset]

				for (int z = z1; z < z2; z++)
				{
					for (int y = y1; y < y2; y++)
					{
						for (int x = x1; x < x2; x++)
						{
							int i = y*m_width + x + z*(m_width*m_height);

							l = m_lvec[i];
							a = m_avec[i];
							b = m_bvec[i];

							dist = (l - kseedsl[n])*(l - kseedsl[n]) +
								(a - kseedsa[n])*(a - kseedsa[n]) +
								(b - kseedsb[n])*(b - kseedsb[n]);

							distxyz = (x - kseedsx[n])*(x - kseedsx[n]) +
								(y - kseedsy[n])*(y - kseedsy[n]) +
								(z - kseedsz[n] + frame_id_start)*(z - kseedsz[n] + frame_id_start) * timedist * timedist;

							dist += distxyz * invwt;

							if (dist < distvec[i])
							{
								distvec[i] = dist;
								klabels[i] = n; //label of voxel i is the id of current seed
							}
						}
					}
				}
			}
		}
		
		int invalid = 0;
		for (int i = 0; i < sz; i++) {
			if (klabels[i] == -1) {
				invalid += 1;
				for (int n = 0; n < numk; n++)
				{
					l = m_lvec[i];
					a = m_avec[i];
					b = m_bvec[i];
					int x = i % m_width;
					int y = (i / m_width) % m_height;
					int z = i % (m_width * m_height);

					dist = (l - kseedsl[n])*(l - kseedsl[n]) +
						(a - kseedsa[n])*(a - kseedsa[n]) +
						(b - kseedsb[n])*(b - kseedsb[n]);
					distxyz = (x - kseedsx[n])*(x - kseedsx[n]) +
						(y - kseedsy[n])*(y - kseedsy[n]) +
						(z - kseedsz[n] + frame_id_start)*(z - kseedsz[n] + frame_id_start) * timedist * timedist;
					
					dist += distxyz * invwt;

					if (dist < distvec[i])
					{
						distvec[i] = dist;
						klabels[i] = n; //label of voxel i is the id of current seed
					}
				}
			}
		}
		cout << "Invalid: " << invalid << " in " << sz << endl;

	}

	void Merge2InPlace(
		const int&					itr,
		const int*                  klabels,
		int*&                       newlabels,
		map<int, int>&				hashtable,
		int&						numlabels,
		const int&					choice,
		const int&					c_merge)
	{
		int sz = m_width * m_height * m_frame;
		for (int i = 0; i < sz; i++) newlabels[i] = -1;

		hashtable[-1] = 0;
		int cnt = 0;
		int curLabel;
			
		if (lastFrameLabel != NULL){
			for (int y = 0; y < m_height; y++) {
				for (int x = 0; x < m_width; x++) {
					int z = 0;
					int idx = y*m_width + x;
					if (newlabels[idx] >= 0)
						continue;
					if (lastFrameLabel[idx] == klabels[idx]){
						curLabel = lastFrameLabel[idx + m_width * m_height];
						addcolor = false;
					}else
						continue;

					newlabels[idx] = curLabel;

					processIter(x, y, z, cnt, itr, klabels, newlabels, hashtable, choice, c_merge);
					
				}
			}
		}
		for (int z = 0; z < m_frame; z++) {
			for (int y = 0; y < m_height; y++) {
				for (int x = 0; x < m_width; x++) {
					int idx = z*m_width*m_height + y*m_width + x;
					if (newlabels[idx] >= 0)
						continue;
					if (hashtable.find(klabels[idx]) == hashtable.end()){
						curLabel = klabels[idx];
						addcolor = false;
					}else{
						curLabel = rvec.size();
						addcolor = true;
					}
					
					newlabels[idx] = curLabel;

					processIter(x, y, z, cnt, itr, klabels, newlabels, hashtable, choice, c_merge);
				}
			}
		}
		cout << "# Kseeds: " << kseedsl.size() << endl;
		cout << "After Merge: " << rvec.size() << endl;
		numlabels = cnt;

		if (lastFrameLabel == NULL){
			lastFrameLabel = new int[m_width * m_height * 2];
		}
		for (int y = 0; y < m_height; y++) {
			for (int x = 0; x < m_width; x++) {
				int idx = y * m_width + x;
				int idx2 = idx + (m_frame - 1) * m_width * m_height;
				lastFrameLabel[idx] = klabels[idx2];
				lastFrameLabel[idx + m_width * m_height] = newlabels[idx2];
			}
		}
	}

	void processIter(
		int&						x,
		int&						y,
		int&						z,
		int&						cnt,
		const int&					itr,
		const int*                  klabels,
		int*&                       newlabels,
		map<int, int>&				hashtable,
		const int&					choice,
		const int&					c_merge)
	{
		const int dx[26] = { 0, -1, 0, 1, 0, 0, -1, 1, -1, 1,
			-1, 0, 1, 0, -1, 0, 1, 0,
			-1, 1, -1, 1, -1, 1, -1, 1 };
		const int dy[26] = { 0, 0, -1, 0, 1, 0, -1, -1, 1, 1,
			0, -1, 0, 1, 0, -1, 0, 1,
			-1, -1, 1, 1, -1, -1, 1, 1 };
		const int dz[26] = { -1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			-1, -1, -1, -1, 1, 1, 1, 1,
			-1, -1, -1, -1, 1, 1, 1, 1 };

		int neighbornum = 6;
		if (choice == 1){//face- adjacent voxels
			neighbornum = 6;
		}
		else if (choice == 2){//same frame: face-, edge- adjacent voxels
			//diff frame: face- adjacent voxels
			neighbornum = 10;
		}
		else if (choice == 3){//face-, edge- adjacent voxels
			neighbornum = 18;
		}
		else if (choice == 4){//face-, edge-, vertex- adjacent voxels
			neighbornum = 26;
		}
		//cout << "connectivity chosen: " << neighbornum << endl;

		double merge_diflab = DBL_MAX;
		if (choice == 5){
			merge_diflab = 4;
		}
		int idx = z*m_width*m_height + y*m_width + x;

		const int cnt_threshold = m_step * m_step * m_step;
		const int cnt_threshold_merge = m_step * m_step * m_step / double(c_merge); // 3; // 2;

		vector<int> xvec;
		vector<int> yvec;
		vector<int> zvec;

		int nx, ny, nz;
		double currentl, currenta, currentb;
		double adjl = 0;
		double adja = 0;
		double adjb = 0;
		double diflab = 0;
		int curLabel;
		// find ADJACENT label node
		int adjlabel = -1;
		for (int n = 0; n < neighbornum; n++){
			int ax = x + dx[n];
			int ay = y + dy[n];
			int az = z + dz[n];

			if (ax >= 0 && ax < m_width && ay >= 0 && ay < m_height && az >= 0 && az < m_frame){
				int aidx = az*m_width*m_height + ay*m_width + ax;
				if (newlabels[aidx] >= 0 && newlabels[aidx] != newlabels[idx]){
					adjlabel = newlabels[aidx];
					adjl = kseedsl[klabels[aidx]];
					adja = kseedsa[klabels[aidx]];
					adjb = kseedsb[klabels[aidx]];
				}
			}
		}
		currentl = kseedsl[klabels[idx]];
		currenta = kseedsa[klabels[idx]];
		currentb = kseedsb[klabels[idx]];
		if (adjlabel >= 0){
			diflab = sqrt((currentl - adjl) * (currentl - adjl) +
				(currenta - adja) * (currenta - adja) +
				(currentb - adjb) * (currentb - adjb));
			//cout << diflab << endl;
		}

		// find voxels belong to SAME label node (in the same frame)
		// Similar to EnforceConnectivity for Superpixel
		xvec.clear(); yvec.clear(); zvec.clear();
		xvec.push_back(x); yvec.push_back(y); zvec.push_back(z);
		for (int c = 0; c < xvec.size(); c++)
		{
			for (int n = 0; n < neighbornum; n++)
			{
				nx = xvec[c] + dx[n];
				ny = yvec[c] + dy[n];
				nz = zvec[c] + dz[n];
				if ((nx >= 0 && nx < m_width) && (ny >= 0 && ny < m_height) && (nz >= 0 && nz < m_frame))
				{
					int nidx = nz*m_width*m_height + ny*m_width + nx;
					if (newlabels[nidx] < 0 && klabels[idx] == klabels[nidx])
					{
						xvec.push_back(nx);
						yvec.push_back(ny);
						zvec.push_back(nz);
						newlabels[nidx] = newlabels[idx];
					}
				}
			}
		}
		hashtable[curLabel] = xvec.size();
		int count = xvec.size();

		if (adjlabel < 0){
			cnt++;
			if (addcolor) addColor();
			return;
		}

		bool flag = false;
		if (choice <= 3){
			if (itr == iteration - 1){
				if (count <= cnt_threshold / 16) //16 //8
					flag = true;
			}
			else{
				if (count <= cnt_threshold_merge)// && count + hashtable[adjlabel] <= cnt_threshold){
					flag = true;
			}
		}
		else{
			if (itr == iteration - 1){
				if (count <= cnt_threshold / 8)
					flag = true;
			}
			else{
				if (count <= cnt_threshold / 8 ||
					(diflab < merge_diflab && count + hashtable[adjlabel] <= 3 * cnt_threshold))
					flag = true;
			}
		}
		if (flag){
			hashtable[adjlabel] += count;
			for (int c = 0; c < count; c++){
				int nidx = zvec[c] * m_width*m_height + yvec[c] * m_width + xvec[c];
				newlabels[nidx] = adjlabel;
			}
			cnt--;
		}
		else{
			if (addcolor) addColor();
		}
		cnt++;
	}
};