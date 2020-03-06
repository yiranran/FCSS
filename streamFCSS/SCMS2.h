#pragma once
#include "CSS.h"

//#define OUTPUT_PICK_INFO

//#define VOL      // use vol or num to get dense and sparse
//#define CHECK1   // use rvt or rand partition to check feasibility

class SCMS2 : public CSS{
public:
	SCMS2(bool pickmode, double timedist, int farthestInit, int iteration, bool ifdraw, int choice)
		: CSS(pickmode, timedist, farthestInit, iteration, ifdraw)
	{
		this->choice = choice;
	}

	// Get from clip
	vector<double>                  pointsl;
	vector<double>                  pointsa;
	vector<double>                  pointsb;
	vector<double>                  pointsx;
	vector<double>                  pointsy;
	vector<double>                  pointsz;
	vector<double>					pointsvox;
	vector<double>					pointsvol;
	vector<double>					pointsdene;

	// Used for split and merge
	int stage;
	int choice; // to choose which connectivity
	std::vector<int> dense;
	std::vector<int> sparse;
	double sm[6];//x y z l a b
	double si[6];
	double sj[6];
	double sk[6];
	double sp[6];
	double sq[6];
	double avg_mass;
	int r1, r2;
	std::vector<int> spv;
	std::vector<int> sqv;

	//=====================================================================================
	//								STRICT S-M
	//=====================================================================================
	double computeVolume(int seedId){
		double vol = 0;
		for (int i = 0; i < rvt[seedId].size(); i++){
			vol += m_volvec[rvt[seedId][i]];
		}
		return vol;
	}

	void ComputeSparseDense()
	{
		dense.clear();
		sparse.clear();

		if (stage == 1){
#ifdef VOL
			double avg_vol = total_vol / kseedsl.size();
			//double sum_vol = 0;
			double coef = 1.5; // 2 // 4
			for (int i = 0; i < kseedsl.size(); i++){
				double vol = computeVolume(i);
				if (vol > coef * avg_vol){
					dense.push_back(i);
				}
				else if (vol != 0 && vol < avg_vol / coef){
					sparse.push_back(i);
				}
				//sum_vol += vol;
			}
			//std::cout << "sum_vol: " << sum_vol << " total_vol: " << total_vol << std::endl; //checked
#else
			avg_mass = (double)(m_width * m_height * m_frame) / kseedsl.size();
			double coef = 2; // 1.5(good) // 1.25 // 1.5 // 2 // 4
			for (int i = 0; i < kseedsl.size(); i++){
				double mass = rvt[i].size();
				if (mass > coef * avg_mass){
					dense.push_back(i);
				}
				else if (mass != 0 && mass < avg_mass / coef){
					sparse.push_back(i);
				}
			}
#endif
		}
		else{
#ifdef VOL
			double avg_vol = total_vol / kseedsl.size();
			double coef = 1.5; // 2 // 4
			for (int i = 0; i < kseedsl.size(); i++){
				double vol = 0;
				for (int j = 0; j < rvt[i].size(); j++)
					vol += pointsvol[rvt[i][j]];
				if (vol > coef * avg_vol){
					dense.push_back(i);
				}
				else if (vol != 0 && vol < avg_vol / coef){
					sparse.push_back(i);
				}
			}
#else
			avg_mass = (double)(m_width * m_height * m_frame) / kseedsl.size();
			double coef = 2; // 1.5(good) // 1.25 // 1.5 // 2 // 4
			for (int i = 0; i < kseedsl.size(); i++){
				double mass = 0;
				for (int j = 0; j < rvt[i].size(); j++)
					mass += pointsvox[rvt[i][j]];
				if (mass > coef * avg_mass){
					dense.push_back(i);
				}
				else if (mass != 0 && mass < avg_mass / coef){
					sparse.push_back(i);
				}
			}
#endif
		}

		std::cout << "dense size: " << dense.size() << std::endl;
		std::cout << "sparse size: " << sparse.size() << std::endl;
	}

	bool HeuristicPick(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int &						toSplit,
		int &						toMerge1,
		int &						toMerge2)
	{
		size_t numk = kseedsl.size();
		double x, y, z, l, a, b;

		if (dense.size() > 2){
			r1 = (int)((dense.size()-1) * ((double)rand() / RAND_MAX));//[0,dense.size()-1]
			toSplit = dense[r1];
		}
		else{
			toSplit = (int)((numk-1) * ((double)rand() / RAND_MAX));//[0,numk-1]
			r1 = -1;
		}
#ifdef OUTPUT_PICK_INFO
		std::cout << "tosplit: " << toSplit << std::endl;
#endif
		if (rvt[toSplit].size() == 0)
			return false;

		if (sparse.size() > 4){//at least 2 pairs
			r2 = (int)((sparse.size()-1) * ((double)rand() / RAND_MAX));//[0,sparse.size()-1]
			toMerge1 = sparse[r2];
#ifdef OUTPUT_PICK_INFO
			std::cout << "r: " << r2 << std::endl;
#endif
			x = kseedsx[toMerge1];
			y = kseedsy[toMerge1];
			z = kseedsz[toMerge1];
			l = kseedsl[toMerge1];
			a = kseedsa[toMerge1];
			b = kseedsb[toMerge1];
			toMerge2 = -1;

			double mindist = 1e30;
			double dist, distxyz;
			/*
			for(int i = 0; i < sparse.size(); i++){
				if(sparse[i] == toMerge1 || sparse[i] == toSplit) continue;
				dist = (l - kseedsl[sparse[i]])*(l - kseedsl[sparse[i]]) +
				(a - kseedsa[sparse[i]])*(a - kseedsa[sparse[i]]) +
				(b - kseedsb[sparse[i]])*(b - kseedsb[sparse[i]]);
				distxyz = (x - kseedsx[sparse[i]])*(x - kseedsx[sparse[i]]) +
				(y - kseedsy[sparse[i]])*(y - kseedsy[sparse[i]]) +
				(z - kseedsz[sparse[i]])*(z - kseedsz[sparse[i]]) * timedist * timedist;
				dist += distxyz * invwt;
				if(dist < mindist){
					mindist = dist;
					toMerge2 = sparse[i];
				}
			}
			*/
			for (int i = 0; i < numk; i++){
				if (i == toMerge1 || i == toSplit) continue;
				//if(rvt[i].size() > 2 * avg_mass) continue;
				dist = (l - kseedsl[i])*(l - kseedsl[i]) +
					(a - kseedsa[i])*(a - kseedsa[i]) +
					(b - kseedsb[i])*(b - kseedsb[i]);
				distxyz = (x - kseedsx[i])*(x - kseedsx[i]) +
					(y - kseedsy[i])*(y - kseedsy[i]) +
					(z - kseedsz[i])*(z - kseedsz[i]) * timedist * timedist;
				dist += distxyz * invwt;
				if (dist < mindist){
					mindist = dist;
					toMerge2 = i;
				}
			}
#ifdef OUTPUT_PICK_INFO
			std::cout << "minDist: " << mindist << " toMerge2: " << toMerge2 << std::endl;
#endif
		}
		else{
			r2 = -1;
			toMerge1 = (int)((numk-1) * ((double)rand() / RAND_MAX));//[0,numk-1]
			x = kseedsx[toMerge1];
			y = kseedsy[toMerge1];
			z = kseedsz[toMerge1];
			l = kseedsl[toMerge1];
			a = kseedsa[toMerge1];
			b = kseedsb[toMerge1];
			toMerge2 = -1;
			if (rvt[toMerge1].size() == 0)
				return false;
			double mindist = 1e30;
			double dist, distxyz;
			for (int i = 0; i < numk; i++){
				if (i == toMerge1 || i == toSplit) continue;
				dist = (l - kseedsl[i])*(l - kseedsl[i]) +
					(a - kseedsa[i])*(a - kseedsa[i]) +
					(b - kseedsb[i])*(b - kseedsb[i]);
				distxyz = (x - kseedsx[i])*(x - kseedsx[i]) +
					(y - kseedsy[i])*(y - kseedsy[i]) +
					(z - kseedsz[i])*(z - kseedsz[i]) * timedist * timedist;
				dist += distxyz * invwt;
				if (dist < mindist){
					mindist = dist;
					toMerge2 = i;
				}
			}
		}
#ifdef OUTPUT_PICK_INFO
		std::cout << "toMerge1: " << toMerge1 << " toMerge2: " << toMerge2 << std::endl;
#endif

		if (toMerge2 == -1 || rvt[toMerge2].size() == 0)
			return false;
		else
			return true;

	}

	bool RandPick(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int &						toSplit,
		int &						toMerge1,
		int &						toMerge2)
	{
		size_t numk = kseedsl.size();
		// (toMerge1, toMerge2) -> toMerge2
		// toSplit -> (toSplit, toMerge1)
		double x, y, z, l, a, b;

		toSplit = (int)((numk-1) * ((double)rand() / RAND_MAX));//[0,numk-1]
#ifdef OUTPUT_PICK_INFO
		std::cout << "toSplit: " << toSplit << std::endl;
#endif
		if (rvt[toSplit].size() == 0)
			return false;

		toMerge1 = (int)((numk-1) * ((double)rand() / RAND_MAX));//[0,numk-1]
#ifdef OUTPUT_PICK_INFO
		std::cout << "toMerge1: " << toMerge1 << std::endl;
#endif
		if (rvt[toMerge1].size() == 0)
			return false;

		x = kseedsx[toMerge1];
		y = kseedsy[toMerge1];
		z = kseedsz[toMerge1];
		l = kseedsl[toMerge1];
		a = kseedsa[toMerge1];
		b = kseedsb[toMerge1];
		toMerge2 = -1;

		double mindist = 1e30;
		double dist, distxyz;
		for (int i = 0; i < numk; i++){
			if (i == toMerge1 || i == toSplit) continue;
			dist = (l - kseedsl[i])*(l - kseedsl[i]) +
				(a - kseedsa[i])*(a - kseedsa[i]) +
				(b - kseedsb[i])*(b - kseedsb[i]);
			distxyz = (x - kseedsx[i])*(x - kseedsx[i]) +
				(y - kseedsy[i])*(y - kseedsy[i]) +
				(z - kseedsz[i])*(z - kseedsz[i]) * timedist * timedist;
			dist += distxyz * invwt;
			if (dist < mindist){
				mindist = dist;
				toMerge2 = i;
			}
		}

		if (toMerge2 == -1 || rvt[toMerge2].size() == 0)
			return false;
		else
			return true;
	}


	void getContent(int k, double *content){
		if (stage == 1){
			content[0] = k % m_width;
			content[1] = (k / m_width) % m_height;
			content[2] = k / (m_width * m_height);
			content[3] = m_lvec[k];
			content[4] = m_avec[k];
			content[5] = m_bvec[k];
		}
		else{
			content[0] = pointsx[k];
			content[1] = pointsy[k];
			content[2] = pointsz[k];
			content[3] = pointsl[k];
			content[4] = pointsa[k];
			content[5] = pointsb[k];
		}
	}
	double dist(double *a, double *b){
		double sum = 0;
		for (int i = 0; i < 3; i++)
			sum += (a[i] - b[i]) * (a[i] - b[i]) * invwt;//xyz
		for (int i = 3; i < 6; i++)
			sum += (a[i] - b[i]) * (a[i] - b[i]);
		return sum;
	}
	double dist2(int k, int k2){
		double a[6];
		double b[6];
		getContent(k, a);
		getContent(k2, b);
		return dist(a, b);
	}
	double dist3(int k, double *b){
		double a[6];
		getContent(k, a);
		return dist(a, b);
	}
	void computeCentroid(int seedId, double *centroid, bool usedensity)
	{
		for (int n = 0; n < 6; n++)
			centroid[n] = 0;
		double kv[6];
		double sum_density = 0;
		double densityE;
		for (int i = 0; i < rvt[seedId].size(); i++){
			int k = rvt[seedId][i];
			getContent(k, kv);
			if (stage == 1){
				if (usedensity)
					densityE = m_densityvec[k];
				else
					densityE = 1.0;
			}else{
				if (usedensity)
					densityE = pointsdene[k];
				else
					densityE = 1.0;
			}
			for (int n = 0; n < 6; n++)
				centroid[n] += kv[n] * densityE;
			sum_density += densityE;
		}
		for (int n = 0; n < 6; n++)
			centroid[n] /= sum_density/*rvt[seedId].size()*/;
	}
	void computeMergeCentroid(int seedId, int seedId2, double *centroid, bool usedensity)
	{
		for (int n = 0; n < 6; n++)
			centroid[n] = 0;
		double kv[6];
		//seedId
		double sum_density1 = 0;
		double densityE;
		for (int i = 0; i < rvt[seedId].size(); i++){
			int k = rvt[seedId][i];
			getContent(k, kv);
			if (stage == 1){
				if (usedensity)
					densityE = m_densityvec[k];
				else
					densityE = 1.0;
			}
			else{
				if (usedensity)
					densityE = pointsdene[k];
				else
					densityE = 1.0;
			}
			for (int n = 0; n < 6; n++)
				centroid[n] += kv[n] * densityE;
			sum_density1 += densityE;
		}
		//seedId2
		double sum_density2 = 0;
		for (int i = 0; i < rvt[seedId2].size(); i++){
			int k = rvt[seedId2][i];
			getContent(k, kv);
			if (stage == 1){
				if (usedensity)
					densityE = m_densityvec[k];
				else
					densityE = 1.0;
			}
			else{
				if (usedensity)
					densityE = pointsdene[k];
				else
					densityE = 1.0;
			}
			for (int n = 0; n < 6; n++)
				centroid[n] += kv[n] * densityE;
			sum_density2 += densityE;
		}
		for (int n = 0; n < 6; n++)
			centroid[n] /= (sum_density1 + sum_density2)/*(rvt[seedId].size() + rvt[seedId2].size())*/;
	}
	bool computeSplitCentroids(int seedId, double *p1, double *p2, double *centroid1, double *centroid2, bool usedensity)
	{
		for (int n = 0; n < 6; n++){
			centroid1[n] = 0;
			centroid2[n] = 0;
		}
		spv.clear(); spv.resize(0);
		sqv.clear(); sqv.resize(0);
		int n1 = 0, n2 = 0;
		double sum1 = 0, sum2 = 0;
		double densityE;
		double content[6];
		for (int i = 0; i < rvt[seedId].size(); i++){
			int k = rvt[seedId][i];
			getContent(k, content);
			if (stage == 1){
				if (usedensity)
					densityE = m_densityvec[k];
				else
					densityE = 1.0;
			}
			else{
				if (usedensity)
					densityE = pointsdene[k];
				else
					densityE = 1.0;
			}
			//std::cout << "kp1: " << dist3(k,p1) << " kp2: " << dist3(k,p2) << std::endl;
			if (dist3(k, p1) < dist3(k, p2)){
				for (int n = 0; n < 6; n++)
					centroid1[n] += content[n] * densityE;
				spv.push_back(k);
				n1++;
				sum1 += densityE;
			}
			else{
				for (int n = 0; n < 6; n++)
					centroid2[n] += content[n] * densityE;
				sqv.push_back(k);
				n2++;
				sum2 += densityE;
			}
		}
#ifdef OUTPUT_PICK_INFO
		std::cout << "n1: " << n1 << " n2: " << n2 << std::endl;
#endif
		if (n1 == 0 || n2 == 0)
			return false;

		for (int n = 0; n < 6; n++){
			centroid1[n] /= sum1/*n1*/;
			centroid2[n] /= sum2/*n2*/;
		}
		return true;
	}
	bool computeSplitCentroids2(int seedId, double *centroid1, double *centroid2, bool usedensity)
	{
		double min[6];
		double max[6];
		for (int n = 0; n < 6; n++){
			min[n] = 1e30;
			max[n] = -1e30;
		}
		double kv[6];
		for (int i = 0; i < rvt[seedId].size(); i++){
			int k = rvt[seedId][i];
			getContent(k, kv);
			for (int j = 0; j < 6; j++){
				if (kv[j] < min[j]){
					min[j] = kv[j];
				}
				if (kv[j] > max[j]){
					max[j] = kv[j];
				}
			}
		}
		for (int n = 0; n < 6; n++){
			centroid1[n] = 0;
			centroid2[n] = 0;
		}
		spv.clear(); spv.resize(0);
		sqv.clear(); sqv.resize(0);
		int n1 = 0, n2 = 0;
		double sum1 = 0, sum2 = 0;
		double densityE;
		double content[6];
		for (int i = 0; i < rvt[seedId].size(); i++){
			int k = rvt[seedId][i];
			getContent(k, content);
			if (stage == 1){
				if (usedensity)
					densityE = m_densityvec[k];
				else
					densityE = 1.0;
			}
			else{
				if (usedensity)
					densityE = pointsdene[k];
				else
					densityE = 1.0;
			}
			int c = 0;//(int)(5 * ((double)rand() / RAND_MAX));
			if (content[c] < (min[c] + max[c]) / 2.0){
				for (int n = 0; n < 6; n++)
					centroid1[n] += content[n] * densityE;
				spv.push_back(k);
				n1++;
				sum1 += densityE;
			}
			else{
				for (int n = 0; n < 6; n++)
					centroid2[n] += content[n] * densityE;
				sqv.push_back(k);
				n2++;
				sum2 += densityE;
			}
		}
#ifdef OUTPUT_PICK_INFO
		std::cout << "n1: " << n1 << " n2: " << n2 << std::endl;
#endif
		if (n1 == 0 || n2 == 0)
			return false;

		for (int n = 0; n < 6; n++){
			centroid1[n] /= sum1/*n1*/;
			centroid2[n] /= sum2/*n2*/;
		}
		return true;
	}
	double computeDiameter(int seedId, double *a, double *b)
	{
		//compute AABB
		double min[6];
		double max[6];
		//int minId[6];
		//int maxId[6];
		for (int n = 0; n < 6; n++){
			min[n] = 1e30;
			max[n] = -1e30;
			//minId[n] = -1;
			//maxId[n] = -1;
		}
		double kv[6];
		for (int i = 0; i < rvt[seedId].size(); i++){
			int k = rvt[seedId][i];
			getContent(k, kv);
			for (int j = 0; j < 6; j++){
				if (kv[j] < min[j]){
					min[j] = kv[j];
					//minId[j] = k;
				}
				if (kv[j] > max[j]){
					max[j] = kv[j];
					//maxId[j] = k;
				}
			}
		}
#ifdef OUTPUT_PICK_INFO
		std::cout << "AABB_max: ";
		for (int n = 0; n < 6; n++)
			std::cout << max[n] << " ";
		std::cout << std::endl;
		std::cout << "AABB_min: ";
		for (int n = 0; n < 6; n++)
			std::cout << min[n] << " ";
		std::cout << std::endl;
		//std::cout << "AABB_max_id: ";
		//for (int n = 0; n < 6; n++)
		//	std::cout << maxId[n] << " ";
		//std::cout << std::endl;
		//std::cout << "AABB_min_id: ";
		//for (int n = 0; n < 6; n++)
		//	std::cout << minId[n] << " ";
		//std::cout << std::endl;
#endif
		/*
		//choose two supporting points as p1, p2
		double maxDist = 0;
		int k = -1, k2 = -1;
		for(int n = 0; n < 6; n++){
			if(maxId[n] != -1 && minId[n] != -1){

				//double d = (max[n] - min[n]) * (max[n] - min[n]);
				//if(n < 3)//xyz
				//	d *= invwt;

				double d = dist2(maxId[n], minId[n]);
				if(d > maxDist){
					maxDist = d;
					k = maxId[n];
					k2 = minId[n];
					if(DEBUG) std::cout << n << " ";
				}
			}
		}
		if(DEBUG) std::cout << std::endl;
		//std::cout << "maxDist: " << maxDist << std::endl;
		getContent(k, a);
		getContent(k2, b);

		if(DEBUG){
			std::cout << "p1: (id: " << k << ") ";
			for(int n = 0; n < 6; n++)
				std::cout << a[n] << " ";
			std::cout << std::endl;
			std::cout << "p2: (id: " << k2 << ") ";
			for(int n = 0; n < 6; n++)
				std::cout << b[n] << " ";
			std::cout << std::endl;
		}
		return maxDist;
		*/

		//choose two points nearest to min[6] and max[6] as p1, p2
		double minDist1 = 1e30;
		double minDist2 = 1e30;
		int k1 = -1, k2 = -1;
		for (int i = 0; i < rvt[seedId].size(); i++){
			double d1 = dist3(rvt[seedId][i], max);
			double d2 = dist3(rvt[seedId][i], min);
			if (minDist1 > d1){
				minDist1 = d1;
				k1 = i;
			}
			if (minDist2 > d2){
				minDist2 = d2;
				k2 = i;
			}
		}

		getContent(k1, a);
		getContent(k2, b);
#ifdef OUTPUT_PICK_INFO
		std::cout << "k1: " << k1 << " k2: " << k2 << std::endl;
		std::cout << "k1_p: ";
		for (int n = 0; n < 6; n++)
			std::cout << a[n] << " ";
		std::cout << std::endl;
		std::cout << "k2_p: ";
		for (int n = 0; n < 6; n++)
			std::cout << b[n] << " ";
		std::cout << std::endl;
		std::cout << "maxDist: " << dist(a, b) << std::endl;
#endif

		return dist(a, b);
	}

	bool CheckFeasibility(
		int&                        toSplit,
		int&						toMerge1,
		int&						toMerge2,
		const bool&					usedensity = true)
	{
		//compute sm, si, sj
		computeCentroid(toSplit, sm, usedensity);
		computeCentroid(toMerge1, si, usedensity);
		computeCentroid(toMerge2, sj, usedensity);

		//compute sp, sq

#ifdef CHECK1
		//two rvc of p1, p2 -- what's strange is most times arbitrary partition is better
		double p1[6];
		double p2[6];
		computeDiameter(toSplit, p1, p2);
		if (!computeSplitCentroids(toSplit, p1, p2, sp, sq, usedensity))
			return false;
#else
		//two arbitrary partition
		if (!computeSplitCentroids2(toSplit, sp, sq, usedensity))
			return false;
#endif

		//compute sk
		computeMergeCentroid(toMerge1, toMerge2, sk, usedensity);

		//check feasibility
#ifdef VOL
		double mi = computeVolume(toMerge1);
		double mj = computeVolume(toMerge2);
		double mm = computeVolume(toSplit);
#else
		double mi = rvt[toMerge1].size();
		double mj = rvt[toMerge2].size();
		double mm = rvt[toSplit].size();
#endif
		double threshold = (mi * mj) / (mm * (mi + mj)) * dist(si, sj);
#ifdef OUTPUT_PICK_INFO
		std::cout << "mm: " << mm << "  mi: " << mi << " mj: " << mj << std::endl;
		std::cout << dist(sp, sm) << " " << dist(sq, sm) << "  " << dist(si, sj) << " " << threshold << std::endl;
		std::cout << "dense: " << dense.size() << " sparse: " << sparse.size() << std::endl;
#endif
		return (dist(sp, sm) > threshold) && (dist(sq, sm) > threshold);
	}

	void SplitAndMerge(
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*                  		klabels,
		int&                        toSplit,
		int&						toMerge1,
		int&						toMerge2)
	{
		///MERGE: 
		//(toMerge1, toMerge2) -> toMerge2
		//(si, sj) -> sk
		//update kseeds
		kseedsx[toMerge2] = sk[0];
		kseedsy[toMerge2] = sk[1];
		kseedsz[toMerge2] = sk[2];
		kseedsl[toMerge2] = sk[3];
		kseedsa[toMerge2] = sk[4];
		kseedsb[toMerge2] = sk[5];

		///SPLIT:
		//toSplit -> (toSplit, toMerge1)
		//sm -> (sp, sq)
		//update kseeds
		kseedsx[toSplit] = sp[0];
		kseedsy[toSplit] = sp[1];
		kseedsz[toSplit] = sp[2];
		kseedsl[toSplit] = sp[3];
		kseedsa[toSplit] = sp[4];
		kseedsb[toSplit] = sp[5];

		kseedsx[toMerge1] = sq[0];
		kseedsy[toMerge1] = sq[1];
		kseedsz[toMerge1] = sq[2];
		kseedsl[toMerge1] = sq[3];
		kseedsa[toMerge1] = sq[4];
		kseedsb[toMerge1] = sq[5];

		///Update klabels and rvt
		for (int i = 0; i < rvt[toMerge1].size(); i++){
			klabels[rvt[toMerge1][i]] = toMerge2;
			rvt[toMerge2].push_back(rvt[toMerge1][i]);
		}

		//voxels belong to sp: in spv
		rvt[toSplit].clear(); rvt[toSplit].resize(0);
		for (int i = 0; i < spv.size(); i++){
			//klabels[spv[i]] = toSplit;
			rvt[toSplit].push_back(spv[i]);
		}
		//voxels belong to sq: in sqv
		rvt[toMerge1].clear(); rvt[toMerge1].resize(0);
		for (int i = 0; i < sqv.size(); i++){
			klabels[sqv[i]] = toMerge1;
			rvt[toMerge1].push_back(sqv[i]);
		}

		std::cout << "spv: " << spv.size() << " sqv: " << sqv.size() << std::endl;
	}


	//=====================================================================================
	//								FAST S-M
	//=====================================================================================

	void Merge2(
		const int&					itr,
		const int*                  klabels,
		int*&                       newlabels,
		map<int, int>&				hashtable,
		int&						numlabels,
		const int&					c_merge)
	{
		int sz = m_width * m_height * m_frame;
		const int cnt_threshold = m_step * m_step * m_step;
		const int cnt_threshold_merge = m_step * m_step * m_step / double(c_merge); // 3; // 2;

		for (int i = 0; i < sz; i++) newlabels[i] = -1;
		int* xvec = new int[sz];
		int* yvec = new int[sz];
		int* zvec = new int[sz];

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
		cout << "connectivity chosen: " << neighbornum << endl;

		double merge_diflab = DBL_MAX;
		if (choice == 5){
			merge_diflab = 4;
		}

		hashtable[-1] = 0;

		int nx, ny, nz;
		int labelNodeId = 0;
		int adjlabel = -1;
		double currentl, currenta, currentb;
		double adjl, adja, adjb;
		double diflab = 0;
		for (int z = 0; z < m_frame; z++) {
			for (int y = 0; y < m_height; y++) {
				for (int x = 0; x < m_width; x++) {
					int idx = z*m_width*m_height + y*m_width + x;
					if (newlabels[idx] >= 0)
						continue;
					newlabels[idx] = labelNodeId;

					// find ADJACENT label node
					adjlabel = -1;
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
						//std::cout << diflab << std::endl;
					}

					// find voxels belong to SAME label node (in the same frame)
					// Similar to EnforceConnectivity for Superpixel
					xvec[0] = x;
					yvec[0] = y;
					zvec[0] = z;
					int count = 1;
					for (int c = 0; c < count; c++)
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
									xvec[count] = nx;
									yvec[count] = ny;
									zvec[count] = nz;
									newlabels[nidx] = newlabels[idx];
									count++;
								}
							}
						}
					}
					hashtable[labelNodeId] = count;

					if (adjlabel < 0){
						labelNodeId++;
						continue;
					}

					bool flag = false;
					if (choice <= 3){
						if (itr == iteration - 1){
							if (count <= cnt_threshold / 16) //8
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
						labelNodeId--;
					}
					labelNodeId++;
				}
			}
		}
		std::cout << "# Kseeds: " << kseedsl.size() << std::endl;
		std::cout << "After Merge: " << labelNodeId << std::endl;
		numlabels = labelNodeId;

		if (xvec) delete[] xvec;
		if (yvec) delete[] yvec;
		if (zvec) delete[] zvec;
	}

	void Split2(
		const int&					itr,
		vector<double>&             kseedsl,
		vector<double>&             kseedsa,
		vector<double>&             kseedsb,
		vector<double>&             kseedsx,
		vector<double>&             kseedsy,
		vector<double>&             kseedsz,
		int*&                       newlabels,
		map<int, int>&				hashtable,
		const int&					numlabels,
		const int&					c_split)
	{
		const int cnt_threshold_split = c_split * m_step * m_step * m_step; //2 * m_step * m_step * m_step;

		kseedsl.assign(numlabels, 0);
		kseedsa.assign(numlabels, 0);
		kseedsb.assign(numlabels, 0);
		kseedsx.assign(numlabels, 0);
		kseedsy.assign(numlabels, 0);
		kseedsz.assign(numlabels, 0);

		vector<double> clusterdensity(numlabels, 0);
		vector<double> inv(numlabels, 0);//to store 1/clustersize[k] values
		vector<double> sigmal(numlabels, 0);
		vector<double> sigmaa(numlabels, 0);
		vector<double> sigmab(numlabels, 0);
		vector<double> sigmax(numlabels, 0);
		vector<double> sigmay(numlabels, 0);
		vector<double> sigmaz(numlabels, 0);

		double densityE;

		int ind = 0;
		for (int f = 0; f < m_frame; f++){
			for (int r = 0; r < m_height; r++){
				for (int c = 0; c < m_width; c++){
					densityE = m_densityvec[ind];
					sigmal[newlabels[ind]] += m_lvec[ind] * densityE;
					sigmaa[newlabels[ind]] += m_avec[ind] * densityE;
					sigmab[newlabels[ind]] += m_bvec[ind] * densityE;
					sigmax[newlabels[ind]] += c * densityE;
					sigmay[newlabels[ind]] += r * densityE;
					sigmaz[newlabels[ind]] += f * densityE;
					clusterdensity[newlabels[ind]] += densityE;
					ind++;
				}
			}
		}
		for (int k = 0; k < numlabels; k++){
			if (clusterdensity[k] <= 0) clusterdensity[k] = 1;
			inv[k] = 1.0 / clusterdensity[k];//computing inverse now to multiply, than divide later
			//if (hashtable[k] <= 0) hashtable[k] = 1;
			//inv[k] = 1.0 / hashtable[k];//computing inverse now to multiply, than divide later
		}
		for (int k = 0; k < numlabels; k++){
			kseedsl[k] = sigmal[k] * inv[k];
			kseedsa[k] = sigmaa[k] * inv[k];
			kseedsb[k] = sigmab[k] * inv[k];
			kseedsx[k] = sigmax[k] * inv[k];
			kseedsy[k] = sigmay[k] * inv[k];
			kseedsz[k] = sigmaz[k] * inv[k];
		}
		if (itr < iteration - 2){
			for (int k = 0; k < numlabels; k++){
				if (hashtable[k] > cnt_threshold_split){
					double kx = kseedsx[k];
					double ky = kseedsy[k];
					double x1 = max(kx - m_step / 2.0, 0);
					double x2 = min(kx + m_step / 2.0, m_width - 1);
					double y1 = max(ky - m_step / 2.0, 0);
					double y2 = min(ky + m_step / 2.0, m_height - 1);
					int r = rand() % 2;
					if (r == 1){
						kseedsx[k] = x1;
						kseedsx.push_back(x2);
						kseedsy.push_back(ky);
					}
					else{
						kseedsy[k] = y1;
						kseedsx.push_back(kx);
						kseedsy.push_back(y2);
					}
					kseedsz.push_back(kseedsz[k]);
					kseedsl.push_back(kseedsl[k]);
					kseedsa.push_back(kseedsa[k]);
					kseedsb.push_back(kseedsb[k]);
				}
			}
		}
		std::cout << "Original: " << numlabels << std::endl;
		std::cout << "After Split: " << kseedsl.size() << std::endl;
	}
};