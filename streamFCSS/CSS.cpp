// CSS.cpp: implementation of the MSLIC class.
#include "CSS.h"
#include <math.h>
#define PI 3.14159265354

CSS::CSS()
{
	m_lvec = NULL;
	m_avec = NULL;
	m_bvec = NULL;

	m_volvec = NULL;

}

CSS::~CSS()
{
	if (m_lvec) delete[] m_lvec;
	if (m_avec) delete[] m_avec;
	if (m_bvec) delete[] m_bvec;

	if (m_volvec) delete[] m_volvec;
}

CSS::CSS(bool pickmode, double timedist, int farthestInit, int iteration, bool ifdraw)
{
	this->pickmode = pickmode;
	this->timedist = timedist;
	this->farthestInit = farthestInit;
	this->iteration = iteration;
	this->ifdraw = ifdraw;

	m_lvec = NULL;
	m_avec = NULL;
	m_bvec = NULL;
	m_volvec = NULL;
}

//==============================================================================
/// RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
void CSS::RGB2XYZ(
	const int&              sR,
	const int&              sG,
	const int&              sB,
	double&                 X,
	double&                 Y,
	double&                 Z)
{
	double R = sR / 255.0;
	double G = sG / 255.0;
	double B = sB / 255.0;

	double r, g, b;

	if (R <= 0.04045)    r = R / 12.92;
	else                r = pow((R + 0.055) / 1.055, 2.4);
	if (G <= 0.04045)    g = G / 12.92;
	else                g = pow((G + 0.055) / 1.055, 2.4);
	if (B <= 0.04045)    b = B / 12.92;
	else                b = pow((B + 0.055) / 1.055, 2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

//===========================================================================
/// RGB2LAB
//===========================================================================
void CSS::RGB2LAB(
	const int&              sR,
	const int&              sG,
	const int&              sB,
	double&                 lval,
	double&                 aval,
	double&                 bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;  //actual CIE standard
	double kappa = 903.3;     //actual CIE standard

	double Xr = 0.950456;   //reference white
	double Yr = 1.0;        //reference white
	double Zr = 1.088754;   //reference white

	double xr = X / Xr;
	double yr = Y / Yr;
	double zr = Z / Zr;

	double fx, fy, fz;
	if (xr > epsilon)    fx = pow(xr, 1.0 / 3.0);
	else                fx = (kappa*xr + 16.0) / 116.0;
	if (yr > epsilon)    fy = pow(yr, 1.0 / 3.0);
	else                fy = (kappa*yr + 16.0) / 116.0;
	if (zr > epsilon)    fz = pow(zr, 1.0 / 3.0);
	else                fz = (kappa*zr + 16.0) / 116.0;

	lval = 116.0*fy - 16.0;
	aval = 500.0*(fx - fy);
	bval = 200.0*(fy - fz);
}

//===========================================================================
/// DoRGBtoLABConversion
///
/// For whole image: overlaoded floating point version
//===========================================================================
void CSS::DoRGBtoLABConversion(
	Mat*&                       img,
	double*&                    lvec,
	double*&                    avec,
	double*&                    bvec)
{
	int sz = m_width*m_height;
	lvec = new double[sz];
	avec = new double[sz];
	bvec = new double[sz];

	/*for( int j = 0; j < sz; j++ )
	{
	int r = (ubuff[j] >> 16) & 0xFF;
	int g = (ubuff[j] >>  8) & 0xFF;
	int b = (ubuff[j]      ) & 0xFF;

	RGB2LAB( r, g, b, lvec[j], avec[j], bvec[j] );
	}*/
	for (int y = 0; y < m_height; y++)
	{
		const uchar* p = img->ptr<uchar>(y);
		for (int x = 0; x < m_width; x++)
		{
			int index = y*m_width + x;
			int b = (int)p[3 * x];
			int g = (int)p[3 * x + 1];
			int r = (int)p[3 * x + 2];

			RGB2LAB(r, g, b, lvec[index], avec[index], bvec[index]);
		}
	}
}

//===========================================================================
/// DoRGBtoLABConversion
///
/// For whole volume (supervoxel)
//===========================================================================
void CSS::DoRGBtoLABConversion(
	Mat**&                      imgs,
	double*&                    lvec,
	double*&                    avec,
	double*&                    bvec)
{
	int sz = m_width*m_height*m_frame;
	lvec = new double[sz];
	avec = new double[sz];
	bvec = new double[sz];

	for (int i = 0; i < m_frame; i++)
	{
		for (int y = 0; y < m_height; y++)
		{
			const uchar* p = imgs[i]->ptr<uchar>(y);
			for (int x = 0; x < m_width; x++)
			{
				int index = y * m_width + x + i * (m_width * m_height);
				int b = (int)p[3 * x];
				int g = (int)p[3 * x + 1];
				int r = (int)p[3 * x + 2];

				RGB2LAB(r, g, b, lvec[index], avec[index], bvec[index]);
			}
		}
	}
}

//===========================================================================
/// computeUnitVolume
///
/// For whole volume
//===========================================================================
void CSS::computeUnitVolume(
	double*&                    vol)
{
	int sz = m_width*m_height*m_frame;
	vol = new double[sz];
	memset(vol, 0, sizeof(double)*sz);
	total_vol = 0;

	invwt = 1.0 / ((1 / 1.0)*(1 / 1.0));

	Matrix mtx;
	vector<vector<double> > m_V, m_Vt, m_A;
	vector<double> m_c, m_b, m_x, m_d;
	m_V.assign(6, vector<double>(3, 0));
	m_Vt.assign(3, vector<double>(6, 0));
	m_A.assign(3, vector<double>(3, 0));
	m_c.assign(6, 0);
	m_b.assign(3, 0);
	m_x.assign(3, 0);
	m_d.assign(6, 0);
	int ind[4];
	for (int f = 0; f < m_frame - 1; f++)
	{
		for (int r = 0; r < m_height - 1; r++)
		{
			for (int c = 0; c < m_width - 1; c++)
			{
				ind[0] = f*m_height*m_width + r*m_width + c;
				ind[1] = f*m_height*m_width + (r + 1)*m_width + c;
				ind[2] = f*m_height*m_width + r*m_width + (c + 1);
				ind[3] = (f + 1)*m_height*m_width + r*m_width + c;

				// calculate 2 * AREA of bottom traingle
				double x1 = (0);
				double y1 = (1);
				double z1 = (0) * timedist;
				double l1 = (m_lvec[ind[1]] - m_lvec[ind[0]]) / sqrt(invwt);
				double a1 = (m_avec[ind[1]] - m_avec[ind[0]]) / sqrt(invwt);
				double b1 = (m_bvec[ind[1]] - m_bvec[ind[0]]) / sqrt(invwt);
				double x2 = (1);
				double y2 = (0);
				double z2 = (0) * timedist;
				double l2 = (m_lvec[ind[2]] - m_lvec[ind[0]]) / sqrt(invwt);
				double a2 = (m_avec[ind[2]] - m_avec[ind[0]]) / sqrt(invwt);
				double b2 = (m_bvec[ind[2]] - m_bvec[ind[0]]) / sqrt(invwt);
				double bottom = sqrt((x1*x1 + y1*y1 + z1*z1 + l1*l1 + a1*a1 + b1*b1)*(x2*x2 + y2*y2 + z2*z2 + l2*l2 + a2*a2 + b2*b2) - (x1*x2 + y1*y2 + z1*z2 + l1*l2 + a1*a2 + b1*b2)*(x1*x2 + y1*y2 + z1*z2 + l1*l2 + a1*a2 + b1*b2));

				// calculate min DISTANNCE from ind[3] to the subspace
				if (m_lvec[ind[0]] == 0 && m_avec[ind[0]] == 0 && m_bvec[ind[0]] == 0 &&
					m_lvec[ind[1]] == 0 && m_avec[ind[1]] == 0 && m_bvec[ind[1]] == 0 &&
					m_lvec[ind[2]] == 0 && m_avec[ind[2]] == 0 && m_bvec[ind[2]] == 0){
					double h = sqrt(1 + (m_lvec[ind[3]] * m_lvec[ind[3]] + m_avec[ind[3]] * m_avec[ind[3]] +
						m_bvec[ind[3]] * m_bvec[ind[3]]) / invwt);
					vol[ind[0]] = bottom * h;
					total_vol += vol[ind[0]];
					if (h == 0 || bottom == 0)
						cout << "0vol " << bottom << " " << h << endl;
					continue;
				}
				// calculate min DISTANNCE from ind[3] to the subspace
				for (int m = 0; m < 3; m++) { // V = [a b c], 6 * 3 matrix
					m_V[0][m] = ind[m] % m_width;
					m_V[1][m] = (ind[m] / m_width) % m_height;
					m_V[2][m] = ind[m] / (m_width * m_height) * timedist;
					m_V[3][m] = m_lvec[ind[m]] / sqrt(invwt);
					m_V[4][m] = m_avec[ind[m]] / sqrt(invwt);
					m_V[5][m] = m_bvec[ind[m]] / sqrt(invwt);
				}
				m_c[0] = ind[3] % m_width;
				m_c[1] = (ind[3] / m_width) % m_height;
				m_c[2] = ind[3] / (m_width * m_height) * timedist;
				m_c[3] = m_lvec[ind[3]] / sqrt(invwt);
				m_c[4] = m_avec[ind[3]] / sqrt(invwt);
				m_c[5] = m_bvec[ind[3]] / sqrt(invwt);
				mtx.Transpose(m_V, m_Vt, 6, 3);             //Vt = V'
				mtx.MatrixMultiply(m_Vt, m_V, m_A, 3, 6, 3);//A = Vt * V
				mtx.MatrixMultiplyVector(m_Vt, m_c, m_b, 3, 6); //b = Vt * c
				int gaus = mtx.Gaussian(m_A, m_b, m_x, 3);             //solve A * x = b
				mtx.MatrixMultiplyVector(m_V, m_x, m_d, 6, 3);  //d = V * x
				double h = mtx.DistVec(m_c, m_d, 6);        //h = ||c - d||
#ifdef _WIN32
				if (gaus == -1 || _isnan(h) || h == 0){
#else
				if (gaus == -1 || isnan(h) || h == 0){
#endif
					/*
					cout << "current_volume: " << bottom << " " << h << " " << bottom * h << endl;
					cout << "r: " << r << " c: " << c << endl;
					cout << "m_V:\n";
					for (int i = 0; i < 6; i++)
						cout << " " << m_V[i][0] << " " << m_V[i][1] << " " << m_V[i][2] << endl;
					cout << "m_c:\n";
					for (int i = 0; i < 6; i++)
						cout << " " << m_c[i] << endl;
					*/
					h = 1; // avoid nan -1.#IND
				}
				// calculate VOLUME
				vol[ind[0]] = bottom * h;
				if (h == 0 || bottom == 0)
					cout << "0vol " << bottom << " " << h << endl;
				total_vol += vol[ind[0]];
			}
		}
	}
}

//============================================================================
// Calculate density from CIELAB information for video
//============================================================================
void CSS::DoDensityCalculation()
{
	if (m_volvec == NULL) computeUnitVolume(m_volvec);
	int sz = m_width*m_height*m_frame;
	m_densityvec = new double[sz];
	maxdensity = -1e30;
	mindensity = 1e30;
	for (int f = 0; f < m_frame - 1; f++)
	{
		for (int r = 0; r < m_height - 1; r++)
		{
			for (int c = 0; c < m_width - 1; c++)
			{
				int ind = f*m_height*m_width + r*m_width + c;
				m_densityvec[ind] = m_volvec[ind];
			}
		}

		//对于最后一列
		int cL = m_width - 1;
		for (int r = 0; r < m_height - 1; r++){
			m_densityvec[f*m_height*m_width + r*m_width + cL] = m_densityvec[f*m_height*m_width + r*m_width + (cL - 1)];
		}

		//对于最后一行
		int rL = m_height - 1;
		for (int c = 0; c < m_width - 1; c++){
			m_densityvec[f*m_height*m_width + rL*m_width + c] = m_densityvec[f*m_height*m_width + (rL - 1)*m_width + c];
		}

		//对于右下角那一个像素
		m_densityvec[f*m_height*m_width + rL*m_width + cL] = 0.5 * (m_densityvec[f*m_height*m_width + (rL - 1)*m_width + cL] +
			m_densityvec[f*m_height*m_width + rL*m_width + (cL - 1)]);
	}

	//对于最后一帧
	int fL = m_frame - 1;
	for (int r = 0; r < m_height; r++)
	{
		for (int c = 0; c < m_width; c++)
		{
			m_densityvec[fL*m_height*m_width + r*m_width + c] = m_densityvec[(fL - 1)*m_height*m_width + r*m_width + c];
		}
	}
	
	for (int f = 0; f < m_frame; f++)
	{
		for (int r = 0; r < m_height; r++)
		{
			for (int c = 0; c < m_width; c++)
			{
				int ind = f*m_height*m_width + r*m_width + c;
				//m_densityvec[ind] = pow(m_densityvec[ind], m_exponent);
				m_densityvec[ind] = atan(m_densityvec[ind]) * 2 / PI;
				m_densityvec[ind] = 1 - m_densityvec[ind] * m_exponent;
				if (m_densityvec[ind] > maxdensity)
					maxdensity = m_densityvec[ind];
				if (m_densityvec[ind] < mindensity)
					mindensity = m_densityvec[ind];
			}
		}
	}
}

//===========================================================================
/// juRandomPermuteRange
///
/// Randomly permute the values 0 to n-1 and store the permuted list in a vector
//===========================================================================
void CSS::juRandomPermuteRange(
	int                         n,
	vector<int>&                v,
	unsigned int*               seed)
{
	v.clear();
	v.resize(n);
	int i, j;
	v[0] = 0;
	for (i = 1; i < n; i++)
	{
#ifdef _WIN32
		j = (int)(i * ((double)rand() / RAND_MAX)); //generate random float in [0, i], convert to int
		//the result is int from 0 to i.
#else
		j = rand_r(seed) % (i + 1);
#endif
		v[i] = v[j];
		v[j] = i;
	}
}

//=================================================================================
/// DrawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void CSS::DrawContoursAroundSegments(
	Mat*&                       img,
	int*&                       labels,
	const int&                  width,
	const int&                  height,
	const unsigned int&         color)
{
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int sz = width*height;
	vector<bool> istaken(sz, false);
	vector<int> contourx(sz); vector<int> contoury(sz);
	int mainindex(0); int cind(0);
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			int np(0);
			for (int i = 0; i < 8; i++)
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height))
				{
					int index = y*width + x;
					//if( false == istaken[index] )//comment this to obtain internal contours
					{
						if (labels[mainindex] != labels[index]) np++;
					}
				}
			}
			//if( np > 1 )//change to 2 or 3 for thinner lines
			if (np > 1)
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				cind++;
			}
			mainindex++;
		}
	}
	if (!ifdraw)
	{
		for (int i = 0; i<height; i++)
		{
			img->row(i).setTo(Scalar(255, 255, 255));
		}
	}

	int numboundpix = cind;//int(contourx.size());
	for (int j = 0; j < numboundpix; j++)
	{
		/*int ii = contoury[j]*width + contourx[j];
		ubuff[ii]=0x000000;*/
		img->at<Vec3b>(contoury[j], contourx[j])[0] = 0;
		img->at<Vec3b>(contoury[j], contourx[j])[1] = 0;
		img->at<Vec3b>(contoury[j], contourx[j])[2] = 0;
	}
}


//==============================================================================
/// DetectLabEdges
//==============================================================================
void CSS::DetectLabEdges(
	const double*               lvec,
	const double*               avec,
	const double*               bvec,
	const int&                  width,
	const int&                  height,
	vector<double>&             edges)
{
	int sz = width*height;

	edges.resize(sz, 0);
	for (int j = 1; j < height - 1; j++)
	{
		for (int k = 1; k < width - 1; k++)
		{
			int i = j*width + k;

			double dx = (lvec[i - 1] - lvec[i + 1])*(lvec[i - 1] - lvec[i + 1]) +
				(avec[i - 1] - avec[i + 1])*(avec[i - 1] - avec[i + 1]) +
				(bvec[i - 1] - bvec[i + 1])*(bvec[i - 1] - bvec[i + 1]);

			double dy = (lvec[i - width] - lvec[i + width])*(lvec[i - width] - lvec[i + width]) +
				(avec[i - width] - avec[i + width])*(avec[i - width] - avec[i + width]) +
				(bvec[i - width] - bvec[i + width])*(bvec[i - width] - bvec[i + width]);

			//edges[i] = fabs(dx) + fabs(dy);
			edges[i] = dx*dx + dy*dy;
		}
	}
}

//==============================================================================
/// DetectLabEdges (volume)
//==============================================================================
void CSS::DetectLabEdges(
	const double*               lvec,
	const double*               avec,
	const double*               bvec,
	const int&                  width,
	const int&                  height,
	const int&                  frame,
	vector<double>&             edges)
{
	int sz = width*height*frame;

	edges.resize(sz, 0);
	int n((width*height) + width + 1);
	for (int i = 1; i < frame - 1; i++)
	{
		for (int j = 1; j < height - 1; j++)
		{
			for (int k = 1; k < width - 1; k++)
			{
				double dx = (lvec[n - 1] - lvec[n + 1])*(lvec[n - 1] - lvec[n + 1]) +
					(avec[n - 1] - avec[n + 1])*(avec[n - 1] - avec[n + 1]) +
					(bvec[n - 1] - bvec[n + 1])*(bvec[n - 1] - bvec[n + 1]);

				double dy = (lvec[n - width] - lvec[n + width])*(lvec[n - width] - lvec[n + width]) +
					(avec[n - width] - avec[n + width])*(avec[n - width] - avec[n + width]) +
					(bvec[n - width] - bvec[n + width])*(bvec[n - width] - bvec[n + width]);

				double dz = (lvec[n - width*height] - lvec[n + width*height])*(lvec[n - width*height] - lvec[n + width*height]) +
					(avec[n - width*height] - avec[n + width*height])*(avec[n - width*height] - avec[n + width*height]) +
					(bvec[n - width*height] - bvec[n + width*height])*(bvec[n - width*height] - bvec[n + width*height]);

				//edges[i] = fabs(dx) + fabs(dy);
				edges[n] = dx*dx + dy*dy + dz*dz;
				n++;
			}
		}
	}
}

//===========================================================================
/// PerturbSeeds
//===========================================================================
void CSS::PerturbSeeds(
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	const vector<double>&       edges)
{
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	size_t numseeds = kseedsl.size();

	for (int n = 0; n < numseeds; n++)
	{
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for (int i = 0; i < 8; i++)
		{
			int nx = ox + dx8[i];//new x
			int ny = oy + dy8[i];//new y

			if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = ny*m_width + nx;
				if (edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if (storeind != oind)
		{
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = storeind / m_width;
			kseedsl[n] = m_lvec[storeind];
			kseedsa[n] = m_avec[storeind];
			kseedsb[n] = m_bvec[storeind];
		}
	}
}

//===========================================================================
/// PerturbSeeds (volume)
//===========================================================================
void CSS::PerturbSeeds(
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	vector<double>&             kseedsz,
	const vector<double>&       edges)
{
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	const int dx26[26] = { -1, -1, 0, 1, 1, 1, 0, -1, 0,
		-1, -1, 0, 1, 1, 1, 0, -1,
		-1, -1, 0, 1, 1, 1, 0, -1, 0 };
	const int dy26[26] = { 0, -1, -1, -1, 0, 1, 1, 1, 0,
		0, -1, -1, -1, 0, 1, 1, 1,
		0, -1, -1, -1, 0, 1, 1, 1, 0 };
	const int dz26[26] = { -1, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1 };

	int numseeds = kseedsl.size();

	for (int n = 0; n < numseeds; n++)
	{
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oz = kseedsz[n];//original z
		int oind = oy*m_width + ox + oz*(m_width*m_height);

		int storeind = oind;
		for (int i = 0; i < 26; i++)
		{
			int nx = ox + dx26[i];//new x
			int ny = oy + dy26[i];//new y
			int nz = oz + dz26[i];//new z

			if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height && nz >= 0 && nz < m_frame)
			{
				int nind = ny*m_width + nx + nz*(m_width*m_height);
				if (edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if (storeind != oind)
		{
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = (storeind % (m_width*m_height)) / m_width;
			kseedsz[n] = storeind / (m_width*m_height);
			kseedsl[n] = m_lvec[storeind];
			kseedsa[n] = m_avec[storeind];
			kseedsb[n] = m_bvec[storeind];
		}
	}
}

//===========================================================================
/// GetLABXYSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void CSS::GetLABXYSeeds_ForGivenStepSize(
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	const int&                  STEP,
	const bool&                 perturbseeds,
	const vector<double>&       edgemag)
{
	const bool hexgrid = false;
	int numseeds(0);
	int n(0);

	int xstrips = (0.5 + double(m_width) / double(STEP));
	int ystrips = (0.5 + double(m_height) / double(STEP));

	int xerr = m_width - STEP*xstrips; if (xerr < 0){ xstrips--; xerr = m_width - STEP*xstrips; }
	int yerr = m_height - STEP*ystrips; if (yerr < 0){ ystrips--; yerr = m_height - STEP*ystrips; }

	double xerrperstrip = double(xerr) / double(xstrips);
	double yerrperstrip = double(yerr) / double(ystrips);

	int xoff = STEP / 2;
	int yoff = STEP / 2;
	//-------------------------
	numseeds = xstrips*ystrips;
	//-------------------------
	kseedsl.resize(numseeds);
	kseedsa.resize(numseeds);
	kseedsb.resize(numseeds);
	kseedsx.resize(numseeds);
	kseedsy.resize(numseeds);

	for (int y = 0; y < ystrips; y++)
	{
		int ye = y*yerrperstrip;
		for (int x = 0; x < xstrips; x++)
		{
			int xe = x*xerrperstrip;
			int seedx = (x*STEP + xoff + xe);
			if (hexgrid){ seedx = x*STEP + (xoff << (y & 0x1)) + xe; seedx = min(m_width - 1, seedx); }//for hex grid sampling
			int seedy = (y*STEP + yoff + ye);
			int i = seedy*m_width + seedx;

			kseedsl[n] = m_lvec[i];
			kseedsa[n] = m_avec[i];
			kseedsb[n] = m_bvec[i];
			kseedsx[n] = seedx;
			kseedsy[n] = seedy;
			n++;
		}
	}
	if (perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemag);
	}
}

//===========================================================================
/// GetLABXYZSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial voxel samples.
//===========================================================================
void CSS::GetLABXYZSeeds_ForGivenStepSize(
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	vector<double>&             kseedsz,
	const int&                  STEP)
{
	//srand((unsigned)time(NULL));
	clock_t start1, finish1;
	start1 = clock();

	int numseeds(0);
	int n(0);

	int xstrips = (0.5 + double(m_width) / double(STEP));
	int ystrips = (0.5 + double(m_height) / double(STEP));
	int zstrips = (0.5 + double(m_frame) / double(STEP));

	int xerr = m_width - STEP*xstrips; if (xerr < 0){ xstrips--; xerr = m_width - STEP*xstrips; }
	int yerr = m_height - STEP*ystrips; if (yerr < 0){ ystrips--; yerr = m_height - STEP*ystrips; }
	int zerr = m_frame - STEP*zstrips; if (zerr < 0){ zstrips--; zerr = m_frame - STEP*zstrips; }

	double xerrperstrip = double(xerr) / double(xstrips);
	double yerrperstrip = double(yerr) / double(ystrips);
	double zerrperstrip = double(zerr) / double(zstrips);

	int xoff = STEP / 2;
	int yoff = STEP / 2;
	int zoff = STEP / 2;
	//-------------------------
	numseeds = xstrips*ystrips*zstrips;
	//-------------------------
	kseedsl.resize(numseeds);
	kseedsa.resize(numseeds);
	kseedsb.resize(numseeds);
	kseedsx.resize(numseeds);
	kseedsy.resize(numseeds);
	kseedsz.resize(numseeds);

	for (int z = 0; z < zstrips; z++)
	{
		int ze = z*zerrperstrip;
		for (int y = 0; y < ystrips; y++)
		{
			int ye = y*yerrperstrip;
			for (int x = 0; x < xstrips; x++)
			{
				int xe = x*xerrperstrip;
				//int i = (y*STEP+yoff+ye)*m_width + (x*STEP+xoff+xe) + (z*STEP+zoff+ze)*(m_width*m_height);

				kseedsl[n] = m_lvec[n];
				kseedsa[n] = m_avec[n];
				kseedsb[n] = m_bvec[n];
				kseedsx[n] = (x*STEP + xoff + xe);
				kseedsy[n] = (y*STEP + yoff + ye);
				kseedsz[n] = (z*STEP + zoff + ze);
				n++;
			}
		}
	}
	finish1 = clock();
	printf("Uniform Init time: %f seconds\n", (double)(finish1 - start1) / CLOCKS_PER_SEC);
}

//===========================================================================
/// NaiveInit
//===========================================================================
void CSS::NaiveInit(
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	vector<double>&             kseedsz,
	const int&                  STEP,
	const double&               M)
{
	//srand((unsigned)time(NULL));
	clock_t start1, finish1;
	start1 = clock();

	int sz = m_width * m_height * m_frame;
	vector<double> distvec(sz, DBL_MAX);

	invwt = 1.0 / ((STEP / M)*(STEP / M));

	kseedsl.resize(m_K);
	kseedsa.resize(m_K);
	kseedsb.resize(m_K);
	kseedsx.resize(m_K);
	kseedsy.resize(m_K);
	kseedsz.resize(m_K);

	int k;
	for (int n = 0; n < m_K; n++){
		k = (int)((sz-1) * ((double)rand() / RAND_MAX));//[0,sz-1]
		kseedsl[n] = m_lvec[k];
		kseedsa[n] = m_avec[k];
		kseedsb[n] = m_bvec[k];
		kseedsx[n] = k % m_width;
		kseedsy[n] = (k / m_width) % m_height;
		kseedsz[n] = k / (m_width * m_height);
	}
	finish1 = clock();
	printf("Naive Init time: %f seconds\n", (double)(finish1 - start1) / CLOCKS_PER_SEC);
}

//===========================================================================
/// KMeansPP
///
/// The k seed values are taken as k-means++.
//===========================================================================
void CSS::KMeansPP(
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	vector<double>&             kseedsz,
	const int&                  STEP,
	const double&               M)
{
	//srand((unsigned)time(NULL));//srand(2018);
	clock_t start1, finish1;
	start1 = clock();

	int sz = m_width * m_height * m_frame;
	vector<double> distvec(sz, DBL_MAX);

	invwt = 1.0 / ((STEP / M)*(STEP / M));

	kseedsl.resize(m_K);
	kseedsa.resize(m_K);
	kseedsb.resize(m_K);
	kseedsx.resize(m_K);
	kseedsy.resize(m_K);
	kseedsz.resize(m_K);

	int k = (int)((sz-1) * ((double)rand() / RAND_MAX));//[0,sz-1]
	kseedsl[0] = m_lvec[k];
	kseedsa[0] = m_avec[k];
	kseedsb[0] = m_bvec[k];
	kseedsx[0] = k % m_width;
	kseedsy[0] = (k / m_width) % m_height;
	kseedsz[0] = k / (m_width * m_height);

	double l, a, b;
	double dist, distxyz;
	double sum, cursum, r;
	for (int n = 1; n < m_K; n++){
		//update disvec
		for (int z = 0; z < m_frame; z++)
		{
			for (int y = 0; y < m_height; y++)
			{
				for (int x = 0; x < m_width; x++)
				{
					int i = z*m_width*m_height + y*m_width + x;

					l = m_lvec[i];
					a = m_avec[i];
					b = m_bvec[i];

					dist = (l - kseedsl[n - 1])*(l - kseedsl[n - 1]) +
						(a - kseedsa[n - 1])*(a - kseedsa[n - 1]) +
						(b - kseedsb[n - 1])*(b - kseedsb[n - 1]);

					distxyz = (x - kseedsx[n - 1])*(x - kseedsx[n - 1]) +
						(y - kseedsy[n - 1])*(y - kseedsy[n - 1]) +
						(z - kseedsz[n - 1])*(z - kseedsz[n - 1]) * timedist * timedist;

					dist += distxyz * invwt;

					if (dist < distvec[i])
					{
						distvec[i] = dist;
					}
				}
			}
		}
		//distvec[i] = D(x)^2
		//Pr(choosing x) = D(x)^2/sum

		//calculate sum
		sum = 0;
		for (int i = 0; i < sz; i++)
			sum += distvec[i];

		//get random number -- select voxel with largest probability
		r = sum * ((double)rand() / RAND_MAX);
		cursum = 0;
		k = 0;
		while (cursum <= r && k < sz){
			cursum += distvec[k];
			k++;
		}
		//cout << k << endl;
		k = min(k, sz - 1);
		//assign new seed
		kseedsl[n] = m_lvec[k];
		kseedsa[n] = m_avec[k];
		kseedsb[n] = m_bvec[k];
		kseedsx[n] = k % m_width;
		kseedsy[n] = (k / m_width) % m_height;
		kseedsz[n] = k / (m_width * m_height);
	}
	finish1 = clock();
	printf("Kmeans++ Init time: %f seconds\n", (double)(finish1 - start1) / CLOCKS_PER_SEC);
}

//===========================================================================
/// KMeansPPBasedOnVol
///
/// The k seed values are taken similar to k-means++.
/// Volume of each voxel is considered.
//===========================================================================
void CSS::KMeansPPBasedOnVol(
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	vector<double>&             kseedsz,
	const int&                  STEP,
	const double&               M)
{
	//srand((unsigned)time(NULL));
	clock_t start1, finish1;
	start1 = clock();

	int sz = m_width * m_height * m_frame;
	vector<double> distvec(sz, DBL_MAX);

	invwt = 1.0 / ((STEP / M)*(STEP / M));

	kseedsl.resize(m_K);
	kseedsa.resize(m_K);
	kseedsb.resize(m_K);
	kseedsx.resize(m_K);
	kseedsy.resize(m_K);
	kseedsz.resize(m_K);

	if (m_volvec) delete[] m_volvec; m_volvec = NULL;
	computeUnitVolume(m_volvec);

	//Choose 1st Point
	double sum_1, cursum_1, r_1;
	sum_1 = 0;
	for (int i = 0; i < sz; i++)
		sum_1 += m_volvec[i];
	r_1 = sum_1 * ((double)rand() / RAND_MAX);
	cursum_1 = 0;
	int k = 0;
	while (cursum_1 <= r_1 && k < sz) {
		cursum_1 += m_volvec[k];
		k++;
	}
	kseedsl[0] = m_lvec[k];
	kseedsa[0] = m_avec[k];
	kseedsb[0] = m_bvec[k];
	kseedsx[0] = k % m_width;
	kseedsy[0] = (k / m_width) % m_height;
	kseedsz[0] = k / (m_width * m_height);

	//Choose remaining (K-1) Points
	double l, a, b;
	double dist, distxyz;
	double sum, cursum, r;
	for (int n = 1; n < m_K; n++) {
		//update disvec
		for (int z = 0; z < m_frame; z++)
		{
			for (int y = 0; y < m_height; y++)
			{
				for (int x = 0; x < m_width; x++)
				{
					int i = z*m_width*m_height + y*m_width + x;

					l = m_lvec[i];
					a = m_avec[i];
					b = m_bvec[i];

					dist = (l - kseedsl[n - 1])*(l - kseedsl[n - 1]) +
						(a - kseedsa[n - 1])*(a - kseedsa[n - 1]) +
						(b - kseedsb[n - 1])*(b - kseedsb[n - 1]);

					distxyz = (x - kseedsx[n - 1])*(x - kseedsx[n - 1]) +
						(y - kseedsy[n - 1])*(y - kseedsy[n - 1]) +
						(z - kseedsz[n - 1])*(z - kseedsz[n - 1]) * timedist * timedist;

					dist += distxyz * invwt;

					if (dist < distvec[i])
					{
						distvec[i] = dist;
					}
				}
			}
		}
		//distvec[i] = D(x)^2
		//Pr(choosing x) = D(x)^2/sum

		//calculate sum
		sum = 0;
		for (int i = 0; i < sz; i++)
			sum += distvec[i] * m_volvec[i]; //p_vi(vj) = Vol(fai(vj))*dist(vj,vi)^2

		//get random number -- select voxel with largest probability
		r = sum * ((double)rand() / RAND_MAX);
		cursum = 0;
		k = 0;
		while (cursum <= r && k < sz) {
			cursum += distvec[k] * m_volvec[k];
			k++;
		}
		//cout << k << endl;
		k = min(k, sz - 1);
		//assign new seed
		kseedsl[n] = m_lvec[k];
		kseedsa[n] = m_avec[k];
		kseedsb[n] = m_bvec[k];
		kseedsx[n] = k % m_width;
		kseedsy[n] = (k / m_width) % m_height;
		kseedsz[n] = k / (m_width * m_height);
	}
	finish1 = clock();
	printf("Kmeans++_vol Init time: %f seconds\n", (double)(finish1 - start1) / CLOCKS_PER_SEC);
}

//===========================================================================
/// SaveSuperpixelLabels
///
/// Save labels in raster scan order.
//===========================================================================
void CSS::SaveSuperpixelLabels(
	const int*&                 labels,
	const int&                  width,
	const int&                  height,
	const string&               filename,
	const string&               path)
{
#ifdef _WIN32
	char fname[256];
	char extn[256];
	_splitpath(filename.c_str(), NULL, NULL, fname, extn);
	string temp = fname;
	string finalpath = path + temp + string(".txt");
#else
	string nameandextension = filename;
	size_t pos = filename.find_last_of("/");
	if (pos != string::npos)
	{
		nameandextension = filename.substr(pos + 1);
	}
	string newname = nameandextension.replace(nameandextension.rfind(".") + 1, 3, "txt");//find the position of the dot and replace the 3 characters following it.
	string finalpath = path + newname;
#endif

	int sz = width*height;
	ofstream outfile;
	outfile.open(finalpath.c_str(), ios::binary);
	for (int i = 0; i < sz; i++)
	{
		outfile.write((const char*)&labels[i], sizeof(int));
	}
	outfile.close();
}

//===========================================================================
/// Print labels
//===========================================================================
void CSS::printlabel(int*& klabels, int width, int height, int itr)
{
	ofstream ofile;
	char filename[255];
	sprintf(filename, "label_%d.txt", itr);
	ofile.open(filename, ios::app);
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			ofile << klabels[i*width + j] << ' ';
		}
		ofile << endl;
	}
}

//===========================================================================
/// SaveSupervoxelLabelNodeIds
///
/// Print labels
//===========================================================================
void CSS::SaveSupervoxelLabels(
	int*&                       labels,
	const int&                  width,
	const int&                  height,
	const int&                  frame,
	char*&                      output_path)
{
	char savepath[256];
	int frame_sz = width*height;
	for (int i = 0; i < frame; i++) {
#ifdef _WIN32
		sprintf(savepath, "%s\\%05d.txt", output_path, i + 1);
#else
		sprintf(savepath, "%s/%05d.txt", output_path, i + 1);
#endif
		FILE *out;
		out = fopen(savepath, "w");
		for (int j = 0; j < height; j++){
			for (int k = 0; k < width; k++) {
				fprintf(out, "%d ", labels[i*frame_sz + j*width + k]);
			}
			fprintf(out, "\n");
		}
		fclose(out);
	}
}

//===========================================================================
/// PerformSuperpixelSLIC
//===========================================================================
void CSS::PerformSuperpixelMSLIC(
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	int*&                       klabels,
	const int                   STEP,
	const vector<double>&       edgemag,
	vector<double>&             adaptk,
	const double&               M,
	const int&                  speed)
{
	int sz = m_width*m_height;
	const int numk = kseedsl.size();
	//----------------
	int offset = STEP;
	//if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
	//----------------

	vector<double> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values

	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<double> distvec(sz, DBL_MAX);

	invwt = 1.0 / ((STEP / M)*(STEP / M));

	int x1, y1, x2, y2;
	double l, a, b;
	double dist;
	double distxy;
	for (int itr = 0; itr < 1; itr++)
	{
		distvec.assign(sz, DBL_MAX);

		for (int n = 0; n < numk; n++)
		{
			if (adaptk[n]<1)
			{
				offset = (int)speed*STEP*adaptk[n];
			}
			else
			{
				offset = (int)speed*STEP*adaptk[n];
			}

			y1 = max(0.0, kseedsy[n] - offset);
			y2 = min((double)m_height, kseedsy[n] + offset);
			x1 = max(0.0, kseedsx[n] - offset);
			x2 = min((double)m_width, kseedsx[n] + offset);


			for (int y = y1; y < y2; y++)
			{
				for (int x = x1; x < x2; x++)
				{
					int i = y*m_width + x;

					l = m_lvec[i];
					a = m_avec[i];
					b = m_bvec[i];

					dist = (l - kseedsl[n])*(l - kseedsl[n]) +
						(a - kseedsa[n])*(a - kseedsa[n]) +
						(b - kseedsb[n])*(b - kseedsb[n]);

					distxy = (x - kseedsx[n])*(x - kseedsx[n]) +
						(y - kseedsy[n])*(y - kseedsy[n]);

					//------------------------------------------------------------------------
					dist += distxy*invwt;//dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
					//dist=distxy;
					//------------------------------------------------------------------------
					if (dist < distvec[i])
					{
						distvec[i] = dist;
						klabels[i] = n;
					}
				}

			}
		}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		//instead of reassigning memory on each iteration, just reset.

		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		clustersize.assign(numk, 0);
		//------------------------------------
		//edgesum.assign(numk, 0);
		//------------------------------------

		{int ind(0);
		for (int r = 0; r < m_height; r++)
		{
			for (int c = 0; c < m_width; c++)
			{
				sigmal[klabels[ind]] += m_lvec[ind];
				sigmaa[klabels[ind]] += m_avec[ind];
				sigmab[klabels[ind]] += m_bvec[ind];
				sigmax[klabels[ind]] += c;
				sigmay[klabels[ind]] += r;
				//------------------------------------
				//edgesum[klabels[ind]] += edgemag[ind];
				//------------------------------------
				clustersize[klabels[ind]] += 1.0;
				ind++;
			}
		}}

		{for (int k = 0; k < numk; k++)
		{
			if (clustersize[k] <= 0) clustersize[k] = 1;
			inv[k] = 1.0 / clustersize[k];//computing inverse now to multiply, than divide later
		}}

		{for (int k = 0; k < numk; k++)
		{
			kseedsl[k] = sigmal[k] * inv[k];
			kseedsa[k] = sigmaa[k] * inv[k];
			kseedsb[k] = sigmab[k] * inv[k];
			kseedsx[k] = sigmax[k] * inv[k];
			kseedsy[k] = sigmay[k] * inv[k];
			//------------------------------------
			//edgesum[k] *= inv[k];
			//------------------------------------
		}}
	}
}

//===========================================================================
/// EnforceLabelConnectivity
//===========================================================================
void CSS::EnforceLabelConnectivity(
	int                         itr,
	vector<double>&             adaptk,
	vector<double>&             newadaptk,
	const int*                  labels,//input labels that need to be corrected to remove stray labels
	const int                   width,
	const int                   height,
	int*&                       nlabels,//new labels
	int&                        numlabels,//the number of labels changes in the end if segments are removed
	const int&                  K,//the number of superpixels desired by the user
	const double&               merge)//diflab threshold
{
	//  const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//  const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };
	newadaptk.clear();
	const int sz = width*height;
	const int SUPSZ = sz / K;
	for (int i = 0; i < sz; i++) nlabels[i] = -1;
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	double adjl;
	double adja;
	double adjb;
	double currentl;
	double currenta;
	double currentb;
	double diflab;
	int currentlabel;
	int adjlabel(0);//adjacent label
	map<int, int> hashtable;
	hashtable[-1] = 0;
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			if (0 > nlabels[oindex])
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				currentlabel = labels[j*width + k];
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{
				adjl = 0;
				adja = 0;
				adjb = 0;
				for (int n = 0; n < 4; n++)
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if ((x >= 0 && x < width) && (y >= 0 && y < height))
					{
						int nindex = y*width + x;
						if (nlabels[nindex] >= 0)
						{
							adjlabel = nlabels[nindex];
							adjl = kseedsl[labels[nindex]];//labels - old
							adja = kseedsa[labels[nindex]];
							adjb = kseedsb[labels[nindex]];
						}
					}
				}}
				currentl = kseedsl[labels[j*width + k]];//labels - old
				currenta = kseedsa[labels[j*width + k]];
				currentb = kseedsb[labels[j*width + k]];
				diflab = sqrt((currentl - adjl)*(currentl - adjl) + (currenta - adja)*(currenta - adja) + (currentb - adjb)*(currentb - adjb));
				newadaptk.push_back(adaptk[currentlabel]);
				int count(1);
				for (int c = 0; c < count; c++)
				{
					for (int n = 0; n < 4; n++)
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];
						if ((x >= 0 && x < width) && (y >= 0 && y < height))
						{
							int nindex = y*width + x;
							if (0 > nlabels[nindex] && labels[oindex] == labels[nindex])
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}
					}
				}
				hashtable[label] = count;
				//-------------------------------------------------------
				// If segment size is less then a limit or is very similar to its neighbor, assign an adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if (itr<iteration - 1)
				{
					if (count <= SUPSZ >> 3 || (diflab<merge&&hashtable[adjlabel] + hashtable[newadaptk.size() - 1] <= 3 * m_step*m_step))
					{
						if ((diflab<merge&&hashtable[adjlabel] + hashtable[newadaptk.size() - 1] <= 3 * m_step*m_step))
						{
							newadaptk[adjlabel] = min(2, (newadaptk[adjlabel] + newadaptk[newadaptk.size() - 1]));
							hashtable[adjlabel] = hashtable[adjlabel] + hashtable[newadaptk.size() - 1];
						}
						for (int c = 0; c < count; c++)
						{
							int ind = yvec[c] * width + xvec[c];
							nlabels[ind] = adjlabel;
						}
						label--;
						newadaptk.pop_back();
					}
				}
				else
				{
					if (count <= SUPSZ >> 3)
					{
						for (int c = 0; c < count; c++)
						{
							int ind = yvec[c] * width + xvec[c];
							nlabels[ind] = adjlabel;
						}
						label--;
						newadaptk.pop_back();
					}
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;
	adaptk.clear();
	adaptk = newadaptk;

	if (xvec) delete[] xvec;
	if (yvec) delete[] yvec;
}

//===========================================================================
/// SuperpixelSplit
//===========================================================================
void CSS::SuperpixelSplit(
	int                         itr,
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	int*&                       klabels,
	const int                   STEP,
	const vector<double>&       edgemag,
	vector<double>&             adaptk,
	int                         numk,//the number of labels after enforce connectivity
	const double&               M,//compactness
	const double&               split)
{
	int sz = m_width*m_height;

	vector<double> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<double> distvec(sz, DBL_MAX);
	invwt = 1.0 / ((STEP / M)*(STEP / M));

	sigmal.assign(numk, 0);
	sigmaa.assign(numk, 0);
	sigmab.assign(numk, 0);
	sigmax.assign(numk, 0);
	sigmay.assign(numk, 0);
	clustersize.assign(numk, 0);
	{int ind(0);
	for (int r = 0; r < m_height; r++)
	{
		for (int c = 0; c < m_width; c++)
		{
			if (klabels[ind] >= 0)
			{
				sigmal[klabels[ind]] += m_lvec[ind];
				sigmaa[klabels[ind]] += m_avec[ind];
				sigmab[klabels[ind]] += m_bvec[ind];
				sigmax[klabels[ind]] += c;
				sigmay[klabels[ind]] += r;
				clustersize[klabels[ind]] += 1.0;
				ind++;
			}
		}
	}}
	{for (int k = 0; k < numk; k++)
	{
		if (clustersize[k] <= 0) clustersize[k] = 1;
		inv[k] = 1.0 / clustersize[k];//computing inverse now to multiply, than divide later
	}}
	if (itr<iteration - 2)
	{
		vector<double> avgl(numk, 1);
		vector<double> avga(numk, 1);
		vector<double> avgb(numk, 1);
		for (int i = 0; i<numk; i++)
		{
			avgl[i] = sigmal[i] * inv[i];
			avga[i] = sigmaa[i] * inv[i];
			avgb[i] = sigmab[i] * inv[i];
		}
		invwt = 1.0 / ((1 / 1.0)*(1 / 1.0));
		vector<double> avglabs(numk, 0);
		vector<double> rate(numk, 1);
		//-------------------------------------------------------
		/// Calculate area of each superpixel
		//-------------------------------------------------------
		{
			for (int r = 0; r < m_height - 1; r++)
			{
				for (int c = 0; c < m_width - 1; c++)
				{
					int ind = r*m_width + c;
					int ind2 = (r + 1)*m_width + c;
					int ind3 = (r)*m_width + c + 1;
					if (klabels[ind] == klabels[ind2] && klabels[ind] == klabels[ind3])
					{
						double x1 = (1);
						double y1 = (0);
						double l1 = (m_lvec[ind2] - m_lvec[ind]) / sqrt(invwt);
						double a1 = (m_avec[ind2] - m_avec[ind]) / sqrt(invwt);
						double b1 = (m_bvec[ind2] - m_bvec[ind]) / sqrt(invwt);
						double x2 = (0);
						double y2 = (1);
						double l2 = (m_lvec[ind3] - m_lvec[ind]) / sqrt(invwt);
						double a2 = (m_avec[ind3] - m_avec[ind]) / sqrt(invwt);
						double b2 = (m_bvec[ind3] - m_bvec[ind]) / sqrt(invwt);
						avglabs[klabels[ind]] += sqrt((x1*x1 + y1*y1 + l1*l1 + a1*a1 + b1*b1)*(x2*x2 + y2*y2 + l2*l2 + a2*a2 + b2*b2) - (x1*x2 + y1*y2 + l1*l2 + a1*a2 + b1*b2)*(x1*x2 + y1*y2 + l1*l2 + a1*a2 + b1*b2));
					}
				}
			}
		}
		for (int i = 0; i<numk; i++)
		{
			avglabs[i] = avglabs[i] / STEP / STEP;
		}
		kseedsl.clear();
		kseedsa.clear();
		kseedsb.clear();
		kseedsx.clear();
		kseedsy.clear();
		for (int i = 0; i<numk; i++)
		{
			kseedsl.push_back(0);
			kseedsa.push_back(0);
			kseedsb.push_back(0);
			kseedsx.push_back(0);
			kseedsy.push_back(0);
		}
		for (int k = 0; k < numk; k++)
		{
			kseedsx[k] = sigmax[k] * inv[k];
			kseedsy[k] = sigmay[k] * inv[k];
			kseedsl[k] = sigmal[k] * inv[k];
			kseedsa[k] = sigmaa[k] * inv[k];
			kseedsb[k] = sigmab[k] * inv[k];
		}
		{for (int k = 0; k < numk; k++)
		{
			int xindex = 0, yindex = 0;
			if (adaptk[k] <= 0.5 || avglabs[k]<split*ratio)
			{

				kseedsx[k] = sigmax[k] * inv[k];
				kseedsy[k] = sigmay[k] * inv[k];
				kseedsl[k] = sigmal[k] * inv[k];
				kseedsa[k] = sigmaa[k] * inv[k];
				kseedsb[k] = sigmab[k] * inv[k];
				adaptk[k] = sqrt(ratio / avglabs[k]);// ratio*STEP*STEP/avglabs_original[k]
				adaptk[k] = max(0.5, adaptk[k]);
				adaptk[k] = min(2, adaptk[k]);
			}
			//-------------------------------------------------------
			// If segment size is too large, split it and calculate four new seeds.
			//-------------------------------------------------------
			else
			{
				xindex = (int)(sigmax[k] * inv[k]);
				yindex = (int)(sigmay[k] * inv[k]);
				adaptk[k] = max(0.5, adaptk[k] / 2);
				int x1 = (int)(xindex - min(1, adaptk[k])*STEP / 2);
				if (x1<0)
				{
					x1 = 0;
				}
				int y1 = (int)(yindex + min(1, adaptk[k])*STEP / 2);
				if (y1 >= m_height)
				{

					y1 = m_height - 1;
				}
				int x2 = (int)(xindex + min(1, adaptk[k])*STEP / 2);
				if (x2 >= m_width)
				{
					x2 = m_width - 1;
				}
				int y2 = (int)(yindex + min(1, adaptk[k])*STEP / 2);
				if (y2 >= m_height)
				{
					y2 = m_height - 1;
				}
				int x3 = (int)(xindex - min(1, adaptk[k])*STEP / 2);
				if (x3<0)
				{
					x3 = 0;
				}
				int y3 = (int)(yindex - min(1, adaptk[k])*STEP / 2);
				if (y3<0)
				{
					y3 = 0;
				}
				int x4 = (int)(xindex + min(1, adaptk[k])*STEP / 2);
				if (x4 >= m_width)
				{
					x4 = m_width - 1;
				}
				int y4 = (int)(yindex - min(1, adaptk[k])*STEP / 2);
				if (y4<0)
				{
					y4 = 0;
				}
				kseedsx[k] = x1;
				kseedsy[k] = y1;
				kseedsl[k] = m_lvec[y1*m_width + x1];
				kseedsa[k] = m_avec[y1*m_width + x1];
				kseedsb[k] = m_bvec[y1*m_width + x1];
				kseedsx.push_back(x2);
				kseedsx.push_back(x3);
				kseedsx.push_back(x4);
				kseedsy.push_back(y2);
				kseedsy.push_back(y3);
				kseedsy.push_back(y4);
				kseedsl.push_back(m_lvec[y2*m_width + x2]);
				kseedsl.push_back(m_lvec[y3*m_width + x3]);
				kseedsl.push_back(m_lvec[y4*m_width + x4]);
				kseedsa.push_back(m_avec[y2*m_width + x2]);
				kseedsa.push_back(m_avec[y3*m_width + x3]);
				kseedsa.push_back(m_avec[y4*m_width + x4]);
				kseedsb.push_back(m_bvec[y2*m_width + x2]);
				kseedsb.push_back(m_bvec[y3*m_width + x3]);
				kseedsb.push_back(m_bvec[y4*m_width + x4]);
				adaptk.push_back(adaptk[k]);
				adaptk.push_back(adaptk[k]);
				adaptk.push_back(adaptk[k]);
				inv.push_back(0);
				inv.push_back(0);
				inv.push_back(0);
			}
		}
		}
	}
	else
	{
		kseedsl.clear();
		kseedsa.clear();
		kseedsb.clear();
		kseedsx.clear();
		kseedsy.clear();
		for (int i = 0; i<numk; i++)
		{
			kseedsl.push_back(0);
			kseedsa.push_back(0);
			kseedsb.push_back(0);
			kseedsx.push_back(0);
			kseedsy.push_back(0);
		}
		for (int k = 0; k < numk; k++)
		{
			kseedsx[k] = sigmax[k] * inv[k];
			kseedsy[k] = sigmay[k] * inv[k];
			kseedsl[k] = sigmal[k] * inv[k];
			kseedsa[k] = sigmaa[k] * inv[k];
			kseedsb[k] = sigmab[k] * inv[k];
		}
	}
}

//===========================================================================
/// DoSuperpixelSegmentation_ForGivenSuperpixelSize
//===========================================================================
void CSS::DoSuperpixelSegmentation_ForGivenSuperpixelSize(
	Mat*&                       img,
	int*&                       klabels,
	int&                        numlabels,
	const int&                  superpixelsize,
	const double&               compactness,
	const double&               merge,
	const double&               split,
	const int&                  speed)
{
	//------------------------------------------------
	const int STEP = sqrt(double(superpixelsize)) + 0.5;
	m_step = STEP;
	//------------------------------------------------
	m_width = img->cols;
	m_height = img->rows;
	int sz = m_width*m_height;
	//klabels.resize( sz, -1 );
	//--------------------------------------------------
	if (!klabels) klabels = new int[sz];
	for (int s = 0; s < sz; s++) klabels[s] = -1;
	//--------------------------------------------------
	if (1)//LAB, the default option
	{
		DoRGBtoLABConversion(img, m_lvec, m_avec, m_bvec);
	}
	else//RGB
	{
		m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz];
		for (int y = 0; y < m_height; y++)
		{
			const uchar* p = img->ptr<uchar>(y);
			for (int x = 0; x < m_width; x++)
			{
				int index = y*m_width + x;
				m_lvec[index] = (int)p[3 * x + 2]; //R
				m_avec[index] = (int)p[3 * x + 1]; //G
				m_bvec[index] = (int)p[3 * x];     //B
			}
		}
	}
	//--------------------------------------------------
	bool perturbseeds(false);
	vector<double> edgemag(0);
	if (perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenStepSize(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, perturbseeds, edgemag);
	int numk = kseedsl.size();
	vector<double> adaptk(numk, 1);
	vector<double> newadaptk(numk, 1);
	for (int itr = 0; itr<iteration; itr++)
	{
		PerformSuperpixelMSLIC(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, edgemag, adaptk, compactness, speed);
		numlabels = kseedsl.size();
		int* nlabels = new int[sz];
		EnforceLabelConnectivity(itr, adaptk, newadaptk, klabels, m_width, m_height, nlabels, numlabels, double(sz) / double(STEP*STEP), merge);
		SuperpixelSplit(itr, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, nlabels, STEP, edgemag, adaptk, numlabels, compactness, split);
		{for (int i = 0; i < sz; i++)
			klabels[i] = nlabels[i]; }
		if (nlabels) delete[] nlabels;
	}
}

//===========================================================================
/// DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(entry)
//===========================================================================
void CSS::DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(
	Mat*&                       img,
	int*&                       klabels,
	int&                        numlabels,
	const int&                  K,//required number of superpixels
	const double&               compactness,
	const double&               merge,
	const double&               split,
	const int&                  speed
	)//weight given to spatial distance
{
	const int superpixelsize = 0.5 + double(img->rows*img->cols) / double(K);
	DoSuperpixelSegmentation_ForGivenSuperpixelSize(img, klabels, numlabels, superpixelsize, compactness, merge, split, speed);
}

//===========================================================================
/// TransformImages
//============================================================================
void CSS::TransformImages(
	Mat**&                      imgs,
	const int&                  frame_num)
{
	//--------------------------------------------------
	DoRGBtoLABConversion(imgs, m_lvec, m_avec, m_bvec);
	//--------------------------------------------------
	bool perturbseeds(false);
	vector<double> edgemag(0);
	if (perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, m_frame, edgemag);

}

//===========================================================================
/// GenerateOutput
///
/// GenerateOutput of supervoxels as images
//============================================================================
void CSS::GenerateOutput(
	char*&                      output_path,
	int*&                       labels,
	const int&                  numlabels,
	const int&                  total_frames,
	const bool&                 debug_info)
{
	// Prepare vector of random/unique values for access by counter
	vector<int> randomNumbers;

	unsigned int r = rand() % 10000000; //seed for rand_r, start index for randomNumbers...
	juRandomPermuteRange(16777215, randomNumbers, &r);

	m_frame = total_frames;
	char savepath[1024];
	Mat** output = new Mat*[m_frame];

	int* m_rvec = new int[numlabels];
	int* m_gvec = new int[numlabels];
	int* m_bvec = new int[numlabels];

	// write out the ppm files.

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

	uchar* p;
	for (int i = 0; i < m_frame; i++) {
#ifdef _WIN32
		sprintf(savepath, "%s\\%05d.png", output_path, i + 1);
#else
		sprintf(savepath, "%s/%05d.png", output_path, i + 1);
#endif
		output[i] = new Mat(m_height, m_width, CV_8UC3, Scalar(0, 0, 0));
		for (int y = 0; y < m_height; y++) {
			p = output[i]->ptr<uchar>(y);
			for (int x = 0; x < m_width; x++) {
				int label = labels[y * m_width + x + i * (m_width * m_height)];
				p[3 * x] = m_bvec[label];
				p[3 * x + 1] = m_gvec[label];
				p[3 * x + 2] = m_rvec[label];
			}
		}
		imwrite(savepath, *output[i]);
		if (debug_info) printf("save --> %s\n", savepath);
	}

	printf("Num of supervoxels: %d\n", numlabels);

	for (int i = 0; i < m_frame; i++)
		delete output[i];
	delete[] output;
	delete[] m_rvec;
	delete[] m_gvec;
	delete[] m_bvec;
}

//===========================================================================
/// AssignLabels
/// calculate klabels from kseeds
//===========================================================================
void CSS::AssignLabels(
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
	int sz = m_width * m_height * m_frame;
	size_t numk = kseedsl.size();
	adaptk.resize(numk, 1);
	invwt = 1.0 / ((STEP / M)*(STEP / M));//D^2 = dc^2 + (Nc/Ns)^2 * ds^2. here invwt = Nc/Ns = M / STEP.
	//M large -> more compact, M small -> better boundary recall

	int step2 = pow(double(sz) / double(numk) + 0.5, 1.0 / 3.0) + 0.5;

	vector<double> distvec(sz, DBL_MAX);
	for (int i = 0; i < sz; i++) klabels[i] = -1;

	energy = 0;
	//int num = 0;

	int x1, y1, x2, y2, z1, z2;
	double l, a, b;
	double dist;
	double distxyz;

	distvec.assign(sz, DBL_MAX);
	for (int i = 0; i < rvt.size(); i++){
		rvt[i].clear();
		rvt[i].resize(0);
	}
	rvt.clear();
	rvt.resize(kseedsl.size());

	//-----------------------------------------------------------------
	// Assign labels -> klabels
	//-----------------------------------------------------------------
	for (int n = 0; n < numk; n++)
	{
		int offset = (int)speed*adaptk[n] * step2;

		//2offset*2offset*2offset search window
		x1 = max(0.0, kseedsx[n] - offset);
		x2 = min((double)m_width, kseedsx[n] + offset);
		y1 = max(0.0, kseedsy[n] - offset);
		y2 = min((double)m_height, kseedsy[n] + offset);
		z1 = max(0.0, kseedsz[n] - offset);
		z2 = min((double)m_frame, kseedsz[n] + offset);

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
						(z - kseedsz[n])*(z - kseedsz[n]) * timedist * timedist;

					dist += distxyz * invwt;

					if (dist < distvec[i])
					{
						if (distvec[i] != DBL_MAX){//not the first time assigned
							energy -= distvec[i];
							//num--;
						}
						distvec[i] = dist;
						klabels[i] = n; //label of voxel i is the id of current seed
						//rvt[n].push_back(i);
						energy += dist;
						//num++;
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
				int z = i / (m_width * m_height);

				dist = (l - kseedsl[n])*(l - kseedsl[n]) +
					(a - kseedsa[n])*(a - kseedsa[n]) +
					(b - kseedsb[n])*(b - kseedsb[n]);
				distxyz = (x - kseedsx[n])*(x - kseedsx[n]) +
					(y - kseedsy[n])*(y - kseedsy[n]) +
					(z - kseedsz[n])*(z - kseedsz[n]) * timedist * timedist;
				//------------------------------------------------------------------------
				dist += distxyz * invwt;
				//------------------------------------------------------------------------
				if (dist < distvec[i])
				{
					distvec[i] = dist;
					klabels[i] = n; //label of voxel i is the id of current seed
					//rvt[n].push_back(i);
					energy += dist;
					//num++;
				}
			}
		}
	}
	for (int i = 0; i < sz; i++){
		rvt[klabels[i]].push_back(i);
	}
	//int sum = 0;
	//for(int i = 0; i < numk; i++){
	//    sum += rvt[i].size();
	//}
	cout << "Invalid: " << invalid << " in " << sz << endl;
	cout << "energy: " << energy << endl;
	//cout << "num: " << num << endl;//checked
	//cout << "sum: " << sum << endl;//checked
}

//===========================================================================
/// AssignLabelsAndLlyod
/// kseeds  ->  klabels  ->  kseeds (centroids)
//===========================================================================
void CSS::AssignLabelsAndLloyd(
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
	int sz = m_width * m_height * m_frame;
	size_t numk = kseedsl.size();
	adaptk.resize(numk, 1);
	invwt = 1.0 / ((STEP / M)*(STEP / M));//D^2 = dc^2 + (Nc/Ns)^2 * ds^2. here invwt = Nc/Ns = M / STEP.
	//M large -> more compact, M small -> better boundary recall

	int step2 = pow(double(sz) / double(numk) + 0.5, 1.0 / 3.0) + 0.5;

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
			int offset = (int)speed*adaptk[n] * step2;

			//2offset*2offset*2offset search window
			x1 = max(0.0, kseedsx[n] - offset);
			x2 = min((double)m_width, kseedsx[n] + offset);
			y1 = max(0.0, kseedsy[n] - offset);
			y2 = min((double)m_height, kseedsy[n] + offset);
			z1 = max(0.0, kseedsz[n] - offset);
			z2 = min((double)m_frame, kseedsz[n] + offset);

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
							(z - kseedsz[n])*(z - kseedsz[n]) * timedist * timedist;
						//------------------------------------------------------------------------
						dist += distxyz * invwt;//dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
						//------------------------------------------------------------------------
						if (dist < distvec[i])
						{
							distvec[i] = dist;
							klabels[i] = n; //label of voxel i is the id of current seed
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
					int z = i / (m_width * m_height);

					dist = (l - kseedsl[n])*(l - kseedsl[n]) +
						(a - kseedsa[n])*(a - kseedsa[n]) +
						(b - kseedsb[n])*(b - kseedsb[n]);
					distxyz = (x - kseedsx[n])*(x - kseedsx[n]) +
						(y - kseedsy[n])*(y - kseedsy[n]) +
						(z - kseedsz[n])*(z - kseedsz[n]) * timedist * timedist;
					//------------------------------------------------------------------------
					dist += distxyz * invwt;
					//------------------------------------------------------------------------
					if (dist < distvec[i])
					{
						distvec[i] = dist;
						klabels[i] = n; //label of voxel i is the id of current seed
					}
				}
			}
		}
		cout << "Invalid: " << invalid << " in " << sz << endl;


		//-----------------------------------------------------------------
		// Lloyd
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		vector<double> clustersize(numk, 0);
		vector<double> inv(numk, 0);//to store 1/clustersize[k] values

		vector<double> sigmal(numk, 0);
		vector<double> sigmaa(numk, 0);
		vector<double> sigmab(numk, 0);
		vector<double> sigmax(numk, 0);
		vector<double> sigmay(numk, 0);
		vector<double> sigmaz(numk, 0);

		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		sigmaz.assign(numk, 0);
		clustersize.assign(numk, 0);
		inv.assign(numk, 0);


		int ind = 0;
		for (int f = 0; f < m_frame; f++)
		{
			for (int r = 0; r < m_height; r++)
			{
				for (int c = 0; c < m_width; c++)
				{
					if (klabels[ind] == -1){
						ind++;
						continue;
					}
					sigmal[klabels[ind]] += m_lvec[ind];
					sigmaa[klabels[ind]] += m_avec[ind];
					sigmab[klabels[ind]] += m_bvec[ind];
					sigmax[klabels[ind]] += c;
					sigmay[klabels[ind]] += r;
					sigmaz[klabels[ind]] += f;

					clustersize[klabels[ind]] += 1.0;
					ind++;
				}
			}
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
	cout << "# Kseeds: " << kseedsl.size() << endl;
}

//===========================================================================
/// Llyod only
/// calculate kseeds from klabels (as centroid)
//===========================================================================
void CSS::Lloyd(
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
	vector<double> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values

	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<double> sigmaz(numk, 0);

	sigmal.assign(numk, 0);
	sigmaa.assign(numk, 0);
	sigmab.assign(numk, 0);
	sigmax.assign(numk, 0);
	sigmay.assign(numk, 0);
	sigmaz.assign(numk, 0);
	clustersize.assign(numk, 0);
	inv.assign(numk, 0);


	int ind = 0;
	for (int f = 0; f < m_frame; f++)
	{
		for (int r = 0; r < m_height; r++)
		{
			for (int c = 0; c < m_width; c++)
			{
				sigmal[klabels[ind]] += m_lvec[ind];
				sigmaa[klabels[ind]] += m_avec[ind];
				sigmab[klabels[ind]] += m_bvec[ind];
				sigmax[klabels[ind]] += c;
				sigmay[klabels[ind]] += r;
				sigmaz[klabels[ind]] += f;

				clustersize[klabels[ind]] += 1.0;
				ind++;
			}
		}
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
/// Llyod with density
/// calculate kseeds from klabels (as centroid)
//===========================================================================
void CSS::LloydWithDensity(
	vector<double>&             kseedsl,
	vector<double>&             kseedsa,
	vector<double>&             kseedsb,
	vector<double>&             kseedsx,
	vector<double>&             kseedsy,
	vector<double>&             kseedsz,
	int*&                       klabels,
	const int&                  STEP,
	const double&               M,
	const int&					withcontrol)
{
	size_t numk = kseedsl.size();
	invwt = 1.0 / ((STEP / M)*(STEP / M));

	//-----------------------------------------------------------------
	// Lloyd
	// Recalculate the centroid and store in the seed values
	//-----------------------------------------------------------------
	vector<double> clusterdensity(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<double> sigmaz(numk, 0);

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
	clusterdensity.assign(numk, 0);
	inv.assign(numk, 0);

	double densityE;

	int ind = 0;
	for (int f = 0; f < m_frame; f++)
	{
		for (int r = 0; r < m_height; r++)
		{
			for (int c = 0; c < m_width; c++)
			{
				densityE = m_densityvec[ind];
				sigmal[klabels[ind]] += m_lvec[ind] * densityE;
				sigmaa[klabels[ind]] += m_avec[ind] * densityE;
				sigmab[klabels[ind]] += m_bvec[ind] * densityE;
				sigmax[klabels[ind]] += c * densityE;
				sigmay[klabels[ind]] += r * densityE;
				sigmaz[klabels[ind]] += f * densityE;
				/*if (_isnan(sigmaz[klabels[ind]]))
					cout << "nan " << densityE << " " << m_densityvec[ind] << " " << klabels[ind] << endl;*/
				clusterdensity[klabels[ind]] += densityE;
				if (withcontrol){
					sigmaluni[klabels[ind]] += m_lvec[ind];
					sigmaauni[klabels[ind]] += m_avec[ind];
					sigmabuni[klabels[ind]] += m_bvec[ind];
					sigmaxuni[klabels[ind]] += c;
					sigmayuni[klabels[ind]] += r;
					sigmazuni[klabels[ind]] += f;
					clusteruni[klabels[ind]] += 1;
				}
				ind++;
			}
		}
	}

	for (int k = 0; k < numk; k++)
	{
		if (clusterdensity[k] <= 0) clusterdensity[k] = 1;
		inv[k] = 1.0 / clusterdensity[k];//computing inverse now to multiply, than divide later
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
			//if (k == 126){
			//	cout << k << " " << sigmaz[k] << " " << inv[k] << " " << clusterdensity[k] << endl;
			//	//					nan					0					nan
			//}
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
				/*
				double dist = (sigmaluni[k] - kseedsl[k])*(sigmaluni[k] - kseedsl[k]) +
				(sigmaauni[k] - kseedsa[k])*(sigmaauni[k] - kseedsa[k]) +
				(sigmabuni[k] - kseedsb[k])*(sigmabuni[k] - kseedsb[k]) +
				((sigmaxuni[k] - kseedsx[k])*(sigmaxuni[k] - kseedsx[k]) +
				(sigmayuni[k] - kseedsy[k])*(sigmayuni[k] - kseedsy[k]) +
				(sigmazuni[k] - kseedsz[k])*(sigmazuni[k] - kseedsz[k])*timedist*timedist) * invwt;
				cout << "dist: " << dist << ", dist2: " << dist2 << endl;//verified
				*/
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
				double maxden = -1e30;
				int mindex = -1;
				double mdist = -1;
				for (int i = 0; i < rvt[k].size(); i++)
				{
					int x = rvt[k][i] % m_width;
					int y = (rvt[k][i] / m_width) % m_height;
					int z = rvt[k][i] / (m_width * m_height);
					double dist = (sigmaluni[k] - m_lvec[rvt[k][i]])*(sigmaluni[k] - m_lvec[rvt[k][i]]) +
						(sigmaauni[k] - m_avec[rvt[k][i]])*(sigmaauni[k] - m_avec[rvt[k][i]]) +
						(sigmabuni[k] - m_bvec[rvt[k][i]])*(sigmabuni[k] - m_bvec[rvt[k][i]]) +
						((sigmaxuni[k] - x)*(sigmaxuni[k] - x) +
						(sigmayuni[k] - y)*(sigmayuni[k] - y) +
						(sigmazuni[k] - z)*(sigmazuni[k] - z)*timedist*timedist) * invwt;
					//cout << "dist: " << dist << ", dist2: " << dist2 << endl;
					if (dist > dist2) continue;
					if (m_densityvec[rvt[k][i]] > maxden){
						maxden = m_densityvec[rvt[k][i]];
						mindex = rvt[k][i];
						mdist = dist;
					}
				}
				//cout << "rvt[k]: " << rvt[k].size() << ", clusteruni[k]: " << clusteruni[k] << ", mindex: " << mindex << ", dist: " << mdist << ", dist1: " << dist1 << ", dist2: " << dist2 << endl;
				if (mindex != -1){
					kseedsl[k] = m_lvec[mindex];
					kseedsa[k] = m_avec[mindex];
					kseedsb[k] = m_bvec[mindex];
					kseedsx[k] = mindex % m_width;
					kseedsy[k] = (mindex / m_width) % m_height;
					kseedsz[k] = mindex / (m_width * m_height);
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