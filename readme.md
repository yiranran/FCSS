## Feature-Aware Uniform Tessellations on Video Manifold for Content-Sensitive Supervoxels.
This code is based on the TPAMI paper subject to further simplification and optimization.

If you use this code, please cite our paper:

[1] R. Yi, Z. Ye, W. Zhao, M. Yu, Y.-K. Lai and Y.-J. Liu. 
Feature-Aware Uniform Tessellations on Video Manifold for Content-Sensitive Supervoxels. 
IEEE Transactions on Pattern Analysis and Machine Intelligence. DOI (identifier) 10.1109/TPAMI.2020.2979714, 2020.

For more information (pdf and demo video) of this paper, please visit [this page](https://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm).


==========================================

Two projects FCSS and streamFCSS correspond to FCSS and streamFCSS algorithm in the paper, respectively.

Each program takes a video (a sequence of images) as input and produces a sequence of images of supervoxel segmentation, where supervoxel labels are encoded in color. 


They have been tested on Windows 7 64-bit OS and Ubuntu 14.04.


Environment:

	The program requires OpenCV 2.4.9.


How to compile:

	On windows, open the .sln file in Visual Studio 2013, and build solution in Release and x64.

	On ubuntu, makefile can be found in CSS and streamCSS folder, run make in each folder.


How to use the program (on linux as an example):

	Run "./FCSS [input_folder] [output_folder] [sv_num]"

    	"./streamFCSS [input_folder] [output_folder] [sv_num]"


Parameters in order are:

	[input_folder] -- the input folder of video frames in ppm format (png format for windows). Image filenames should be in the format "00001.ppm". Folder "girl" is an example input folder.

	[output_folder] -- the output folder of results, where supervoxel labels are encoded in color. The folder will be created.

	[sv_num] -- the desired number of supervoxels. The value should be in the range 50 ~ 10000.


An example:

	./CSS girl girl-css 300

	./streamCSS girl girl-stream 300


==========================================

We provide an example testing video in the "girl" folder.

The example testing video is taken from SegTrackv2 dataset in LIBSVXv4.0 format:

http://www.cs.rochester.edu/~cxu22/d/libsvx/

You can find more testing videos in the link above.


The program may take longer time for longer videos and in the streaming mode.