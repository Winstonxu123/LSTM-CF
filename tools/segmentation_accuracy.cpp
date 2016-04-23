#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/confusion_matrix.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
using namespace std;

int main(int argc, char* argv[])
{
	if (argc != 6)
	{
		LOG(FATAL) << "Usage: command model weights device_query iterations";
		return 1;
	}
	int device = atoi(argv[4]);
	int iter_num = atoi(argv[5]);
	//Caffe::set_phase(Caffe::TEST);
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(device);

	string feature_prefix = "/home/ganyk/results/nyu/features/";
	string label_prefix = "/home/ganyk/results/nyu/labels/";

	ofstream out1("/home/ganyk/results/nyu/accuracy.txt");
	ofstream out(argv[3]);
	ConfusionMatrix confusion_matrix;

	
	//get the net
	Net<float> caffe_test_net(argv[1], caffe::TEST);
	//get trained net
	caffe_test_net.CopyTrainedLayersFrom(argv[2]);


	const shared_ptr< Blob<float> > local_global_conv = caffe_test_net.blob_by_name("local_global_conv");
	//resize the confusion matrix
	confusion_matrix.resize(local_global_conv->channels());
	confusion_matrix.clear(); 

    int top_k = 1;  // only support for top_k = 1
	vector<Blob<float>* > bottom_vec;
	for (int iter_ = 0; iter_ < iter_num; iter_++)
	{
		LOG(INFO) << "Batch " << iter_;
		//const vector<Blob<float>*>& result = 
		caffe_test_net.Forward(bottom_vec);
		//get the pointer of two blobs
		const shared_ptr< Blob<float> > data_blob = caffe_test_net.blob_by_name("local_global_conv");
		const shared_ptr< Blob<float> > label_blob = caffe_test_net.blob_by_name("label_shrink");
		const shared_ptr< Blob<float> > label_blob1 = caffe_test_net.blob_by_name("label");
		const shared_ptr< Blob<float> > accuracy_blob = caffe_test_net.blob_by_name("accuracy");
		const float* data = data_blob->cpu_data();
		const float* label = label_blob->cpu_data();
		const float* seg_accuracy = accuracy_blob->cpu_data();
		int num = data_blob->num();
		//int dim = bottom[0]->count() / bottom[0]->num();
		int channels = data_blob->channels();
		int height = data_blob->height();
		int width = data_blob->width();
		int data_index, label_index;

		for (int i = 0; i < num; ++i) 
		{
    		for (int h = 0; h < height; ++h) 
    		{
      			for (int w = 0; w < width; ++w) 
    			{
      				// Top-k accuracy
					std::vector< std::pair<float, int> > bottom_data_vector;

					for (int c = 0; c < channels; ++c) 
					{
	  					data_index = (c * height + h) * width + w;
	  					bottom_data_vector.push_back(std::make_pair(data[data_index], c));
					}

					std::partial_sort(
	  					bottom_data_vector.begin(), bottom_data_vector.begin() + top_k,
	  					bottom_data_vector.end(), std::greater<std::pair<float, int> >());
					// check if true label is in top k predictions
					label_index = h * width + w;
					const int gt_label = static_cast<int>(label[label_index]);
					confusion_matrix.accumulate(gt_label, bottom_data_vector[0].second);
      			}
      		}

      		data += data_blob->offset(1);
      		label += label_blob->offset(1);	
      	}

      	//output confidence map
      	std::ostringstream oss;
      	oss << feature_prefix << "iter_" << iter_ << ".mat";
      	//data_blob->ToMat(oss.str().c_str(), false);

      	//clear the stringstream
      	oss.str("");

      	//output lable map
      	oss << label_prefix << "iter_" << iter_ << ".mat";
      	//label_blob1->ToMat(oss.str().c_str(), false);
      	oss.str("");
      	//output segmentation accuracy for each batch
      	//out1 << seg_accuracy[0] << " " << seg_accuracy[1] << " " << seg_accuracy[2] << " " << seg_accuracy[3] << std::endl;
	}

	//report accuracy
	LOG(INFO) << "           Pixel accuracy: " << (float)confusion_matrix.accuracy();
	LOG(INFO) << "	 Class accuracy(recall): " << (float)confusion_matrix.avgRecall(false);
	LOG(INFO) << "Class accuracy(precision): " << (float)confusion_matrix.avgPrecision();
	LOG(INFO) << "                 PixelIOU: " << (float)confusion_matrix.avgJaccard();
	
	//save confusion matrix to txt file
	
	
	for (int i = 0; i < confusion_matrix.numRows(); i++)
	{
		for (int j = 0; j < confusion_matrix.numRows(); j++)
		{
			out << confusion_matrix(i, j) << " ";
		}

		out << std::endl;
	}

	out.close();
	//out1.close();
	
	return 0;
}