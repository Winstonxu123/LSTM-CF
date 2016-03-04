#include <vector>
#include "caffe/layers/mask_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	Dtype* top_data = top[0]->mutable_gpu_data();
	switch (this->layer_param_.mask_param().masktype()) {
	case MaskParameter_MaskType_VERTICAL:
		if (this->layer_param_.mask_param().direction() 
				== MaskParameter_Direction_TOP2BOTTOM)
		{
			caffe_gpu_set(width_, (Dtype)0, top_data);
			top_data += top[0]->offset(1);
			caffe_gpu_set((height_ - 1)* width_, (Dtype)1, top_data);
		}
		else if (this->layer_param_.mask_param().direction() 
			== MaskParameter_Direction_BOTTOM2TOP)
		{
			caffe_gpu_set((height_ - 1)* width_, (Dtype)1, top_data);
			top_data += top[0]->offset(1);
			caffe_gpu_set(width_, (Dtype)0, top_data);
		}
		break;
	case MaskParameter_MaskType_HORIZONTAL:
		if (this->layer_param_.mask_param().direction()
				== MaskParameter_Direction_LEFT2RIGHT)
		{
			for (int i = 0; i < height_; i++)
			{
				caffe_gpu_set(1, (Dtype)0, top_data);
				caffe_gpu_set(width_ - 1, (Dtype)1, top_data + 1);
				top_data += top[0]->offset(1);
			}
		}
		else if (this->layer_param_.mask_param().direction()
					 == MaskParameter_Direction_RIGHT2LEFT)
		{
			for (int i = 0; i < height_; i++)
			{	
				caffe_gpu_set(width_ - 1, (Dtype)1, top_data + 1);
				caffe_gpu_set(1, (Dtype)0, top_data + width_ - 1);
				top_data += top[0]->offset(1);
			}			
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskLayer);
}