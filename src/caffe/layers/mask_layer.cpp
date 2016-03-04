#include <vector>
#include "caffe/layers/mask_layer.hpp"

namespace caffe{

template <typename Dtype>
void MaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const MaskParameter& param = this->layer_param_.mask_param();
	width_ = param.width();
	height_ = param.height();

	vector<int> top_shape(2);

	top_shape[0] = height_;
	top_shape[1] = width_;

	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	Dtype* top_data = top[0]->mutable_cpu_data();
	switch (this->layer_param_.mask_param().masktype()) {
	case MaskParameter_MaskType_VERTICAL:
		if (this->layer_param_.mask_param().direction() 
				== MaskParameter_Direction_TOP2BOTTOM)
		{
			caffe_set(width_, (Dtype)0, top_data);
			top_data += top[0]->offset(1);
			caffe_set((height_ - 1)* width_, (Dtype)1, top_data);
		}
		else if (this->layer_param_.mask_param().direction() 
			== MaskParameter_Direction_BOTTOM2TOP)
		{
			caffe_set((height_ - 1)* width_, (Dtype)1, top_data);
			top_data += top[0]->offset(1);
			caffe_set(width_, (Dtype)0, top_data);
		}
		break;
	case MaskParameter_MaskType_HORIZONTAL:
		if (this->layer_param_.mask_param().direction()
				== MaskParameter_Direction_LEFT2RIGHT)
		{
			for (int i = 0; i < height_; i++)
			{
				top_data[0] = (Dtype)0;
				caffe_set(width_ - 1, (Dtype)1, top_data + 1);
				top_data += top[0]->offset(1);
			}
		}
		else if (this->layer_param_.mask_param().direction()
					 == MaskParameter_Direction_RIGHT2LEFT)
		{
			for (int i = 0; i < height_; i++)
			{	
				caffe_set(width_ - 1, (Dtype)1, top_data + 1);
				top_data[width_ - 1] = (Dtype)0;
				top_data += top[0]->offset(1);
				
			}			
		}
	}
}

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}

