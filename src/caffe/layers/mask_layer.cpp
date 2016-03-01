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
	caffe_set(width_, (Dtype)0, top_data);
	top_data += top[0]->offset(1);
	caffe_set((height_ - 1)* width_, (Dtype)1, top_data);
}

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}

