#include <vector>
#include "caffe/layers/mask_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	Dtype* top_data = top[0]->mutable_gpu_data();
	caffe_gpu_set(width_, (Dtype)0, top_data);
	top_data += top[0]->offset(1);
	caffe_gpu_set((height_ - 1)* width_, (Dtype)1, top_data);
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskLayer);
}