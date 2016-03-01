#include <vector>

#include "caffe/layers/last_row_layer.hpp"

namespace caffe {

template <typename Dtype>
void LastRowLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	int num = bottom[0]->shape(0);
	int num1 = bottom[0]->shape(1);
	int channels = bottom[0]->shape(2);

	bottom_data += bottom[0]->offset(num - 1);
	caffe_copy(channels * num1, bottom_data, top_data);
}

template <typename Dtype>
void LastRowLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

	int num = bottom[0]->shape(0);
	int num1 = bottom[0]->shape(1);
	int channels = bottom[0]->shape(2);

	bottom_diff += bottom[0]->offset(num - 1);
	caffe_copy(channels * num1, top_diff, bottom_diff);	

}

INSTANTIATE_LAYER_GPU_FUNCS(LastRowLayer);

}