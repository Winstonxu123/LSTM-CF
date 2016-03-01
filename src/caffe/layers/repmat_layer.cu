#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/repmat_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RepMatForward(const int nthreads, const Dtype* bottom_data, Dtype* top_data, 
const int top_n, const int top_c, const int top_h, const int top_w, 
const int bottom_n, const int bottom_c, const int bottom_h, const int bottom_w) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = (index % top_w) % bottom_w;
    int h = ((index / top_w) % top_h) % bottom_h;     
    int c = (((index / top_w) / top_h) % top_c) % bottom_c;
    int n = (((index / top_w) / top_h) / top_c) % bottom_n;
    top_data[index] = bottom_data[ w + bottom_w*(h + bottom_h*(c + bottom_c*n)) ];





  }

}

template <typename Dtype>
void RepMatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  RepMatForward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()),CAFFE_CUDA_NUM_THREADS>>>(
      top[0]->count(), bottom_data, top_data, top_n, top_c, top_h, top_w, 
      bottom_n, bottom_c, bottom_h, bottom_w); 
}


template <typename Dtype>
__global__ void RepMatBackward(const int nthreads, const Dtype* top_diff, Dtype* bottom_diff, 
const int top_n, const int top_c, const int top_h, const int top_w, 
const int bottom_n, const int bottom_c, const int bottom_h, const int bottom_w,
int rep_n, int rep_c, int rep_h, int rep_w) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % bottom_w;
    int h = (index / bottom_w) % bottom_h;     
    int c = ((index / bottom_w) / bottom_h) % bottom_c;
    int n = ((index / bottom_w) / bottom_h) / bottom_c;
    for(int n_o = 0; n_o<rep_n; n_o++){
      for(int c_o = 0; c_o<rep_c; c_o++){
        for(int h_o = 0; h_o<rep_h; h_o++){
          for(int w_o = 0; w_o<rep_w; w_o++){
            int offset = (((n_o*bottom_n + n)*top_c + c_o*bottom_c + c)*top_h + h_o*bottom_h + h)*top_w + w_o*bottom_w + w;
            bottom_diff[index] += top_diff[offset];
          }
        }
      }
    }



  }

}


template <typename Dtype>
void RepMatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
 
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0.0), bottom_diff);
  RepMatBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->count(), top_diff, bottom_diff, top_n, top_c, top_h, top_w, 
      bottom_n, bottom_c, bottom_h, bottom_w,
      rep_n, rep_c, rep_h, rep_w); 

//Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(RepMatLayer);

}  // namespace caffe
