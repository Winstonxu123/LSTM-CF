#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/repmat_layer.hpp"

namespace caffe {

template <typename Dtype>
void RepMatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const RepMatParameter& repmat_param = this->layer_param_.repmat_param();
  rep_w = repmat_param.rep_w();
  rep_h = repmat_param.rep_h();
  rep_c = repmat_param.rep_c();
  rep_n = repmat_param.rep_n();
}

template <typename Dtype>
void RepMatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //get the shape of bottom
  vector<int> bottom_shape = bottom[0]->shape();


  //If width and height aren't there, we set them to 1;
  if(bottom_shape.size() < 4)
  {
    int curr=bottom_shape.size();
    for(int i = curr; i<4; i++){
      bottom_shape.push_back(1);
    }
  }

  //set variables
  bottom_n = bottom_shape[0];
  bottom_c = bottom_shape[1];
  bottom_h = bottom_shape[2];
  bottom_w = bottom_shape[3];
  //multiply
  bottom_shape[0] = bottom_shape[0]*rep_n;
  bottom_shape[1] = bottom_shape[1]*rep_c;
  bottom_shape[2] = bottom_shape[2]*rep_h;
  bottom_shape[3] = bottom_shape[3]*rep_w;

  top_n = bottom_shape[0];
  top_c = bottom_shape[1];
  top_h = bottom_shape[2];
  top_w = bottom_shape[3];

  //reshape top
  top[0]->Reshape(bottom_shape);
}

template <typename Dtype>
void RepMatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  for(int i = 0; i < top[0]->count(); i++) {
    int w = (i % top_w) % bottom_w;
    int h = ((i / top_w) % top_h) % bottom_h;     
    int c = (((i / top_w) / top_h) % top_c) % bottom_c;
    int n = (((i / top_w) / top_h) / top_c) % bottom_n;
    top_data[i] = bottom_data[bottom[0]->offset(n,c,h,w)];
  }
}

template <typename Dtype>
void RepMatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  caffe_set(bottom[0]->count(), static_cast<Dtype>(0.0), bottom_diff);
  for(int i=0; i<top[0]->count(); i++) {
    int w = (i % top_w) % bottom_w;
    int h = ((i / top_w) % top_h) % bottom_h;     
    int c = (((i / top_w) / top_h) % top_c) % bottom_c;
    int n = (((i / top_w) / top_h) / top_c) % bottom_n;
    bottom_diff[bottom[0]->offset(n,c,h,w)] += top_diff[i];
  }
}

#ifdef CPU_ONLY
STUB_GPU(RepMatLayer);
#endif

INSTANTIATE_CLASS(RepMatLayer);
REGISTER_LAYER_CLASS(RepMat);

}  // namespace caffe
