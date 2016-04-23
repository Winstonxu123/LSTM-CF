#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/transpose_layer.hpp"

namespace caffe {

template <typename Dtype>
void TransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  const BlobShape& top_blob_shape = this->layer_param_.transpose_param().shape();
  for (int i = 0; i < top_blob_shape.dim_size(); i++)
  {
    top_shape.push_back(top_blob_shape.dim(i));
  }

  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());  
}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  int spatial_dim, channels;
  spatial_dim = bottom[0]->height() * bottom[0]->width();
  channels = bottom[0]->channels();
  switch (this->layer_param_.transpose_param().transposetype()) {
    case TransposeParameter_TransposeType_CONVTOLSTM:
      for (int n = 0; n < bottom[0]->num(); ++n) 
      {
        const int offset = bottom[0]->count() / bottom[0]->num() * n;
        for (int c = 0; c < channels; ++c) 
        {
          for (int i = 0; i < spatial_dim; ++i) 
          {
            top_data[offset + i * channels + c]
                = bottom_data[offset + c * spatial_dim + i];
          }
        }
      }
      break;
    case TransposeParameter_TransposeType_LSTMTOCONV:
        for (int n = 0; n < bottom[0]->num(); ++n) 
        {
          const int offset = bottom[0]->count() / bottom[0]->num() * n;
          for (int c = 0; c < channels; ++c) 
          {
            for (int i = 0; i < spatial_dim; ++i) 
            {
              top_data[offset + c * spatial_dim + i] =
                  bottom_data[offset + i * channels + c];
            }
          }
        }
      break;
  }
  
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int spatial_dim, channels;
	spatial_dim = bottom[0]->height() * bottom[0]->width();
	channels = bottom[0]->channels();

  switch (this->layer_param_.transpose_param().transposetype()) {
    case TransposeParameter_TransposeType_CONVTOLSTM:
    	for (int n = 0; n < bottom[0]->num(); ++n) 
      {
    		const int offset = bottom[0]->count() / bottom[0]->num() * n;
    		for (int c = 0; c < channels; ++c) 
        {
    		  for (int i = 0; i < spatial_dim; ++i) 
          {
    		    bottom_diff[offset + c * spatial_dim + i] =
    		        top_diff[offset + i * channels + c];
    		  }
    		}
    	}
    break;
    case TransposeParameter_TransposeType_LSTMTOCONV:
      for (int n = 0; n < bottom[0]->num(); ++n) 
      {
        const int offset = bottom[0]->count() / bottom[0]->num() * n;
        for (int c = 0; c < channels; ++c) 
        {
          for (int i = 0; i < spatial_dim; ++i) 
          {
            bottom_diff[offset + i * channels + c]
                = top_diff[offset + c * spatial_dim + i];
          }
        }
      }
    break;
  }
}

#ifdef CPU_ONLY
STUB_GPU(TransposeLayer);
#endif

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe
