#ifndef CAFFE_LAST_ROW_LAYER_HPP
#define CAFFE_LAST_ROW_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

template <typename Dtype>
class LastRowLayer : public Layer<Dtype>{
public:
	explicit LastRowLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    
  	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      	const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "LastRow"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

}

#endif