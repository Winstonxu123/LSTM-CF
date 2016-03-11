#ifndef CAFFE_MAT_WRITE_LAYER_HPP
#define CAFFE_MAT_WRITE_LAYER_HPP

namespace caffe {
template <typename Dtype>
class MatWriteLayer : public Layer<Dtype> {
 public:
  explicit MatWriteLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "MatWrite"; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int iter_;
  int period_;
  string prefix_;
  vector<string> fnames_;
};

}

#endif