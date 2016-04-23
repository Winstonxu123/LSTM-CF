addpath('.');
addpath('./matlab/caffe');

%specific the path to the model and the weights
model = '/home/ganyk/caffe-lstm-ganyk/models/deeplab/deeplab_lstm_v4_conv7.prototxt';
weights = '/home/ganyk/project/2016-3-4/caffe_deeplab_lstm_v4_conv7_train_iter_40000.caffemodel';

%set mode and device
caffe('set_mode_gpu');
%caffe.set_device(0);

caffe('init', model, weights, 'test');
scores = caffe('forward');
