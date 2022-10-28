# Generative Adversarial Network application based on libtorch

This repository is a "general" application for implementing Generative Adversarial Network  using libtorch.
It uses [yaml](https://github.com/jbeder/yaml-cpp) [input files](INPUT) for configuration of the trainning process. 

## Dependencies 

* libtorch from [Pytorch](https://pytorch.org/)

* libpng  from [libpng](http://www.libpng.org/pub/png/libpng.html)

## CMake variables for configuration

|name|values|description|
|----|------|-----------|
|DISCRIMINATOR|MPS,TTN,DGAN...|The torch module to use as discriminator. One can create custom modules and add it to the namespace custom_models.|
|GENERATOR|GGAN...|The torch module to use as generator. One can create custom modules and add it to the namespace custom_models.|
|DATASET|FP2,IRIS,CMNIST,MNIST|The torch dataset to use for training the generator. One can create custom datasets and add it to the namespace custom_models::datasets.|
|TRAIN|ON,OFF|Perform training on the model.|
|TEST|ON,OFF|Perform testing on the model.|

### Note on this

Custom modules and custom datasets must  have a constructor of the form 

|Object|Object constructor|
|-------------|------------------------------------|
|Custom_Module|Custom_ModuleImpl(YAML::Node config)|
|Custom_Dataset|Custom_DatasetImpl(const std::string& root, Mode mode = Mode::kTrain)|

### Testing the generator
 
The code test the generator by producing a image from the resulting tensor. The later tensor is the result of applying the generator to random data. This only work for the case of generators that produce images with certain restrictions. 
## Install and Execute

### Build and install
```
git clone git@github.com:EddyTheCo/GAN.git GAN 
cd GAN 
cmake -DCMAKE_INSTALL_PREFIX=install -DTEST=ON -DTRAIN=ON -DDISCRIMINATOR=DGAN -DGENERATOR=GGAN -DDATASET=MNIST -DCUSTOM_MODULES="DGAN;GGAN" ../
cmake --build . --target install -- -j4
```

### Execute
```
./install/bin/GAN install/INPUT/gan_mnist.yaml
```
