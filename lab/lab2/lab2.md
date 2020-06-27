# 实验二

---

## BANGC 算子与风格迁移实验（下）

### 模型量化

由于MLU270对于mlp和conv算子可使用int8数据类型进行计算以提高效率，所以要将原始的float32数据类型的pb模型量化成为int 类型，所以 /opt/AICSE-demo-student/demo/style_transfer_bcl/tools/
fppb_to_intpb 目录中执行：python fppb_to_intpb.py udnie_int8.ini。

其中udnie_int8.ini 中的关键内容如下：

```bash
[preprocess]
mean = 0.0, 0.0, 0.0 
std = 1.0 
color_mode = rgb 
crop = 256, 256 

[config]
quantization_type = int8
int_op_list = Conv, FC, LRN 

[model]
output_tensor_names = add_37:0
original_models_path = ../../models/pb_models/udnie.pb
save_model_path = ../../models/int_pb_models/udnie_int8.pb
input_tensor_names = X_content:0

[data]
num_runs = 1
data_path = ./image_list
batch_size = 1

```

主要包括了四个部分，分别是预处理，设置了输入图片的均值，rgb通道以及图片大小(256,256)， 在config部分，设置了需要量化的过程，包括conv,FC和 LRN层，以及量化成的是int8类型， 在model部分，设定了输入输出的内容，原始模型和量化的模型的位置，最后在data部分设定了相应的数据。

最终可以看到在`/opt/AICSE-demo-student/demo/style_transfer_bcl/models/int_pb_models`中保存了以下模型：

![image-20200604165024462](https://i.loli.net/2020/06/04/hkiL8UrBAzjxQJ2.png)

至此，量化过程完成。

可以看到，量化后的模型，对比量化前的模型，其具体的差距在于

![image-20200604220047970](https://i.loli.net/2020/06/04/YKiba95VCT6mnlq.png)

conv2D变为了Int8Conv2D，故也就是在卷积过程中进行了量化。

### 在线推理程序

#### CPU推理

在CPU推理阶段，需要完成对于三个模型的调用，分别进行相应的推理，下面分别进行分析（以下关于模型的可视化使用[netron](https://github.com/lutzroeder/netron) 完成

##### udnie.pb

对于最原始的模型，不会用到我们实现的`power difference`算子，通过可视化模型可以知道，对于输入，其为：

![image-20200604165927916](https://i.loli.net/2020/06/04/aA9NzOKhvBDp54t.png)

可以看到，其输入的模块为`X_content`，而根据最终的输入，其可视化模型为：

![image-20200604170221155](https://i.loli.net/2020/06/04/DyJ1MHTPs53vfwS.png)故输出的模块为`add_37`，所以，根据此，代码中为：

```python
input_tensor = sess.graph.get_tensor_by_name('X_content:0')
output_tensor = sess.graph.get_tensor_by_name('add_37:0')

start_time = time.time()
ret =sess.run(output_tensor, feed_dict={input_tensor:[X]})
end_time = time.time()
```

可以看到，得到输入，然后传输数据X，并根据此得到输出结果并进行保存。

##### udnie_power_diff.pb

这里采用了我们实现的power_diff算子来进行计算，其输入输出与之前的udnie.pb是相同的，故这里只显示不同的地方，可视化模型中不同的位置在于：

![image-20200604170656684](https://i.loli.net/2020/06/04/jVPmkLigaY1nCBq.png)

![image-20200604170552583](https://i.loli.net/2020/06/04/hm94Qg3xYbjd2f7.png)

可以看到，图2是`udnie_power_diff.pb`与图1不同的部分，即`PowerDifference`算子的不同，图1采用的是`SquaredDifference`，对于我们的`PowerDifference`算子，应该有三个输入，现在只有两个，根据分析，显然第三个参数Z的值为2（squared），所以加上第三个输入，并将其输入为2，即可。

```python
input_differ = sess.graph.get_tensor_by_name('moments_15/PowerDifference_z:0')
input_tensor = sess.graph.get_tensor_by_name('X_content:0')
output_tensor = sess.graph.get_tensor_by_name('add_37:0')

start_time = time.time()
# ret =sess.run(...)
ret =sess.run(output_tensor, feed_dict={input_differ:2,input_tensor:[X]})
end_time = time.time()
print("C++ inference(CPU) time is: ",end_time-start_time)

```

可以看到，这里获得了input_differ的占位符，并将其输入置为2，这样就可以得到最终的结果。

##### udnie_power_diff_numpy.pb

对于`udnie_power_diff_numpy.pb`而言，其使用的是之前实现的`power_diff_numpy.py`的内容来进行power_difference的计算，进行对比可以看到其具体实现的不同

![image-20200604170656684](https://i.loli.net/2020/06/04/jVPmkLigaY1nCBq.png)

![image-20200604171602543](D:\leliy\mine\大三下\智能计算系统\作业\image-20200604171602543.png)

可以看到，我们可以得到相应被用来计算powerdifference的输入，并根据此计算好power_difference并重新输入得到最终结果，故实际上要运行两次该推理网络，具体实现为：

```python
input_tensor = sess.graph.get_tensor_by_name('X_content:0')
input_differ = sess.graph.get_tensor_by_name('moments_15/PowerDifference:0')
output_tensor = sess.graph.get_tensor_by_name('add_37:0')
conv_tensor = sess.graph.get_tensor_by_name('Conv2D_13:0')
grad_tensor = sess.graph.get_tensor_by_name('moments_15/StopGradient:0')
temp_feed = {input_differ:[X],input_tensor:[X]}
conv = sess.run(conv_tensor,feed_dict=temp_feed)
grad = sess.run(grad_tensor,feed_dict=temp_feed)
#conv = conv.eval(session=sess)
#grad = grad.eval(session=sess)
differ = power_diff_numpy(conv,grad,2) 
start_time = time.time()
# ret =sess.run(...)
ret =sess.run(output_tensor, feed_dict={input_differ:differ,input_tensor:[X]})
#start_time = time.time()
#ret = sess.run(...)
end_time = time.time()
print("Numpy inference(CPU) time is: ",end_time-start_time)

```

可以看到，其首先通过一次运行得到conv 和 grad，然后将这两个参数传递到`power_diff_numpy`中进行计算，得到power Difference的计算结果之后，再将其作为输入传入到网络中，得到最终结果。 

最终，得到三张图片，没有明显差异：

![image-20200604172009418](https://i.loli.net/2020/06/04/DLi5anFIxBoUENR.png)

显然实现正确。

#### MLU推理

MLU推理中的代码在很大程度上是与CPU中的实现相同的，故不再说明三个不同模型的代码执行过程。在MLU执行中，一个更主要的内容在于保存离线模型用于后续的执行，其只需要加上一行代码即可：

```python
config.mlu_options.save_offline_model = True
```

除此之外，为了使得能够并行加速推理过程，我们还可以设置其它的部分：

![image-20200604172852904](https://i.loli.net/2020/06/04/VjbEWowryRIHXSk.png)

因此，通过这样的设置，我们能够加速推理过程，采用并行方式进行推理。

![image-20200604173005892](https://i.loli.net/2020/06/04/bf1rDkA6oVwHj2Q.png)

生成的最终三张图片如上所示，显然实现正确。

于此同时，也生成了相应的离线模型，如下所示：

![image-20200604173108140](https://i.loli.net/2020/06/04/WmYOvQj91AhdEyZ.png)

因此，可以执行离线推理。

### 离线推理程序

对于离线推理程序而言，其main函数如下所示：

```c++
int main(int argc, char** argv){
    // parse args
    std::string file_list = "../../images/" + std::string(argv[1]) + ".jpg";
    std::string offline_model = "../../models/offline_models/" + std::string(argv[2]) + ".cambricon";

    //creat data 
    DataTransfer* DataT =(DataTransfer*) new DataTransfer();
    DataT->image_name = argv[1];
    DataT->model_name = argv[2];
    //process image
    DataProvider *image = new DataProvider(file_list); 
    image->run(DataT);

    //running inference
    Inference *infer = new Inference(offline_model);
    infer->run(DataT);

    //postprocess image
    PostProcessor *post_process = new PostProcessor();
    post_process->run(DataT);
    
    delete DataT;
    DataT = NULL;
    delete image;
    image = NULL;
    delete infer;
    infer = NULL;
    delete post_process;
    post_process = NULL;
}
```

可以看到，其传输了两个参数，参数1是图片的地址(name)，参数2是离线模型的地址(name)。 可以看到，整个的推理过程主要分为三步：`DataProvider`， `Inference`和`PostProcessor`，其中第一部分主要用于读取图片并得到输入数据，而第二部分是主要的推理部分（也是我们实现的部分），第三部分是将得到的最终输出数据保存为图片。

根据提示，实现的inference.cpp中的内容如下所示，实现重点在注释中显示。 关于模型推理部分的代码，主要参考为文件中的`powerDiff.cpp`和[寒武纪运行时库用户手册](http://www.cambricon.com/docs/cnrt/user_guide_html/example/offline_mode.html)

```cpp
namespace StyleTransfer{

typedef unsigned short half;
//参考实验1中powerDiff.cpp中实现相关转换的代码
void cnrtConvertFloatToHalfArray(uint16_t* x, const float* y, int len) {
  for (int i = 0; i < len; i++){
    cnrtConvertFloatToHalf(x+i,y[i]);
  }
}

void cnrtConvertHalfToFloatArray(float* x, const uint16_t* y, int len) {
  for (int i = 0; i < len; i++){
    cnrtConvertHalfToFloat(x+i,y[i]);
  }
}


Inference :: Inference(std::string offline_model){
    offline_model_ = offline_model;
}

void Inference :: run(DataTransfer* DataT){
    cnrtInit(0);
    //初始化cnrt
    // load model
    cnrtModel_t model;
    cnrtLoadModel(&model, offline_model_.c_str());
	//根据名称和地址来加载当前的model
    // set current device
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, 0);
    cnrtSetCurrentDevice(dev);
    //设置当前的device，由于只有一个MLU 270,所以device 为0
    
    float* input_data = reinterpret_cast<float*>(malloc(256*256*3*sizeof(float)));
    float* output_data = reinterpret_cast<float*>(malloc(256*256*3*sizeof(float)));
    //创建输入和输出的数据
    int t = 256*256;
    for(int i=0;i<t;i++)
        for(int j=0;j<3;j++)
            input_data[i*3+j] = DataT->input_data[t*j+i];       //转换原输入数据到当前输入数据，由于tensorflow实际上使用的是
    //NHWC,故要将数据转换为NHWC格式
    int number = 0;
    cnrtGetFunctionNumber(model, &number);
    printf("%d function\n", number);
	//计算当前的model一共有多少的function number
    // load extract function
    cnrtFunction_t function;
    if (CNRT_RET_SUCCESS != cnrtCreateFunction(&function)) {
      printf("cnrtCreateFunction Failed!\n");
      exit(-1);
    }
    
    if (CNRT_RET_SUCCESS != cnrtExtractFunction(&function, model, "subnet0")) {
      printf("cnrtExtractFunction Failed!\n");
      exit(-1);
    }
    //加载function

    int inputNum, outputNum;
    int64_t *inputSizeS, *outputSizeS;
    cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
    cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);  // prepare data on cpu
    printf("the input num :%d , the output num : %d\n",inputNum,outputNum);
    printf("the input size: %lld, the output size : %lld\n",inputSizeS[0],outputSizeS[0]);
    //计算输入和输出的number 数目 （根据size 可判断实际上确实使用的是half数据）

    DataT->output_data = reinterpret_cast<float*>(malloc(256 * 256 * 3 * sizeof(float)));
    half* input_half = (half*)malloc(256 * 256 * 3 * sizeof(half));
    half* output_half = (half*)malloc(256 * 256 * 3 * sizeof(half));
  
    cnrtConvertFloatToHalfArray(input_half, input_data, 256 * 256 * 3);
    cnrtConvertFloatToHalfArray(output_half, DataT->output_data, 256 * 256 * 3);
    //转换数据类型从 float 到 half
  
    // allocate I/O data memory on MLU
    void *mlu_input, *mlu_output;

    // prepare input buffer
    if (CNRT_RET_SUCCESS != cnrtMalloc(&(mlu_input), inputSizeS[0])) {
      printf("cnrtMalloc Failed!\n");
      exit(-1);
    }
    if (CNRT_RET_SUCCESS != cnrtMalloc(&(mlu_output), outputSizeS[0])) {
      printf("cnrtMalloc output Failed!\n");
      exit(-1);
    }
    //为MLU中的数据分配I/O数据空间
    if (CNRT_RET_SUCCESS != cnrtMemcpy(mlu_input, input_half, 256 * 256 * 3 * sizeof(half), CNRT_MEM_TRANS_DIR_HOST2DEV)) {
      printf("cnrtMemcpy Failed!\n");
      exit(-1);
    }
    //将CPU中的数据拷贝到MLU中

    // setup runtime ctx
    cnrtRuntimeContext_t ctx;
    cnrtCreateRuntimeContext(&ctx, function, NULL);
	//创建一个runtime ctx 并绑定设备
    // bind device
    cnrtSetRuntimeContextDeviceId(ctx, 0);
    cnrtInitRuntimeContext(ctx, NULL);
    
    void *param[2];
    param[0] = mlu_input;
    param[1] = mlu_output;
    //设置参数，由于只有两个参数，输入和输出
    // compute offline
    cnrtQueue_t queue;
    cnrtRuntimeContextCreateQueue(ctx, &queue);
    cnrtInvokeRuntimeContext(ctx, (void**)param, queue, nullptr);
    cnrtSyncQueue(queue);
    //开始执行整个推理过程
    
    printf("run success\n");
    
    if (CNRT_RET_SUCCESS != cnrtMemcpy(output_half, mlu_output, 256 * 256 * 3 * sizeof(half), CNRT_MEM_TRANS_DIR_DEV2HOST)) {
      printf("cnrtMemcpy output Failed!\n");
      exit(-1);
    }
    //将数据拷贝回CPU中
    cnrtConvertHalfToFloatArray(output_data, output_half, 256 * 256 * 3);
    printf("memcpy output success\n");
    for(int i=0;i<t;i++)
        for(int j=0;j<3;j++)
            DataT->output_data[t*j+i] = output_data[i*3+j];
	//将half数据转换为float数据，并将其转换为所需要的output 格式
    
    // free memory spac
    if (CNRT_RET_SUCCESS != cnrtFree(mlu_input)) {
      printf("cnrtFree Failed!\n");
      exit(-1);
    }
    if (CNRT_RET_SUCCESS != cnrtFree(mlu_output)) {
      printf("cnrtFree output Failed!\n");
      exit(-1);
    }
    printf("free mlu success\n");
    if (CNRT_RET_SUCCESS != cnrtDestroyQueue(queue)) {
      printf("cnrtDestroyQueue Failed!\n");
      exit(-1);
    }
    printf("free queue success\n");
    cnrtDestroy();
    //free(param);
    free(input_half);
    free(output_half);
}

} // namespace StyleTransfer

```

##### 数据转换

在这里重点强调一下其中数据转换的部分，这是很容易忽略的一部分。

对于输入数据，可以看到，在DataProvider中，其会得到相应的输入，其中最重要的函数为：

```cpp
void DataProvider :: split_image(DataTransfer* DataT){
    DataT->input_data = reinterpret_cast<float*>(malloc(sizeof(float) * 256*256*3*batch_size));
    float *data_tmp = DataT->input_data;
    printf("the batch size is : %d\n",batch_size);
    for(int i = 0; i < batch_size; i++){
        DataT->split_images.push_back(std::vector<cv::Mat>());
        for(int j = 0; j < 3; j++){
            cv::Mat img(256, 256, CV_32FC1, data_tmp);
            DataT->split_images[i].push_back(img);
            data_tmp += 256*256;
        }
        cv::split(DataT->image_processed[i], DataT->split_images[i]);
    }
}
```

可以看到，这里得到的数据实际上是NCHW，也就是第二维是channel，根据channel进行了分片的，但是对于实际模型的输入，以在线为例，其是NHWC的(tensorflow本身输入的要求也为如此)

![image-20200604215616178](https://i.loli.net/2020/06/04/e6U1yYDFvjf5Idr.png)

可以看到，输入为(256,256,3)，所以需要进行调整

具体的调整在程序中为：

```cpp
float* input_data = reinterpret_cast<float*>(malloc(256*256*3*sizeof(float)));
float* output_data = reinterpret_cast<float*>(malloc(256*256*3*sizeof(float)));
//创建输入和输出的数据
int t = 256*256;
for(int i=0;i<t;i++)
    for(int j=0;j<3;j++)
        input_data[i*3+j] = DataT->input_data[t*j+i];  
```

可以看到，这样能够将NCHW的数据转换为NHWC的数据，这样能够得到正确的结果。

而对于输出而言，其具体存储代码为：

```cpp
void PostProcessor :: save_image(DataTransfer* DataT){

    std::vector<cv::Mat> mRGB(3);
    for(int i = 0; i < 3; i++){
        cv::Mat img(256, 256, CV_32FC1, DataT->output_data + 256 * 256 * i);
        mRGB[i] = img;
    }
    cv::Mat im(256, 256, CV_8UC3);
    cv::merge(mRGB,im);

    std::string file_name = DataT->image_name + std::string("_") + DataT->model_name + ".jpg";
    cv::imwrite(file_name, im);
    std::cout << "style transfer result file: " << file_name << std::endl;   
}
```

可以看到，也是通过NCHW来进行存储的，故对于输出，就需要将NHWC的数据转换为NCHW的数据进行输出，其具体内容这里不再继续展示，参考`inference.cpp`即可。

##### 实验完成

完成之后，重新cmake 和 make ，最后执行`./run.sh`程序，得到最终结果为：

![image-20200604214810613](https://i.loli.net/2020/06/04/qPX5aNhSRIwG9c2.png)

最终得到的生成图片为：

![image-20200604214908773](https://i.loli.net/2020/06/04/K19l3XgxckYmIU4.png)

至此，完成实验