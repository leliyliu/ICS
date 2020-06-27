# 实验1

---

## 必做（BANGC算子实现与TensorFlow的集成）

#### PowerDifference BANGC算子实现

##### plugin_power_difference_kernel.h

对于`plugin_power_difference_kernel.h`文件而言，只需要对于BCL接口进行定义即可，即填写函数`PowerDifferenceKernel`的相关参数，下面进行具体分析

根据整个项目文件的执行顺序，从`main.cpp`入手，在main函数中，其调用了函数`MLUPowerDifferenceOp(input_x,input_y,POW_COUNT,output_data,DATA_COUNT);`来进行PowerDifference的计算，具体函数实现在`powerDiff.cpp`文件中，对于其中代码具体分析：

```c++
  //prepare data
  half* input1_half = (half*)malloc(dims_a * sizeof(half));
  half* input2_half = (half*)malloc(dims_a * sizeof(half));
  half* output_half = (half*)malloc(dims_a * sizeof(half));

  cnrtConvertFloatToHalfArray(input1_half, input1, dims_a);
  cnrtConvertFloatToHalfArray(input2_half, input2, dims_a);
  cnrtConvertFloatToHalfArray(output_half, output, dims_a);

  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  cnrtKernelParamsBufferAddParam(params, &mlu_input1, sizeof(half*)); 
  cnrtKernelParamsBufferAddParam(params, &mlu_input2, sizeof(half*)); 
  cnrtKernelParamsBufferAddParam(params, &pow, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &mlu_output, sizeof(half*)); 
  cnrtKernelParamsBufferAddParam(params, &dims_a, sizeof(int)); 
  cnrtPlaceNotifier(event_start, pQueue);

  // TODO：完成cnrtInvokeKernel函数
  cnrtInvokeKernel_V2((void*)&PowerDifferenceKernel,dim,params,c,pQueue);
```

可以看到，首先通过`cnrtConvertFloatToHalfArray`函数将float* 转换为 half* ，然后调用`cnrtInvokeKernel_V2`来实现具体内容，故而可知，接口定义为：

```c++
void PowerDifferenceKernel(half* input1,half* input2,int32_t pow,half* output,int32_t dims_a);
```

##### plugin_power_difference_kernel.mlu

在`plugin_power_difference_kernel.mlu`中，对于power_difference进行了具体的实现，按照注释可以分为几个步骤（可以参考手册中给出的代码模仿进行计算），下面直接给出代码并在代码中进行相应的注释：

```c++
#define ONELINE 64
__mlu_entry__ void PowerDifferenceKernel(half* input1,half* input2,int32_t pow,half* output,int32_t dims_a)
{
  if (taskId > 0) return;
  // TODO：循环条件判断
  int32_t quotient = dims_a/ONELINE;
  int32_t rem = dims_a % ONELINE;
  if(rem != 0)
  {
    quotient+=1;
  }
 //判断需要进行多少次循环，每次循环设定为64个half的数据同时进行，因此，通过这样的方式可以计算得到
  
  // TODO：内存申请
  __nram__ half inputx_nram[ONELINE];
  __nram__ half inputy_nram[ONELINE];
  __nram__ half temp_nram[ONELINE];
  //进行内存申请，包括两个输入和一个中间操作

  // TODO：For循环计算
  for (int i = 0; i < quotient; i++)
  {
    // TODO：拷入操作
//在开始计算之前，需要将两个输入中相关的内容拷贝的NRAM中，然后进行计算
      __memcpy(inputx_nram,input1+i*ONELINE,ONELINE*sizeof(half),GDRAM2NRAM);
 __memcpy(inputy_nram,input2+i*ONELINE,ONELINE*sizeof(half),GDRAM2NRAM);
    // TODO：实际计算部分
    __bang_sub(temp_nram,inputx_nram,inputy_nram,ONELINE);
    __bang_active_abs(temp_nram,temp_nram,ONELINE);
    //首先将两个数据进行相减，然后求其绝对值
    for(int i=0;i<pow-1;i++)
    {
      __bang_mul(temp_nram,temp_nram,temp_nram,temp_nram,ONELINE);//循环相乘来求得其最终的pow次幂
    }
    // TODO：结果拷出操作
    __memcpy(output+i*ONELINE,temp_nram,ONELINE*sizeof(half),NRAM2GDRAM);
  }
}
```

#### PowerDifference BANGC算子测试

##### 补全powerDiff.cpp

对于`powerDiff.cpp`文件，主要有四个部分需要进行补全，下面分别进行分析

**补充PowerDifferenceKernel参数**

根据上一个部分的分析，非常容易就能够知道如何补充该参数，即为：

```c++
void PowerDifferenceKernel(half* input1,half* input2,int32_t pow,half* output,int32_t dims_a);
```

**完成拷入**

根据提示，要利用cnrtMemcpy函数进行拷入，也就是要将数据从host端拷入到device端，所以具体的代码实现非常简单：

```c++
  cnrtMemcpy(mlu_input1,input1_half,dims_a*sizeof(half),CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(mlu_input2,input2_half,dims_a*sizeof(half),CNRT_MEM_TRANS_DIR_HOST2DEV);

```

也就是将input1_half 和 input2_half拷贝到MLU设备当中

**完成cnrtInvokeKernel**

`cnrtInvokeKernel`函数会利用相应的参数来调用具体的内容，这里只需要注意一点，由于需要的第一个参数必须要为静态，那么不能直接写为：

```c++
cnrtInvokeKernel_V2(PowerDifferenceKernel,dim,params,c,pQueue);
//而需要写成
cnrtInvokeKernel_V2((void*)&PowerDifferenceKernel,dim,params,c,pQueue);
```

##### 执行测试

首先执行`bash make.sh`操作，操作结果如下所示：

![image-20200519161741055](https://i.loli.net/2020/05/19/pxICTNsib5ghV2U.png)

可以看到，完成之后生成了power_diff_test文件，执行该文件：

![image-20200519161851756](https://i.loli.net/2020/05/19/zv4km7ACqG82U6D.png)

执行成功，说明BANGC算子实现正确

#### cnplugin集成

##### 补全plugin_power_difference_op.cc

根据所给出的提示以及PPT中的说明，可以知道：

![image-20200519162334290](https://i.loli.net/2020/05/19/4IjZxDmYJMNnHhF.png)

由于`cnmlDestroyPluginPowerDifferenceOpParam`已经给出了具体实现，故而需要实现的函数共有三个，下面具体分析：

**cnmlCreatePluginPowerDifferenceOpParam**

关于此处的实现，可以参考其他算子的相应实现，以`/opt/AICSE-demo-student/env/Cambricon-CNPlugin-MLU270/pluginops/PluginNonMaxSuppressionOP`中的相应代码为例：

```c++
cnmlStatus_t cnmlCreatePluginNonMaxSuppressionOpParam(
    cnmlPluginNonMaxSuppressionOpParam_t *param,
    int len,
    int max_num,
    float iou_threshold,
    float score_threshold,
    cnmlCoreVersion_t core_version)
```

而其相应的输入为

```c++
__mlu_entry__ void NonMaxSuppressionKernel(int32_t* out_index, half* box_gdram, half* scores_gdram,
 half* const_gdram, half* max_score_gdram,int len, int max_num, half iou_threshold, half score_threshold)
```

所以实际上，除了param和core_version之外，其添加了四个其他参数，参考此实现除了基础的输入输出之外的参数设置，`cnmlCreatePluginPowerDifferenceOpParam`具体代码为：

```c++
cnmlStatus_t cnmlCreatePluginPowerDifferenceOpParam(
  cnmlPluginPowerDifferenceOpParam_t *param,
  int pow,
  int dims_a,
  cnmlCoreVersion_t core_version
  // TODO：添加变量
) {
  *param = new cnmlPluginPowerDifferenceOpParam();

  int static_num = 0;//无常量数据

  (*param)->pow = pow;
  (*param)->dims_a = dims_a;
  // TODO：配置变量

  return CNML_STATUS_SUCCESS;
}
```

**cnmlCreatePluginPowerDifferenceOp**

关于该接口所需要的参数，可以通过`/opt/AICSE-demo-student/demo/style_transfer_bcl/src/tf-implementation/tf-a
dd-power-diff`中的`mlu_lib_ops.cc`来找到，包括下一个接口也可以知道，其具体代码为：

```c++
tensorflow::Status CreatePowerDifferenceOp(MLUBaseOp** op, MLUTensor* input1,
                                             MLUTensor* input2,
                                             int input3,
                                             MLUTensor* output, int len) {
  MLUTensor* inputs_ptr[2] = {input1, input2};
  MLUTensor* outputs_ptr[1] = {output};

  CNML_RETURN_STATUS(cnmlCreatePluginPowerDifferenceOp(op, inputs_ptr, input3, outputs_ptr, len));
}
```

故而，可以看到，其一共传递了五个参数，因此根据此，可以得到算子的构建接口，需要设定好输入，输出以及静态变量，故而需要的参数除了op之外，还有input,pow,output 和len，具体代码实现如下：

```c++
cnmlStatus_t cnmlCreatePluginPowerDifferenceOp(
  cnmlBaseOp_t *op,
  cnmlTensor_t *pd_input_tensors,
  int pow,
  cnmlTensor_t *pd_output_tensors,
  int len
  // TODO：添加变量
) {
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  // TODO：配置变量

  cnrtKernelParamsBufferMarkInput(params);
  cnrtKernelParamsBufferMarkInput(params);
  cnrtKernelParamsBufferAddParam(params,&pow,sizeof(int));
  cnrtKernelParamsBufferMarkOutput(params);
  cnrtKernelParamsBufferAddParam(params,&len,sizeof(int));
  void **InterfacePtr = reinterpret_cast<void**>(&PowerDifferenceKernel);
  cnmlCreatePluginOp(op,
                     "PowerDifference",
                     InterfacePtr,
                     params,
                     pd_input_tensors,
                     2,
                     pd_output_tensors,
                     1,
                     nullptr,
                     0);
  cnrtDestroyKernelParamsBuffer(params);
  return CNML_STATUS_SUCCESS;
}
```

其中关于`cnmlCreatePluginOp`函数的参数，可以在`cnml.h`中查询到，其具体为：

```c++
CNML_DLL_API cnmlStatus_t cnmlCreatePluginOp(cnmlBaseOp_t *op,const char *name, void *kernel,cnrtKernelParamsBuffer_t params,cnmlTensor_t *input_tensors,int input_num,cnmlTensor_t *output_tensors,int output_num,
  cnmlTensor_t *statics_tensor,int static_num);
```

**cnmlComputePluginPowerDifferenceOpForward**

根据上一个接口实现的过程，同样也可以在`mlu_lib_ops.cc`中找到相应需要的参数，如下所示，只需要四个参数：

```c++
tensorflow::Status ComputePowerDifferenceOp(MLUBaseOp* op,
 MLUCnrtQueue* queue, void* input1,void* input2, void* output) {
  void* inputs_ptr[2] = {input1, input2};
  void* outputs_ptr[1] = {output};
  CNML_RETURN_STATUS(cnmlComputePluginPowerDifferenceOpForward(op, inputs_ptr, outputs_ptr, queue));
}
```



实际上，最后这个函数时用来声明单算子运行接口的，而整个计算只需要调用`cnmlComputePluginOpForward_V4`函数即可，在`cnml.h`中，有关于`cnmlComputePluginOpForward_V4`函数的具体声明，如下所示：

```c++
CNML_DLL_API cnmlStatus_t cnmlComputePluginOpForward_V4(cnmlBaseOp_t op,cnmlTensor_t input_tensors[], void *inputs[],int input_num, cnmlTensor_t output_tensors[], void *outputs[], int output_num, cnrtQueue_t queue, void *extra);
```

故而，可以实现为下：

```c++
cnmlStatus_t cnmlComputePluginPowerDifferenceOpForward(
  cnmlBaseOp_t op,
  void *inputs[],
  void *outputs[],
  // TODO：添加变量
  cnrtQueue_t queue,
) {
  // TODO：完成Compute函数
  cnmlComputePluginOpForward_V4(op,
                                nullptr,
                                inputs,
                                2,
                                nullptr,
                                outputs,
                                1,
                                queue,
                                nullptr);
  return CNML_STATUS_SUCCESS;
}
```

##### 补全cnplugin.h

根据前面对于`plugin_power_difference_op.cc`的补全，故而只需要在`cnplugin.h`中添加对于这几个函数的声明，以及对于结构体`cnmlPluginPowerDifferenceOpParam`的声明即可，具体实现代码如下：

```c++
struct cnmlPluginPowerDifferenceOpParam{
  int pow;
  int dims_a;
  cnmlCoreVersion_t core_version;
};

typedef cnmlPluginPowerDifferenceOpParam *cnmlPluginPowerDifferenceOpParam_t;

cnmlStatus_t cnmlCreatePluginPowerDifferenceOpParam(
  cnmlPluginPowerDifferenceOpParam_t *params,
  int pow,
  int dims_a,
  cnmlCoreVersion_t core_version
);

cnmlStatus_t cnmlDestroyPluginPowerDifferenceOpParam(
  cnmlPluginPowerDifferenceOpParam_t *params
);

cnmlStatus_t cnmlCreatePluginPowerDifferenceOp(
  cnmlBaseOp_t *op，
  cnmlTensor_t *pd_input_tensors,
  int pow,
  cnmlTensor_t *pd_output_tensors,
  int len
);

cnmlStatus_t cnmlComputePluginPowerDifferenceOpForward(
  cnmlBaseOp_t op,
  void *inputs[],
  void *outputs[],
  cnrtQueue_t queue
);
```

#### Tensorflow算子集成

根据tensorflow集成新算子的方式，分析可知，需要将`/opt/AICSE-demo-student/demo/style_transfer_bcl/src/tf-implementation/tf-a
dd-power-diff`中的文件分别拷贝的`/opt/AICSE-demo-student/env/tensorflow-v1.10/tensorflow`相应目录下，具体地址如下所示

>(以/opt/AICSE-demo-student/env/tensorflow-v1.10/tensorflow为根目录)
>
>cwise_op_power_difference_mlu.h & cwise_op_power_difference.cc -> ./core/kernels
>
>math_ops.cc -> ./core/ops/
>
>mlu_lib_ops.cc & mlu_lib_ops.h ->  ./stream_executor/mlu/mlu_api/lib_ops/
>
>mlu_ops.h -> ./stream_executor/mlu/mlu_api/ops/
>
>mlu_stream.h -> ./stream_executor/
>
>power_difference.cc -> ./stream_executor/mlu/mlu_api/ops/

其中，拷贝到`./core/kernels`中的两个文件实现了具体的部分，包括CPU计算和MLU计算的具体内容。而文件`math_ops.cc`,`mlu_lib_ops.cc`,`mlu_lib_ops.h`,`mlu_ops.h`以及`mlu_stream.h`文件都是替换原来的文件，其中新增加的部分都是关于power_difference算子注册的部分，而`power_difference.cc`文件则是注册到了MLU中，是关于具体实现的一个部分。

关于替换的部分，简单以`mlu_stream.h`为例，可以进行对比：

![image-20200519170323429](https://i.loli.net/2020/05/19/3L7HrNujDMmaO8p.png)

可以看到，其新增的部分就是用于PowerDifference的相关注册。

在将这些内容拷贝之后，并按照实验教程将`PluginPowerDifferenceOp`的相关内容拷贝到相应文件夹之后，对于tensorflow进行编译，进入到`/opt/AICSE-demo-student/env/tensorflow-v1.10`中执行相应的内容，注意需要在执行前，将`build_tensorflow-v1.10_mlu.sh`中89行`jobs_num`改为16或8，否则在编译过程中无法完成，这是由于内存较小导致的。

编译完成之后，会显示如下结果：

![image-20200519163805943](https://i.loli.net/2020/05/19/x9PBJ3nmdKT7khj.png)

#### 框架算子测试

最后只需要修改`power_difference_test_bcl.py`和`power_difference_test_cpu.py`的代码即可，实现方式相同，代码均为：

```python
  def power_difference_op(input_x,input_y,input_pow):
      with tf.Session() as sess:
        x = tf.placeholder(tf.float32,shape = input_x.shape)
        y = tf.placeholder(tf.float32,shape = input_y.shape)
        pow = tf.placeholder(tf.float32)
        out = tf.power_difference(x,y,pow);
        return sess.run(out,feed_dict = \{x:input_x,y:input_y,pow:input_pow})

```

在补全代码之后进行执行，其结果分别为：

**CPU代码测试**

![image-20200519164104720](D:\leliy\mine\大三下\智能计算系统\作业\image-20200519164104720.png)

**MLU代码测试**

![image-20200519164139449](D:\leliy\mine\大三下\智能计算系统\作业\image-20200519164139449.png)

其中，MLU所消耗时间长是因为其需要进行初始化，下面进行多次试验，修改代码：

![image-20200519164437244](https://i.loli.net/2020/05/19/6ik8lfseORPT5da.png)

 ![image-20200519164424177](https://i.loli.net/2020/05/19/t6CEpwjisGeFmnv.png)

故而，证明实现没有问题，到此，实验完成。

## 四选一选做(Cosine 相似度)

根据所给出的公式，可以很清楚的知道计算的方法：
$$
c(X,Y) = \frac{X\cdot Y}{|X||Y|} = \frac{\sum_{i=1}^nX_iY_i}{\sqrt{\sum_{i=1}^n X_i^2} \sqrt{\sum_{i=1}^n Y_i^2}}
$$
参考CPU实现的具体代码，可以知道，在实际过程中，有一个边界值的处理过程：

```c++
int CPUCOMCosineOp(float* output, float* inputX, float* inputY){
	float sqX[N];
	float sqY[N];
	for(int i=0; i<N; i++){
		sqX[i]=0.0;
		sqY[i]=0.0;
		for(int j=0; j<M; j++){
			sqX[i]+=*(inputX+j*N+i)*(*(inputX+j*N+i));
			sqY[i]+=*(inputY+j*N+i)*(*(inputY+j*N+i));
		}
		sqX[i]=sqrt(sqX[i]+eps);
		sqY[i]=sqrt(sqY[i]+eps);
	}
	for(int i=0; i<N; i++){
		output[i]=0.0;
		for(int j=0; j<M; j++){
			output[i]+=*(inputX+j*N+i)*(*(inputY+j*N+i));
		}
		output[i]/=(sqX[i]*sqY[i]);
		//printf("output[%d]: %f\n",i, output[i]);
	}
	return 0;
}
```

可以看到，实际上在最终结果之前加上了一个eps来进行计算，故而参考公式和CPU实现代码，最终实现的代码如下：

```c++
__mlu_entry__ void CosineKernel(half* inputX, half* inputY, half* output)
{
    __nram__ half inputx_nram[N];
    __nram__ half inputy_nram[N];
    __nram__ half output_nram[N];
    __nram__ half temp1_nram[N];
    __nram__ half temp2_nram[N];
    __nram__ half sumx_nram[N];
    __nram__ half sumy_nram[N];
    __nram__ half addelement_nram[N];

    __nramset_half(output_nram,N,0.0);
    __nramset_half(temp1_nram,N,0.0);
    __nramset_half(sumx_nram,N,0.0);
    __nramset_half(sumy_nram,N,0.0);
    __nramset_half(addelement_nram,N,EPS);

    for(int i=0;i<N;i++){
        __memcpy(inputx_nram,inputX + i*M,M*sizeof(half),GDRAM2NRAM);
        __memcpy(inputy_nram,inputY + i*M,M*sizeof(half),GDRAM2NRAM);

        __bang_mul(temp2_nram,inputx_nram,inputy_nram,N);
        __bang_add(temp1_nram,temp1_nram,temp2_nram,N);
        __bang_mul(inputx_nram,inputx_nram,inputx_nram,N);
        __bang_mul(inputy_nram,inputy_nram,inputy_nram,N);
        __bang_add(sumx_nram,sumx_nram,inputx_nram,N);
        __bang_add(sumy_nram,sumy_nram,inputy_nram,N);
    }
    __bang_add(sumx_nram,sumx_nram,addelement_nram,N);
    __bang_add(sumy_nram,sumy_nram,addelement_nram,N);
    __bang_active_sqrt(sumx_nram,sumx_nram,N);
    __bang_active_sqrt(sumy_nram,sumy_nram,N);
    __bang_mul(output_nram,sumx_nram,sumy_nram,N);
    __bang_active_recip(output_nram,output_nram,M);
    __bang_mul(output_nram,temp1_nram,output_nram,M);
    __memcpy(output,output_nram,M*sizeof(half),NRAM2GDRAM);
}
```

由于最后每一列计算结果为一个，所以实际上是一行一行进行处理，具体处理过程是将每一行拷贝之后，进行相乘，然后再对拷贝的元素平方加和，最后开方。

补全代码之后进行make，由于环境原因，需要修改`Makefile`中的具体内容，即gcc 和 lcnrt lcnml 等的相关地址。

![image-20200520212735864](https://i.loli.net/2020/05/20/H9oB8daD6rNp7mx.png)

修改结束后进行make，生成test文件，然后执行文件，最终结果如下所示。

![image-20200520211948445](https://i.loli.net/2020/05/20/Piamx2MpfdleL7R.png)

至此，选修作业完成。