# **NVDLA Compiler**

The NVDLA compiler is available [here](https://github.com/nvdla/sw/tree/master/prebuilt/x86-ubuntu) and it can be used as follows,

<a name="nvdla-compiler-help"></a>

    Usage: ./nvdla_compiler [options] --prototxt <prototxt_file> --caffemodel <caffemodel_file>
    where options include:
    -h                                              print this help message
    -o <outputpath>                                 outputs wisdom files in 'outputpath' directory
    --profile <basic|default|performance|fast-math> computation profile (default: fast-math)
    --cprecision <fp16|int8>                        compute precision (default: fp16)
    --configtarget <nv_full|nv_large|nv_small>      target platform (default: nv_full)
    --calibtable <int8 calibration table>           calibration table for INT8 networks (default: 0.00787)
    --quantizationMode <per-kernel|per-filter>      quantization mode for INT8 (default: per-kernel)
    --batch                                         batch size (default: 1)
    --informat <ncxhwx|nchw|nhwc>                   input data format (default: nhwc)

<a name="nvdla-compiler-example"></a>

We found two Caffe models online that successfully compiled. They are as follows,
- [LeNet](https://www.esp.cs.columbia.edu/docs/thirdparty_acc/thirdparty_acc-guide/)
- ResNet-50 ([Caffe model](https://cknowledge.org/repo/web.php?wcid=1dc07ee0f4742028:4b439b412770d1a6), [calibtable](https://github.com/nvdla/sw/tree/master/umd/utils/calibdata))

For the ResNet-50 .prototxt file we need to manually change the ``$#batch_size#$`` value to 1.

We used the following command when creating the NVDLA loadables,

    ./nvdla_compiler --prototxt <filename>.prototxt --caffemodel <filename>.caffemodel -o . --cprecision <selected_precision> --calibtable <filename>.json

The rest of the arguements were not explicitely defined, hence default values were used.

The caffe models used and the loadables generated are included in this directory.
