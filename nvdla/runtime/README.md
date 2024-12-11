# **NVDLA Runtime**

All the prebuilt files needed inside the simululator is available [here](https://github.com/nvdla/sw/tree/master/prebuilt/arm64-linux). The important files are,
- drm.ko - User-Mode Driver (UMD)
- opendla_1.ko - Kernel-Mode Driver (KMD) for ``nv_full`` config.
- opendla_2.ko - Kernel-Mode Driver (KMD) for ``nv_large``/``nv_small`` config.
- nvdla_runtime - Runtime application

The NVDLA loadables, images used and any SW regression loadables (only needed for AWS FPGA) also needs to copied to accessible folder.

Once the virtual platform is deployed, the following needs to be done before runtime,
- Mount the drive (any file needed by the simulator needs to be inside the directory the simulator is launched from)
```
mount -t 9p -o trans=virtio r /mnt
```
- Insert kernel driver modules
```
cd /mnt
insmod drm.ko
insmod opendla_1.ko
```

The runtime application available here can be run in 3 modes,
1. Run with only the NVDLA loadable (sanity test)
```
./nvdla_runtime --loadable <file_name>.nvdla
```
2. Run with NVDLA loadable and a sample image (network test)
```
./nvdla_runtime --loadable <file_name>.nvdla --image <sample_image_file>
```
3. Run in server mode (server test)
```
./nvdla_runtime -s
```

The 4th mode of running regressions is not given here, since the flow is very different. Details on how to do it in AWS FPGA is given [here](https://nvdla.org/vp_fpga.html).

The NVDLA loadables used and the terminal outputs are given in this directory.
