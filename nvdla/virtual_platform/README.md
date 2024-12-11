# **NVDLA Virtual Platform**

We followed the flow given [here](https://nvdla.org/vp.html) when building the virtual simulator platform. Since it needed older versions, we used a virtual machine with Ubuntu 14.04 booted in it to build the files needed for the virtual platform.

We need the following files to deploy the virtual simulator,
- ``aarch64_toplevel`` - This is the virtual simulator executable built during the process
- ``aarch64_nvdla.lua`` - Configuration file used by the executable (available in [nvdla/vp](https://github.com/nvdla/vp/tree/master/conf) repository)
- ``Image`` - Kernel image during bootup of the virtual simulator
- ``rootfs.ext2`` - Filesystem for the virtual simulator

The configuration file needs to be changed to give the location of the kernel image and the filesystem.

We built the remaining files following the steps given. But except for executable file, prebuilt files for all the other files are given [here](https://github.com/nvdla/sw/tree/master/prebuilt/arm64-linux).

The command to launch the virtual simulator is as follows,
```
export SC_SIGNAL_WRITE_CHECK=DISABLE
aarch64_toplevel -c aarch64_nvdla.lua
```

All the files that we built for the virtual simulator are given in this directory.