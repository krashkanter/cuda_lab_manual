
**Essential Commands**

Compile without nvcc adding aggressive optimizations to mitigate warp divergence

	nvcc -g -G {.cu} -o {path_ouput}

Warp Divergence

    sudo ncu --section SourceCounters {program_to_profile}

