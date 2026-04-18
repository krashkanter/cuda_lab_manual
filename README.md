
**Essential Commands**

Compile without nvcc adding aggressive optimizations that mitigate warp divergence

	nvcc -g -G {.cu} -o {path_ouput}

Profiling the program

    sudo ncu --section SourceCounters {program_to_profile}

