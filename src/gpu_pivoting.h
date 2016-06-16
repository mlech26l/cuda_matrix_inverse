#ifndef GPU_PIVOTING_H_
#define GPU_PIVOTING_H_

typedef struct {int index; float value;} max_entry;

void preload_device_properties(int n);
void unload_device_properties(void);

/* d_A must be on the device!!!! */
max_entry find_pivot_semi_gpu(float *d_A, int n, int row);

#endif
