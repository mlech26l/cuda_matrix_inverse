#ifndef GPU_PIVOTING_H_
#define GPU_PIVOTING_H_

typedef struct {int index; float value;} pivoting_max_entry;

void pivoting_preload_device_properties(int n);
void pivoting_unload_device_properties(void);

/* d_A must be on the device!!!! */
pivoting_max_entry pivoting_find_pivot_semi_gpu(float *d_A, int n, int row);

#endif
