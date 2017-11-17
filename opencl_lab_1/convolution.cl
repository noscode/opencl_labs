__kernel void convolution(__global float *a, __global float *b, __global float *c, int n, int m) 
{
	int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= n || col >= n)
        return;

	int HM = (m - 1) / 2;
    float sum = 0;

    for (int k = -HM; k <= HM; ++k) {
        for (int l = -HM; l <= HM; ++l) {
            int a_x = row + k;
			int a_y = col + l;
			int b_x = k + HM;
			int b_y = l + HM;
            if (a_x >= 0 && a_y >= 0 && a_x < n && a_y < n) {
                sum += a[a_x * n + a_y] * b[b_x * m + b_y];
            }
        }
    }
    c[row * n + col] = sum;
}
