#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <queue>
#include <windows.h>
#include <mutex>
#include <thread>
#define TTMATH_NOASM
#include "ttmath.h"
using namespace std;
typedef ttmath::Big<TTMATH_BITS(64), TTMATH_BITS(256)> lf;
typedef double ld;
#define logscale 6.514417228548728556637719668407e-15
#define conpi 6.283185307179586476925286766559
#define cone 2.7182818284590452353602874713527
cudaError_t addWithCuda(long long *crr,long long num);
__global__ void addKernel(long long *crr,long long num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int kid = tid;
	for (int i = 0; i < 16384; i++) {
		long long X = i | tid << 14 | num << 28;
		ld k = log10(X * conpi) * .5; k = k - floor(k);
		ld tmp = X / cone; 
		ld k3 = 0;
		for (long long Y = X; Y; Y >>= 1) {
			tmp /= pow(10., round(log10(tmp)));
			ld k2 = log10(tmp);
			if (Y & 1) {
				k3 += k2;
				k3 -= round(k3);
			}
			tmp = tmp * tmp;
		}
		
		k += k3+ log10(1 + 1. / 12 / X + 1. / 288 / X / X - 139. / 51840 / X / X / X - 571. / 2488320 / X / X / X / X);
		k -= floor(k);
		double kk = round(pow(10., k));
		k -= log10(kk);
		if (abs(k)*kk< 1e-6) {
			while (crr[kid])
				kid = (kid + i) % 163840;
			crr[kid] = X;
		}
	}
}
ld mink = 1; long long minx = -1;
lf expi = lf(2) * ACos(lf(-1));
lf exe = Exp(lf(1));
queue<long long> RQ;
mutex mutRQ;
HANDLE ghSemaphore;
HANDLE fhSemaphore;
void threadfunc() {
	for (;;) {
		WaitForSingleObject(ghSemaphore,1000000000000L);
		mutRQ.lock();
		//cout << (int)sig <<' '<< RQ.size() << endl;
		long long CC= RQ.front(); RQ.pop();
		mutRQ.unlock();
		ReleaseSemaphore(
			fhSemaphore,  // handle to semaphore
			1, NULL);
		lf X = CC;
		lf k = Ln(X * expi) * .5;;
		lf k2 = Ln(X / exe); k2 = k2 * X;
		k += k2 + Ln(lf(1.) + lf(1.) / 12 / X + lf(1.) / 288 / X / X - lf(139.) / 51840 / X / X / X - lf(571.) / 2488320 / X / X / X / X);
		k /= Ln(lf(10.));
		k -= Floor(k);
		int kk = round(pow(10., k.ToDouble()));
		k -= log10(kk);
		k = Abs(k) * kk;
		//cout << "\t=\t" << k.ToDouble();
		if (k.ToDouble() < mink) {
			mink = k.ToDouble();
			minx = CC;
		}
		if (k.ToDouble() > 0.01)
			cout << k.ToDouble() << endl;
	}
}
int main() 
{
	ghSemaphore = CreateSemaphore(
		NULL,           // default security attributes
		0,  // initial count
		1024,  // maximum count
		NULL);          // unnamed semaphore
	fhSemaphore = CreateSemaphore(
		NULL,           // default security attributes
		1024,  // initial count
		1024,  // maximum count
		NULL);          // unnamed semaphore
	thread** thrr = new thread * [16];
	for (int i = 0; i < 16; i++)
		thrr[i]=new thread(threadfunc);
	static long long crr[163840];
	
	for(long long iiii= 1;;iiii++){
		if (iiii % 30 == 0) {
			system("cls");
			if ((iiii << 28) > 100000000000000)
				Sleep(10000);
			cout << (iiii << 28) << "/100000000000000---------------------" << endl;
			cout << minx << ':' << mink << endl;
			if ((iiii << 28) > 100000000000000)
				Sleep(100000000);
		}
		cudaError_t cudaStatus = addWithCuda(crr,iiii);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
		cout<<"---";int cnt = 0;
		for (int iii = 0; iii < 163840; iii++) {
			if (crr[iii]) {
				WaitForSingleObject(fhSemaphore, 1000000000L);
				mutRQ.lock();
				RQ.push(crr[iii]);
				mutRQ.unlock();
				ReleaseSemaphore(
					ghSemaphore,  // handle to semaphore
					1,NULL);
			}
		}
		cout << endl;
	}
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(long long *c, long long num)
{
    long long *dev_c = 0;
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_c, 163840*sizeof(long long));
	cudaMemset(dev_c, 0, 163840 * sizeof(long long));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    addKernel<<<128,128>>>(dev_c,num);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    cudaStatus = cudaMemcpy(c, dev_c, 163840 * sizeof(long long), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(dev_c);
    return cudaStatus;
}
