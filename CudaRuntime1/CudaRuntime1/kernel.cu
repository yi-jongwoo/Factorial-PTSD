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
typedef ttmath::Big<TTMATH_BITS(64), TTMATH_BITS(1024)> lf;
typedef double ld;
#define logscale 6.514417228548728556637719668407e-15
#define conpi 6.283185307179586476925286766559
#define cone 2.7182818284590452353602874713527
cudaError_t addWithCuda(long long *crr,long long num);
#define N 4 // under point precision
#define M 2 // over point precision
#define NM (N+M)
typedef unsigned int ui;
typedef unsigned long long ull;
__device__
void m_plus(ui* r, ui* x, ui* y) {
	ull k = 0;
	for (int i = 0; i < NM; i++) {
		k += (ull)x[i] + y[i];
		r[i] = k;
		k >>= 32;
	}
}
__device__
ui m_minus(ui* r, ui* x, ui* y) {
	signed long long k = 0;
	for (int i = 0; i < NM; i++) {
		k += (ull)x[i] - y[i];
		r[i] = k;
		k >>= 32;
	}
	return k;
}
__device__
void m_times(ui* r, ui* x, ui* y) {
	ull tmp[2 * NM] = { 0 };
	ull atmp[2 * NM] = { 0 };
	for (int i = 0; i < NM; i++)
		for (int j = 0; j < NM; j++) {
			tmp[i + j] += (ull)x[i] * y[j];
			atmp[i+j] += tmp[i+j]>>32;
			tmp[i+j] &= 0xffffffff;
		}
	for (int i = 0; i < (2 * N + M); i++) {
		tmp[i + 1] += tmp[i] >> 32;
		atmp[i + 1] += atmp[i] >> 32;
		tmp[i + 1] += atmp[i] & 0xffffffff;
	}
	for (int i = 0; i < NM; i++)
		r[i] = tmp[i + N];
}
__device__
void m_div(ui* r, ui* x, ui* y) { // *r <-> *y
	ui t[NM * 2] = { 0 };
	for (int i = 0; i < NM; i++)
		t[i + N - M] = x[i]; // assert that doesnt exceed 1e64
	for (int i = 0; i < NM; i++)
		r[i] = 0;
	for (int i = NM * 32 - 1; i >= 0; i--) {
		for (int j = NM * 2 - 1; j >= 0; j--) {
			t[j] <<= 1;
			if (j)
				t[j] |= t[j - 1] >> 31;
		}
		for (int j = NM * 2 - 1; j >= 0; j--) {
			ui yy = j >= N && j < NM + N ? y[j - N] : 0;
			if (t[j] > yy) {
				r[i / 32] |= 1u << (i % 32);
				if (m_minus(t + N, t + N, y))
					for (int k = NM + N; !t[k]--; k++);
				break;
			}
			else if (t[j] < yy)
				break;
		}
	}
}
__device__
void m_log2(ui* r, ui* x) {
	ui t[6];
	int kk = 0;
	for (int i = 0; i < 6; i++)
		t[i] = x[i];
	while (t[5] || t[4] > 1) {
		//std::cout<<r[5]<<std::endl;
		kk++;
		for (int j = 0; j < NM; j++) {
			t[j] >>= 1;
			if (j != NM - 1)
				t[j] |= t[j + 1] << 31;
		}
	}
	while (!t[4]) {
		kk--;
		for (int j = NM - 1; j >= 0; j--) {
			t[j] <<= 1;
			if (j)
				t[j] |= t[j - 1] >> 31;
		}
	}
	if (kk >= 0)
		r[5] = 0;
	else
		r[5] = -1;
	r[4] = kk;
	r[3] = r[2] = r[1] = r[0] = 0;
	for (int i = N * 32 - 1; i >= 0; i--) {
		m_times(t, t, t);
		if (t[4] > 1) {
			//std::cout<<i<<std::endl;
			r[i / 32] |= 1u << (i % 32);
			for (int j = 0; j < NM; j++) {
				t[j] >>= 1;
				if (j < NM - 1)
					t[j] |= t[j + 1] << 31;
			}
		}
	}
}
__device__
double ns6(long long CC) {
	ui expi[6] = { 88371514,1838622596,3337599704,1399077284,1,0 };
	ui exe[6] = { 767918830,2196767327,1204821640,2393606573,-2u,-1u };
	ui one[6] = { 0,0,0,0,1,0 };
	ui o1[6] = { 1431655765,1431655765,1431655765,357913941,0,0 };
	ui o2[6] = { 954437176,2386092942,3817748707,14913080,0,0 };
	ui o3[6] = { 2311858939,4061660430,1993713213,11516212,0,0 };
	ui o4[6] = { 2541630852,1056950799,1640217963,892368,0 };
	ui lten[6] = { 918514995,615504893,879635449,1382670639,3,0 };
	ui X[6] = { 0,0,0,0,CC & 0xffffffff,CC >> 32 };
	ui K12[6]; m_log2(K12, X);
	ui D[6]; m_div(D, one, X);
	m_times(o1, o1, D);
	m_plus(one, one, o1);
	ui E[6]; m_times(E, D, D);
	m_times(o2, o2, E);
	m_plus(one, one, o2);
	m_times(E, E, D);
	m_times(o3, o3, E);
	m_plus(one, one, o3);
	m_times(E, E, D);
	m_times(o4, o4, E);
	m_plus(one, one, o4);
	m_log2(one, one);
	m_plus(expi, expi, one); // 0.5log2(2pi) + log2(1+1/12x+...)
	m_times(exe, exe, X); // -Xlog2(e)
	X[3] = 1u << 31;
	m_times(K12, K12, X); // (X+0.5)log2(X)
	m_plus(K12, K12, expi);
	m_plus(K12, K12, exe);
	//cout<<':';for(int i=0;i<6;i++)cout<<K12[i]<<' ';
	m_div(K12, K12, lten);
	//cout<<':';for(int i=0;i<6;i++)cout<<K12[i]<<' ';cout<<endl;
	double res = K12[2];
	res /= (1ll << 32);
	res += K12[3];
	res /= (1ll << 32);
	return res;
}
__global__ void addKernel(long long *crr,long long num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int kid = tid;
	for (int i = 1; i < 16384; i++) {
		long long X = i | tid << 14 | num << 28;
		ld k = ns6(X);
		double kk = round(pow(10., k));
		k -= log10(kk);
		if (abs(k) * kk < 10000*logscale ) {
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
		WaitForSingleObject(ghSemaphore,1000000000000000L);
		mutRQ.lock();
		//cout << (int)sig <<' '<< RQ.size() << endl;
		long long CC = RQ.front(); RQ.pop();
		mutRQ.unlock();
		ReleaseSemaphore(
			fhSemaphore,  // handle to semaphore
			1, NULL);   //cout << CC << endl; continue;
		lf X = CC; //cout << X << endl;
		lf k = Ln(X * expi) * .5;;
		lf k2 = Ln(X / exe); k2 = k2 * X;
		k += k2 + Ln(lf(1.) + lf(1.) / 12 / X + lf(1.) / 288 / X / X - lf(139.) / 51840 / X / X / X - lf(571.) / 2488320 / X / X / X / X);
		k /= Ln(lf(10.));
		k -= Floor(k);
		int kk = round(pow(10., k.ToDouble()));
		k -= log10(kk);
		k = Abs(k) * kk;
		cout << CC<<"\t=\t" << k.ToDouble()<<endl;
		if (k.ToDouble() < mink) {
			mink = k.ToDouble();
			minx = CC;
		}
		cout << minx << "\t=\t" << mink << endl;
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
	static long long crr[16384];
	
	for(long long iiii= 1;;iiii++){
		cout << iiii<<' '<<':'<<(iiii << 28) << "/100000000000000---------------------" << endl;
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
		for (int iii = 0; iii < 16384; iii++) {
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
    cudaStatus = cudaMalloc((void**)&dev_c, 16384*sizeof(long long));
	cudaMemset(dev_c, 0, 16384 * sizeof(long long));
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
    cudaStatus = cudaMemcpy(c, dev_c, 16384 * sizeof(long long), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(dev_c);
    return cudaStatus;
}
