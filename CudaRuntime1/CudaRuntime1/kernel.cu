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
#define o_plus(r0,r1,r2,r3,r4,r5,x0,x1,x2,x3,x4,x5,y0,y1,y2,y3,y4,y5) { \
	ull k=(ull)x0+y0; r0=k; k>>=32; \
	k+=(ull)x1+y1; r1=k; k>>=32; \
	k+=(ull)x2+y2; r2=k; k>>=32; \
	k+=(ull)x3+y3; r3=k; k>>=32; \
	k+=(ull)x4+y4; r4=k; k>>=32; \
	k+=(ull)x5+y5; r5=k; \
}
#define o_minus(r0,r1,r2,r3,r4,r5,x0,x1,x2,x3,x4,x5,y0,y1,y2,y3,y4,y5) { \
	signed long long k=(ull)x0-y0; r0=k; k>>=32; \
	k+=(ull)x1-y1; r1=k; k>>=32; \
	k+=(ull)x2-y2; r2=k; k>>=32; \
	k+=(ull)x3-y3; r3=k; k>>=32; \
	k+=(ull)x4-y4; r4=k; k>>=32; \
	k+=(ull)x5-y5; r5=k; \
}
#define o_times(r0,r1,r2,r3,r4,r5,x0,x1,x2,x3,x4,x5,y0,y1,y2,y3,y4,y5) { \
	ull t0=(ull)x0*y0; \
	ull t1=(t0>>32)+(ull)x0*y1; \
	ull t2=t1>>32; t1&=0xffffffff; t1+=(ull)x1*y0; \
	t2+=(t1>>32)+(ull)x0*y2; ull t3=t2>>32; t2&=0xffffffff; t2+=(ull)x1*y1; \
		t3+=t2>>32; t2&=0xffffffff; t2+=(ull)x2*y0; \
	t3+=(t2>>32)+(ull)x0*y3; ull t4=t3>>32; t3&=0xffffffff; t3+=(ull)x1*y2; \
		t4+=t3>>32; t3&=0xffffffff; t3+=(ull)x2*y1; \
		t4+=t3>>32; t3&=0xffffffff; t3+=(ull)x3*y0; \
	t4+=(t3>>32)+(ull)x0*y4; ull t5=t4>>32; t4&=0xffffffff; t4+=(ull)x1*y3; \
		t5+=t4>>32; t4&=0xffffffff; t4+=(ull)x2*y2; \
		t5+=t4>>32; t4&=0xffffffff; t4+=(ull)x3*y1; \
		t5+=t4>>32; t4&=0xffffffff; t4+=(ull)x4*y0; \
	t5+=(t4>>32)+(ull)x0*y5; ull t6=t5>>32; t5&=0xffffffff; t5+=(ull)x1*y4; \
		t6+=t5>>32; t5&=0xffffffff; t5+=(ull)x2*y3; \
		t6+=t5>>32; t5&=0xffffffff; t5+=(ull)x3*y2; \
		t6+=t5>>32; t5&=0xffffffff; t5+=(ull)x4*y1; \
		t6+=t5>>32; t5&=0xffffffff; t5+=(ull)x5*y0; \
	t6+=(t5>>32)+(ull)x1*y5; ull t7=t6>>32; t6&=0xffffffff; t6+=(ull)x2*y4; \
		t7+=t6>>32; t6&=0xffffffff; t6+=(ull)x3*y3; \
		t7+=t6>>32; t6&=0xffffffff; t6+=(ull)x4*y2; \
		t7+=t6>>32; t6&=0xffffffff; t6+=(ull)x5*y1; \
	t7+=(t6>>32)+(ull)x2*y5; ull t8=t7>>32; t7&=0xffffffff; t7+=(ull)x3*y4; \
		t8+=t7>>32; t7&=0xffffffff; t7+=(ull)x4*y3; \
		t8+=t7>>32; t7&=0xffffffff; t7+=(ull)x5*y2; \
	t8+=(t7>>32)+(ull)x3*y5; ui t9=t8>>32; t8&=0xffffffff; t8+=(ull)x4*y4; \
		t9+=t8>>32; t8&=0xffffffff; t8+=(ull)x5*y3; \
	t9+=(ull)x4*y5+(ull)x5*y4; \
	r0=t4; r1=t5; r2=t6; r3=t7; r4=t8; r5=t9; \
}
#define o_div(r0,r1,r2,r3,r4,r5,x0,x1,x2,x3,x4,x5,y0,y1,y2,y3,y4,y5) { \
	ui t2=x0,t3=x1,t4=x2,t5=x3,t6=x4,t7=x5,t8=0,t9=0,tA=0; \
	r0=0; r1=0; r2=0; r3=0; r4=0; r5=0; \
	for (int i = 31; i >= 0; i--) { \
		tA=t9>>31; \
		t9=t9<<1|t8>>31; \
		t8=t8<<1|t7>>31; \
		t7=t7<<1|t6>>31; \
		t6=t6<<1|t5>>31; \
		t5=t5<<1|t4>>31; \
		t4=t4<<1|t3>>31; \
		t3=t3<<1|t2>>31; \
		t2<<=1; \
		if(tA || \
			t9>y5 || t9==y5&&( \
			t8>y4 || t8==y4&&( \
			t7>y3 || t7==y3&&( \
			t6>y2 || t6==y2&&( \
			t5>y1 || t5==y1&&( \
			t4>=y0 \
		)))))){ \
			r5|=1u<<i; \
			o_minus(t4,t5,t6,t7,t8,t9,t4,t5,t6,t7,t8,t9,y0,y1,y2,y3,y4,y5) \
		} \
	} \
	for (int i = 31; i >= 0; i--) { \
		tA=t9>>31; \
		t9=t9<<1|t8>>31; \
		t8=t8<<1|t7>>31; \
		t7=t7<<1|t6>>31; \
		t6=t6<<1|t5>>31; \
		t5=t5<<1|t4>>31; \
		t4=t4<<1|t3>>31; \
		t3<<=1; \
		if(tA || \
			t9>y5 || t9==y5&&( \
			t8>y4 || t8==y4&&( \
			t7>y3 || t7==y3&&( \
			t6>y2 || t6==y2&&( \
			t5>y1 || t5==y1&&( \
			t4>=y0 \
		)))))){ \
			r4|=1u<<i;\
			o_minus(t4,t5,t6,t7,t8,t9,t4,t5,t6,t7,t8,t9,y0,y1,y2,y3,y4,y5) \
		} \
	} \
	for (int i = 31; i >= 0; i--) { \
		tA=t9>>31; \
		t9=t9<<1|t8>>31; \
		t8=t8<<1|t7>>31; \
		t7=t7<<1|t6>>31; \
		t6=t6<<1|t5>>31; \
		t5=t5<<1|t4>>31; \
		t4<<=1; \
		if(tA || \
			t9>y5 || t9==y5&&( \
			t8>y4 || t8==y4&&( \
			t7>y3 || t7==y3&&( \
			t6>y2 || t6==y2&&( \
			t5>y1 || t5==y1&&( \
			t4>=y0 \
		)))))){ \
			r3|=1u<<i;\
			o_minus(t4,t5,t6,t7,t8,t9,t4,t5,t6,t7,t8,t9,y0,y1,y2,y3,y4,y5) \
		} \
	} \
	for (int i = 31; i >= 0; i--) { \
		tA=t9>>31; \
		t9=t9<<1|t8>>31; \
		t8=t8<<1|t7>>31; \
		t7=t7<<1|t6>>31; \
		t6=t6<<1|t5>>31; \
		t5=t5<<1|t4>>31; \
		t4<<=1; \
		if(tA || \
			t9>y5 || t9==y5&&( \
			t8>y4 || t8==y4&&( \
			t7>y3 || t7==y3&&( \
			t6>y2 || t6==y2&&( \
			t5>y1 || t5==y1&&( \
			t4>=y0 \
		)))))){ \
			r2|=1u<<i;\
			o_minus(t4,t5,t6,t7,t8,t9,t4,t5,t6,t7,t8,t9,y0,y1,y2,y3,y4,y5) \
		} \
	} \
	for (int i = 31; i >= 0; i--) { \
		tA=t9>>31; \
		t9=t9<<1|t8>>31; \
		t8=t8<<1|t7>>31; \
		t7=t7<<1|t6>>31; \
		t6=t6<<1|t5>>31; \
		t5=t5<<1|t4>>31; \
		t4<<=1; \
		if(tA || \
			t9>y5 || t9==y5&&( \
			t8>y4 || t8==y4&&( \
			t7>y3 || t7==y3&&( \
			t6>y2 || t6==y2&&( \
			t5>y1 || t5==y1&&( \
			t4>=y0 \
		)))))){ \
			r1|=1u<<i;\
			o_minus(t4,t5,t6,t7,t8,t9,t4,t5,t6,t7,t8,t9,y0,y1,y2,y3,y4,y5) \
		} \
	} \
	for (int i = 31; i >= 0; i--) { \
		tA=t9>>31; \
		t9=t9<<1|t8>>31; \
		t8=t8<<1|t7>>31; \
		t7=t7<<1|t6>>31; \
		t6=t6<<1|t5>>31; \
		t5=t5<<1|t4>>31; \
		t4<<=1; \
		if(tA || \
			t9>y5 || t9==y5&&( \
			t8>y4 || t8==y4&&( \
			t7>y3 || t7==y3&&( \
			t6>y2 || t6==y2&&( \
			t5>y1 || t5==y1&&( \
			t4>=y0 \
		)))))){ \
			r0|=1u<<i; \
			o_minus(t4,t5,t6,t7,t8,t9,t4,t5,t6,t7,t8,t9,y0,y1,y2,y3,y4,y5) \
		} \
	} \
}
__device__
void m_plus(ui* r, ui* x, ui* y) {
	o_plus(r[0], r[1], r[2], r[3], r[4], r[5], x[0], x[1], x[2], x[3], x[4], x[5], y[0], y[1], y[2], y[3], y[4], y[5]);
}
__device__
void m_times(ui* r, ui* x, ui* y) {
	o_times(r[0], r[1], r[2], r[3], r[4], r[5], x[0], x[1], x[2], x[3], x[4], x[5], y[0], y[1], y[2], y[3], y[4], y[5]);
}
__device__
void m_div(ui* r, ui* x, ui* y) { // *r <-> *y
	o_div(r[0], r[1], r[2], r[3], r[4], r[5], x[0], x[1], x[2], x[3], x[4], x[5], y[0], y[1], y[2], y[3], y[4], y[5]);
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
