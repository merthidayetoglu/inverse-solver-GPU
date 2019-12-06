#include "vars.h"

#include <cuda.h>
#include <cuComplex.h>

__global__ void P2Mkernel(cuDoubleComplex *C, cuDoubleComplex *A, cuDoubleComplex *B, int M, int N, int L){
  extern __shared__ cuDoubleComplex temp[];
  int tile = blockDim.x;
  int row = blockIdx.y*tile+threadIdx.y;
  int col = blockIdx.x*tile+threadIdx.x;
  int numtile = N/tile;
  if(N%tile)numtile++;
  cuDoubleComplex reduce = make_cuDoubleComplex(0,0);
  int index = threadIdx.y*tile+threadIdx.x;
  int block = tile*tile;
  int start = 0;
  for(int k = 0; k < numtile; k++){
    if(row < M && start+threadIdx.x < N)
        temp[index] = A[row*N+start+threadIdx.x];
    if(col < L && start+threadIdx.y < N)
      temp[block+index] = B[(start+threadIdx.y)*L+col];
    __syncthreads();
    if(col < L && row < M)
      for(int i = 0; i < tile && start + i < N; i++)
        reduce = cuCadd(reduce,cuCmul(temp[threadIdx.y*tile+i],temp[block+i*tile+threadIdx.x]));
    __syncthreads();
    start = start+tile;
  }
  if(col < L && row < M)C[row*L+col] = reduce;
}

__global__ void L2Pkernel(cuDoubleComplex *C, cuDoubleComplex *A, cuDoubleComplex *B, int M, int N, int L){
  extern __shared__ cuDoubleComplex temp[];
  int tile = blockDim.x;
  int row = blockIdx.y*tile+threadIdx.y;
  int col = blockIdx.x*tile+threadIdx.x;
  int numtile = N/tile;
  if(N%tile)numtile++;
  cuDoubleComplex reduce = make_cuDoubleComplex(0,0);
  int index = threadIdx.y*tile+threadIdx.x;
  int block = tile*tile;
  int start = 0;
  for(int k = 0; k < numtile; k++){
    if(row < M && start+threadIdx.x < N)
      temp[index] = A[row*N+start+threadIdx.x];
    if(col < L && start+threadIdx.y < N)
      temp[block+index] = B[(start+threadIdx.y)*L+col];
    __syncthreads();
    if(col < L && row < M)
      for(int i = 0; i < tile && start + i < N; i++)
        reduce = cuCadd(reduce,cuCmul(temp[threadIdx.y*tile+i],temp[block+i*tile+threadIdx.x]));
    __syncthreads();
    start = start+tile;
  }
  if(col < L && row < M)C[row*L+col] = cuCadd(C[row*L+col],reduce);
}
__global__ void L2Pkernelh(cuDoubleComplex *C, cuDoubleComplex *A, cuDoubleComplex *B, int M, int N, int L){
  extern __shared__ cuDoubleComplex temp[];
  int tile = blockDim.x;
  int row = blockIdx.y*tile+threadIdx.y;
  int col = blockIdx.x*tile+threadIdx.x;
  int numtile = N/tile;
  if(N%tile)numtile++;
  cuDoubleComplex reduce = make_cuDoubleComplex(0,0);
  int index = threadIdx.y*tile+threadIdx.x;
  int block = tile*tile;
  int start = 0;
  for(int k = 0; k < numtile; k++){
    if(row < M && start+threadIdx.x < N)
      temp[index] = A[row*N+start+threadIdx.x];
    if(col < L && start+threadIdx.y < N)
      temp[block+index] = B[(start+threadIdx.y)*L+col];
    __syncthreads();
    if(col < L && row < M)
      for(int i = 0; i < tile && start + i < N; i++)
        reduce = cuCadd(reduce,cuCmul(temp[threadIdx.y*tile+i],temp[block+i*tile+threadIdx.x]));
    __syncthreads();
    start = start+tile;
  }
  if(col < L && row < M)C[row*L+col] = reduce;
}

__global__ void M2Mkernel(cuDoubleComplex *agg, cuDoubleComplex *aggl, int sampl, double *interp, int *intind, int ninter, cuDoubleComplex *shift){
  extern __shared__ cuDoubleComplex temp[];
  int clusm = blockIdx.y*gridDim.x+blockIdx.x;
  int start = clusm*4;
  if(threadIdx.x<sampl)
    for(int cn = 0; cn < 4; cn++)
      temp[cn*sampl+threadIdx.x] = aggl[(start+cn)*sampl+threadIdx.x];
  cuDoubleComplex reduce = make_cuDoubleComplex(0,0);
  cuDoubleComplex reg[4] = {0, 0, 0, 0};
  cuDoubleComplex intval;
  int indval = intind[threadIdx.x];
  __syncthreads();
  for(int k = 0; k < ninter; k++){
    intval = make_cuDoubleComplex(interp[k*blockDim.x+threadIdx.x],0);
    for(int m = 0; m < 4; m++)
      reg[m] = cuCadd(reg[m],cuCmul(intval,temp[m*sampl+(indval+k)%sampl]));
  }
  for(int m = 0; m < 4; m++)
    reduce = cuCadd(reduce,cuCmul(reg[m],shift[m*blockDim.x+threadIdx.x]));
  agg[clusm*blockDim.x+threadIdx.x] = reduce;
}
__global__ void M2Mkernel_output(cuDoubleComplex *agg, cuDoubleComplex *aggl, int samp, int sampl, int numclus, double *interp, int *intind, int ninter, cuDoubleComplex *shift){
  extern __shared__ cuDoubleComplex temp[];
  cuDoubleComplex *shifttemp = &temp[blockDim.x*ninter];
  int *tempind = (int*)&shifttemp[4*blockDim.x];
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  cuDoubleComplex reduce = make_cuDoubleComplex(0,0);
  cuDoubleComplex reg[4] = {0,0,0,0};
  int index = threadIdx.y*blockDim.x+threadIdx.x;
  int size = blockDim.y*blockDim.x;
  int start = blockIdx.x*blockDim.x*ninter;
  int ind = threadIdx.y;
  if(col < samp){
    while(ind < ninter){
      temp[ind*blockDim.x+threadIdx.x] = make_cuDoubleComplex(interp[ind*samp+col],0);
      ind = ind + blockDim.y;
    }
    if(threadIdx.y < 4)shifttemp[index] = shift[threadIdx.y*samp+col];
    if(threadIdx.y == 0)tempind[threadIdx.x] = intind[col];
  }
  __syncthreads();
  if(col < samp && row < numclus){
    for(int k = 0; k < ninter; k++){
      ind = (tempind[threadIdx.x]+k)%sampl;
      cuDoubleComplex coeff = temp[k*blockDim.x+threadIdx.x];
      for(int m = 0; m < 4; m++)
        reg[m] = cuCadd(reg[m],cuCmul(coeff,aggl[(4*row+m)*sampl+ind]));
    }
    for(int m = 0; m < 4; m++)
      reduce = cuCadd(reduce,cuCmul(reg[m],shifttemp[m*blockDim.x+threadIdx.x]));
    agg[row*samp+col] = reduce;
  }
}
__global__ void L2Lkernel_output(cuDoubleComplex *agg, cuDoubleComplex *aggl, int samp, int sampl, int numclus, double *interp, int *intind, int ninter, cuDoubleComplex *shift){
  extern __shared__ cuDoubleComplex temp[];
  int *tempind = (int*)&temp[ninter*blockDim.x];
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  cuDoubleComplex reg[4] = {0,0,0,0};
  int index = threadIdx.y*blockDim.x+threadIdx.x;
  int size = blockDim.y*blockDim.x;
  int start = blockIdx.x*blockDim.x*ninter;
  int ind = threadIdx.y;
  if(col < sampl){
    while(ind < ninter){
      temp[ind*blockDim.x+threadIdx.x] = make_cuDoubleComplex(interp[ind*sampl+col],0);
      ind = ind + blockDim.y;
    }
  }
  if(threadIdx.y==0)tempind[threadIdx.x] = intind[col];
  __syncthreads();
  if(col < sampl && row < numclus){
    for(int k = 0; k < ninter; k++){
      int ind = (tempind[threadIdx.x]+k)%samp;
      cuDoubleComplex coeff = temp[k*blockDim.x+threadIdx.x];
      cuDoubleComplex sample = cuCmul(coeff,agg[row*samp+ind]);
      for(int m = 0; m < 4; m++)
        reg[m] = cuCadd(reg[m],cuCmul(sample,shift[m*samp+ind]));
    }
    for(int m = 0; m < 4; m++)
      aggl[(4*row+m)*sampl+col] = cuCadd(aggl[(4*row+m)*sampl+col],reg[m]);
  }
}
__global__ void L2Lkernelh_output(cuDoubleComplex *agg, cuDoubleComplex *aggl, int samp, int sampl, int numclus, double *interp, int *intind, int ninter, cuDoubleComplex *shift){
  extern __shared__ cuDoubleComplex temp[];
  int *tempind = (int*)&temp[ninter*blockDim.x];
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  cuDoubleComplex reg[4] = {0,0,0,0};
  int index = threadIdx.y*blockDim.x+threadIdx.x;
  int size = blockDim.y*blockDim.x;
  int start = blockIdx.x*blockDim.x*ninter;
  int ind = threadIdx.y;
  if(col < sampl){
    while(ind < ninter){
      temp[ind*blockDim.x+threadIdx.x] = make_cuDoubleComplex(interp[ind*sampl+col],0);
      ind = ind + blockDim.y;
    }
  }
  if(threadIdx.y==0)tempind[threadIdx.x] = intind[col];
  __syncthreads();
  if(col < sampl && row < numclus){
    for(int k = 0; k < ninter; k++){
      int ind = (tempind[threadIdx.x]+k)%samp;
      cuDoubleComplex coeff = temp[k*blockDim.x+threadIdx.x];
      cuDoubleComplex sample = cuCmul(coeff,agg[row*samp+ind]);
      for(int m = 0; m < 4; m++)
        reg[m] = cuCadd(reg[m],cuCmul(sample,shift[m*samp+ind]));
    }
    for(int m = 0; m < 4; m++)
      aggl[(4*row+m)*sampl+col] = reg[m];
  }
}
__global__ void L2Lkernel(cuDoubleComplex *agg, cuDoubleComplex *aggl, int sampl, double *interp, int *intind, int ninter, cuDoubleComplex *shift){
  extern __shared__ cuDoubleComplex temp[];
  int clusn = blockIdx.y*gridDim.x+blockIdx.x;
  cuDoubleComplex samp = agg[clusn*blockDim.x+threadIdx.x];
  for(int cn = 0; cn < 4; cn++)
    temp[cn*blockDim.x+threadIdx.x] = cuCmul(samp,shift[cn*blockDim.x+threadIdx.x]);
  cuDoubleComplex reg[4] = {0,0,0,0};
  cuDoubleComplex intval;
  int indval = intind[threadIdx.x];
  __syncthreads();
  if(threadIdx.x < sampl){
    for(int k = 0; k < ninter; k++){
      intval = make_cuDoubleComplex(interp[k*sampl+threadIdx.x],0);
      int ind = (indval+k)%blockDim.x;
      for(int m = 0; m < 4; m++)
        reg[m] = cuCadd(reg[m],cuCmul(intval,temp[m*blockDim.x+ind]));
    }
    for(int m = 0; m < 4; m++)
      aggl[(clusn*4+m)*sampl+threadIdx.x] = cuCadd(aggl[(clusn*4+m)*sampl+threadIdx.x],reg[m]);
  }
}
__global__ void L2Lkernelh(cuDoubleComplex *agg, cuDoubleComplex *aggl, int sampl, double *interp, int *intind, int ninter, cuDoubleComplex *shift){
  extern __shared__ cuDoubleComplex temp[];
  int clusn = blockIdx.y*gridDim.x+blockIdx.x;
  cuDoubleComplex samp = agg[clusn*blockDim.x+threadIdx.x];
  for(int cn = 0; cn < 4; cn++)
    temp[cn*blockDim.x+threadIdx.x] = cuCmul(samp,shift[cn*blockDim.x+threadIdx.x]);
  cuDoubleComplex reg[4] = {0,0,0,0};
  cuDoubleComplex intval;
  int indval = intind[threadIdx.x];
  __syncthreads();
  if(threadIdx.x < sampl){
    for(int k = 0; k < ninter; k++){
      intval = make_cuDoubleComplex(interp[k*sampl+threadIdx.x],0);
      int ind = (indval+k)%blockDim.x;
      for(int m = 0; m < 4; m++)
        reg[m] = cuCadd(reg[m],cuCmul(intval,temp[m*blockDim.x+ind]));
    }
    for(int m = 0; m < 4; m++)
      aggl[(clusn*4+m)*sampl+threadIdx.x] = reg[m];
  }
}
__global__ void M2Lkernel_output(cuDoubleComplex *loc, cuDoubleComplex *agg, int samp, int numclus, int *far,int *traid, cuDoubleComplex *trans){
  extern __shared__ int tempfar[];
  int *tempid = &tempfar[blockDim.y*27];
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  cuDoubleComplex reduce = make_cuDoubleComplex(0,0);
  int index = threadIdx.y*blockDim.x+threadIdx.x;
  int size = blockDim.y*blockDim.x;
  int start = blockIdx.y*blockDim.y*27;
  int ind = index;
  while(ind < blockDim.y*27){
    tempfar[ind] = far[start+ind];
    tempid[ind] = traid[start+ind];
    ind = ind + size;
  }
  __syncthreads();
  if(col < samp && row < numclus){
    for(int cn = 0; cn < 27; cn++){
      int clusn = tempfar[threadIdx.y*27+cn];
      if(clusn != -1){
        int indmulti = clusn*samp;
        int index = tempid[threadIdx.y*27+cn]*samp;
        reduce = cuCadd(reduce,cuCmul(trans[index+col],agg[indmulti+col]));
      }
    }
    loc[row*samp+col] = reduce;
  }
}
/*__global__ void P2Pkernel(cuDoubleComplex *r, cuDoubleComplex *x, int *clusnear, cuDoubleComplex *near){
  extern __shared__ cuDoubleComplex temp[];
  int box = blockDim.x;
  int boxx = box*box;
  int *clutemp = (int*)&temp[boxx];
  int t = threadIdx.y*box+threadIdx.x;
  int clusm = blockIdx.y*gridDim.x+blockIdx.x;
  if(t < 9)clutemp[t] = clusnear[clusm*9+t];
  __syncthreads();
  cuDoubleComplex reduce = make_cuDoubleComplex(0,0);
  int clusn, indboxbase;
  for(int cn = 0; cn < 9; cn++){
    clusn = clutemp[cn];
    if(clusn != -1){
      __syncthreads();
      temp[t] = x[clusn*boxx+t];
      indboxbase = cn*boxx*boxx;
      __syncthreads();
      for(int n = 0; n < boxx; n++)
        reduce = cuCadd(reduce,cuCmul(temp[n],near[indboxbase+n*boxx+t]));
    }
  }
  r[clusm*boxx+t] = reduce;
}*/
//LARGE BLOCK
__global__ void P2Pkernel(cuDoubleComplex *r, cuDoubleComplex *x, int *clusnear, cuDoubleComplex *near){
  extern __shared__ cuDoubleComplex temp[];
  cuDoubleComplex *neartemp = &temp[blockDim.x];
  int boxx = blockDim.x/4;
  int t = threadIdx.x%boxx;
  int clusl = threadIdx.x/boxx;
  int clusm = blockIdx.x*4+clusl;
  int clusn,indboxbase;
  int *nearlist = (int*)&neartemp[blockDim.x];
  cuDoubleComplex reduce = make_cuDoubleComplex(0,0);
  if(t<9)nearlist[clusl*9+t] = clusnear[clusm*9+t];
  __syncthreads();
  for(int cn = 0; cn < 9; cn++){
    clusn = nearlist[clusl*9+cn];
    indboxbase = cn*boxx*boxx;
    __syncthreads();
    if(clusn!=-1)temp[threadIdx.x] = x[clusn*boxx+t];
    for(int n = 0; n < boxx; n++){
      if(n%4==0){
        __syncthreads();
        neartemp[threadIdx.x]=near[indboxbase+n*boxx+threadIdx.x];
        __syncthreads();
      }
      if(clusn != -1)reduce = cuCadd(reduce,cuCmul(neartemp[(n%4)*boxx+t],temp[clusl*boxx+n]));
    }
  }
  r[clusm*boxx+t] = reduce;
}
__global__ void locate(cuDoubleComplex *sendbuff, int *sendmap, cuDoubleComplex *aggmulti, int numsamp){
  int loop = 0;
  for(loop = 0; loop < numsamp/blockDim.x; loop++)
    sendbuff[blockIdx.x*numsamp+loop*blockDim.x+threadIdx.x] = aggmulti[sendmap[blockIdx.x]*numsamp+loop*blockDim.x+threadIdx.x];
  if(threadIdx.x < numsamp%blockDim.x)
    sendbuff[blockIdx.x*numsamp+loop*blockDim.x+threadIdx.x] = aggmulti[sendmap[blockIdx.x]*numsamp+loop*blockDim.x+threadIdx.x];
}
__global__ void relocate(cuDoubleComplex *aggmulti, int *recvmap, cuDoubleComplex *recvbuff, int numsamp){
  int loop = 0;
  for(loop = 0; loop < numsamp/blockDim.x; loop++)
    aggmulti[recvmap[blockIdx.x]*numsamp+loop*blockDim.x+threadIdx.x] = recvbuff[blockIdx.x*numsamp+loop*blockDim.x+threadIdx.x];
  if(threadIdx.x < numsamp%blockDim.x)
    aggmulti[recvmap[blockIdx.x]*numsamp+loop*blockDim.x+threadIdx.x] = recvbuff[blockIdx.x*numsamp+loop*blockDim.x+threadIdx.x];
}
__global__ void norm2partc(double *part, cuDoubleComplex *a){
  extern __shared__ double tempd[];
  int start = 2*blockIdx.x*blockDim.x;
  cuDoubleComplex var;
  var = a[start+threadIdx.x];
  tempd[threadIdx.x] = cuCreal(var)*cuCreal(var)+cuCimag(var)*cuCimag(var);
  var = a[start+blockDim.x+threadIdx.x];
  tempd[blockDim.x+threadIdx.x] = cuCreal(var)*cuCreal(var)+cuCimag(var)*cuCimag(var);
  for(int stride = blockDim.x; stride >= 1; stride >>= 1){
    __syncthreads();
    if(threadIdx.x < stride)
      tempd[threadIdx.x]=tempd[threadIdx.x]+tempd[threadIdx.x+stride];
  }
  if(threadIdx.x==0)
    part[blockIdx.x] = tempd[0];
}
__global__ void norm2partd(double *part){
  extern __shared__ double tempd[];
  int start = 2*blockIdx.x*blockDim.x;
  tempd[threadIdx.x] = part[start+threadIdx.x];
  tempd[blockDim.x+threadIdx.x] = part[start+blockDim.x+threadIdx.x];
  for(int stride = blockDim.x; stride >= 1; stride >>= 1){
    __syncthreads();
    if(threadIdx.x < stride)
      tempd[threadIdx.x]=tempd[threadIdx.x]+tempd[threadIdx.x+stride];
  }
  if(threadIdx.x==0)
    part[blockIdx.x] = tempd[0];
}
__global__ void innerpartc(cuDoubleComplex *part, cuDoubleComplex *a, cuDoubleComplex *b){
  extern __shared__ cuDoubleComplex temp[];
  int start = 2*blockIdx.x*blockDim.x;
  cuDoubleComplex var1, var2;
  var1 = a[start+threadIdx.x];
  var2 = b[start+threadIdx.x];
  temp[threadIdx.x] = cuCmul(cuConj(var1),var2);
  var1 = a[start+blockDim.x+threadIdx.x];
  var2 = b[start+blockDim.x+threadIdx.x];
  temp[blockDim.x+threadIdx.x] = cuCmul(cuConj(var1),var2);
  for(int stride = blockDim.x; stride >= 1; stride >>= 1){
    __syncthreads();
    if(threadIdx.x < stride)
      temp[threadIdx.x]=cuCadd(temp[threadIdx.x],temp[threadIdx.x+stride]);
  }
  if(threadIdx.x==0)
    part[blockIdx.x] = temp[0];
}
__global__ void innerpartd(cuDoubleComplex *part){
  extern __shared__ cuDoubleComplex temp[];
  int start = 2*blockIdx.x*blockDim.x;
  temp[threadIdx.x] = part[start+threadIdx.x];
  temp[blockDim.x+threadIdx.x] = part[start+blockDim.x+threadIdx.x];
  for(int stride = blockDim.x; stride >= 1; stride >>= 1){
    __syncthreads();
    if(threadIdx.x < stride)
      temp[threadIdx.x]=cuCadd(temp[threadIdx.x],temp[threadIdx.x+stride]);
  }
  if(threadIdx.x==0)
    part[blockIdx.x] = temp[0];
}
__global__ void prep(cuDoubleComplex *buff, cuDoubleComplex *x, cuDoubleComplex *o){
  int t = blockIdx.x*blockDim.x+threadIdx.x;
  buff[t] = cuCmul(x[t],o[t]);
}
__global__ void prep(cuDoubleComplex *buff, cuDoubleComplex *x){
  int t = blockIdx.x*blockDim.x+threadIdx.x;
  buff[t] = cuConj(x[t]);
}
__global__ void post(cuDoubleComplex *r_d, cuDoubleComplex *x_d){
  int t = blockIdx.x*blockDim.x+threadIdx.x;
  r_d[t] = cuCsub(x_d[t],r_d[t]);
}
__global__ void post(cuDoubleComplex *r, cuDoubleComplex *x, cuDoubleComplex *o){
  int t = blockIdx.x*blockDim.x+threadIdx.x;
  r[t] = cuCsub(x[t],cuConj(cuCmul(o[t],r[t])));
}
__global__ void saxp(cuDoubleComplex *a,cuDoubleComplex *b,cuDoubleComplex *c,cuDoubleComplex alpha){
  int t = blockIdx.x*blockDim.x+threadIdx.x;
  c[t] = cuCadd(a[t],cuCmul(alpha,b[t]));
}
