#include"NN.hpp"
using namespace std;
struct Timer {
#ifdef _OPENMP
  double st;

  Timer() { reset(); }

  inline void reset() { st = omp_get_wtime(); }

  inline double elapsed() {
    return omp_get_wtime()-st;
  }
#else
  chrono::high_resolution_clock::time_point st;

  Timer() { reset(); }

  void reset() { st = chrono::high_resolution_clock::now(); }

  double elapsed() {
    auto ed = chrono::high_resolution_clock::now();
    return double(chrono::duration_cast<chrono::nanoseconds>(ed - st).count())/1000000000;
  }
#endif
};
constexpr int LN = 60000,TN = 10000;
constexpr int R = 28;
constexpr int C = 28;
FILE *fp1 = fopen("train-images-idx3-ubyte","r");
FILE *fp2 = fopen("train-labels-idx1-ubyte","r");
FILE *fp3 = fopen("t10k-images-idx3-ubyte","r");
FILE *fp4 = fopen("t10k-labels-idx1-ubyte","r");
void skip(){
  for(int i=0;i<16;i++) getc_unlocked(fp1);
  for(int i=0;i<8;i++) getc_unlocked(fp2);
  for(int i=0;i<16;i++) getc_unlocked(fp3);
  for(int i=0;i<8;i++) getc_unlocked(fp4);
}
void input1(F x[],F t[]){
  for(int i=0;i<R*C;i++) x[i] = F(getc_unlocked(fp1))/256;
  int num = getc_unlocked(fp2);
  fill(t,t+10,0);t[num] = 1;
}
void input2(F x[],F t[]){
  for(int i=0;i<R*C;i++) x[i] = F(getc_unlocked(fp3))/256;
  int num = getc_unlocked(fp4);
  fill(t,t+10,0);t[num] = 1;
}
F lx[LN][R*C],lt[LN][10];
F tx[TN][R*C],tt[TN][10],ret[10];
int main(){
  NN<relu,softmax,Optimizer::SGD,R*C,800,10> network(0.01);
  skip();
  for(int i=0;i<LN;i++) input1(lx[i],lt[i]);
  for(int i=0;i<TN;i++) input2(tx[i],tt[i]);
  Timer timer;
  for(int i=0;i<LN;i++){
    network.training(lx[i],lt[i]);
    if((i&255)==0) printf("Done:%d\n",i);
  }
  printf("Training Time: %.6lfsec\n",timer.elapsed());
  int AC_count = 0;
  for(int i=0;i<TN;i++){
    network.output(tx[i],ret);
    AC_count += (max_element(tt[i],tt[i]+10)-tt[i]==max_element(ret,ret+10)-ret);
  }
  printf("Accuracy: %.4lf\n",double(AC_count)/TN);
}