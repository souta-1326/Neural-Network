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
void func1(){
  NN<sigmoid,id,Optimizer::SGD,3,4,2> network(0.1);
  vector<vector<vector<F>>> w
  ({{{0.73,0.42,0.34,0.53},{0.96,0.37,-0.43,0.06},{0.98,-0.79,-0.58,-0.96},{0.93,0.32,0.47,-1.00}},
    {{-0.98,0.08},{-0.52,-0.09},{-0.59,0.92},{-0.59,-0.07},{-0.67,0.70}}});
  network.W_set(w);
  F x[3] = {-0.65,-0.05,-0.78};
  F t[2] = {1,1};
  network.training(x,t);
  network.output(x,t);
  for(F a:t) printf("%lf\n",a);
}
inline void test_func(F x[],F t[]){t[0] = sin(x[0]);}
inline F rand(F lef,F rig){
  static random_device rng;
  return F(rng())/UINT_MAX*(rig-lef)+lef;
}

void func2(){
  constexpr int TRAIN_SIZE = 1000000;
  constexpr int BATCH_SIZE = 1;
  constexpr int X_SIZE = 1;
  constexpr int Y_SIZE = 1;
  static_assert(TRAIN_SIZE%BATCH_SIZE == 0);
  static F x[TRAIN_SIZE][X_SIZE],t[TRAIN_SIZE][Y_SIZE],tx[X_SIZE],tt[Y_SIZE],ty[Y_SIZE];
  std::random_device rng;
  for(int i=0;i<TRAIN_SIZE;i++){
    for(int j=0;j<X_SIZE;j++) x[i][j] = rand(-acos(-1),acos(-1));
    for(int j=0;j<Y_SIZE;j++) test_func(x[i],t[i]);
  }
  Timer timer;
  NN<sigmoid,id,Optimizer::Adam,X_SIZE,10,Y_SIZE> NN_sin(0.001,0.9,0.999);
  for(int i=0;i<TRAIN_SIZE;i+=BATCH_SIZE){
    //NN_sin.training(x[i],t[i]);
    NN_sin.training_minibatch(BATCH_SIZE,x+i,t+i);
    //printf("%lf %lf\n",x[0],t[0]);
  }
  //NN_sin.out_W();
  for(int i=0;i<10;i++){
    for(int j=0;j<X_SIZE;j++) tx[j] = rand(-acos(-1),acos(-1));
    test_func(tx,ty);
    NN_sin.output(tx,tt);
    F error = 0;
    for(int j=0;j<Y_SIZE;j++) error += abs(tt[j]-ty[j]);
    error /= Y_SIZE;
    printf("error:%lf loss:%.10lf\n",error,error*error/2);
  }
  printf("time:%fsec\n",timer.elapsed());
}
int main(){
  func2();
}
