#include"NN.hpp"
using namespace std;
constexpr F lr1(){return 0.1;}
constexpr F lr2(){return 0.2;}
void func1(){
  NN<sigmoid,id,3,4,2> network(0.1);
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
inline F test_func(F x){return sin(x);}
inline F rand(F lef,F rig){
  static random_device rng;
  return F(rng())/UINT_MAX*(rig-lef)+lef;
}

void func2(){
  constexpr int TRAIN_SIZE = 1000000;
  static F x[TRAIN_SIZE][1],t[TRAIN_SIZE][1],tx[1],tt[1];
  std::random_device rng;
  for(int i=0;i<TRAIN_SIZE;i++){
    x[i][0] = rand(0,2*acos(-1));
    t[i][0] = test_func(x[i][0]);
  }
  int time = clock();
  NN<sigmoid,id,1,10,1> NN_sin(0.001,0.9,0.9);
  for(int i=0;i<TRAIN_SIZE;i++){
    NN_sin.training(x[i],t[i]);
    //printf("%lf %lf\n",x[0],t[0]);
  }
  //NN_sin.out_W();
  for(int i=0;i<10;i++){
    tx[0] = rand(0,2*acos(-1));
    NN_sin.output(tx,tt); 
    printf("test_func(%lf):%lf error:%lf loss:%.10lf\n",tx[0],tt[0],abs(test_func(tx[0])-tt[0]),pow(abs(test_func(tx[0])-tt[0]),2)/2);
  }
  printf("time:%lu\n",clock()-time);
}
int main(){
  func2();
}
