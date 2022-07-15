#include"NN.hpp"
using namespace std;
void func1(){
  NN<sigmoid,sigmoid_dash,id,id_dash,3,4,2> network(0.1);
  vector<Matrix> w
  ({{{0.73,0.42,0.34,0.53},{0.96,0.37,-0.43,0.06},{0.98,-0.79,-0.58,-0.96},{0.93,0.32,0.47,-1.00}},
    {{-0.98,0.08},{-0.52,-0.09},{-0.59,0.92},{-0.59,-0.07},{-0.67,0.70}}});
  network.W_set(w);
  F x[3] = {-0.65,-0.05,-0.78};
  F t[2] = {1,1};
  network.training(x,t);
  network.output(x,t);
  for(F a:t) cout << a << endl;
}
void func2(){
  int time = clock();
  NN<tanh,tanh_dash,id,id_dash,1,10,1> NN_sin(0.01);
  F x[1],t[1];
  random_device rng;
  for(int i=0;i<1000000;i++){
    x[0] = double(rng())/UINT_MAX*8-4;
    t[0] = sin(x[0]);
    NN_sin.training(x,t);
    //printf("%lf %lf\n",x[0],t[0]);
  }
  //NN_sin.out_W();
  for(int i=0;i<10;i++){
    x[0] = double(rng())/UINT_MAX*8-4;
    NN_sin.output(x,t);
    printf("sin(%lf):%lf error:%lf loss:%lf\n",x[0],t[0],abs(sin(x[0])-t[0]),pow(abs(sin(x[0])-t[0]),2)/2);
  }
  cout << clock()-time << endl;
}
int main(){
  func2();
}
