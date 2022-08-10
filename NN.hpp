#include<bits/stdc++.h>
using F = float;
constexpr inline void sigmoid(int N,F A[],F ret[]){
  for(int i=0;i<N;i++) ret[i] = 1/(1+std::exp(-A[i]));
}
constexpr inline F sigmoid_dash(F x){
  F s = 1/(1+std::exp(-x));
  return s*(1-s);
}
constexpr inline F sigmoid_dash(F x,F y,F t){
  F s = 1/(1+std::exp(-x));
  return s*(1-s)*(y-t);
}
inline void id(int N,F A[],F ret[]){memcpy(ret,A,N*sizeof(F));}
constexpr inline F id_dash(F x,F y,F t){return (y-t);}
constexpr inline void tanh(int N,F A[],F ret[]){
  for(int i=0;i<N;i++) ret[i] = std::tanh(A[i]);
}
constexpr inline F tanh_dash(F x){
  F cosh_x = std::cosh(x);return 1/(cosh_x*cosh_x);
}
constexpr inline void softmax(int N,F A[],F ret[]){
  F sum = 0;
  for(int i=0;i<N;i++) sum += (ret[i] = std::exp(A[i]));
  for(int i=0;i<N;i++) ret[i] /= sum;
}
constexpr inline F softmax_dash(F x,F y,F t){
  return y-t;
}
constexpr inline void relu(int N,F A[],F ret[]){
  for(int i=0;i<N;i++) ret[i] = std::max(A[i],F(0));
}
inline F relu_dash(F x){
  return x>=0;
}
template <void(*Hidden_Act)(int,F[],F[]),void(*Output_Act)(int,F[],F[]),int... _node_count> class NN{
  static constexpr int layer_count = sizeof...(_node_count);
  static constexpr int node_count[layer_count] = {_node_count...};
  static constexpr int max_node_count = *std::max_element(node_count,node_count+layer_count);
  static constexpr F(*Hidden_Act_dash)(F) = 
  (
    Hidden_Act == sigmoid ? (F(*)(F))sigmoid_dash:
    Hidden_Act == (void(*)(int,F[],F[]))tanh ? (F(*)(F))tanh_dash:
    Hidden_Act == relu ? (F(*)(F))relu_dash:
  nullptr);
  static constexpr F(*Output_loss_Act_dash)(F,F,F) =
  (
    Output_Act == id ? (F(*)(F,F,F))id_dash:
    Output_Act == sigmoid ? (F(*)(F,F,F))sigmoid_dash:
    Output_Act == softmax ? (F(*)(F,F,F))softmax_dash:
  nullptr);
  //F A[layer_count][max_node_count+1],Act_A[layer_count][max_node_count+1];
  F **A,**Act_A;
  //F W[layer_count-1][max_node_count_left+1][max_node_count_right],dv[layer_count-1][max_node_count_left+1][max_node_count_right];
  F ***W,***dv,***ds;//重み,motentum,RMSProp
  F **delta;
  F alpha;//学習率
  F momentum_beta;
  F sub_1_momentum_beta;
  F momentum_beta_powt;
  F RMSProp_beta;
  F sub_1_RMSProp_beta;
  F RMSProp_beta_powt;
  static constexpr F RMSProp_eps = 1e-6;
public:
  NN(F _alpha = 0.001,F _momentum_beta = 0.9,F _RMSProp_beta = 0.999):
  alpha(_alpha),
  momentum_beta(_momentum_beta),
  sub_1_momentum_beta(1-momentum_beta),
  momentum_beta_powt(momentum_beta),
  RMSProp_beta(_RMSProp_beta),
  sub_1_RMSProp_beta(1-RMSProp_beta),
  RMSProp_beta_powt(RMSProp_beta)
  {
    A = new F*[layer_count];
    for(int i=0;i<layer_count;i++) A[i] = new F[node_count[i]+1];
    Act_A = new F*[layer_count];
    for(int i=0;i<layer_count;i++) Act_A[i] = new F[node_count[i]+1];
    W = new F**[layer_count-1];
    for(int i=0;i<layer_count-1;i++){
      W[i] = new F*[node_count[i]+1];
      for(int j=0;j<node_count[i]+1;j++){
        W[i][j] = new F[node_count[i+1]];
      }
    }
    dv = new F**[layer_count-1];
    for(int i=0;i<layer_count-1;i++){
      dv[i] = new F*[node_count[i]+1];
      for(int j=0;j<node_count[i]+1;j++){
        dv[i][j] = new F[node_count[i+1]];
      }
    }
    ds = new F**[layer_count-1];
    for(int i=0;i<layer_count-1;i++){
      ds[i] = new F*[node_count[i]+1];
      for(int j=0;j<node_count[i]+1;j++){
        ds[i][j] = new F[node_count[i+1]];
      }
    }
    delta = new F*[layer_count];
    for(int i=1;i<layer_count;i++) delta[i] = new F[node_count[i]];
    for(int i=0;i<layer_count-1;i++) A[i][node_count[i]] = Act_A[i][node_count[i]] = 1;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    for(int i=0;i<layer_count-1;i++){
      std::normal_distribution dist(0.0,sqrt((Hidden_Act == relu ? 2.0:1.0)/node_count[i]));
      for(int j=0;j<node_count[i]+1;j++){
        for(int k=0;k<node_count[i+1];k++){
          W[i][j][k] = dist(engine);
        }
      }
    }
  }
  template<class T> void W_set(T &w){
    for(int i=0;i<layer_count-1;i++){
      for(int j=0;j<node_count[i]+1;j++){
        for(int k=0;k<node_count[i+1];k++) W[i][j][k] = w[i][j][k];
      }
    }
  }
  void training(F x[node_count[0]],F t[node_count[layer_count-1]]){
    //順伝播
    memcpy(Act_A[0],x,node_count[0]*sizeof(F));
    for(int i=0;i<layer_count-1;i++){
      std::fill(A[i+1],A[i+1]+node_count[i+1],0);
      for(int k=0;k<node_count[i]+1;k++){
        for(int j=0;j<node_count[i+1];j++) A[i+1][j] += Act_A[i][k]*W[i][k][j];
      }
      (i==layer_count-2 ? Output_Act:Hidden_Act)(node_count[i+1],A[i+1],Act_A[i+1]);
    }
    //逆伝播
    for(int i=layer_count-1;i>=1;i--){
      if(i == layer_count-1){
        for(int j=0;j<node_count[i];j++){
          delta[i][j] = Output_loss_Act_dash(A[i][j],Act_A[i][j],t[j]);
        }
      }
      else{
        std::fill(delta[i],delta[i]+node_count[i],0);
        for(int j=0;j<node_count[i];j++){
          for(int k=0;k<node_count[i+1];k++) delta[i][j] += W[i][j][k]*delta[i+1][k];
          delta[i][j] *= Hidden_Act_dash(A[i][j]);
        }
        //memcpy(delta,delta_new,node_count[i]*sizeof(F));
      }
    }
    for(int i=0;i<layer_count-1;i++){
      for(int j=0;j<node_count[i]+1;j++){
        for(int k=0;k<node_count[i+1];k++){
          F now_grad = Act_A[i][j]*delta[i+1][k];
          //dv[i-1][j][k] = dv[i-1][j][k]*momentum_beta+now_grad*(1-momentum_beta);
          dv[i][j][k] = dv[i][j][k]*momentum_beta+now_grad*sub_1_momentum_beta;
          //ds[i-1][j][k] = ds[i-1][j][k]*RMSProp_beta+(now_grad**2)*(1-RMSProp_beta);
          ds[i][j][k] = ds[i][j][k]*RMSProp_beta+now_grad*now_grad*sub_1_RMSProp_beta;
        }
      }
    }
    //反映
    F alphadiv_1submomentum_beta_powt = alpha/(1-momentum_beta_powt);
    F div_1subRMSProp_beta_powt = 1/(1-RMSProp_beta_powt);
    for(int i=0;i<layer_count-1;i++){
      for(int j=0;j<node_count[i]+1;j++){
        for(int k=0;k<node_count[i+1];k++){
          //W[i][j][k] -= dv[i][j][k]*alpha/(1-momentum_beta**t)/sqrtf(ds[i][j][k]/(1-RMSProp_beta**t)+RMSProp_eps);
          W[i][j][k] -= dv[i][j][k]*alphadiv_1submomentum_beta_powt/sqrtf(ds[i][j][k]*div_1subRMSProp_beta_powt+RMSProp_eps);
        }
      }
    }
    momentum_beta_powt *= momentum_beta;
    RMSProp_beta_powt *= RMSProp_beta;
  }
  void W_out(){
    for(int i=0;i<layer_count-1;i++){
      for(int j=0;j<node_count[i]+1;j++){
        for(int k=0;k<node_count[i+1];k++){
          fprintf(stderr,"%d %d %d: %lf\n",i,j,k,W[i][j][k]);
        }
      }
    }
  }
  void output(F x[node_count[0]],F ret[node_count[layer_count-1]]){
    memcpy(Act_A[0],x,node_count[0]*sizeof(F));
    for(int i=0;i<layer_count-1;i++){
      std::fill(A[i+1],A[i+1]+node_count[i+1],0);
      for(int k=0;k<node_count[i]+1;k++){
        for(int j=0;j<node_count[i+1];j++) A[i+1][j] += Act_A[i][k]*W[i][k][j];
      }
      (i==layer_count-2 ? Output_Act:Hidden_Act)(node_count[i+1],A[i+1],Act_A[i+1]);
    }
    memcpy(ret,Act_A[layer_count-1],node_count[layer_count-1]*sizeof(F));
  }
  ~NN() = default;
};
