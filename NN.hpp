#include<bits/stdc++.h>
using namespace std;
using F = float;
inline void sigmoid(int N,F A[],F ret[]){
  for(int i=0;i<N;i++) ret[i] = 1/(1+exp(-A[i]));
}
inline F sigmoid_dash(F x){
  F s = 1/(1+exp(-x));
  return s*(1-s);
}
inline F sigmoid_dash(F x,F y,F t){
  F s = 1/(1+exp(-x));
  return s*(1-s)*(y-t);
}
inline void id(int N,F A[],F ret[]){memcpy(ret,A,N*sizeof(F));}
inline F id_dash(F x,F y,F t){return (y-t);}
inline void tanh(int N,F A[],F ret[]){
  for(int i=0;i<N;i++) ret[i] = tanh(A[i]);
}
inline F tanh_dash(F x){
  return pow(cosh(x),-2);
}
inline void softmax(int N,F A[],F ret[]){
  F sum = 0;
  for(int i=0;i<N;i++) sum += (ret[i] = exp(A[i]));
  for(int i=0;i<N;i++) ret[i] /= sum;
}
inline F softmax_dash(F x,F y,F t){
  return y-t;
}
inline void relu(int N,F A[],F ret[]){
  for(int i=0;i<N;i++) ret[i] = max(A[i],F(0));
}
inline F relu_dash(F x){
  return x>=0;
}
template <void(*Hidden_Act)(int,F[],F[]),F(*Hidden_Act_dash)(F),void(*Output_Act)(int,F[],F[]),F(*Output_loss_Act_dash)(F,F,F),int... _node_count> class NN{
  static constexpr int layer_count = sizeof...(_node_count);
  static constexpr int node_count[layer_count] = {_node_count...};
  static constexpr int max_node_count = *max_element(node_count,node_count+layer_count);
  //F A[layer_count][max_node_count+1],Act_A[layer_count][max_node_count+1];
  F **A,**Act_A;
  //F W[layer_count-1][max_node_count_left+1][max_node_count_right],dv[layer_count-1][max_node_count_left+1][max_node_count_right];
  F ***W,***dv,***ds;
  F lr;
  static constexpr F Momentum_beta = 0.9;
  static constexpr F RMSProp_beta = 0.999;
  static constexpr F RMSProp_eps = 1e-8;
public:
  NN(F _lr = 0.001):lr(_lr){
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
    for(int i=0;i<layer_count-1;i++) A[i][node_count[i]] = Act_A[i][node_count[i]] = 1;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    for(int i=0;i<layer_count-1;i++){
      normal_distribution dist(0.0,2/sqrt(node_count[i]));
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
      fill(A[i+1],A[i+1]+node_count[i+1],0);
      for(int k=0;k<node_count[i]+1;k++){
        for(int j=0;j<node_count[i+1];j++) A[i+1][j] += Act_A[i][k]*W[i][k][j];
      }
      (i==layer_count-2 ? Output_Act:Hidden_Act)(node_count[i+1],A[i+1],Act_A[i+1]);
    }
    //逆伝播
    static F delta[max_node_count],delta_new[max_node_count];//転置されている
    for(int i=layer_count-1;i>=1;i--){
      if(i == layer_count-1){
        for(int j=0;j<node_count[i];j++){
          delta[j] = Output_loss_Act_dash(A[i][j],Act_A[i][j],t[j]);
        }
      }
      else{
        fill(delta_new,delta_new+node_count[i],0);
        for(int j=0;j<node_count[i];j++){
          for(int k=0;k<node_count[i+1];k++) delta_new[j] += W[i][j][k]*delta[k];
          delta_new[j] *= Hidden_Act_dash(A[i][j]);
        }
        memcpy(delta,delta_new,node_count[i]*sizeof(F));
      }
      for(int j=0;j<node_count[i-1]+1;j++){
        for(int k=0;k<node_count[i];k++){
          F w_now = Act_A[i-1][j]*delta[k];
          dv[i-1][j][k] = dv[i-1][j][k]*Momentum_beta+w_now*(1-Momentum_beta);
          ds[i-1][j][k] = ds[i-1][j][k]*RMSProp_beta+w_now*w_now*(1-RMSProp_beta);
        }
      }
    }
    //反映
    for(int i=0;i<layer_count-1;i++){
      for(int j=0;j<node_count[i]+1;j++){
        for(int k=0;k<node_count[i+1];k++){
          W[i][j][k] -= lr*dv[i][j][k]/(1-Momentum_beta)/sqrtf(ds[i][j][k]/(1-RMSProp_beta)+RMSProp_eps);
        }
      }
    }
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
      fill(A[i+1],A[i+1]+node_count[i+1],0);
      for(int k=0;k<node_count[i]+1;k++){
        for(int j=0;j<node_count[i+1];j++) A[i+1][j] += Act_A[i][k]*W[i][k][j];
      }
      (i==layer_count-2 ? Output_Act:Hidden_Act)(node_count[i+1],A[i+1],Act_A[i+1]);
    }
    memcpy(ret,Act_A[layer_count-1],node_count[layer_count-1]*sizeof(F));
  }
  ~NN() = default;
};
