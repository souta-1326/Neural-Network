#include<bits/stdc++.h>
using namespace std;
using F = double;
inline F sigmoid(F x){
  return 1/(1+exp(-x));
}
inline F sigmoid_dash(F x){
  F s = sigmoid(x);
  return s*(1-s);
}
inline F id(F x){return x;}
inline F id_dash(F x){return 1;}
inline F tanh_dash(F x){
  return 1/pow(cosh(x),2);
}
using Matrix = vector<vector<F>>;
template <F(*Hidden_Act)(F),F(*Hidden_Act_dash)(F),F(*Output_Act)(F),F(*Output_Act_dash)(F),int... _node_count> class NN{
  static constexpr int layer_count = sizeof...(_node_count);
  static constexpr int node_count[layer_count] = {_node_count...};
  static constexpr int max_node_count = *max_element(node_count,node_count+layer_count);
  F A[layer_count][max_node_count+1],Act_A[layer_count][max_node_count+1];
  F W[layer_count-1][max_node_count+1][max_node_count],dW[layer_count-1][max_node_count+1][max_node_count];
  F lr;
public:
  NN(F _lr = 0.01):lr(_lr){
    for(int i=0;i<layer_count-1;i++) A[i][node_count[i]] = Act_A[i][node_count[i]] = 1;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    for(int i=0;i<layer_count-1;i++){
      normal_distribution dist(0.0,1/sqrt(node_count[i]));
      for(int j=0;j<node_count[i]+1;j++){
        for(int k=0;k<node_count[i+1];k++){
          W[i][j][k] = dist(engine);
        }
      }
    }
  }
  void W_set(vector<Matrix> &w){
    for(int i=0;i<layer_count-1;i++){
      for(int j=0;j<node_count[i]+1;j++){
        for(int k=0;k<node_count[i+1];k++) W[i][j][k] = w[i][j][k];
      }
    }
  }
  void training(F x[node_count[0]],F t[node_count[layer_count-1]]){
    //順伝播
    memcpy(A[0],x,node_count[0]*sizeof(F));
    memcpy(Act_A[0],A[0],node_count[0]*sizeof(F));
    for(int i=0;i<layer_count-1;i++){
      fill(A[i+1],A[i+1]+node_count[i+1],0);
      for(int k=0;k<node_count[i]+1;k++){
        for(int j=0;j<node_count[i+1];j++) A[i+1][j] += Act_A[i][k]*W[i][k][j];
      }
      for(int j=0;j<node_count[i+1];j++) Act_A[i+1][j] = (i==layer_count-2 ? Output_Act:Hidden_Act)(A[i+1][j]);
    }
    //逆伝播
    static F delta[max_node_count],delta_new[max_node_count];//転置されている
    for(int i=layer_count-1;i>=1;i--){
      if(i == layer_count-1){
        for(int j=0;j<node_count[i];j++){
          delta[j] = Output_Act_dash(A[i][j])*(Act_A[i][j]-t[j]);
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
        F nex_A = Act_A[i-1][j];
        for(int k=0;k<node_count[i];k++) dW[i-1][j][k] = nex_A*delta[k];
      }
    }
    //反映
    for(int i=0;i<layer_count-1;i++){
      for(int j=0;j<node_count[i]+1;j++){
        for(int k=0;k<node_count[i+1];k++){
          W[i][j][k] -= dW[i][j][k]*lr;
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
    memcpy(A[0],x,node_count[0]*sizeof(F));
    for(int i=0;i<layer_count-1;i++){
      fill(A[i+1],A[i+1]+node_count[i+1],0);
      for(int k=0;k<node_count[i]+1;k++){
        for(int j=0;j<node_count[i+1];j++) A[i+1][j] += A[i][k]*W[i][k][j];
      }
      for(int j=0;j<node_count[i+1];j++) A[i+1][j] = (i==layer_count-2 ? Output_Act:Hidden_Act)(A[i+1][j]);
    }
    memcpy(ret,A[layer_count-1],node_count[layer_count-1]*sizeof(F));
  }
};
