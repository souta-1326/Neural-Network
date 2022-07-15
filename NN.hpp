#include<bits/stdc++.h>
using namespace std;
using F = double;
inline F sigmoid(F x){
  return 1/(1+exp(-x));
}
inline F sigmoid_dash(F x){
  return x*(1-x);
}
inline F id(F x){return x;}
inline F id_dash(F x){return 1;}
using Matrix = vector<vector<F>>;
template <F(*Hidden_Act)(F),F(*Hidden_Act_dash)(F),F(*Output_Act)(F),F(*Output_Act_dash)(F),int... node_count> class NN{
  static const int Layer_count = sizeof...(node_count);
  const int Node_count[Layer_count] = {node_count...};
  vector<F> A[Layer_count];
  Matrix W[Layer_count-1],dW[Layer_count-1];
  F lr;
public:
  NN(F I_lr):lr(I_lr){
    for(int i=0;i<Layer_count;i++){
      A[i].resize(Node_count[i]+(i<Layer_count-1));
    }
    for(int i=0;i<Layer_count-1;i++) A[i][Node_count[i]] = 1;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    for(int i=0;i<Layer_count-1;i++){
      normal_distribution dist(0.0,1/sqrt(Node_count[i]));
      W[i].resize(Node_count[i]+1,vector<F>(Node_count[i+1]));
      for(int j=0;j<Node_count[i]+1;j++){
        for(int k=0;k<Node_count[i+1];k++){
          W[i][j][k] = dist(engine);
        }
      }
    }
    for(int i=0;i<Layer_count-1;i++){
      dW[i].resize(Node_count[i]+1,vector<F>(Node_count[i+1]));
    }
  }
  void W_set(vector<Matrix> &w){
    for(int i=0;i<Layer_count-1;i++){
      for(int j=0;j<Node_count[i]+1;j++){
        for(int k=0;k<Node_count[i+1];k++) W[i][j][k] = w[i][j][k];
      }
    }
  }
  void training(vector<F> &x,vector<F> &t){
    //順伝播
    for(int i=0;i<Node_count[0];i++) A[0][i] = x[i];
    for(int i=0;i<Layer_count-1;i++){
      fill(A[i+1].begin(),A[i+1].begin()+Node_count[i+1],0);
      for(int k=0;k<Node_count[i]+1;k++){
        for(int j=0;j<Node_count[i+1];j++) A[i+1][j] += A[i][k]*W[i][k][j];
      }
      for(int j=0;j<Node_count[i+1];j++) A[i+1][j] = (i==Layer_count-2 ? Output_Act:Hidden_Act)(A[i+1][j]);
    }
    // for(int i=0;i<Layer_count;i++){
    //   for(int j=0;j<Node_count[i];j++) cout << A[i][j] << " ";
    //   cout << endl;
    // }
    //逆伝播
    vector<F> delta(Node_count[Layer_count-1]);//転置されている
    for(int i=Layer_count-1;i>=1;i--){
      vector<F> delta_new(Node_count[i]);
      if(i == Layer_count-1){
        for(int j=0;j<Node_count[i];j++){
          delta_new[j] = Output_Act_dash(A[i][j])*(A[i][j]-t[j]);
        }
      }
      else{
        for(int j=0;j<Node_count[i];j++){
          for(int k=0;k<Node_count[i+1];k++) delta_new[j] += W[i][j][k]*delta[k];
          delta_new[j] *= Hidden_Act_dash(A[i][j]);
          //delta_new[j] *= Hidden_Act_dash(A[i][j]);
        }
      }
      delta = delta_new;
      for(int j=0;j<Node_count[i-1]+1;j++){
        for(int k=0;k<Node_count[i];k++) dW[i-1][j][k] = A[i-1][j]*delta[k];
      }
    }
    for(int i=0;i<Layer_count-1;i++){
      for(int j=0;j<Node_count[i]+1;j++){
        for(int k=0;k<Node_count[i+1];k++){
          //cout << i << j << k << ":" << dW[i][j][k] << endl;
          W[i][j][k] -= dW[i][j][k]*lr;
          //cout << i << j << k << ":" << W[i][j][k] << endl;
        }
      }
    }
  }
  void out_W(){
    for(int i=0;i<Layer_count-1;i++){
      for(int j=0;j<Node_count[i]+1;j++){
        for(int k=0;k<Node_count[i+1];k++){
          cout << i << j << k << ":" << W[i][j][k] << endl;
        }
      }
    }
  }
  vector<F> output(vector<F> &x){
    for(int i=0;i<Node_count[0];i++) A[0][i] = x[i];
    for(int i=0;i<Layer_count-1;i++){
      fill(A[i+1].begin(),A[i+1].begin()+Node_count[i+1],0);
      for(int k=0;k<Node_count[i]+1;k++){
        for(int j=0;j<Node_count[i+1];j++) A[i+1][j] += A[i][k]*W[i][k][j];
      }
      for(int j=0;j<Node_count[i+1];j++) A[i+1][j] = (i==Layer_count-2 ? Output_Act:Hidden_Act)(A[i+1][j]);
    }
    return A[Layer_count-1];
  }
};
