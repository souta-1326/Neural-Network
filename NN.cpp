#include<bits/stdc++.h>
using namespace std;
using F = double;
inline F sigmoid(F x){
  return 1/(1+exp(-x));
}
inline F sigmoid_dash(F x){
  return x*(1-x);
}
using Matrix = vector<vector<F>>;
template <F(*Act)(F),F(*Act_dash)(F),int... node_count> class NN{
  static const int Layer_count = sizeof...(node_count);
  int Node_count[Layer_count] = {node_count...};
  vector<F> A[Layer_count];
  Matrix W[Layer_count-1],dW[Layer_count-1];
  F lr;
public:
  NN(F I_lr):lr(I_lr){
    for(int i=0;i<Layer_count;i++){
      A[i].resize(Node_count[i]+(i<Layer_count-1));
    }
    for(int i=0;i<Layer_count-1;i++) A[i][Node_count[i]] = 1;
    for(int i=0;i<Layer_count-1;i++){
      W[i].resize(Node_count[i]+1,vector<F>(Node_count[i+1]));
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
      memset(&A[i+1][0],0,Node_count[i+1]*4);
      for(int k=0;k<Node_count[i]+1;k++){
        F bef_A = (i==0 || k==Node_count[i] ? A[i][k]:Act(A[i][k]));
        for(int j=0;j<Node_count[i+1];j++) A[i+1][j] += bef_A*W[i][k][j];
      }
    }
    for(int i=0;i<Layer_count;i++){
      for(int j=0;j<Node_count[i];j++) cout << A[i][j] << " ";
      cout << endl;
    }
    //逆伝播
    vector<F> delta(Node_count[Layer_count-1]);//転置されている
    for(int i=Layer_count-1;i>=1;i--){
      vector<F> delta_new(Node_count[i]);
      if(i == Layer_count-1){
        for(int j=0;j<Node_count[i];j++) delta_new[j] = (A[i][j]-t[j]);
      }
      else{
        for(int j=0;j<Node_count[i];j++){
          for(int k=0;k<Node_count[i+1];k++) delta_new[j] += W[i][j][k]*delta[k];
          delta_new[j] *= Act_dash(A[i][j]);
        }
      }
      delta = delta_new;
      for(int j=0;j<Node_count[i-1]+1;j++){
        for(int k=0;k<Node_count[i];k++) dW[i-1][j][k] = (i==1 || j==Node_count[i-1]? A[i-1][j]:Act(A[i-1][j]))*delta[k];
      }
    }
    for(int i=0;i<Layer_count-1;i++){
      for(int j=0;j<Node_count[i]+1;j++){
        for(int k=0;k<Node_count[i+1];k++){
          cout << i << j << k << ":" << dW[i][j][k] << endl;
          W[i][j][k] -= dW[i][j][k]*lr;
        }
      }
    }
  }
};
int main(){
  NN<sigmoid,sigmoid_dash,3,4,2> network(0.1);
  vector<Matrix> w
  ({{{0.73,0.42,0.34,0.53},{0.96,0.37,-0.43,0.06},{0.98,-0.79,-0.58,-0.96},{0.93,0.32,0.47,-1.00}},
    {{-0.98,0.08},{-0.52,-0.09},{-0.59,0.92},{-0.59,-0.07},{-0.67,0.70}}});
  network.W_set(w);
  vector<F> x({-0.65,-0.05,-0.78});
  vector<F> t({1,1});
  network.training(x,t);
  //network.training(x,t);
}
