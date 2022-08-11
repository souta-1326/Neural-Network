# Neural-Network
将来的にAlphaZeroを自作するために、C++で高速に動作するようにしたニューラルネットワークです。\
まだ未完成なので、現時点ではオンライン学習しか対応しておらず、可読性も皆無です。
## Reference
### 前提
- `using F = float` とします。
- 出力層の活性化関数に`id`、`sigmoid`を用いた場合は損失関数は二乗和誤差、`softmax`を用いた場合は損失関数は交差エントロピー誤差となります。
### Constructor
```cpp
NN<Hidden_Act, Output_Act, Optimize, ..._node_count> network(F alpha,F momentum_beta,F RMSProp_beta);
```
- `Hidden_Act`:隠れ層の活性化関数 (sigmoid,tanh,relu)
- `Output_Act`:出力層の活性化関数 (id,sigmoid,softmax)
- `Optimize`:最適化アルゴリズム (Optimizer::SGD,Optimizer::Adam)
- `_node_count`:各層のノード数 (最初の値が入力層のノード数、最後の値が出力層のノード数)
- `alpha`:学習率 デフォルト:0.001
- `momentum_beta`:モーメンタムのハイパーパラメータ デフォルト:0.9
- `RMSProp_beta`:RMSPropのハイパーパラメータ デフォルト:0.999
### training
```cpp
void network.training(F x[],F t[]);
```
入力データ`x`、それに対応する出力データ`t`を引数に取ってオンライン学習をします。
### output
```cpp
void network.output(F x[],F ret[]);
````
入力データ`x`に対する予測を`ret`に埋め込みます。
