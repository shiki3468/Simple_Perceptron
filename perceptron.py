import numpy as np
'''
単純パーセプトロンを提供するクラス。複数の特徴量を重みと掛け集めて正解を推測する
'''
class perceptron(object):
    '''
    eta:学習率（Π）。重みを修正するときに使用する。
    n_iter:機械学習を行う回数。
    random_state:ランダム関数のSEED値。
    ーーーーーーーーーーーーーーーーーーーーーー
　　クラスを呼び出すことで__init__()が呼び出され、この変数が生成される
    '''
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    '''
    X:対象の特徴量を持つ配列。
    y:重み付けの値が入った配列。
    ーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    トレーニング用メソッド
    '''
    def fit(self,X,y):
        #ランダム値のSEED値を設定する
        rgen = np.random.RandomState(self.random_state)
        #0.0~1.0の間の値を配列Xの2つ目の要素数分、重み変数に入れる
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        #配列errors_を生成
        self.errors_ = []
        
        #50回繰り返す
        for _ in range(self.n_iter):
            #初期化
            errors = 0
            #xiに特徴量の配列、targetに正解ラベルを入れる
            for xi,target in zip(X,y):
                #Δwi = Π(yi - y^i)(xi)
                #updataにΠ(yi - y^i)を入れる。
                update = self.eta * (target - self.predict(xi))
                #重みづけ変数にΔwiを追加する
                self.w_[1:] += update * xi
                #重みづけの初期値にΠ(yi - y^i)を入れる
                self.w_[0] += update
                #エラーの回数（trueの数)
                errors += int(update != 0.0)
            #エラー回数を配列の最後に追加する
            self.errors_.append(errors)
        return self

    #配列Xと配列w_のない席を求め、結果を返している
    #Σ
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    #内積の結果が0.0よりも大きければ1を、そうでなければ-1を返す。
    #閾値
    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
