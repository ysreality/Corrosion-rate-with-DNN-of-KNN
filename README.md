# Corrosion rate predict(Keras/KNN/alpha/beta)

## Summary
使用Keras架構下的DNN模型與其結合KNN模型，共同預測化工製程的H2O、pH值，以及腐蝕率。
分四階段進行，如下列所示：

第一階段：使用DNN預測含水量與pH值；

第二階段：使用knn model建立與API 581(腐蝕係數表)的相同的數據模型，先行計算出初步的腐蝕率(beta)；

第三階段：根據廠方提供的修正係數公式，計算出修正係數(alpha)；

第四階段：把alpha與beta相乘，得出修正後的腐蝕率(fixed cor rate)

### AI-model 1：Keras model(DNN)

安裝keras於虛擬環境中，還有須安裝tensorflow(2.2以上(含))，兩者是連動的
```
pip install keras
```

數據匯入後，先把數據分成85/15，15為驗證集，再把85拆成80/20的訓練集與測試集

訓練集與測試集的數據型態都需轉成矩陣，並使用正規化(min-max)，讓數據都壓縮至0~1之間

#訓練與測試的x&y正規化需分開設定，Ex：
```
trainx = preprocessing.MinMaxScaler()
```

#fit_transform()為正規化(0~1之間)
```
trainx_minmax = trainx.fit_transform(npx_train).reshape(npx_train.shape[0],npx_train.shape[1])
```

ps.若要還原到正常數值得化就使用原本的訓練集數據逆推
```
retrain_x = trainx.inverse_transform(trainx_minmax)
```
#Keras model build

input is 5, output is 1, dense is 1 step

經迭代搜尋法找尋後，神經元數128為較佳、batch_size為16較佳、使用500次迭代

活化函數使用ReLU，kernel使用he_normal，輸出層使用sigmoid收斂

使用模型評估指標，訓練集與測試集的R^2都在90%以上，RMSE與MAPE均極小，具有高度預測準確率

pH值要跟各管段操作溫度一起進入AI-model 2預測出初步的腐蝕率(beta)

### AI-model 2：knn model for API 581

因為API 581屬於非線性的腐蝕係數表資料庫，因此使用多種機器學習方法來嘗試建立，但都會使預測值產生負值，與原表差異過大。
經過多組嘗試後，最後找到了KNN，能順利解決負值的問題

```
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)
```

用混淆矩陣呈現預測結果
```
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(ytest,pred))
```

### AI-model 1 and 2 to calculate corrosion rate

輸入各管段的操作溫度與經AI-model 1預測出來的pH值，進AI-model 2得出初步的腐蝕率(beta)

經AI-model 1預測出來的含水量預測值，透過廠方提供的修正係數公式表，得出各管段的修正係數(alpha)

最後再將各管段修正係數(alpha)與各管段初步的腐蝕率(beta)相乘，得出修正後的腐蝕率，即為各管段的腐蝕率
