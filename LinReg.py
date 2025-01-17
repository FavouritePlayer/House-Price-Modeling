import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Datasets
df = pd.read_csv('House Prices/train.csv')
X = df[['LotArea']]
y = df.SalePrice
pred_df = pd.read_csv('House Prices/test.csv')

#Modeling
linReg = LinearRegression().fit(X, y)

#Evaluate Model on Test Data
X_test = pred_df[['LotArea']]
y_pred_test = linReg.predict(X_test)

#Results
coef = linReg.coef_[0]
intercept = linReg.intercept_
r2 = r2_score(y_true=y, y_pred=linReg.predict(X))

#Plot
plt.figure(figsize=(12, 6))

#Plot Training Data
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.5, label='Training Data')
plt.plot(X, linReg.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title('Lot Area vs Sale Price (Training Data)')
plt.xlabel('Lot Area (Square Feet)')
plt.ylabel('Sale Price (USD)')
plt.legend()
plt.grid(True)

#Plot for Test Data vs Predicted Sale Price
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_pred_test, color='green', alpha=0.5, label='Predicted Sale Price')
plt.title('Lot Area vs Predicted Sale Price (Test Data)')
plt.xlabel('Lot Area (Square Feet)')
plt.ylabel('Predicted Sale Price (USD)')
plt.legend()
plt.grid(True)

#Add Statistics
stats_text = f"$R^2$: {r2:.3f}\nCoefficient: {coef:.2f}\nIntercept: {intercept:.2f}"
plt.subplot(1, 2, 1)
plt.text(0.95, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.show()
