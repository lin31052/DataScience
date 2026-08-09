"""电商预测 Step5 最优模型 - 最终准确率评估 (独立脚本)"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.chdir("/root/DataScience")
df = pd.read_csv("test/retail_sales/online_retail.csv", parse_dates=["InvoiceDate"])
uk = df[df["Country"]=="United Kingdom"].copy()
uk = uk[uk["CustomerID"].notna()]; uk = uk[uk["Quantity"]>0]
uk = uk[~uk["InvoiceNo"].astype(str).str.startswith("C")]
uk["Revenue"] = uk["Quantity"]*uk["UnitPrice"]
uk["Day"] = uk["InvoiceDate"].dt.floor("D")
rev = uk.groupby("Day")["Revenue"].sum().astype(float).sort_index()

dn = ["周一","周二","周三","周四","周五","周六","周日"]
f = pd.DataFrame(index=rev.index); f["target"]=rev
for d in range(7): f[dn[d]] = (rev.index.dayofweek==d).astype(int)
for k in (1,2,3,7,14,21): f[f"lag{k}"] = rev.shift(k)
f["roll7"] = rev.shift(1).rolling(7).mean(); f["dom"] = rev.index.day
f = f.dropna(); X = f.drop(columns=["target"]); y = f["target"]

def wfm(nf=14, mt=60):
    preds={}
    for start in range(mt, len(y)-nf+1, nf):
        m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        m.fit(X.iloc[:start], y.iloc[:start])
        n2 = min(nf, len(y)-start)
        fp = m.predict(X.iloc[start:start+n2])
        for j in range(n2): preds[start+j] = fp[j]
    pa = np.full(len(y), np.nan)
    for k,v in preds.items(): pa[k]=v
    mk = ~np.isnan(pa)
    return y.values[mk], pa[mk]

rt, pt = wfm()
mae = mean_absolute_error(rt, pt); rmse = np.sqrt(mean_squared_error(rt, pt))
mape = np.mean(np.abs((rt-pt)/np.maximum(rt,1)))*100
med = np.median(np.abs(pt-rt)); r2 = r2_score(rt, pt)

print("="*46)
print("电商预测 当前最优模型 最终准确率 (Step5日度RF)")
print("="*46)
print(f"测试天数     : {len(rt)} 天")
print(f"日均销售额   : £{rt.mean():,.0f}")
print(f"MAE         : £{mae:,.0f}")
print(f"RMSE        : £{rmse:,.0f}")
print(f"MAPE        : {mape:.1f}%")
print(f"单日误差中位 : £{med:,.0f}")
print(f"R²          : {r2:.3f}")
print()
print("对比历史: 周度RF RMSE=£76k → 周度Naive £42k → 日度RF £15k")
