"""
泰坦尼克号 核心数据分析 (Titanic EDA)
数据: test/titanic/titanic.csv (seaborn 版, 891 乘客)
分析: 生存率总览 / 性别 / 舱位 / 年龄 / 票价 / 亲属数 / 港口
输出: 控制台统计 + 中文可视化图表 /tmp/titanic_chart.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ---------- 中文字体 (必须在 set_theme 之后设置) ----------
sns.set_theme(style="whitegrid", palette="muted")

def setup_chinese_font():
    import glob
    font_files = []
    for pat in [
        "/usr/share/fonts/**/wqy-microhei.ttc",
        "/usr/share/fonts/**/wqy-zenhei.ttc",
        "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/*CJK*.ttc",
        "/usr/share/fonts/**/*WenQuanYi*",
    ]:
        font_files += glob.glob(pat, recursive=True)
    for fp in sorted(set(font_files)):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    zh_names = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                "Noto Sans CJK JP", "Noto Sans CJK SC"]
    available = [n for n in zh_names if any(f.name == n for f in fm.fontManager.ttflist)]
    if not available:
        available = sorted({f.name for f in fm.fontManager.ttflist
                            if any(k in f.name for k in ("CJK", "WenQuanYi"))})
    return available

zh_fonts = setup_chinese_font()
print("可用中文字体:", zh_fonts if zh_fonts else "无(默认)")
if zh_fonts:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = zh_fonts + ["DejaVu Sans"]
    fm.findfont(fm.FontProperties(family=zh_fonts[0]))
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 读数据 ----------
df = pd.read_csv("test/titanic/titanic.csv")
print("=" * 56)
print("泰坦尼克号 891 名乘客数据分析")
print("=" * 56)
print(f"数据形状: {df.shape}  ({len(df)} 乘客, {df.shape[1]} 字段)")

# ---------- 2. 数据质量 ----------
print("\n【缺失值】")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if (missing > 0).any() else "无缺失")
print(f"年龄缺失: {df['age'].isnull().sum()} 人 ({df['age'].isnull().mean()*100:.1f}%)")
print(f"票价缺失: {df['fare'].isnull().sum()} 人")

# ---------- 3. 核心生存率分析 ----------
print("\n【总生存率】")
surv_rate = df["survived"].mean() * 100
print(f"总体生存率: {surv_rate:.2f}%  (生存 {df['survived'].sum()} / {len(df)})")

print("\n【按性别生存率】")
sex_surv = df.groupby("sex")["survived"].agg(["mean", "count"])
sex_surv["mean"] = (sex_surv["mean"] * 100).round(2)
print(sex_surv.rename(columns={"mean": "生存率%", "count": "人数"}).to_string())

print("\n【按舱位生存率】")
class_surv = df.groupby("class")["survived"].agg(["mean", "count"])
class_surv["mean"] = (class_surv["mean"] * 100).round(2)
print(class_surv.rename(columns={"mean": "生存率%", "count": "人数"}).to_string())

print("\n【按登船港口生存率】")
port_surv = df.groupby("embark_town")["survived"].agg(["mean", "count"]).dropna()
port_surv["mean"] = (port_surv["mean"] * 100).round(2)
print(port_surv.rename(columns={"mean": "生存率%", "count": "人数"}).to_string())

print("\n【按是否独身生存率】")
alone_surv = df.groupby("alone")["survived"].agg(["mean", "count"])
alone_surv["mean"] = (alone_surv["mean"] * 100).round(2)
print(alone_surv.rename(columns={"mean": "生存率%", "count": "人数"}).to_string())

# ---------- 4. 年龄分布 ----------
print("\n【年龄统计】")
print(df["age"].describe().round(1).to_string())
print(f"儿童(<=14岁)生存率: {(df[df['age']<=14]['survived'].mean()*100):.1f}%")
print(f"老人(>=60岁)生存率: {(df[df['age']>=60]['survived'].mean()*100):.1f}%")

# ---------- 5. 可视化 (中文) ----------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 总生存率
sizes = [df["survived"].sum(), len(df) - df["survived"].sum()]
axes[0, 0].pie(sizes, labels=["生存", "遇难"], autopct="%1.1f%%",
               startangle=90, colors=["#55A868", "#C44E52"], explode=(0.05, 0))
axes[0, 0].set_title(f"总体生存率 {surv_rate:.1f}%")

# 性别生存
sns.barplot(data=df, x="sex", y="survived", ax=axes[0, 1])
axes[0, 1].set_title("按性别生存率")
axes[0, 1].set_ylabel("生存率")
axes[0, 1].set_xlabel("性别")

# 舱位生存
sns.barplot(data=df, x="class", y="survived", order=["First", "Second", "Third"], ax=axes[0, 2])
axes[0, 2].set_title("按舱位生存率")
axes[0, 2].set_ylabel("生存率")
axes[0, 2].set_xlabel("舱位")

# 年龄分布(生存 vs 遇难)
sns.histplot(data=df, x="age", hue="survived", kde=True,
             palette={0: "#C44E52", 1: "#55A868"}, multiple="dodge", ax=axes[1, 0])
axes[1, 0].set_title("年龄分布(生存 vs 遇难)")
axes[1, 0].set_xlabel("年龄")

# 票价箱线图(生存 vs 遇难)
sns.boxplot(data=df, x="survived", y="fare", ax=axes[1, 1])
axes[1, 1].set_title("票价与生存")
axes[1, 1].set_xlabel("生存(0=否,1=是)")
axes[1, 1].set_ylabel("票价")

# 家庭规模影响
df["家庭规模"] = df["sibsp"] + df["parch"]
fam = df.groupby("家庭规模")["survived"].mean() * 100
axes[1, 2].plot(fam.index, fam.values, marker="o", color="#4C72B0", linewidth=2)
axes[1, 2].set_title("家庭规模与生存率")
axes[1, 2].set_xlabel("家庭规模(同行亲属数)")
axes[1, 2].set_ylabel("生存率%")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
out = "/tmp/titanic_chart.png"
plt.savefig(out, dpi=130)
print(f"\n✅ 中文图表已保存: {out}")
print("\n" + "=" * 56)
print("泰坦尼克核心数据分析完成!")
print("TITANIC_ANALYSIS_OK")
