import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import statsmodels .api as sm
from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency

conn=sqlite3.connect("universite_ogrencileri.db")

df=pd.read_sql("select * from ogrenciler",conn)
print(df.head())
print(df.describe())
print(df.isnull().sum())
#sınav notları dağılımları
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df["Sinav_1"],kde=True,bins=20,color='skyblue')
plt.title("1.Sınav dağılımları")
plt.subplot(1,2,2)
sns.histplot(df["Sinav_2"],kde=True,bins=20,color='salmon')
plt.title("2.Sınav dağılımları")
plt.tight_layout()
plt.show()

#Sınav notları arası korelasyon
sns.heatmap(df[["Sinav_1","Sinav_2","Memnuniyet_Skoru"]].corr(),annot=True,cmap='coolwarm')
plt.title("Korelasyon metrisi")
plt.show()

#Bölümlere göre ortalama sınav sonuçları
df.groupby("Bolum")[["Sinav_1","Sinav_2"]].mean().plot(kind='bar',figsize=(12,7))
plt.title("Bölümlere göre ortalama sınav sonuçları")
plt.ylabel("Not ortalaması")
plt.show()

#Kadın ve erkek gruplarını ayırma
erkek=df[df["Cinsiyet"]=="Erkek"]["Sinav_2"].dropna()
kadın=df[df["Cinsiyet"]=="Kadın"]["Sinav_2"].dropna()

#cinsiyete göre not farkı
sns.boxplot(x="Cinsiyet",y="Sinav_2",data=df)
plt.title("Cinsiyete göre 2.sınav notları")
plt.show()

#Shapiro-Wilk testi
print("Shapiro test sonuçları")
print("Erkek",shapiro(erkek))
print("Kadın",shapiro(kadın))
#histogram
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(erkek,kde=True,color='blue',bins=10)
plt.title("Erkek 2.sınav dağılımı")
plt.subplot(1,2,2)
sns.histplot(kadın,kde=True,color='pink',bins=10)
plt.title("Kadın 2.sınav dağılımı")
plt.show()

#QQplot(verinin dağılımı)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sm.qqplot(erkek,line='s',ax=plt.gca())
plt.title("Erkek QQplot")
plt.subplot(1,2,2)
sm.qqplot(kadın,line='s',ax=plt.gca())
plt.title("Kadın QQplot")
plt.tight_layout()
plt.show()

#Levene testi(Gruplar arası varyanslar eşitmi)
erkek_sinav2=df[df["Cinsiyet"]=="Erkek"]["Sinav_2"]
kadin_sinav2=df[df["Cinsiyet"]=="Kadın"]["Sinav_2"]
print("Levene test sonuçları")
levene_test=levene(erkek_sinav2,kadin_sinav2)
print(levene_test)

#Bağımsız ttest
ttest_result=ttest_ind(erkek_sinav2,kadin_sinav2,equal_var=levene_test.pvalue>0.05)
print("Bağımsız t-testi sonucu:",ttest_result)

#paired t-test 
paired_ttest=ttest_rel(df["Sinav_1"],df["Sinav_2"])
print("Paired t-test sonucu:",paired_ttest)

#Bölümlere göre sınav 2
bolumler=df["Bolum"].unique()
gruplar=[df[df["Bolum"]==bolum]["Sinav_2"].dropna() for bolum in bolumler]
anova_result=f_oneway(*gruplar)
print("Anova sonucu:",anova_result)

#Cinsiyet-mezun olma(ki kare testi)
ctab=pd.crosstab(df["Cinsiyet"],df["Mezun_Olma"])
chi2,p,dof,expected=chi2_contingency(ctab)
print("Ki kare testi sonucu:")
print(f"chi2: {chi2}, p-value: {p}")

print("Korelasyonlar:")
print("Ders_Katilim_Yuzdesi ↔ Memnuniyet_Skoru:", df["Ders_Katilim_Yuzdesi"].corr(df["Memnuniyet_Skoru"]))
print("Stres_Skoru ↔ Sinav_2:", df["Stres_Skoru"].corr(df["Sinav_2"]))
