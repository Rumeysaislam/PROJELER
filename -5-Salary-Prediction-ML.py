# PROJECT: SALARY PREDICTION WITH MACHINE LEARNING

# IS Problemi

# Maas bilgileri ve 1986 yilina ait kariyer istatistikleri paylasilan beyzbol
# oyuncularinin maas tahminleri icin bir makine ogrenmesi projesi gerceklestirilebilir mi?

# Veri seti hikayesi

# Bu veri seti orijinal olarak Carnegie Mellon Universitesi'nde bulunan StatLib kutuphanesinden alinmistir.
# Veri seti 1988 ASA Grafik Bolumu Poster Oturumu'nda kullanilan verilerin bir parcasidir.
# Maas verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alinmistir.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayinlanan
# 1987 Beyzbol Ansiklopedisi Guncellemesinden elde edilmistir.


# AtBat         : 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vurus sayisi
# Hits          : 1986-1987 sezonundaki isabet sayisi
# HmRun         : 1986-1987 sezonundaki en degerli vurus sayisi
# Runs          : 1986-1987 sezonunda takimina kazandirdigi sayi
# RBI           : Bir vurucunun vurus yaptiginda kosu yaptirdigi oyuncu sayisi
# Walks         : Karsi oyuncuya yaptirilan hata sayisi
# Years         : Oyuncunun major liginde oynama suresi (sene)
# CAtBat        : Oyuncunun kariyeri boyunca topa vurma sayisi
# CHits         : Oyuncunun kariyeri boyunca yaptigi isabetli vurus sayisi
# CHmRun        : Oyucunun kariyeri boyunca yaptigi en degerli sayisi
# CRuns         : Oyuncunun kariyeri boyunca takimina kazandirdigi sayi
# CRBI          : Oyuncunun kariyeri boyunca kosu yaptirdirdigi oyuncu sayisi
# CWalks        : Oyuncun kariyeri boyunca karsi oyuncuya yaptirdigi hata sayisi
# League        : Oyuncunun sezon sonuna kadar oynadigi ligi gosteren A ve N seviyelerine sahip bir faktor
# Division      : 1986 sonunda oyuncunun oynadıgı pozisyonu gosteren E ve W seviyelerine sahip bir faktor
# PutOuts       : Oyun icinde takim arkadasinla yardimlasma
# Assits        : 1986-1987 sezonunda oyuncunun yaptıgı asist sayisi
# Errors        : 1986-1987 sezonundaki oyuncunun hata sayisi
# Salary        : Oyuncunun 1986-1987 sezonunda aldigi maas(bin uzerinden)
# NewLeague     : 1987 sezonunun basinda oyuncunun ligini gosteren A ve N seviyelerine sahip bir faktor



# GOREV: Veri on isleme ve ozellik muhendisligi tekniklerini kullanarak maas tahmin modeli gelistiriniz.



# Gerekli Kutuphane ve Fonksiyonlar;

import warnings
import pandas as pd
import missingno as msno
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from helpers.data_prep import *
from helpers.eda import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Tum Base Modeller
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

# Bazi uyarilari devre disi birakmak icin;
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


df = pd.read_csv("hitters.csv")
df.head()



### EDA ANALIZI (Exploratory Data Analysis);

df.describe()
check_df(df)

# Bagımlı degiskende 59 tane NA var!
# CAtBat, CHits outlier olabilir.


# BAGIMLI DEGISKEN ANALIZI;
import seaborn as sns
import matplotlib.pyplot as plt


df["Salary"].describe()                                        # max. degerin outlier oldugunu gorebiliyoruz.
sns.distplot(df.Salary)
plt.show(block=True)
# Grafigimiz simetrik degil, saga carpik dagilim.
# Bu dagilimi normallestirmek istersek; log. donusumu, karekok donusumu, min-max. donusumu kullanabiliriz. :)

sns.boxplot(df["Salary"])
plt.show()


# KATEGORIK VE NUMERIK DEGISKENLERIN SECILMESI;
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols


# KATEGORIK DEGISKEN ANALIZI;
rare_analyser(df, "Salary", cat_cols)


# SAYISAL DEGISKEN ANALIZI;
for col in num_cols:
    num_summary(df, col, plot=False)


# AYKIRI GOZLEM ANALIZI;
for col in num_cols:
    print(col, check_outlier(df, col, q1=0.1, q3=0.9))

# 1300 den sonraki degerleri veri setinden cikartiyorum.
# Atama yapmak dogru degil; yanliligi onlemek icin veriyi baskiliyoruz. (Aykiri degerler goz onune alinarak)

print(df.shape)
df = df[(df['Salary'] < 1350) | (df['Salary'].isnull())]                    # Eksik degerleri de istiyoruz.
print(df.shape)                                                             # Baskilama islemi sonrasinda degerler azaldi.
sns.distplot(df.Salary)
plt.show()


# AYKIRI DEGERLERİ BASKILAMA;
for col in num_cols:
    if check_outlier(df, col, q1=0.05, q3=0.95):
        replace_with_thresholds(df, col, q1=0.05, q3=0.95)

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))                    # Outlier olup olmadigini sorguladik.


# EKSİK GOZLEM ANALIZI;
missing_values_table(df)
# Salary bagimli degiskeninde 59 Eksik Gozlem bulunmakta. Bunlari cikartmak bir cozum yolu olabilir.
# Bagimli degisken old. icin doldurmak cok dogru olmuyor.

# KORELASYON ANALIZI;
import numpy as np

target_correlation_matrix(df, corr_th=0.3, target="Salary")
# Acik renge gittikce korelasyon artar. Korelasyonun artmasi iki degiskenin birbirini acikladigi anlamina gelir.
# Bu da modelin basarisini arttirir.
# % 75 uzeri yuksek korelasyon kabul edilebilir.
high_correlated_cols(df, plot=False, corr_th=0.90)
# Verilen esik degerine gore yuksek korelasyonlu olanlari bir listeye atadik.
# Su an icin bir sey yapmiyoruz, ileride lazim old. liste halinde kullanacagiz.



### VERI ONISLEME

# Yeni degiskenler olusturuyoruz;
df['NEW_HitRatio'] = df['Hits'] / df['AtBat']
df['NEW_RunRatio'] = df['HmRun'] / df['Runs']
df['NEW_CHitRatio'] = df['CHits'] / df['CAtBat']
df['NEW_CRunRatio'] = df['CHmRun'] / df['CRuns']

df['NEW_Avg_AtBat'] = df['CAtBat'] / df['Years']
df['NEW_Avg_Hits'] = df['CHits'] / df['Years']
df['NEW_Avg_HmRun'] = df['CHmRun'] / df['Years']
df['NEW_Avg_Runs'] = df['CRuns'] / df['Years']
df['NEW_Avg_RBI'] = df['CRBI'] / df['Years']
df['NEW_Avg_Walks'] = df['CWalks'] / df['Years']

# Paydaya sifir gelme ihtimaline karsilik paydadaki degiskenlere 1 eklenebilir. Sorun olmaz.


# One Hot Encoder

df = one_hot_encoder(df, cat_cols, drop_first=True)




### MODELLEME

df_null = df[df["Salary"].isnull()]                                     # Salary icerisindeki bos degerleri ayiralim.
df.dropna(inplace=True)                                                 # Salarydeki eksik degerleri cikartma

y = df['Salary']                                                        # Veri setinden bagimli degiskenimizi seciyoruz.
X = df.drop("Salary", axis=1)                                           # Veri setinden bag.li degiskeni ayikladik. Boylece bag.siz degiskenlerimi de secmis oldum.



# HOLD OUT - MODEL VALIDATION

# Train ve test setlerini ayiriyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)





# # ROBUST SCALER:
# ALL DATA FINAL RMSE: 219.83899058361285

# cols = X.columns
# index = X.index
# from sklearn.preprocessing import RobustScaler
# transformer = RobustScaler().fit(X)
# X = transformer.transform(X)
# X = pd.DataFrame(X, columns=cols, index=index)



# # STANDARD SCALER:
# ALL DATA FINAL RMSE: 186.16240421879607

# num_cols.remove("Salary")
# scaler = StandardScaler()
# df[num_cols] = scaler.fit_transform(df[num_cols])



# BASE MODELS

# Tum base modellerde kullanilabilecek bir fonk.;

# pip install catboost
# pip install lightgbm
import lightgbm as lgb
# pip install xgboost

def all_models(X, y, test_size=0.2, random_state=12345, classification=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)                                     # Modeli fit ediyoruz. Model burada ogreniyor.
            y_pred_train = model.predict(X_train)                           # Prediction elde ediyoruz.
            y_pred_test = model.predict(X_test)                             # Prediction elde ediyoruz.
            acc_train = accuracy_score(y_train, y_pred_train)               # Tahmin ile gercek arasindaki dogruluga bakiyoruz.
            acc_test = accuracy_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
            all_models.append(values)                                       # Olusturulan all_models listesine ekleniyor.

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df


all_models = all_models(X, y, test_size=0.2, random_state=46, classification=False)



# RANDOM FORESTS MODEL TUNING
# Tune etmek; Parametre optimizasyonu. Algoritma icindeki parametrelerin on tanımlı degerlerini uygun hale getiriyorum.

# Tuning icin hazirlanan parametreler. Tuning zaman aldigi icin cikan parametre degerlerini girdim.
rf_params = {"max_depth": [4, 5, 7, 10],
             "max_features": [4, 5, 6, 8, 10, 12],
             "n_estimators": [80, 100, 150, 250, 400, 500],
             "min_samples_split": [8, 10, 12, 15]}

# rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1).fit(X_train , y_train)
# rf_cv_model.best_params_

# Tune etmedik, islem uzun surdugu icin. :)
# GridSearchCV; girdigimiz parametre degerlerini deneyerek en uygununu seciyor.
# Cross validation yapiyoruz (cv). 10 parcaya bolup 9'u ile ogretip 1'i ile test yapiyor.


# Tune etseydik asagidaki optimum degerli vermis olacakti. Bu parametreleri veriyorum.
best_params = {'max_depth': 10,
               'max_features': 8,
               'min_samples_split': 10,
               'n_estimators': 80}

rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)                         # rf-modeli calistirdik.


# RANDOM FORESTS TUNED MODEL;
rf_tuned = RandomForestRegressor(max_depth=10, max_features=8, n_estimators=80,
                                 min_samples_split=10, random_state=42).fit(X_train, y_train)


# TUNED MODEL TRAIN HATASI;
y_pred = rf_tuned.predict(X_train)

print("RF Tuned Model Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred)))

# TUNED MODEL TEST HATASI

y_pred = rf_tuned.predict(X_test)
print("RF Tuned Model Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Onceki hata oranlariyla karsilastirmak istersek;
print(all_models)

# Hata oranlarin daha uygun hale geldigi gozlenmistir.
# Amacimiz; train ve test seti arasindaki farki azaltip tutarliligi saglamak.
# Modelin icerisinde olan plot_importance fonksiyonu biz onu gorsellestirecegiz;
# Hangi degiskenler en onemli, en etkili gormek istiyoruz;

def plot_importance(model, features, num=len(X), save=False):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_tuned, X_train)


# Tuned edilmis model nesnesinin kaydedilmesi "pickle" ile;
import pickle
pickle.dump(rf_tuned, open("rf_final_model.pkl", 'wb'))


# Tuned edilmis model nesnesinin yuklenmesi;
df_prep = pickle.load(open('rf_final_model.pkl', 'rb'))
