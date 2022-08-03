#Telco Customer Churn Feature Engineering

# Problem: sirketi terk edecek musterileri tahmin edebilecek bir makine ogrenmesi modeli gelistirilmesi
# istenmektedir. Modeli gelistirmeden once gerekli olan veri analizi ve ozellik muhendisligi adimlarini
# gerceklestirmeniz beklenmektedir.

# Telco musteri churn verileri, ucuncu ceyrekte Kaliforniya'daki 7043 musteriye ev telefonu ve internet hizmetleri
# saglayan hayali bir telekom sirketi hakkinda bilgi icerir. Hangi musterilerin hizmetlerinden ayrildigini,
# kaldigini veya hizmete kaydoldugunu icermektedir.

# 21 Degisken 7043 Gozlem

# CustomerId                : Musteri İd’si
# Gender                    : Cinsiyet
# SeniorCitizen             : Musterinin yasli olup olmadigi (1, 0)
# Partner                   : Musterinin bir ortagi olup olmadigi (Evet, Hayir) Evli olup olmama
# Dependents                : Musterinin bakmakla yukumlu oldugu kisiler olup olmadigi (Evet, Hayir) (cocuk, anne, baba, buyukanne)
# tenure                    : Musterinin sirkette kaldigi ay sayisi
# PhoneService              : Musterinin telefon hizmeti olup olmadigi (Evet, Hayir)
# MultipleLines             : Musterinin birden fazla hatti olup olmadigi (Evet, Hayir, Telefon hizmeti yok)
# InternetService           : Musterinin internet servis saglayicisi (DSL, Fiber optik, Hayir)
# OnlineSecurity            : Musterinin cevrimici guvenliginin olup olmadigi (Evet, Hayir, internet hizmeti yok)
# OnlineBackup              : Musterinin online yedeginin olup olmadigi (Evet, Hayir, internet hizmeti yok)
# DeviceProtection          : Musterinin cihaz korumasina sahip olup olmadigi (Evet, Hayir, internet hizmeti yok)
# TechSupport               : Musterinin teknik destek alip almadigi (Evet, Hayir, internet hizmeti yok)
# StreamingTV               : Musterinin TV yayini olup olmadigi (Evet, Hayir, internet hizmeti yok) Musterinin, bir ucuncu taraf saglayicidan televizyon programlari yayinlamak icin internet hizmetini kullanip kullanmadigini gosterir.
# StreamingMovies           : Musterinin film akisi olup olmadigi (Evet, Hayir, Internet hizmeti yok) Musterinin bir ucuncu taraf saglayicidan film akisi yapmak icin İnternet hizmetini kullanip kullanmadigini gosterir.
# Contract                  : Musterinin sozlesme suresi (Aydan aya, Bir yil, İki yil)
# PaperlessBilling          : Musterinin kagitsiz faturasi olup olmadigi (Evet, Hayir)
# PaymentMethod             : Musterinin odeme yontemi (Elektronik cek, Posta ceki, Banka havalesi (otomatik), Kredi karti (otomatik))
# MonthlyCharges            : Musteriden aylik olarak tahsil edilen tutar
# TotalCharges              : Musteriden tahsil edilen toplam tutar
# Churn                     : Musterinin kullanip kullanmadigi (Evet veya Hayir) - Gecen ay veya ceyreklik icerisinde ayrilan musteriler


# Her satır benzersiz bir musteriyi temsil etmekte.
# Degiskenler musteri hizmetleri, hesap ve demografik veriler hakkinda bilgiler icerir.
# Musterilerin kaydoldugu hizmetler - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Musteri hesap bilgileri – ne kadar suredir musteri olduklari, sozlesme, odeme yontemi, kagitsiz faturalandirma, aylik ucretler ve toplam ucretler
# Musteriler hakkinda demografik bilgiler - cinsiyet, yas araligi ve ortaklari ve bakmakla yukumlu olduklari kisiler olup olmadigi


# GOREV 1: KESİFCİ VERİ ANALİZİ
           # Adım 1: Genel resmi inceleyiniz.
           # Adım 2: Numerik ve kategorik degiskenleri yakalayiniz.
           # Adım 3: Numerik ve kategorik degiskenlerin analizini yapiniz.
           # Adım 4: Hedef degisken analizi yapiniz. (Kategorik degiskenlere gore hedef degiskenin ortalamasi, hedef degiskene gore numerik degiskenlerin ortalamasi)
           # Adım 5: Aykiri gozlem analizi yapiniz.
           # Adım 6: Eksik gozlem analizi yapiniz.
           # Adım 7: Korelasyon analizi yapiniz.

# GOREV 2: FEATURE ENGINEERING
           # Adım 1: Eksik ve aykiri degerler icin gerekli islemleri yapiniz.
           # Adım 2: Yeni degiskenler olusturunuz.
           # Adım 3: Encoding islemlerini gerceklestiriniz.
           # Adım 4: Numerik degiskenler icin standartlastirma yapiniz.
           # Adım 5: Model olusturunuz.


# Gerekli Kutuphane ve Fonksiyonlar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("feature_engineering/datasets/Telco-Customer-Churn.csv")

df.head()
df.shape
df.info()
df.isnull().sum()
# TotalCharges sayisal bir degisken olmali
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df.head()
df["Churn"].unique()
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

df.head()


# GOREV 1: KESIFCI VERI ANALIZI


# GENEL RESIM;

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


# NUMERİK VE KATEGORIK DEGISKENLERIN YAKALANMASI;

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


# KATEGORIK DEGİSKENLERIN ANALIZI;

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


# NUMERIK DEGISKENLERIN ANALIZI;

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)

# Tenure'e bakildiginda 1 aylik musterilerin cok fazla oldugunu ardindan da 72 aylik musterilerin geldigini goruyoruz.
# tenure : Musterinin sirkette kaldigi ay sayisi

df["tenure"].value_counts().head()


# Farkli kontratlardan dolayi gerceklesmis olabilir, aylik sozlesmesi olan kisilerin tenure ile 2 yillik sozlesmesi olan kisilerin tenure'ne bakalim.
# MonthlyCharges : Musteriden aylik olarak tahsil edilen tutar

df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show()

# MonthyChargers'a bakildiginda aylik sozlesmesi olan musterilerin aylik ortalama odemeleri daha fazla olabilir.
df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show()

# 66.39849032258037
df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].mean()

df[df["Contract"] == "Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Two year")
plt.show()

# 60.770412979351
df[df["Contract"] == "Two year"]["MonthlyCharges"].mean()



# NUMERIK DEGISKENLERIN TARGET GORE ANALIZI;

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Churn", col)



# KATEGORIK DEGİSKENLERIN TARGET'A GORE ANALIZI;

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)



# KORELASYON;

df[num_cols].corr()

df.corr()


# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# TotalChargers'in aylik ucretler ve tenure ile yuksek korelasyonlu oldugu gorulmekte




# GOREV 2: FEATURE ENGINEERING;


# EKSIK DEGER ANALIZI;

df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df.isnull().sum()



# BASE MODEL KURULUMU;

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


dff = one_hot_encoder(dff, cat_cols, drop_first=True)


y = dff["Churn"]
X = dff.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")

# Accuracy: 0.7837
# Recall: 0.6333
# Precision: 0.4843
# F1: 0.5489
# Auc: 0.7282



# AYKIRI DEGER ANALIZI;

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Deger Analizi ve Baskilama Islemi
for col in num_cols:
    print(col, check_outlier(df, col))



# OZELLIK CIKARIMI;

# Tenure  degiskeninden yıllık kategorik degisken olusturma
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"


# Kontratı 1 veya 2 yıllık musterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kisiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sozlesmesi bulunan ve genc olan musteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# Kisinin toplam aldıgı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)


# Herhangi bir streaming hizmeti alan kisiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kisi otomatik odeme yapıyor mu?
df["PaymentMethod"].unique()
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# ortalama aylık odeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] +0.1)

# Guncel Fiyatın ortalama fiyata gore artısı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / (df["MonthlyCharges"] + 1)

# Servis basına ucret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

df.head()

df.shape



# ENCODING;

# Degiskenlerin tiplerine gore ayrılması islemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding İslemi
# cat_cols listesinin guncelleme islemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape



# MODELLEME;

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Nihai Model Sonuclari
# Accuracy: 0.79
# Recall: 0.66
# Precision: 0.51
# F1: 0.57
# Auc: 0.74


# Base Model
# # Accuracy: 0.7837
# # Recall: 0.6333
# # Precision: 0.4843
# # F1: 0.5489
# # Auc: 0.7282


def plot_feature_importance(importance,names,model_type):
    # Feature importance ve feature adlarindan olusan array olusturursak;
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Sozluk kullanarak dataframe olusturalim;
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Feature importance azalacak sekilde siralarsak;
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Bar plot'un boyutunu tanimlarsak;
    plt.figure(figsize=(25, 10))

    # Plot Searborn bar chart;
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Grafige labels eklersek;
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()


plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST')
