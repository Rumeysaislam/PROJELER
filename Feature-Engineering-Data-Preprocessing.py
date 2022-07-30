# Titanic Veri Seti icin; Uçtan Uca Feature Engineering & Data Preprocessing

df = load()         # Titanic veri setini cektik.
df.shape            # 891 tane gozlem birimi ve 12 tane degiskenden olusmaktadir.
df.head()

# Butun degiskenlerin isimlerini tek bir formata (hepsi buyuk harf) getirirsek;
df.columns = [col.upper() for col in df.columns]    # df'in sutunlarinda gez, yakaladigin ismi buyult.



# 1. Feature Engineering (Degisken Muhendisligi)

# Urettigimiz yeni degiskenler;

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')

# Name count, isim saydirma
df["NEW_NAME_COUNT"] = df["NAME"].str.len()

# name word count; kelime saydirma
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

# name dr; İsminde "Dr" olanlar;
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

# name title; text sekli
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)

# family size; ailedeki kisi sayisi;
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

# age_pclass ; age * Pclass ile olusturulan degisken;
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

# is alone; Yalniz mi, degil mi degiskeni;
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

# age level; araliklara gore yeni yas degiskenimiz;
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

# sex x age; cinsiyet ve yasa gore olusturulan yeni degisken;
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape                                           # Baslangicta 12 olan degisken sayimiz 22 oldu.

# Kategorik ve sayisal degiskenlerimi anlayabilmek icin "grab_col_names" fonksiyonumu cagiriyorum.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# "PassegerId" numaerik bir degisken degil, bunu disarida birakmak icin;
num_cols = [col for col in num_cols if "PASSENGERID" not in col]




# 2. Outliers (Aykiri Degerler)

# "check_outlier" ile aykiri degerleri sorguluyoruz;

for col in num_cols:
    print(col, check_outlier(df, col))

# Esik degerlerle bu aykiri degerleri degistirmek istersek;
for col in num_cols:
    replace_with_thresholds(df, col)

# Tekrar aykiri degerlere bakarsak;
for col in num_cols:
    print(col, check_outlier(df, col))       # Aykiri degerlerden kurtulmusuz. :)


    

# 3. Missing Values (Eksik Degerler)

# Daha once yazdigimiz "missing_values_table" fonksiyonumuzu cagiriyoruz;
missing_values_table(df)

# "Cabin" yerine "cabin_bool" adinda yeni bir degisken olusturdugumuz icin "Cabiné degiskenini sileriz;
df.drop("CABIN", inplace=True, axis=1)          # inplace=True ile silme islemini kalici hale getirdik.

# Istemedigimiz diger degiskenleri de kaldiralim (Name uzerinden yeni degiskenler olusturmustuk zaten);
remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

# Yas degiskeninde eksiklikler old. icin yasa bagli olusan degiskenlerde de eksiklik cikti !
# Olusturdugumuz "NEW_TITLE"a gore groupby'a alip, yas degiskenin eksik degerlerini medyan ile dolduralim;
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
# Yas degiskenindeki eksiklikler gitti.

# Peki yas degiskeni uzerinden olustrulan degiskenlerdeki eksiklikler ne olacak?
# Yasa bagli degiskenleri tekrar olusturuyoruz;
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

missing_values_table(df)                                # Sadece "embarked" kaldi.

# programatik sekilde "embarked"dan kurtulmak istersek;
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
# Tipi object olan ve essiz deger sayisi 10'dan kucuk esit olan kategorik degiskenleri modlari ile doldurduk. Bir tane old. dolayi sadece embarked degiskeni doldu. :)

missing_values_table(df)                                # Eksik degerlerden kurtulmus olduk. :)

# Kullandigimiz yontemlere gore ( Orn. agac yontemine gore) eksik degerlerden kurtulmamayi tercih edebilirdik. :)




# 4. Label Encoding

# Iki sinifli kategorik degiskenleri donusturuyoruz;
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]              # Essiz 2 sinifa sahip olan kategorik degiskenleri secmis olduk.
# "SEX" ile yeni olusturdugumuz; "NEW_IS_ALONE" (Yes-No ifadelerine sahip) ifadeleri geldi.

for col in binary_cols:
    df = label_encoder(df, col)




# 5. Rare Encoding

# One-hot encoding yapmadan once olasi indirgemeleri yapiyoruz;

# "rare_analyser" fonksiyonumu getiriyorum;
rare_analyser(df, "SURVIVED", cat_cols)
# Gozlenme oranlari az olan siniflar birlesme durumlari ifade etmis oldu.

df = rare_encoder(df, 0.01)                     # Orani 0,01 altinda olanlari birlestirdik.

df["NEW_TITLE"].value_counts()                  # Bircok sinif "Rare" olarak bir araya getirilmis.




# 6. One-Hot Encoding

# Butun kategorik degiskenleri cevirecegiz;
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
# Zaten essiz deger sayisi 2 olan degiskenleri donusturmustuk, simdi 2 den buyuk 10'dan kucuk esit olanlari donusturecegim;
df = one_hot_encoder(df, ohe_cols)

df.head()                                                                                  # Butun olasi kategoriler degiskenlere donustu.
df.shape # 12 tane iken 50 tane degiskenimiz oldu. one_hot_encoder'in ilk siniflarini drop ettik (drop_first =True, on tanimli degerimiz).


# Yeni olusturdugum degiskenlerde ornegin 1 ve 0 siniflarinin dagilimi cok dusuk olanlar var mı?
# Olusturdugum degiskenler bilgi tasiyor mu? (Gerekli mi, gereksiz mi?)
# "rare_analyser" fonksiyonumu tekrar cagiriyorum;

cat_cols, num_cols, cat_but_car = grab_col_names(df)                            # Yeni degiskenlerim de eklenince yeni bir veri seti oldu.
# Degiskenlerime tekrar baktim.

num_cols = [col for col in num_cols if "PASSENGERID" not in col]                # Numerik degiskenlerden "PASSENGERID"i kaldirdik.

rare_analyser(df, "SURVIVED", cat_cols)                                         # cat_cols'lari rar_analyser'dan gecirdik.

# Yeni olsturdugum degiskenlerde frekanslar ve oranlar birbirine yakin olsun istiyorum;
# Orani dusuk; bilgi tasimayanlari analiz etmem gerekiyor.

# Essiz 2 sinifi olup, siniflarindan herhangi biri 0,01'den az olan var mi?;
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
# Kategorik degiskenlerin sinif sayilarini gozlem sayilarina bolduk,
# bunlardan herhangi birtanesinde 0,01'den kucuk olan 2 sinifli bir kategorik degisken varsa "useless_cols" diye bir degisken olusturduk.


# df.drop(useless_cols, axis=1, inplace=True) ; Bunlari silmeyi tercih edebilirsin.



# 7. Standart Scaler

# Bu problemde gerekli degil ama standartlastirmaya ihtiyacimiz olursa;
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])                                # Sayisal degiskenler donusturuldu.

df[num_cols].head()

df.head() # Son haline baktigimizda; kullanacagimiz herhangi bir makine ogrenmesi algoritmasi bu veri uzerinde calisabilir. :)
df.shape

# Veri on isleme ile ilgili islemlerimiz bitmistir. :))




# 8. Model

y = df["SURVIVED"]                                  # Bagimli degiskenim olan "SURVIVED"i sectim.
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)    # Bagimsiz degiskenleri ("PASSENGERID", "SURVIVED" disindaki degiskenler) de sectim.

# Veri setini; train ve test olarak ikiye ayiriyorum;
# train seti uzerinde model kuracagim, test seti ile kurdugum modeli test edecegim.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Agac tabanli yontem kullaniyoruz; sklearn.ensemble icerisinden RandomForestClassifier nesnesini getiriyorum;
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)            # Bir satirda modeli kurduk. X_train: bagimsiz degiskenler, y_train: bagimli degiskenler
y_pred = rf_model.predict(X_test)                                                   # Modeli kurup tahmin etme adimi; test setindeki x bagimsiz degisken degerlerini modele sorduk.
accuracy_score(y_pred, y_test)                                                      # Test setindeki y bagimli degiskeni ile modelin tahmin ettigi degerleri kiyasliyorum.

# %80 tahmin dogruluk oranina (accuracy_score) ulastik.





# Hic bir islem yapilmadan model kursaydik elde edilecek skor ne olurdu?


dff = load()                                                                        # Veri setini bastan okudum.
dff.dropna(inplace=True)                                                            # Eksik degerleri ucurdum. :)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)             # one-hot Encoding yaptim (Binary encoding'de gerceklesmis oldu.)
y = dff["Survived"]                                                                 # Bagimli degiskenleri sectim.
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)        # Bagimsiz degiskenleri sectim.
# Modeli tekrar kuruyorum;
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# accuracy_score = 0,70 (%70) cikti.
# Eksik degerden kurtulmasaydik hata alirdik. (Bazi agaca dayali modellerde drop etmeseydik cok rahat calisirdi)
# "get_dummies" yapmasaydik hata alirdik; algoritma string ifadeyi float'a ceviremezdi.




# Yeni ürettigimiz degiskenler anlamli mi, anlamsiz mi?
y = df["SURVIVED"]                                                                  # Bagimli degiskenim olan "SURVIVED"i sectim.
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)                                    # Bagimsiz degiskenleri ("PASSENGERID", "SURVIVED" disindaki degiskenler) de sectim.

# Veri setini; train ve test olarak ikiye ayiriyorum;
# train seti uzerinde model kuracagim, test seti ile kurdugum modeli test edecegim.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Agac tabanli yontem kullaniyoruz; sklearn.ensemble icerisinden RandomForestClassifier nesnesini getiriyorum;
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)            # Bir satirda modeli kurduk. X_train: bagimsiz degiskenler, y_train: bagimli degiskenler
y_pred = rf_model.predict(X_test)                                                   # Modeli kurup tahmin etme adimi; test setindeki x bagimsiz degisken degerlerini modele sorduk.
accuracy_score(y_pred, y_test)                                                      # Test setindeki y bagimli degiskeni ile modelin tahmin ettigi degerleri kiyasliyorum.


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)                                                  # Onem sirasina gore degiskenleri; anlamli ve anlamsiz olanlari gormus olduk. :)
