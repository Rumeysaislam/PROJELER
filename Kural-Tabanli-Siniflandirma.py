# PROJE: Kural Tabanli Siniflandirma ile Potansiyel Musteri Getirisi Hesaplama
#############################################

# İs Problemi
#############################################
# Bir oyun sirketi musterilerinin bazi ozelliklerini kullanarak seviye tabanli (level based) yeni musteri tanimlari (persona)
# olusturmak ve bu yeni musteri tanimlarina gore segmentler olusturup bu segmentlere gore yeni gelebilecek musterilerin sirkete
# ortalama ne kadar kazandirabilecegini tahmin etmek istemektedir.

# Orn.: Turkiye’den IOS kullanicisi olan 25 yasindaki bir erkek kullanicinin ortalama ne kadar kazandirabilecegi belirlenmek isteniyor.

# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararasi bir oyun sirketinin sattigi urunlerin fiyatlarini ve bu urunleri satin alan kullanicilarin bazi
# demografik bilgilerini barindirmaktadir. Veri seti her satis isleminde olusan kayitlardan meydana gelmektedir. Bunun anlami tablo
# tekillestirilmemistir. Diger bir ifade ile belirli demografik ozelliklere sahip bir kullanici birden fazla alisveris yapmis olabilir.


################# Uygulama Oncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrasi #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


import pandas as pd

print("#################  GOREV 1  #################\n")
## SORU 1: """Dosyayi okut ve bazi bilgileri gsster"""
df = pd.read_csv("/home/rumeysa/Desktop/Miuul_summercamp/2.Hafta/2.hafta-odevler/Kural-tabanli-siniflandirma/persona.csv")
print(f"persona.csv veri seti:\n\n{df.head(3)}\n\n{df.tail(3)}")


## SORU 2: """Benzersiz "SOURCE" degerleri ve frekanslar"""
print("\n\nBenzersiz degerler: ", df["SOURCE"].unique())
print("Benzersiz degerlerin sayisi: ", df["SOURCE"].unique().size)
print("Benzersiz degerlerin sayisi: ", df["SOURCE"].nunique())                   # df["SOURCE"].unique().size = df["SOURCE"].nunique()
print(f"Frekanslar:\n{df['SOURCE'].value_counts()}")

# print("Frekanslar: \n", file["SOURCE"].value_counts())                         # "android" onunde bosluk oluyor; sevmedim.
# print("{}\n{}".format("Frekanslar", file["SOURCE"].value_counts()))   ; bu sekilde de yapilabilirdi.


## SORU 3: """Benzersiz "PRICE" degerleri"""
print("\n\nBenzersiz degerler: ", df["PRICE"].unique())
print("Benzersiz degerlerin sayisi: ", df["PRICE"].nunique())


## SORU 4: """Benzersiz "PRICE" degerlerinden kac tane gerceklesmis?"""
print(f"\n\nFrekanslar:\n{file['PRICE'].value_counts()}")


## SORU 5: """Hangi ulkede kac satis olmus?"""
print(f"\n\nSayilar:\n{file['COUNTRY'].value_counts()}")
print(f"Turkiye'den kac satıs olmus:\n{file['COUNTRY'].value_counts()['tur']}")

# ya da;
df["COUNTRY"].value_counts()                                                     # Hangi ulkede kac satis oldugu
df.groupby("COUNTRY")["PRICE"].count()                                           # Ulkelere gore harcama sayisi

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")


## SORU 6: """Ulkelere gore toplam kazanclar"""
country_groupped_prices = file.groupby("COUNTRY")["PRICE"]
print(f"\n\nUlkelere gore toplam kazanclar:\n{country_groupped_prices.sum()}")

# ya da;
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# ya da;
df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="sum")


## SORU 7: """Kaynaklara gore satis sayilari"""
print(f"\n\nKaynaklara gore satıs sayiları:\n{file['SOURCE'].value_counts()}")


## SORU 8: """Ulkelere gore ortalama kazanclar"""
print(f"\n\nUlkelere gore ortalama kazanclar:\n{country_groupped_prices.mean()}")

# ya da;
df.groupby(by=['COUNTRY']).agg({"PRICE": "mean"})


## SORU 9: """Kaynaklara gore ortalama kazanclar"""
source_groupped_prices = file.groupby("SOURCE")["PRICE"]
print(f"\n\nKaynaklara gore ortalama kazanclar:\n{source_groupped_prices.mean()}")

# ya da;
df.groupby(by=['SOURCE']).agg({"PRICE": "mean"})

## SORU 10: """Ulke-Kaynak kirilimina gore ortalama kazanclar"""
source_groupped_prices = file.groupby(["SOURCE", "COUNTRY"])["PRICE"]
print(f"\n\nUlke-Kaynak kırılımına gore ortalama kazanclar:\n{source_groupped_prices.mean()}")

# ya da;
df.groupby(by=["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})



## GOREV 2: """COUNTRY, SOURCE, SEX, AGE kiriliminda ortalama kazanclar"""
print("\n#################  GOREV 2  #################\n")
print(f'COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazanclar:\n\n{file.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean()}')

# ya da;
df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head()



## GOREV 3: """Ciktiyi PRICE’a gore siralama"""
print("\n#################  GOREV 3  #################\n")
agg_df = file.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).mean().sort_values("PRICE", ascending=False)
print(f"Ciktiyi PRICE’a gore siralama:\n{agg_df.head()}")

# ya da;
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()

## GOREV 4: """Indekste yer alan isimleri degisken ismine cevirme"""
print("\n#################  GOREV 4  #################\n")
agg_df = agg_df.reset_index()
print(f"Indekste yer alan isimleri degisken ismine cevirme:\n\n{agg_df.head()}")


## GOREV 5: """Age degiskenini kategorik degiskene cevirme ve agg_df’e ekleme"""
# Araliklari ikna edici olacagini dusundugunuz sekilde olusturunuz.
# Ornegin: '0_18', '19_23', '24_30', '31_40', '41_70'
print("\n#################  GOREV 5  #################\n")
# AGE degiskeninin nerelerden bolunecegini belirtelim:
age_bins = [0, 18, 23, 30, 40, 70]

# Bolunen noktalara karsilik isimlendirmelerin ne olacagini ifade edelim:
age_categories = ['0_18', '19_23', '24_30', '31_40', '41_70']

# ya da;
age_categories = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

# age'i bolelim:
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], age_bins, labels=age_categories)
print(f"Age degiskenini kategorik degiskene cevirme:\n\n{agg_df.head()}")



## GOREV 6: """Yeni seviye tabanli musterileri (persona) tanimlama ve veri setine ekleme"""
print("\n#################  GOREV 6  #################\n")
# YONTEM 2
agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].agg(lambda x: '_'.join(x).upper(), axis=1)


# YONTEM 3
agg_df["customers_level_based"] = ['_'.join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]


# YONTEM 1
# degisken isimleri:
agg_df.columns

# Gozlem degerlerine nasıl erisiriz?
for row in agg_df.values:
    print(row)

# Amacımıza bir adim daha yaklastik.
# Burada ufak bir problem var. Bircok ayni segment olacak.
# Ornegin USA_ANDROID_MALE_0_18 segmentinden bircok sayida olabilir.
# Kontrol edelim:
agg_df["customers_level_based"].value_counts()

# Bu sebeple segmentlere gore groupby yaptiktan sonra price ortalamalarini almali ve segmentleri tekillestirmeliyiz;
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})

# customers_level_based index'te yer almaktadir. Bunu degiskene cevirelim;
agg_df = agg_df.reset_index()
agg_df.head()

# kontrol edelim; her bir persona'nin 1 tane olmasini bekleriz;
agg_df["customers_level_based"].value_counts()
agg_df.head()

###################################################################################################
# ya da;
# customers_level_categories = ["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]
# agg_df["customers_level_based"] = agg_df[customers_level_categories[0]].str.upper()
# for cat in customers_level_categories[1:]:
#    agg_df["customers_level_based"] += "_" + agg_df[cat].str.upper()

# persona_groupped_agg_price_mean = agg_df.groupby(["customers_level_based"])["PRICE"].mean()
# print(f"Yeni seviye tabanli musterileri (persona) tanımlama:\n{persona_groupped_agg_price_mean.head()}")
##################################################################################################

## GOREV 7: """Yeni musterileri (personalari) segmentlere ayirma"""
print("\n#################  GOREV 7  #################\n")
agg_df['SEGMENT'] = pd.qcut(agg_df['PRICE'], 4, labels=["D", "C", "B", "A"])
print(f"Yeni musterileri (personaları) segmentlere ayirma:\n{agg_df.groupby('SEGMENT').agg({'PRICE': ['mean', 'max', 'sum']})}")



## GOREV 8: """Yeni gelen musterileri siniflandirip, ne kadar gelir getirebileceklerini tahmin etme"""
print("\n#################  GOREV 8  #################\n")
TUR_ANDROID_FEMALE_31_40 = agg_df[agg_df['customers_level_based'] == 'TUR_ANDROID_FEMALE_31_40']
FRA_IOS_FEMALE_31_40 = agg_df[agg_df['customers_level_based'] == 'FRA_IOS_FEMALE_31_40']
print("31-40 yas arası Turk kadini Android: \n", "Ortalama Kazanc: ", TUR_ANDROID_FEMALE_31_40["PRICE"].mean().__round__(2), "Segment: ", TUR_ANDROID_FEMALE_31_40["SEGMENT"].unique())
print("31-40 yas arası Fransiz kadini iOS: \n", "Ortalama Kazanc: ", FRA_IOS_FEMALE_31_40["PRICE"].mean().__round__(2), "Segment: ", FRA_IOS_FEMALE_31_40["SEGMENT"].unique())
