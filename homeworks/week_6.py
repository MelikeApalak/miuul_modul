import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro,levene,ttest_ind,mannwhitneyu, \
    pearsonr,spearmanr,kendalltau,f_oneway,kruskal
from statsmodels.stats.proportion import proportions_ztest


#Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri
#ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.
#Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri
#ab_testing.xlsxexcel’ininayrı sayfalarında yer almaktadır.
#Kontrol grubuna Maximum Bidding, test grubuna AverageBidding uygulanmıştır
#4 değişken, 40 gözlem, 26 KB

#Impression -> reklam görüntülenme sayısı
#Click -> görüntülenen reklama tıklama sayısı
#Purchase -> tıklanan reklamlar sonrası satın alınan ürün sayısı
#Earning -> satın alınan ürünler sonrası elde edilen kazanç

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',10)
pd.set_option('display.float_format',lambda  x: '%.5f' % x)

#####
#Veriyi Hazırlama ve Analiz Etme

#Adım 1:  ab_testing_data.xlsxadlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz.
#Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
sheet_names= ['Control Group','Test Group']
df_control = pd.read_excel('homeworks/ab_testing.xlsx',sheet_name=sheet_names[1])
df_test = pd.read_excel('homeworks/ab_testing.xlsx',sheet_name=sheet_names[1])


#Adım 2: Kontrol ve test grubu verilerini analiz ediniz
df_control.describe().T
df_test.describe().T

df_control["group_type"] = "control"
df_test["group_type"] = "test"

#Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz
frames=[df_control,df_test]
df_concat = pd.concat(frames)
df_concat.describe().T
df_concat.info
####
#Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#istatistsel testler için Purchase metriğine odaklanılmalıdır.
#"maximumbidding" adı verilen teklif verme türüne alternatif olarak
#yeni bir teklif türü olan "averagebidding"
#hipotez ->teklif verme türü değiştikten sonra kullanıcıların satın alınan
#ürün sayısında artış miktarında ist. olarak anlamlı bir fark var mı ?
#click, purchase
#H0 -> ... anlamlı bir farklılık yoktur.
#H1 -> ... vardır.


#varsayım kontrolü
#normallik varsayımı
#varyans homojenliği

#shapiro testi bir değişkenin dağılımının normal olup olmadığını test eder.
test_stat, pvalue = shapiro(df_concat.loc[df_concat["group_type"] == "control","Purchase"])
print('Test Stat= %.4f, p-value = %.4f' % (test_stat,pvalue))
#p-value > 0.05. H0 reddedilemez. Normallik varsayımı sağlanmaktadır.

test_stat, pvalue = shapiro(df_concat.loc[df_concat["group_type"] == "test","Purchase"])
print('Test Stat= %.4f, p-value = %.4f' % (test_stat,pvalue))
#p-value > 0.05. H0 reddedilemez. Normallik varsayımı sağlanmaktadır.
#normallik varsayımı sağlanıyor. varyans homojenliğine bakılır.

test_stat, pvalue = levene(df_concat.loc[df_concat["group_type"] == "control","Purchase"],
                           df_concat.loc[df_concat["group_type"] == "test","Purchase"])
print('Test Stat= %.4f, p-value=%.4f' % (test_stat,pvalue))
#p-value>0.05. H0 reddedilemez. varyans homojenliği sağlandı.

#varsayımlar sağlandığı için parametrik test yapılır.
# t-testing
test_stat, pvalue = ttest_ind(df_concat.loc[df_concat["group_type"] == "control","Purchase"],
                           df_concat.loc[df_concat["group_type"] == "test", "Purchase"],
                              equal_var=True)

print('Test Stat= %.4f, p-value= %.4f' % (test_stat,pvalue))

#p-value > 0.05. H0 reddedilemez. İstatistiksel olarak anlamlı bir farklılık yoktur.