#Gelişmiş Fonksiyonel Veri Analizi (Advanced Functional Eda)

#Genel resim
#1. Kategorik değişken analizi
#2. Sayısal değişken analizi
#3. Hedef Değişken Analizi
#4. Korelasyon Analizi

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None) #gösterilecek olan maksimum kolon sayısı olmasın yani her kolonu göster diyoruz.
pd.set_option('display.width', 500)
df=sns.load_dataset("titanic")
df.head()

df.tail()

df.info()
#class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype
#---  ------       --------------  -----
# 0   survived     891 non-null    int64
# 1   pclass       891 non-null    int64
# 2   sex          891 non-null    object
# 3   age          714 non-null    float64
# 4   sibsp        891 non-null    int64
# 5   parch        891 non-null    int64
# 6   fare         891 non-null    float64
# 7   embarked     889 non-null    object
# 8   class        891 non-null    category
# 9   who          891 non-null    object
# 10  adult_male   891 non-null    bool
# 11  deck         203 non-null    category
# 12  embark_town  889 non-null    object
# 13  alive        891 non-null    object
# 14  alone        891 non-null    bool
#dtypes: bool(2), category(2), float64(2), int64(4), object(5)
#memory usage: 80.7+ KB


df.shape
#(891, 15)


df.columns
#Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'], dtype='object')

df.index
# RangeIndex(start=0, stop=891, step=1)

df.describe().T
#         count       mean        std   min      25%      50%   75%       max
#survived  891.0   0.383838   0.486592  0.00   0.0000   0.0000   1.0    1.0000
#pclass    891.0   2.308642   0.836071  1.00   2.0000   3.0000   3.0    3.0000
#age       714.0  29.699118  14.526497  0.42  20.1250  28.0000  38.0   80.0000
#sibsp     891.0   0.523008   1.102743  0.00   0.0000   0.0000   1.0    8.0000
#parch     891.0   0.381594   0.806057  0.00   0.0000   0.0000   0.0    6.0000
#fare      891.0  32.204208  49.693429  0.00   7.9104  14.4542  31.0  512.3292


df.isnull().values.any()
#True


df.isnull().sum()
#survived         0
#pclass           0
#sex              0
#age            177
#sibsp            0
#parch            0
#fare             0
#embarked         2
#class            0
#who              0
#adult_male       0
#deck           688
3embark_town      2
#alive            0
#alone            0
#dtype: int64

def check_df(dataframe, head=5):
    print("##############Shape#############")
    print(dataframe.shape)
    print("##############Types#############")
    print(dataframe.dtypes)
    print("##############Head#############")
    print(dataframe.head(head))
    print("##############Tail#############")
    print(dataframe.tail(head))
    print("##############NA#############")
    print(dataframe.isnull().sum())
    #Sayısal değişkenlerin dağılım bilgisi, QUANTILES
    print("##############QUANTILES#############")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)


check_df(df)

df=sns.load_dataset("flights")
check_df(df)

#1. Kategorik değişken analizi
#Programatik olarak bazı tanımlamalar , kategorik değişkeni analiz etmek
#tek değişkeni analiz etmek ayrı çok değişkeni analiz etmek ayrı

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df=sns.load_dataset("titanic")
df.head()


df["embarked"].value_counts() #sınıf sayılarına erişmek
#S    644
#C    168
#Q     77
#Name: embarked, dtype: int64

df["sex"].unique() #bir başka değişkenin unique değerlerine erişmek istersek
#array(['male', 'female'], dtype=object)

df["sex"].nunique() #toplamda kaç tane eşsiz değer var
#2

#Öyle bir şey yapmalıyız ki bu veri seti içerisinden otomatik bir şekilde bana olası bütün kategorik değişkenleri seçsin
#bunu birkaç aşamada yapıcaz
#öncelikle tip bilgisine göre seçicez
#daha sonra tip konusunda bizi atlatmış, baka tipte gözüken ama aslında kategorik (istediğimiz tip yani) olan sinsirellaları yakalayacağız.

#1. tip bilgisine göre bundan hareketle:  bool , category, ve object

[col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"] ]

#tip bilgisini kontrol sağladık
#['sex',
# 'embarked',
# 'class',
# 'who',
# 'adult_male',
# 'deck',
# 'embark_town',
# 'alive',
# 'alone']

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"] ]
cat_cols
#['sex',
# 'embarked',
# 'class',
# 'who',
# 'adult_male',
# 'deck',
# 'embark_town',
# 'alive',
# 'alone']

num_but_cat=[col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
num_but_cat
#['survived', 'pclass', 'sibsp', 'parch']

cat_but_car= [col for col in df.columns if df[col].nunique()>20 and str(df[col].dtypes) in ["category", "object"]]
cat_but_car
#[]

cat_cols=cat_cols+num_but_cat
cat_cols=[col for col in cat_cols if col not in cat_but_car]
#üstteki iki satır da önemli

df[cat_cols].nunique()
#sex            2
#embarked       3
#class          3
#who            3
#adult_male     2
#deck           7
#embark_town    3
#alive          2
#alone          2
#dtype: int64


[col for col in df.columns if col not in cat_cols]
#['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']

df["survived"].value_counts()
#0    549
#1    342
#Name: survived, dtype: int64

100*df["survived"].value_counts() / len(df)
#0    61.616162
#1    38.383838
#Name: survived, dtype: float64

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df,col)

#Veri setindeki bütün kategorik değişkenleri yazdırdı.

#KATEGORİK DEĞİŞKEN ANALİZİ PART 2

#BASİTLİK, GENELLENEBİLİRLİK ÖNEMLİ, sürdürülebilirlik kavramı önemli
#kategorik değişken sütun grafiği countplot fonksiyonu
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex")
#        sex      Ratio
#male    577  64.758698
#female  314  35.241302
##########################################


cat_summary(df, "sex", plot=True)
#Çıktıda grafiğin oluştuğunu görürüz


for col in cat_cols:
    cat_summary(df,col, plot=True)

#Her şey yolundayken bir hata aldık, kontrollü bir senaryo, genellenebilirlik , ölçeklenebilirlik kavramını test etmek için: adult_male bool tipli olduğu için görselleştirememiştir.



for col in cat_cols:
    if df[col].dtypes == "bool":
        print("###################sadgsdgs############") #countplot methodunu kullanırken exception olanları görüntülemek için
    else:
    cat_summary(df,col, plot=True)

df["adult_male"].head()
#0     True
#1    False
#2    False
#3    False
#4     True
#Name: adult_male, dtype: bool

df["adult_male"].astype(int).head()
#0    1
#1    0
#2    0
#3    0
#4    1
#Name: adult_male, dtype: int32

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col]=df[col].astype(int)
        cat_summary(df,col,plot=True) #adult_male'de başarılı bir şekilde geldi
    else:
    cat_summary(df,col, plot=True)


#Büyük ölçekli işlerde daha az ve kullanılabilir özellik takip edilebilmeli.
#özellik arttıkça ve eklemeler arttıkça bu beraberinde başka riskleri getiriyor.






def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
             sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)
cat_summary(df, "adult_male", plot=True) #Arızalı değişken


#2. SAYISAL DEĞİŞKEN ANALİZİ

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df=sns.load_dataset("titanic")
df.head()

df[["age", "fare"]].describe().T
#      count       mean        std   min      25%      50%   75%       max
#age   714.0  29.699118  14.526497  0.42  20.1250  28.0000  38.0   80.0000
#fare  891.0  32.204208  49.693429  0.00   7.9104  14.4542  31.0  512.3292

#yaş ve fare değişkenleri sayısal değişken olduğunu biliyorum
#programatik olarak nümerik değişkenleri nasıl seçerim?


cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"] ]
cat_cols

num_but_cat=[col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
num_but_cat

cat_but_car= [col for col in df.columns if df[col].nunique()>20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols=cat_cols+num_but_cat
cat_cols=[col for col in cat_cols if col not in cat_but_car]

num_cols=[col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
#['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
#sayısalları seçtik ama bazı değişkenler sayısal görünüm olsa da sayısal değişken değildir
num_cols=[col for col in num_cols if col not in cat_cols]


cat_cols
#['sex',
# 'embarked',
# 'class',
# 'who',
# 'deck',
# 'embark_town',
# 'alive',
# 'survived',
# 'pclass',
# 'sibsp',
# 'parch']
num_cols
#['age', 'fare']

#ölçeklenebilirlik kavramının asıl zorluğu veri yapılarını seçebilmek, programatik olarak genellenebilirlik kaygılarıyla seçebilmek

def num_summary(dataframe, numerical_col):
    quantiles=[0.05, 0.10,0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")
#count    714.000000
#mean      29.699118
#std       14.526497
#min        0.420000
#5%         4.000000
#10%       14.000000
#20%       19.000000
#30%       22.000000
#40%       25.000000
#50%       28.000000
#60%       31.800000
#70%       36.000000
#80%       41.000000
#90%       50.000000
#95%       56.000000
#99%       65.870000
#max       80.000000
#Name: age, dtype: float64

for col in num_cols:
    num_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles=[0.05, 0.10,0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age", plot=True) #Görsel geldi çıktıda

for col in num_cols:
    num_summary(df, col, plot=True)

#3.DEĞİŞKENLERİN YAKALANMASI VE İŞLEMLERİN GENELLEŞTİRİLMESİ

#BU BÖLÜMDE BİR FONKSİYON TANIMLIYCAZ VE AYNI ZAMANDA DOCSTRİNG KAVRAMINA DOKUNUP BİR FONKSİYON DİĞER KULLANICILAR TARAFINDAN KULLANILIRKEN HANGİ NOKTALARA DOKUNULMASI GERREKTİĞİNDEN BAHSEDECEĞİZ.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df=sns.load_dataset("titanic")
df.head()
df.info()
#<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype
#---  ------       --------------  -----
# 0   survived     891 non-null    int64
# 1   pclass       891 non-null    int64
# 2   sex          891 non-null    object
# 3   age          714 non-null    float64
# 4   sibsp        891 non-null    int64
# 5   parch        891 non-null    int64
# 6   fare         891 non-null    float64
# 7   embarked     889 non-null    object
# 8   class        891 non-null    category
# 9   who          891 non-null    object
# 10  adult_male   891 non-null    bool
# 11  deck         203 non-null    category
# 12  embark_town  889 non-null    object
# 13  alive        891 non-null    object
# 14  alone        891 non-null    bool
#dtypes: bool(2), category(2), float64(2), int64(4), object(5)
#memory usage: 80.7+ KB

#Konumuz DOCSTRING: bir fonksiyona argüman yazma konusu
#""" üç tırnak girip aşağıya enter dediğimizde ön tanımlı olarak bize bir docstring oluşturma yöntemi önerdi.

def grab_col_names(dataframe, cat_th=10, car_th=30):
   """

   Parameters
   ----------
   dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
   cat_th: int, float
         nümerik veya kategorik olan değişkenler için sınıf eşik değeridir.
   car_th: int, float
         kategorik veya kardinal değişkenler için sınıf eşik değeridir.

   Returns
   -------
   cat cols: List
        Kategorik değişken listesi
   num_cols: list
        Nümerik değişken listesi
   cat_but_car:list
        Kategorik görünümlü kardinal değişken listesi

   Notes:
   cat_cols + num_cols + cat_but_car =toplam değişken sayısı
       num_but_cat cat_cols'un içerisindedir
    ----
   """
#   """

#   Parameters
#   ----------
#   dataframe: dataframe
#        değişken isimleri alınmak istenen dataframe'dir.
#   cat_th: int, float
#         nümerik veya kategorik olan değişkenler için sınıf eşik değeridir.
#   car_th: int, float
#         kategorik veya kardinal değişkenler için sınıf eşik değeridir.
#
#   Returns
#   -------
#   cat cols: List
#        Kategorik değişken listesi
##   num_cols: list
#        Nümerik değişken listesi
#   cat_but_car:list
#        Kategorik görünümlü kardinal değişken listesi

#   Notes:
#   cat_cols + num_cols + cat_but_car =toplam değişken sayısı
#       num_but_cat cat_cols'un içerisindedir
#    ----
#   """

#help(grab_col_names): açıklamaları getiren kod.
   cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
   num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
   cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
   cat_cols = cat_cols + num_but_cat
   cat_cols = [col for col in cat_cols if col not in cat_but_car]
   num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
   # ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
   # sayısalları seçtik ama bazı değişkenler sayısal görünüm olsa da sayısal değişken değildir
   num_cols = [col for col in num_cols if col not in cat_cols]
   print(f"observations: {dataframe.shape[0]}")
   print(f"variables: {dataframe.shape[1]}")
   print(f"cat_cols: {len(cat_cols)}")
   print(f"num_cols: {len(num_cols)}")
   print(f"cat_but_car: {len(cat_but_car)}")
   print(f"num_but_cat: {len(num_but_cat)}")

   return cat_cols, num_cols, cat_but_car



grab_col_names(df)
#(['sex',
#  'embarked',
#  'class',
#  'who',
#  'adult_male',
#  'deck',
#  'embark_town',
#  'alive',
#  'alone',
#  'survived',
#  'pclass',
#  'sibsp',
#  'parch'],
# ['age', 'fare'],
# [])

cat_cols, num_cols, cat_but_car=grab_col_names(df)
#observations: 891
#variables: 15
#cat_cols: 13
#num_cols: 2
#cat_but_car: 0
#num_but_cat: 4


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


cat_summary(df, "sex")
#        sex      Ratio
#male    577  64.758698
#female  314  35.241302
##########################################

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles=[0.05, 0.10,0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


#cat_summary fonk.nu da plot özelliğiyle biçimlendirilecek şekilde rahatça bir ele alalım:

#BONUS

df= sns.load_dataset("titanic")
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col]=df[col].astype(int)

cat_cols,num_cols,cat_but_car=grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)




for col in cat_cols:
    cat_summary(df, col, plot=True)


#3. HEDEF DEĞİŞKEN ANALİZİ
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df=sns.load_dataset("titanic")


for col in df.columns:
    if df[col].dtypes == "bool":
        df[col]=df[col].astype(int)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
def grab_col_names(dataframe, cat_th=10, car_th=20):
   """

   Parameters
   ----------
   dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
   cat_th: int, float
         nümerik veya kategorik olan değişkenler için sınıf eşik değeridir.
   car_th: int, float
         kategorik veya kardinal değişkenler için sınıf eşik değeridir.

   Returns
   -------
   cat cols: List
        Kategorik değişken listesi
   num_cols: list
        Nümerik değişken listesi
   cat_but_car:list
        Kategorik görünümlü kardinal değişken listesi

   Notes:
   cat_cols + num_cols + cat_but_car =toplam değişken sayısı
       num_but_cat cat_cols'un içerisindedir
    ----
   """
#   """

#   Parameters
#   ----------
#   dataframe: dataframe
#        değişken isimleri alınmak istenen dataframe'dir.
#   cat_th: int, float
#         nümerik veya kategorik olan değişkenler için sınıf eşik değeridir.
#   car_th: int, float
#         kategorik veya kardinal değişkenler için sınıf eşik değeridir.
#
#   Returns
#   -------
#   cat cols: List
#        Kategorik değişken listesi
##   num_cols: list
#        Nümerik değişken listesi
#   cat_but_car:list
#        Kategorik görünümlü kardinal değişken listesi

#   Notes:
#   cat_cols + num_cols + cat_but_car =toplam değişken sayısı
#       num_but_cat cat_cols'un içerisindedir
#    ----
#   """

#help(grab_col_names): açıklamaları getiren kod.
   cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
   num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
   cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
   cat_cols = cat_cols + num_but_cat
   cat_cols = [col for col in cat_cols if col not in cat_but_car]
   num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
   # ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
   # sayısalları seçtik ama bazı değişkenler sayısal görünüm olsa da sayısal değişken değildir
   num_cols = [col for col in num_cols if col not in cat_cols]
   print(f"observations: {dataframe.shape[0]}")
   print(f"variables: {dataframe.shape[1]}")
   print(f"cat_cols: {len(cat_cols)}")
   print(f"num_cols: {len(num_cols)}")
   print(f"cat_but_car: {len(cat_but_car)}")
   print(f"num_but_cat: {len(num_but_cat)}")

   return cat_cols, num_cols, cat_but_car

cat_cols,num_cols,cat_but_car=grab_col_names(df)
df.head()

df["survived"].value_counts()
#0    549
#1    342
#Name: survived, dtype: int64

cat_summary(df, "survived")

#insanların hayatta kalma durumunu etkileyen şey nedir
#bunu anlamanın yolu değişkenleri tek başına incelemek değildir, değişkenleri çaprazlamamız lazım
#yani bağımlı değişkenlere göre diğer değişkenleri göz önünde bulundurarak analizler yapmamız lazım

#öncelikle survived değişkeni ile kategorik değişkeni nasıl çarpıştırırız, nasıl çaprazlarız ve survived durumunun nasıl ortaya çıktığını analiz ederiz
#1-kategorik değişkene göre veriyi group by'a alsam
df.groupby("sex")["survived"].mean()
#sex
#female    0.742038
#male      0.188908
#Name: survived, dtype: float64
#Kadın olmak hayatta kalma faktörünü etkileyen bir değer olabilir.















