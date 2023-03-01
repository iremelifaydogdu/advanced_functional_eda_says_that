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
#bu işlemi bir fonksiyonla tanımlayacak olursak:

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), )

target_summary_with_cat(df,"survived", "pclass")
#        TARGET_MEAN
#pclass
#1          0.629630
#2          0.472826
#3          0.242363


for col in cat_cols:
    target_summary_with_cat(df, "survived", col)
#bütün kategorik değişkenlerle bağımlı değişkenimiz analize sokulmuş oldu
#        TARGET_MEAN
#sex
#female     0.742038
#male       0.188908
#          TARGET_MEAN
#embarked
#C            0.553571
#Q            0.389610
#S            0.336957
#        TARGET_MEAN
#class
#First      0.629630
#Second     0.472826
#Third      0.242363
#       TARGET_MEAN
#who
#child     0.590361
#man       0.163873
#woman     0.756458
#            TARGET_MEAN
#adult_male
#False          0.717514
#True           0.163873
#      TARGET_MEAN
#deck
#A        0.466667
#B        0.744681
#C        0.593220
#D        0.757576
#E        0.750000
#F        0.615385
#G        0.500000
#             TARGET_MEAN
#embark_town
#Cherbourg       0.553571
#Queenstown      0.389610
#Southampton     0.336957
#       TARGET_MEAN
#alive
#no             0.0
#yes            1.0
#       TARGET_MEAN
#alone
#False     0.505650
#True      0.303538
#          TARGET_MEAN
#survived
#0                 0.0
#1                 1.0
#        TARGET_MEAN
#pclass
#1          0.629630
#2          0.472826
#3          0.242363
#       TARGET_MEAN
#sibsp
#0         0.345395
#1         0.535885
#2         0.464286
#3         0.250000
#4         0.166667
#5         0.000000
#8         0.000000
#       TARGET_MEAN
#parch
#0         0.343658
#1         0.550847
#2         0.500000
#3         0.600000
#4         0.000000
#5         0.200000
#6         0.000000

#HEDEF DEĞİŞKENİN SAYISAL DEĞİŞKENLER İLE ANALİZİ

#group by'ı bu sefer target'a alıp nümerik değişkenlerin ort.sını alabiliriz. yani eksen değiştiriyoruz.


df.groupby("survived")["age"].mean()
#survived
#0    30.626179
#1    28.343690
#Name: age, dtype: float64


df.groupby("survived").agg({"age":"mean"})
#                age
#survived
#0         30.626179
#1         28.343690

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df,"survived","age")
#                age
#survived
#0         30.626179
#1         28.343690

for col in num_cols:
    target_summary_with_num(df, "survived", col)

#Hedef değişkeni kategorik değişkene göre ya da nümerik değişkene göre analiz eden fonksiyonlardır.

#4. Korelasyon Analizi

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df=pd.read_csv("HAFTA 2/breast_cancer.csv")
df=df.iloc[:,1:-1] #veri setinde problemli değişkenleri dışarıda bırakmak için böyle bir seçim yapılmış
df.head()

#Amacımız elimize veri seti geldiğinde bunun ısı haritası aracılığıyla korelasyonlarına bakmak ve daha sonra yüksek korelasyonlu bir değişken setinde yüksek korelasyonlu değişkenleri dışarıda bırakabilmeyi görmektir.
#Burada görmüş olduğunuz yüksek korelasyonlu değişkenlerden her birisini silmeyi illa ki her çalışmada yapılacaktır diye bir şey yoktur.
#Sadece ihtiyacımız olduğunda yüksek korelasyonlu değişkenleri nasıl yakalarız ciddi bir problemdir.
#Tek tek değişkenlerle korelasyona bakmak ve buna göre karar vermek değişken sayısı çok yüksek olduğunda mümkün olmadığından dolayı bir fonksiyon yazımı paylaşılacaktır.
#ihtiyacınız olduğunda sadece bir analiz aracı olarak kullanmalısınız.


#veri setindeki sayısal değişkenleri seçmemiz lazım.
#bize sadece nümerik değişkenleri seçen fonksiyon lazım

num_cols = [col for col in df. columns if df[col].dtype in [int, float]]


corr=df[num_cols].corr()
corr
#                         radius_mean  texture_mean  perimeter_mean  area_mean  smoothness_mean  compactness_mean  concavity_mean  concave points_mean  symmetry_mean  fractal_dimension_mean  radius_se  texture_se  perimeter_se   area_se  smoothness_se  compactness_se  concavity_se  concave points_se  symmetry_se  fractal_dimension_se  radius_worst  texture_worst  perimeter_worst  area_worst  smoothness_worst  compactness_worst  concavity_worst  concave points_worst  symmetry_worst  \
#radius_mean                 1.000000      0.323782        0.997855   0.987357         0.170581          0.506124        0.676764             0.822529       0.147741               -0.311631   0.679090   -0.097317      0.674172  0.735864      -0.222600        0.206000      0.194204           0.376169    -0.104321             -0.042641      0.969539       0.297008         0.965137    0.941082          0.119616           0.413463         0.526911              0.744214        0.163953
#texture_mean                0.323782      1.000000        0.329533   0.321086        -0.023389          0.236702        0.302418             0.293464       0.071401               -0.076437   0.275869    0.386358      0.281673  0.259845       0.006614        0.191975      0.143293           0.163851     0.009127              0.054458      0.352573       0.912045         0.358040    0.343546          0.077503           0.277830         0.301025              0.295316        0.105008
#perimeter_mean              0.997855      0.329533        1.000000   0.986507         0.207278          0.556936        0.716136             0.850977       0.183027               -0.261477   0.691765   -0.086761      0.693135  0.744983      -0.202694        0.250744      0.228082           0.407217    -0.081629             -0.005523      0.969476       0.303038         0.970387    0.941550          0.150549           0.455774         0.563879              0.771241        0.189115
#area_mean                   0.987357      0.321086        0.986507   1.000000         0.177028          0.498502        0.685983             0.823269       0.151293               -0.283110   0.732562   -0.066280      0.726628  0.800086      -0.166777        0.212583      0.207660           0.372320    -0.072497             -0.019887      0.962746       0.287489         0.959120    0.959213          0.123523           0.390410         0.512606              0.722017        0.143570
#smoothness_mean             0.170581     -0.023389        0.207278   0.177028         1.000000          0.659123        0.521984             0.553695       0.557775                0.584792   0.301467    0.068406      0.296092  0.246552       0.332375        0.318943      0.248396           0.380676     0.200774              0.283607      0.213120       0.036072         0.238853    0.206718          0.805324           0.472468         0.434926              0.503053        0.394309
#compactness_mean            0.506124      0.236702        0.556936   0.498502         0.659123          1.000000        0.883121             0.831135       0.602641                0.565369   0.497473    0.046205      0.548905  0.455653       0.135299        0.738722      0.570517           0.642262     0.229977              0.507318      0.535315       0.248133         0.590210    0.509604          0.565541           0.865809         0.816275              0.815573        0.510223
#concavity_mean              0.676764      0.302418        0.716136   0.685983         0.521984          0.883121        1.000000             0.921391       0.500667                0.336783   0.631925    0.076218      0.660391  0.617427       0.098564        0.670279      0.691270           0.683260     0.178009              0.449301      0.688236       0.299879         0.729565    0.675987          0.448822           0.754968         0.884103              0.861323        0.409464
#concave points_mean         0.822529      0.293464        0.850977   0.823269         0.553695          0.831135        0.921391             1.000000       0.462497                0.166917   0.698050    0.021480      0.710650  0.690299       0.027653        0.490424      0.439167           0.615634     0.095351              0.257584      0.830318       0.292752         0.855923    0.809630          0.452753           0.667454         0.752399              0.910155        0.375744
#symmetry_mean               0.147741      0.071401        0.183027   0.151293         0.557775          0.602641        0.500667             0.462497       1.000000                0.479921   0.303379    0.128053      0.313893  0.223970       0.187321        0.421659      0.342627           0.393298     0.449137              0.331786      0.185728       0.090651         0.219169    0.177193          0.426675           0.473200         0.433721              0.430297        0.699826
#fractal_dimension_mean     -0.311631     -0.076437       -0.261477  -0.283110         0.584792          0.565369        0.336783             0.166917       0.479921                1.000000   0.000111    0.164174      0.039830 -0.090170       0.401964        0.559837      0.446630           0.341198     0.345007              0.688132     -0.253691      -0.051269        -0.205151   -0.231854          0.504942           0.458798         0.346234              0.175325        0.334019
#radius_se                   0.679090      0.275869        0.691765   0.732562         0.301467          0.497473        0.631925             0.698050       0.303379                0.000111   1.000000    0.213247      0.972794  0.951830       0.164514        0.356065      0.332358           0.513346     0.240567              0.227754      0.715065       0.194799         0.719684    0.751548          0.141919           0.287103         0.380585              0.531062        0.094543
#texture_se                 -0.097317      0.386358       -0.086761  -0.066280         0.068406          0.046205        0.076218             0.021480       0.128053                0.164174   0.213247    1.000000      0.223171  0.111567       0.397243        0.231700      0.194998           0.230283     0.411621              0.279723     -0.111690       0.409003        -0.102242   -0.083195         -0.073658          -0.092439        -0.068956             -0.119638       -0.128215
#perimeter_se                0.674172      0.281673        0.693135   0.726628         0.296092          0.548905        0.660391             0.710650       0.313893                0.039830   0.972794    0.223171      1.000000  0.937655       0.151075        0.416322      0.362482           0.556264     0.266487              0.244143      0.697201       0.200371         0.721031    0.730713          0.130054           0.341919         0.418899              0.554897        0.109930
#area_se                     0.735864      0.259845        0.744983   0.800086         0.246552          0.455653        0.617427             0.690299       0.223970               -0.090170   0.951830    0.111567      0.937655  1.000000       0.075150        0.284840      0.270895           0.415730     0.134109              0.127071      0.757373       0.196497         0.761213    0.811408          0.125389           0.283257         0.385100              0.538166        0.074126
#smoothness_se              -0.222600      0.006614       -0.202694  -0.166777         0.332375          0.135299        0.098564             0.027653       0.187321                0.401964   0.164514    0.397243      0.151075  0.075150       1.000000        0.336696      0.268685           0.328429     0.413506              0.427374     -0.230691      -0.074743        -0.217304   -0.182195          0.314457          -0.055558        -0.058298             -0.102007       -0.107342
#compactness_se              0.206000      0.191975        0.250744   0.212583         0.318943          0.738722        0.670279             0.490424       0.421659                0.559837   0.356065    0.231700      0.416322  0.284840       0.336696        1.000000      0.801268           0.744083     0.394713              0.803269      0.204607       0.143003         0.260516    0.199371          0.227394           0.678780         0.639147              0.483208        0.277878
#concavity_se                0.194204      0.143293        0.228082   0.207660         0.248396          0.570517        0.691270             0.439167       0.342627                0.446630   0.332358    0.194998      0.362482  0.270895       0.268685        0.801268      1.000000           0.771804     0.309429              0.727372      0.186904       0.100241         0.226680    0.188353          0.168481           0.484858         0.662564              0.440472        0.197788
#concave points_se           0.376169      0.163851        0.407217   0.372320         0.380676          0.642262        0.683260             0.615634       0.393298                0.341198   0.513346    0.230283      0.556264  0.415730       0.328429        0.744083      0.771804           1.000000     0.312780              0.611044      0.358127       0.086741         0.394999    0.342271          0.215351           0.452888         0.549592              0.602450        0.143116
#symmetry_se                -0.104321      0.009127       -0.081629  -0.072497         0.200774          0.229977        0.178009             0.095351       0.449137                0.345007   0.240567    0.411621      0.266487  0.134109       0.413506        0.394713      0.309429           0.312780     1.000000              0.369078     -0.128121      -0.077473        -0.103753   -0.110343         -0.012662           0.060255         0.037119             -0.030413        0.389402
#fractal_dimension_se       -0.042641      0.054458       -0.005523  -0.019887         0.283607          0.507318        0.449301             0.257584       0.331786                0.688132   0.227754    0.279723      0.244143  0.127071       0.427374        0.803269      0.727372           0.611044     0.369078              1.000000     -0.037488      -0.003195        -0.001000   -0.022736          0.170568           0.390159         0.379975              0.215204        0.111094
#radius_worst                0.969539      0.352573        0.969476   0.962746         0.213120          0.535315        0.688236             0.830318       0.185728               -0.253691   0.715065   -0.111690      0.697201  0.757373      -0.230691        0.204607      0.186904           0.358127    -0.128121             -0.037488      1.000000       0.359921         0.993708    0.984015          0.216574           0.475820         0.573975              0.787424        0.243529
#texture_worst               0.297008      0.912045        0.303038   0.287489         0.036072          0.248133        0.299879             0.292752       0.090651               -0.051269   0.194799    0.409003      0.200371  0.196497      -0.074743        0.143003      0.100241           0.086741    -0.077473             -0.003195      0.359921       1.000000         0.365098    0.345842          0.225429           0.360832         0.368366              0.359755        0.233027
#perimeter_worst             0.965137      0.358040        0.970387   0.959120         0.238853          0.590210        0.729565             0.855923       0.219169               -0.205151   0.719684   -0.102242      0.721031  0.761213      -0.217304        0.260516      0.226680           0.394999    -0.103753             -0.001000      0.993708       0.365098         1.000000    0.977578          0.236775           0.529408         0.618344              0.816322        0.269493
#area_worst                  0.941082      0.343546        0.941550   0.959213         0.206718          0.509604        0.675987             0.809630       0.177193               -0.231854   0.751548   -0.083195      0.730713  0.811408      -0.182195        0.199371      0.188353           0.342271    -0.110343             -0.022736      0.984015       0.345842         0.977578    1.000000          0.209145           0.438296         0.543331              0.747419        0.209146
#smoothness_worst            0.119616      0.077503        0.150549   0.123523         0.805324          0.565541        0.448822             0.452753       0.426675                0.504942   0.141919   -0.073658      0.130054  0.125389       0.314457        0.227394      0.168481           0.215351    -0.012662              0.170568      0.216574       0.225429         0.236775    0.209145          1.000000           0.568187         0.518523              0.547691        0.493838
#compactness_worst           0.413463      0.277830        0.455774   0.390410         0.472468          0.865809        0.754968             0.667454       0.473200                0.458798   0.287103   -0.092439      0.341919  0.283257      -0.055558        0.678780      0.484858           0.452888     0.060255              0.390159      0.475820       0.360832         0.529408    0.438296          0.568187           1.000000         0.892261              0.801080        0.614441
#concavity_worst             0.526911      0.301025        0.563879   0.512606         0.434926          0.816275        0.884103             0.752399       0.433721                0.346234   0.380585   -0.068956      0.418899  0.385100      -0.058298        0.639147      0.662564           0.549592     0.037119              0.379975      0.573975       0.368366         0.618344    0.543331          0.518523           0.892261         1.000000              0.855434        0.532520
#concave points_worst        0.744214      0.295316        0.771241   0.722017         0.503053          0.815573        0.861323             0.910155       0.430297                0.175325   0.531062   -0.119638      0.554897  0.538166      -0.102007        0.483208      0.440472           0.602450    -0.030413              0.215204      0.787424       0.359755         0.816322    0.747419          0.547691           0.801080         0.855434              1.000000        0.502528
#symmetry_worst              0.163953      0.105008        0.189115   0.143570         0.394309          0.510223        0.409464             0.375744       0.699826                0.334019   0.094543   -0.128215      0.109930  0.074126      -0.107342        0.277878      0.197788           0.143116     0.389402              0.111094      0.243529       0.233027         0.269493    0.209146          0.493838           0.614441         0.532520              0.502528        1.000000
#fractal_dimension_worst     0.007066      0.119205        0.051019   0.003738         0.499316          0.687382        0.514930             0.368661       0.438413                0.767297   0.049559   -0.045655      0.085433  0.017539       0.101480        0.590973      0.439329           0.310655     0.078079              0.591328      0.093492       0.219122         0.138957    0.079647          0.617624           0.810455         0.686511              0.511114        0.537848
#                         fractal_dimension_worst
#radius_mean                             0.007066
#texture_mean                            0.119205
#perimeter_mean                          0.051019
#area_mean                               0.003738
#smoothness_mean                         0.499316
#compactness_mean                        0.687382
#concavity_mean                          0.514930
#concave points_mean                     0.368661
#symmetry_mean                           0.438413
#fractal_dimension_mean                  0.767297
#radius_se                               0.049559
#texture_se                             -0.045655
#perimeter_se                            0.085433
#area_se                                 0.017539
#smoothness_se                           0.101480
#compactness_se                          0.590973
#concavity_se                            0.439329
#concave points_se                       0.310655
#symmetry_se                             0.078079
#fractal_dimension_se                    0.591328
#radius_worst                            0.093492
#texture_worst                           0.219122
#perimeter_worst                         0.138957
#area_worst                              0.079647
#smoothness_worst                        0.617624
#compactness_worst                       0.810455
#concavity_worst                         0.686511
#concave points_worst                    0.511114
#symmetry_worst                          0.537848
#fractal_dimension_worst                 1.000000

#Korelasyon: değişkenlerin birbiriyle ilgisini ifade eden bir istatistiksel ölçümdür.
#-1 ile +1 arasında değerler alır. -1'e ya da +1'e yaklaştıkça ilişkinin şiddeti kuvetlenir.
#Bir değişkenin değeri arttıkça diğerinin de değeri artar.
#iki değişken arası ilişki negatifse bir değişkenin değerleri artarken diğer değişkenin değerleri azalır.
#bu korelasyonlar da ifade edildiği üzere 1'e yaklaştıkça ilişki şiddetli kuvvetli -1'e yaklaştıkça da ilişki şiddeti kuvvetlidir. -1 negatif yönlü +1 pozitif yönlüdür.
#0'a yakınsa korelasyon olmadığı anlamına gelir. =düşük korelasyon

#Genelde analitik çalışmalarda birbiriyle yüksek korelasyonlu değişkenlerin çalışmalarda bulunmamasını isteriz. Birisini çalışma dışında bırakmak isteriz.


sns.set(rc={'figure.figsize': (12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)














