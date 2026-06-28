# Proje Dokümanı

Bu doküman, projede hangi verilerin kullanıldığını, hangi işlemlerin yapıldığını, neden bu yöntemlerin seçildiğini ve hangi sonuçlara ulaşıldığını anlayabilmesi için hazırlanmıştır.

## 1. Proje Ne Hakkında?

Bu proje, restoran müşterilerinin davranışlarının hava koşullarıyla nasıl değiştiğini inceleyen bir veri analizi çalışmasıdır. Temel fikir şudur: insanların restoranda ne kadar süre kaldığı, açık alan mı kapalı alan mı tercih ettiği, bir günde kaç oturum oluştuğu ve ne kadar harcama yaptığı gibi davranışlar sadece işletme içi faktörlerle değil, dış çevre koşullarıyla da ilişkili olabilir.

Bu nedenle çalışma, **Internet of Behaviors (IoB)** yaklaşımıyla ele alındı. IoB, kullanıcı davranışlarını çevresel ve dijital sinyallerle birlikte inceleyen bir yaklaşımdır. Burada çevresel sinyal olarak hava durumu verileri kullanıldı.

## 2. Bu Projede Hangi Sorulara Cevap Arandı?

Çalışmanın ana soruları şunlardı:

- Hava koşulları oturum süresini etkiliyor mu?
- Yağışlı günlerde müşteri sayısı azalıyor mu?
- Bahçe, İç Salon ve Paket tercihleri hava durumuna göre değişiyor mu?
- COVID öncesi, COVID ve sonrası dönemlerde davranış kalıpları farklı mı?
- Hava değişkenleri toplam harcama ve sipariş miktarını açıklayabiliyor mu?

Bu soruların ortak amacı, restoran yönetimi için daha bilinçli kararlar üretmektir. Örneğin doğru stok planı yapmak, yoğun günleri önceden tahmin etmek, Bahçe alanını nasıl kullanmak gerektiğine karar vermek ve dönemsel değişimleri anlamak gibi.

## 3. Kullanılan Veriler

Projede iki farklı veri türü birleştirildi:

- Restorana ait sipariş ve masa kayıtları
- Open-Meteo Archive API üzerinden alınan saatlik hava durumu verileri

Restoran verisinde masa numarası, oturum başlangıç ve bitiş saatleri, ürün bilgileri, miktar, fiyat ve gün bilgileri gibi alanlar vardı. Hava verisinde ise sıcaklık, hissedilen sıcaklık, yağış, rüzgar, nem, bulutluluk, basınç, kısa dalga radyasyonu ve benzeri değişkenler yer aldı.

Veriler birleştirildikten sonra her restoran oturumunun yanında o ana ait hava koşulları da bulunur hale geldi. Bu, davranışı çevre koşullarıyla birlikte incelemeyi mümkün kıldı.

## 4. Veri Nasıl Hazırlandı?

### [data-cleaning.ipynb](data-cleaning.ipynb)

Bu dosya, çalışmanın veri hazırlama omurgasını oluşturur. Buradaki amaç ham veriyi analiz edilebilir hale getirmektir.

Bu notebook’ta yapılanlar:

- Açılış ve kapanış saatlerinden oturum süresi hesaplandı.
- Aynı oturuma ait kayıtlar tek satır altında toplandı.
- Masa numarasına göre masa grubu üretildi: Bahçe, İç Salon, Paket.
- Tarih, saat, gün, ay, yıl ve hafta numarası gibi zaman değişkenleri çıkarıldı.
- Hava verisi ile oturum verisi saat bazında eşleştirildi.
- Yağış kategorisi ve sıcaklık aralığı gibi yorumlamayı kolaylaştıran yeni alanlar üretildi.
- Çok kısa veya çok uzun oturumlar outlier olarak işaretlendi.

Bu aşamanın sonunda şu ana veri seti üretildi:

- [Veriler/oturum_hava_birlesik.csv](Veriler/oturum_hava_birlesik.csv)

Bu dosya, sonraki tüm analizlerin ana girdisidir.

### Veri Hazırlama Neden Gerekiyordu?

Ham veri tek başına modelleme için uygun değildi çünkü:

- Aynı oturum birden fazla satırda yer alıyordu.
- Hava verisi ile doğrudan eşleşmiyordu.
- Oturum süresi gibi temel metrikler doğrudan hazır değildi.
- Bazı kayıtlar olağandışı davranışlar içeriyordu.

Bu nedenle önce veri temizlendi, sonra anlamlı özellikler üretildi, ardından hava verisiyle birleştirildi.

## 5. İlk Keşifsel Analizlerde Ne Görüldü?

### [eda.ipynb](eda.ipynb)

Bu notebook’un amacı, veri setindeki genel desenleri görmekti. Başka bir ifadeyle, model kurmadan önce “veri ne söylüyor?” sorusuna cevap arandı.

İncelenen başlıca başlıklar:

- Yıllık oturum sayısı değişimi
- Aylık oturum yoğunluğu
- Saatlik oturum dağılımı
- Saat × gün ısı haritası
- Yağış kategorisine göre oturum süresi
- Sıcaklık aralığına göre müşteri davranışı
- Bahçe ve İç Salon gruplarının karşılaştırılması
- COVID döneminin zamansal etkisi

Bu analizler şu tip sonuçlar gösterdi:

- Öğle saatleri en yoğun zaman dilimi olarak öne çıktı.
- Bahar ve yaz aylarında Bahçe kullanımı artma eğiliminde oldu.
- Yağışlı ve soğuk dönemlerde İç Salon daha baskın hale geldi.
- COVID dönemi boyunca oturum sayılarında belirgin dalgalanmalar yaşandı.

Bu bölümün önemi şudur: daha karmaşık analizlere geçmeden önce verinin davranışını anlamak gerekir. Hangi değişkenlerin ilgi çekici olduğu burada görülür.

## 6. EDA Sonrası Neden İstatistiksel Test Yapıldı?

### [deep_analysis.ipynb](deep_analysis.ipynb)

EDA’da grafik olarak görülen farkların gerçekten anlamlı olup olmadığını test etmek için istatistiksel analiz yapıldı.

Burada kullanılan yöntemler ve amaçları:

- **Kruskal-Wallis testi**: Birden fazla grubun medyanlarının farklı olup olmadığını kontrol etmek için.
- **Dunn post-hoc testi**: Hangi iki grup arasında fark olduğunu görmek için.
- **Mann-Whitney U testi**: İki grubun dağılım farkını kıyaslamak için.
- **Cohen’s d**: Farkın sadece anlamlı değil, aynı zamanda pratik olarak ne kadar büyük olduğunu yorumlamak için.

Bu notebook’ta ele alınan konular:

- Yağış kategorileri ve oturum süresi
- Sıcaklık aralıkları ve oturum süresi
- Dönemler arası davranış farkı
- Yağışlı ve kuru günlerde Bahçe/İç Salon davranışı
- Yağmur kaynaklı tahmini müşteri kaybı
- Post-COVID döneminde masa devir hızı
- Paket segmentinin ayrı davranış profili
- Bahçe doluluk tahmin modeli

Bu bölümün ana sonucu şudur:

- Gözlenen farklar yalnızca görsel değil, bazı durumlarda istatistiksel olarak da anlamlıdır.
- Özellikle yağış ve sıcaklık, oturum süresini ve mekan tercihini etkileyen güçlü değişkenlerdir.

## 7. Oturum Süresi Analizi Neden Önemli?

### [Oturum_sure.ipynb](Oturum_sure.ipynb)

Bu notebook, oturum süresi davranışını detaylı olarak açıklamaya çalışır. Çünkü oturum süresi, müşteri bağlılığını ve mekanda kalma eğilimini gösteren önemli bir metriktir.

Yapılan işlemler:

- Veri kalite kontrolü
- Feature engineering
- Cohen’s d etki büyüklüğü hesapları
- Yağışlı günlerde gelir kaybı yaklaşımı
- Oturum süresi tahmin modelleri
- 5-fold cross validation
- Feature importance analizi

Bu bölümde ulaşılan temel düşünce şudur:

> Hava koşulları oturum süresini etkiler, ancak tek başına bütün davranışı açıklamaz. Modelin çok yüksek olmaması, veri setinde olmayan başka etkilerin de önemli olduğunu gösterir.

Bu nokta önemlidir çünkü modelin zayıf görünmesi her zaman kötü sonuç değildir. Bazen düşük ya da orta düzey açıklama gücü, problemin doğası gereği beklenen bir durumdur.

## 8. Dönemler Arasındaki Karşılaştırma Neden Yapıldı?

### [donem_karsilastirma.py](donem_karsilastirma.py)

Bu dosyada üç dönem karşılaştırıldı:

- Pre-COVID
- COVID
- Post-COVID

Bu karşılaştırma neden gerekliydi?

Çünkü aynı hava koşulu farklı dönemlerde aynı davranışı üretmeyebilir. Toplumsal, ekonomik ve alışkanlık değişimleri davranışı dönüştürebilir.

Bu dosyada yapılan çalışmalar:

- Dönem bazlı betimsel istatistikler
- Kruskal-Wallis testi
- Cohen’s d ile etki büyüklüğü analizi
- Dönem × yağış ilişkisi
- Dönem bazlı Random Forest modeli
- Feature importance karşılaştırması

Bu analizle görülen ana fikir:

- Müşteri davranışı yalnızca hava durumuna bağlı değildir.
- Zamanın bağlamı, özellikle COVID gibi kırılma noktaları, hava etkisini değiştirebilir.

## 9. Günlük Talep Analizinde Neye Bakıldı?

### [gunluk_talep.py](gunluk_talep.py)

Bu analizde hedef, günlük oturum sayısını tahmin etmekti. Böylece hava koşullarına bakarak o gün işletmeye yaklaşık kaç oturum gelebileceği anlaşılmak istendi.

Yapılanlar:

- Hava değişkenleri ile günlük oturum sayısı arasındaki korelasyonlar hesaplandı.
- Yağışlı ve kuru günler karşılaştırıldı.
- Dönem bazlı yağış etkisi incelendi.
- Birden fazla regresyon modeli karşılaştırıldı.
- 5-fold cross validation uygulandı.
- Feature importance ile etkili değişkenler belirlendi.

Kullanılan modeller:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting

Bu analizin pratik karşılığı şudur:

- Personel planlaması
- Mutfak ve stok hazırlığı
- Günlük yoğunluk öngörüsü

## 10. Mekan Tercihi Analizinde Neden Sınıflandırma Kullanıldı?

### [mekan_tercihi.py](mekan_tercihi.py)

Bu dosyada müşterinin Bahçe mi yoksa İç Salon mu seçtiği tahmin edilmeye çalışıldı. Bu problem bir sınıflandırma problemidir çünkü çıktı iki kategoriden biri olur.

Kullanılan yöntemler:

- Logistic Regression
- Random Forest
- Gradient Boosting

Değerlendirme için kullanılan metrikler:

- Accuracy
- AUC-ROC
- F1-score
- Confusion matrix
- Feature importance
- Stratified cross validation

Bu analiz bize şunu anlattı:

- Hava koşulları açık alan tercihini ciddi biçimde etkiler.
- Yağış arttıkça Bahçe kullanımı düşer.
- Sıcaklık ve oturum saatinin de etkisi vardır.

Bu analiz, restoranın açık alan yönetimi açısından önemlidir. Çünkü hava durumuna göre kaç masanın kullanılacağı önceden tahmin edilebilir.

## 11. Satış Davranışı Analizinde Ne İncelendi?

### [satis_davranisi.py](satis_davranisi.py)

Bu dosyada iki ayrı hedef ele alındı:

- Toplam tutar
- Toplam miktar

Burada amaç, hava koşullarının müşterinin harcama davranışını etkileyip etkilemediğini görmekti.

İncelenen başlıklar:

- Betimsel istatistikler
- Yağış kategorisine göre karşılaştırmalar
- Sıcaklık aralığına göre karşılaştırmalar
- Masa grubuna göre farklılıklar
- Kruskal-Wallis testi
- Regresyon modelleri
- Cross validation
- Feature importance

Bu bölümün ana yorumu:

- Hava koşulları harcamayı tamamen açıklamaz.
- Ancak yağış ve sıcaklık gibi değişkenler bazı dönemlerde belirgin farklılıklar oluşturabilir.

Bu da işletme açısından önemlidir çünkü harcama davranışı stok ve gelir planlamasını doğrudan etkiler.

## 12. Üretilen Temel Çıktılar Nelerdir?

Bu projede çeşitli görseller ve model çıktıları üretildi. Bunlar analiz sonuçlarının görsel olarak sunulmasını sağlar.

Önemli çıktı dosyaları:

- [Outputs/Donem_karsilastirma_cikti/feature_importance_donem_karsilastirma.png](Outputs/Donem_karsilastirma_cikti/feature_importance_donem_karsilastirma.png)
- [Outputs/Gunluk_talep_cikti/feature_importance_gunluk.png](Outputs/Gunluk_talep_cikti/feature_importance_gunluk.png)
- [Outputs/Mekan_tercihi_cikti/confusion_matrix.png](Outputs/Mekan_tercihi_cikti/confusion_matrix.png)
- [Outputs/Mekan_tercihi_cikti/feature_importance_mekan.png](Outputs/Mekan_tercihi_cikti/feature_importance_mekan.png)
- [Outputs/Satis_davranisi_cikti/feature_importance_toplam_tutar.png](Outputs/Satis_davranisi_cikti/feature_importance_toplam_tutar.png)
- [Outputs/Satis_davranisi_cikti/feature_importance_toplam_miktar.png](Outputs/Satis_davranisi_cikti/feature_importance_toplam_miktar.png)

## 13. Bu Çalışmanın Genel Sonucu Nedir?

Bu proje gösteriyor ki restoran müşteri davranışı sadece işletmenin iç dinamikleriyle açıklanamaz. Hava koşulları, dönem etkisi ve mekansal tercih gibi dışsal faktörler de davranışı anlamlı biçimde etkiler.

En genel sonuçlar:

- Oturum süresi hava koşullarına duyarlıdır.
- Yağışlı günlerde Bahçe kullanımı düşer.
- Günlük yoğunluk hava koşullarıyla değişebilir.
- COVID gibi dönemsel kırılmalar davranış kalıplarını etkiler.
- Satış ve sipariş miktarı bazı hava değişkenlerine tepki verebilir.

## 14. Bu Proje İşletme Açısından Ne Fayda Sağlar?

Bu analizler doğrudan operasyonel kararlara katkı sağlayabilir:

- Stok planlaması
- Personel vardiya planı
- Açık alan kullanım stratejisi
- Promosyon zamanlaması
- Talep tahmini
- Dönemsel davranış karşılaştırması

## 15. Dokümanı Okuyan Birinin Bilmesi Gereken Kısa Özet

Bu proje önce restoran verisini ve hava verisini birleştirir, sonra müşteri davranışını farklı açılardan inceler. Önce grafiklerle genel eğilimler görülür, sonra istatistiksel testlerle bu eğilimler doğrulanır, en sonunda da makine öğrenmesi modelleriyle tahmin yapılır.

Tek cümlelik özet:

> Bu çalışma, restoran oturum verilerini hava durumu verileriyle birleştirerek oturum süresi, günlük yoğunluk, mekan tercihi ve satış davranışının nasıl değiştiğini inceleyen veri odaklı bir IoB analizidir.

