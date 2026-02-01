# IoB Tabanlı Hava Durumu Analizleriyle Gıda Sektöründe Stratejik Müşteri Davranışı Yönetimi

Bu proje, **Internet of Behaviors (IoB)** yaklaşımını kullanarak lokanta müşterilerinin davranışlarını hava durumu verileriyle ilişkilendirmeyi ve gıda sektöründe veri odaklı stratejik karar alma süreçlerini geliştirmeyi amaçlamaktadır.

## Proje Amacı

Gerçek lokanta verileri (sipariş kayıtları, masa kullanım süreleri, ürün tercihleri) ile saatlik hava durumu verileri (sıcaklık, hissedilen sıcaklık, yağış, nem, rüzgar vb.) entegre edilerek şu sorulara cevap aranmaktadır:

- Hava koşulları (sıcaklık, yağış, nem) müşteri ürün tercihlerini ve tüketim paternlerini nasıl etkiliyor?  
- Çevresel faktörler, müşterilerin lokantada geçirdiği süreyi ve sipariş hacmini nasıl değiştiriyor?  
- Bu veriler kullanılarak işletmelere gerçek zamanlı stok yönetimi, menü optimizasyonu ve promosyon stratejileri önerilebilir mi?

Proje kapsamında lokantanın SQL veritabanından elde edilen ham veriler temizlenmekte, oturum bazında gruplanmakta ve açık kaynaklı hava durumu API'leri ile birleştirilmektedir.

## Kullanılan Teknolojiler

- Veri kaynağı: SQL Server (lokanta sipariş ve masa kayıtları)  
- Veri işleme & analiz: Python, pandas  
- Hava durumu verisi: Open-Meteo Archive API (saatlik tarihi veriler, 2012–2025)  
- Geliştirme ortamı: Yerel Python ortamı (Jupyter Notebook / VS Code)  

## Proje Kapsamı

- Lokanta verilerinin temizlenmesi ve oturum (masa + zaman aralığı) bazında özetlenmesi  
- Saatlik hava durumu verilerinin lokantanın tam koordinatlarına göre çekilmesi  
- Verilerin zaman damgası üzerinden entegrasyonu  
- Çevresel faktörlerin müşteri davranışı üzerindeki etkisinin analizi  

Proje, TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı kapsamında geliştirilmektedir.

## Lisans

Eğitim ve araştırma amaçlıdır. Kaynak gösterilerek kullanılabilir.

Son güncelleme: 2026