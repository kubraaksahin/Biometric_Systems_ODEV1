# Biometric_Systems_ODEV1
Calculating FAR, FRR, Threshold, and ERR.


ÖDEV DETAYLARI :
Bir biyometrik sistemin özellik çıkaran modülü (Feature Extractor) ile 10 farklı zamanda 110 kişiye ait 
biyometrik veriler üzerinden her biri 6 elemanlı olan özellik vektörleri (Feature Vectors) çıkarılmıştır. 
Özellik vektörleri Features.npz adlı dosyaya python numpy dosyası halinde kaydedilmiştir. Dosyadan 
okutulacak Features adlı numpy array’i 10, 110, 6 boyutlarındadır. 
Örnek: Features[0, 3] ifadesi (Numaraları 0’dan başlatılan) ilk seferde çıkarılan (yine numaraları 0’dan 
başlatılan) 3. kişiye ait 6 elemanlı özellik vektörünü vermektedir. 
>>> Features[0, 3] 
array([4.03230, 4.20211, 2.86105, 5.81975, 2.61711, 6.42673]) 
Örnek: Features[9, 56] ifadesi son seferde çıkarılan 56. kişiye ait özellik vektörünü vermektedir. 
 >>> Features[9, 56] 
array([5.44869, 4.15940, 5.43823, 6.11477, 2.04102, 6.08631]) 
Features array’inin ilk 100 kişiye ait tüm verilerini alın ve işlemlerinize ilk 100 kişinin verileri ile devam 
edin. (Son 10 kişinin verilerini bu aşamadan itibaren dikkate almayın.) 
Özellik vektörlerini tüm elemanlarının değerleri [0 1] aralığında kalacak şekilde normalize edin. 
Bu biyometrik sistemin karşılaştırıcı modülü (Matcher) ile iki özellik vektörünün öklid mesafesi (euclidiean 
distance) hesaplanıp 1/(1+ ÖklidMesafesi) formülü ile skor hesabı yapılıyor olsun.

 Yazacağınız python betiği (script) ile elde ettiğiniz veriler üzerinde: 
Hesaplanabilecek tüm muhtemel gerçek skor (Genuine Score) hesaplarını yapıp gerçek skor dağılımını 
(Genuine Score Distribution) elde edin. 
Hesaplanabilecek tüm muhtemel sahteci skor (Imposter Score) hesaplarını yapıp sahteci skor dağılımını 
(Imposter Score Distribution) elde edin. 
Sahteci skor dağılımını (Imposter Score Distribution) ve gerçek skor dağılımını (Genuine Score 
Distribution) aynı çizim üzerinde gösterin. 
Eşik değerlerine karşı Yanlış Kabul Oranı (False Acceptance Rate) – FAR hesaplamalarını yapın. 
Eşik değerlerine karşı Yanlış Ret Oranı (False Reject Rate) – FRR hesaplamalarını yapın. 
Eşit Hata Oranı (Equal Error Rate) – EER değerini hesaplayın. 
Eşik değerlerine karşı FAR ile FRR değerlerinin değişimi ve ERR değerini aynı çizimde gösterin. 
FAR değerlerine karşı FRR değerlerinin değişimini çizim üzerinde gösterin.
