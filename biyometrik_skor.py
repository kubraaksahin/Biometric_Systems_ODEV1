import numpy as np
import matplotlib.pyplot as plt

def minmax_01_global(X, eps=1e-12):
    """
        Özellik vektörlerinin tüm elemanlarını [0,1] aralığına çektim.
    """
    xmin = X.min()
    xmax = X.max()
    rng = max(xmax - xmin, eps)
    return (X - xmin) / rng

def score_from_vectors(a, b):
    """
        Matcher modülünün iki vektörü karşılaştırarak skor üretmesini simüle ettim.
    """
    d = np.linalg.norm(a - b)
    return 1.0 / (1.0 + d)

def compute_genuine_imposter_scores(F):
   
    T, N, D = F.shape

    # 1) Genuine skorları hesapla (aynı kişi, farklı zaman)
    genuine = []
    for i in range(N):
        for t1 in range(T):
            for t2 in range(t1 + 1, T):
                genuine.append(score_from_vectors(F[t1, i], F[t2, i]))
    genuine = np.array(genuine, dtype=np.float64)

    # 2) Imposter skorları hesapla (farklı kişi, tüm zaman kombinasyonları)
    imposter = np.empty((T * T * (N * (N - 1) // 2),), dtype=np.float64)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            for t1 in range(T):
                a = F[t1, i]
                for t2 in range(T):
                    imposter[k] = score_from_vectors(a, F[t2, j])
                    k += 1

    return genuine, imposter

def far_frr_over_thresholds(genuine, imposter, thresholds):
    """
        Eşik değerine (threshold) bağlı olarak FAR ve FRR değerlerini hesaplamak.
    """
    fars = np.empty_like(thresholds, dtype=np.float64)
    frrs = np.empty_like(thresholds, dtype=np.float64)

    for idx, th in enumerate(thresholds):
        fars[idx] = np.mean(imposter >= th)  # sahteci içinde kabul oranı
        frrs[idx] = np.mean(genuine < th)    # gerçek içinde ret oranı

    return fars, frrs

def eer_from_curves(thresholds, fars, frrs):
    """
        EER (Equal Error Rate - Eşit Hata Oranı) değerini bulmak.
    """
    diff = np.abs(fars - frrs)
    i = np.argmin(diff)
    eer = (fars[i] + frrs[i]) / 2.0
    return eer, thresholds[i], fars[i], frrs[i], i

def main():
    # 1) VERİYİ DOSYADAN OKU
    Features = np.load("Features.npy")

    # 2) SADECE İLK 100 KİŞİYİ AL
    Features = Features[:, :100, :]  

    # 3) NORMALİZASYON [0,1]
    
    Features = minmax_01_global(Features)

    print("Loaded Features shape:", Features.shape)
    print("Min/Max after normalization:", Features.min(), Features.max())

    # 4) GENUINE ve IMPOSTER SKORLARINI HESAPLA
    genuine, imposter = compute_genuine_imposter_scores(Features)
    print("Genuine scores count:", genuine.size)     # 100*45 = 4500
    print("Imposter scores count:", imposter.size)   # 495000

    # 5) SKOR DAĞILIMLARININ ÖZET İSTATİSTİKLERİ
    def summary(x):
        return {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p01": float(np.quantile(x, 0.01)),
            "p50": float(np.quantile(x, 0.50)),
            "p99": float(np.quantile(x, 0.99)),
        }

    print("\nGenuine summary:", summary(genuine))
    print("Imposter summary:", summary(imposter))

    # 6) GENUINE ve IMPOSTER DAĞILIMLARINI AYNI GRAFİKTE ÇİZ
    # density=True -> histogramları olasılık yoğunluğu gibi normalize eder.
    plt.figure(figsize=(9, 5))
    bins = 60
    plt.hist(imposter, bins=bins, density=True, alpha=0.55, label="Imposter", color="tab:orange")
    plt.hist(genuine,  bins=bins, density=True, alpha=0.55, label="Genuine",  color="tab:blue")
    plt.xlabel("Skor (1 / (1 + Öklid Mesafesi))")
    plt.ylabel("Yoğunluk (Density)")
    plt.title("Genuine ve Imposter Skor Dağılımları (İlk 100 Kişi)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("score_distributions.png", dpi=200)
    plt.close()

    # 7) FAR / FRR EĞRİLERİNİ EŞİK DEĞERLERİNE GÖRE HESAPLA
    # Eşik aralığını, skorların kapsadığı min-max aralıktan seçiyoruz.
    smin = float(min(genuine.min(), imposter.min()))
    smax = float(max(genuine.max(), imposter.max()))
    thresholds = np.linspace(smin, smax, 1500)

    fars, frrs = far_frr_over_thresholds(genuine, imposter, thresholds)

    # 8) EER HESAPLA
    eer, th_eer, far_eer, frr_eer, idx = eer_from_curves(thresholds, fars, frrs)

    print("\nEER results:")
    print("  EER =", eer)
    print("  Threshold@EER =", th_eer)
    print("  FAR@EER =", far_eer)
    print("  FRR@EER =", frr_eer)

    # 9) FAR-FRR-EER'İ AYNI GRAFİKTE GÖSTER
    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, fars, label="FAR", color="tab:orange")
    plt.plot(thresholds, frrs, label="FRR", color="tab:blue")
    plt.axvline(th_eer, color="k", linestyle="--", linewidth=1, label=f"EER eşiği={th_eer:.4f}")
    plt.scatter([th_eer], [eer], color="red", zorder=5, label=f"EER={eer:.4f}")
    plt.xlabel("Eşik (Threshold)")
    plt.ylabel("Hata Oranı (Error Rate)")
    plt.title("Eşiğe Göre FAR/FRR Değişimi ve EER")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("far_frr_eer_vs_threshold.png", dpi=200)
    plt.close()

    # 10) FRR'nin FAR'a GÖRE DEĞİŞİMİ (FRR vs FAR) ÇİZ
    # FAR artan sırada daha düzgün bir eğri için sıralama yapıyoruz.
    order = np.argsort(fars)
    plt.figure(figsize=(6.5, 6))
    plt.plot(fars[order], frrs[order], color="tab:green")
    plt.scatter([far_eer], [frr_eer], color="red", zorder=5, label=f"EER={eer:.4f}")
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("FRR'nin FAR'a Göre Değişimi")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("frr_vs_far.png", dpi=200)
    plt.close()

    # 11) HESAPLANAN TÜM ÇIKTILARI KAYDET
    # computed_scores.npz içine genuine, imposter, FAR, FRR, threshold ve EER bilgileri yazılır.
    np.savez_compressed(
        "computed_scores.npz",
        genuine=genuine,
        imposter=imposter,
        thresholds=thresholds,
        FAR=fars,
        FRR=frrs,
        EER=np.array([eer]),
        EER_threshold=np.array([th_eer]),
    )

    print("\nSaved outputs:")
    print("  score_distributions.png")
    print("  far_frr_eer_vs_threshold.png")
    print("  frr_vs_far.png")
    print("  computed_scores.npz")

if __name__ == "__main__":
    main()