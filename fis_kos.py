# ===== Tugas Pengganti UTS Sistem Cerdas: FIS Mamdani pemilihan kos mahasiswa =====
import os
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')                       # gambar disimpan ke file, tanpa jendela
import matplotlib.pyplot as plt

# 1) Semesta pembicaraan & variabel linguistik (4 input, 1 output)
jarak     = ctrl.Antecedent(np.arange(0, 5.01, 0.1), 'jarak')       # 0..5 km ke kampus
keamanan  = ctrl.Antecedent(np.arange(0, 101, 1), 'keamanan')       # skor kenyamanan & keamanan 0..100
harga     = ctrl.Antecedent(np.arange(200, 2001, 10), 'harga')      # 200..2000 ribu Rp/bulan
fasilitas = ctrl.Antecedent(np.arange(0, 101, 1), 'fasilitas')      # skor kelengkapan fasilitas 0..100
kelayakan = ctrl.Consequent(np.arange(0, 101, 1), 'kelayakan', defuzzify_method='centroid')

# 2) Fungsi keanggotaan (trapmf: [a,b,c,d], trimf: [a,b,c])
jarak['dekat']  = fuzz.trapmf(jarak.universe, [0, 0, 0.5, 1.5])
jarak['sedang'] = fuzz.trimf (jarak.universe, [0.5, 1.5, 3])
jarak['jauh']   = fuzz.trapmf(jarak.universe, [1.5, 3, 5, 5])
harga['murah']  = fuzz.trapmf(harga.universe, [200, 200, 400, 700])
harga['sedang'] = fuzz.trimf (harga.universe, [400, 800, 1200])
harga['mahal']  = fuzz.trapmf(harga.universe, [900, 1300, 2000, 2000])
for var, (a, b, c) in ((keamanan,  ('rendah', 'sedang', 'tinggi')),
                       (fasilitas, ('kurang', 'cukup',  'lengkap'))):
    var[a] = fuzz.trapmf(var.universe, [0, 0, 25, 50])
    var[b] = fuzz.trimf (var.universe, [25, 50, 75])
    var[c] = fuzz.trapmf(var.universe, [50, 75, 100, 100])
OUT = ['sangat_kurang', 'kurang', 'cukup', 'baik', 'sangat_baik']
kelayakan['sangat_kurang'] = fuzz.trapmf(kelayakan.universe, [0, 0, 10, 30])
kelayakan['kurang']        = fuzz.trimf (kelayakan.universe, [0, 25, 50])
kelayakan['cukup']         = fuzz.trimf (kelayakan.universe, [25, 50, 75])
kelayakan['baik']          = fuzz.trimf (kelayakan.universe, [50, 75, 100])
kelayakan['sangat_baik']   = fuzz.trapmf(kelayakan.universe, [70, 90, 100, 100])

# 3) Basis aturan 3x3x3x3 = 81 aturan (matriks tulis tangan, gaya matriks R pada contoh Bapak)
#    Satu blok = satu pasangan (harga, fasilitas). Baris matriks = jarak, kolom = keamanan.
#    Logika: makin dekat, aman, murah, dan lengkap -> makin layak
J = ['jauh', 'sedang', 'dekat']
K = ['rendah', 'sedang', 'tinggi']
R = {
    ('mahal', 'kurang'): [      # harga mahal, fasilitas kurang
        ['sangat_kurang','sangat_kurang','kurang'],           # jarak jauh
        ['sangat_kurang','kurang',       'kurang'],           # jarak sedang
        ['kurang',       'cukup',        'cukup'],            # jarak dekat
    ],
    ('mahal', 'cukup'): [       # harga mahal, fasilitas cukup
        ['sangat_kurang','kurang',       'kurang'],           # jarak jauh
        ['kurang',       'kurang',       'cukup'],            # jarak sedang
        ['cukup',        'cukup',        'baik'],             # jarak dekat
    ],
    ('mahal', 'lengkap'): [     # harga mahal, fasilitas lengkap
        ['kurang',       'kurang',       'cukup'],            # jarak jauh
        ['kurang',       'cukup',        'cukup'],            # jarak sedang
        ['cukup',        'baik',         'baik'],             # jarak dekat
    ],
    ('sedang', 'kurang'): [     # harga sedang, fasilitas kurang
        ['sangat_kurang','kurang',       'kurang'],           # jarak jauh
        ['kurang',       'cukup',        'cukup'],            # jarak sedang
        ['cukup',        'cukup',        'baik'],             # jarak dekat
    ],
    ('sedang', 'cukup'): [      # harga sedang, fasilitas cukup
        ['kurang',       'kurang',       'cukup'],            # jarak jauh
        ['cukup',        'cukup',        'baik'],             # jarak sedang
        ['cukup',        'baik',         'baik'],             # jarak dekat
    ],
    ('sedang', 'lengkap'): [    # harga sedang, fasilitas lengkap
        ['kurang',       'cukup',        'cukup'],            # jarak jauh
        ['cukup',        'baik',         'baik'],             # jarak sedang
        ['baik',         'baik',         'sangat_baik'],      # jarak dekat
    ],
    ('murah', 'kurang'): [      # harga murah, fasilitas kurang
        ['kurang',       'cukup',        'cukup'],            # jarak jauh
        ['cukup',        'cukup',        'baik'],             # jarak sedang
        ['baik',         'baik',         'sangat_baik'],      # jarak dekat
    ],
    ('murah', 'cukup'): [       # harga murah, fasilitas cukup
        ['cukup',        'cukup',        'baik'],             # jarak jauh
        ['cukup',        'baik',         'baik'],             # jarak sedang
        ['baik',         'baik',         'sangat_baik'],      # jarak dekat
    ],
    ('murah', 'lengkap'): [     # harga murah, fasilitas lengkap
        ['cukup',        'baik',         'baik'],             # jarak jauh
        ['baik',         'baik',         'sangat_baik'],      # jarak sedang
        ['sangat_baik',  'sangat_baik',  'sangat_baik'],      # jarak dekat
    ],
}
rules = [ctrl.Rule(jarak[J[i]] & keamanan[K[j]] & harga[h] & fasilitas[f], kelayakan[R[(h, f)][i][j]])
         for (h, f) in R for i in range(3) for j in range(3)]
print("=" * 74)
print(" FIS MAMDANI - KELAYAKAN KOS MAHASISWA (4 input, 1 output)")
print("=" * 74)
print(f"Jumlah aturan = {len(rules)}")

# 4) Simulasi
fis = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def hitung(d, a, h, f):                      # d=jarak, a=keamanan, h=harga, f=fasilitas
    fis.input['jarak'], fis.input['keamanan'] = d, a
    fis.input['harga'], fis.input['fasilitas'] = h, f
    fis.compute()
    return fis.output['kelayakan']

print("\n[4] Simulasi FIS")
print(f"Kelayakan kos (0.8 km, aman 80, Rp 700rb, fasilitas 70) = {hitung(0.8, 80, 700, 70):.2f}")

# 5) Verifikasi manual langkah demi langkah (input yang sama)
x = dict(jarak=0.8, keamanan=80, harga=700, fasilitas=70)
var = dict(jarak=jarak, keamanan=keamanan, harga=harga, fasilitas=fasilitas)
mu = {v: {n: float(fuzz.interp_membership(var[v].universe, var[v][n].mf, x[v]))
          for n in var[v].terms} for v in var}
print("\n[5] Verifikasi manual (input: 0.8 km, aman 80, Rp 700rb, fasilitas 70)")
print("  Fuzzifikasi (derajat keanggotaan > 0):")
for v in mu:
    print(f"    {v:9s}", ", ".join(f"{n} = {m:.2f}" for n, m in mu[v].items() if m > 0))

alpha = dict.fromkeys(OUT, 0.0)
for (h, f), M in R.items():
    for i in range(3):
        for j in range(3):
            w = min(mu['jarak'][J[i]], mu['keamanan'][K[j]], mu['harga'][h], mu['fasilitas'][f])  # AND = min
            alpha[M[i][j]] = max(alpha[M[i][j]], w)                                               # OR  = max
print("  Evaluasi aturan (AND = min, OR = max), alpha himpunan output:")
print("   ", ", ".join(f"{n} = {a:.2f}" for n, a in alpha.items() if a > 0))
z = kelayakan.universe
agg = np.zeros(len(z))
for n in OUT:
    agg = np.fmax(agg, np.fmin(alpha[n], kelayakan[n].mf))                                        # agregasi
print("  Defuzzifikasi:")
print(f"    centroid manual  = {fuzz.defuzz(z, agg, 'centroid'):.2f}   (skfuzzy: {hitung(0.8, 80, 700, 70):.2f})")
for m in ('mom', 'som', 'lom', 'bisector'):
    print(f"    {m:8s}         = {fuzz.defuzz(z, agg, m):.1f}")

# 6) Uji 4 kos contoh (urutan angka: jarak km, keamanan, harga ribu Rp, fasilitas)
contoh = {'Kos A (jarak dekat, aman tinggi, harga mahal, fasilitas lengkap)':  (0.3, 85, 1500, 90),
          'Kos B (jarak sedang, aman sedang, harga murah, fasilitas cukup)':   (1.5, 60, 500, 50),
          'Kos C (jarak jauh, aman sedang, harga murah, fasilitas kurang)':    (4.0, 40, 350, 25),
          'Kos D (jarak dekat, aman tinggi, harga sedang, fasilitas lengkap)': (0.5, 80, 600, 70)}
hasil = {n: hitung(*v) for n, v in contoh.items()}
print("\n[6] Uji 4 kos contoh (diurutkan dari skor tertinggi)")
for n, skor in sorted(hasil.items(), key=lambda t: -t[1]):
    print(f"  {n:68s} {skor:6.2f}")

# 7) Dataset: hitung seluruh kos di data_kos.csv -> peringkat -> hasil_kos.csv
df = pd.read_csv('data_kos.csv')
df['kelayakan'] = [round(hitung(r.jarak, r.keamanan, r.harga, r.fasilitas), 2) for r in df.itertuples()]
df['peringkat'] = df.kelayakan.rank(ascending=False, method='min').astype(int)
df = df.sort_values('peringkat')
df.to_csv('hasil_kos.csv', index=False)
print(f"\n[7] Peringkat {len(df)} kos dari data_kos.csv (tersimpan di hasil_kos.csv)")
print(df.to_string(index=False))

# 8) Gambar untuk makalah & slide: MF tiap variabel (5 file) dan control surface
os.makedirs('gambar', exist_ok=True)

def gambar_mf(ax, v):
    for n, t in v.terms.items():
        ax.plot(v.universe, t.mf, label=n, linewidth=2)
    ax.set_title(f'Fungsi keanggotaan: {v.label}'); ax.set_xlabel(v.label); ax.set_ylabel('derajat keanggotaan')
    ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(alpha=0.3)
    ax.legend(loc='upper center', ncol=len(v.terms), fontsize=8)

for v in (jarak, keamanan, harga, fasilitas, kelayakan):
    gambar_mf(plt.figure(figsize=(6, 3.2)).add_subplot(), v)
    plt.tight_layout(); plt.savefig(f'gambar/mf_{v.label}.png', dpi=300); plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 7))                       # 1 gambar, 4 layout (2x2)
for ax, v in zip(axes.flat, (jarak, keamanan, harga, fasilitas)):
    gambar_mf(ax, v)
plt.tight_layout(); plt.savefig('gambar/mf_gabungan.png', dpi=300); plt.close()

plt.figure(figsize=(6, 3.2))
for n in OUT:
    plt.plot(z, kelayakan[n].mf, '--', alpha=0.4, label=n)
plt.fill_between(z, agg, alpha=0.4, color='tab:blue', label='agregasi')
c = fuzz.defuzz(z, agg, 'centroid')
plt.axvline(c, color='r', label=f'centroid = {c:.2f}')
plt.title('Hasil agregasi dan centroid'); plt.xlabel('kelayakan'); plt.ylabel('derajat keanggotaan')
plt.ylim(0, 1.1); plt.grid(alpha=0.3); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig('gambar/agregasi_centroid.png', dpi=300); plt.close()

Jg, Hg = np.meshgrid(np.linspace(0, 5, 15), np.linspace(200, 2000, 15))      # keamanan=70, fasilitas=60 tetap
Z = np.array([[hitung(d, 70, h, 60) for d, h in zip(bd, bh)] for bd, bh in zip(Jg, Hg)])
ax = plt.figure(figsize=(7, 5)).add_subplot(projection='3d')
ax.plot_surface(Jg, Hg, Z, cmap='viridis')
ax.set_xlabel('Jarak (km)'); ax.set_ylabel('Harga (ribu Rp)'); ax.set_zlabel('Kelayakan')
plt.savefig('gambar/surface_jarak_harga.png', dpi=300); plt.close()
print("\n[8] Gambar tersimpan di folder gambar/ (5 grafik fungsi keanggotaan + 1 gabungan 2x2 + 1 control surface)")

# 9) Demo interaktif: ketik data kos, hasil tampil di output (Enter = ya, q = selesai)
def baca(teks, v):
    lo, hi = v.universe.min(), v.universe.max()
    while True:
        try:
            nilai = float(input(f"  {teks} ({lo:g}-{hi:g}) : ").replace(',', '.'))
            if lo <= nilai <= hi:
                return nilai
        except ValueError:
            pass
        print("   -> bukan angka atau di luar rentang, ulangi")

print("\n[9] Demo interaktif")
while input("\nNilai kos baru? (Enter = ya, q = selesai) ").strip().lower() != 'q':
    d, a = baca('Jarak ke kampus, km', jarak), baca('Keamanan, skor', keamanan)
    h, f = baca('Harga, ribu Rp', harga), baca('Fasilitas, skor', fasilitas)
    print(f"  >>> Skor kelayakan = {hitung(d, a, h, f):.2f} / 100")