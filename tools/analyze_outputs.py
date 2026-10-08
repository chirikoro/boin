import sys, numpy as np, soundfile as sf, librosa, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pairs = [("male_ja", "/Users/sakakibaratakashiyu/boin/out/samples/male_ja.aiff", "/Users/sakakibaratakashiyu/boin/out/converted/male_ja_sarasara.wav"),
         ("female_ja", "/Users/sakakibaratakashiyu/boin/out/samples/female_ja.aiff", "/Users/sakakibaratakashiyu/boin/out/converted/female_ja_sarasara.wav"),
         ("sample", "/Users/sakakibaratakashiyu/boin/out/samples/sample_sarasara.mp3", "/Users/sakakibaratakashiyu/boin/out/converted/sample_sarasara_sarasara.wav")]
fig, axes = plt.subplots(len(pairs), 2, figsize=(16, 4*len(pairs)))
for r,(name, a, b) in enumerate(pairs):
    for c,(label, path) in enumerate([("input", a), ("output", b)]):
        y, sr = librosa.load(path, sr=None, mono=True)
        dur = len(y)/sr
        peak = float(np.abs(y).max()); rms = float(np.sqrt(np.mean(y**2)))
        nan = int(np.isnan(y).sum())
        f0, vf, vp = librosa.pyin(y, fmin=60, fmax=600, sr=sr, frame_length=2048)
        f0v = f0[~np.isnan(f0)]
        med = float(np.median(f0v)) if len(f0v) else 0.0
        voiced = float(len(f0v))/max(1,len(f0))
        print(f"{name:10s} {label:6s} sr={sr} dur={dur:6.2f}s peak={peak:.3f} rms={rms:.4f} nan={nan} f0_median={med:6.1f}Hz voiced={voiced:.2f}")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
        ax = axes[r][c]
        librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set_ylim(0, 8000); ax.set_title(f"{name} {label} (sr={sr}, f0med={med:.0f}Hz)")
plt.tight_layout(); plt.savefig("spectrograms.png", dpi=70)
print("saved")
