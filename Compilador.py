"""
Compilador: pega el URL de un stream (VOD) y arma UN video largo
(por ejemplo 40 min) con los mejores momentos y subtítulos.

Es independiente de clipper.py y app.py. Corre en el puerto 5001.
Uso:  python3 compilador.py
"""
import os
import shutil
import subprocess
import textwrap
import threading
import time

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "compilaciones")
os.makedirs(WORK, exist_ok=True)

S = {"estado": "inactivo", "etapa": "", "pct": 0, "msg": "", "archivo": None}
cancelar = threading.Event()
proc = [None]


class Cancelado(Exception):
    pass


def st(**kw):
    S.update(kw)


def check():
    if cancelar.is_set():
        raise Cancelado()


def tam_carpeta(c):
    t = 0
    for r, _, fs in os.walk(c):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


def duracion(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True).stdout.strip()
    return float(out)


def descargar(url, carpeta, ini_min, fin_min):
    import yt_dlp
    parar = threading.Event()

    def vigilar():
        while not parar.is_set():
            st(msg=f"{tam_carpeta(carpeta) / 1e9:.2f} GB descargados")
            parar.wait(3)

    def hook(d):
        check()
        if d.get("status") == "downloading":
            fi, fc = d.get("fragment_index"), d.get("fragment_count")
            if fi and fc:
                st(pct=round(fi * 100 / fc, 1))

    opts = {
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(carpeta, "origen.%(ext)s"),
        "quiet": True, "noprogress": True,
        "progress_hooks": [hook],
    }
    if fin_min and fin_min > 0:
        opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(ini_min * 60, fin_min * 60)])
    threading.Thread(target=vigilar, daemon=True).start()
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([url])
    finally:
        parar.set()
    cand = [f for f in os.listdir(carpeta)
            if f.startswith("origen.") and not f.endswith((".part", ".ytdl"))]
    if not cand:
        raise RuntimeError("No se encontró el video descargado.")
    return os.path.join(carpeta, sorted(cand)[0])


def rms_por_segundo(path, total):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", "8000", "-f", "s16le", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    proc[0] = p
    vals, hechos = [], 0
    while True:
        check()
        b = p.stdout.read(8000 * 2 * 60)
        if not b:
            break
        a = np.frombuffer(b[:len(b) // 2 * 2], dtype=np.int16).astype(np.float32)
        n = len(a) // 8000
        if n:
            m = a[:n * 8000].reshape(n, 8000)
            vals.append(np.sqrt((m ** 2).mean(axis=1)))
            hechos += n
            st(pct=min(99, round(hechos * 100 / max(total, 1), 1)))
    p.wait()
    return np.concatenate(vals) if vals else np.zeros(0)


def elegir(rms, L, n):
    x = np.log1p(rms)
    T = len(x)
    if T <= L:
        return [(0, T)]
    c1 = np.concatenate([[0], np.cumsum(x)])
    c2 = np.concatenate([[0], np.cumsum(x * x)])
    k = T - L + 1
    mean = (c1[L:L + k] - c1[:k]) / L
    sd = np.sqrt(np.maximum((c2[L:L + k] - c2[:k]) / L - mean ** 2, 0))

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    score = z(mean) + 0.5 * z(sd)
    ocupado = np.zeros(T, bool)
    out = []
    for i in np.argsort(-score):
        if len(out) >= n:
            break
        if ocupado[max(0, i - 1):i + L + 1].any():
            continue
        ocupado[i:i + L] = True
        out.append((int(i), int(i + L)))
    return sorted(out)


def escribir_lista(path, origen, tramos):
    with open(path, "w") as f:
        for a, b in tramos:
            f.write(f"file '{origen}'\ninpoint {a}\noutpoint {b}\n")


def correr_ffmpeg(cmd, total, carpeta, log="ffmpeg.log"):
    full = ["ffmpeg", "-y", "-progress", "pipe:1", "-nostats"] + cmd
    with open(os.path.join(carpeta, log), "wb") as errf:
        p = subprocess.Popen(full, cwd=carpeta, stdout=subprocess.PIPE,
                             stderr=errf, text=True)
        proc[0] = p
        for line in p.stdout:
            if cancelar.is_set():
                p.kill()
                raise Cancelado()
            if line.startswith("out_time_ms="):
                try:
                    us = int(line.split("=")[1])
                    st(pct=min(99.9, round(us / 1e6 * 100 / total, 1)))
                except ValueError:
                    pass
        p.wait()
    if p.returncode != 0:
        with open(os.path.join(carpeta, log), errors="ignore") as f:
            raise RuntimeError("ffmpeg falló: " + f.read()[-300:])


def fmt_t(t):
    t = max(t, 0)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"


def bloques(texto, ini, fin):
    lin = textwrap.wrap(texto, 40)
    grupos = ["\n".join(lin[i:i + 2]) for i in range(0, len(lin), 2)]
    tot = sum(len(g) for g in grupos) or 1
    t0 = ini
    for g in grupos:
        d = (fin - ini) * len(g) / tot
        yield t0, t0 + d, g
        t0 += d


def transcribir(wav, modelo, total, srt):
    from faster_whisper import WhisperModel
    m = WhisperModel(modelo, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(wav, vad_filter=True, beam_size=1)
    k = 0
    with open(srt, "w", encoding="utf-8") as f:
        for s in segs:
            check()
            st(pct=min(99, round(s.end * 100 / max(total, 1), 1)))
            texto = s.text.strip()
            if not texto:
                continue
            for a, b, g in bloques(texto, s.start, s.end):
                k += 1
                f.write(f"{k}\n{fmt_t(a)} --> {fmt_t(b)}\n{g}\n\n")
    return k


def trabajo(url, ini_min, fin_min, minutos, L, modelo):
    carpeta = os.path.join(WORK, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(carpeta)
    ok = False
    try:
        st(estado="trabajando", archivo=None, pct=0, msg="")
        st(etapa="1/5 Descargando el video (720p)")
        origen = descargar(url, carpeta, ini_min, fin_min)

        st(etapa="2/5 Buscando los mejores momentos", pct=0, msg="")
        dur = duracion(origen)
        rms = rms_por_segundo(origen, dur)
        n = max(1, round(minutos * 60 / L))
        tramos = elegir(rms, L, n)
        total = sum(b - a for a, b in tramos)
        st(msg=f"{len(tramos)} momentos, {total / 60:.1f} min en total")
        lista = os.path.join(carpeta, "lista.txt")
        escribir_lista(lista, origen, tramos)

        st(etapa="3/5 Preparando el audio", pct=0)
        correr_ffmpeg(["-f", "concat", "-safe", "0", "-i", "lista.txt", "-vn",
                       "-ac", "1", "-ar", "16000", "audio.wav"], total, carpeta)

        st(etapa="4/5 Creando subtítulos con IA", pct=0)
        srt = os.path.join(carpeta, "subs.srt")
        cues = transcribir(os.path.join(carpeta, "audio.wav"), modelo, total, srt)

        st(etapa="5/5 Armando el video final (tarda)", pct=0)
        cmd = ["-f", "concat", "-safe", "0", "-i", "lista.txt"]
        if cues:
            estilo = "FontSize=14,Bold=1,Outline=2,Shadow=0,MarginV=25"
            cmd += ["-vf", f"subtitles=subs.srt:force_style='{estilo}'"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                "compilacion.mp4"]
        correr_ffmpeg(cmd, total, carpeta)

        os.remove(origen)
        os.remove(os.path.join(carpeta, "audio.wav"))
        ok = True
        st(estado="listo", etapa="Listo", pct=100, msg="Video terminado",
           archivo=os.path.basename(carpeta) + "/compilacion.mp4")
    except Cancelado:
        st(estado="cancelado", etapa="Cancelado", msg="Cancelado por ti")
    except Exception as e:
        st(estado="error", etapa="Error", msg=str(e)[-400:])
    finally:
        if not ok:
            shutil.rmtree(carpeta, ignore_errors=True)


@app.route("/")
def inicio():
    return PAGINA


@app.route("/iniciar", methods=["POST"])
def iniciar():
    if S["estado"] == "trabajando":
        return jsonify(error="Ya hay un trabajo en curso"), 409
    d = request.get_json(force=True)
    url = (d.get("url") or "").strip()
    if not url.startswith("http"):
        return jsonify(error="Pega un enlace válido"), 400
    cancelar.clear()
    args = (url, int(d.get("desde") or 0), int(d.get("hasta") or 0),
            int(d.get("minutos") or 40), int(d.get("seg") or 45),
            d.get("modelo") or "base")
    st(estado="trabajando", etapa="Iniciando...", pct=0, msg="", archivo=None)
    threading.Thread(target=trabajo, args=args, daemon=True).start()
    return jsonify(ok=True)


@app.route("/estado")
def estado():
    return jsonify(S)


@app.route("/cancelar", methods=["POST"])
def cancelar_ruta():
    cancelar.set()
    if proc[0] and proc[0].poll() is None:
        proc[0].kill()
    return jsonify(ok=True)


@app.route("/bajar/<path:nombre>")
def bajar(nombre):
    return send_from_directory(WORK, nombre, as_attachment=True)


PAGINA = """<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Compilador de mejores momentos</title>
<style>
body{background:#0f1015;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:16px}
.w{max-width:560px;margin:auto}
h1{font-size:22px;text-align:center}
.c{background:#181a22;border:1px solid #2a2d38;border-radius:12px;padding:16px;margin-top:14px}
label{display:block;font-size:13px;margin:10px 0 4px;font-weight:600}
input,select{width:100%;box-sizing:border-box;background:#0a0b0f;color:#fff;border:1px solid #2a2d38;border-radius:8px;padding:10px;font-size:15px}
.r{display:flex;gap:10px}.r>div{flex:1}
button{width:100%;margin-top:14px;padding:13px;border:0;border-radius:8px;font-size:16px;font-weight:700;cursor:pointer}
#go{background:#53fc18;color:#000}#no{background:#3a2020;color:#fff;display:none}
.bar{height:12px;background:#0a0b0f;border-radius:6px;overflow:hidden;margin-top:10px}
#fill{height:100%;width:0;background:#53fc18;transition:width .5s}
#res a{display:block;background:#53fc18;color:#000;text-align:center;padding:13px;border-radius:8px;font-weight:700;text-decoration:none;margin-top:10px}
small{color:#9aa}
</style></head><body><div class="w">
<h1>Compilador de mejores momentos</h1>
<div class="c">
<label>URL del stream (VOD de Kick, Twitch, YouTube)</label>
<input id="url" placeholder="https://kick.com/.../videos/...">
<div class="r"><div><label>Duración final (min)</label><input id="minutos" type="number" value="40"></div>
<div><label>Cada momento (seg)</label><input id="seg" type="number" value="45"></div></div>
<div class="r"><div><label>Analizar desde el minuto</label><input id="desde" type="number" value="0"></div>
<div><label>hasta el minuto (0 = todo)</label><input id="hasta" type="number" value="180"></div></div>
<label>Precisión de subtítulos</label>
<select id="modelo"><option value="tiny">Rápida (tiny)</option><option value="base" selected>Normal (base)</option><option value="small">Mejor, más lenta (small)</option></select>
<small>Para no llenar el disco, analiza máximo 3 horas por vez.</small>
<button id="go" onclick="ir()">Crear compilación</button>
<button id="no" onclick="fetch('/cancelar',{method:'POST'})">Cancelar</button>
</div>
<div class="c" id="est" style="display:none">
<div id="et" style="font-weight:700"></div>
<div class="bar"><div id="fill"></div></div>
<small id="ms"></small><div id="res"></div></div>
</div>
<script>
async function ir(){
 const b={url:url.value,minutos:minutos.value,seg:seg.value,desde:desde.value,hasta:hasta.value,modelo:modelo.value};
 const r=await fetch('/iniciar',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
 const j=await r.json(); if(j.error){alert(j.error);return}
 sondear();
}
async function sondear(){
 const s=await (await fetch('/estado')).json();
 const t=s.estado=='trabajando';
 est.style.display=s.estado=='inactivo'?'none':'block';
 go.style.display=t?'none':'block'; no.style.display=t?'block':'none';
 et.textContent=s.etapa; fill.style.width=s.pct+'%';
 ms.textContent=(s.pct?s.pct+'% · ':'')+(s.msg||'');
 res.innerHTML=s.archivo?'<a href="/bajar/'+s.archivo+'">Descargar video</a>':'';
 setTimeout(sondear,2000);
}
sondear();
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
