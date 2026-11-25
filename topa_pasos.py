# =============================================================================
#  topa_pasos.py — funciones de los cuadernos «paso a paso» de la tesis Ochroma pyramidale (topa)
#  Cada paso del cuaderno llama a UNA función de aquí. El cálculo es el mismo de los scripts con que se
#  entrenaron los modelos de la tesis (RF_fuentes, SVM_fuentes, XGBoost_word, YOLO, ViT y U-Net).
#  Autor: Elvis Venancio · Tesis Ochroma pyramidale · UNAS
# =============================================================================
import os, sys, time, shutil, warnings, json, subprocess
import numpy as np, pandas as pd, geopandas as gpd, rasterio, joblib
import matplotlib.pyplot as plt
from affine import Affine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Polygon as PolyMPL
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes
from rasterio.windows import Window
from scipy import ndimage as ndi
from shapely.geometry import shape, Polygon
warnings.filterwarnings("ignore")
T0 = time.time()
ALFA = 8                                    # banda de transparencia del DJI Mavic 3M (marca dónde hay imagen)
_VISTA = {}


# ───────────────────────────── colores en español ─────────────────────────────
import unicodedata
import matplotlib.colors as _mc
COLORES = {"verde": "#00C853", "verde claro": "#76FF03", "verde oscuro": "#1B5E20", "lima": "#C6FF00",
           "rojo": "#FF0000", "rojo vivo": "#FF1744", "rojo oscuro": "#B71C1C",
           "azul": "#1565C0", "azul claro": "#42A5F5", "azul oscuro": "#0D47A1", "celeste": "#00FFFF", "turquesa": "#1DE9B6",
           "amarillo": "#FFEB3B", "dorado": "#FFC400", "naranja": "#FF9100",
           "morado": "#9C27B0", "violeta": "#7C4DFF", "rosado": "#FF4081", "fucsia": "#F50057", "magenta": "#FF00FF",
           "marron": "#795548", "cafe": "#795548", "gris": "#9E9E9E", "gris claro": "#E0E0E0", "gris oscuro": "#424242",
           "negro": "#000000", "blanco": "#FFFFFF"}
SIN_COLOR = {"ninguno", "ninguna", "sin color", "sin relleno", "transparente", "nada"}
PALETAS = {"amarillo a verde": "YlGn", "rojo a verde": "RdYlGn", "verdes": "Greens", "rojos": "Reds", "azules": "Blues",
           "naranjas": "Oranges", "morados": "Purples", "grises": "Greys", "fuego": "hot", "arcoiris": "turbo", "arco iris": "turbo"}
POSICIONES = {"arriba derecha": "upper right", "arriba izquierda": "upper left", "abajo derecha": "lower right", "abajo izquierda": "lower left",
              "arriba": "upper center", "abajo": "lower center", "derecha": "center right", "izquierda": "center left", "centro": "center",
              "automatico": "best", "donde quepa": "best"}


def _normal(t):
    """Minúsculas, sin tildes y sin «a la»: «Arriba a la Derecha» → «arriba derecha»."""
    t = "".join(ch for ch in unicodedata.normalize("NFD", str(t).strip().lower()) if unicodedata.category(ch) != "Mn")
    return " ".join(t.replace(" a la ", " ").replace(" al ", " ").split())


def _col(c):
    """Color escrito en español (o en inglés) → color del dibujo. «ninguno» = sin color."""
    if c is None: return c
    t = _normal(c)
    return "none" if t in SIN_COLOR else COLORES.get(t, c)


def _pal(p): return PALETAS.get(_normal(p), p)


def _pos(p): return POSICIONES.get(_normal(p), p)


_mc.get_named_colors_mapping().update({**COLORES, "marrón": COLORES["marron"], "café": COLORES["cafe"]})   # matplotlib entiende «verde», «rojo»… en cualquier parte


def ver_colores():
    """Muestra los colores que puedes escribir en las celdas (también valen en inglés: green, red, black…)."""
    n = list(COLORES); f = (len(n) + 3) // 4
    fig, ax = plt.subplots(figsize=(12, 0.55 * f + 0.8))
    for i, k in enumerate(n):
        x, y = (i % 4) * 3, -(i // 4)
        ax.add_patch(Rectangle((x, y - 0.4), 0.8, 0.8, facecolor=COLORES[k], edgecolor="black", lw=0.6)); ax.text(x + 1.0, y, {"marron": "marrón", "cafe": "café"}.get(k, k), va="center", fontsize=12)
    ax.set_xlim(-0.2, 12); ax.set_ylim(-f + 0.4, 0.6); ax.axis("off")
    ax.set_title("Colores que puedes escribir  ·  «ninguno» = sin relleno", fontsize=13); plt.show()


# ───────────────────────────── datos ─────────────────────────────
def _orto_rapido(TIF):
    """Ruta del ortomosaico en el disco de Colab: lo copia una vez por sesión, porque leerlo desde Drive es muy lento."""
    tam = os.path.getsize(TIF); local = "/content/" + os.path.basename(TIF)
    ya = os.path.exists(local) and os.path.getsize(local) == tam
    if not ya and "google.colab" in sys.modules and tam < shutil.disk_usage("/content").free - 5e9:     # solo en Colab
        print(f"Copiando el ortomosaico a Colab ({tam / 1e9:.1f} GB, una vez por sesión)…", flush=True)
        try: shutil.copy(TIF, local); ya = os.path.getsize(local) == tam
        except OSError: pass
    return local if ya else TIF


def abrir_orto(TIF):
    """Abre el ortomosaico. Si cabe, primero lo copia al disco de Colab (leer desde Drive es lento)."""
    src = rasterio.open(_orto_rapido(TIF))
    print(f"Ortomosaico: {os.path.basename(TIF)} · {src.width} × {src.height} px · {src.res[0] * 100:.2f} cm por píxel · {src.count} bandas")
    return src


def leer_poligonos(SHP, src, CAMPO, TOPA):
    """Lee tus polígonos y marca cuáles son topa (cl = 1) y cuáles otras especies (cl = 2)."""
    shp = gpd.read_file(SHP); shp = shp.set_crs(src.crs) if shp.crs is None else shp.to_crs(src.crs)
    assert CAMPO in shp.columns, f"El shapefile no tiene el campo {CAMPO}. Tiene: {list(shp.columns)}"
    shp["geometry"] = shp.geometry.buffer(0); shp = shp[~shp.geometry.is_empty].reset_index(drop=True)
    shp["cl"] = np.where(shp[CAMPO].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True) == str(TOPA), 1, 2)
    assert shp.cl.nunique() == 2, f"Se necesitan topa y otras especies. Revisa CAMPO y TOPA: {shp[CAMPO].unique()[:10]}"
    print(f"Polígonos: {len(shp)} · topa {int((shp.cl == 1).sum())} · otras especies {int((shp.cl == 2).sum())}")
    return shp


# ───────────────────────────── figuras ─────────────────────────────
def _vista(src):
    """Vista del ortomosaico en color natural (estiramiento 2–98 %), para todas las figuras."""
    if src.name not in _VISTA:
        k = max(src.width, src.height) / 1500; h, w = int(src.height / k), int(src.width / k)
        rgb = src.read([1, 2, 3], out_shape=(3, h, w)).astype(np.float32).transpose(1, 2, 0)
        mv = src.read(ALFA, out_shape=(h, w)) > 0 if src.count >= ALFA else rgb.sum(2) > 0
        lo, hi = np.percentile(rgb[mv], [2, 98], axis=0); img = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1); img[~mv] = 1
        b = src.bounds; _VISTA[src.name] = (img, (b.left, b.right, b.bottom, b.top))
    return _VISTA[src.name]


def _vista_zona(src, zona, lado_px=1500):
    """El acercamiento leído con más detalle (no la vista reducida de todo el ortomosaico), con su propio estiramiento 2–98 %."""
    from rasterio.windows import from_bounds
    w = from_bounds(*zona, transform=src.transform); k = max(1.0, max(w.width, w.height) / lado_px)
    h_, w_ = max(1, int(w.height / k)), max(1, int(w.width / k))
    rgb = src.read([1, 2, 3], window=w, out_shape=(3, h_, w_), boundless=True, fill_value=0).astype(np.float32).transpose(1, 2, 0)
    mv = rgb.sum(2) > 0
    if not mv.any(): return np.ones((h_, w_, 3)), (zona[0], zona[2], zona[1], zona[3])
    lo, hi = np.percentile(rgb[mv], [2, 98], axis=0); img = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1); img[~mv] = 1
    return img, (zona[0], zona[2], zona[1], zona[3])


def estilo_figuras(tam_letra=11, fuente="DejaVu Sans"):
    """Tamaño de letra y tipo de letra de TODAS las figuras del cuaderno (ejes, números, textos)."""
    plt.rcParams.update({"font.size": tam_letra, "font.family": fuente})


def capa(gdf, relleno, borde, etiqueta, grosor=0.6, transparencia=1.0, contar=True):
    """Una capa para ver_mapa: polígonos, color de relleno ("ninguno" = sin relleno), color del borde, etiqueta de la leyenda,
    grosor del borde y transparencia (0 = invisible, 1 = opaco). contar=True agrega a la etiqueta cuántos polígonos hay."""
    return (gdf, _col(relleno), _col(borde), grosor, transparencia, f"{etiqueta} ({len(gdf)})" if contar and "(" not in str(etiqueta) else etiqueta)


def ver_mapa(src, capas, titulo, zona=None, linea=None, tam_titulo=13, tam_leyenda=10, pos_leyenda="arriba derecha", leyenda=True,
             color_linea="rojo", grosor_linea=2, texto_linea="línea de corte 70/30", color_recuadro="rojo", ancho=None, alto=None):
    """Dibuja el ortomosaico con las capas encima. Con zona: además un acercamiento. Con linea = (eje, valor): la línea de corte.
    Todo se puede cambiar: título y su tamaño, leyenda (tamaño, lugar, mostrarla o no), color y grosor de la línea, tamaño de la figura."""
    color_linea, color_recuadro, pos_leyenda = _col(color_linea), _col(color_recuadro), _pos(pos_leyenda)
    img, EXT = _vista(src)
    tam = (ancho or (11 if zona is None else 20), alto or (12 if zona is None else 11))
    fig, axs = plt.subplots(1, 1 if zona is None else 2, figsize=tam, squeeze=False); axs = axs[0]; hs = []
    for k, ax in enumerate(axs):
        chico = zona is not None and max(zona[2] - zona[0], zona[3] - zona[1]) < 600 * (EXT[1] - EXT[0]) / img.shape[1]
        if k == 1 and chico: zi, ze = _vista_zona(src, zona); ax.imshow(zi, extent=ze)      # acercamiento chico: leído con más detalle
        else: ax.imshow(img, extent=EXT)
        for g, cara, borde, gros, alfa, etq in capas:
            if len(g): g.plot(ax=ax, facecolor=cara, edgecolor=borde, linewidth=gros, alpha=alfa)
            if k == 0: hs.append(Patch(facecolor="white" if cara == "none" else cara, edgecolor=borde, alpha=alfa, label=etq))
        if linea: (ax.axvline if linea[0] == "x" else ax.axhline)(linea[1], color=color_linea, ls="--", lw=grosor_linea)
        ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_xticks([]); ax.set_yticks([])
    if linea: hs.append(Line2D([0], [0], color=color_linea, ls="--", lw=grosor_linea, label=texto_linea))
    if zona is not None:
        axs[0].add_patch(Rectangle((zona[0], zona[1]), zona[2] - zona[0], zona[3] - zona[1], fill=False, edgecolor=color_recuadro, lw=2.5))
        axs[0].set_title("Todo el ortomosaico (recuadro = acercamiento)", fontsize=tam_leyenda + 1); axs[1].set_xlim(zona[0], zona[2]); axs[1].set_ylim(zona[1], zona[3])
        axs[1].set_title("Acercamiento", fontsize=tam_leyenda + 1)
    if leyenda: axs[-1].legend(handles=hs, loc=pos_leyenda, framealpha=0.95, fontsize=tam_leyenda)
    fig.suptitle(titulo, fontsize=tam_titulo); plt.tight_layout(); plt.show()


def zona_topas(shp, margen=15):
    """Recuadro alrededor de tus topas (con margen en metros), para los acercamientos."""
    return shp[shp.cl == 1].total_bounds + np.array([-margen, -margen, margen, margen])


def zona_cerca(shp, lado=40):
    """Acercamiento chico: un cuadrado de «lado» metros en el centro de tus topas (para ver bien muchas copas pequeñas)."""
    c = shp[shp.cl == 1].geometry.centroid; x, y = float(np.median(c.x)), float(np.median(c.y)); m = lado / 2
    return np.array([x - m, y - m, x + m, y + m])


# ───────────────────────────── copas (watershed) ─────────────────────────────
def copas_watershed(src, PIX_WS=0.05, WS_NDVI=0.25, WS_DIST_M=1.5, WS_SIGMA_M=0.15, WS_HUECO_M2=2, AREA_MIN_M2=1, AREA_MAX_M2=400, NIR=7, ROJO=5):
    """Separa las copas: deja lo verde (NDVI), busca el centro de cada copa y la hace crecer hasta tocar a sus vecinas (watershed)."""
    from skimage.filters import gaussian
    from skimage.segmentation import watershed
    res = src.res[0]; vmax = float(np.iinfo(src.dtypes[0]).max) if np.issubdtype(np.dtype(src.dtypes[0]), np.integer) else 1.0
    ds = max(1, round(PIX_WS / res))
    while (src.width // ds) * (src.height // ds) * 6 * 4 > 3e9: ds += 1
    W4, H4 = src.width // ds, src.height // ds; TRD = src.transform * Affine.scale(ds); px = res * ds
    leer = lambda b: src.read(b, window=Window(0, 0, W4 * ds, H4 * ds), out_shape=(H4, W4), resampling=Resampling.average).astype(np.float32)
    nir, rr = leer(NIR), leer(ROJO)
    dat = (leer(ALFA) > 0) & ~((nir >= 0.98 * vmax) & (rr >= 0.98 * vmax)); dat = ndi.binary_erosion(dat, iterations=3)
    ndvi = (nir - rr) / (nir + rr + 1e-6); sup = nir * (1 + ndvi); veg = (ndvi >= WS_NDVI) & dat
    if WS_SIGMA_M / px > 0.3: sup = gaussian(sup, sigma=WS_SIGMA_M / px, preserve_range=True)
    veg = ndi.binary_closing(veg, iterations=2)
    hue, _ = ndi.label(~veg); chico = np.bincount(hue.ravel()) < WS_HUECO_M2 / px ** 2; chico[0] = False; veg = (veg | chico[hue]) & dat
    md = max(1, int(round(WS_DIST_M / px))); sm = np.where(veg, sup, np.nanmin(sup) - 1).astype(np.float32)
    marcas, _ = ndi.label((sm == ndi.maximum_filter(sm, size=2 * md + 1)) & veg); reg = watershed(-sm, marcas, mask=veg)
    COP = gpd.GeoDataFrame(pd.DataFrame([{"id": int(v), "geometry": shape(g)} for g, v in shapes(reg.astype(np.int32), mask=reg > 0, transform=TRD)],
                                        columns=["id", "geometry"]), geometry="geometry", crs=src.crs)
    COP = COP.assign(a=COP.area).sort_values("a").drop_duplicates("id", keep="last").drop(columns="a")
    COP["geometry"] = COP.geometry.apply(lambda g: Polygon(g.exterior)); COP["area_m2"] = COP.area.round(1)
    COP = COP[(COP.area_m2 >= AREA_MIN_M2) & ((AREA_MAX_M2 <= 0) | (COP.area_m2 <= AREA_MAX_M2))].reset_index(drop=True)
    receta = dict(WS_NDVI=WS_NDVI, WS_DIST_M=WS_DIST_M, WS_SIGMA_M=WS_SIGMA_M, WS_HUECO_M2=WS_HUECO_M2, AREA_MIN_M2=AREA_MIN_M2, AREA_MAX_M2=AREA_MAX_M2)
    print(f"{len(COP)} copas de {AREA_MIN_M2} a {AREA_MAX_M2} m² · {time.time() - T0:.0f} s")
    return COP, receta


def etiquetar_copas(COP, shp, SOLAPE_MIN_M2=1):
    """Clase de cada copa = la del polígono con el que más se solapa (los solapes < 1 m² no cuentan). Sin polígono = no topa."""
    ic, ip = shp.sindex.query(COP.geometry, predicate="intersects")
    PARES = pd.DataFrame({"copa": ic, "cl": shp.cl.values[ip]})
    PARES["a"] = gpd.GeoSeries(COP.geometry.values[ic], crs=COP.crs).intersection(gpd.GeoSeries(shp.geometry.values[ip], crs=COP.crs), align=False).area.values
    AREA = PARES[PARES.a >= SOLAPE_MIN_M2].groupby(["copa", "cl"]).a.sum().unstack(fill_value=0).reindex(columns=[1, 2], fill_value=0)
    ETQ = COP.loc[AREA.index].assign(y=(AREA[1] > AREA[2]).astype(int).values); ETQ["tipo"] = np.where(ETQ.y == 1, "topa", "otras")
    FON = COP.drop(index=np.unique(ic)).assign(y=0, tipo="fondo")
    GEO = gpd.GeoDataFrame(pd.concat([ETQ, FON])[["geometry", "area_m2", "y", "tipo"]], geometry="geometry", crs=COP.crs)
    print(f"topa {(GEO.tipo == 'topa').sum()} · otras especies {(GEO.tipo == 'otras').sum()} · sin polígono (= no topa) {(GEO.tipo == 'fondo').sum()} · "
          f"sin usar (solo solapes < {SOLAPE_MIN_M2} m²) {len(COP) - len(GEO)}")
    return GEO


def capas_etiquetas(GEO, topa="verde", otras="rojo", fondo="gris", borde="negro", grosor=0.2, transparencia=0.9):
    """Capas de la clase de cada copa: color de relleno de topa, otras y sin polígono; color y grosor del borde; transparencia."""
    return [capa(GEO[GEO.tipo == "fondo"], fondo, borde, "sin polígono = no topa", grosor / 2, transparencia * 0.4),
            capa(GEO[GEO.tipo == "otras"], otras, borde, "otras especies", grosor, transparencia * 0.9),
            capa(GEO[GEO.tipo == "topa"], topa, borde, "topa", grosor, transparencia)]


# ───────────────────────────── rasgos de cada copa ─────────────────────────────
BANDAS = {"B1": 1, "B2": 2, "B3": 3, "G": 4, "R": 5, "RE": 6, "NIR": 7}
INDICES = ["NDVI", "NDRE", "GNDVI", "SAVI", "MCARI", "CIRE", "PSRI", "GRNDVI"]


def calcular_rasgos(src, GEO, GLCM_NIVELES=32):
    """Para cada copa: mediana y desviación de 7 bandas y 8 índices + 5 medidas de textura del infrarrojo (GLCM)."""
    from skimage.feature import graycomatrix, graycoprops
    e = 1e-6; nd = lambda a, b: (a - b) / np.where(a + b == 0, np.inf, a + b)
    IDXF = {"NDVI": lambda b: nd(b["NIR"], b["R"]), "NDRE": lambda b: nd(b["NIR"], b["RE"]), "GNDVI": lambda b: nd(b["NIR"], b["G"]),
            "SAVI": lambda b: 1.5 * (b["NIR"] - b["R"]) / (b["NIR"] + b["R"] + 0.5), "MCARI": lambda b: ((b["RE"] - b["R"]) - 0.2 * (b["RE"] - b["G"])) * (b["RE"] / (b["R"] + e)),
            "CIRE": lambda b: b["NIR"] / (b["RE"] + e) - 1, "PSRI": lambda b: (b["R"] - b["G"]) / (b["RE"] + e), "GRNDVI": lambda b: nd(b["NIR"], b["G"] + b["R"])}
    PROPS = ["contrast", "dissimilarity", "homogeneity", "correlation", "ASM"]; BL = list(BANDAS.values())

    def una_copa(g):
        inv = ~src.transform; (ca, ra), (cb, rb) = inv * (g.bounds[0], g.bounds[3]), inv * (g.bounds[2], g.bounds[1])
        c0, r0 = int(np.floor(min(ca, cb))) - 2, int(np.floor(min(ra, rb))) - 2; w, h = int(np.ceil(max(ca, cb))) + 3 - c0, int(np.ceil(max(ra, rb))) + 3 - r0
        a = src.read(BL, window=Window(c0, r0, w, h), boundless=True, fill_value=0).astype(np.float32)
        m = geometry_mask([g], out_shape=(h, w), transform=src.transform * Affine.translation(c0, r0), invert=True)
        if m.sum() < 10: return None
        b = {k: a[i] for i, k in enumerate(BANDAS)}; f = {}
        for k in list(BANDAS) + INDICES:
            v = b[k] if k in BANDAS else IDXF[k](b); v = v[m]; v = v[np.isfinite(v)]
            f[f"{k}_med"], f[f"{k}_std"] = (float(np.median(v)), float(np.std(v))) if v.size else (np.nan, np.nan)
        p = b["NIR"][m]; qi = np.clip((b["NIR"] - p.min()) / max(np.ptp(p), 1e-6) * (GLCM_NIVELES - 1), 0, GLCM_NIVELES - 1).astype(np.uint8)
        gl = graycomatrix(np.where(m, qi, GLCM_NIVELES).astype(np.uint8), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=GLCM_NIVELES + 1,
                          symmetric=True, normed=False)[:GLCM_NIVELES, :GLCM_NIVELES].astype(np.float64).sum(axis=3, keepdims=True)
        gl /= max(gl.sum(), 1.0)
        for pr in PROPS: f[f"GLCM_{pr}_NIR"] = float(graycoprops(gl, pr)[0, 0])
        return f

    filas = {}
    for n_, (i, g) in enumerate(GEO.geometry.items()):
        f = una_copa(g)
        if f is not None: filas[i] = f
        if (n_ + 1) % 200 == 0: print(f"   rasgos: {n_ + 1}/{len(GEO)} copas", end="\r", flush=True)
    DATOS = pd.DataFrame(filas).T.astype(float).replace([np.inf, -np.inf], np.nan); RAS0 = list(DATOS.columns); n0 = len(DATOS); DATOS = DATOS.dropna()
    DATOS["y"], DATOS["tipo"] = GEO.loc[DATOS.index, "y"].astype(int), GEO.loc[DATOS.index, "tipo"]
    print(f"\n{len(DATOS)} copas con {len(RAS0)} rasgos ({n0 - len(DATOS)} excluidas por datos incompletos) · {time.time() - T0:.0f} s")
    return DATOS, RAS0


def ver_rasgo(src, GEO, DATOS, rasgo, zona=None, titulo=None, tam_titulo=13, colores="amarillo a verde", borde="negro", grosor=0.1, transparencia=1.0, ancho=11, alto=11):
    """Pinta cada copa con el valor de un rasgo. colores = paleta: «amarillo a verde», «rojo a verde», «verdes», «azules», «rojos», «grises», «fuego», «arcoiris»."""
    img, EXT = _vista(src); gv = GEO.loc[DATOS.index].assign(v=DATOS[rasgo]); v0, v1 = float(gv.v.quantile(0.02)), float(gv.v.quantile(0.98))
    fig, ax = plt.subplots(figsize=(ancho, alto)); ax.imshow(img, extent=EXT)
    gv.plot(ax=ax, column="v", cmap=_pal(colores), vmin=v0, vmax=v1, edgecolor=_col(borde), linewidth=grosor, alpha=transparencia)
    if zona is not None: ax.set_xlim(zona[0], zona[2]); ax.set_ylim(zona[1], zona[3])
    fig.colorbar(plt.cm.ScalarMappable(cmap=_pal(colores), norm=plt.Normalize(v0, v1)), ax=ax, shrink=0.6, label=rasgo)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titulo or f"{rasgo} de cada copa", fontsize=tam_titulo); plt.show()


# ───────────────────────────── división 70/30 ─────────────────────────────
def corte_espacial(shp, GEO, DATOS, PRUEBA=0.30):
    """Línea de corte: el 70 % de tus topas queda a un lado (aprende) y el 30 % al otro (examen). Cada copa va al lado de su centro."""
    t_ = shp[shp.cl == 1].geometry.centroid; EJE = "x" if np.ptp(t_.x) >= np.ptp(t_.y) else "y"
    SV = float(np.percentile(t_.x if EJE == "x" else t_.y, (1 - PRUEBA) * 100))
    c_ = GEO.loc[DATOS.index].geometry.centroid; es_examen = (c_.x if EJE == "x" else c_.y).values > SV
    TR, TE = DATOS[~es_examen], DATOS[es_examen]
    assert TR.y.nunique() == 2 and TE.y.nunique() == 2, "Un lado del corte quedó sin topa o sin no topa: revisa el SHP"
    print(f"Corte en el eje {EJE.upper()} = {SV:.1f} m\n" + pd.crosstab(DATOS.tipo, np.where(es_examen, "30 % examen", "70 % aprende")).rename_axis(None, axis=1).rename_axis(None).to_string())
    return TR, TE, es_examen, EJE, SV


def capas_corte(GEO, TR, TE, aprende="azul", examen="naranja", borde="negro", grosor=0.2, transparencia_no_topa=0.3, transparencia_topa=1.0):
    """Capas del 70/30: color de lo que aprende y del examen, borde, grosor y transparencia (no topa más tenue que topa)."""
    GTR, GTE = GEO.loc[TR.index], GEO.loc[TE.index]
    return [capa(GTR[GTR.y == 0], aprende, borde, "70 % aprende: no topa", grosor / 2, transparencia_no_topa),
            capa(GTE[GTE.y == 0], examen, borde, "30 % examen: no topa", grosor / 2, transparencia_no_topa),
            capa(GTR[GTR.y == 1], aprende, borde, "70 % aprende: topa", grosor, transparencia_topa),
            capa(GTE[GTE.y == 1], examen, borde, "30 % examen: topa", grosor, transparencia_topa)]


# ───────────────────────────── selección de rasgos ─────────────────────────────
def seleccionar_rasgos(TR, RAS0, ALFA_FDR=0.05, RHO_MAX=0.80, QUITAR=()):
    """Solo con el 70 %: quedan los rasgos que separan topa de no topa (Kruskal-Wallis + Benjamini-Hochberg) y, de dos repetidos (Spearman), el que más separa."""
    from scipy.stats import kruskal
    RAS = [r for r in RAS0 if r not in QUITAR]; H, P_ = {}, {}
    for r in RAS:
        if TR[r].nunique() > 1: H[r], P_[r] = kruskal(TR.loc[TR.y == 0, r], TR.loc[TR.y == 1, r])
    p_ = pd.Series(P_).sort_values(); q_ = (p_ * len(p_) / np.arange(1, len(p_) + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    SEP = [r for r in q_.index if q_[r] < ALFA_FDR]; rho = TR[SEP].corr(method="spearman").abs(); SEL = []
    for r in sorted(SEP, key=H.get, reverse=True):
        if all(rho.loc[r, s] < RHO_MAX for s in SEL): SEL.append(r)
    assert SEL, "Ningún rasgo pasó la selección"
    print(f"{len(RAS)} rasgos → {len(SEP)} separan las clases → {len(SEL)} sin repetidos: " + ", ".join(SEL))
    T = pd.DataFrame({"H": pd.Series(H), "q": q_}).reindex(RAS); T["decisión"] = np.where(T.index.isin(SEL), "entra", np.where(T.index.isin(SEP), "repite a otro", "no separa"))
    T = T.sort_values("H", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 4.5)); ax.bar(T.index, T.H.fillna(0), color=T["decisión"].map({"entra": "#2E7D32", "repite a otro": "#FF9100", "no separa": "#BDBDBD"}))
    ax.tick_params(axis="x", rotation=90); ax.set_ylabel("cuánto separa (H de Kruskal-Wallis)")
    ax.legend(handles=[Patch(color=c, label=t) for t, c in (("entra", "#2E7D32"), ("repite a otro", "#FF9100"), ("no separa", "#BDBDBD"))]); plt.tight_layout(); plt.show()
    return SEL


# ───────────────────────────── Random Forest ─────────────────────────────
def buscar_mtry(TR, SEL, ARBOLES=1000, PLIEGUES=5, REPETICIONES=5, SEMILLA=42):
    """Prueba cuántos rasgos mira cada pregunta de los árboles (mtry, de 1 al total) con validación cruzada 5 × 5 por Kappa."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import cohen_kappa_score, make_scorer
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
    print(f"Buscando mtry entre 1 y {len(SEL)} con {PLIEGUES} × {REPETICIONES} pliegues y {ARBOLES} árboles (puede tardar)…", flush=True)
    BUS = GridSearchCV(RandomForestClassifier(n_estimators=ARBOLES, random_state=SEMILLA), {"max_features": list(range(1, len(SEL) + 1))}, scoring=make_scorer(cohen_kappa_score),
                       cv=RepeatedStratifiedKFold(n_splits=PLIEGUES, n_repeats=REPETICIONES, random_state=SEMILLA), n_jobs=-1, refit=False).fit(TR[SEL], TR.y)
    MTRY, K_CV, K_SD = BUS.best_params_["max_features"], BUS.best_score_, BUS.cv_results_["std_test_score"][BUS.best_index_]
    CV = pd.DataFrame(BUS.cv_results_); m_ = CV.param_max_features.astype(int).values
    fig, ax = plt.subplots(figsize=(9, 4)); ax.errorbar(m_, CV.mean_test_score, yerr=CV.std_test_score, marker="o", capsize=3, color="#1565C0")
    ax.scatter([MTRY], [K_CV], s=250, marker="*", color="#F9A825", zorder=5, label=f"elegido: mtry = {MTRY}"); ax.set_xlabel("mtry"); ax.set_ylabel("Kappa (validación 5 × 5)")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
    print(f"mtry = {MTRY} · Kappa en validación {K_CV:.3f} ± {K_SD:.3f}")
    return MTRY, (float(K_CV), float(K_SD))


def entrenar_rf(TR, SEL, ARBOLES, MTRY, SEMILLA=42):
    """Entrena el bosque UNA sola vez con el 70 % y mide cuánto usó cada rasgo (importancia por permutación)."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=ARBOLES, max_features=MTRY, n_jobs=-1, random_state=SEMILLA).fit(TR[SEL], TR.y)
    return rf, importancia(rf, TR, SEL, SEMILLA)


def importancia(modelo, TR, SEL, SEMILLA=42):
    from sklearn.inspection import permutation_importance
    PI = permutation_importance(modelo, TR[SEL], TR.y, n_repeats=10, scoring="accuracy", random_state=SEMILLA)
    IMP = pd.Series(PI.importances_mean, index=SEL).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(IMP) + 1.4)); ax.barh(IMP.index[::-1], IMP.values[::-1], color="#2E7D32")
    ax.set_xlabel("cuánto cae el acierto al desordenar el rasgo"); ax.set_title("Rasgos que más usó el modelo"); plt.tight_layout(); plt.show()
    return IMP


# ───────────────────────────── copas del RF (SVM y XGBoost rinden el mismo examen) ─────────────────────────────
def cargar_copas_rf(BASE_RF, src):
    """Lee del modelo del RF las MISMAS copas con sus rasgos, clases y corte 70/30, y rehace sus contornos (el mismo watershed) para dibujarlas."""
    B = joblib.load(BASE_RF); DATOS = B["muestras"].copy(); DATOS["y"] = DATOS["y"].astype(int)
    es_examen = (DATOS.grupo == "test").values; TR, TE = DATOS[~es_examen], DATOS[es_examen]
    EJE, SV, WS = B["corte"]["eje"], B["corte"]["valor"], B["ws"]
    COP, _ = copas_watershed(src, PIX_WS=B["receta"]["ws_pix"], **WS)
    assert DATOS.index.isin(COP.index).all(), "Las copas no coinciden con las del RF: revisa ORTO (debe ser el mismo ortomosaico del RF)."
    GEO = gpd.GeoDataFrame(COP.loc[DATOS.index, ["geometry", "area_m2"]].assign(y=DATOS.y.values, tipo=DATOS.tipo.values), geometry="geometry", crs=src.crs)
    c_ = GEO.geometry.centroid; assert (((c_.x if EJE == "x" else c_.y).values > SV) == es_examen).all(), "El corte no coincide con el del RF: revisa ORTO."
    print(f"{len(DATOS)} copas del RF · corte en el eje {EJE.upper()} = {SV:.1f} m\n"
          + pd.crosstab(DATOS.tipo, np.where(es_examen, "30 % examen", "70 % aprende")).rename_axis(None, axis=1).rename_axis(None).to_string())
    return B, DATOS, TR, TE, GEO, es_examen, EJE, SV, WS


# ───────────────────────────── SVM ─────────────────────────────
def _mapa_calor(G, titulo):
    R = pd.DataFrame(G.cv_results_["params"]).assign(k=G.cv_results_["mean_test_score"], lc=lambda d: np.log2(d.svc__C.astype(float)), lg=lambda d: np.log2(d.svc__gamma.astype(float)))
    M = R.pivot_table(index="lg", columns="lc", values="k").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.5)); im = ax.imshow(M.values, cmap="Blues", aspect="auto"); plt.colorbar(im, ax=ax, label="Kappa en validación")
    pc, pg = max(1, M.shape[1] // 8), max(1, M.shape[0] // 8)
    ax.set_xticks(range(0, M.shape[1], pc), [f"{v:g}" for v in M.columns[::pc]]); ax.set_yticks(range(0, M.shape[0], pg), [f"{v:g}" for v in M.index[::pg]])
    bc, bg = np.log2(float(G.best_params_["svc__C"])), np.log2(float(G.best_params_["svc__gamma"])); ax.plot(list(M.columns).index(bc), list(M.index).index(bg), marker="*", color="#D50000", ms=18)
    ax.set_xlabel("log2(C)  ·  C grande = frontera más ajustada"); ax.set_ylabel("log2(gamma)  ·  gamma grande = cada copa influye más cerca"); ax.set_title(f"{titulo} · ★ = elegido"); plt.tight_layout(); plt.show()


def grilla_gruesa(TR, SEL, C_GRUESO, G_GRUESO, SUB_GRUESA=1500, PLIEGUES=5, REPETICIONES=5, SEMILLA=42):
    """Prueba pares de C y gamma muy separados (potencias de 2) con una parte de las copas, para ubicar la zona buena."""
    from sklearn.metrics import cohen_kappa_score, make_scorer
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    i_g = TR.index if len(TR) <= SUB_GRUESA else train_test_split(TR.index, train_size=SUB_GRUESA, stratify=TR.y, random_state=SEMILLA)[0]
    print(f"Grilla gruesa con {len(i_g)} copas: {len(C_GRUESO) * len(G_GRUESO)} pares × {PLIEGUES * REPETICIONES} ajustes (puede tardar)…", flush=True)
    G1 = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf", cache_size=1000)), {"svc__C": C_GRUESO, "svc__gamma": G_GRUESO}, scoring=make_scorer(cohen_kappa_score),
                      cv=RepeatedStratifiedKFold(n_splits=PLIEGUES, n_repeats=REPETICIONES, random_state=SEMILLA), n_jobs=-1, refit=False).fit(TR.loc[i_g, SEL], TR.loc[i_g, "y"])
    lc, lg = np.log2(G1.best_params_["svc__C"]), np.log2(G1.best_params_["svc__gamma"]); _mapa_calor(G1, "Grilla gruesa")
    print(f"Mejor zona: log2(C) = {lc:.0f} · log2(gamma) = {lg:.0f} · Kappa {G1.best_score_:.3f} · {time.time() - T0:.0f} s")
    return lc, lg


def grilla_fina(TR, SEL, lc, lg, RANGO_FINO=2, PASO_FINO=0.25, PLIEGUES=5, REPETICIONES=5, SEMILLA=42):
    """Afina C y gamma alrededor de la mejor zona, con todas las copas del 70 %."""
    from sklearn.metrics import cohen_kappa_score, make_scorer
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    FINO = {"svc__C": 2.0 ** np.arange(lc - RANGO_FINO, lc + RANGO_FINO + 1e-9, PASO_FINO), "svc__gamma": 2.0 ** np.arange(lg - RANGO_FINO, lg + RANGO_FINO + 1e-9, PASO_FINO)}
    print(f"Grilla fina con {len(TR)} copas: {len(FINO['svc__C']) * len(FINO['svc__gamma'])} pares × {PLIEGUES * REPETICIONES} ajustes (es el paso más largo)…", flush=True)
    G2 = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf", cache_size=1000)), FINO, scoring=make_scorer(cohen_kappa_score),
                      cv=RepeatedStratifiedKFold(n_splits=PLIEGUES, n_repeats=REPETICIONES, random_state=SEMILLA), n_jobs=-1, refit=False).fit(TR[SEL], TR.y)
    C, G = float(G2.best_params_["svc__C"]), float(G2.best_params_["svc__gamma"]); K, KD = float(G2.best_score_), float(G2.cv_results_["std_test_score"][G2.best_index_])
    _mapa_calor(G2, "Grilla fina"); print(f"C = {C:.4g} · gamma = {G:.4g} · Kappa en validación {K:.3f} ± {KD:.3f} · {time.time() - T0:.0f} s")
    return C, G, (K, KD)


def entrenar_svm(TR, SEL, C, GAMMA, SEMILLA=42):
    """Entrena el SVM UNA sola vez con el 70 % (rasgos escalados + núcleo RBF) y le agrega probabilidades; mide cuánto usó cada rasgo."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    svm = CalibratedClassifierCV(make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=GAMMA, cache_size=1000)), method="sigmoid", cv=5, ensemble=False).fit(TR[SEL], TR.y)
    return svm, importancia(svm, TR, SEL, SEMILLA)


# ───────────────────────────── XGBoost ─────────────────────────────
def rasgos_del_rf(B, TR):
    """Usa los MISMOS rasgos que eligió el RF; repite su cálculo solo para mostrar cuánto separa cada rasgo."""
    SEL = seleccionar_rasgos(TR, list(B["rasgos_todos"]))
    if sorted(SEL) != sorted(B["rasgos"]): print("⚠️ el cálculo no da los mismos rasgos que el RF: se usan los del RF.")
    return list(B["rasgos"])


def validar_xgb(TR, SEL, PARAM, PLIEGUES=5, REPETICIONES=5, SEMILLA=42):
    """Mide, solo con el 70 %, cuánto rinden los parámetros fijos con validación cruzada 5 × 5 (el Kappa de cada ajuste)."""
    from xgboost import XGBClassifier
    from sklearn.metrics import cohen_kappa_score, make_scorer
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    print(f"Validando XGBoost en {PLIEGUES * REPETICIONES} ajustes…", flush=True)
    KS = cross_val_score(XGBClassifier(**PARAM, random_state=SEMILLA, n_jobs=-1), TR[SEL], TR.y, scoring=make_scorer(cohen_kappa_score),
                         cv=RepeatedStratifiedKFold(n_splits=PLIEGUES, n_repeats=REPETICIONES, random_state=SEMILLA), n_jobs=1)
    K, KD = float(KS.mean()), float(KS.std())
    fig, ax = plt.subplots(figsize=(9, 4)); rep = np.repeat(np.arange(1, REPETICIONES + 1), PLIEGUES)
    ax.scatter(rep + np.tile(np.linspace(-0.2, 0.2, PLIEGUES), REPETICIONES), KS, color="#2E7D32", s=40, zorder=3); ax.axhline(K, color="#D50000", ls="--", lw=1.5, label=f"promedio {K:.3f}")
    ax.set_xticks(range(1, REPETICIONES + 1), [f"repetición {r}" for r in range(1, REPETICIONES + 1)]); ax.set_ylabel("Kappa en validación"); ax.grid(axis="y", alpha=0.25); ax.legend()
    ax.set_title("El Kappa de cada pliegue (5 pliegues × 5 repeticiones)"); plt.tight_layout(); plt.show()
    print(f"Kappa en validación {K:.3f} ± {KD:.3f} (mínimo {KS.min():.3f} · máximo {KS.max():.3f})")
    return (K, KD), KS


def entrenar_xgb(TR, SEL, PARAM, SEMILLA=42):
    """Entrena UN XGBoost con el 70 %: árboles que se suman uno tras otro, cada uno corrige los errores de los anteriores."""
    from xgboost import XGBClassifier
    xgb = XGBClassifier(**PARAM, random_state=SEMILLA, n_jobs=-1).fit(TR[SEL], TR.y)
    GAN = pd.Series(xgb.get_booster().get_score(importance_type="gain")).reindex(SEL).fillna(0).sort_values(ascending=False)
    return xgb, importancia(xgb, TR, SEL, SEMILLA), GAN


# ───────────────────────────── examen ─────────────────────────────
def evaluar(modelo, TE, SEL, N_BOOT=1000, SEMILLA=42):
    """Examen en el 30 % que el modelo nunca vio: métricas con intervalo de confianza del 95 % (bootstrap), matriz de confusión y curva ROC."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
    yt = TE.y.values; p_t = modelo.predict_proba(TE[SEL])[:, 1]; y_p = modelo.predict(TE[SEL]).astype(int)
    return _examen(yt, y_p, p_t, N_BOOT, SEMILLA)


def _examen(yt, y_p, p_t, N_BOOT=1000, SEMILLA=42):
    """Métricas con IC95 (bootstrap), matriz de confusión y curva ROC, para cualquier modelo (1 = topa)."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
    def met(y, yp, p):
        return {"OA": accuracy_score(y, yp), "Kappa": cohen_kappa_score(y, yp), "Precisión topa": precision_score(y, yp, zero_division=0),
                "Recall topa": recall_score(y, yp, zero_division=0), "F1 topa": f1_score(y, yp, zero_division=0), "AUC": roc_auc_score(y, p)}
    rng = np.random.default_rng(SEMILLA); BOOT = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[i])) == 2: BOOT.append(met(yt[i], y_p[i], p_t[i]))
    BOOT = pd.DataFrame(BOOT); TAB = pd.DataFrame({"valor": met(yt, y_p, p_t), "IC95 inferior": BOOT.quantile(0.025), "IC95 superior": BOOT.quantile(0.975)}).round(3)
    cm = confusion_matrix(yt, y_p, labels=[1, 0]); VP, FN, FP, VN = (int(v) for v in cm.ravel())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5)); a1.imshow(cm, cmap="Greens")
    for (i, j), v in np.ndenumerate(cm): a1.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
    a1.set_xticks([0, 1], ["dice topa", "dice no topa"]); a1.set_yticks([0, 1], ["topa real", "no topa real"]); a1.set_title(f"Matriz de confusión · Kappa {TAB.loc['Kappa', 'valor']:.3f}")
    fx, fy, _ = roc_curve(yt, p_t); a2.plot(fx, fy, color="#1565C0", lw=2); a2.plot([0, 1], [0, 1], color="#9E9E9E", ls="--", lw=0.8)
    a2.set_xlabel("falsas alarmas"); a2.set_ylabel("topas encontradas"); a2.set_title(f"Curva ROC · AUC {TAB.loc['AUC', 'valor']:.3f}"); plt.tight_layout(); plt.show()
    print(f"VP {VP} · FN {FN} · FP {FP} · VN {VN}")
    return TAB, {"VP": VP, "FN": FN, "FP": FP, "VN": VN}, y_p


def capas_examen(GEO, TR, TE, y_p, aprendio="azul", acierto="verde", falsa_alarma="rojo", no_reconocida="naranja",
                 borde="negro", borde_errores="blanco", grosor=0.2, grosor_errores=1.2, transparencia_no_topa=0.3):
    """Capas del examen: color de lo que aprendió, de los aciertos y de los dos errores; bordes, grosores y transparencia."""
    yt = TE.y.values; GT, GTR = GEO.loc[TE.index], GEO.loc[TR.index]
    vp, vn, fp, fn = (yt == 1) & (y_p == 1), (yt == 0) & (y_p == 0), (yt == 0) & (y_p == 1), (yt == 1) & (y_p == 0)
    return [capa(GTR[GTR.y == 0], aprendio, borde, "70 %: no topa con que aprendió", grosor / 2, transparencia_no_topa),
            capa(GT[vn], acierto, borde, "30 %: no topa acertada", grosor / 2, transparencia_no_topa),
            capa(GTR[GTR.y == 1], aprendio, borde, "70 %: topa con que aprendió", grosor, 1),
            capa(GT[vp], acierto, borde, "30 %: topa acertada", grosor, 1),
            capa(GT[fp], falsa_alarma, borde_errores, "30 %: marcada como topa sin serlo", grosor_errores, 1),
            capa(GT[fn], no_reconocida, borde_errores, "30 %: topa no reconocida", grosor_errores, 1)]


# ═════════════════════════════ YOLO (copia de las celdas 9 a 13 de Ultima_prueba_YOLO26.ipynb) ═════════════════════════════
def yolo_poligonos(SHP, src, CAMPO, TOPA):
    """Lee tus polígonos tal cual (sin corregirlos) y les da la clase de YOLO: 0 = topa, 1 = otras."""
    g = gpd.read_file(SHP); g = g.set_crs(src.crs) if g.crs is None else g.to_crs(src.crs)
    g["cls"] = np.where(g[CAMPO].astype(str) == TOPA, 0, 1); g["cenx"] = g.geometry.centroid.x; g["ceny"] = g.geometry.centroid.y
    print(f"topa = {(g.cls == 0).sum()} · otras = {(g.cls == 1).sum()}")
    return g


def yolo_corte(g, TRAIN_FRAC=0.70):
    """Línea de corte por los centros de las topas: el 70 % aprende y el 30 % se valida."""
    t = g[g.cls == 0]; t = t if len(t) >= 3 else g
    if np.ptp(t["cenx"]) >= np.ptp(t["ceny"]): coord, SV, EJE = g["cenx"].values, np.percentile(t["cenx"], TRAIN_FRAC * 100), "x"
    else: coord, SV, EJE = g["ceny"].values, np.percentile(t["ceny"], TRAIN_FRAC * 100), "y"
    g["side"] = np.where(coord <= SV, "train", "val")
    print(f"corte eje {EJE} en {SV:.1f} → aprende {(g.side == 'train').sum()} (topa {((g.side == 'train') & (g.cls == 0)).sum()}) · "
          f"valida {(g.side == 'val').sum()} (topa {((g.side == 'val') & (g.cls == 0)).sum()}) polígonos")
    return g, EJE, float(SV)


def capas_yolo(g, aprende="celeste", valida="magenta", relleno="ninguno", grosor=0.8):
    """Capas del 70/30 de YOLO: color del borde de lo que aprende y de lo que valida, relleno y grosor."""
    return [capa(g[g.side == "train"], relleno, aprende, "70 % aprende", grosor), capa(g[g.side == "val"], relleno, valida, "30 % valida", grosor)]


def yolo_teselas(src, g, OUT="/content/ds_topa_seg_7030", TILE=640, RGB=(1, 2, 3)):
    """Una tesela de 640 × 640 px centrada en cada polígono, con los contornos de los polígonos de su mismo lado (formato YOLO)."""
    import random
    from PIL import Image
    from shapely.geometry import box, MultiPolygon
    random.seed(42); np.random.seed(42); RGB = list(RGB)
    for s in ["train", "val"]: os.makedirs(f"{OUT}/images/{s}", exist_ok=True); os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)
    nb = src.count; F = max(1, int(max(src.width, src.height) / 2000))
    pv = src.read(RGB + ([8] if nb >= 8 else []), out_shape=(len(RGB) + (1 if nb >= 8 else 0), src.height // F, src.width // F))
    msk = pv[3] > 0 if nb >= 8 else ~((pv[:3].sum(0) == 0) | ((pv[:3] >= pv[:3].max()).all(0)))
    LO = [np.percentile(pv[c][msk], 2) for c in range(3)]; HI = [np.percentile(pv[c][msk], 98) for c in range(3)]
    def leer(win):
        a = src.read(RGB, window=win, boundless=True, fill_value=0).astype(float)
        for c in range(3): a[c] = np.clip((a[c] - LO[c]) / (HI[c] - LO[c] + 1e-9), 0, 1)
        return (np.dstack([a[0], a[1], a[2]]) * 255).astype(np.uint8)
    TOT = {"train": 0, "val": 0}; INST = {"train": [0, 0], "val": [0, 0]}; SKIP = 0
    for i, cr in g.iterrows():
        sd = cr["side"]; row, col = src.index(cr["cenx"], cr["ceny"])
        win = Window(col - TILE // 2, row - TILE // 2, TILE, TILE); wt = src.window_transform(win); inv = ~wt
        gminx, gmaxy = wt * (0, 0); gmaxx, gminy = wt * (TILE, TILE); tbox = box(gminx, gminy, gmaxx, gmaxy)
        sub = g[(g.side == sd) & (g.geometry.centroid.within(tbox))]; lines = []
        for _, cc in sub.iterrows():
            it = cc.geometry.intersection(tbox)
            if it.is_empty: continue
            for p in (it.geoms if isinstance(it, MultiPolygon) else [it]):
                if not isinstance(p, Polygon) or p.area == 0: continue
                xs, ys = p.exterior.coords.xy; nm = []
                for x, y in zip(xs, ys):
                    cxp, ryp = inv * (x, y); nm += [min(max(cxp / TILE, 0), 1), min(max(ryp / TILE, 0), 1)]
                if len(nm) >= 6: lines.append(str(int(cc.cls)) + " " + " ".join(f"{v:.6f}" for v in nm)); INST[sd][int(cc.cls)] += 1
        if not lines: SKIP += 1; continue
        img = leer(win)
        if (img.sum(2) > 0).mean() < 0.15: SKIP += 1; continue
        Image.fromarray(img).save(f"{OUT}/images/{sd}/c{i}.jpg", quality=92)
        open(f"{OUT}/labels/{sd}/c{i}.txt", "w").write("\n".join(lines)); TOT[sd] += 1
    open(f"{OUT}/data.yaml", "w").write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: 2\nnames: ['topa','otras']\n")
    print(f"aprende {TOT['train']} teselas (topa {INST['train'][0]} / otras {INST['train'][1]}) · valida {TOT['val']} teselas (topa {INST['val'][0]} / otras {INST['val'][1]}) · saltadas {SKIP}")
    return OUT


def ver_teselas(OUT, SPLIT="train", NVER=6):
    """Algunas teselas con sus contornos: verde = topa, rojo = otras."""
    import glob, random
    from PIL import Image
    files = sorted(glob.glob(f"{OUT}/images/{SPLIT}/*.jpg"))
    if not files:
        print(f"No hay teselas en {OUT}/images/{SPLIT}. Corre primero el paso «Cortar las teselas de 640 px» "
              "(si Colab se reinició, se borran: hay que cortarlas otra vez)."); return
    sel = random.sample(files, min(NVER, len(files)))
    fig, axs = plt.subplots(1, len(sel), figsize=(4 * len(sel), 4), squeeze=False)
    for ax, f in zip(axs[0], sel):
        im = np.array(Image.open(f)); H, W = im.shape[:2]; ax.imshow(im)
        for ln in open(f.replace("images", "labels").replace(".jpg", ".txt")):
            p = ln.split(); v = np.array(p[1:], float).reshape(-1, 2); v[:, 0] *= W; v[:, 1] *= H
            ax.plot(*np.vstack([v, v[0]]).T, lw=1.5, color="lime" if int(p[0]) == 0 else "red")
        ax.set_title(os.path.basename(f)); ax.axis("off")
    plt.tight_layout(); plt.show()


def entrenar_yolo(OUT, SALIDA, NOMBRE="yolo_topa_seg_7030", MODELO_BASE="yolo26l-seg.pt", EPOCAS=150, TAM=640, LOTE=16, PACIENCIA=150, SEMILLA=42):
    """Ajusta YOLO26l-seg (preentrenado en COCO) a tus teselas; en cada época copia el avance a Drive por si Colab se corta."""
    import glob
    from ultralytics import YOLO
    if not os.path.exists(f"{OUT}/data.yaml"):
        raise FileNotFoundError(f"No hay teselas en {OUT}. Corre primero el paso «Cortar las teselas de 640 px» "
                                "(si Colab se reinició, se borran: hay que cortarlas otra vez).")
    _pedir_gpu()
    CKPT = f"{SALIDA}/ckpt_{NOMBRE}"; os.makedirs(CKPT, exist_ok=True)
    for c in glob.glob(f"{OUT}/**/*.cache", recursive=True): os.remove(c)
    m = YOLO(MODELO_BASE)
    def copiar(tr):
        for w in ["best.pt", "last.pt"]:
            p = os.path.join(tr.save_dir, "weights", w)
            if os.path.exists(p): shutil.copy(p, f"{CKPT}/{w}")
    m.add_callback("on_fit_epoch_end", copiar)
    r = m.train(data=f"{OUT}/data.yaml", epochs=EPOCAS, imgsz=TAM, batch=LOTE, project="runs_topa", name=NOMBRE, exist_ok=True, patience=PACIENCIA, seed=SEMILLA, cache=False)
    SD = str(r.save_dir); MODELO = f"{SALIDA}/{NOMBRE}.pt"; shutil.copy(f"{SD}/weights/best.pt", MODELO)
    if os.path.exists(f"{SD}/results.png"):
        plt.figure(figsize=(14, 7)); plt.imshow(plt.imread(f"{SD}/results.png")); plt.axis("off"); plt.title("Cómo aprendió en cada época"); plt.show()
    print(f"Modelo guardado (mejor época): {MODELO}")
    return MODELO


def _hay_modelo(MODELO):
    """Avisa claro si todavía no existe la red o el modelo entrenado."""
    if not os.path.exists(str(MODELO)):
        tesis = f'/content/drive/MyDrive/{os.path.basename(str(MODELO))}'
        raise FileNotFoundError(f"No encontré el modelo entrenado: {MODELO}. Corre primero el paso de entrenar. "
                                f"Para usar el de la tesis sin entrenar, escribe antes de esta celda: MODELO = \"{tesis}\" "
                                "(en RF, SVM y XGBoost la ruta está en la nota del paso «Detectar en todo el ortomosaico»).")


def _pedir_gpu():
    """Entrenar una red sin GPU tardaría días: se detiene y explica cómo activarla."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Colab está sin GPU (abajo a la derecha dice «Python 3» y no «T4»). Ve a «Entorno de ejecución → Cambiar tipo de entorno "
                           "de ejecución», elige GPU T4 y guarda. Colab se reinicia: corre otra vez el Paso 1 y el Paso 2 (en ViT y U-Net, también el paso "
                           "«recuperar»; en YOLO, el de cortar las teselas) y luego entrena.")


def validar_yolo(MODELO, OUT, TAM=640):
    """Valida el modelo en las teselas del 30 % con los valores de Ultralytics: precisión, recuperación y mAP50 por clase."""
    import glob
    from ultralytics import YOLO
    _hay_modelo(MODELO)
    mej = YOLO(MODELO); res = mej.val(data=f"{OUT}/data.yaml", split="val", imgsz=TAM, project="runs_topa", name="validacion", exist_ok=True, cache=False)
    names = mej.names; filas = {}      # métricas estándar de Ultralytics (sus conteos de la matriz incluyen cajas de confianza casi cero: no se muestran)
    for i, c in enumerate(res.box.ap_class_index):
        filas[names[c]] = {"P (caja)": res.box.p[i], "R (caja)": res.box.r[i], "mAP50 (caja)": res.box.ap50[i], "mAP50 (máscara)": res.seg.ap50[i], "mAP50-95 (caja)": res.box.ap[i]}
    TAB = pd.DataFrame(filas).T.round(3); VD = str(res.save_dir)
    for pats, tit in ((["confusion_matrix.png"], "Matriz de confusión"), (["MaskPR_curve.png", "BoxPR_curve.png"], "Precisión-recuperación"), (["val_batch0_pred.jpg"], "Predicciones en el 30 %")):
        fs = [x for p in pats for x in glob.glob(f"{VD}/{p}")]
        if fs: plt.figure(figsize=(11, 8)); plt.imshow(plt.imread(fs[0])); plt.axis("off"); plt.title(tit); plt.show()
    return TAB


# ═════════════════════════════ ViT (celdas «creadores» de Ultima_prueba_VIT.ipynb) ═════════════════════════════
def vit_watershed(src, MAXDIM=4000, MIN_DIST=18, SIGMA=2, NDVI_MIN=0.3, AREA_MIN=2, AREA_MAX=200, NIR=7, ROJO=5):
    """Separa las copas con su watershed: NDVI > 0,3, suavizado, centros a 18 px y copas de 2 a 200 m²."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    esc = max(src.width, src.height) / MAXDIM; H, W = int(src.height / esc), int(src.width / esc)
    nir = src.read(NIR, out_shape=(H, W)).astype(np.float32); red = src.read(ROJO, out_shape=(H, W)).astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-9); veg = ndvi > NDVI_MIN
    surf = ndi.gaussian_filter(np.where(veg, nir, 0), SIGMA); peaks = peak_local_max(surf, min_distance=MIN_DIST, labels=veg)
    mk = np.zeros(surf.shape, np.int32)
    for i, (r, c) in enumerate(peaks, 1): mk[r, c] = i
    labels = watershed(-surf, mk, mask=veg); transform = src.transform * src.transform.scale(esc, esc)
    polys = [shape(g) for g, v in shapes(labels.astype(np.int32), mask=labels > 0, transform=transform) if v > 0]
    gw = gpd.GeoDataFrame(geometry=polys, crs=src.crs); gw = gw[(gw.area >= AREA_MIN) & (gw.area <= AREA_MAX)].reset_index(drop=True)
    print(f"{len(gw)} copas de {AREA_MIN} a {AREA_MAX} m²"); return gw


def vit_etiquetar(gw, SHP, src, CAMPO, TOPA, NEG_X=5):
    """Clase de cada copa = la del polígono con más área en común (0 = topa, 1 = no topa); las no topa se reducen a 5 por cada topa."""
    shp = gpd.read_file(SHP).to_crs(src.crs); shp["_topa"] = (shp[CAMPO].astype(str) == TOPA).astype(int); gw = gw.copy(); gw["y"] = np.nan
    for i, g in gw.iterrows():
        sub = shp[shp.intersects(g.geometry)]
        if len(sub) == 0: continue
        inter = sub.geometry.intersection(g.geometry).area; gw.at[i, "y"] = 0 if sub.loc[inter.idxmax(), "_topa"] == 1 else 1
    topa = gw[gw.y == 0].copy(); noto = gw[gw.y == 1].copy()
    if len(topa) == 0:
        raise ValueError("Ninguna copa quedó como topa: revisa CAMPO y TOPA en el paso 2 (el campo de especie y el valor que es topa).")
    noto = noto.sample(n=min(len(noto), NEG_X * len(topa)), random_state=42)
    gw = gpd.GeoDataFrame(np.r_[topa.values, noto.values], columns=gw.columns, crs=src.crs); gw["cenx"] = gw.geometry.centroid.x; gw["ceny"] = gw.geometry.centroid.y
    print(f"topa {len(topa)} · no topa {len(noto)}"); return gw, topa


def vit_corte(gw, topa, SPLIT_PCT=0.70):
    """Línea Norte–Sur por el 70 % de los centros de las topas: al sur aprende, al norte se valida."""
    corte = float(np.quantile(topa.geometry.centroid.y.values, SPLIT_PCT)); gw["split"] = np.where(gw["ceny"] <= corte, 0, 1)
    print(f"corte en y = {corte:.1f} · aprende {(gw.split == 0).sum()} · valida {(gw.split == 1).sum()} copas"); return gw, corte


def vit_parches(src, gw, OUT_NPZ, CANALES=(1, 2, 3), PATCH=224, ver=True):
    """Recorta el recuadro de cada copa en color natural y lo lleva a 224 × 224 px (la entrada del ViT). ver = dibujar algunos parches."""
    import cv2
    from rasterio.windows import from_bounds
    CANALES = list(CANALES); samp = src.read(CANALES, out_shape=(3, 1000, 1000)).astype(np.float32)
    LO = np.array([np.percentile(samp[i], 2) for i in range(3)]); HI = np.array([np.percentile(samp[i], 98) for i in range(3)])
    def parche(geom):
        a = src.read(CANALES, window=from_bounds(*geom.bounds, transform=src.transform), boundless=True, fill_value=0).astype(np.float32)
        for c in range(3): a[c] = np.clip((a[c] - LO[c]) / (HI[c] - LO[c] + 1e-9), 0, 1)
        return (cv2.resize(np.transpose(a, (1, 2, 0)), (PATCH, PATCH), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    X, Y, S = [], [], []
    for _, g in gw.iterrows(): X.append(parche(g.geometry)); Y.append(int(g.y)); S.append(int(g.split))
    X, Y, S = np.stack(X), np.array(Y), np.array(S); os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True); np.savez_compressed(OUT_NPZ, X=X, Y=Y, S=S)
    if ver: ver_parches(X, Y, gw)
    print(f"{len(X)} parches guardados en {OUT_NPZ}"); return X, Y, S


def ver_parches(X, Y, gw=None, N=8, color_topa="celeste", color_otras="magenta", grosor=2, tam_titulo=13, SEMILLA=42):
    """Algunos parches que mira el ViT: arriba topa y abajo no topa, con el borde de la copa encima (si se da gw).
    N = parches por fila · colores en palabras · SEMILLA = cambia el número para ver otras copas."""
    rng = np.random.default_rng(SEMILLA); P = X.shape[1]
    fig, axs = plt.subplots(2, N, figsize=(2.1 * N, 5.4), squeeze=False)
    for f, (cl, nombre, col) in enumerate(((0, "TOPA", color_topa), (1, "NO TOPA", color_otras))):
        idx = np.where(Y == cl)[0]; idx = np.sort(rng.choice(idx, min(N, len(idx)), replace=False)) if len(idx) else idx
        for j in range(N):
            ax = axs[f, j]; ax.set_xticks([]); ax.set_yticks([])
            if j >= len(idx): ax.axis("off"); continue
            k = idx[j]; ax.imshow(X[k])
            if gw is not None:
                g = gw.geometry.iloc[k]; x0, y0, x1, y1 = g.bounds
                for pol in getattr(g, "geoms", [g]):
                    xs, ys = np.asarray(pol.exterior.xy)
                    ax.plot((xs - x0) / (x1 - x0 + 1e-9) * P, (y1 - ys) / (y1 - y0 + 1e-9) * P, color=_col(col), lw=grosor)
                ax.set_title(f"copa {k} · {g.area:.0f} m²", fontsize=tam_titulo - 4)
            for borde_ in ax.spines.values(): borde_.set_edgecolor(_col(col)); borde_.set_linewidth(4)
            ax.set_xlim(0, P); ax.set_ylim(P, 0)
        axs[f, 0].set_ylabel(nombre, fontsize=tam_titulo, fontweight="bold")
    fig.suptitle("Lo que mira el ViT: el recuadro de cada copa llevado a 224 × 224 px (la línea es el borde de la copa)", fontsize=tam_titulo)
    plt.tight_layout(); plt.show()


def cargar_parches(PARCHES):
    """Si Colab se reinició: lee los parches que guardó vit_parches en tu Drive (no hace falta recortar otra vez)."""
    if not os.path.exists(PARCHES):
        raise FileNotFoundError(f"No encontré {PARCHES}. Corre primero el paso «Recortar cada copa (parches de 224 px)».")
    d = np.load(PARCHES); X, Y, S = d["X"], d["Y"], d["S"]
    print(f"{len(X)} parches recuperados · aprende {(S == 0).sum()} · valida {(S == 1).sum()} · topa {(Y == 0).sum()} · no topa {(Y == 1).sum()}")
    return X, Y, S


VIT_MEDIA, VIT_DESV = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])   # normalización de ImageNet (RGB)


def entrenar_vit(X, Y, S, OUT_PT, EPOCHS=150, BS=32, LR=5e-5, SEMILLA=42):
    """Ajusta un ViT-Small/16 preentrenado en ImageNet: giros, espejos, brillo, gamma y color al azar; guarda la época con mejor Kappa."""
    import torch, torch.nn as nn, timm
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import cohen_kappa_score
    _pedir_gpu(); np.random.seed(SEMILLA); dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr, Xva, Yva = X[S == 0], Y[S == 0], X[S == 1], Y[S == 1]
    class DS(Dataset):
        def __init__(s, X, Y, aug): s.X, s.Y, s.aug = X, Y, aug
        def __len__(s): return len(s.X)
        def __getitem__(s, i):
            a = s.X[i].astype(np.float32) / 255.
            if s.aug:
                if np.random.rand() < .5: a = a[:, ::-1]
                if np.random.rand() < .5: a = a[::-1, :]
                a = np.rot90(a, np.random.randint(4)); a = np.clip(a * np.random.uniform(.85, 1.15), 0, 1)
                a = np.clip(a ** np.random.uniform(.85, 1.15), 0, 1); a = np.clip(a * np.random.uniform(.9, 1.1, size=3), 0, 1)
            a = (a - VIT_MEDIA) / VIT_DESV
            return torch.tensor(np.ascontiguousarray(a.transpose(2, 0, 1)), dtype=torch.float32), int(s.Y[i])
    tl = DataLoader(DS(Xtr, Ytr, True), BS, shuffle=True, num_workers=0); vl = DataLoader(DS(Xva, Yva, False), BS, shuffle=False, num_workers=0)
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=2, in_chans=3).to(dev)
    w = torch.tensor([len(Ytr) / (2 * (Ytr == 0).sum()), len(Ytr) / (2 * (Ytr == 1).sum())], dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); scaler = torch.cuda.amp.GradScaler(); best = -1; hist = []
    os.makedirs(os.path.dirname(OUT_PT), exist_ok=True)
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad()
            with torch.cuda.amp.autocast(): loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step(); model.eval(); P = []
        with torch.no_grad():
            for xb, _ in vl:
                with torch.cuda.amp.autocast(): o = model(xb.to(dev)).float().cpu().numpy()
                P += list(o.argmax(1))
        k = cohen_kappa_score(Yva, np.array(P)); hist.append(k)
        if k > best: best = k; torch.save(model.state_dict(), OUT_PT)
        print(f"época {ep + 1:03d} · Kappa {k:.3f} (mejor {best:.3f})")
    plt.figure(figsize=(8, 4)); plt.plot(range(1, EPOCHS + 1), hist, marker="o", ms=3); plt.axhline(best, color="g", ls="--", label=f"mejor Kappa = {best:.3f}")
    plt.xlabel("época"); plt.ylabel("Kappa en el 30 %"); plt.legend(); plt.grid(alpha=.3); plt.title("Cómo aprendió el ViT"); plt.tight_layout(); plt.show()
    print(f"Mejor modelo guardado: {OUT_PT}"); return OUT_PT, hist


def examen_vit(OUT_PT, X, Y, S, N_BOOT=1000, SEMILLA=42):
    """Examen del ViT guardado en los parches del 30 %: métricas con IC95, matriz de confusión y curva ROC (1 = topa)."""
    import torch, timm
    _hay_modelo(OUT_PT); dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=2, in_chans=3); model.load_state_dict(torch.load(OUT_PT, map_location=dev)); model.to(dev).eval()
    Xva, Yva = X[S == 1], Y[S == 1]; pr = []
    with torch.no_grad():
        for i in range(0, len(Xva), 64):
            a = (Xva[i:i + 64].astype(np.float32) / 255. - VIT_MEDIA) / VIT_DESV
            pr.append(torch.softmax(model(torch.tensor(a.transpose(0, 3, 1, 2), dtype=torch.float32).to(dev)), 1)[:, 0].cpu().numpy())
    p_t = np.concatenate(pr); return _examen((Yva == 0).astype(int), (p_t > 0.5).astype(int), p_t, N_BOOT, SEMILLA)


# ═════════════════════════════ U-Net («CELDA 1 RGB» y «CELDA 2 RGB» de Ultima_prueba_U-NET.ipynb) ═════════════════════════════
def unet_teselas(src, SHP, CAMPO, TOPA, OUT_NPZ, CANALES=(1, 2, 3), TILE=640, STRIDE=640, MIN_VALID=0.30, EJE_SPLIT="Y", TRAIN_PCT=0.70):
    """Corta la zona de tus polígonos en teselas de 640 px sin solape; cada tesela lleva su máscara (topa = 1). 70 % al sur aprende, 30 % al norte se valida."""
    from rasterio.features import rasterize
    from rasterio.windows import bounds as win_bounds
    CANALES = list(CANALES); shp = gpd.read_file(SHP).to_crs(src.crs); topa = shp[shp[CAMPO].astype(str) == TOPA]
    minx, miny, maxx, maxy = shp.total_bounds; r0, c0 = src.index(minx, maxy); r1, c1 = src.index(maxx, miny)
    r0, r1 = sorted([max(0, r0), min(src.height, r1)]); c0, c1 = sorted([max(0, c0), min(src.width, c1)])
    samp = src.read(CANALES, out_shape=(3, 1000, 1000)).astype(np.float32)
    LO = np.array([np.percentile(samp[i], 2) for i in range(3)]); HI = np.array([np.percentile(samp[i], 98) for i in range(3)])
    X, M, CX, CY, TB = [], [], [], [], []
    for r in range(r0, r1, STRIDE):
        for c in range(c0, c1, STRIDE):
            win = Window(c, r, TILE, TILE); a = src.read(CANALES, window=win, boundless=True, fill_value=0).astype(np.float32)
            if (a.sum(0) > 0).mean() < MIN_VALID: continue
            m = rasterize([(g, 1) for g in topa.geometry], out_shape=(TILE, TILE), transform=src.window_transform(win), fill=0, dtype="uint8")
            for i in range(3): a[i] = np.clip((a[i] - LO[i]) / (HI[i] - LO[i] + 1e-9), 0, 1)
            l, b, rr, t = win_bounds(win, src.transform)
            X.append((np.transpose(a, (1, 2, 0)) * 255).astype(np.uint8)); M.append(m); CX.append((l + rr) / 2); CY.append((b + t) / 2); TB.append((l, b, rr, t))
    X, M, CX, CY, TB = np.stack(X), np.stack(M), np.array(CX), np.array(CY), np.array(TB)
    coord = CY if EJE_SPLIT == "Y" else CX; corte = float(np.quantile(coord, TRAIN_PCT)); S = np.where(coord <= corte, "train", "val")
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True); np.savez_compressed(OUT_NPZ, X=X, M=M, S=S, CX=CX, CY=CY, TB=TB)
    print(f"{len(X)} teselas · aprende {(S == 'train').sum()} · valida {(S == 'val').sum()} · con topa {(M.reshape(len(M), -1).sum(1) > 0).sum()}")
    return X, M, S, TB, corte


def cargar_teselas(TESELAS):
    """Si Colab se reinició: lee las teselas y máscaras que guardó unet_teselas en tu Drive (no hace falta cortar otra vez)."""
    if not os.path.exists(TESELAS):
        raise FileNotFoundError(f"No encontré {TESELAS}. Corre primero el paso «Teselas de 640 px con su máscara y división 70/30».")
    d = np.load(TESELAS); X, M, S = d["X"], d["M"], d["S"]
    print(f"{len(X)} teselas recuperadas · aprende {(S == 'train').sum()} · valida {(S == 'val').sum()} · con topa {(M.reshape(len(M), -1).sum(1) > 0).sum()}")
    return X, M, S


def ver_teselas_unet(src, X, M, S, TB, corte, EJE_SPLIT="Y", aprende="celeste", valida="naranja", transparencia=0.30, color_linea="rojo",
                     titulo=None, tam_titulo=13, color_mascara="rojo", transparencia_mascara=0.55):
    """El 70/30 sobre el ortomosaico (colores de lo que aprende y valida, línea de corte) y 3 teselas con su máscara de topa."""
    img, EXT = _vista(src); fig, ax = plt.subplots(figsize=(11, 11)); ax.imshow(img, extent=EXT)
    for (l, b, rr, t), sp in zip(TB, S): ax.add_patch(Rectangle((l, b), rr - l, t - b, facecolor=_col(aprende) if sp == "train" else _col(valida), edgecolor="none", alpha=transparencia))
    (ax.axhline if EJE_SPLIT == "Y" else ax.axvline)(corte, color=_col(color_linea), lw=2, ls="--"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo or f"70 % aprende {(S == 'train').sum()} · 30 % valida {(S == 'val').sum()} teselas", fontsize=tam_titulo); plt.tight_layout(); plt.show()
    idx = np.where(M.reshape(len(M), -1).sum(1) > 0)[0][:3]
    if len(idx) == 0:
        print("Ninguna tesela tiene topa: revisa CAMPO y TOPA en el paso 2 (el campo de especie y el valor que es topa)."); return
    fig, axs = plt.subplots(1, len(idx), figsize=(15, 5), squeeze=False)
    for ax, i in zip(axs[0], idx): ax.imshow(X[i]); ax.imshow(np.ma.masked_where(M[i] == 0, M[i]), cmap=_mc.ListedColormap([_col(color_mascara)]), alpha=transparencia_mascara); ax.set_title(f"tesela {i}"); ax.axis("off")
    plt.suptitle(f"Máscara de topa ({color_mascara}) sobre la tesela"); plt.tight_layout(); plt.show()


def entrenar_unet(X, M, S, SALIDA_PT, ENCODER="resnet34", EPOCAS=150, BATCH=6, LR=1e-4):
    """Ajusta una U-Net (codificador ResNet34 de ImageNet) con pérdida Focal + Dice y cambios al azar de giro, brillo, contraste, gamma y color; guarda la época con mejor IoU."""
    import torch, segmentation_models_pytorch as smp
    from torch.utils.data import Dataset, DataLoader
    _pedir_gpu(); dev = "cuda" if torch.cuda.is_available() else "cpu"; tr = (S.astype(str) == "train"); va = (S.astype(str) == "val")
    class DS(Dataset):
        def __init__(s, X, M, aug): s.X, s.M, s.aug = X, M, aug
        def __len__(s): return len(s.X)
        def __getitem__(s, i):
            x = s.X[i].astype(np.float32) / 255.0; m = (s.M[i] > 0).astype(np.float32)
            if s.aug:
                if np.random.rand() < .5: x = x[:, ::-1]; m = m[:, ::-1]
                if np.random.rand() < .5: x = x[::-1, :]; m = m[::-1, :]
                k = np.random.randint(4); x = np.rot90(x, k); m = np.rot90(m, k)
                x = np.clip(x * np.random.uniform(0.8, 1.2), 0, 1); x = np.clip((x - 0.5) * np.random.uniform(0.8, 1.2) + 0.5, 0, 1)
                x = np.clip(x ** np.random.uniform(0.8, 1.25), 0, 1); x = np.clip(x * np.random.uniform(0.9, 1.1, size=3), 0, 1)
            return (torch.tensor(np.ascontiguousarray(x.transpose(2, 0, 1)), dtype=torch.float32), torch.tensor(np.ascontiguousarray(m)[None], dtype=torch.float32))
    tl = DataLoader(DS(X[tr], M[tr], True), BATCH, shuffle=True, num_workers=2); vl = DataLoader(DS(X[va], M[va], False), BATCH, shuffle=False, num_workers=2)
    model = smp.Unet(ENCODER, encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    dice = smp.losses.DiceLoss(mode="binary"); focal = smp.losses.FocalLoss(mode="binary"); lossf = lambda p, y: dice(p, y) + focal(p, y)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS)
    scaler = torch.cuda.amp.GradScaler(); best = -1; hist = []; os.makedirs(os.path.dirname(SALIDA_PT), exist_ok=True)
    for ep in range(EPOCAS):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad()
            with torch.cuda.amp.autocast(): loss = lossf(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step(); model.eval(); TP = FP = FN = TN = 0
        with torch.no_grad():
            for xb, yb in vl:
                with torch.cuda.amp.autocast(): pr = torch.sigmoid(model(xb.to(dev)))
                tp, fp, fn, tn = smp.metrics.get_stats(pr, yb.to(dev).long(), mode="binary", threshold=0.5); TP += tp.sum(); FP += fp.sum(); FN += fn.sum(); TN += tn.sum()
        iou = (TP / (TP + FP + FN + 1e-9)).item(); hist.append(iou)
        if iou > best: best = iou; torch.save(model.state_dict(), SALIDA_PT)
        print(f"época {ep + 1:03d} · IoU {iou:.3f} (mejor {best:.3f})")
    plt.figure(figsize=(8, 4)); plt.plot(range(1, EPOCAS + 1), hist, marker="o", ms=3); plt.axhline(best, color="g", ls="--", label=f"mejor IoU = {best:.3f}")
    plt.xlabel("época"); plt.ylabel("IoU en el 30 %"); plt.legend(); plt.grid(alpha=.3); plt.title("Cómo aprendió la U-Net"); plt.tight_layout(); plt.show()
    print(f"Mejor modelo guardado: {SALIDA_PT}"); return SALIDA_PT, hist


def examen_unet(SALIDA_PT, X, M, S, ENCODER="resnet34", N_BOOT=1000, SEMILLA=42, N_EJEMPLOS=3, color_real="celeste", color_unet="magenta",
                acierto="verde", falsa_alarma="rojo", no_reconocida="naranja", transparencia=0.45, tam_titulo=12):
    """Examen por píxel en las teselas del 30 %: IoU, Dice, precisión, recuperación, OA y Kappa (IC95 sorteando teselas) y ejemplos sobre la tesela:
    tus polígonos, lo que marcó la U-Net y la comparación (acierto, falsa alarma, no reconocida). Colores y cantidad de ejemplos se pueden cambiar."""
    import torch, segmentation_models_pytorch as smp
    _hay_modelo(SALIDA_PT); dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(ENCODER, encoder_weights=None, in_channels=3, classes=1); model.load_state_dict(torch.load(SALIDA_PT, map_location=dev)); model.to(dev).eval()
    va = np.where(S.astype(str) == "val")[0]; C = []; PRED = {}
    with torch.no_grad():
        for i in va:
            x = torch.tensor(X[i].astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(dev); p = (torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5); m = M[i] > 0
            C.append([(p & m).sum(), (p & ~m).sum(), (~p & m).sum(), (~p & ~m).sum()]); PRED[i] = p
    C = np.array(C, dtype=np.float64)
    def met(c):
        tp, fp, fn, tn = c; n = tp + fp + fn + tn; po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
        return {"IoU": tp / (tp + fp + fn + 1e-9), "Dice": 2 * tp / (2 * tp + fp + fn + 1e-9), "Precisión": tp / (tp + fp + 1e-9), "Recuperación": tp / (tp + fn + 1e-9),
                "OA": po, "Kappa": (po - pe) / (1 - pe + 1e-12)}
    rng = np.random.default_rng(SEMILLA); BT = pd.DataFrame([met(C[rng.integers(0, len(C), len(C))].sum(0)) for _ in range(N_BOOT)])
    TAB = pd.DataFrame({"valor": met(C.sum(0)), "IC95 inferior": BT.quantile(0.025), "IC95 superior": BT.quantile(0.975)}).round(3)
    _ejemplos_unet(X, M, PRED, va, N_EJEMPLOS, color_real, color_unet, acierto, falsa_alarma, no_reconocida, transparencia, tam_titulo)
    tp, fp, fn, tn = (int(v) for v in C.sum(0)); print(f"píxeles: VP {tp} · FP {fp} · FN {fn} · VN {tn}")
    return TAB


def _rgba(mascara, color, alfa):
    """Máscara → capa de color semitransparente para poner encima de la tesela."""
    r, g, b, _ = _mc.to_rgba(_col(color)); a = np.zeros(mascara.shape + (4,)); a[mascara] = (r, g, b, alfa); return a


def _ejemplos_unet(X, M, PRED, va, N, color_real, color_unet, acierto, falsa_alarma, no_reconocida, transparencia, tam_titulo):
    """Ejemplos del examen dibujados SOBRE la tesela: grande, mediana y chica cantidad de topa (no los mejores, sino repartidos)."""
    area = {i: (M[i] > 0).mean() for i in va}
    con = sorted([i for i in va if area[i] >= 0.02], key=lambda i: -area[i]) or sorted([i for i in va if area[i] > 0], key=lambda i: -area[i])
    if not con: print("Ninguna tesela del 30 % tiene topa: no hay ejemplos que mostrar."); return
    sel = [con[int(round(j))] for j in np.linspace(0, len(con) - 1, min(N, len(con)))]
    fig, axs = plt.subplots(len(sel), 3, figsize=(15, 5 * len(sel)), squeeze=False)
    for f_, i in enumerate(sel):
        m, p = M[i] > 0, PRED[i]; vp, fp, fn = m & p, ~m & p, m & ~p; iou = vp.sum() / max(1, (m | p).sum())
        for ax in axs[f_]: ax.imshow(X[i]); ax.axis("off")
        for ax, k, col in ((axs[f_, 0], m, color_real), (axs[f_, 1], p, color_unet)):
            ax.imshow(_rgba(k, col, transparencia))
            if k.any(): ax.contour(k.astype(float), levels=[0.5], colors=[_col(col)], linewidths=2)
        for k, col in ((vp, acierto), (fp, falsa_alarma), (fn, no_reconocida)): axs[f_, 2].imshow(_rgba(k, col, 0.7))
        axs[f_, 0].set_title(f"Tesela {i}: tus polígonos (topa real)", fontsize=tam_titulo)
        axs[f_, 1].set_title("Lo que marcó la U-Net", fontsize=tam_titulo)
        axs[f_, 2].set_title(f"¿Acertó?  IoU de esta tesela = {iou:.2f}".replace(".", ","), fontsize=tam_titulo)
    hs = [Patch(facecolor=_col(acierto), label="acierto: topa bien marcada"), Patch(facecolor=_col(falsa_alarma), label="falsa alarma: marcó topa donde no hay"),
          Patch(facecolor=_col(no_reconocida), label="no reconocida: topa que no marcó")]
    fig.legend(handles=hs, loc="lower center", ncol=3, fontsize=tam_titulo, frameon=False); plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.show()


# ───────────────────────────── guardar ─────────────────────────────
def guardar_modelo(ruta, modelo, SEL, WS, res, extra):
    """Guarda el modelo con todo lo que necesita el detector (rasgos, bandas, receta de las copas…)."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    meta = {"modelo": modelo, "rasgos": SEL, "bandas": {"B": 3, "G": 4, "R": 5, "RE": 6, "NIR": 7}, "clases": {1: "topa", 0: "otras"}, "umbral": 0.5,
            "ws": WS, "receta": {"pix": res, "ws_pix": 0.05, "ws": WS, "glcm": "dentro"}}
    meta.update(extra); joblib.dump(meta, ruta); print(f"Modelo guardado: {ruta} ({os.path.getsize(ruta) / 1e6:.1f} MB)")
    return ruta


# ───────────────────────────── mapa final ─────────────────────────────
MOTOR = "/content/drive/MyDrive/MAPAS_TOPA/mapa_topa_motor.py"


def detectar(MODELO, ORTO, SALIDA, minimo=0.5, motor=MOTOR):
    """Corre el modelo en todo el ortomosaico con la clase que aprendió (umbral de decisión 0,50) y devuelve las copas de topa (copa entera).
    minimo = en YOLO, la confianza más baja que se guarda (0,50 = la de la tesis)."""
    _hay_modelo(MODELO); os.makedirs(SALIDA, exist_ok=True)
    nombre = f"{os.path.splitext(os.path.basename(MODELO))[0]}_{os.path.splitext(os.path.basename(ORTO))[0]}_ws_esri"
    cfg = dict(MODELO=MODELO, ORTO=_orto_rapido(ORTO), SALIDA=SALIDA, COPAS_MRS=None, FORMA_SALIDA="watershed", IGUALAR_BRILLO=False, REF_BRILLO="", REFERENCIA=None,
               JUNTAR_TOCAN=False, UMBRAL_TOPA=0.5, FUENTE_UMBRAL="tutorial de Esri", NOMBRE_SALIDA=nombre)
    if MODELO.endswith(".pt") and "yolo" in os.path.basename(MODELO).lower(): cfg.update(CONFIANZA_MINIMA=minimo, NMS_GLOBAL=0.5, SOLO_TOPA=False)
    ruta = f"{SALIDA}/topa_{nombre}_configuracion.json"; json.dump(cfg, open(ruta, "w"), ensure_ascii=False)
    print("Corriendo el modelo en todo el ortomosaico (puede tardar varios minutos)…", flush=True)
    p = subprocess.Popen(["python", "-u", motor, ruta], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for linea in p.stdout:
        if not linea.startswith("@@RESULTADO@@") and "Clasificando copas" not in linea: print(linea.rstrip())
    p.wait(); shp = f"{SALIDA}/topa_{nombre}.shp"
    return gpd.read_file(shp) if os.path.exists(shp) else gpd.GeoDataFrame(geometry=[], crs=None), shp


def _escala(L_total):
    for x in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if x >= L_total / 5: return x
    return round(L_total / 5)


def mapa_final(ORTO, TOP, ALGORITMO, SALIDA, titulo="Detección de", especie="Ochroma pyramidale", tam_titulo=13,
               relleno="rojo", transparencia=0.45, borde="rojo", grosor=0.7,
               texto_leyenda=None, titulo_leyenda=None, tam_leyenda=9, pos_leyenda="abajo derecha",
               tam_letra=8, color_marco="negro", grosor_marco=1.4,
               norte=True, tam_norte=1.0, pos_norte="arriba derecha", fondo_norte="blanco", color_norte="negro",
               escala=True, largo_escala=None, tramos_escala=4, tam_escala=1.0, pos_escala="abajo izquierda", fondo_escala="blanco",
               coordenadas=True, tam_coord=None, cada_coord=None, lados_coord="cuatro", sistema=True, ancho=9, dpi=300):
    """Mapa de presentación: ortomosaico en color natural, copas de Ochroma pyramidale, marco con coordenadas en los 4 lados,
    coordenadas Norte en vertical, barra de escala, símbolo del norte, leyenda y título. TODO se puede cambiar con sus parámetros:
    título y especie (en cursiva), tamaños de letra, color de relleno y de contorno de las copas, transparencia y grosor,
    texto, título, tamaño y lugar de la leyenda, marco, ancho de la figura y resolución (dpi).
    Norte: tam_norte (1 = normal, 0.6 = más chico), pos_norte, fondo_norte («ninguno» = sin recuadro), color_norte.
    Escala: largo_escala en metros (None = automático), tramos_escala, tam_escala (grosor), pos_escala, fondo_escala.
    Coordenadas: tam_coord (letra), cada_coord (metros entre números; None = automático), lados_coord («cuatro» o «dos» = abajo e izquierda), sistema (texto del sistema de coordenadas)."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from pyproj import CRS as CRS_
    cuatro = _normal(lados_coord) not in ("dos", "2", "2 lados", "dos lados", "abajo e izquierda"); tam_coord = tam_coord or tam_letra
    relleno, borde, color_marco, pos_leyenda = _col(relleno), _col(borde), _col(color_marco), _pos(pos_leyenda)
    with rasterio.open(_orto_rapido(ORTO)) as s:
        k = max(s.width, s.height) / 3000; h, w = int(s.height / k), int(s.width / k)
        rgb = s.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average).astype(np.float32)
        val = (s.read(ALFA, out_shape=(h, w)) > 0) if s.count >= ALFA else rgb.sum(0) > 0
        lo, hi = np.percentile(rgb[:, val], 2, axis=1), np.percentile(rgb[:, val], 98, axis=1)
        img = np.clip((rgb - lo[:, None, None]) / (hi - lo + 1e-9)[:, None, None], 0, 1).transpose(1, 2, 0); img[~val] = 1
        b, crs = s.bounds, s.crs
    nom = (CRS_.from_user_input(crs).name or "").split(" + ")[0].replace("zone", "zona") if crs else ""
    cursiva = r"$\it{" + especie.replace(" ", r"\ ") + "}$"
    fig, ax = plt.subplots(figsize=(ancho, ancho * h / w + 1.4))
    ax.imshow(img, extent=(b.left, b.right, b.bottom, b.top), zorder=0)
    if len(TOP): TOP.to_crs(crs).plot(ax=ax, facecolor=relleno, alpha=transparencia, edgecolor="none", zorder=2); TOP.to_crs(crs).boundary.plot(ax=ax, color=borde, linewidth=grosor, zorder=3)
    ax.set_xlim(b.left, b.right); ax.set_ylim(b.bottom, b.top)
    for lado in ax.spines.values(): lado.set_color(color_marco); lado.set_linewidth(grosor_marco)                     # marco
    if coordenadas:                                                                                               # coordenadas en los 4 lados, Norte en vertical
        ent = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")); ax.xaxis.set_major_formatter(ent); ax.yaxis.set_major_formatter(ent)
        if cada_coord: ax.xaxis.set_major_locator(MultipleLocator(cada_coord)); ax.yaxis.set_major_locator(MultipleLocator(cada_coord))
        ax.tick_params(direction="out", length=4, top=cuatro, right=cuatro, labeltop=cuatro, labelright=cuatro, labelsize=tam_coord)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va="center")
        ax.set_xlabel("Este (m)", fontsize=tam_coord + 1); ax.set_ylabel("Norte (m)", fontsize=tam_coord + 1)
    else: ax.set_xticks([]); ax.set_yticks([])
    if escala:                                                                                                    # barra de escala en tramos blanco y negro
        W_, H_ = b.right - b.left, b.top - b.bottom; L = largo_escala or _escala(W_); n_ = max(1, int(tramos_escala)); gros = H_ * 0.008 * tam_escala
        pe = _pos(pos_escala); derecha, arriba = "right" in pe, "upper" in pe
        x0 = b.right - W_ * 0.05 - L * 1.1 if derecha else b.left + W_ * 0.05
        y0 = b.top - H_ * 0.045 - gros * 4 if arriba else b.bottom + H_ * 0.045
        if _col(fondo_escala) != "none":
            ax.add_patch(Rectangle((x0 - L * 0.08, y0 - gros * 2.2), L * 1.16, gros * 6.2, facecolor=_col(fondo_escala), edgecolor="none", alpha=0.85, zorder=4))
        for i in range(n_): ax.add_patch(Rectangle((x0 + i * L / n_, y0), L / n_, gros, facecolor="black" if i % 2 == 0 else "white", edgecolor="black", lw=0.8, zorder=5))
        for v in (0, L / 2, L) if n_ % 2 == 0 else (0, L): ax.text(x0 + v, y0 + gros * 1.6, f"{v:g}", ha="center", va="bottom", fontsize=tam_letra, zorder=6)
        ax.text(x0 + L + L * 0.03, y0 + gros / 2, "m", ha="left", va="center", fontsize=tam_letra, zorder=6)
    if norte:                                                                                                     # símbolo del norte (flecha de dos mitades)
        t = tam_norte; w_, h_, m_ = 0.09 * t, 0.175 * t, 0.02; pn = _pos(pos_norte)
        bx = m_ if "left" in pn else 1 - m_ - w_; by = m_ if "lower" in pn else 1 - m_ - h_
        nx, ny, nh = bx + w_ / 2, by + 0.02 * t, 0.10 * t; cn = _col(color_norte)
        if _col(fondo_norte) != "none":
            ax.add_patch(Rectangle((bx, by), w_, h_, transform=ax.transAxes, facecolor=_col(fondo_norte), edgecolor="black", lw=0.6, alpha=0.9, zorder=5))
        ax.add_patch(PolyMPL([(nx, ny + nh), (nx - 0.025 * t, ny), (nx, ny + nh * 0.3)], closed=True, transform=ax.transAxes, facecolor=cn, edgecolor=cn, zorder=6))
        ax.add_patch(PolyMPL([(nx, ny + nh), (nx + 0.025 * t, ny), (nx, ny + nh * 0.3)], closed=True, transform=ax.transAxes, facecolor="white", edgecolor=cn, zorder=6))
        ax.text(nx, ny + nh + 0.008 * t, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=max(5, (tam_letra + 4) * t), fontweight="bold", color=cn, zorder=6)
    hs = [Patch(facecolor=relleno, alpha=transparencia, edgecolor=borde, label=texto_leyenda or f"{cursiva} ({len(TOP)} copas)")]
    ax.legend(handles=hs, title=titulo_leyenda or f"Leyenda · {ALGORITMO}", loc=pos_leyenda, framealpha=0.95, fontsize=tam_leyenda, title_fontsize=tam_leyenda, edgecolor="black")
    if titulo: ax.set_title(f"{titulo} {cursiva}", fontsize=tam_titulo, pad=26 if coordenadas and cuatro else 10)
    if nom and coordenadas and sistema: ax.annotate(f"Sistema de coordenadas: {nom}", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -34), textcoords="offset points", ha="center", va="top", fontsize=tam_letra)
    plt.tight_layout(); os.makedirs(SALIDA, exist_ok=True)
    png = f"{SALIDA}/mapa_{ALGORITMO.replace(' ', '_').replace('-', '')}_{os.path.splitext(os.path.basename(ORTO))[0]}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.15); plt.show(); print(f"Mapa guardado: {png}")
    return png


# ───────────────────────────── ortomosaico interactivo con barra de umbral ─────────────────────────────
_PAGINA = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body{margin:0;font-family:Arial,Helvetica,sans-serif;font-size:__LETRA__px;background:#121212;color:#eee}
#barra{padding:8px 12px;display:flex;gap:22px;align-items:center;flex-wrap:wrap;background:#1d1d1d}
#mapa{height:__ALTO__px;background:#fff} input[type=range]{width:240px;vertical-align:middle} b.v{color:#B7F34A}
.titulo{font-weight:bold;color:#B7F34A;font-size:1.15em}
</style></head><body>
<div id="barra"><span class="titulo">__TITULO__</span>
<label>Umbral (probabilidad mayor que): <input id="u" type="range" min="__MIN__" max="1" step="0.01" value="__UMBRAL__"> <b class="v" id="uv"></b></label>
<span>Copas marcadas: <b class="v" id="n"></b> de __N__</span>
<label>Claridad del ortomosaico: <input id="c" type="range" min="50" max="220" step="5" value="100"></label></div>
<div id="mapa"></div>
<script>
const W=__W__, H=__H__, geo=__GEO__;
const m=L.map('mapa',{crs:L.CRS.Simple,minZoom:-8,maxZoom:8,zoomSnap:0.25,preferCanvas:true});
const ov=L.imageOverlay('data:image/jpeg;base64,__FOTO__',[[0,0],[H,W]]).addTo(m); m.fitBounds([[0,0],[H,W]]);
const est={color:'__BORDE__',weight:__GROSOR__,fillColor:'__RELLENO__',fillOpacity:__OPA__};
const todas=[]; L.geoJSON(geo,{style:()=>est,onEachFeature:(f,l)=>{l.bindTooltip('probabilidad: '+f.properties.p.toFixed(2).replace('.',','));todas.push(l);}});
const grupo=L.featureGroup().addTo(m);
function pinta(){const u=parseFloat(document.getElementById('u').value);let n=0;grupo.clearLayers();
 for(const l of todas){if(l.feature.properties.p>u){grupo.addLayer(l);n++;}}
 document.getElementById('uv').textContent=u.toFixed(2).replace('.',',');document.getElementById('n').textContent=n;}
document.getElementById('u').addEventListener('input',pinta);
document.getElementById('c').addEventListener('input',e=>{ov.getElement().style.filter='brightness('+e.target.value+'%)';});
pinta();
function encuadra(){m.invalidateSize();m.fitBounds([[0,0],[H,W]]);}
window.addEventListener('load',encuadra); setTimeout(encuadra,400); setTimeout(encuadra,1500);
</script></body></html>"""


def _copas_con_probabilidad(SHP):
    """Copas enteras con su probabilidad, las MISMAS que cuenta el mapa final.
    RF, SVM, XGBoost y ViT clasifican copas pequeñas (copas_evaluadas_…); el detector las junta en copas enteras (segmentacion_…) con la regla
    del motor: una copa pequeña va a las copas enteras que tienen más de la mitad dentro de ella o, si no hay, a la que más comparte con ella.
    Cada copa entera lleva la probabilidad más alta de sus copas pequeñas. YOLO y U-Net ya vienen por copa entera (confianza o p_media)."""
    ev, seg = SHP.replace("/topa_", "/copas_evaluadas_"), SHP.replace("/topa_", "/segmentacion_")
    if os.path.exists(ev) and os.path.exists(seg):
        E, C = gpd.read_file(ev), gpd.read_file(seg); E = E.to_crs(C.crs).reset_index(drop=True); C = C[["geometry"]].reset_index(drop=True)
        ie, ic = C.sindex.query(E.geometry, predicate="intersects")
        A = pd.DataFrame({"e": ie, "c": ic, "a": gpd.GeoSeries(E.geometry.values[ie], crs=C.crs).intersection(gpd.GeoSeries(C.geometry.values[ic], crs=C.crs), align=False).area.values})
        A["fr"] = A.a / C.area.values[A.c]; mayor = A[A.fr > 0.5]; resto = A[~A.e.isin(mayor.e)].sort_values("a").drop_duplicates("e", keep="last")
        A = pd.concat([mayor, resto]); A["p"] = E.p_topa.values[A.e]
        C["p_topa"] = A.groupby("c").p.max().reindex(C.index).fillna(0).values
        return C, "p_topa"
    G = gpd.read_file(ev if os.path.exists(ev) else SHP)
    col = next((c for c in ("p_topa", "confianza", "p_media") if c in G.columns), None)
    assert col, "Este SHP no trae probabilidad (p_topa, confianza o p_media): corre antes el paso «Detectar»."
    return G[G[col].notna()].reset_index(drop=True), col


def mapa_interactivo(ORTO, SHP, umbral=0.50, titulo="Detección de Ochroma pyramidale", relleno="rojo vivo", borde="blanco",
                     transparencia=0.40, grosor=1.2, tam_letra=14, alto=650, lado_px=2400):
    """Ortomosaico que se mueve y se acerca con el ratón, con una barra de umbral: se ven las copas con probabilidad mayor que el umbral,
    y cuántas son. Al pasar el ratón sobre una copa se ve su probabilidad. También guarda la página .html junto al SHP."""
    import base64, io, re, html as html_
    from PIL import Image
    from IPython.display import HTML, display
    from shapely.geometry import mapping
    from shapely.affinity import translate
    G, col = _copas_con_probabilidad(SHP)
    relleno, borde = _col(relleno), _col(borde)
    if relleno == "none": relleno, transparencia = "black", 0
    if borde == "none": borde, grosor = "black", 0
    relleno, borde = _mc.to_hex(relleno), _mc.to_hex(borde)
    with rasterio.open(_orto_rapido(ORTO)) as s:
        k = max(1.0, max(s.width, s.height) / lado_px); h, w = int(s.height / k), int(s.width / k)
        rgb = s.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average).astype(np.float32)
        val = (s.read(ALFA, out_shape=(h, w)) > 0) if s.count >= ALFA else rgb.sum(0) > 0
        lo, hi = np.percentile(rgb[:, val], 2, axis=1), np.percentile(rgb[:, val], 98, axis=1)
        img = (np.clip((rgb - lo[:, None, None]) / (hi - lo + 1e-9)[:, None, None], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8); img[~val] = 255
        b, crs = s.bounds, s.crs
    buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=85); foto = base64.b64encode(buf.getvalue()).decode()
    G = G.to_crs(crs)
    feats = [{"type": "Feature", "properties": {"p": round(float(p), 3)}, "geometry": mapping(translate(g.simplify(0.05), -b.left, -b.bottom))} for g, p in zip(G.geometry, G[col])]
    geo = re.sub(r"(\d+\.\d\d)\d{2,}", r"\1", json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))   # coordenadas a 1 cm; la probabilidad (3 decimales) queda igual
    rep = {"__FOTO__": foto, "__GEO__": geo, "__W__": f"{b.right - b.left:.2f}", "__H__": f"{b.top - b.bottom:.2f}", "__UMBRAL__": f"{umbral:.2f}",
           "__MIN__": f"{min(umbral, float(np.floor(G[col].min() * 100) / 100)) if len(G) else 0:.2f}", "__N__": str(len(G)), "__TITULO__": html_.escape(titulo),
           "__RELLENO__": relleno, "__BORDE__": borde, "__OPA__": f"{transparencia:.2f}", "__GROSOR__": f"{grosor:g}", "__LETRA__": str(tam_letra), "__ALTO__": str(alto)}
    pagina = _PAGINA
    for a_, b_ in rep.items(): pagina = pagina.replace(a_, b_)
    ruta = SHP.replace(".shp", "_interactivo.html"); open(ruta, "w", encoding="utf-8").write(pagina)
    display(HTML(f'<iframe srcdoc="{html_.escape(pagina)}" style="width:100%;height:{alto + 70}px;border:0"></iframe>'))
    print(f"{len(G)} copas con su probabilidad · la página también quedó en {ruta} (ábrela en el navegador si Colab no la muestra)")
    return ruta


def guardar_umbral(SHP, umbral=0.50):
    """Guarda en un SHP las copas con probabilidad mayor que el umbral que elegiste en la barra."""
    G, col = _copas_con_probabilidad(SHP); T = G[G[col] > umbral]
    ruta = SHP.replace("/topa_", f"/topa_umbral_{umbral:.2f}_".replace(".", "_")); T.to_file(ruta)
    print(f"{len(T)} copas con probabilidad mayor que {umbral:.2f} → {ruta}".replace(".", ",", 1)); return T, ruta

