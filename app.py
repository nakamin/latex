# 楽ちんLatex — 左右2カラム/黄色テーマ/整列修正/コメント付き 最終版

import io, base64, json, re
from typing import Tuple
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, Affine2D

# 先にページ設定
st.set_page_config(page_title="楽ちんLatex", page_icon="🔤", layout="wide")
st.set_option("client.showErrorDetails", False)  # ユーザーにトレースバックを出さない

# =========================
#           CSS
# =========================
ACCENT_BASE = "#f2c94c"   # 基本の黄色
ACCENT      = "#f1b70a"   # 濃い黄色（hover/active）
ACCENT_SOFT = "#ffd874"   # ボーダー用
ACCENT_LIGHT= "#fff7dc"   # 背景用
TEXT_DARK   = "#5a4a00"   # 読みやすい濃色

st.markdown(f"""
<style>
:root {{
  --accent-base: {ACCENT_BASE}; --accent: {ACCENT};
  --accent-soft: {ACCENT_SOFT}; --accent-light: {ACCENT_LIGHT};
  --text-dark: {TEXT_DARK};
}}

/* 上端に見えていた残り枠や空hrを無効化 */
hr {{ border:none; height:0; margin:0; padding:0; }}

/* 共通ボタン（数字/挿入/操作） */
.stButton > button {{
  background: var(--accent-light); border:1.5px solid var(--accent-soft);
  color: var(--text-dark); padding:.52rem 1.0rem; border-radius:10px;
  font-weight:700; transition:all .15s ease;
}}
.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {{
  background: var(--accent) !important; border-color: var(--accent) !important;
  color:#fff !important; box-shadow:none !important; outline:none !important;
}}

/* 枠なし・薄い区切り感だけ */
.card-col{{
  border: none;
  box-shadow: none;
  background: transparent;
  padding: 8px 0 12px;
  margin-bottom: 8px;
}}

/* 大ボタン（コピー/ダウンロード） */
.bigbtn, .bigbtn:link, .bigbtn:visited {{
  display:inline-flex; align-items:center; justify-content:center;
  min-width:320px; padding:1.1rem 1.4rem; font-size:1.14rem; font-weight:800;
  background:var(--accent-light); color:var(--text-dark);
  border:1.5px solid var(--accent-soft); border-radius:12px;
  text-decoration:none; transition:all .15s ease;
}}
.bigbtn:hover, .bigbtn:focus, .bigbtn:active {{
  background: var(--accent); border-color: var(--accent); color:#fff; outline:none;
}}
.cta-row {{ display:flex; gap:20px; justify-content:center; align-items:center; flex-wrap:wrap; margin-top:10px; }}
.cta-note {{ font-size:.95rem; color:#666; width:100%; text-align:center; }}

/* 小プレビューの詰め */
.block {{ margin-bottom:.26rem; }}

/* タイトルロゴの余白 */
.logo-wrap {{ display:flex; justify-content:center; margin:10px 0 2px; }}
</style>
""", unsafe_allow_html=True)

# =========================
#       Matplotlib 設定
# =========================
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# =========================
#  MathJax が必要な式の判定
# =========================
MJ_PAT = re.compile(
    r"""\\begin\{(?:pmatrix|bmatrix|matrix|cases|align\*?|aligned)\}|\\text\{|\\overset|\\underset""",
    re.X,
)
def needs_mathjax(s: str) -> bool:
    """行列・方程式など MathJax でしか描けない構文を検出して True。"""
    return bool(MJ_PAT.search(s or ""))

# =========================
# Matplotlib で LaTeX を画像化（余白px最小）
# =========================
def render_latex_to_image_bytes(
    tex: str,
    fmt: str = "png",
    fontsize: int = 56,
    dpi: int = 300,
    extra_pad_px: int = 2,
    text_color: str = "#000000",
    face_color: str = "#FFFFFF",
    transparent: bool = False,
) -> Tuple[bytes, str]:
    """数式を Matplotlib(mathtext) で描画し、最小の外接 bbox で PNG/SVG を返す。"""
    fig = plt.figure(dpi=dpi)
    fig.patch.set_alpha(0 if transparent else 1)
    s = (tex or "").strip()
    if not (s.startswith("$") and s.endswith("$")):
        s = f"${s}$"
    t = fig.text(0.0, 0.0, s, ha="left", va="bottom", fontsize=fontsize, color=text_color)
    fig.canvas.draw()
    tb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    tb = Bbox.from_extents(tb.x0-extra_pad_px, tb.y0-extra_pad_px, tb.x1+extra_pad_px, tb.y1+extra_pad_px)
    bbox_inches = tb.transformed(Affine2D().scale(1.0/fig.dpi))
    buf = io.BytesIO()
    save_kw = dict(bbox_inches=bbox_inches, pad_inches=0, transparent=transparent,
                   facecolor=None if transparent else face_color)
    if fmt == "png":
        fig.savefig(buf, format="png", dpi=dpi, **save_kw); mime = "image/png"
    elif fmt == "svg":
        fig.savefig(buf, format="svg", **save_kw); mime = "image/svg+xml"
    else:
        plt.close(fig); raise ValueError("fmt は png / svg")
    plt.close(fig)
    return buf.getvalue(), mime

# =========================
# MathJax プレビュー（SVG→PNG 生成/コピー/ダウンロード付き）
# =========================
def render_mathjax_component(tex: str, font_size_px: int = 50,
                             text_color: str = "#000000",
                             transparent: bool = False, bg_color: str = "#FFFFFF") -> None:
    """MathJax で SVG を生成し、PNG 化・コピー・DL をブラウザで行う。"""
    tex_js = json.dumps(tex)
    bg_css = "transparent" if transparent else bg_color
    html = f"""
<div id="wrap"><div id="stage" style="display:inline-block;background:{bg_css};
padding:2px;border:1px dashed #ddd;border-radius:6px;"></div>
<div style="display:flex;gap:.6rem;align-items:center;margin-top:10px;flex-wrap:wrap">
  <button id="copy" class="bigbtn" style="cursor:pointer;">画像をコピー</button>
  <a id="dlsvg" download="latex.svg" class="bigbtn">SVGをダウンロード</a>
  <a id="dlpng" download="latex.png" class="bigbtn">PNGをダウンロード</a>
  <span id="status" style="font-size:.95rem;color:#666;"></span>
</div></div>
<script>
  window.MathJax = window.MathJax || {{tex:{{packages:['base','ams','newcommand','noerrors','noundefined']}}, svg:{{fontCache:'none'}}}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<script>
  const TEX = {tex_js};
  const FONTSIZE = {font_size_px};
  const COLOR = {json.dumps(text_color)};
  const BG = {json.dumps(bg_css)};
  function toDataURL(svg) {{
    const s = new XMLSerializer().serializeToString(svg);
    return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(s)));
  }}
  function render() {{
    const mj = MathJax.tex2svg(TEX, {{display:true}});
    const svg = mj.querySelector('svg');
    svg.setAttribute('xmlns','http://www.w3.org/2000/svg');
    svg.style.fontSize = FONTSIZE + 'px';
    svg.setAttribute('fill', COLOR);
    const stage = document.getElementById('stage');
    stage.innerHTML = '';
    stage.appendChild(svg.cloneNode(true));
    document.getElementById('dlsvg').href = toDataURL(svg);
    const img = new Image();
    img.onload = function() {{
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      const ctx = c.getContext('2d');
      if (BG !== 'transparent') {{ ctx.fillStyle = BG; ctx.fillRect(0,0,c.width,c.height); }}
      ctx.drawImage(img,0,0);
      const png = c.toDataURL('image/png');
      document.getElementById('dlpng').href = png;
      document.getElementById('copy').onclick = () => {{
        c.toBlob(async (blob) => {{
          try {{
            await navigator.clipboard.write([new ClipboardItem({{'image/png': blob}})]);
            document.getElementById('status').textContent = "コピー完了";
          }} catch(e) {{
            document.getElementById('status').textContent = "コピー失敗（HTTPS/ブラウザ制限）";
          }}
        }});
      }};
    }};
    img.src = toDataURL(svg);
  }}
  if (document.readyState === 'complete') render(); else window.addEventListener('load', render);
</script>
"""
    st.components.v1.html(html, height=348, scrolling=False)

def wrap_tex(src: str) -> str:
    """与えられた文字列を $...$ で包む。空ならそのまま返す。"""
    s = (src or "").strip()
    return f"${s.strip('$')}$" if s else s

def render_copy_download_cta(image_bytes: bytes, mime: str) -> None:
    """Matplotlib時の“画像をコピー/ダウンロード”ボタン群。"""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    filename = "latex.png" if mime == "image/png" else "latex.svg"
    html = f"""
<div class="cta-row">
  <button id="copy-btn" class="bigbtn">📋 画像をコピー</button>
  <a id="dl" class="bigbtn" href="{data_url}" download="{filename}">⬇️ ダウンロード</a>
  <span id="copy-status" class="cta-note"></span>
</div>
<script>
  const btn = document.getElementById('copy-btn');
  const status = document.getElementById('copy-status');
  btn.addEventListener('click', async () => {{
    try {{
      const res = await fetch("{data_url}");
      const blob = await res.blob();
      await navigator.clipboard.write([new ClipboardItem({{"{mime}": blob}})]);
      status.textContent = "コピー完了";
    }} catch(e) {{
      status.textContent = "コピー失敗（HTTPS/ブラウザ制限）";
    }}
  }});
</script>
"""
    st.components.v1.html(html, height=124)

# =========================
#            UI
# =========================
def render_title_logo() -> None:
    """中央に SVG ロゴを描画。"""
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='840' height='140' viewBox='0 0 840 140'>
  <defs><linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>
    <stop stop-color='{ACCENT_BASE}' offset='0'/><stop stop-color='{ACCENT_SOFT}' offset='1'/>
  </linearGradient></defs>
  <rect x='0' y='0' width='840' height='140' rx='18' fill='url(#g)' opacity='0.18'/>
  <g font-family='DejaVu Sans, Segoe UI, Arial, sans-serif' fill='{TEXT_DARK}'>
    <text x='70' y='92' font-size='56' font-weight='800'>楽ちん</text>
    <text x='280' y='92' font-size='56' font-weight='800'>La</text>
    <text x='345' y='92' font-size='56' font-style='italic' font-weight='800'>T</text>
    <text x='382' y='92' font-size='56' font-weight='800'>eX</text>
    <text x='490' y='92' font-size='52'>∫</text>
    <text x='530' y='92' font-size='44'>Σ</text>
    <text x='572' y='92' font-size='48'>π</text>
  </g>
</svg>
"""
    st.markdown(f"<div class='logo-wrap'>{svg}</div>", unsafe_allow_html=True)

render_title_logo()

# ---- 状態 ----
FORM_KEY = "formula"
if FORM_KEY not in st.session_state:
    st.session_state[FORM_KEY] = r"\int_0^{\infty} e^{-x^2}\,dx=\frac{\sqrt{\pi}}{2}"
if "undo" not in st.session_state: st.session_state.undo = []
if "force_mj" not in st.session_state: st.session_state.force_mj = False
if "last_img" not in st.session_state: st.session_state.last_img = (None, None)

def push_undo() -> None:
    """現在の式を undo スタックに積む。"""
    st.session_state.undo.append(st.session_state[FORM_KEY])

def insert(tok: str, force_mj: bool = False) -> None:
    """式の末尾にトークンを追記。必要なら MathJax を強制。"""
    push_undo()
    st.session_state[FORM_KEY] += tok
    st.session_state.force_mj |= force_mj

# ---- サイドバー ----
with st.sidebar:
    st.header("描画設定")
    font_px = st.slider("フォントサイズ(px)", 12, 160, 56, 2)
    fmt = st.selectbox("出力フォーマット（通常描画）", ["png", "svg"], index=0)
    dpi = st.slider("DPI（PNG）", 72, 600, 300, 12)
    extra_pad_px = st.slider("追加余白（px）", 0, 16, 2, 1)
    text_color = st.color_picker("文字色", "#000000")
    transparent = st.checkbox("背景を透過（PNG推奨）", value=False)
    bg_color = "#FFFFFF" if not transparent else "#FFFFFF"
    if not transparent:
        bg_color = st.color_picker("背景色（透過OFF時）", "#FFFFFF")

# ===== 2カラム（左: 入力/表示、右: パレット） =====
left_col, right_col = st.columns([1.1, 0.9], gap="large")

# -------- 左：入力・操作・プレビュー --------
with left_col:
    st.markdown('<div class="card-col">', unsafe_allow_html=True)
    st.subheader("LaTeX 入力")

    def cb_undo() -> None:
        if st.session_state.get("undo"):
            st.session_state[FORM_KEY] = st.session_state.undo.pop()
    def cb_clear() -> None:
        st.session_state[FORM_KEY] = ""
    def cb_wrap() -> None:
        st.session_state[FORM_KEY] = wrap_tex(st.session_state[FORM_KEY])

    st.text_area("ここにLaTeXを入力またはパレットで追加", key=FORM_KEY, height=140)
    c1, c2, c3 = st.columns(3)
    with c1: st.button("戻る", on_click=cb_undo, use_container_width=True)
    with c2: st.button("消去", on_click=cb_clear, use_container_width=True)
    with c3: st.button("$で囲む", on_click=cb_wrap, use_container_width=True)

    # 自動プレビュー（MathJax 必要なら切替）
    user_tex = st.session_state[FORM_KEY]
    use_mj = st.session_state.force_mj or needs_mathjax(user_tex)

    # Matplotlib はダブルバックスラッシュを嫌うことがあるので、そのときだけ正規化
    tex_for_render = user_tex if use_mj else user_tex.replace("\\\\", "\\")

    if tex_for_render.strip():
        if use_mj:
            render_mathjax_component(tex_for_render, font_size_px=font_px,
                                    text_color=text_color, transparent=transparent,
                                    bg_color=bg_color)
            st.session_state.last_img = (None, None)  # 下のコピー/DLは非表示
        else:
            try:
                img_bytes, mime = render_latex_to_image_bytes(
                    tex=tex_for_render, fmt=fmt, fontsize=font_px, dpi=dpi,
                    extra_pad_px=extra_pad_px, text_color=text_color,
                    face_color=bg_color, transparent=transparent,
                )
                if mime == "image/png":
                    st.image(img_bytes)
                else:
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    st.markdown(f'![svg](data:image/svg+xml;base64,{b64})', unsafe_allow_html=True)
                st.session_state.last_img = (img_bytes, mime)
            except Exception:
                st.warning("表示エラー！行列/方程式を使用してみてください。")
                st.session_state.last_img = (None, None)
    else:
        st.write("（数式を入力するとここに表示されます）")

    # Matplotlib 時のみコピー/ダウンロードを出す
    img_bytes, mime = st.session_state.last_img
    if img_bytes is not None:
        render_copy_download_cta(img_bytes, mime)

    st.session_state.force_mj = False
    st.markdown("</div>", unsafe_allow_html=True)

# -------- 右：パレット（単一展開） --------
# -------- 右：パレット（単一展開） --------
with right_col:
    st.subheader("パレット")

    section = st.radio(
        "開くパレットを選ぶ",
        ["数字パレット", "微分・偏微分・勾配", "行列・方程式", "ギリシャ・演算子"],
        horizontal=True, label_visibility="collapsed",
    )

    # タイル（MathJax 直描画）— 行列でも安全。height は引数で可変。
    def tile(code: str, key: str, force_mj: bool = False, height_px: int = 60) -> None:
        code_js = json.dumps(code)  # JS 文字列に安全に渡す
        html = f"""
<div style='display:inline-block; overflow:visible;'>
  <script>
    window.MathJax = window.MathJax || {{tex:{{packages:['base','ams','newcommand','noerrors','noundefined']}},
                                         svg:{{fontCache:'none'}}}};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <span id="mj-{key}"></span>
  <script>
    (function(){{
      const el  = document.getElementById("mj-{key}");
      const svg = MathJax.tex2svg({code_js}, {{display:true}}).querySelector('svg');
      svg.setAttribute('xmlns','http://www.w3.org/2000/svg');
      svg.style.maxWidth = '100%';   // 横切れ防止
      el.innerHTML = svg.outerHTML;
    }})();
  </script>
</div>
"""
        st.components.v1.html(html, height=height_px)
        st.button("挿入", key=key, use_container_width=True,
                  on_click=insert, args=(code, force_mj))

    # 数字パレット 🔢
    with st.expander("数字パレット 🔢", expanded=(section == "数字パレット")):
        cols = st.columns(5)
        for i in range(10):
            with cols[i % 5]:
                tile(str(i), f"num{i}", height_px=30)

    # 微分・偏微分・勾配
    with st.expander("微分・偏微分・勾配 ✏️", expanded=(section == "微分・偏微分・勾配")):
        items = [
            r"\frac{d}{dx} f(x)", r"\frac{d^2}{dx^2} f(x)", r"\frac{\partial f}{\partial x}",
            r"\nabla f", r"\nabla^2 f", r"J_{ij}=\frac{\partial f_i}{\partial x_j}",
            r"\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}", r"\sum_{i=1}^{n}",
        ]
        cols = st.columns(2)
        for idx, code in enumerate(items):
            with cols[idx % 2]:
                tile(code, f"d{idx}")

    # 行列・整列（ここだけタイル高め）
    with st.expander("行列・方程式 ▦", expanded=(section == "行列・方程式")):
        items = [
            (r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}", True),
            (r"\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}", True),
            (r"\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}", True),
            (r"\begin{cases} ax+by=c \\ dx+ey=f \end{cases}", True),
            (r"\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad-bc", True),
            (r"y=ax+b", False),
        ]
        cols = st.columns(2)
        for idx, (code, fmj) in enumerate(items):
            with cols[idx % 2]:
                tile(code, f"m{idx}", fmj, height_px=100)  # ← 行列だけ高め

    # ギリシャ・演算子
    with st.expander("ギリシャ・演算子 🧮", expanded=(section == "ギリシャ・演算子")):
        greek = [r"\alpha ", r"\beta ", r"\gamma ", r"\lambda ", r"\mu ", r"\sigma ", r"\phi ", r"\pi "]
        ops   = ["+", "-", r"\times ", r"\div ", "=", r"\ne ", r"\pm ", r"\cdot ", "<", ">"]
        items = greek + ops
        NCOL = 5
        for r in range((len(items)+NCOL-1)//NCOL):
            cols = st.columns(NCOL)
            for c in range(NCOL):
                i = r*NCOL + c
                if i >= len(items): break
                with cols[c]:
                    tile(items[i], f"gx{i}", height_px=30)

    st.markdown("</div>", unsafe_allow_html=True)