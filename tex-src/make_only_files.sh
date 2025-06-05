# --- ❶ tex-src ディレクトリへ移動（既に居るなら不要）-------------
cd /Users/hiroyukiyamanaka/Desktop/t3Stock_stash/tex-src

# --- ❷ 必要な .tex ファイルを一気に生成 --------------------------
cat <<'EOF' > __make_only_tex.sh
#!/usr/bin/env bash
set -e

# 親ファイル
cat > parent.tex <<'PARENT'
%=== parent.tex ==========================================
\documentclass[dvipdfmx,openany]{jsbook}
\usepackage{amsmath,amssymb,tabularx,booktabs}
\renewcommand{\arraystretch}{1.2}
\begin{document}
\input{basic_form.tex}
\input{phase0.tex}
\input{phase1.tex}
\input{phase2.tex}
\input{phase3.tex}
\input{phase4.tex}
\end{document}
%=========================================================
PARENT

# --- 子／孫ファイル -------------------------------------
cat > basic_form.tex <<'BASIC'
%=== 基本形（共通） ======================================
% 1. 基本形：日中レンジを伴うモデル
% （ここに先ほどの「基本形」数式と表を貼り付け）
BASIC

cat > phase0.tex <<'P0'
%=== フェーズ0（EWMA ギャップ） ==========================
% （sample.tex のフェーズ0 相当部分をコピー）
P0

cat > phase1.tex <<'P1'
%=== フェーズ1（ギャップ分布スケーリング） ===============
% （詳細説明＋数式）
P1

cat > phase2.tex <<'P2'
%=== フェーズ2（ボラティリティ連動補正） ================
% （詳細説明＋数式）
P2

cat > phase3.tex <<'P3'
%=== フェーズ3（Proxy Board Gap 補正） ===================
% （詳細説明＋数式）
P3

cat > phase4.tex <<'P4'
%=== フェーズ4（自己適応 \lambda 更新） ==================
% （ステップ1〜5の丁寧な解説＋数式）
P4
EOF

# スクリプトを走らせてファイルを生成
bash __make_only_tex.sh
rm  __make_only_tex.sh          # 後片付け