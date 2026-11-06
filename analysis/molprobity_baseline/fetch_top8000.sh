#!/usr/bin/env bash
# Top8000 rotamer and Ramachandran percentile contour grids (Williams et al. 2018, Protein Sci 27:293),
# from github.com/rlabduke/reference_data (CC BY 4.0), checksum-verified. Idempotent.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)/top8000"; B=https://raw.githubusercontent.com/rlabduke/reference_data/master
mkdir -p "$D/rotamer" "$D/rama"
get() {  # sub-dir, file, repo path, sha256
  f="$D/$1/$2"
  if [ ! -s "$f" ] || [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" != "$4" ]; then curl -sf --retry 3 -o "$f" "$B/$3/$2"; fi
  [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" = "$4" ] || { echo "checksum mismatch: $2"; exit 1; }
}
get rotamer rota8000-arg.data Top8000/Top8000_rotamer_pct_contour_grids 7632f3b256c1cb9adfcdb9b9498f9d8ce6f7a7d641f668e4929e98718f6f52c9
get rotamer rota8000-asn.data Top8000/Top8000_rotamer_pct_contour_grids fa42b353e8c8fe662d78764f8346d8c65d8cc2a2fcbdb9681a8e3244cf39a5e1
get rotamer rota8000-asp.data Top8000/Top8000_rotamer_pct_contour_grids a144e6188b5358fa34b0ac2e81a2ee80446384bf7cec62d1dabf1eaf817f82c6
get rotamer rota8000-cys.data Top8000/Top8000_rotamer_pct_contour_grids 1d4a83389f6fd5c6ddb73799ca42927c80c62db7c742361df44a740155e60484
get rotamer rota8000-gln.data Top8000/Top8000_rotamer_pct_contour_grids 4182caba7f26506522365d6b4e7ec940f010ecbb32bcd536011496a9ecd506ad
get rotamer rota8000-glu.data Top8000/Top8000_rotamer_pct_contour_grids e4806705048e0a8e750c3848f5db1f9b5d4b52bec95ac243da6a5794c26af2e4
get rotamer rota8000-his.data Top8000/Top8000_rotamer_pct_contour_grids 42885096613cc8e75416a837304e353b37f0104d3838b3ba348b92b027c7ebdd
get rotamer rota8000-ile.data Top8000/Top8000_rotamer_pct_contour_grids 381cb7275d5490720dc94361a1eaf3561fb7f2e75ecb1c5bd8badd4698f42590
get rotamer rota8000-leu.data Top8000/Top8000_rotamer_pct_contour_grids d938e27e4b1bd6e6afeb0a62b157e57e70ff3d0d1a31fee5052d5ff01217f20e
get rotamer rota8000-lys.data Top8000/Top8000_rotamer_pct_contour_grids 2ba990118adb9ef63e2e230a89675b7ba4b7d1bdfd4b9d703c0f1f8079d835e9
get rotamer rota8000-met.data Top8000/Top8000_rotamer_pct_contour_grids 1ff4dc554a85a34cd53b16e1276277c772caa810a0f7fc0cf7938af1cfda7603
get rotamer rota8000-phetyr.data Top8000/Top8000_rotamer_pct_contour_grids 50fb18a362cb114478e9c5e8375eb28f8c49403c3e44af03fd6e8dd17c5a3a88
get rotamer rota8000-pro.data Top8000/Top8000_rotamer_pct_contour_grids 63836f5d6ea38af5b413094811266d9cd320f9a1cfdbad2e180be38030a808e1
get rotamer rota8000-ser.data Top8000/Top8000_rotamer_pct_contour_grids 29cfe12bbd2e5d83c26ad56e139f088275989748e097306cd7a3e230c5b062c8
get rotamer rota8000-thr.data Top8000/Top8000_rotamer_pct_contour_grids 22a824fefb6ce9b55d2f6a394465b58527196e4530669b4ac6009e3ca989327a
get rotamer rota8000-trp.data Top8000/Top8000_rotamer_pct_contour_grids a08b04e57f485898d0bcf22a2fe0f61acdd3401fb9bd84f0715c7a6971c7d7b7
get rotamer rota8000-val.data Top8000/Top8000_rotamer_pct_contour_grids 38ab7a37fe6fcff545160f2bb620a3ae4b20a9a033de838049a0b141997e3660
get rama rama8000-cispro.data Top8000/Top8000_ramachandran_pct_contour_grids 143a3004668baefaf6749bd9c2acbe829d3b1a93059c74ba766b04f1e00f43ab
get rama rama8000-general-noGPIVpreP.data Top8000/Top8000_ramachandran_pct_contour_grids ccdbc6a201ca2510119e77b0dd169ca2cdea7f2c9ed348555a3b1129f8c2b00a
get rama rama8000-gly-sym.data Top8000/Top8000_ramachandran_pct_contour_grids 89c75a5ac036ff3309c51b46a2412f30137827abaff4606209ed41e04fcba637
get rama rama8000-ileval-nopreP.data Top8000/Top8000_ramachandran_pct_contour_grids 567e1318128a50b8f44362427bf717d564749936d22d5dbfd34e5b0c17819276
get rama rama8000-prepro-noGP.data Top8000/Top8000_ramachandran_pct_contour_grids 1cc10b92911d47ca775f0a8131a029b42634dbe26364cece4cfdc8c85b9c3fab
get rama rama8000-transpro.data Top8000/Top8000_ramachandran_pct_contour_grids 092b4c0bcd2fe846a063000c83d13cbe62ff43b2dafd67ecb8a47042cce747ec
echo "[top8000] 23 grids present and verified in $D"
