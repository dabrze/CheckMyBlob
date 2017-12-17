
for x in "1fpx_SAM_1699_A/1fpx" "1yqs_GOL_502_A/1yqs" "3a0k_ABA_240_G/3a0k" "4rk3_GOL_401_A/4rk3" "5aou_PO4_5001_A/5aou" "1kcz_EDO_903_A/1kcz" "1yuk_NDG_463_B/1yuk" "3s1y_IPA_400_A/3s1y" "4xfw_ACY_304_A/4xfw" "5fle_FES_1002_X/5fle" "1kwn_TAR_501_A/1kwn" "2e9m_GAL_1402_A/2e9m" "4j4z_PEG_203_A/4j4z" "4xgo_GOL_1008_A/4xgo"; do
echo "Run $x"
pdb="$x.pdb"
mtz="$x""_refmac.mtz"
map="$x"

echo "$pdb $mtz $map"
fft hklin $mtz mapout $map'_2FoFc.map.ccp4' \
<<eof
title 2mFo-dFc map
labi F1=FWT PHI=PHWT
end
eof

fft hklin $mtz mapout $map'_FoFc.map.ccp4' \
<<eof
title mFo-dFc map
labi F1=DELFWT PHI=PHDELWT
end 
eof

mapmask mapin $map'_2FoFc.map.ccp4' xyzin $pdb mapout $map'_2FoFc.cover.ccp4' \
<<eof
BORDER 5.0
end 
eof

mapmask mapin $map'_FoFc.map.ccp4' xyzin $pdb mapout $map'_FoFc.cover.ccp4' \
<<eof
BORDER 5.0
end 
eof
done
