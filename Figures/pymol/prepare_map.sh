
for x in "3wh1" "4b9a" "4j4z" "4y1u" "4iun" "3mb5" "1ogv" "5n0h"; do
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
