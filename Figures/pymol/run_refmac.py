import ccp4
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

#change the path if different path to ccp4
ccp4.setup('~/ccp4-7.0/bin/ccp4.setup-sh')

runs=[
    {'XYZIN': '1fpx_SAM_1699_A/1fpx_SAH.pdb', 'XYZOUT': '1fpx_SAM_1699_A/1fpx_SAH_ref.pdb', 'HKLIN': '1fpx_SAM_1699_A/1fpx.mtz', 'HKLOUT': '1fpx_SAM_1699_A/1fpx_SAH_ref.mtz'},
    {'XYZIN': '1kcz_EDO_903_A/1kcz_GOL.pdb', 'XYZOUT': '1kcz_EDO_903_A/1kcz_GOL_ref.pdb', 'HKLIN': '1kcz_EDO_903_A/1kcz.mtz', 'HKLOUT': '1kcz_EDO_903_A/1kcz_ref.mtz'},
    {'XYZIN': '1kwn_TAR_501_A/1kwn_TLA.pdb', 'XYZOUT': '1kwn_TAR_501_A/1kwn_TLA_ref.pdb', 'HKLIN': '1kwn_TAR_501_A/1kwn.mtz', 'HKLOUT': '1kwn_TAR_501_A/1kwn_ref.mtz'},
    {'XYZIN': '1yqs_GOL_502_A/1yqs_EDO_H2O.pdb', 'XYZOUT': '1yqs_GOL_502_A/1yqs_EDO_H2O_ref.pdb', 'HKLIN': '1yqs_GOL_502_A/1yqs.mtz', 'HKLOUT': '1yqs_GOL_502_A/1yqs_ref.mtz'},
    {'XYZIN': '1yuk_NDG_463_B/1yuk_NAG.pdb', 'XYZOUT': '1yuk_NDG_463_B/1yuk_NAG_ref.pdb', 'HKLIN': '1yuk_NDG_463_B/1yuk.mtz', 'HKLOUT': '1yuk_NDG_463_B/1yuk_ref.mtz'},
    {'XYZIN': '2e9m_GAL_1402_A/2e9m_TRS.pdb', 'XYZOUT': '2e9m_GAL_1402_A/2e9m_TRS_ref.pdb', 'HKLIN': '2e9m_GAL_1402_A/2e9m.mtz', 'HKLOUT': '2e9m_GAL_1402_A/2e9m_ref.mtz'},
    {'XYZIN': '2pdt_FAD_204_D/2pdt_FMN.pdb', 'XYZOUT': '2pdt_FAD_204_D/2pdt_FMN_ref.pdb', 'HKLIN': '2pdt_FAD_204_D/2pdt.mtz', 'HKLOUT': '2pdt_FAD_204_D/2pdt_ref.mtz'},
    {'XYZIN': '3a0k_ABA_240_G/3a0k_GOL.pdb', 'XYZOUT': '3a0k_ABA_240_G/3a0k_GOL_ref.pdb', 'HKLIN': '3a0k_ABA_240_G/3a0k.mtz', 'HKLOUT': '3a0k_ABA_240_G/3a0k_ref.mtz'},
    {'XYZIN': '3s1y_IPA_400_A/3s1y_IMD.pdb', 'XYZOUT': '3s1y_IPA_400_A/3s1y_IMD_ref.pdb', 'HKLIN': '3s1y_IPA_400_A/3s1y.mtz', 'HKLOUT': '3s1y_IPA_400_A/3s1y_ref.mtz'},
    {'XYZIN': '4j4z_PEG_203_A/4j4z_EDO.pdb', 'XYZOUT': '4j4z_PEG_203_A/4j4z_EDO_ref.pdb', 'HKLIN': '4j4z_PEG_203_A/4j4z.mtz', 'HKLOUT': '4j4z_PEG_203_A/4j4z_ref.mtz'},
    {'XYZIN': '4rk3_GOL_401_A/4rk3_TRS.pdb', 'XYZOUT': '4rk3_GOL_401_A/4rk3_TRS_ref.pdb', 'HKLIN': '4rk3_GOL_401_A/4rk3.mtz', 'HKLOUT': '4rk3_GOL_401_A/4rk3_ref.mtz'},
    {'XYZIN': '4xfw_ACY_304_A/4xfw_NO3.pdb', 'XYZOUT': '4xfw_ACY_304_A/4xfw_NO3_ref.pdb', 'HKLIN': '4xfw_ACY_304_A/4xfw.mtz', 'HKLOUT': '4xfw_ACY_304_A/4xfw_ref.mtz'},
    {'XYZIN': '4xgo_GOL_1008_A/4xgo_NAG.pdb', 'XYZOUT': '4xgo_GOL_1008_A/4xgo_NAG_ref.pdb', 'HKLIN': '4xgo_GOL_1008_A/4xgo.mtz', 'HKLOUT': '4xgo_GOL_1008_A/4xgo_ref.mtz'},
    {'XYZIN': '5aou_PO4_5001_A/5aou_SO4.pdb', 'XYZOUT': '5aou_PO4_5001_A/5aou_SO4_ref.pdb', 'HKLIN': '5aou_PO4_5001_A/5aou.mtz', 'HKLOUT': '5aou_PO4_5001_A/5aou_ref.mtz'},
    {'XYZIN': '5aou_PO4_5001_A/5aou_PO4.pdb', 'XYZOUT': '5aou_PO4_5001_A/5aou_PO4_ref.pdb', 'HKLIN': '5aou_PO4_5001_A/5aou.mtz', 'HKLOUT': '5aou_PO4_5001_A/5aou_PO4_ref.mtz'},
    {'XYZIN': '5fle_FES_1002_X/5fle_SF4.pdb', 'XYZOUT': '5fle_FES_1002_X/5fle_SF4_ref.pdb', 'HKLIN': '5fle_FES_1002_X/5fle.mtz', 'HKLOUT': '5fle_FES_1002_X/5fle_ref.mtz', 'LIBIN': '5fle_FES_1002_X/5fle_SF4_ref.cif', 'LIBOUT': '5fle_FES_1002_X/5fle_SF4_ref.cif'},
]


LABIN_FP = 'FP'
LABIN_SIGFP = 'SIGFP'
REFMAC_NCYC = 5
REFMAC_WEIGHT = 'AUTO'

keywords = {
    'LABIN': 'FP=%s SIGFP=%s' % (LABIN_FP, LABIN_SIGFP),
    'LABOUT': 'FC=FC FWT=FWT PHIC=PHIC PHWT=PHWT DELFWT=DELFWT PHDELWT=PHDELWT FOM=FOM',
    'NCYC': REFMAC_NCYC,
    'WEIGHT': REFMAC_WEIGHT,
}

for files in runs:

    print('Running REFMAC', files)
    refmac = ccp4.CCP4_REFMAC(files, keywords)
    refmac.run()

    with open(files['XYZIN'].replace('.pdb', '_ref.log'), 'w') as refmac_log:
        refmac_log.write(refmac.output)
