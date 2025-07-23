"""dictionary to store config for all the variables for plotting"""
# todo add variable binning or limited ranges where needed.
var_dict = {
    'xsec': {
        "label": "cross section [fb]",
        "dtype": "float32",
     },
    'nEvents': {
        "label": "# events",
        "dtype": "int32",
     },
    'genWeight': {
        "label": "generator weight",
        "dtype": "float32",
     },
    'bjet1eta': {
        "label": r"$\eta(b_{1})$",
        "dtype": "float32",
     },
    'bjet2eta': {
        "label": r"$\eta(b_{2})$",
        "dtype": "float32",
     },
    'bjet1phi': {
        "label": r"$\phi(b_{1})$",
        "dtype": "float32",
     },
    'bjet2phi': {
        "label": r"$\phi(b_{2})$",
        "dtype": "float32",
     },
    'bjet1pt': {
        "label": r"$p_T(b_{1})$ [GeV]",
        "dtype": "float32",
     },
    'bjet2pt': {
        "label": r"$p_T(b_{2})$ [GeV]",
        "dtype": "float32",
     },
    'lep1eta': {
        "label": r"$\eta(l_{1})$",
        "dtype": "float32",
     },
    'lep2eta': {
        "label": r"$\eta(l_{2})$",
        "dtype": "float32",
     },
    'lep1phi': {
        "label": r"$\phi(l_{1})$",
        "dtype": "float32",
     },
    'lep2phi': {
        "label": r"$\phi(l_{2})$",
        "dtype": "float32",
     },
    'lep1pt': {
        "label": r"$p_T(l_{1})$ [GeV]",
        "dtype": "float32",
     },
    'lep2pt': {
        "label": r"$p_T(l_{2})$ [GeV]",
        "dtype": "float32",
     },
    'njets': {
        "label": r"$n_{jets}$",
        "dtype": "int32",
     },
    'nbjets': {
        "label": r"$\n_{b}$",
        "dtype": "int32",
     },
    'met': {
        "label": r"$p_{T}^{miss}$ [GeV]",
        "dtype": "float32",
     },
    'metphi': {
        "label": r"$\phi^{miss}$",
        "dtype": "float32",
     },
    'metsigHt': {
        "label": r"$p_{T}^{miss}/H_{T}~[\sqrt{GeV}]$",
        "dtype": "float32",
     },
    'sumptllbb': {
        "label": r"$H_{T}$ [GeV]",
        "dtype": "float32",
     },
    'sumptllbbMET': {
        "label": r"$H_{T} + p_{T}^{miss}$ [GeV]",
        "dtype": "float32",
     },
    'mt2': {
        "label": r"$M_{T2}$ [GeV]",
        "dtype": "float32",
     },
    'mindPhiMETl': {
        "label": r"$min(\Delta\phi(p_{T}^{miss},l))$",
        "dtype": "float32",
     },
    'maxdPhiMETl': {
        "label": r"$max(\Delta\phi(p_{T}^{miss},l))$",
        "dtype": "float32",
     },
    'mindPhiMETb': {
        "label": r"$min(\Delta\phi(p_{T}^{miss},b))$",
        "dtype": "float32",
     },
    'maxdPhiMETb': {
        "label": r"$max(\Delta\phi(p_{T}^{miss},b))$",
        "dtype": "float32",
     },
    'avedPhiMETl': {
        "label": r"$<(\Delta\phi(p_{T}^{miss},l))>$",
        "dtype": "float32",
     },
    'avedPhiMETb': {
        "label": r"$<(\Delta\phi(p_{T}^{miss},b))>$",
        "dtype": "float32",
     },
    'mtl1': {
        "label": r"$m_{T}(l_{1})$ [GeV]",
        "dtype": "float32",
     },
    'mtl2': {
        "label": r"$m_{T}(l_{2})$ [GeV]",
        "dtype": "float32",
     },
    'mtlb1': {
        "label": r"$m_{T}(l,b)-close$ [GeV]",
        "dtype": "float32",
     },
    'mtlb2': {
        "label": r"$m_{T}(l,b)-far$ [GeV]",
        "dtype": "float32",
     },
    'mtlmin': {
        "label": r"$min(m_{T}(l))$ [GeV]",
        "dtype": "float32",
     },
    'mtlbmin': {
        "label": r"$min(m_{T}(l,b))$ [GeV]",
        "dtype": "float32",
     },
    'summtlb': {
        "label": r"$\Sigma(m_{T}(l,b))$ [GeV]",
        "dtype": "float32",
     },
    'summtl': {
        "label": r"$\Sigma(m_{T}(p_{T}^{miss},l))$ [GeV]",
        "dtype": "float32",
     },
    'dPhil1MET': {
        "label": r"$\Delta\phi(p_{T}^{miss},l_{1})$",
        "dtype": "float32",
     },
    'dPhil2MET': {
        "label": r"$\Delta\phi(p_{T}^{miss},l_{2})",
        "dtype": "float32",
     },
    'dPhib1MET': {
        "label": r"$\Delta\phi(p_{T}^{miss},b_{1})$",
        "dtype": "float32",
     },
    'dPhib2MET': {
        "label": r"$\Delta\phi(p_{T}^{miss},b_{2})$",
        "dtype": "float32",
     },
    'dRl1b1': {
        "label": r"$\Delta R(l_{1}, b_{1})$",
        "dtype": "float32",
     },
    'dRl1b2': {
        "label": r"$\Delta R(l_{1}, b_{2})$",
        "dtype": "float32",
     },
    'dRl2b1': {
        "label": r"$\Delta R(l_{2}, b_{1})$",
        "dtype": "float32",
     },
    'dRl2b2': {
        "label": r"$\Delta R(l_{2}, b_{2})$",
        "dtype": "float32",
     },
    'sumdRlb': {
        "label": r"$\Sigma(\Delta R(l,b))$",
        "dtype": "float32",
     },
    'mindRlb': {
        "label": r"$min(\Delta R(l,b))$",
        "dtype": "float32",
     },
    'invsumdRlb': {
        "label": r"$1/Sigma(\Delta R(l,b))$",
        "dtype": "float32",
     },
    'invmindRlb': {
        "label": r"$1/min(\Delta R(l,b))",
        "dtype": "float32",
     },
    'mH1': {
        "label": r'$m(H_{1})$ [GeV]',
        "dtype": "float32",
     },
    'mH2': {
        "label": r'$m(H_{2})$ [GeV]',
        "dtype": "float32",
     },
    'mH3': {
        "label": r'$m(H_{3})$ [GeV]',
        "dtype": "float32",
     },
    'mHHH': {
        "label": r'$m(HHH)$ [GeV]',
        "dtype": "float32",
     },
    'dRH1': {
        "label": r'$\Delta R(H_{1})$',
        "dtype": "float32",
     },
    'dRH2': {
        "label": r'$\Delta R(H_{2})$',
        "dtype": "float32",
     },
    'dRH3': {
        "label": r'$\Delta R(H_{3})$',
        "dtype": "float32",
     },
    'meandRBB': {
        "label": r'$<\Delta R(jj)>$',
        "dtype": "float32",
     },
    'sphere3dv2b': {
        "label": r'Sphericity$_{6j}$',
        "dtype": "float32",
     },
    'sphere3dv2btrans': {
        "label": 'Transverse Sphericity$_{6j}$',
        "dtype": "float32",
     },
    'aplan3dv2b': {
        "label": r'Aplanarity$_{6jets}$',
        "dtype": "float32",
     },
    'theta3dv2b': {
        "label": r'$\theta_{6jets}$',
        "dtype": "float32",
    }
}

for feature in range(0, 21):
    var_dict[f"feat_{feature+1:02d}"] = {
        "label": f"Feature {feature+1:02d}",
        "dtype": "float32",
    }
