########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Two dimensional, seven energy group problem using multiple materials 
# based off of M.A. Smith et al. "Benchmark on Deterministic Transport 
# Calculations Without Spatial Homogenisation: A 2-D/3-D MOX Fuel 
# Assembly Benchmark" OECD/NEA report, NEA/NSC/DOC(2003)16 (2003)
# Appendix A. This document is for creating the cross sections and 
# saving them to the file "critical_2d_C5G7_xs.h5".
# 
########################################################################

import numpy as np
import h5py

groups = 7

########################################################################
# UO2 fuel XS
########################################################################
# uo2_total = np.array([2.12450E-01, 3.55470E-01, 4.85540E-01, 5.59400E-01, \
#                         3.18030E-01, 4.01460E-01, 5.70610E-01])
uo2_total = np.array([1.77949E-1, 3.29805E-1, 4.80388E-1, 5.54367E-1,
                      3.11801E-1, 3.95168E-1, 5.64406E-1])

uo2_sigmaf = np.array([7.21206E-03, 8.19301E-04, 6.45320E-03, 1.85648E-02, \
                        1.78084E-02, 8.30348E-02, 2.16004E-01])
uo2_nu = np.array([2.78145E+00, 2.47443E+00, 2.43383E+00, 2.43380E+00, \
                        2.43380E+00, 2.43380E+00, 2.43380E+00])
uo2_chi = np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, \
                        0.00000E+00, 0.00000E+00, 0.00000E+00])
uo2_fission = (uo2_chi[:,None] @ (uo2_nu * uo2_sigmaf)[None,:]).T
uo2_scatter = np.array([
[1.27537E-01, 4.23780E-02, 9.43740E-06, 5.51630E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 3.24456E-01, 1.63140E-03, 3.14270E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 4.50940E-01, 2.67920E-03, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 4.52565E-01, 5.56640E-03, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 1.25250E-04, 2.71401E-01, 1.02550E-02, 1.00210E-08],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.29680E-03, 2.65802E-01, 1.68090E-02],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 8.54580E-03, 2.73080E-01]]) 

UO2 = {"xs_total": uo2_total, "xs_scatter": uo2_scatter, "xs_fission": uo2_fission, \
       "chi": uo2_chi, "nu": uo2_nu, "sigmaf": uo2_sigmaf}

########################################################################
# 4.3% MOX fuel XS
########################################################################
# mox43_total = np.array([2.11920E-01, 3.55810E-01, 4.88900E-01, 5.71940E-01, \
#                             4.32390E-01, 6.84950E-01, 6.88910E-01])
mox43_total = np.array([1.78731E-01, 3.30849E-01, 4.83772E-01, 5.66922E-01, \
                            4.26227E-01, 6.78997E-01, 6.82852E-01])

mox43_sigmaf = np.array([7.62704E-03, 8.76898E-04, 5.69835E-03, 2.28872E-02, \
                            1.07635E-02, 2.32757E-01, 2.48968E-01])
mox43_nu = np.array([2.85209E+00, 2.89099E+00, 2.85486E+00, 2.86073E+00, \
                            2.85447E+00, 2.86415E+00, 2.86780E+00])
mox43_chi = np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, \
                            0.00000E+00, 0.00000E+00, 0.00000E+00])
mox43_fission = (mox43_chi[:,None] @ (mox43_nu * mox43_sigmaf)[None,:]).T
mox43_scatter = np.array([
[1.28876E-01, 4.14130E-02, 8.22900E-06, 5.04050E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 3.25452E-01, 1.63950E-03, 1.59820E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 4.53188E-01, 2.61420E-03, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 4.57173E-01, 5.53940E-03, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 1.60460E-04, 2.76814E-01, 9.31270E-03, 9.16560E-09],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 2.00510E-03, 2.52962E-01, 1.48500E-02],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 8.49480E-03, 2.65007E-01]])

MOX43 = {"xs_total": mox43_total, "xs_scatter": mox43_scatter, "xs_fission": mox43_fission, \
         "chi": mox43_chi, "nu": mox43_nu, "sigmaf": mox43_sigmaf}

########################################################################
# 7.0% MOX fuel XS
########################################################################
# mox70_total = np.array([2.14540E-01, 3.59350E-01, 4.98910E-01, 5.96220E-01, \
#                             4.80350E-01, 8.39360E-01, 8.59480E-01])
mox70_total = np.array([1.81323E-01, 3.34368E-01, 4.93785E-01, 5.91216E-01, \
                            4.74198E-01, 8.33601E-01, 8.53603E-01])

mox70_sigmaf = np.array([8.25446E-03, 1.32565E-03, 8.42156E-03, 3.28730E-02, \
                            1.59636E-02, 3.23794E-01, 3.62803E-01])
mox70_nu = np.array([2.88498E+00, 2.91079E+00, 2.86574E+00, 2.87063E+00, \
                            2.86714E+00, 2.86658E+00, 2.87539E+00])
mox70_chi = np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, \
                            0.00000E+00, 0.00000E+00, 0.00000E+00])
mox70_fission = (mox70_chi[:,None] @ (mox70_nu * mox70_sigmaf)[None,:]).T
mox70_scatter = np.array([
[1.30457E-01, 4.17920E-02, 8.51050E-06, 5.13290E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 3.28428E-01, 1.64360E-03, 2.20170E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 4.58371E-01, 2.53310E-03, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 4.63709E-01, 5.47660E-03, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 1.76190E-04, 2.82313E-01, 8.72890E-03, 9.00160E-09],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 2.27600E-03, 2.49751E-01, 1.31140E-02],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 8.86450E-03, 2.59529E-01]])

MOX70 = {"xs_total": mox70_total, "xs_scatter": mox70_scatter, "xs_fission": mox70_fission, \
         "chi": mox70_chi, "nu": mox70_nu, "sigmaf": mox70_sigmaf}


########################################################################
# 8.7% MOX fuel XS  
########################################################################
# mox87_total = np.array([2.16280E-01, 3.61700E-01, 5.05630E-01, 6.11170E-01, \
#                             5.08900E-01, 9.26670E-01, 9.60990E-01])
mox87_total = np.array([1.83045E-01, 3.36705E-01, 5.00507E-01, 6.06174E-01, \
                            5.02754E-01, 9.21028E-01, 9.55231E-01])

mox87_sigmaf = np.array([8.67209E-03, 1.62426E-03, 1.02716E-02, 3.90447E-02, \
                            1.92576E-02, 3.74888E-01, 4.30599E-01])
mox87_nu = np.array([2.90426E+00, 2.91795E+00, 2.86986E+00, 2.87491E+00, \
                            2.87175E+00, 2.86752E+00, 2.87808E+00])
mox87_chi = np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, \
                            0.00000E+00, 0.00000E+00, 0.00000E+00])
mox87_fission = (mox87_chi[:,None] @ (mox87_nu * mox87_sigmaf)[None,:]).T
mox87_scatter = np.array([
[1.31504E-01, 4.20460E-02, 8.69720E-06, 5.19380E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 3.30403E-01, 1.64630E-03, 2.60060E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 4.61792E-01, 2.47490E-03, 0.00000E+00, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 4.68021E-01, 5.43300E-03, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 1.85970E-04, 2.85771E-01, 8.39730E-03, 8.92800E-09],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 2.39160E-03, 2.47614E-01, 1.23220E-02],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 8.96810E-03, 2.56093E-01]])

MOX87 = {"xs_total": mox87_total, "xs_scatter": mox87_scatter, "xs_fission": mox87_fission, \
         "chi": mox87_chi, "nu": mox87_nu, "sigmaf": mox87_sigmaf}


########################################################################
# Fission chamber XS
########################################################################
# fc_total = np.array([1.90730E-01, 4.56520E-01, 6.40700E-01, 6.49840E-01, \
#                         6.70630E-01, 8.75060E-01, 1.43450E+00])
fc_total = np.array([1.26032E-01, 2.93160E-01, 2.84250E-01, 2.81020E-01, \
                        3.34460E-01, 5.65640E-01, 1.17214E+00])

fc_sigmaf = np.array([4.79002E-09, 5.82564E-09, 4.63719E-07, 5.24406E-06, \
                        1.45390E-07, 7.14972E-07, 2.08041E-06])
fc_nu = np.array([2.76283E+00, 2.46239E+00, 2.43380E+00, 2.43380E+00, \
                        2.43380E+00, 2.43380E+00, 2.43380E+00])
fc_chi = np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, \
                        0.00000E+00, 0.00000E+00, 0.00000E+00])
fc_fission = (fc_chi[:,None] @ (fc_nu * fc_sigmaf)[None,:]).T
fc_scatter = np.array([
[6.61659E-02, 5.90700E-02, 2.83340E-04, 1.46220E-06, 2.06420E-08, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 2.40377E-01, 5.24350E-02, 2.49900E-04, 1.92390E-05, 2.98750E-06, 4.21400E-07],
[0.00000E+00, 0.00000E+00, 1.83425E-01, 9.22880E-02, 6.93650E-03, 1.07900E-03, 2.05430E-04],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 7.90769E-02, 1.69990E-01, 2.58600E-02, 4.92560E-03],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 3.73400E-05, 9.97570E-02, 2.06790E-01, 2.44780E-02],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 9.17420E-04, 3.16774E-01, 2.38760E-01],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.97930E-02, 1.09910E+00]])

fission_chamber = {"xs_total": fc_total, "xs_scatter": fc_scatter, "xs_fission": fc_fission, \
                    "chi": fc_chi, "nu": fc_nu, "sigmaf": fc_sigmaf}

########################################################################
# Guide Tube XS
########################################################################
# gt_total = np.array([1.90730E-01, 4.56520E-01, 6.40670E-01, 6.49670E-01, \
#                         6.70580E-01, 8.75050E-01, 1.43450E+00])
gt_total = np.array([1.26032E-01, 2.93160E-01, 2.84240E-01, 2.80960E-01, \
                        3.34440E-01, 5.65640E-01, 1.17215E+00])

gt_fission = np.zeros((groups, groups))
gt_scatter = np.array([
[6.61659E-02, 5.90700E-02, 2.83340E-04, 1.46220E-06, 2.06420E-08, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 2.40377E-01, 5.24350E-02, 2.49900E-04, 1.92390E-05, 2.98750E-06, 4.21400E-07],
[0.00000E+00, 0.00000E+00, 1.83297E-01, 9.23970E-02, 6.94460E-03, 1.08030E-03, 2.05670E-04],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 7.88511E-02, 1.70140E-01, 2.58810E-02, 4.92970E-03],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 3.73330E-05, 9.97372E-02, 2.06790E-01, 2.44780E-02],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 9.17260E-04, 3.16765E-01, 2.38770E-01],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.97920E-02, 1.09912E+00]])

guide_tubes = {"xs_total": gt_total, "xs_scatter": gt_scatter, "xs_fission": gt_fission}

########################################################################
# Moderator
########################################################################
# md_total = np.array([2.30070E-01, 7.76460E-01, 1.48420E+00, 1.50520E+00, \
#                         1.55920E+00, 2.02540E+00, 3.30570E+00])
md_total = np.array([1.59206E-01, 4.12970E-01, 5.90310E-01, 5.84350E-01, \
                        7.18000E-01, 1.25445E+00, 2.65038E+00])

md_fission = np.zeros((groups, groups))
md_scatter = np.array([
[4.44777E-02, 1.13400E-01, 7.23470E-04, 3.74990E-06, 5.31840E-08, 0.00000E+00, 0.00000E+00],
[0.00000E+00, 2.82334E-01, 1.29940E-01, 6.23400E-04, 4.80020E-05, 7.44860E-06, 1.04550E-06],
[0.00000E+00, 0.00000E+00, 3.45256E-01, 2.24570E-01, 1.69990E-02, 2.64430E-03, 5.03440E-04],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 9.10284E-02, 4.15510E-01, 6.37320E-02, 1.21390E-02],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 7.14370E-05, 1.39138E-01, 5.11820E-01, 6.12290E-02],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 2.21570E-03, 6.99913E-01, 5.37320E-01],
[0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.32440E-01, 2.48070E+00]])

moderator = {"xs_total": md_total, "xs_scatter": md_scatter, "xs_fission": md_fission}



with h5py.File("critical_2d_C5G7_xs.h5", "w") as f:

    for key, item in UO2.items():
        f["UO2/" + key] = item

    for key, item in MOX43.items():
        f["MOX43/" + key] = item

    for key, item in MOX70.items():
        f["MOX70/" + key] = item

    for key, item in MOX87.items():
        f["MOX87/" + key] = item

    for key, item in fission_chamber.items():
        f["fission_chamber/" + key] = item
        
    for key, item in guide_tubes.items():
        f["guide_tubes/" + key] = item

    for key, item in moderator.items():
        f["moderator/" + key] = item


print("Saved!")