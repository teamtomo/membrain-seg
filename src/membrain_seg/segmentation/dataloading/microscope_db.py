#
# THIS CODE IS COPIED FROM PYTO:
# https://github.com/vladanl/Pyto/tree/master/pyto/io
# THIS IS TO AVOID THE INSTALLATION ISSUES AND NO PYPI AVAILABLITY
# PLASE CITE ACCORDINGLY
#
#
# Lorenz Lamm, July 2023


# pixelsize (in pm) at the specimen level for different nominal magnifications
pixelsize = {}

# physical CCD pixelsize (in um)
ccd_pixelsize = {}

# number of pixels on CCD
n_pixels = {}

# nominal magnification for different real magnifications
nominal_mag = {}

# number of counts per electron
conversion = {}


###################################################
#
# Titan 2 with K2 in counting mode recorded with SerialEM
#

# screen up (nominal) magnification vs. pixel size (at specimen level)
pixelsize["titan-2_k2-count_sem"] = {}

# counts per electron
conversion["titan-2_k2-count_sem"] = 15


###################################################
#
# Polara 2 with K2 in counting mode recorded with SerialEM
#

# screen up (nominal) magnification vs. pixel size (at specimen level)
pixelsize["polara-2_k2-count_sem"] = {}

# counts per electron
conversion["polara-2_k2-count_sem"] = 19

# CCD pixel size
# ccd_pixelsize['polara-2_01-09'] = 30000

# number of pixels at CCD
# n_pixels['polara-2_01-09'] = 2048


###################################################
#
# Polara 1 from 01.07
#

# screen up (nominal) magnification vs. pixel size (at specimen level)
pixelsize["polara-1_01-07"] = {
    18000: 1230,
    22500: 979,
    27500: 805,
    34000: 661,
    41000: 545,
    50000: 446,
    61000: 364,
}

# counts per electron
conversion["polara-1_01-07"] = 5.91

# CCD pixel size
ccd_pixelsize["polara-1_01-07"] = 30000

# number of pixels at CCD
n_pixels["polara-1_01-07"] = 2048


###################################################
#
# Polara 1 from 01.09
#

# screen up (nominal) magnification vs. pixel size (at specimen level)
pixelsize["polara-1_01-09"] = pixelsize["polara-1_01-07"]

# counts per electron
conversion["polara-1_01-09"] = 2.3

# CCD pixel size
ccd_pixelsize["polara-1_01-09"] = 30000

# number of pixels at CCD
n_pixels["polara-1_01-09"] = 2048


###################################################
#
# Polara 2 from 01.09
#

# screen up (nominal) magnification vs. pixel size (at specimen level)
pixelsize["polara-2_01-09"] = {
    9300: 1372,
    13500: 956,
    18000: 713,
    22500: 572,
    27500: 468,
    34000: 381,
}

# counts per electron
conversion["polara-2_01-09"] = 8.1

# CCD pixel size
ccd_pixelsize["polara-2_01-09"] = 30000

# number of pixels at CCD
n_pixels["polara-2_01-09"] = 2048


###################################################
#
# Krios 2, Falcon detector from 09.2011
#

# screen up (nominal) magnification vs. pixel size [fm] (at specimen level)
pixelsize["krios-2_falcon_05-2011"] = {18000: 475}

# counts per electron (JP 12.2011)
conversion["krios-2_falcon_05-2011"] = 134.0

# CCD pixel size (not determined yet)
ccd_pixelsize["krios-2_falcon_05-2011"] = 1

# number of pixels at CCD
n_pixels["krios-2_falcon_05-2011"] = 4096


###################################################
#
# F20 eagle camera
#

# screen up (nominal) magnification vs. pixel size [fm] (at specimen level)
pixelsize["f20_eagle"] = {}

# counts per electron
conversion["f20_eagle"] = 73.0


###################################################
#
# CM 300
#

# screen up (nominal) magnification vs. pixel size (at specimen level)
pixelsize["cm300"] = {13500: 1147, 17500: 821, 23000: 682, 27500: 547}

nominal_mag["cm300"] = {26157: 13500, 36527: 17500, 43974: 23000, 54844: 27500}

# counts per electron
conversion["cm300"] = 5.5

# CCD pixel size
ccd_pixelsize["cm300"] = 30000

# number of pixels at CCD
n_pixels["cm300"] = 2048
