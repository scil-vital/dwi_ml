
# We may want to define a script to preprocess data with steps that will never
# be included in scilpy because it doesn't fit with their goal.

# RAW DWI:
# - Resample raw DWI to sh_basis, using
"""
from dwi_ml.data.processing.dwi.dwi import resample_raw_dwi_from_sh
if self.resample:
    # Brings to SH and then back to directions.
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())
    output = resample_raw_dwi_from_sh(dwi_image, gtab, sh_order=sh_order)
else:
    output = dwi_image.get_fdata(dtype=np.float32)
"""


# SH DWI:
# - Resample SH order
"""
from scilpy.reconst.fodf import compute_sh_coefficients
gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())
output = compute_sh_coefficients(dwi_image, gtab, sh_order=sh_order)
"""


# Peaks fODF:
# - ?

