from .digitize import *

spectrum_top = 50
def process_spec(spectrum, output_dim):
    spectrum = spectrum/spectrum_top
    return digitizex(spectrum, output_dim)