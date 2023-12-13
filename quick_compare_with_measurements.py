""" 
Compare with MO-Powering
------------------------

In this scritp the measured detuning corrections are compared to the powering in MO-Powering,
which has been calculated before via the lhc_run_check_mo_powering.py script.
"""
    
    
import numpy as np
import pandas as pd
from lhc_run_check_mo_powering import get_detuning_from_ptc_output
import tfs

AMPDET_FILENAME = "results/b{beam:d}/ampdet.lhc.b{beam:d}.mo{mo:d}.tfs"

MEASURED_FLAT = {
    1: {"X":-18, "Y":0, "XY":27},
    2: {"X":-19, "Y":3.4, "XY":12.5},
}

MEASURED_BEFORE = {
    1: {"X":9, "Y":2.1, "XY":32},
    2: {"X":20, "Y":20, "XY":-41},
}

MEASURED_AFTER = {
    1: {"X":-12.7, "Y":17.5, "XY":31.5},
    2: {"X":-46, "Y":-18, "XY":32.75},
}

PREDICTED_IMPROVED_CORRECTION={  # inverted sign, as this is correction on full (i.e. before-flat+correction)
    1: {"X":20, "Y":-12, "XY":-7},
    2: {"X":50, "Y":23, "XY":-53},
}

def prnt(label: str, args):
    try:
        print(f"{label:<10}: " +  (("{:<+7.1f} ")*len(args)).format(*args))
    except ValueError:
        print(f"{label:<10}: " +  (("{:<7s} ")*len(args)).format(*args))

def add_rms(s: pd.Series) -> pd.Series:
    s = s.copy()
    s["RMS"] = np.sqrt((s**2).mean())
    return s


if __name__ == "__main__":

    for beam in (1, 2):
        flat = pd.Series(MEASURED_FLAT[beam])
        nob6 = pd.Series(MEASURED_BEFORE[beam])
        wb6 = pd.Series(MEASURED_AFTER[beam])
        improved = nob6 - pd.Series(PREDICTED_IMPROVED_CORRECTION[beam])

        ampdet_mo_none = get_detuning_from_ptc_output(tfs.read(AMPDET_FILENAME.format(beam=beam, mo=0)))
        ampdet_mo_powered = get_detuning_from_ptc_output(tfs.read(AMPDET_FILENAME.format(beam=beam, mo=433)))
        mo_power_delta = (pd.Series(ampdet_mo_powered) - pd.Series(ampdet_mo_none)) * 1e-3


        

        # prnt("MO", rms_dict(mo_power_delta).values())


        # mo_power_delta = {"X": 1e2, "Y": 1e2, "XY": 1e2}  # to check actual detuning values

        diff_nob6 = nob6-flat 
        diff_wb6 = wb6-flat
        diff_improved = improved-flat

        pct_nob6 = (diff_nob6.abs() / mo_power_delta.abs()) * 100
        pct_wb6 = (diff_wb6.abs() / mo_power_delta.abs()) * 100
        pct_correction = pct_wb6 - pct_nob6

        pct_improved = (diff_improved.abs() / mo_power_delta.abs()) * 100
        pct_improved_correction = pct_improved - pct_nob6
        

        prnt(f"Beam {beam}", add_rms(diff_nob6).index)
        prnt("", ["[ % ]"]*4)
        prnt("-"*10, ["-"*7]*4)
        prnt("Before Cor.", add_rms(pct_nob6))
        prnt("After Cor.", add_rms(pct_wb6))
        prnt("DAfter", pct_correction)
        prnt("Improved", add_rms(pct_improved))
        prnt("DImproved", pct_improved_correction)
        print()